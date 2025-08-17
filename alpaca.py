#!/usr/bin/env python3
"""
alpaca — Ollama-style wrapper for vLLM (Docker backend) with optional Ray shared inference.

Commands (local):
  pull <hf_repo>[:rev] [--alias <name>]
  serve <model-or-alias> [--port <p>] [--gpu|--cpu] [--dtype auto|float32|bf16|fp16] [--max-seqs N]
  run <model-or-alias> [-p "…"] [--path /v1/chat/completions] [--json FILE]
  ps
  ls [--size]
  stop <model-or-alias>
  rm <model-or-alias> [--purge]
  logs <model-or-alias> [-f]
  config [--set KEY=VAL] [--show]

Ray shared inference:
  ray-head [--dashboard-port 8265] [--client-port 10001] [--gcs-port 6379]
  ray-worker --address <HEAD_IP:6379> [--cpus N] [--gpu]
  serve-ray <model-or-alias> --address ray://<HEAD_IP>:10001 [--namespace vllm]
            [--port <p>] [--dtype auto|float32|bf16|fp16] [--max-seqs N]
  ray-down  (stop & remove Ray head/workers launched by alpaca)

Convenience cluster helpers (single host):
  cluster-up   [--cpu-workers N] [--gpu-workers N] [--cpus-per-worker M]
               [--dashboard-port 8265] [--client-port 10001] [--gcs-port 6379]
  cluster-down (alias of ray-down)

Notes:
  - Uses Docker; mounts HF cache at ~/.cache/huggingface.
  - Respects HUGGING_FACE_HUB_TOKEN for private/gated models.
  - GPU detection is automatic for local 'serve'; for Ray workers use --gpu.
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# ----------------------------
# Configuration & paths
# ----------------------------
CACHE_DIR = Path(os.environ.get("ALPACA_CACHE_DIR", str(Path.home() / ".cache" / "huggingface")))
STATE_DIR = Path(os.environ.get("ALPACA_STATE_DIR", str(Path.home() / ".alpaca")))
REGISTRY_PATH = STATE_DIR / "registry.json"
DEFAULT_PORT_START = int(os.environ.get("ALPACA_PORT_START", "8000"))
DTYPE_CPU_DEFAULT = os.environ.get("ALPACA_DTYPE_CPU", "float32")
DTYPE_GPU_DEFAULT = os.environ.get("ALPACA_DTYPE_GPU", "auto")
MAX_SEQS_DEFAULT = int(os.environ.get("ALPACA_MAX_SEQS", "32"))

# Ray defaults
RAY_DASHBOARD_PORT = int(os.environ.get("ALPACA_RAY_DASHBOARD_PORT", "8265"))
RAY_CLIENT_PORT = int(os.environ.get("ALPACA_RAY_CLIENT_PORT", "10001"))
RAY_GCS_PORT = int(os.environ.get("ALPACA_RAY_GCS_PORT", "6379"))

# ----------------------------
# Utilities
# ----------------------------
def die(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)

def info(msg: str):
    print(msg, file=sys.stderr)

def run(cmd: List[str], check=True, capture=False, env=None):
    kwargs = {}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        kwargs["text"] = True
    return subprocess.run(cmd, check=check, env=env, **kwargs)

def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def ensure_vllm():
    try:
        import vllm
    except ImportError:
        die("vLLM not installed. Run: pip install vllm")

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name.replace("/", "_").replace(":", "_"))

def alias_to_state_dir(alias: str) -> Path:
    return STATE_DIR / "models" / sanitize(alias)

def process_name(alias: str) -> str:
    return f"alpaca_vllm_{sanitize(alias)}"

def read_registry() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text())
        except Exception:
            pass
    return {"aliases": {}, "servers": {}, "config": {}}

def write_registry(reg: Dict[str, Any]):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))

def resolve_alias_or_repo(token: str, reg: Dict[str, Any]) -> Tuple[str, str]:
    aliases = reg.get("aliases", {})
    if token in aliases:
        return token, aliases[token]["repo"]
    return sanitize(token), token

def next_free_port(start: int = DEFAULT_PORT_START, max_attempts: int = 200) -> int:
    for port in range(start, start + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.1)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    die("No free port found in range.")

def gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def ensure_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def ensure_state():
    (STATE_DIR / "models").mkdir(parents=True, exist_ok=True)

def hf_snapshot_download(repo_id: str, revision: Optional[str] = None, token: Optional[str] = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        die("huggingface_hub is not installed. Run: pip install huggingface_hub")
    kwargs = dict(repo_id=repo_id, allow_patterns=None, local_files_only=False)
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token
    path = snapshot_download(**kwargs)
    return Path(path)

def parse_repo_spec(spec: str) -> Tuple[str, Optional[str]]:
    if ":" in spec and "/" in spec.split(":")[0]:
        repo, rev = spec.split(":", 1)
        return repo, rev
    return spec, None

def human_bytes(n: int) -> str:
    for unit in ["B","KiB","MiB","GiB","TiB"]:
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024

def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except Exception:
            pass
    return total

def get_running_processes() -> List[Dict[str, str]]:
    """Get running alpaca vLLM processes."""
    import psutil
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = proc.info['cmdline'] or []
            if any('vllm.entrypoints.openai.api_server' in arg for arg in cmdline):
                # Check if this is an alpaca-managed process
                if any('alpaca_vllm' in arg for arg in cmdline):
                    processes.append({
                        'pid': str(proc.info['pid']),
                        'name': proc.info['name'],
                        'cmdline': ' '.join(cmdline),
                        'status': proc.status(),
                        'create_time': proc.info['create_time']
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes

def fetch_env_token() -> Optional[str]:
    return os.environ.get("HUGGING_FACE_HUB_TOKEN")

def kill_processes_by_pattern(pattern: str):
    """Kill processes matching a pattern."""
    import psutil
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline'] or []
            if any(pattern in arg for arg in cmdline):
                proc.terminate()
                proc.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            try:
                proc.kill()
            except:
                pass

# ----------------------------
# Subcommand handlers (local)
# ----------------------------
def cmd_pull(args):
    ensure_vllm()
    ensure_cache()
    ensure_state()
    reg = read_registry()

    repo_spec = args.repo
    alias = args.alias
    if alias and not re.match(r"^[A-Za-z0-9_.-]+$", alias):
        die("Alias may contain only letters, numbers, dot, underscore, and dash.")

    repo, rev = parse_repo_spec(repo_spec)
    token = fetch_env_token()

    info(f"Pulling {repo_spec} into cache…")
    path = hf_snapshot_download(repo, revision=rev, token=token)
    size = dir_size(path)

    if alias is None:
        base = repo.split("/")[-1]
        alias = base
        existing = set(read_registry().get("aliases", {}).keys())
        suffix = 0
        while alias in existing:
            suffix += 1
            alias = f"{base}-{suffix}"

    reg.setdefault("aliases", {})
    reg["aliases"][alias] = {
        "repo": repo_spec,
        "resolved_repo": repo,
        "revision": rev,
        "cache_path": str(path),
        "size_bytes": size,
        "created_at": int(time.time())
    }
    write_registry(reg)
    info(f"Pulled ✓  alias: {alias}  size: {human_bytes(size)}")
    print(alias)

def cmd_ls(args):
    reg = read_registry()
    aliases = reg.get("aliases", {})
    if not aliases:
        print("(no cached models)")
        return
    for alias, meta in aliases.items():
        size = meta.get("size_bytes")
        size_s = human_bytes(size) if size else "?"
        print(f"{alias:24s}  {meta.get('repo')}  rev={meta.get('revision') or 'latest'}  size≈{size_s}")
    if args.size:
        total = sum(m.get("size_bytes") or 0 for m in aliases.values())
        print(f"\nTotal cache size: {human_bytes(total)}")

def cmd_ps(_args):
    processes = get_running_processes()
    reg = read_registry()
    servers = reg.get("servers", {})
    
    if not processes and not servers:
        print("(no running alpaca-managed processes)")
        return
        
    for alias, server_info in servers.items():
        pid = server_info.get("pid")
        port = server_info.get("port", "")
        mode = server_info.get("mode", "?")
        repo = server_info.get("repo", "?")
        
        # Check if process is still running
        try:
            import psutil
            proc = psutil.Process(int(pid)) if pid else None
            status = proc.status() if proc and proc.is_running() else "stopped"
        except:
            status = "stopped"
            
        print(f"{alias:32s}  {mode:10s}  port={port:5s}  {repo:40s}  pid={pid}  {status}")

def cmd_stop(args):
    reg = read_registry()
    alias, _ = resolve_alias_or_repo(args.model, reg)
    servers = reg.get("servers", {})
    
    if alias not in servers:
        info(f"No running server found for '{alias}'")
        return
        
    pid = servers[alias].get("pid")
    if pid:
        try:
            import psutil
            proc = psutil.Process(int(pid))
            proc.terminate()
            proc.wait(timeout=10)
            info(f"Stopped server for {alias} (PID {pid})")
        except Exception as e:
            info(f"Failed to stop process {pid}: {e}")
    
    # Remove from registry
    servers.pop(alias, None)
    write_registry(reg)

def cmd_rm(args):
    reg = read_registry()
    alias, _ = resolve_alias_or_repo(args.model, reg)
    
    # Stop the server first
    servers = reg.get("servers", {})
    if alias in servers:
        cmd_stop(args)
    
    info(f"Removed server {alias}")
    if args.purge:
        if alias in reg.get("aliases", {}):
            reg["aliases"].pop(alias, None)
            write_registry(reg)
            info(f"Removed alias '{alias}' from registry.")
        else:
            info("Alias not found in registry; nothing to purge.")

def _decide_backend() -> str:
    if not gpu_available():
        die("GPU not available. This version requires CUDA GPU support.")
    return "gpu"

def _dtype_for_mode(user_dtype: Optional[str]) -> str:
    if user_dtype:
        return user_dtype
    return DTYPE_GPU_DEFAULT

def _find_or_allocate_port(reg, alias: str, preferred: Optional[int]) -> int:
    srv = reg.get("servers", {}).get(alias)
    if srv and srv.get("port"):
        return int(srv["port"])
    if preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", preferred)) != 0:
                return preferred
            info(f"Port {preferred} busy; allocating automatically.")
    return next_free_port()

def _model_for_token(reg, token: str) -> Tuple[str, str]:
    alias, repo_spec = resolve_alias_or_repo(token, reg)
    if alias not in reg.get("aliases", {}):
        reg.setdefault("aliases", {})
        reg["aliases"][alias] = {"repo": repo_spec, "created_at": int(time.time())}
        write_registry(reg)
    return alias, repo_spec

def _docker_labels(base: Dict[str, str]) -> List[str]:
    res = []
    for k, v in base.items():
        res += ["--label", f"{k}={v}"]
    return res

def cmd_serve(args):
    ensure_vllm(); ensure_cache(); ensure_state()
    reg = read_registry()
    alias, repo_spec = _model_for_token(reg, args.model)
    mode = _decide_backend()  # Always GPU now
    dtype = _dtype_for_mode(args.dtype)
    port = _find_or_allocate_port(reg, alias, args.port)

    # Stop any existing server for this alias
    servers = reg.get("servers", {})
    if alias in servers:
        cmd_stop(args)

    # Build vLLM command
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", repo_spec,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", dtype,
        "--max-num-seqs", str(args.max_seqs or MAX_SEQS_DEFAULT)
    ]

    # Set environment variables
    env = os.environ.copy()
    env["VLLM_USE_MODELSCOPE"] = "false"
    if token := fetch_env_token():
        env["HUGGING_FACE_HUB_TOKEN"] = token

    info(f"Starting vLLM (GPU, dtype={dtype}) on port {port} for {repo_spec} …")
    
    # Start the process
    proc = subprocess.Popen(
        vllm_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    _post_start_health(alias, port, repo_spec, proc.pid, mode, dtype)

def _post_start_health(alias, port, repo_spec, pid, mode, dtype):
    reg = read_registry()
    reg.setdefault("servers", {})
    reg["servers"][alias] = {
        "pid": pid,
        "port": port, "mode": mode, "dtype": dtype,
        "repo": repo_spec, "started_at": int(time.time())
    }
    write_registry(reg)
    # health
    try:
        import httpx
        base = f"http://127.0.0.1:{port}"
        ok=False
        for _ in range(120):
            try:
                r = httpx.get(f"{base}/v1/models", timeout=2)
                if r.status_code == 200: ok=True; break
            except Exception:
                pass
            time.sleep(0.5)
        if not ok:
            info("warning: server did not become healthy in time; check process logs.")
        print(f"ready: {repo_spec} as {alias}  mode={mode}  dtype={dtype}  port={port}  pid={pid}")
        print(f"Try: curl {base}/v1/models")
    except Exception:
        pass

def cmd_logs(args):
    reg = read_registry()
    alias, _ = resolve_alias_or_repo(args.model, reg)
    servers = reg.get("servers", {})
    
    if alias not in servers:
        die(f"No server found for alias '{alias}'")
    
    pid = servers[alias].get("pid")
    if not pid:
        die(f"No PID found for server '{alias}'")
    
    info(f"Note: vLLM process logs are handled by the process itself.")
    info(f"Server '{alias}' is running with PID {pid}")
    info("To see vLLM logs, check the terminal where alpaca was started or system logs.")

def cmd_run(args):
    reg = read_registry()
    alias, _ = resolve_alias_or_repo(args.model, reg)
    server = reg.get("servers", {}).get(alias)
    if not server:
        die(f"No running server for alias '{alias}'. Start with: alpaca serve {alias} OR serve-ray.")
    import httpx, json as _json
    base = f"http://127.0.0.1:{server['port']}"
    path = args.path or "/v1/chat/completions"
    url = f"{base}{path}"
    if args.json:
        payload = _json.loads(Path(args.json).read_text())
    else:
        prompt = args.prompt or "Hello!"
        payload = {"model":"unused","messages":[{"role":"user","content":prompt}],"max_tokens":128}
    r = httpx.post(url, json=payload, timeout=120)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))

def cmd_config(args):
    reg = read_registry()
    if args.show or not args.set:
        print(json.dumps(reg.get("config", {}), indent=2)); return
    reg.setdefault("config", {})
    for kv in args.set:
        if "=" not in kv: die(f"--set expects KEY=VAL, got: {kv}")
        k, v = kv.split("=", 1); reg["config"][k]=v
    write_registry(reg); print("OK")

# ----------------------------
# Ray shared inference (simplified - requires manual Ray setup)
# ----------------------------
def cmd_serve_ray(args):
    """Launch vLLM server with Ray distributed executor (requires pre-existing Ray cluster)."""
    try:
        import ray
    except ImportError:
        die("Ray not installed. Run: pip install ray")
        
    ensure_vllm(); ensure_cache(); ensure_state()
    
    if not args.address.startswith("ray://"):
        die("RAY address must start with ray:// (example: ray://127.0.0.1:10001)")
    
    reg = read_registry()
    alias, repo_spec = _model_for_token(reg, args.model)
    dtype = args.dtype or "auto"
    port = _find_or_allocate_port(reg, alias, args.port)

    # Stop any existing server for this alias
    servers = reg.get("servers", {})
    if alias in servers:
        cmd_stop(args)

    # Build vLLM command with Ray backend
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", repo_spec,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", dtype,
        "--distributed-executor-backend", "ray",
        "--max-num-seqs", str(args.max_seqs or MAX_SEQS_DEFAULT)
    ]

    # Set environment variables
    env = os.environ.copy()
    env["RAY_ADDRESS"] = args.address
    env["RAY_NAMESPACE"] = args.namespace
    env["VLLM_USE_MODELSCOPE"] = "false"
    if token := fetch_env_token():
        env["HUGGING_FACE_HUB_TOKEN"] = token

    info(f"Starting vLLM (Ray backend) on port {port}, address={args.address}")
    
    # Start the process
    proc = subprocess.Popen(
        vllm_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    _post_start_health(alias, port, repo_spec, proc.pid, "ray", dtype)

# ----------------------------
# Argument parsing
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser(prog="alpaca", description="Ollama-style wrapper for vLLM (Docker) + Ray.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pull", help="Prefetch a HF repo into local cache.")
    sp.add_argument("repo", help="HF repo spec, e.g. org/name[:rev]")
    sp.add_argument("--alias", help="Local alias name to register.")
    sp.set_defaults(func=cmd_pull)

    sp = sub.add_parser("ls", help="List cached models (aliases).")
    sp.add_argument("--size", action="store_true", help="Show total cache size.")
    sp.set_defaults(func=cmd_ls)

    sp = sub.add_parser("serve", help="Serve a model via vLLM (GPU only).")
    sp.add_argument("model", help="Alias or HF repo spec")
    sp.add_argument("--port", type=int, help="Host port (default auto).")
    sp.add_argument("--dtype", choices=["auto", "float32", "bf16", "fp16"], help="Override dtype.")
    sp.add_argument("--max-seqs", type=int, help=f"Max concurrent sequences (default {MAX_SEQS_DEFAULT}).")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("ps", help="List running alpaca-managed processes.")
    sp.set_defaults(func=cmd_ps)

    sp = sub.add_parser("stop", help="Stop a running server.")
    sp.add_argument("model", help="Alias or repo spec used to start it.")
    sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("rm", help="Remove server process and (optionally) alias.")
    sp.add_argument("model", help="Alias or repo spec.")
    sp.add_argument("--purge", action="store_true", help="Also remove alias from registry.")
    sp.set_defaults(func=cmd_rm)

    sp = sub.add_parser("logs", help="Show server process info.")
    sp.add_argument("model", help="Alias or repo spec.")
    sp.add_argument("-f", "--follow", action="store_true", help="(ignored - for compatibility)")
    sp.set_defaults(func=cmd_logs)

    sp = sub.add_parser("run", help="Send a single request to a running server.")
    sp.add_argument("model", help="Alias or repo spec.")
    sp.add_argument("-p", "--prompt", help="Prompt text (for chat/completions).")
    sp.add_argument("--path", help="Override API path (default /v1/chat/completions).")
    sp.add_argument("--json", help="Path to JSON request body (overrides --prompt).")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("config", help="Show or set config.")
    sp.add_argument("--show", action="store_true", help="Show current config.")
    sp.add_argument("--set", nargs="*", help="Set KEY=VAL pairs.")
    sp.set_defaults(func=cmd_config)

    # Ray shared inference (simplified)
    sp = sub.add_parser("serve-ray", help="Serve a model using Ray distributed executor (requires pre-existing Ray cluster).")
    sp.add_argument("model", help="Alias or HF repo spec")
    sp.add_argument("--address", required=True, help="Ray client address, e.g. ray://10.0.0.5:10001")
    sp.add_argument("--namespace", default="vllm", help="Ray namespace to use.")
    sp.add_argument("--port", type=int, help="Host port for API (default auto).")
    sp.add_argument("--dtype", choices=["auto", "float32", "bf16", "fp16"], help="Override dtype (default auto).")
    sp.add_argument("--max-seqs", type=int, help=f"Max concurrent sequences (default {MAX_SEQS_DEFAULT}).")
    sp.set_defaults(func=cmd_serve_ray)

    return p

# ----------------------------
# Main
# ----------------------------
def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
