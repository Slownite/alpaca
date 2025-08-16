
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
IMAGE = os.environ.get("ALPACA_IMAGE", "vllm/vllm-openai:latest")
RAY_IMAGE = os.environ.get("ALPACA_RAY_IMAGE", "rayproject/ray:2.34.0")
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

def ensure_docker():
    if not have("docker"):
        die("Docker not found in PATH. Install Docker first.")
    try:
        run(["docker", "version"], check=True)
    except Exception as e:
        die(f"Cannot talk to Docker daemon: {e}")

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name.replace("/", "_").replace(":", "_"))

def alias_to_state_dir(alias: str) -> Path:
    return STATE_DIR / "models" / sanitize(alias)

def container_name(prefix: str, alias: str) -> str:
    return f"{prefix}_{sanitize(alias)}"

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
        out = run(["docker", "info", "--format", "{{json .Runtimes}}"], capture=True)
        if '"nvidia"' not in out.stdout:
            return False
        probe = run([
            "docker","run","--rm","--gpus","all","--entrypoint","bash",
            "nvidia/cuda:12.2.0-base","-lc","nvidia-smi -L"
        ], check=False, capture=True)
        return probe.returncode == 0 and "GPU" in (probe.stdout or "")
    except Exception:
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

def docker_ps_alpaca() -> List[Dict[str, str]]:
    out = run([
        "docker", "ps", "--filter", "label=alpaca=true",
        "--format", "{{.ID}}|{{.Names}}|{{.Status}}|{{.Ports}}|{{.Labels}}"
    ], check=False, capture=True)
    rows = []
    if out.returncode != 0:
        return rows
    for line in out.stdout.strip().splitlines():
        parts = line.split("|", 4)
        if len(parts) == 5:
            rows.append({"id": parts[0], "name": parts[1], "status": parts[2], "ports": parts[3], "labels": parts[4]})
    return rows

def fetch_env_token() -> Optional[str]:
    return os.environ.get("HUGGING_FACE_HUB_TOKEN")

def docker_rm_by_filter(label_filter: str):
    out = run(["docker", "ps", "-aq", "--filter", f"label={label_filter}"], capture=True, check=False)
    ids = out.stdout.strip().splitlines() if out.stdout else []
    if not ids:
        return
    run(["docker", "rm", "-f"] + ids, check=False)

# ----------------------------
# Subcommand handlers (local)
# ----------------------------
def cmd_pull(args):
    ensure_docker()
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
    rows = docker_ps_alpaca()
    if not rows:
        print("(no running alpaca-managed containers)")
        return
    for r in rows:
        labels = {}
        for kv in r["labels"].split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                labels[k.strip()] = v.strip()
        mode = labels.get("alpaca.mode", labels.get("alpaca.role", "?"))
        model = labels.get("alpaca.model", labels.get("alpaca.role", "?"))
        port = labels.get("alpaca.port", "")
        extra = labels.get("alpaca.extra", "")
        print(f"{r['name']:32s}  {mode:10s}  port={port:5s}  {model:40s}  {extra}  {r['status']:s}")

def cmd_stop(args):
    reg = read_registry()
    alias, _ = resolve_alias_or_repo(args.model, reg)
    name = container_name("alpaca_vllm", alias)
    run(["docker", "stop", name], check=False)
    info(f"Stopped {name}")

def cmd_rm(args):
    reg = read_registry()
    alias, _ = resolve_alias_or_repo(args.model, reg)
    name = container_name("alpaca_vllm", alias)
    run(["docker", "rm", "-f", name], check=False)
    info(f"Removed container {name}")
    if args.purge:
        if alias in reg.get("aliases", {}):
            reg["aliases"].pop(alias, None)
            write_registry(reg)
            info(f"Removed alias '{alias}' from registry.")
        else:
            info("Alias not found in registry; nothing to purge.")

def _decide_backend(args_backend: Optional[str]) -> str:
    if args_backend == "gpu":
        return "gpu"
    if args_backend == "cpu":
        return "cpu"
    return "gpu" if gpu_available() else "cpu"

def _dtype_for_mode(mode: str, user_dtype: Optional[str]) -> str:
    if user_dtype:
        return user_dtype
    return DTYPE_GPU_DEFAULT if mode == "gpu" else DTYPE_CPU_DEFAULT

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
    ensure_docker(); ensure_cache(); ensure_state()
    reg = read_registry()
    alias, repo_spec = _model_for_token(reg, args.model)
    mode = _decide_backend("gpu" if args.gpu else "cpu" if args.cpu else None)
    dtype = _dtype_for_mode(mode, args.dtype)
    port = _find_or_allocate_port(reg, alias, args.port)
    name = container_name("alpaca_vllm", alias)

    labels = {
        "alpaca":"true","alpaca.alias":alias,"alpaca.model":repo_spec,
        "alpaca.mode":mode,"alpaca.port":str(port)
    }
    cmd = ["docker","run","-d","--name",name,"--restart","unless-stopped",
           "-p",f"{port}:8000","-e","VLLM_USE_MODELSCOPE=false",
           "-v",f"{str(CACHE_DIR)}:/root/.cache/huggingface"] + _docker_labels(labels)
    if mode == "gpu":
        cmd += ["--gpus","all","-e","NVIDIA_VISIBLE_DEVICES=all"]

    vllm_args = ["python","-m","vllm.entrypoints.openai.api_server",
                 "--model",repo_spec,"--host","0.0.0.0","--port","8000",
                 "--dtype",dtype,"--max-num-seqs",str(args.max_seqs or MAX_SEQS_DEFAULT)]
    cmd += [IMAGE] + vllm_args

    run(["docker","rm","-f",name],check=False)
    info(f"Starting vLLM ({mode}, dtype={dtype}) on port {port} for {repo_spec} …")
    run(cmd, check=True)
    _post_start_health(alias, port, repo_spec, IMAGE, mode, dtype)

def _post_start_health(alias, port, repo_spec, image, mode, dtype):
    reg = read_registry()
    reg.setdefault("servers", {})
    reg["servers"][alias] = {
        "container": container_name("alpaca_vllm", alias),
        "port": port, "mode": mode, "dtype": dtype,
        "repo": repo_spec, "image": image, "started_at": int(time.time())
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
            info("warning: server did not become healthy in time; use `alpaca logs`.")
        print(f"ready: {repo_spec} as {alias}  mode={mode}  dtype={dtype}  port={port}")
        print(f"Try: curl {base}/v1/models")
    except Exception:
        pass

def cmd_logs(args):
    reg = read_registry()
    alias, _ = resolve_alias_or_repo(args.model, reg)
    name = container_name("alpaca_vllm", alias)
    cmd = ["docker","logs"]
    if args.follow: cmd.append("-f")
    cmd.append(name)
    os.execvp(cmd[0], cmd)

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
# Ray shared inference
# ----------------------------
def cmd_ray_head(args):
    ensure_docker()
    name = "alpaca_ray_head"
    dash = args.dashboard_port or RAY_DASHBOARD_PORT
    client = args.client_port or RAY_CLIENT_PORT
    gcs = args.gcs_port or RAY_GCS_PORT

    labels = {"alpaca":"true","alpaca.role":"ray-head",
              "alpaca.extra":f"dash={dash} client={client} gcs={gcs}"}
    cmd = ["docker","run","-d","--name",name,"--restart","unless-stopped"] + \
          ["-p",f"{dash}:{dash}","-p",f"{client}:{client}","-p",f"{gcs}:{gcs}"] + \
          _docker_labels(labels) + [RAY_IMAGE,"bash","-lc",
          f"ray start --head --port={gcs} --dashboard-host=0.0.0.0 --dashboard-port={dash} "
          f"--ray-client-server-port={client} --num-cpus=0 && tail -f /dev/null"]
    run(["docker","rm","-f",name],check=False)
    info(f"Starting Ray head (dashboard {dash}, client {client}, gcs {gcs}) …")
    run(cmd, check=True)

    print("Ray head ready.")
    print(f"Dashboard: http://127.0.0.1:{dash}")
    print(f"Workers join with: alpaca ray-worker --address  <HEAD_IP:{gcs}>")
    print(f"vLLM connects with: alpaca serve-ray … --address ray://<HEAD_IP>:{client}")

def cmd_ray_worker(args):
    ensure_docker()
    if not args.address:
        die("--address is required (format HEAD_IP:6379)")

    name = f"alpaca_ray_worker_{int(time.time())}"
    labels = {"alpaca":"true","alpaca.role":"ray-worker"}
    cmd = ["docker","run","-d","--name",name,"--restart","unless-stopped"] + _docker_labels(labels)

    cpus = max(1, int(args.cpus)) if args.cpus else 0  # 0 = Ray auto-detects

    if args.gpu:
        cmd += ["--gpus","all","-e","NVIDIA_VISIBLE_DEVICES=all"]

    cmd += ["-v",f"{str(CACHE_DIR)}:/root/.cache/huggingface"]

    ray_cmd = f"ray start --address={args.address} " + (f"--num-cpus={cpus} " if cpus>0 else "") + "&& tail -f /dev/null"
    cmd += [RAY_IMAGE,"bash","-lc",ray_cmd]
    info(f"Starting Ray worker (gpu={args.gpu}, cpus={'auto' if cpus==0 else cpus}) -> {args.address}")
    run(cmd, check=True)
    print("Ray worker started.")

def cmd_ray_down(_args):
    """Stop and remove Ray head/workers launched by alpaca."""
    ensure_docker()
    info("Stopping and removing alpaca Ray head/workers …")
    docker_rm_by_filter("alpaca.role=ray-head")
    docker_rm_by_filter("alpaca.role=ray-worker")
    print("Ray cluster containers removed.")

def cmd_serve_ray(args):
    """Launch vLLM server attached to a Ray cluster (multi-node shared inference)."""
    ensure_docker(); ensure_cache(); ensure_state()
    if not args.address.startswith("ray://"):
        die("RAY address must start with ray:// (example: ray://<HEAD_IP>:10001)")
    reg = read_registry()
    alias, repo_spec = _model_for_token(reg, args.model)

    dtype = args.dtype or "auto"
    port = _find_or_allocate_port(reg, alias, args.port)
    name = container_name("alpaca_vllm", alias)

    labels = {
        "alpaca":"true","alpaca.alias":alias,"alpaca.model":repo_spec,
        "alpaca.mode":"ray","alpaca.port":str(port),
        "alpaca.extra":f"namespace={args.namespace}"
    }

    cmd = ["docker","run","-d","--name",name,"--restart","unless-stopped",
           "-p",f"{port}:8000",
           "-e","VLLM_USE_MODELSCOPE=false",
           "-e",f"RAY_ADDRESS={args.address}",
           "-e",f"RAY_NAMESPACE={args.namespace}",
           "-v",f"{str(CACHE_DIR)}:/root/.cache/huggingface"] + _docker_labels(labels)

    # API container itself doesn't need GPUs; Ray places executors on workers.

    vllm_args = ["python","-m","vllm.entrypoints.openai.api_server",
                 "--model",repo_spec,"--host","0.0.0.0","--port","8000",
                 "--dtype",dtype,
                 "--distributed-executor-backend","ray",
                 "--max-num-seqs",str(args.max_seqs or MAX_SEQS_DEFAULT)]
    cmd += [IMAGE] + vllm_args

    run(["docker","rm","-f",name],check=False)
    info(f"Starting vLLM (Ray backend) on port {port}, namespace={args.namespace}, address={args.address}")
    run(cmd, check=True)
    _post_start_health(alias, port, repo_spec, IMAGE, "ray", dtype)

# ----------------------------
# Cluster helpers (single host)
# ----------------------------
def cmd_cluster_up(args):
    """Start a Ray head + local workers on the same machine."""
    # Head first
    cmd_ray_head(argparse.Namespace(
        dashboard_port=args.dashboard_port,
        client_port=args.client_port,
        gcs_port=args.gcs_port
    ))
    head_addr = f"127.0.0.1:{args.gcs_port}"

    # CPU workers
    for i in range(args.cpu_workers or 0):
        info(f"Starting CPU worker #{i+1} …")
        cmd_ray_worker(argparse.Namespace(address=head_addr, cpus=args.cpus_per_worker, gpu=False))
    # GPU workers
    for i in range(args.gpu_workers or 0):
        info(f"Starting GPU worker #{i+1} …")
        cmd_ray_worker(argparse.Namespace(address=head_addr, cpus=args.cpus_per_worker, gpu=True))

    print("cluster-up complete.")
    print(f"Use serve-ray … --address ray://127.0.0.1:{args.client_port}")

def cmd_cluster_down(_args):
    cmd_ray_down(_args)

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

    sp = sub.add_parser("serve", help="Serve a model via vLLM (local CPU/GPU).")
    sp.add_argument("model", help="Alias or HF repo spec")
    mgroup = sp.add_mutually_exclusive_group()
    mgroup.add_argument("--gpu", action="store_true", help="Force GPU mode.")
    mgroup.add_argument("--cpu", action="store_true", help="Force CPU mode.")
    sp.add_argument("--port", type=int, help="Host port (default auto).")
    sp.add_argument("--dtype", choices=["auto", "float32", "bf16", "fp16"], help="Override dtype.")
    sp.add_argument("--max-seqs", type=int, help=f"Max concurrent sequences (default {MAX_SEQS_DEFAULT}).")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("ps", help="List running alpaca-managed containers.")
    sp.set_defaults(func=cmd_ps)

    sp = sub.add_parser("stop", help="Stop a running server.")
    sp.add_argument("model", help="Alias or repo spec used to start it.")
    sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("rm", help="Remove server and (optionally) alias.")
    sp.add_argument("model", help="Alias or repo spec.")
    sp.add_argument("--purge", action="store_true", help="Also remove alias from registry.")
    sp.set_defaults(func=cmd_rm)

    sp = sub.add_parser("logs", help="Tail server logs.")
    sp.add_argument("model", help="Alias or repo spec.")
    sp.add_argument("-f", "--follow", action="store_true", help="Follow logs.")
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

    # Ray shared inference
    sp = sub.add_parser("ray-head", help="Start a Ray head node.")
    sp.add_argument("--dashboard-port", type=int, default=RAY_DASHBOARD_PORT)
    sp.add_argument("--client-port", type=int, default=RAY_CLIENT_PORT)
    sp.add_argument("--gcs-port", type=int, default=RAY_GCS_PORT)
    sp.set_defaults(func=cmd_ray_head)

    sp = sub.add_parser("ray-worker", help="Start a Ray worker that joins a head.")
    sp.add_argument("--address", required=True, help="Ray head GCS address, e.g. 10.0.0.5:6379")
    sp.add_argument("--cpus", type=int, help="CPUs to contribute (default: Ray auto).")
    sp.add_argument("--gpu", action="store_true", help="Contribute all GPUs from this host.")
    sp.set_defaults(func=cmd_ray_worker)

    sp = sub.add_parser("serve-ray", help="Serve a model using Ray distributed executor (multi-node).")
    sp.add_argument("model", help="Alias or HF repo spec")
    sp.add_argument("--address", required=True, help="Ray client address, e.g. ray://10.0.0.5:10001")
    sp.add_argument("--namespace", default="vllm", help="Ray namespace to use.")
    sp.add_argument("--port", type=int, help="Host port for API (default auto).")
    sp.add_argument("--dtype", choices=["auto", "float32", "bf16", "fp16"], help="Override dtype (default auto).")
    sp.add_argument("--max-seqs", type=int, help=f"Max concurrent sequences (default {MAX_SEQS_DEFAULT}).")
    sp.set_defaults(func=cmd_serve_ray)

    sp = sub.add_parser("ray-down", help="Stop and remove Ray head/workers launched by alpaca.")
    sp.set_defaults(func=cmd_ray_down)

    sp = sub.add_parser("cluster-up", help="Start Ray head + local workers on this host.")
    sp.add_argument("--cpu-workers", type=int, default=1, help="Number of local CPU workers.")
    sp.add_argument("--gpu-workers", type=int, default=0, help="Number of local GPU workers.")
    sp.add_argument("--cpus-per-worker", type=int, help="CPUs per worker (default Ray auto).")
    sp.add_argument("--dashboard-port", type=int, default=RAY_DASHBOARD_PORT)
    sp.add_argument("--client-port", type=int, default=RAY_CLIENT_PORT)
    sp.add_argument("--gcs-port", type=int, default=RAY_GCS_PORT)
    sp.set_defaults(func=cmd_cluster_up)

    sp = sub.add_parser("cluster-down", help="Tear down local Ray head/workers started by alpaca.")
    sp.set_defaults(func=cmd_cluster_down)

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
