#!/usr/bin/env python3
"""
alpaca — Ollama-style wrapper for vLLM with native Python processes and Ray cluster management.

Commands (local):
  pull <hf_repo>[:rev] [--alias <name>]
  serve <model-or-alias> [--port <p>] [--dtype auto|float32|bf16|fp16] [--max-seqs N]
  run <model-or-alias> [-p "…"] [--path /v1/chat/completions] [--json FILE]
  ps
  ls [--size]
  stop <model-or-alias>
  rm <model-or-alias> [--purge]
  logs <model-or-alias> [-f]
  config [--set KEY=VAL] [--show]

Ray cluster management:
  ray-head     [--dashboard-port 8265] [--client-port 10001] [--gcs-port 6379]
  ray-worker   --address <HEAD_IP:6379> [--cpus N] [--gpus N]
  ray-status   (show cluster status and resources)
  ray-down     (stop & cleanup Ray cluster)

Ray distributed inference:
  serve-ray <model-or-alias> [--address ray://<HEAD_IP>:10001] [--namespace vllm]
            [--port <p>] [--dtype auto|float32|bf16|fp16] [--max-seqs N]
            (auto-detects local Ray cluster if no address provided)

Convenience cluster helpers (single host):
  cluster-up   [--cpu-workers N] [--gpu-workers N] [--cpus-per-worker M]
               [--dashboard-port 8265] [--client-port 10001] [--gcs-port 6379]
  cluster-down (alias of ray-down)

Notes:
  - Uses native Python processes with GPU acceleration via vLLM.
  - Respects HUGGING_FACE_HUB_TOKEN for private/gated models.
  - Ray cluster management provides full lifecycle control of distributed setups.
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

def ensure_httpx():
    try:
        import httpx
    except ImportError:
        die("httpx not installed. Run: pip install httpx")

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
            
        print(f"{alias:32s}  {mode:10s}  port={str(port):5s}  {repo:40s}  pid={str(pid)}  {status}")

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
    
    # Extended health check for Ray mode
    if mode == "ray":
        _ray_health_check(alias, port, repo_spec, pid, dtype)
    else:
        _standard_health_check(alias, port, repo_spec, pid, mode, dtype)

def _ray_health_check(alias, port, repo_spec, pid, dtype):
    """Enhanced health check for Ray distributed servers."""
    try:
        import httpx
        import ray
        base = f"http://127.0.0.1:{port}"
        
        # First check Ray cluster connectivity
        ray_ok = False
        http_ok = False
        cluster_resources = {}
        available_resources = {}
        
        info("Checking Ray cluster connectivity...")
        try:
            # Check if Ray is already initialized
            if not ray.is_initialized():
                ray_address = os.environ.get("RAY_ADDRESS")
                if ray_address:
                    ray.init(address=ray_address)
            
            # Verify Ray cluster has resources
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            if cluster_resources.get('CPU', 0) > 0:
                ray_ok = True
                info(f"Ray cluster OK: {cluster_resources.get('CPU', 0)} total CPUs, {available_resources.get('CPU', 0)} available")
            else:
                info("warning: Ray cluster has no CPU resources available")
                
        except Exception as e:
            info(f"warning: Ray cluster connectivity failed: {e}")
        
        # Extended timeout for Ray servers (3 minutes)
        info("Checking HTTP endpoint health...")
        info("Note: vLLM with Ray backend may take several minutes to initialize...")
        for i in range(360):
            try:
                r = httpx.get(f"{base}/v1/models", timeout=5)
                if r.status_code == 200:
                    http_ok = True
                    break
            except Exception as e:
                if i % 60 == 0 and i > 0:  # Log every 30 seconds (60 * 0.5s)
                    info(f"HTTP health check attempt {i//60 + 1}/6: [Errno {getattr(e, 'errno', 'Unknown')}] {e}")
                elif i == 0:  # Log the first attempt to show what we're checking
                    info(f"Starting health checks on {base}/v1/models...")
            time.sleep(0.5)
        
        if ray_ok and http_ok:
            print(f"ready: {repo_spec} as {alias}  mode=ray  dtype={dtype}  port={port}  pid={pid}")
            print(f"Ray cluster: {cluster_resources.get('CPU', 0)} CPUs available")
            print(f"Try: curl {base}/v1/models")
        elif http_ok and not ray_ok:
            info("warning: HTTP server healthy but Ray cluster issues detected")
            print(f"partial: {repo_spec} as {alias}  mode=ray  dtype={dtype}  port={port}  pid={pid}")
            print(f"Try: curl {base}/v1/models")
        else:
            info("warning: server did not become healthy in time; check Ray cluster and process logs")
            if not ray_ok:
                info("- Ray cluster connectivity failed")
            if not http_ok:
                info("- HTTP endpoint not responding")
            info(f"- Check process logs for PID {pid}")
            info(f"- Try manually: curl {base}/v1/models")
                
    except ImportError:
        info("warning: httpx or Ray not available for health check")
        _standard_health_check(alias, port, repo_spec, pid, "ray", dtype)
    except Exception as e:
        info(f"warning: Ray health check failed: {e}")
        _standard_health_check(alias, port, repo_spec, pid, "ray", dtype)

def _standard_health_check(alias, port, repo_spec, pid, mode, dtype):
    """Standard health check for non-Ray servers."""
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

def _validate_ray_cluster(ray_address):
    """Validate Ray cluster connectivity and resources before starting server."""
    try:
        import ray
        
        info("Validating Ray cluster connectivity...")
        
        # Try to connect to Ray cluster
        try:
            if ray.is_initialized():
                ray.shutdown()
            ray.init(address=ray_address, namespace="alpaca")
        except Exception as e:
            die(f"Failed to connect to Ray cluster at {ray_address}: {e}")
        
        # Check cluster resources
        try:
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            total_cpus = cluster_resources.get('CPU', 0)
            available_cpus = available_resources.get('CPU', 0)
            total_gpus = cluster_resources.get('GPU', 0)
            available_gpus = available_resources.get('GPU', 0)
            
            if total_cpus == 0:
                die("Ray cluster has no CPU resources. Check that workers are running.")
            
            if available_cpus < 1:
                die(f"Ray cluster has insufficient CPU resources: {available_cpus} available, need at least 1")
            
            info(f"Ray cluster validated: {total_cpus} CPUs ({available_cpus} available)")
            if total_gpus > 0:
                info(f"GPU resources: {total_gpus} total ({available_gpus} available)")
            
            # Check for active nodes
            nodes = ray.nodes()
            alive_nodes = [n for n in nodes if n['Alive']]
            if len(alive_nodes) == 0:
                die("No alive nodes found in Ray cluster")
            
            info(f"Ray cluster ready: {len(alive_nodes)} active nodes")
            
        except Exception as e:
            die(f"Failed to validate Ray cluster resources: {e}")
            
    except ImportError:
        die("Ray not installed. Run: pip install ray")
    except Exception as e:
        die(f"Ray cluster validation failed: {e}")
    finally:
        # Leave Ray initialized for vLLM to use
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
        payload = {"model":server.get("repo", "unused"),"messages":[{"role":"user","content":prompt}],"max_tokens":128}
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
# Ray cluster management
# ----------------------------
def ensure_ray():
    try:
        import ray
    except ImportError:
        die("Ray not installed. Run: pip install ray")

def get_ray_processes() -> List[Dict[str, str]]:
    """Get running Ray processes (head/worker nodes)."""
    import psutil
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = proc.info['cmdline'] or []
            # Look for ray processes
            if (any('ray' in arg for arg in cmdline) and 
                any(cmd in ' '.join(cmdline) for cmd in ['ray start', 'ray::'])):
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

def cmd_ray_head(args):
    """Start Ray head node."""
    ensure_ray()
    ensure_state()
    
    reg = read_registry()
    ray_cluster = reg.get("ray_cluster", {})
    
    # Check if head node is already running
    if ray_cluster.get("head_node"):
        head_info = ray_cluster["head_node"]
        try:
            import psutil
            proc = psutil.Process(int(head_info["pid"]))
            if proc.is_running():
                info(f"Ray head node already running (PID {head_info['pid']}) on ports: dashboard={head_info['dashboard_port']}, client={head_info['client_port']}, gcs={head_info['gcs_port']}")
                return
        except:
            pass
    
    # Allocate ports
    dashboard_port = args.dashboard_port or next_free_port(RAY_DASHBOARD_PORT)
    client_port = args.client_port or next_free_port(RAY_CLIENT_PORT)
    gcs_port = args.gcs_port or next_free_port(RAY_GCS_PORT)
    
    # Build Ray head command
    ray_cmd = [
        "ray", "start", "--head",
        "--dashboard-port", str(dashboard_port),
        "--ray-client-server-port", str(client_port),
        "--port", str(gcs_port)
    ]
    
    if args.no_dashboard:
        ray_cmd.extend(["--include-dashboard", "false"])
    
    info(f"Starting Ray head node on ports: dashboard={dashboard_port}, client={client_port}, gcs={gcs_port}")
    
    # Start Ray head node
    result = run(ray_cmd, capture=True)
    if result.returncode != 0:
        die(f"Failed to start Ray head node: {result.stderr}")
    
    # Find the raylet process (Ray head node)
    time.sleep(2)  # Give it time to start
    import psutil
    head_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            proc_name = proc.info['name'] or ''
            if 'raylet' in proc_name.lower():
                head_pid = str(proc.info['pid'])
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not head_pid:
        info("Warning: Could not find raylet process, but Ray may still be running")
    
    # Update registry
    reg.setdefault("ray_cluster", {})
    reg["ray_cluster"]["head_node"] = {
        "pid": head_pid or "unknown",
        "dashboard_port": dashboard_port,
        "client_port": client_port,
        "gcs_port": gcs_port,
        "address": f"127.0.0.1:{gcs_port}",
        "ray_address": f"ray://127.0.0.1:{client_port}",
        "started_at": int(time.time())
    }
    reg["ray_cluster"]["workers"] = reg["ray_cluster"].get("workers", {})
    write_registry(reg)
    
    print(f"Ray head node started successfully!")
    print(f"  Dashboard: http://127.0.0.1:{dashboard_port}")
    print(f"  Client address: ray://127.0.0.1:{client_port}")
    print(f"  GCS address: 127.0.0.1:{gcs_port}")
    print(f"  PID: {head_pid}")
    print(f"\nTo add workers: alpaca ray-worker --address 127.0.0.1:{gcs_port}")
    print(f"To serve models: alpaca serve-ray <model> --address ray://127.0.0.1:{client_port}")

def cmd_ray_worker(args):
    """Add Ray worker node to existing cluster."""
    ensure_ray()
    ensure_state()
    
    if not args.address:
        die("Worker address is required. Use --address <HEAD_IP>:<GCS_PORT>")
    
    reg = read_registry()
    
    # Build Ray worker command
    ray_cmd = ["ray", "start", "--address", args.address]
    
    if args.cpus:
        ray_cmd.extend(["--num-cpus", str(args.cpus)])
    if args.gpus:
        ray_cmd.extend(["--num-gpus", str(args.gpus)])
    
    info(f"Adding Ray worker node to cluster at {args.address}")
    
    # Start Ray worker
    result = run(ray_cmd, capture=True)
    if result.returncode != 0:
        die(f"Failed to start Ray worker: {result.stderr}")
    
    # Find the Ray worker process (python process running default_worker.py)
    time.sleep(2)
    import psutil
    worker_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            proc_name = proc.info['name'] or ''
            cmdline = proc.info['cmdline'] or []
            cmdline_str = ' '.join(cmdline)
            # Look for python processes running ray worker
            if ('python' in proc_name.lower() and 
                'default_worker.py' in cmdline_str and
                (time.time() - proc.info['create_time']) < 30):  # Recently started
                worker_pid = str(proc.info['pid'])
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not worker_pid:
        info("Warning: Could not find Ray worker process, but worker may still be running")
    
    # Update registry
    worker_id = f"worker_{int(time.time())}"
    reg.setdefault("ray_cluster", {}).setdefault("workers", {})
    reg["ray_cluster"]["workers"][worker_id] = {
        "pid": worker_pid,
        "address": args.address,
        "cpus": args.cpus,
        "gpus": args.gpus,
        "started_at": int(time.time())
    }
    write_registry(reg)
    
    print(f"Ray worker node added successfully!")
    print(f"  Connected to: {args.address}")
    print(f"  PID: {worker_pid}")
    if args.cpus:
        print(f"  CPUs: {args.cpus}")
    if args.gpus:
        print(f"  GPUs: {args.gpus}")

def cmd_ray_status(args):
    """Show Ray cluster status."""
    ensure_ray()
    
    reg = read_registry()
    ray_cluster = reg.get("ray_cluster", {})
    
    if not ray_cluster:
        print("No Ray cluster found in registry")
        return
    
    # Check head node
    head_info = ray_cluster.get("head_node")
    if head_info:
        try:
            import psutil
            proc = psutil.Process(int(head_info["pid"]))
            head_status = "running" if proc.is_running() else "stopped"
        except:
            head_status = "unknown"
        
        print("Ray Cluster Status:")
        print(f"  Head Node (PID {head_info['pid']}): {head_status}")
        print(f"    Dashboard: http://127.0.0.1:{head_info['dashboard_port']}")
        print(f"    Client: {head_info['ray_address']}")
        print(f"    GCS: {head_info['address']}")
    else:
        print("No head node found")
        return
    
    # Check workers
    workers = ray_cluster.get("workers", {})
    if workers:
        print(f"  Worker Nodes ({len(workers)}):")
        for worker_id, worker_info in workers.items():
            try:
                import psutil
                proc = psutil.Process(int(worker_info["pid"]))
                worker_status = "running" if proc.is_running() else "stopped"
            except:
                worker_status = "unknown"
            
            print(f"    {worker_id} (PID {worker_info['pid']}): {worker_status}")
            print(f"      Address: {worker_info['address']}")
            if worker_info.get('cpus'):
                print(f"      CPUs: {worker_info['cpus']}")
            if worker_info.get('gpus'):
                print(f"      GPUs: {worker_info['gpus']}")
    else:
        print("  No worker nodes")
    
    # Try to get live cluster info from Ray
    try:
        import ray
        if head_status == "running":
            ray.init(address=head_info['ray_address'])
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            print(f"\nLive Cluster Resources:")
            print(f"  Total CPUs: {cluster_resources.get('CPU', 0)}")
            print(f"  Available CPUs: {available_resources.get('CPU', 0)}")
            if 'GPU' in cluster_resources:
                print(f"  Total GPUs: {cluster_resources.get('GPU', 0)}")
                print(f"  Available GPUs: {available_resources.get('GPU', 0)}")
            ray.shutdown()
    except Exception:
        pass

def cmd_ray_down(args):
    """Stop Ray cluster."""
    ensure_ray()
    
    reg = read_registry()
    ray_cluster = reg.get("ray_cluster", {})
    
    if not ray_cluster:
        info("No Ray cluster found in registry")
        return
    
    stopped_processes = []
    
    # Stop worker nodes first
    workers = ray_cluster.get("workers", {})
    for worker_id, worker_info in workers.items():
        pid = worker_info.get("pid")
        if pid:
            try:
                import psutil
                proc = psutil.Process(int(pid))
                if proc.is_running():
                    proc.terminate()
                    proc.wait(timeout=10)
                    stopped_processes.append(f"worker {worker_id} (PID {pid})")
            except Exception as e:
                info(f"Warning: Could not stop worker {worker_id}: {e}")
    
    # Stop head node
    head_info = ray_cluster.get("head_node")
    if head_info:
        pid = head_info.get("pid")
        if pid:
            try:
                import psutil
                proc = psutil.Process(int(pid))
                if proc.is_running():
                    proc.terminate()
                    proc.wait(timeout=10)
                    stopped_processes.append(f"head node (PID {pid})")
            except Exception as e:
                info(f"Warning: Could not stop head node: {e}")
    
    # Also try ray stop command
    try:
        run(["ray", "stop"], check=False)
    except Exception:
        pass
    
    # Clear registry
    reg.pop("ray_cluster", None)
    write_registry(reg)
    
    if stopped_processes:
        print("Stopped Ray cluster:")
        for proc in stopped_processes:
            print(f"  - {proc}")
    else:
        print("Ray cluster stopped (no running processes found)")

def cmd_cluster_up(args):
    """Start a local Ray cluster with multiple workers."""
    ensure_ray()
    ensure_state()
    
    # Start head node first
    head_args = type('obj', (object,), {
        'dashboard_port': args.dashboard_port,
        'client_port': args.client_port, 
        'gcs_port': args.gcs_port,
        'no_dashboard': False
    })
    cmd_ray_head(head_args)
    
    # Get head node info for worker connections
    reg = read_registry()
    head_info = reg.get("ray_cluster", {}).get("head_node")
    if not head_info:
        die("Failed to start head node")
    
    # Start worker nodes
    num_workers = args.cpu_workers + args.gpu_workers
    if num_workers > 0:
        time.sleep(3)  # Give head node time to fully start
        
        # CPU workers
        for i in range(args.cpu_workers):
            worker_args = type('obj', (object,), {
                'address': head_info['address'],
                'cpus': args.cpus_per_worker,
                'gpus': None
            })
            try:
                cmd_ray_worker(worker_args)
                time.sleep(1)  # Brief pause between workers
            except Exception as e:
                info(f"Warning: Failed to start CPU worker {i+1}: {e}")
        
        # GPU workers
        for i in range(args.gpu_workers):
            worker_args = type('obj', (object,), {
                'address': head_info['address'],
                'cpus': args.cpus_per_worker,
                'gpus': 1
            })
            try:
                cmd_ray_worker(worker_args)
                time.sleep(1)
            except Exception as e:
                info(f"Warning: Failed to start GPU worker {i+1}: {e}")
    
    print(f"\nLocal Ray cluster started with {num_workers} workers!")
    print(f"Use 'alpaca ray-status' to check cluster status")
    print(f"Use 'alpaca serve-ray <model> --address {head_info['ray_address']}' to serve models")

# ----------------------------
# Ray distributed inference
# ----------------------------
def cmd_serve_ray(args):
    """Launch vLLM server with Ray distributed executor."""
    try:
        import ray
    except ImportError:
        die("Ray not installed. Run: pip install ray")
        
    ensure_vllm(); ensure_httpx(); ensure_cache(); ensure_state()
    
    reg = read_registry()
    
    # Auto-detect Ray cluster if no address provided
    ray_address = args.address
    if not ray_address:
        ray_cluster = reg.get("ray_cluster", {})
        head_info = ray_cluster.get("head_node")
        if head_info:
            try:
                import psutil
                proc = psutil.Process(int(head_info["pid"]))
                if proc.is_running():
                    ray_address = head_info["ray_address"]
                    info(f"Auto-detected Ray cluster at {ray_address}")
                else:
                    die("Local Ray head node found but not running. Start with: alpaca ray-head")
            except:
                die("Local Ray head node found but status unknown. Start with: alpaca ray-head")
        else:
            die("No Ray address provided and no local cluster found. Start with: alpaca ray-head or provide --address")
    
    if not ray_address.startswith("ray://"):
        die("Ray address must start with ray:// (example: ray://127.0.0.1:10001)")
    
    # Validate Ray cluster before proceeding
    _validate_ray_cluster(ray_address)
    
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
    env["RAY_ADDRESS"] = ray_address
    env["RAY_NAMESPACE"] = args.namespace
    env["VLLM_USE_MODELSCOPE"] = "false"
    if token := fetch_env_token():
        env["HUGGING_FACE_HUB_TOKEN"] = token

    info(f"Starting vLLM (Ray backend) on port {port}, address={ray_address}")
    info(f"Command: {' '.join(vllm_cmd)}")
    
    # Start the process - don't capture output so we can see startup messages
    proc = subprocess.Popen(
        vllm_cmd,
        env=env,
        # Allow output to be displayed directly to console during startup
        stdout=None,
        stderr=None,
        text=True
    )
    
    # Give process a moment to start and potentially fail fast
    time.sleep(2)
    
    # Check if process is still running
    if proc.poll() is not None:
        die(f"vLLM process failed to start (exit code {proc.returncode}). Check the output above for errors.")
    
    _post_start_health(alias, port, repo_spec, proc.pid, "ray", dtype)

# ----------------------------
# Argument parsing
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser(prog="alpaca", description="Ollama-style wrapper for vLLM with Ray cluster management.")
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

    # Ray cluster management
    sp = sub.add_parser("ray-head", help="Start Ray head node.")
    sp.add_argument("--dashboard-port", type=int, help=f"Dashboard port (default {RAY_DASHBOARD_PORT}).")
    sp.add_argument("--client-port", type=int, help=f"Ray client server port (default {RAY_CLIENT_PORT}).")
    sp.add_argument("--gcs-port", type=int, help=f"Ray GCS port (default {RAY_GCS_PORT}).")
    sp.add_argument("--no-dashboard", action="store_true", help="Disable Ray dashboard.")
    sp.set_defaults(func=cmd_ray_head)

    sp = sub.add_parser("ray-worker", help="Add Ray worker node to existing cluster.")
    sp.add_argument("--address", required=True, help="Head node address, e.g. 127.0.0.1:6379")
    sp.add_argument("--cpus", type=int, help="Number of CPUs for this worker.")
    sp.add_argument("--gpus", type=int, help="Number of GPUs for this worker.")
    sp.set_defaults(func=cmd_ray_worker)

    sp = sub.add_parser("ray-status", help="Show Ray cluster status and resources.")
    sp.set_defaults(func=cmd_ray_status)

    sp = sub.add_parser("ray-down", help="Stop Ray cluster and cleanup.")
    sp.set_defaults(func=cmd_ray_down)

    sp = sub.add_parser("cluster-up", help="Start local Ray cluster with multiple workers.")
    sp.add_argument("--cpu-workers", type=int, default=0, help="Number of CPU-only workers (default 0).")
    sp.add_argument("--gpu-workers", type=int, default=1, help="Number of GPU workers (default 1).")
    sp.add_argument("--cpus-per-worker", type=int, help="CPUs per worker (default auto).")
    sp.add_argument("--dashboard-port", type=int, help=f"Dashboard port (default {RAY_DASHBOARD_PORT}).")
    sp.add_argument("--client-port", type=int, help=f"Ray client server port (default {RAY_CLIENT_PORT}).")
    sp.add_argument("--gcs-port", type=int, help=f"Ray GCS port (default {RAY_GCS_PORT}).")
    sp.set_defaults(func=cmd_cluster_up)

    sp = sub.add_parser("cluster-down", help="Stop local Ray cluster (alias for ray-down).")
    sp.set_defaults(func=cmd_ray_down)

    # Ray shared inference (enhanced)
    sp = sub.add_parser("serve-ray", help="Serve a model using Ray distributed executor.")
    sp.add_argument("model", help="Alias or HF repo spec")
    sp.add_argument("--address", help="Ray client address, e.g. ray://10.0.0.5:10001 (auto-detects if not provided)")
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
