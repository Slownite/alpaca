
#!/usr/bin/env python3
"""
alpaca — Ollama-style wrapper for vLLM (native) with optional Ray shared inference.

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

vLLM Build commands:
  build-vllm [--device cpu|gpu|auto] [--ref main] [--wheel] [--no-test]
  build-status
  build-clean [--source] [--cache] [--all]

Ray shared inference:
  ray-head [--dashboard-port 8265] [--client-port 10001] [--gcs-port 6379]
  ray-worker --address <HEAD_IP:6379> [--cpus N] [--gpu]
  serve-ray <model-or-alias> --address ray://<HEAD_IP>:10001 [--namespace vllm]
            [--port <p>] [--dtype auto|float32|bf16|fp16] [--max-seqs N]
  ray-down  (stop Ray head/workers launched by alpaca)

Convenience cluster helpers (single host):
  cluster-up   [--cpu-workers N] [--gpu-workers N] [--cpus-per-worker M]
               [--dashboard-port 8265] [--client-port 10001] [--gcs-port 6379]
  cluster-down (alias of ray-down)

Notes:
  - Native vLLM execution; uses HF cache at ~/.cache/huggingface.
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

try:
    import argcomplete
    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False

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

def warn(msg: str):
    print(f"warning: {msg}", file=sys.stderr)

def run(cmd: List[str], check=True, capture=False, env=None, cwd=None):
    kwargs = {}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        kwargs["text"] = True
    if cwd:
        kwargs["cwd"] = cwd
    return subprocess.run(cmd, check=check, env=env, **kwargs)

def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def ensure_vllm():
    try:
        import vllm
    except ImportError:
        die("vLLM not installed. Options:\n" +
            "  pip install vllm                    # Install from PyPI\n" +
            "  alpaca build-vllm --device cpu      # Build from source for CPU\n" +
            "  alpaca build-vllm --device gpu      # Build from source for GPU\n" +
            "  alpaca build-vllm --device auto     # Auto-detect CPU/GPU")

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name.replace("/", "_").replace(":", "_"))

def alias_to_state_dir(alias: str) -> Path:
    return STATE_DIR / "models" / sanitize(alias)

def process_name(alias: str) -> str:
    return f"alpaca_{sanitize(alias)}"

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
    """Check if GPU is available for native vLLM."""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except ImportError:
        try:
            # Fallback: check nvidia-smi directly
            result = run(["nvidia-smi", "-L"], check=False, capture=True)
            return result.returncode == 0 and "GPU" in (result.stdout or "")
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

def get_native_processes() -> List[Dict[str, str]]:
    """Get running native alpaca processes."""
    reg = read_registry()
    servers = reg.get("servers", {})
    rows = []
    
    for alias, info in servers.items():
        if not info.get("native"):
            continue
            
        pid = info.get("process_id")
        if not pid:
            continue
            
        # Check if process is still running
        try:
            import psutil
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                status = f"Running ({proc.status()})"
            else:
                status = "Stopped"
        except ImportError:
            # Fallback without psutil
            try:
                os.kill(pid, 0)  # Send null signal to check if process exists
                status = "Running"
            except (ProcessLookupError, OSError):
                status = "Stopped"
        except Exception:
            status = "Unknown"
            
        rows.append({
            "name": f"native_{alias}",
            "status": status,
            "port": str(info.get("port", "")),
            "model": info.get("repo", ""),
            "pid": str(pid),
            "mode": info.get("mode", "native")
        })
    
    return rows

def fetch_env_token() -> Optional[str]:
    return os.environ.get("HUGGING_FACE_HUB_TOKEN")

# ----------------------------
# vLLM Build Functions
# ----------------------------
def get_vllm_build_dir() -> Path:
    return STATE_DIR / "vllm_source"

def get_build_status() -> Dict[str, Any]:
    reg = read_registry()
    return reg.get("build", {})

def update_build_status(status: Dict[str, Any]):
    reg = read_registry()
    reg.setdefault("build", {}).update(status)
    write_registry(reg)

def have_cuda() -> bool:
    """Check if NVIDIA CUDA is available."""
    if sys.platform == "darwin":  # macOS doesn't support NVIDIA CUDA
        return False
    return have("nvidia-smi")

def detect_cuda_version() -> Optional[str]:
    """Detect CUDA version from nvidia-smi."""
    if not have_cuda():
        return None
    try:
        result = run(["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"], 
                    capture=True, check=False)
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip().split('\n')[0].strip()
    except Exception:
        pass
    return None

def have_avx512() -> bool:
    """Check if CPU supports AVX-512 for vLLM CPU backend."""
    try:
        if sys.platform == "darwin":  # macOS
            result = run(["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"], 
                        capture=True, check=False)
            if result.returncode == 0:
                return "avx512" in result.stdout.lower()
        elif have("lscpu"):
            result = run(["lscpu"], capture=True, check=False)
            if result.returncode == 0:
                return "avx512" in result.stdout.lower()
        elif Path("/proc/cpuinfo").exists():
            with open("/proc/cpuinfo") as f:
                return "avx512" in f.read().lower()
    except Exception:
        pass
    return False

def ensure_vllm_source(ref: str = "main") -> Path:
    """Ensure vLLM source code is available."""
    vllm_dir = get_vllm_build_dir()
    
    if vllm_dir.exists() and (vllm_dir / ".git").exists():
        info(f"vLLM source found at {vllm_dir}")
        # Update existing repo
        try:
            run(["git", "fetch", "--tags", "--quiet"], cwd=vllm_dir, check=False)
            run(["git", "checkout", ref], cwd=vllm_dir)
            run(["git", "submodule", "update", "--init", "--recursive"], cwd=vllm_dir, check=False)
        except Exception as e:
            warn(f"Could not update vLLM source: {e}")
    else:
        info(f"Cloning vLLM source to {vllm_dir}")
        vllm_dir.parent.mkdir(parents=True, exist_ok=True)
        if vllm_dir.exists():
            shutil.rmtree(vllm_dir)
        
        run(["git", "clone", "--recursive", "https://github.com/vllm-project/vllm", str(vllm_dir)])
        run(["git", "checkout", ref], cwd=vllm_dir)
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=vllm_dir, check=False)
    
    return vllm_dir

def install_build_deps():
    """Install build dependencies."""
    info("Installing build dependencies...")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools", "packaging"])
    run([sys.executable, "-m", "pip", "install", "ninja", "cmake", "numpy"])

def install_torch_cpu():
    """Install PyTorch for CPU."""
    info("Installing PyTorch (CPU)")
    
    # Check if torch is already installed
    try:
        result = run([sys.executable, "-c", "import torch; print(torch.__version__)"], 
                    capture=True, check=False)
        if result.returncode == 0:
            info("Existing PyTorch detected; will build vLLM against it.")
            return
    except Exception:
        pass
    
    # Check Python version for compatibility
    py_version = sys.version_info
    if py_version >= (3, 13):
        warn("Python 3.13+ detected. Installing PyTorch nightly build...")
        # Install nightly build for Python 3.13
        run([sys.executable, "-m", "pip", "install", 
             "--pre", "torch", "torchvision", "torchaudio", 
             "--index-url", "https://download.pytorch.org/whl/nightly/cpu"])
    elif sys.platform == "darwin":  # macOS
        run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
    else:  # Linux and others
        run([sys.executable, "-m", "pip", "install", 
             "--index-url", "https://download.pytorch.org/whl/cpu",
             "torch", "torchvision", "torchaudio"])

def install_torch_cuda():
    """Install PyTorch for CUDA."""
    # Check if torch is already installed
    try:
        result = run([sys.executable, "-c", "import torch; print(torch.__version__)"], 
                    capture=True, check=False)
        if result.returncode == 0:
            info("Existing PyTorch detected; will build vLLM against it.")
            return
    except Exception:
        pass
    
    cuda_ver = detect_cuda_version()
    if not cuda_ver:
        die("No CUDA detected. Install CUDA drivers or use --device cpu")
    
    info(f"Installing PyTorch for CUDA {cuda_ver}")
    # Use cu118 for CUDA 11.8+ compatibility
    run([sys.executable, "-m", "pip", "install",
         "--index-url", "https://download.pytorch.org/whl/cu118", 
         "torch", "torchvision", "torchaudio"])

def build_vllm_from_source(device: str, vllm_dir: Path, editable: bool = True):
    """Build vLLM from source."""
    info(f"Building vLLM for {device}")
    
    env = os.environ.copy()
    if device == "cpu":
        env["VLLM_TARGET_DEVICE"] = "cpu"
        # CPU-specific optimizations
        env["CUDA_VISIBLE_DEVICES"] = ""
        
        if not have_avx512():
            warn("AVX-512 not detected; vLLM CPU backend may not work optimally")
        
        # Set Intel CPU optimizations if available
        if sys.platform.startswith('linux'):
            # Enable Intel CPU optimizations
            env["VLLM_CPU_ENABLE_AVX512"] = "1" if have_avx512() else "0"
        
        info("Building with CPU optimizations enabled")
    else:
        # GPU-specific settings
        info("Building with GPU support")
    
    # Use existing torch helper if available
    use_torch_script = vllm_dir / "use_existing_torch.py"
    if use_torch_script.exists():
        run([sys.executable, str(use_torch_script)], cwd=vllm_dir, check=False)
    
    if editable:
        run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation", "-v"], 
            cwd=vllm_dir, env=env)
    else:
        run([sys.executable, "setup.py", "bdist_wheel"], cwd=vllm_dir, env=env)
        # Find and install the wheel
        dist_dir = vllm_dir / "dist"
        wheels = list(dist_dir.glob("vllm-*.whl"))
        if wheels:
            run([sys.executable, "-m", "pip", "install", str(wheels[-1])])

def test_vllm_installation(device: str):
    """Test vLLM installation with a small model."""
    info(f"Testing vLLM ({device}) installation...")
    
    test_code = '''
import torch
from vllm import LLM, SamplingParams

# Use a small model for testing
model = "facebook/opt-125m"
prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

try:
    llm = LLM(model=model, enforce_eager=True)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"Generated: {output.outputs[0].text.strip()}")
    print("✓ vLLM installation test passed")
except Exception as e:
    print(f"✗ vLLM test failed: {e}")
    exit(1)
'''
    
    result = run([sys.executable, "-c", test_code], capture=True, check=False)
    if result.returncode == 0:
        info("vLLM installation test passed ✓")
        if result.stdout:
            print(result.stdout)
    else:
        warn("vLLM installation test failed")
        if result.stderr:
            print(result.stderr, file=sys.stderr)

def cleanup_stopped_processes():
    """Clean up registry entries for stopped native processes."""
    reg = read_registry()
    servers = reg.get("servers", {})
    
    for alias in list(servers.keys()):
        server_info = servers[alias]
        if not server_info.get("native"):
            continue
            
        pid = server_info.get("process_id")
        if not pid:
            continue
            
        # Check if process is still running
        try:
            import psutil
            if not psutil.pid_exists(pid):
                del servers[alias]
        except ImportError:
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, OSError):
                del servers[alias]
        except Exception:
            pass
    
    write_registry(reg)

# ----------------------------
# Argument completion functions
# ----------------------------
def complete_models(prefix, parsed_args, **kwargs):
    """Complete model names/aliases from registry."""
    try:
        reg = read_registry()
        aliases = list(reg.get("aliases", {}).keys())
        return [alias for alias in aliases if alias.startswith(prefix)]
    except Exception:
        return []

def complete_running_models(prefix, parsed_args, **kwargs):
    """Complete running model names/aliases."""
    try:
        reg = read_registry()
        servers = list(reg.get("servers", {}).keys())
        return [server for server in servers if server.startswith(prefix)]
    except Exception:
        return []

# ----------------------------
# Subcommand handlers (local)
# ----------------------------
def cmd_pull(args):
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
    # Clean up stopped processes first
    cleanup_stopped_processes()
    
    # Show native processes
    native_rows = get_native_processes()
    
    if not native_rows:
        print("(no running alpaca-managed processes)")
        return
    
    # Native processes
    for r in native_rows:
        print(f"{r['name']:32s}  {r['mode']:10s}  port={r['port']:5s}  {r['model']:40s}  pid={r['pid']}  {r['status']}")

def cmd_stop(args):
    reg = read_registry()
    alias, _ = resolve_alias_or_repo(args.model, reg)
    
    server_info = reg.get("servers", {}).get(alias)
    if not server_info:
        die(f"No running server found for alias '{alias}'.")
    
    pid = server_info.get("process_id")
    if not pid:
        die(f"No process ID found for {alias}")
    
    try:
        import signal
        os.kill(pid, signal.SIGTERM)
        info(f"Stopped native process {alias} (pid={pid})")
        
        # Remove from registry
        del reg["servers"][alias]
        write_registry(reg)
    except (ProcessLookupError, OSError):
        info(f"Process {alias} (pid={pid}) not found or already stopped")
        # Clean up registry entry anyway
        if alias in reg.get("servers", {}):
            del reg["servers"][alias]
            write_registry(reg)

def cmd_rm(args):
    reg = read_registry()
    alias, _ = resolve_alias_or_repo(args.model, reg)
    
    # Stop the process if running
    server_info = reg.get("servers", {}).get(alias)
    if server_info:
        pid = server_info.get("process_id")
        if pid:
            try:
                import signal
                os.kill(pid, signal.SIGTERM)
                info(f"Stopped process {alias} (pid={pid})")
            except (ProcessLookupError, OSError):
                pass
        
        # Remove from servers registry
        del reg["servers"][alias]
        write_registry(reg)
        info(f"Removed server {alias}")
    
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


def cmd_serve(args):
    cmd_serve_native(args)

def cmd_serve_native(args):
    """Serve model using native vLLM installation."""
    ensure_vllm()
    ensure_cache(); ensure_state()
    reg = read_registry()
    alias, repo_spec = _model_for_token(reg, args.model)
    mode = _decide_backend("gpu" if args.gpu else "cpu" if args.cpu else None)
    dtype = _dtype_for_mode(mode, args.dtype)
    port = _find_or_allocate_port(reg, alias, args.port)
    
    # Build vLLM command for native execution
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", repo_spec,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", dtype,
        "--max-num-seqs", str(args.max_seqs or MAX_SEQS_DEFAULT)
    ]
    
    if mode == "cpu":
        vllm_cmd += ["--device", "cpu", "--tensor-parallel-size", "1"]
    
    # Set environment variables
    env = os.environ.copy()
    env["VLLM_USE_MODELSCOPE"] = "false"
    if mode == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    
    info(f"Starting native vLLM ({mode}, dtype={dtype}) on port {port} for {repo_spec} …")
    
    # Start as background process
    import subprocess
    proc = subprocess.Popen(vllm_cmd, env=env)
    
    # Store process info
    reg.setdefault("servers", {})
    reg["servers"][alias] = {
        "process_id": proc.pid,
        "port": port, "mode": f"native-{mode}", "dtype": dtype,
        "repo": repo_spec, "native": True, "started_at": int(time.time())
    }
    write_registry(reg)
    
    print(f"Native vLLM started: {repo_spec} as {alias}  mode={mode}  dtype={dtype}  port={port}  pid={proc.pid}")
    print(f"Try: curl http://127.0.0.1:{port}/v1/models")



def cmd_logs(args):
    reg = read_registry()
    alias, _ = resolve_alias_or_repo(args.model, reg)
    server_info = reg.get("servers", {}).get(alias)
    
    if not server_info:
        die(f"No running server found for alias '{alias}'.")
    
    pid = server_info.get("process_id")
    if not pid:
        die(f"No process ID found for {alias}")
    
    # For native processes, we can't get logs directly like Docker
    # Instead, we'll suggest checking system logs or running with output redirection
    info(f"Native process logs for {alias} (pid={pid}):")
    info("Note: Native processes don't provide centralized logs like Docker.")
    info("Consider running the process in the foreground or redirecting output to a log file.")
    info(f"You can check if the process is running with: ps -p {pid}")
    
    # Try to show basic process info
    try:
        import psutil
        proc = psutil.Process(pid)
        info(f"Process status: {proc.status()}")
        info(f"Memory usage: {proc.memory_info().rss / 1024 / 1024:.1f} MB")
        info(f"CPU percent: {proc.cpu_percent()}%")
        info(f"Command: {' '.join(proc.cmdline())}")
    except ImportError:
        info("Install psutil for more process information: pip install psutil")
    except Exception as e:
        info(f"Could not get process info: {e}")

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
# vLLM Build Commands
# ----------------------------
def cmd_build_vllm(args):
    """Build vLLM from source."""
    ensure_state()
    
    device = args.device
    if device == "auto":
        device = "gpu" if have_cuda() else "cpu"
        info(f"Auto-detected device: {device}")
    
    # Validate device
    if device not in ["cpu", "gpu"]:
        die("Device must be 'cpu', 'gpu', or 'auto'")
    
    if device == "gpu" and not have_cuda():
        die("GPU device requested but NVIDIA CUDA not found. Use --device cpu or install CUDA.")
    
    # Check if already built and not forcing rebuild
    status = get_build_status()
    if not args.force_rebuild and status.get("status") == "completed" and status.get("device") == device:
        try:
            import vllm
            info(f"vLLM already built for {device}. Use --force-rebuild to rebuild.")
            return
        except ImportError:
            info("vLLM build exists but not importable, rebuilding...")
    
    start_time = time.time()
    
    try:
        # Update build status
        update_build_status({
            "status": "building",
            "device": device,
            "started_at": int(start_time),
            "ref": args.ref
        })
        
        # Step 1: Get source
        info("Step 1/5: Fetching vLLM source code...")
        vllm_dir = ensure_vllm_source(args.ref)
        
        # Step 2: Install build dependencies
        info("Step 2/5: Installing build dependencies...")
        install_build_deps()
        
        # Step 3: Install PyTorch
        info("Step 3/5: Installing PyTorch...")
        if device == "cpu":
            install_torch_cpu()
        else:
            install_torch_cuda()
        
        # Step 4: Build vLLM
        info("Step 4/5: Building vLLM from source...")
        build_vllm_from_source(device, vllm_dir, editable=not args.wheel)
        
        # Step 5: Test installation
        if not args.no_test:
            info("Step 5/5: Testing vLLM installation...")
            test_vllm_installation(device)
        else:
            info("Step 5/5: Skipped (--no-test)")
        
        # Update build status
        build_time = int(time.time() - start_time)
        update_build_status({
            "status": "completed",
            "device": device,
            "completed_at": int(time.time()),
            "build_time_seconds": build_time,
            "ref": args.ref,
            "source_path": str(vllm_dir)
        })
        
        info(f"✅ vLLM build completed successfully in {build_time}s")
        print(f"Built vLLM for {device} from {args.ref}")
        
    except subprocess.CalledProcessError as e:
        # Update build status
        error_msg = f"Command failed: {' '.join(e.cmd)}"
        update_build_status({
            "status": "failed",
            "error": error_msg,
            "failed_at": int(time.time())
        })
        die(f"vLLM build failed: {error_msg}")
    except Exception as e:
        # Update build status
        update_build_status({
            "status": "failed",
            "error": str(e),
            "failed_at": int(time.time())
        })
        die(f"vLLM build failed: {e}")

def cmd_build_status(_args):
    """Show vLLM build status."""
    status = get_build_status()
    if not status:
        print("No build information available.")
        return
    
    print("vLLM Build Status:")
    print(f"  Status: {status.get('status', 'unknown')}")
    
    if status.get("device"):
        print(f"  Device: {status['device']}")
    
    if status.get("ref"):
        print(f"  Version: {status['ref']}")
    
    if status.get("source_path"):
        print(f"  Source: {status['source_path']}")
    
    if status.get("build_time_seconds"):
        print(f"  Build time: {status['build_time_seconds']}s")
    
    if status.get("completed_at"):
        completed = status["completed_at"]
        print(f"  Completed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(completed))}")
    
    if status.get("error"):
        print(f"  Error: {status['error']}")
    
    # Check if vLLM is actually importable
    try:
        import vllm
        print(f"  Import status: ✓ Available (version: {vllm.__version__})")
    except ImportError:
        print("  Import status: ✗ Not available")

def cmd_build_clean(args):
    """Clean vLLM build artifacts."""
    vllm_dir = get_vllm_build_dir()
    
    if args.source and vllm_dir.exists():
        info(f"Removing vLLM source directory: {vllm_dir}")
        shutil.rmtree(vllm_dir)
    
    if args.cache:
        # Clean pip cache
        info("Cleaning pip cache...")
        run([sys.executable, "-m", "pip", "cache", "purge"], check=False)
    
    if args.all:
        # Remove build status
        reg = read_registry()
        if "build" in reg:
            del reg["build"]
            write_registry(reg)
            info("Cleared build status")
        
        # Try to uninstall vLLM
        try:
            info("Uninstalling vLLM...")
            run([sys.executable, "-m", "pip", "uninstall", "vllm", "-y"], check=False)
        except Exception:
            pass
    
    print("Build cleanup completed.")

# ----------------------------
# Ray shared inference
# ----------------------------
def cmd_ray_head(args):
    """Start a native Ray head node."""
    try:
        import ray
    except ImportError:
        die("Ray not installed. Install with: pip install ray")
    
    dash = args.dashboard_port or RAY_DASHBOARD_PORT
    client = args.client_port or RAY_CLIENT_PORT
    gcs = args.gcs_port or RAY_GCS_PORT
    
    # Check if Ray is already running
    try:
        ray.init(address="auto", ignore_reinit_error=True)
        info("Ray cluster already running")
        ray.shutdown()
        return
    except Exception:
        pass
    
    info(f"Starting Ray head (dashboard {dash}, client {client}, gcs {gcs}) …")
    
    # Start Ray head
    ray.init(
        _node_ip_address="0.0.0.0",
        port=gcs,
        dashboard_host="0.0.0.0",
        dashboard_port=dash,
        ray_client_server_port=client,
        num_cpus=0  # Head node contributes no CPUs
    )
    
    print("Ray head ready.")
    print(f"Dashboard: http://127.0.0.1:{dash}")
    print(f"Workers join with: alpaca ray-worker --address  <HEAD_IP:{gcs}>")
    print(f"vLLM connects with: alpaca serve-ray … --address ray://<HEAD_IP>:{client}")
    print("Press Ctrl+C to stop the head node.")
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ray.shutdown()
        print("\nRay head stopped.")

def cmd_ray_worker(args):
    """Start a native Ray worker node."""
    try:
        import ray
    except ImportError:
        die("Ray not installed. Install with: pip install ray")
    
    if not args.address:
        die("--address is required (format HEAD_IP:6379)")
    
    cpus = max(1, int(args.cpus)) if args.cpus else None  # None = Ray auto-detects
    
    info(f"Starting Ray worker (gpu={args.gpu}, cpus={'auto' if cpus is None else cpus}) -> {args.address}")
    
    # Configure Ray worker
    init_kwargs = {
        "address": args.address,
        "ignore_reinit_error": True
    }
    
    if cpus is not None:
        init_kwargs["num_cpus"] = cpus
    
    # Start Ray worker
    ray.init(**init_kwargs)
    
    print("Ray worker started.")
    print("Press Ctrl+C to stop the worker.")
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ray.shutdown()
        print("\nRay worker stopped.")

def cmd_ray_down(_args):
    """Stop Ray cluster."""
    try:
        import ray
        ray.shutdown()
        info("Ray cluster stopped.")
    except ImportError:
        die("Ray not installed.")
    except Exception as e:
        info(f"Ray cluster may already be stopped: {e}")

def cmd_serve_ray(args):
    """Launch native vLLM server attached to a Ray cluster (multi-node shared inference)."""
    try:
        import vllm
        import ray
    except ImportError:
        die("vLLM and Ray must be installed. Install with: pip install vllm ray")
    
    ensure_cache(); ensure_state()
    if not args.address.startswith("ray://"):
        die("RAY address must start with ray:// (example: ray://<HEAD_IP>:10001)")
    
    reg = read_registry()
    alias, repo_spec = _model_for_token(reg, args.model)
    
    dtype = args.dtype or "auto"
    port = _find_or_allocate_port(reg, alias, args.port)
    
    # Build vLLM command for Ray backend
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
    env["VLLM_USE_MODELSCOPE"] = "false"
    env["RAY_ADDRESS"] = args.address
    env["RAY_NAMESPACE"] = args.namespace
    
    info(f"Starting vLLM (Ray backend) on port {port}, namespace={args.namespace}, address={args.address}")
    
    # Start as background process
    import subprocess
    proc = subprocess.Popen(vllm_cmd, env=env)
    
    # Store process info
    reg.setdefault("servers", {})
    reg["servers"][alias] = {
        "process_id": proc.pid,
        "port": port, "mode": "ray", "dtype": dtype,
        "repo": repo_spec, "native": True, 
        "ray_address": args.address, "ray_namespace": args.namespace,
        "started_at": int(time.time())
    }
    write_registry(reg)
    
    print(f"Ray vLLM started: {repo_spec} as {alias}  mode=ray  dtype={dtype}  port={port}  pid={proc.pid}")
    print(f"Try: curl http://127.0.0.1:{port}/v1/models")

# ----------------------------
# Cluster helpers (single host)
# ----------------------------
def cmd_cluster_up(args):
    """Start a Ray head + local workers on the same machine."""
    try:
        import ray
    except ImportError:
        die("Ray not installed. Install with: pip install ray")
    
    dash = args.dashboard_port or RAY_DASHBOARD_PORT
    client = args.client_port or RAY_CLIENT_PORT
    gcs = args.gcs_port or RAY_GCS_PORT
    
    info(f"Starting Ray cluster (dashboard {dash}, client {client}, gcs {gcs}) …")
    
    # Calculate total CPUs and GPUs
    cpu_workers = args.cpu_workers or 1
    gpu_workers = args.gpu_workers or 0
    cpus_per_worker = args.cpus_per_worker
    
    total_cpus = None
    if cpus_per_worker:
        total_cpus = cpu_workers * cpus_per_worker
    
    # Start Ray cluster
    ray_kwargs = {
        "_node_ip_address": "0.0.0.0",
        "port": gcs,
        "dashboard_host": "0.0.0.0",
        "dashboard_port": dash,
        "ray_client_server_port": client,
        "ignore_reinit_error": True
    }
    
    if total_cpus:
        ray_kwargs["num_cpus"] = total_cpus
    
    # Auto-detect GPUs if requested
    if gpu_workers > 0 and gpu_available():
        try:
            import torch
            ray_kwargs["num_gpus"] = min(gpu_workers, torch.cuda.device_count())
        except ImportError:
            pass
    
    ray.init(**ray_kwargs)
    
    print("cluster-up complete.")
    print(f"Dashboard: http://127.0.0.1:{dash}")
    print(f"Use serve-ray … --address ray://127.0.0.1:{client}")
    print("Press Ctrl+C to stop the cluster.")
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ray.shutdown()
        print("\nRay cluster stopped.")

def cmd_cluster_down(_args):
    """Tear down local Ray cluster."""
    cmd_ray_down(_args)

# ----------------------------
# Argument parsing
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser(prog="alpaca", description="Ollama-style wrapper for vLLM (native) + Ray.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pull", help="Prefetch a HF repo into local cache.")
    sp.add_argument("repo", help="HF repo spec, e.g. org/name[:rev]")
    sp.add_argument("--alias", help="Local alias name to register.")
    sp.set_defaults(func=cmd_pull)

    sp = sub.add_parser("ls", help="List cached models (aliases).")
    sp.add_argument("--size", action="store_true", help="Show total cache size.")
    sp.set_defaults(func=cmd_ls)

    sp = sub.add_parser("serve", help="Serve a model via vLLM (local CPU/GPU).")
    model_arg = sp.add_argument("model", help="Alias or HF repo spec")
    if ARGCOMPLETE_AVAILABLE:
        model_arg.completer = complete_models
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
    model_arg = sp.add_argument("model", help="Alias or repo spec used to start it.")
    if ARGCOMPLETE_AVAILABLE:
        model_arg.completer = complete_running_models
    sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("rm", help="Remove server and (optionally) alias.")
    model_arg = sp.add_argument("model", help="Alias or repo spec.")
    if ARGCOMPLETE_AVAILABLE:
        model_arg.completer = complete_models
    sp.add_argument("--purge", action="store_true", help="Also remove alias from registry.")
    sp.set_defaults(func=cmd_rm)

    sp = sub.add_parser("logs", help="Tail server logs.")
    model_arg = sp.add_argument("model", help="Alias or repo spec.")
    if ARGCOMPLETE_AVAILABLE:
        model_arg.completer = complete_running_models
    sp.add_argument("-f", "--follow", action="store_true", help="Follow logs.")
    sp.set_defaults(func=cmd_logs)

    sp = sub.add_parser("run", help="Send a single request to a running server.")
    model_arg = sp.add_argument("model", help="Alias or repo spec.")
    if ARGCOMPLETE_AVAILABLE:
        model_arg.completer = complete_running_models
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
    model_arg = sp.add_argument("model", help="Alias or HF repo spec")
    if ARGCOMPLETE_AVAILABLE:
        model_arg.completer = complete_models
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

    # vLLM Build commands
    sp = sub.add_parser("build-vllm", help="Build vLLM from source for CPU or GPU.")
    sp.add_argument("--device", choices=["cpu", "gpu", "auto"], default="auto", 
                    help="Target device (default: auto-detect)")
    sp.add_argument("--ref", default="main", help="Git reference to build (default: main)")
    sp.add_argument("--wheel", action="store_true", help="Build wheel instead of editable install")
    sp.add_argument("--no-test", action="store_true", help="Skip installation test")
    sp.add_argument("--force-rebuild", action="store_true", help="Force rebuild even if already built")
    sp.add_argument("--cpu-backend", choices=["onednn", "x86"], help="CPU backend for optimizations")
    sp.set_defaults(func=cmd_build_vllm)

    sp = sub.add_parser("build-status", help="Show vLLM build status and information.")
    sp.set_defaults(func=cmd_build_status)

    sp = sub.add_parser("build-clean", help="Clean vLLM build artifacts and cache.")
    sp.add_argument("--source", action="store_true", help="Remove source directory")
    sp.add_argument("--cache", action="store_true", help="Clear pip cache")
    sp.add_argument("--all", action="store_true", help="Remove everything including vLLM installation")
    sp.set_defaults(func=cmd_build_clean)

    return p

# ----------------------------
# Main
# ----------------------------
def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    parser = build_parser()
    if ARGCOMPLETE_AVAILABLE:
        argcomplete.autocomplete(parser)
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
