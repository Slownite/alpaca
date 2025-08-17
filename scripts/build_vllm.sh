
#!/usr/bin/env bash
# build_vllm.sh — add vLLM as a submodule (or clone) and build for CPU or GPU
# Usage examples:
#   ./build_vllm.sh --device cpu
#   ./build_vllm.sh --device gpu
#   ./build_vllm.sh --device auto --vllm-path third_party/vllm --ref v0.10.0
#
# Notes:
# - GPU build expects NVIDIA CUDA. If none is found and --device auto is used, it falls back to CPU.
# - CPU backend in vLLM currently targets x86 with AVX-512; this script warns if not detected.
# - The script creates a Python venv (default: .venv) unless you pass --no-venv.

set -Eeuo pipefail

### Defaults #############################################################
DEVICE="auto"                     # cpu | gpu | auto
VLLM_PATH="third_party/vllm"      # where to place the submodule/clone
VLLM_REF="main"                   # tag/branch/commit (e.g. v0.10.0)
PYTHON_BIN="python3"              # python executable to use
VENV_DIR=".venv"                  # virtualenv location
INSTALL_MODE="editable"           # editable | wheel
USE_SUBMODULE="yes"               # yes | no (fallback to clone if repo has no .git)
CREATE_VENV="yes"                 # yes | no
QUIET_PIP=""                      # set to -q if you prefer quieter installs
##########################################################################

log()  { printf "\033[1;34m[build_vllm]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[build_vllm]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[build_vllm]\033[0m %s\n" "$*" >&2; exit 1; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# ---------- arg parsing ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)          DEVICE="${2:-}"; shift 2 ;;
    --vllm-path)       VLLM_PATH="${2:-}"; shift 2 ;;
    --ref)             VLLM_REF="${2:-}"; shift 2 ;;
    --python)          PYTHON_BIN="${2:-}"; shift 2 ;;
    --venv)            VENV_DIR="${2:-}"; shift 2 ;;
    --editable)        INSTALL_MODE="editable"; shift 1 ;;
    --wheel)           INSTALL_MODE="wheel"; shift 1 ;;
    --no-submodule)    USE_SUBMODULE="no"; shift 1 ;;
    --no-venv)         CREATE_VENV="no"; shift 1 ;;
    --quiet)           QUIET_PIP="-q"; shift 1 ;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0 ;;
    *) err "Unknown option: $1" ;;
  esac
done

# ---------- helpers ----------
detect_cuda_tag() {
  # Map CUDA version to PyTorch index tag (e.g., 12.4 -> cu124, 12.1 -> cu121, 11.8 -> cu118)
  local ver="$1"
  local major="${ver%%.*}"
  local minor="${ver#*.}"; minor="${minor%%.*}"
  if [[ "$major" -ge 12 ]]; then
    printf "cu12%s" "${minor:-1}"; return 0
  elif [[ "$major" -eq 11 && "$minor" -ge 8 ]]; then
    printf "cu118"; return 0
  fi
  return 1
}

detect_cuda_version() {
  if have_cmd nvidia-smi; then
    nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -n1 | awk '{print $1}'
  fi
}

have_avx512() {
  if have_cmd lscpu; then
    lscpu | tr '[:upper:]' '[:lower:]' | grep -q 'avx512'
  elif [[ -r /proc/cpuinfo ]]; then
    tr '[:upper:]' '[:lower:]' </proc/cpuinfo | grep -q 'avx512'
  else
    return 1
  fi
}

ensure_python() {
  have_cmd "$PYTHON_BIN" || err "Python not found: $PYTHON_BIN"
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1 || err "Python 3.9–3.12 required."
import sys
maj, min = sys.version_info[:2]
sys.exit(0 if (maj==3 and 9<=min<=12) else 1)
PY
}

ensure_venv() {
  if [[ "$CREATE_VENV" == "yes" ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
      log "Creating venv at $VENV_DIR"
      "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    PYTHON_BIN="python"
  fi
}

pip_install() {
  # Wrapper to keep things consistent
  $PYTHON_BIN -m pip install $QUIET_PIP "$@"
}

ensure_base_build_tools() {
  log "Upgrading pip and base build tools"
  pip_install --upgrade pip
  pip_install --upgrade wheel setuptools packaging
  pip_install ninja cmake numpy
}

ensure_repo() {
  if [[ -d "$VLLM_PATH/.git" ]]; then
    log "Found existing vLLM repo at $VLLM_PATH"
    (cd "$VLLM_PATH" && git fetch --tags --quiet || true)
  elif [[ -d .git && "$USE_SUBMODULE" == "yes" ]]; then
    log "Adding vLLM as a submodule -> $VLLM_PATH (ref: $VLLM_REF)"
    git submodule add https://github.com/vllm-project/vllm "$VLLM_PATH" || true
    git submodule update --init --recursive "$VLLM_PATH"
    (cd "$VLLM_PATH" && git checkout "$VLLM_REF" && git submodule update --init --recursive)
  else
    warn "Not in a git repo or --no-submodule set; cloning instead."
    git clone --recursive https://github.com/vllm-project/vllm "$VLLM_PATH"
    (cd "$VLLM_PATH" && git checkout "$VLLM_REF" && git submodule update --init --recursive)
  fi
}

install_torch_cpu() {
  log "Installing PyTorch (CPU)"
  pip_install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
}

install_torch_cuda() {
  if $PYTHON_BIN - <<'PY' 2>/dev/null; then
import torch; print(torch.__version__)
PY
  then
    log "Existing PyTorch detected; will build vLLM against it."
    return 0
  fi

  local cuda_ver cuda_tag
  cuda_ver="$(detect_cuda_version || true)"
  if [[ -z "${cuda_ver:-}" ]]; then
    err "No CUDA detected by nvidia-smi and no existing torch; aborting GPU install. Try --device cpu or install CUDA/torch first."
  fi
  cuda_tag="$(detect_cuda_tag "$cuda_ver" || true)"
  if [[ -z "${cuda_tag:-}" ]]; then
    err "Unsupported/unknown CUDA version '$cuda_ver' for auto-install. Install torch manually, then re-run."
  fi

  log "Installing PyTorch (CUDA $cuda_ver → index tag $cuda_tag)"
  pip_install --index-url "https://download.pytorch.org/whl/${cuda_tag}" torch torchvision torchaudio
}

build_vllm_cpu() {
  log "Building vLLM (CPU backend)"
  if ! have_avx512; then
    warn "AVX-512 not detected; vLLM CPU backend may not work on this machine."
  fi

  export VLLM_TARGET_DEVICE=cpu
  pushd "$VLLM_PATH" >/dev/null
  if [[ "$INSTALL_MODE" == "editable" ]]; then
    pip_install -e . --no-build-isolation -v
  else
    $PYTHON_BIN setup.py bdist_wheel
    pip_install dist/vllm-*.whl
  fi
  popd >/dev/null
}

build_vllm_gpu() {
  log "Building vLLM (GPU backend)"
  pushd "$VLLM_PATH" >/dev/null
  # If the repo provides a helper to link against the already-installed torch, use it.
  if [[ -f "use_existing_torch.py" ]]; then
    $PYTHON_BIN use_existing_torch.py || true
  fi
  if [[ "$INSTALL_MODE" == "editable" ]]; then
    pip_install -e . --no-build-isolation -v
  else
    $PYTHON_BIN setup.py bdist_wheel
    pip_install dist/vllm-*.whl
  fi
  popd >/dev/null
}

smoke_test_cpu() {
  log "Running CPU smoke test"
  $PYTHON_BIN - <<'PY'
from vllm import LLM, SamplingParams
m = "facebook/opt-125m"
llm = LLM(model=m)  # CPU if built for CPU
out = llm.generate(["Hello from CPU!"], SamplingParams(max_tokens=6))
print("OK CPU:", out[0].outputs[0].text.strip()[:60])
PY
}

smoke_test_gpu() {
  log "Running GPU smoke test"
  $PYTHON_BIN - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available at runtime"
from vllm import LLM, SamplingParams
m = "facebook/opt-125m"
llm = LLM(model=m)  # should use GPU
out = llm.generate(["Hello from GPU!"], SamplingParams(max_tokens=6))
print("OK GPU:", out[0].outputs[0].text.strip()[:60])
PY
}

# ---------- main ----------
ensure_python
ensure_repo
ensure_venv
ensure_base_build_tools

case "$DEVICE" in
  cpu)
    install_torch_cpu
    build_vllm_cpu
    smoke_test_cpu
    ;;
  gpu)
    if ! have_cmd nvidia-smi; then
      err "nvidia-smi not found; cannot proceed with GPU build. Use --device cpu or install NVIDIA drivers/CUDA."
    fi
    install_torch_cuda
    build_vllm_gpu
    smoke_test_gpu
    ;;
  auto)
    if have_cmd nvidia-smi; then
      warn "Auto mode: NVIDIA GPU detected → building GPU."
      install_torch_cuda
      build_vllm_gpu
      smoke_test_gpu
    else
      warn "Auto mode: no NVIDIA GPU detected → building CPU."
      install_torch_cpu
      build_vllm_cpu
      smoke_test_cpu
    fi
    ;;
  *)
    err "Unknown --device '$DEVICE' (expected cpu|gpu|auto)"
    ;;
esac

log "vLLM installation complete."
