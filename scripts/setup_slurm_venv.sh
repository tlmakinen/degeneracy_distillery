#!/usr/bin/env bash

set -euo pipefail

# Build the GPU Slurm environment used by scripts/slurm_gpu_sweep.sh.
# Defaults mirror the local jupylab Slurm helpers: load system modules, then
# activate a venv from /home/makinen/venvs.

ENV_NAME="${ENV_NAME:-degen}"
VENV_BASE="${VENV_BASE:-/home/makinen/venvs}"
VENV_DIR="${VENV_DIR:-$VENV_BASE/$ENV_NAME}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
JAX_CUDA_EXTRA="${JAX_CUDA_EXTRA:-cuda12-local}"
MODULE_PURGE="${MODULE_PURGE:-1}"

INSTALL_ESR="${INSTALL_ESR:-1}"
INSTALL_ILI="${INSTALL_ILI:-1}"
ESR_SPEC="${ESR_SPEC:-git+https://github.com/DeaglanBartlett/ESR.git}"

init_modules() {
  if command -v module >/dev/null 2>&1; then
    return
  fi

  if [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
}

load_module_if_set() {
  local module_name="$1"
  if [[ -n "$module_name" ]]; then
    module load "$module_name"
  fi
}

init_modules
if command -v module >/dev/null 2>&1; then
  if [[ "$MODULE_PURGE" == "1" ]]; then
    module purge || true
  fi
  load_module_if_set "$CUDA_MODULE"
  load_module_if_set "$PYTHON_MODULE"
else
  echo "Warning: environment modules are not available; using python from PATH."
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is not available. Set PYTHON_MODULE to a module that provides Python >=3.10."
  exit 1
fi

export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"

echo "Creating venv: $VENV_DIR"
mkdir -p "$VENV_BASE"
python -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo "Installing PyTorch wheels from https://download.pytorch.org/whl/${TORCH_CUDA}"
python -m pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
  torch torchvision torchaudio

echo "Installing degeneracy_distillery in editable mode"
cd "$REPO_DIR"
python -m pip install -e .

echo "Installing JAX with local system CUDA support: jax[${JAX_CUDA_EXTRA}]"
python -m pip install --upgrade "jax[${JAX_CUDA_EXTRA}]"

if [[ "$INSTALL_ILI" == "1" ]]; then
  echo "Installing ltu-ili with PyTorch extras"
  python -m pip install "ltu-ili[pytorch] @ git+https://github.com/maho3/ltu-ili.git"
fi

if [[ "$INSTALL_ESR" == "1" ]]; then
  echo "Installing ESR package"
  python -m pip install --no-cache-dir "$ESR_SPEC"
else
  echo "Skipping ESR install. Set INSTALL_ESR=1 if postprocessing needs ESR."
fi

echo "Verifying environment"
python - <<'PY'
import degeneracy_distillery
import torch

print(f"degeneracy_distillery: {degeneracy_distillery.__file__}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")

try:
    import jax

    print(f"JAX: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
except Exception as exc:
    print(f"JAX verification failed: {exc}")

try:
    import ili

    print(f"ili: {ili.__file__}")
except Exception as exc:
    print(f"ili verification failed: {exc}")
PY

echo ""
echo "Slurm venv ready:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Submit a smoke test with:"
echo "  sbatch scripts/slurm_gpu_sweep.sh"
