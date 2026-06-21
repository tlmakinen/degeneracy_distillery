#!/usr/bin/env bash
#SBATCH --job-name=degen-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Parameterized SLURM runner for the command-line sweep scripts.
#
# Smoke test:
#   sbatch scripts/slurm_gpu_sweep.sh
#
# Full Rosenbrock run:
#   MODE=full TARGET=rosen sbatch --time=24:00:00 scripts/slurm_gpu_sweep.sh
#
# Site-specific settings can be supplied as environment variables:
#   REPO_DIR=$SCRATCH/degeneracy_distillery
#   ENV_NAME=degen
#   VENV_DIR=/home/makinen/venvs/degen
#   PYTHON_MODULE=intelpython/3-2025.1.0
#   CUDA_MODULE=cuda/12.8
#   OUT_BASE=$SCRATCH/degen_runs

MODE="${MODE:-smoke}"
TARGET="${TARGET:-rosen}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
OUT_BASE="${OUT_BASE:-${SCRATCH:-$REPO_DIR/results}/degen_runs}"

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
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Venv '$VENV_DIR' does not exist."
  echo "Create it with: scripts/setup_slurm_venv.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"
mkdir -p logs "$OUT_BASE"

export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

case "$TARGET" in
  rosen)
    SCRIPT="scripts/rosen_nsims_logprob_sweep.py"
    ;;
  sir)
    SCRIPT="scripts/sir_nsims_logprob_sweep.py"
    ;;
  wl)
    SCRIPT="scripts/wl_2d_nsims_logprob_sweep.py"
    ;;
  gw_imr)
    SCRIPT="scripts/gw_imr_nsims_logprob_sweep.py"
    ;;
  gw_waveform)
    SCRIPT="scripts/gw_waveform_nsims_logprob_sweep.py"
    ;;
  heater)
    SCRIPT="scripts/heater_dim_scaling_sweep.py"
    ;;
  *)
    echo "Unknown TARGET '$TARGET'. Expected one of: rosen, sir, wl, gw_imr, gw_waveform, heater."
    exit 1
    ;;
esac

python - <<'PY'
import torch

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

try:
    import jax

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
except Exception as exc:
    print(f"JAX check failed: {exc}")
PY

RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_BASE/${TARGET}_${MODE}_${RUN_ID}"

if [[ "$MODE" == "smoke" ]]; then
  python "$SCRIPT" \
    --device cuda \
    --nsims 100 \
    --epochs 2 \
    --n-test 32 \
    --n-posterior-samples 128 \
    --out-dir "$OUT_DIR"
elif [[ "$MODE" == "full" ]]; then
  python "$SCRIPT" \
    --device cuda \
    --out-dir "$OUT_DIR" \
    "$@"
else
  echo "Unknown MODE '$MODE'. Expected 'smoke' or 'full'."
  exit 1
fi

echo "Run complete: $OUT_DIR"
