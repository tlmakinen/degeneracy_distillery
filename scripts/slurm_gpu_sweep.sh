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
#   CONDA_MODULE=miniforge
#   CUDA_MODULE=cuda/12.1
#   OUT_BASE=$SCRATCH/degen_runs

MODE="${MODE:-smoke}"
TARGET="${TARGET:-rosen}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
CONDA_MODULE="${CONDA_MODULE:-}"
CUDA_MODULE="${CUDA_MODULE:-}"
OUT_BASE="${OUT_BASE:-${SCRATCH:-$REPO_DIR/results}/degen_runs}"

load_module_if_set() {
  local module_name="$1"
  if [[ -n "$module_name" ]]; then
    module load "$module_name"
  fi
}

if command -v module >/dev/null 2>&1; then
  module purge || true
  load_module_if_set "$CONDA_MODULE"
  load_module_if_set "$CUDA_MODULE"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available. Set CONDA_MODULE to your cluster's Miniforge/Anaconda module."
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd "$REPO_DIR"
mkdir -p logs "$OUT_BASE"

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
