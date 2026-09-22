#!/usr/bin/env bash
#SBATCH --job-name=heater1s
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Heater one-step discovery sweep. Each array task is one (dim, trial).
#
#   7 dims x 10 trials = 70 tasks:
#     sbatch --array=0-69 scripts/slurm/heater_oneshot_sweep_gpu.sh
#
# One dim, one trial first:
#     sbatch --array=0 scripts/slurm/heater_oneshot_sweep_gpu.sh
#
# TASK = DIM_IDX * N_TRIALS + TRIAL
# DIMS default: 2 3 4 6 8 10 12

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
OUT_BASE="${OUT_BASE:-$SCRATCH/heater_oneshot_scaling}"
N_TRIALS="${N_TRIALS:-10}"
NSIMS="${NSIMS:-1000}"
N_TEST="${N_TEST:-1000}"
SEED="${SEED:-0}"
read -r -a DIMS <<< "${DIMS:-2 3 4 6 8 10 12}"

TASK="${SLURM_ARRAY_TASK_ID:-${TASK:-0}}"
DIM_IDX=$(( TASK / N_TRIALS ))
TRIAL=$(( TASK % N_TRIALS ))
DIM="${DIMS[$DIM_IDX]}"
OUT_DIR="$OUT_BASE/d${DIM}/trial_${TRIAL}"

init_modules() {
  if command -v module >/dev/null 2>&1; then return; fi
  if [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
}

init_modules
if command -v module >/dev/null 2>&1; then
  [[ "$MODULE_PURGE" == "1" ]] && module purge || true
  module load "$PYTHON_MODULE" 2>/dev/null || true
  module load "$CUDA_MODULE" 2>/dev/null || true
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Without this JAX fails at the first jitted op with
# "INTERNAL: libdevice not found at ./libdevice.10.bc".
export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"
export MPLBACKEND=Agg

cd "$REPO_DIR"
mkdir -p logs "$OUT_DIR"

echo "dim:      $DIM"
echo "trial:    $TRIAL"
echo "out_dir:  $OUT_DIR"
echo "git head: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "XLA_FLAGS: $XLA_FLAGS"

python scripts/heater_oneshot_discovery_sweep.py \
  --dims "$DIM" \
  --num-trials 1 \
  --trial-start "$TRIAL" \
  --nsims "$NSIMS" \
  --n-test "$N_TEST" \
  --seed "$SEED" \
  --resume \
  --out-dir "$OUT_DIR"
