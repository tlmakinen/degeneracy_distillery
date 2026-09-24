#!/usr/bin/env bash
#SBATCH --job-name=oneshot-sw
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Generic one-step sweep. Each array task is one (arm, dim, trial).
#
#   2 arms x 5 dims x 10 trials = 100 tasks:
#     PROBLEM=rosenbrock sbatch --array=0-99 scripts/slurm/oneshot_sweep_gpu.sh
#
# One dim, one trial first:
#     PROBLEM=rosenbrock sbatch --array=0 scripts/slurm/oneshot_sweep_gpu.sh
#
# TASK = ARM_IDX * (N_DIMS * N_TRIALS) + DIM_IDX * N_TRIALS + TRIAL
# ARMS default: oneshot threestep
# DIMS default: 2 4 8 16 32
# Three-step tasks at d>16 exit as skipped_by_budget.

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
PROBLEM="${PROBLEM:-rosenbrock}"
OUT_BASE="${OUT_BASE:-$SCRATCH/${PROBLEM}_oneshot_scaling}"
N_TRIALS="${N_TRIALS:-10}"
NSIMS="${NSIMS:-2000}"
N_TEST="${N_TEST:-1000}"
SEED="${SEED:-0}"
PROBLEM_ARGS="${PROBLEM_ARGS:-}"
read -r -a DIMS <<< "${DIMS:-2 4 8 16 32}"
read -r -a ARMS <<< "${ARMS:-oneshot threestep}"

N_DIMS="${#DIMS[@]}"
N_ARMS="${#ARMS[@]}"
TASK="${SLURM_ARRAY_TASK_ID:-${TASK:-0}}"
ARM_IDX=$(( TASK / (N_DIMS * N_TRIALS) ))
REM=$(( TASK % (N_DIMS * N_TRIALS) ))
DIM_IDX=$(( REM / N_TRIALS ))
TRIAL=$(( REM % N_TRIALS ))
ARM="${ARMS[$ARM_IDX]}"
DIM="${DIMS[$DIM_IDX]}"
OUT_DIR="$OUT_BASE/${ARM}/d${DIM}/trial_${TRIAL}"

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

export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
mkdir -p logs "$OUT_DIR"

echo "problem:  $PROBLEM"
echo "arm:      $ARM"
echo "dim:      $DIM"
echo "trial:    $TRIAL"
echo "out_dir:  $OUT_DIR"
echo "git head: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "XLA_FLAGS: $XLA_FLAGS"

CMD=(python scripts/oneshot_sweep.py
  --problem "$PROBLEM"
  --arms "$ARM"
  --dims "$DIM"
  --num-trials 1
  --trial-start "$TRIAL"
  --nsims "$NSIMS"
  --n-test "$N_TEST"
  --seed "$SEED"
  --resume
  --out-dir "$OUT_DIR")

if [[ -n "$PROBLEM_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA=($PROBLEM_ARGS)
  for item in "${EXTRA[@]}"; do
    CMD+=(--problem-arg "$item")
  done
fi

echo "cmd: ${CMD[*]}"
"${CMD[@]}"
