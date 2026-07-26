#!/usr/bin/env bash
#SBATCH --job-name=rb-nb-gpu
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Array runner for scripts/rayleigh_benard_notebook_run.py -- the intractable
# example #4 seed campaign (see neurips_rayleigh_benard.md). Each array task runs
# one master-seed, so the default --array=0-9 launches the whole 10-seed campaign
# with one submission.
#
# Smoke test (all 10 seeds, tiny budget, relaxed gates so the pipeline runs end
# to end instead of exiting at the gate):
#   MODE=smoke MIN_NUSSELT_CORR=0 MIN_NUSSELT_COSINE=0 sbatch scripts/slurm/rayleigh_benard_notebook_gpu.sh
#
# Validate ONE full seed before the campaign (recommended -- full budgets are
# untested at time of writing):
#   MODE=full sbatch --array=0 scripts/slurm/rayleigh_benard_notebook_gpu.sh
#
# Full 10-seed campaign:
#   MODE=full sbatch scripts/slurm/rayleigh_benard_notebook_gpu.sh
#
# Common escalation if symbolic regression is the bottleneck (heldout_geometry.nn
# small but .pruned large): rerun with a longer SR budget, e.g.
#   MODE=full SR_TIME_LIMIT=600 sbatch --array=0 scripts/slurm/rayleigh_benard_notebook_gpu.sh
#
# Site-specific settings can be supplied as environment variables:
#   REPO_DIR=$SCRATCH/degeneracy_distillery
#   ENV_NAME=degen
#   VENV_DIR=/home/makinen/venvs/degen
#   PYTHON_MODULE=intelpython/3-2025.1.0
#   CUDA_MODULE=cuda/12.8

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
MODE="${MODE:-full}"
OUT_BASE="${OUT_BASE:-${SCRATCH:-$REPO_DIR/results}/rebuttal_discovery/rayleigh_benard}"
MIN_NUSSELT_CORR="${MIN_NUSSELT_CORR:-0.9}"
MIN_NUSSELT_COSINE="${MIN_NUSSELT_COSINE:-0.9}"
SR_TIME_LIMIT="${SR_TIME_LIMIT:-}"

# One master-seed per array task; falls back to seed 0 for a bare local run.
SEED="${SLURM_ARRAY_TASK_ID:-0}"

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
  echo "Venv '$VENV_DIR' does not exist. Run scripts/setup_slurm_venv.sh first."
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

OUT_DIR="$OUT_BASE/seed_${SEED}"

echo "node: $(hostname)"
echo "repo: $REPO_DIR"
echo "venv: $VENV_DIR"
echo "cuda module: $CUDA_MODULE"
echo "python module: $PYTHON_MODULE"
echo "mode: $MODE"
echo "master seed: $SEED"
echo "out_dir: $OUT_DIR"
echo "CUDA_PATH: ${CUDA_PATH:-}"
echo "XLA_FLAGS: $XLA_FLAGS"

CMD=(python scripts/rayleigh_benard_notebook_run.py
  --mode "$MODE"
  --master-seed "$SEED"
  --out-dir "$OUT_DIR"
  --require-gpu
  --min-nusselt-corr "$MIN_NUSSELT_CORR"
  --min-nusselt-cosine "$MIN_NUSSELT_COSINE")

if [[ -n "$SR_TIME_LIMIT" ]]; then
  CMD+=(--sr-time-limit "$SR_TIME_LIMIT")
fi

echo "cmd: ${CMD[*]}"
"${CMD[@]}"
