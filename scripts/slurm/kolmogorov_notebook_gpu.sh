#!/usr/bin/env bash
#SBATCH --job-name=kolmogorov-nb-gpu
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Batch runner for scripts/kolmogorov_notebook_run.py (forced 2D Kolmogorov-flow
# turbulence discovery). Defaults to a shorter monitored smoke run; use MODE=full
# for the notebook-sized run, MODE=rebuttal for the NeurIPS rebuttal
# configuration. Note: "rebuttal" is currently a same-values alias of "full" for
# this experiment -- "full" already trains on 500 simulations with 2000
# augmented coordinate evaluations, matching the campaign convention used by
# the other *_notebook_run.py scripts, so there is nothing to bump.
#
# This is the most compute-expensive of the three intractable-likelihood
# experiments (~30-45 min pure solver time on CPU at full/rebuttal size per the
# handoff doc, much faster on GPU); the --time budget below is set well above
# the rosenbrock template's 4 hours to leave headroom for fishnet/flatten/SR
# stages on top of the solver time.
#
# Launch as a seed array (one seed per task, master seed = array task ID):
#   sbatch --array=0-9 scripts/slurm/kolmogorov_notebook_gpu.sh
# MODE/OUT_BASE/etc. are still overridable via env, e.g.:
#   MODE=rebuttal sbatch --array=0-9 scripts/slurm/kolmogorov_notebook_gpu.sh
# A plain (non-array) submission falls back to SEED (default 0).

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
MODE="${MODE:-smoke}"
# Default scratch root. Experiment outputs must NOT land under $HOME: that
# filesystem is quota-capped (17.5G) and a single rebuttal campaign exceeds it.
# Hardcoded rather than relying on the caller exporting SCRATCH, because sbatch
# propagates the *submitting* environment and a non-interactive shell (e.g. an
# automated submission) does not source ~/.bashrc -- which silently routed output
# back into the repo. Override by exporting SCRATCH or OUT_BASE.
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
OUT_BASE="${OUT_BASE:-${SCRATCH:-$REPO_DIR/follow_up_results}/kolmogorov/$MODE}"
MIN_REYNOLDS_CORR="${MIN_REYNOLDS_CORR:-0.9}"
MAX_EXPONENT_ERROR="${MAX_EXPONENT_ERROR:-0.4}"
SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-0}}"

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
echo "seed: $SEED (SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<none>})"
echo "out_dir: $OUT_DIR"
echo "CUDA_PATH: ${CUDA_PATH:-}"
echo "XLA_FLAGS: $XLA_FLAGS"

python scripts/kolmogorov_notebook_run.py \
  --mode "$MODE" \
  --master-seed "$SEED" \
  --out-dir "$OUT_DIR" \
  --require-gpu \
  --min-reynolds-corr "$MIN_REYNOLDS_CORR" \
  --max-exponent-error "$MAX_EXPONENT_ERROR"
