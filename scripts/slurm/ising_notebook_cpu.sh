#!/usr/bin/env bash
#SBATCH --job-name=ising-nb-cpu
#SBATCH --partition=comp
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# CPU fallback runner for scripts/ising_notebook_run.py (2D Ising, intractable
# partition function). Companion to scripts/slurm/ising_notebook_gpu.sh.
#
# Why this exists: the GPU path reproducibly diverges to NaN during fishnet
# ensemble training on this cluster (confirmed on two independent GPU seeds --
# a pre-existing bug in the shared training-loop code, not introduced by the
# schema adaptation, and out of scope to fix here). CPU training is
# demonstrably stable for the identical config/seed, so the NeurIPS rebuttal
# seed campaign runs on CPU instead, via --no-require-gpu. Revisit the GPU
# path separately later.
#
# Uses the `comp` partition (no --gres=gpu request, 2-day time limit) rather
# than `pscomp`, so these ten jobs do not compete for the scarce GPU pool.
# Smoke-mode CPU fishnet training alone took ~100 minutes in prior runs
# (scripts/ising_notebook_run.py docstring / neurips_intractable_examples.md);
# rebuttal mode uses a larger lattice and many more fishnet/flatten epochs, so
# the 1-day time budget below is generous headroom, not a tight estimate.
#
# Launch as a seed array (one seed per task, master seed = array task ID):
#   sbatch --array=0-9 scripts/slurm/ising_notebook_cpu.sh
# MODE/OUT_BASE/etc. are still overridable via env, e.g.:
#   MODE=rebuttal sbatch --array=0-9 scripts/slurm/ising_notebook_cpu.sh
# A plain (non-array) submission falls back to SEED (default 0).

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
MODULE_PURGE="${MODULE_PURGE:-1}"
MODE="${MODE:-smoke}"
# Default scratch root. Experiment outputs must NOT land under $HOME: that
# filesystem is quota-capped (17.5G) and a single rebuttal campaign exceeds it.
# Hardcoded rather than relying on the caller exporting SCRATCH, because sbatch
# propagates the *submitting* environment and a non-interactive shell (e.g. an
# automated submission) does not source ~/.bashrc -- which silently routed output
# back into the repo. Override by exporting SCRATCH or OUT_BASE.
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
OUT_BASE="${OUT_BASE:-${SCRATCH:-$REPO_DIR/follow_up_results}/ising/$MODE}"
MIN_IDENTIFIABLE_CORR="${MIN_IDENTIFIABLE_CORR:-0.7}"
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

# No CUDA/XLA GPU flags needed on the CPU path. JAX_PLATFORMS=cpu keeps JAX
# from even attempting to discover a GPU plugin on a node that may still
# expose one, avoiding the (harmless but noisy) CUDA plugin init errors seen
# in the GPU logs.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

OUT_DIR="$OUT_BASE/seed_${SEED}"

echo "node: $(hostname)"
echo "repo: $REPO_DIR"
echo "venv: $VENV_DIR"
echo "python module: $PYTHON_MODULE"
echo "mode: $MODE"
echo "seed: $SEED (SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<none>})"
echo "out_dir: $OUT_DIR"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

python scripts/ising_notebook_run.py \
  --mode "$MODE" \
  --master-seed "$SEED" \
  --out-dir "$OUT_DIR" \
  --no-require-gpu \
  --min-identifiable-corr "$MIN_IDENTIFIABLE_CORR"
