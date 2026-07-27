#!/usr/bin/env bash
#SBATCH --job-name=qm7b-nb-cpu
#SBATCH --partition=comp
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# CPU runner for scripts/qm7b_notebook_run.py, for the alignment A/B only.
#
# Why this can be CPU: the only CUDA-hardcoded stage is fishnet training
# (train_fishnets_dataset(device="cuda")). Coordinate alignment sits downstream
# of both fishnet and flattener training, so an alignment comparison can reuse
# already-trained fishnets via --skip-fishnets --from-dir and re-run only
# flattener fit -> align -> SR -> postprocess, all of which run on CPU (JAX CPU
# + PyOperon). That also keeps the GPU pool free for the live ising array.
#
# Both arms MUST run on CPU from the SAME fishnet artifacts, otherwise the
# comparison is confounded: the flattener refit and the regrouping rotation
# search are both sensitive to CPU-vs-GPU numerics (see
# notes/postprocessing_flatness_bug_fix.md and the rotation-search
# nondeterminism measured on 2026-07-27).
#
#   # treatment: procrustes + separate_nonlinearity=True
#   MODE=rebuttal_procrustes sbatch --array=0-9 scripts/slurm/qm7b_notebook_cpu.sh
#   # matched control: existing kabsch settings, same code path, same CPU
#   MODE=rebuttal OUT_SUFFIX=_cpuctl sbatch --array=0-9 scripts/slurm/qm7b_notebook_cpu.sh

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
MODULE_PURGE="${MODULE_PURGE:-1}"
MODE="${MODE:-rebuttal_procrustes}"
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
OUT_SUFFIX="${OUT_SUFFIX:-}"
OUT_BASE="${OUT_BASE:-${SCRATCH}/follow_up_results/qm7b/${MODE}${OUT_SUFFIX}}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/qm7b}"
MIN_GAP_CORR="${MIN_GAP_CORR:-0.5}"
GEOMETRIC_IMPROVEMENT_MARGIN="${GEOMETRIC_IMPROVEMENT_MARGIN:-0.8}"
SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-0}}"

# Reuse the trained fishnets from the scaler-fixed rebuttal run, which is the
# current correct QM7b result (the earlier follow_up_results/qm7b/rebuttal/ used
# the buggy [0,1] theta scaler).
FROM_BASE="${FROM_BASE:-${SCRATCH}/follow_up_results/qm7b/rebuttal_scalefix}"
FROM_DIR="${FROM_DIR:-${FROM_BASE}/seed_${SEED}}"

init_modules() {
  if command -v module >/dev/null 2>&1; then
    return
  fi
  if [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
}

init_modules
if command -v module >/dev/null 2>&1; then
  [[ "$MODULE_PURGE" == "1" ]] && module purge || true
  module load "$PYTHON_MODULE" 2>/dev/null || true
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"
mkdir -p logs "$OUT_BASE"

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

OUT_DIR="$OUT_BASE/seed_${SEED}"
mkdir -p "$OUT_DIR"

echo "repo:      $REPO_DIR"
echo "mode:      $MODE"
echo "seed:      $SEED"
echo "out_dir:   $OUT_DIR"
echo "from_dir:  $FROM_DIR"
echo "git head:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

if [[ ! -f "$FROM_DIR/fishnets-qm7b/fishnets_outputs.npz" ]]; then
  echo "ERROR: no fishnet artifacts at $FROM_DIR/fishnets-qm7b/fishnets_outputs.npz" >&2
  exit 2
fi

python scripts/qm7b_notebook_run.py \
  --mode "$MODE" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR" \
  --data-dir "$DATA_DIR" \
  --no-require-gpu \
  --min-gap-corr "$MIN_GAP_CORR" \
  --geometric-improvement-margin "$GEOMETRIC_IMPROVEMENT_MARGIN" \
  --skip-fishnets \
  --skip-flatten \
  --from-dir "$FROM_DIR"
