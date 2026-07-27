#!/usr/bin/env bash
#SBATCH --job-name=repost-flatfix
#SBATCH --partition=comp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Re-run the regrouping step of completed runs under the fixed signed flatness
# acceptance test (commit 4514c64, notes/postprocessing_flatness_bug_fix.md).
#
# Training, flattening and the symbolic regression search are NOT redone: this
# only reloads mdl_coords from each run's sr_expressions.pkl, rebuilds X/Fs
# from the saved *_flatten.npz with that run's own align seed and
# align_subsample, and calls regroup_like_terms again. Pure CPU, minutes per
# run -- hence the `comp` partition, so this does not compete for the GPU pool
# with the live ising/rayleigh-benard training arrays.
#
# One array task per sweep, so a sweep is always redone as a unit and never
# ends up with some seeds pre-fix and some post-fix:
#   sbatch --array=0-10 scripts/slurm/repostprocess_cpu.sh
# Dry run first (writes nothing):
#   DRY_RUN=1 sbatch --array=0-10 scripts/slurm/repostprocess_cpu.sh

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
MODULE_PURGE="${MODULE_PURGE:-1}"
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
MANIFEST="${MANIFEST:-$SCRATCH/repost_manifest.json}"
RESULT_DIR="${RESULT_DIR:-$SCRATCH/repost_results}"
DRY_RUN="${DRY_RUN:-0}"

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
mkdir -p logs "$RESULT_DIR"

# This job is CPU-only; keep JAX off the GPU even if one is visible.
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
SWEEP="$(python -c "
import json,sys
m=json.load(open('$MANIFEST'))
sw=sorted({i['sweep'] for i in m})
print(sw[int('$TASK_ID')])
")"

echo "repo:      $REPO_DIR"
echo "manifest:  $MANIFEST"
echo "task:      $TASK_ID"
echo "sweep:     $SWEEP"
echo "dry_run:   $DRY_RUN"
echo "git head:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

SAFE_SWEEP="${SWEEP//\//_}"
CMD=(python scripts/repostprocess_flatness_fix.py
     --manifest "$MANIFEST"
     --sweep "$SWEEP"
     --json-out "$RESULT_DIR/$SAFE_SWEEP.json")
[[ "$DRY_RUN" == "1" ]] && CMD+=(--dry-run)

echo "cmd: ${CMD[*]}"
"${CMD[@]}"
