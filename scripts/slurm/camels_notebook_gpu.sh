#!/usr/bin/env bash
#SBATCH --job-name=camels-nb-gpu
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# CAMELS-TNG fishnets + Fisher structure analysis.
# Uses JAX tabular fishnets (no torch_scatter / GNN) → V100 is fine.
#
# Required env var:
#   DATA_PATH  — absolute path to the CAMELS HDF5 file (data_L50_TNG_v3.hdf5)
#
# Optional env vars:
#   MODE       — "smoke" | "full"  (default: full)
#   OUT_BASE   — base output directory    (default: <repo>/results/camels_notebook)
#   ENV_NAME   — venv name                (default: degen)
#
# Example submission:
#   sbatch --export=ALL,DATA_PATH=/path/to/data_L50_TNG_v3.hdf5 scripts/slurm/camels_notebook_gpu.sh
# Or rely on the default DATA_PATH (data_scratch/data_L50_TNG_v3.hdf5):
#   sbatch scripts/slurm/camels_notebook_gpu.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
MODE="${MODE:-full}"
# Default scratch root. Experiment outputs must NOT land under $HOME: that
# filesystem is quota-capped (17.5G) and a single rebuttal campaign exceeds it.
# Hardcoded rather than relying on the caller exporting SCRATCH, because sbatch
# propagates the *submitting* environment and a non-interactive shell (e.g. an
# automated submission) does not source ~/.bashrc -- which silently routed output
# back into the repo. Override by exporting SCRATCH or OUT_BASE.
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
OUT_BASE="${OUT_BASE:-${SCRATCH:-$REPO_DIR/results}/camels_notebook}"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/data_scratch/data_L50_TNG_v3.hdf5}"
PARAM_INDICES="${PARAM_INDICES:-}"
FROM_DIR="${FROM_DIR:-}"
SKIP_FISHNETS="${SKIP_FISHNETS:-0}"
RETRAIN_SUBSET="${RETRAIN_SUBSET:-0}"

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

# h5py is required; install quietly if missing
python -c "import h5py" 2>/dev/null || pip install -q h5py

cd "$REPO_DIR"
mkdir -p logs "$OUT_BASE"

export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_BASE/${MODE}_${RUN_ID}"

echo "node:          $(hostname)"
echo "repo:          $REPO_DIR"
echo "venv:          $VENV_DIR"
echo "cuda module:   $CUDA_MODULE"
echo "mode:          $MODE"
echo "out_dir:       $OUT_DIR"
echo "data_path:     $DATA_PATH"
echo "param_indices: ${PARAM_INDICES:-<auto>}"
echo "from_dir:      ${FROM_DIR:-<none>}"
echo "skip_fishnets: $SKIP_FISHNETS"
echo "retrain_subset:$RETRAIN_SUBSET"
echo "CUDA_PATH:     ${CUDA_PATH:-}"
echo "XLA_FLAGS:     $XLA_FLAGS"

EXTRA_ARGS=()
[[ -n "$PARAM_INDICES"    ]] && EXTRA_ARGS+=(--param-indices "$PARAM_INDICES")
[[ -n "$FROM_DIR"         ]] && EXTRA_ARGS+=(--from-dir "$FROM_DIR")
[[ "$SKIP_FISHNETS"  == "1" ]] && EXTRA_ARGS+=(--skip-fishnets)
[[ "$RETRAIN_SUBSET" == "1" ]] && EXTRA_ARGS+=(--retrain-subset)

python scripts/camels_notebook_run.py \
  --mode "$MODE" \
  --out-dir "$OUT_DIR" \
  --data-path "$DATA_PATH" \
  --require-gpu \
  "${EXTRA_ARGS[@]}"
