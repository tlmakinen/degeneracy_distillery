#!/usr/bin/env bash
#SBATCH --job-name=camels-square
#SBATCH --partition=pscompl,pscomp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Square-completed loss on CAMELS. d = participating set, K = active.
# One eta architecture; the data network still varies per member so y_std
# stays calibrated for the SR step.
#
#   MODE=smoke sbatch scripts/slurm/camels_square.sh
#   sbatch scripts/slurm/camels_square.sh

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/${ENV_NAME:-degen}}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"

MODE="${MODE:-full}"
D_KEEP="${D_KEEP:-12}"
K_ACTIVE="${K_ACTIVE:-7}"
N_PCA="${N_PCA:-0}"
BOX_MARGIN="${BOX_MARGIN:-0.05}"
FLOW_LAYERS="${FLOW_LAYERS:-4}"
FLOW_WIDTH="${FLOW_WIDTH:-32}"
SEED="${SEED:-0}"
if [[ "$MODE" == "smoke" ]]; then
  STEPS="${STEPS:-300}"; ENSEMBLE_K="${ENSEMBLE_K:-3}"
else
  STEPS="${STEPS:-1500}"; ENSEMBLE_K="${ENSEMBLE_K:-10}"
fi
TAG="${TAG:-${MODE}_d${D_KEEP}_K${K_ACTIVE}}"
OUT_DIR="${OUT_DIR:-$SCRATCH/camels_square/${TAG}_seed${SEED}}"
export PILOT_FISHER="${PILOT_FISHER:-$SCRATCH/camels_pilot_fisher.npz}"

if command -v module >/dev/null 2>&1; then
  module purge || true
  [[ -n "$CUDA_MODULE" ]] && module load "$CUDA_MODULE" 2>/dev/null || true
  module load "$PYTHON_MODULE" 2>/dev/null || true
fi
[[ -f "$VENV_DIR/bin/activate" ]] || { echo "no venv at $VENV_DIR" >&2; exit 1; }
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"
mkdir -p logs "$OUT_DIR"
export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MPLBACKEND=Agg

echo "node:      $(hostname)"
echo "mode:      $MODE   d_keep $D_KEEP   K $K_ACTIVE   n_pca $N_PCA"
echo "steps:     $STEPS   ensemble_k $ENSEMBLE_K   flow ${FLOW_LAYERS}x${FLOW_WIDTH}"
echo "out_dir:   $OUT_DIR"
echo "git head:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

python -u scripts/camels_square.py \
  --n-pca "$N_PCA" --d-keep "$D_KEEP" --k-active "$K_ACTIVE" \
  --prior-reg "${PRIOR_REG:-1.0}" --box-margin "$BOX_MARGIN" --flow-layers "$FLOW_LAYERS" --flow-width "$FLOW_WIDTH" \
  --eval-every "${EVAL_EVERY:-10}" --ensemble-k "$ENSEMBLE_K" --steps "$STEPS" --seed "$SEED" \
  --skip-sr --out "$OUT_DIR"
