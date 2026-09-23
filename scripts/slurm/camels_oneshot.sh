#!/usr/bin/env bash
#SBATCH --job-name=camels-oneshot
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# One-step map + SR on CAMELS-SB35, all 35 parameters, no pre-selection
# (new_idea/camels_oneshot.py).
#
# The three-step run needed a hand-built Fisher heuristic to cut 35 -> 6 before
# it could flatten anything, because the 35D mean Fisher has condition number 61
# and a relative floor reads rank 35. This reads r_hat off the eta spectrum
# instead.
#
# Only 819 training pairs for a 35-dim map, so the fit is small and the volume
# term (a per-sample jacfwd over BATCH_AUG fresh theta draws) dominates -- that
# is the GPU-shaped part. SR is pyoperon and CPU-bound, hence the core request.
#
#   MODE=smoke sbatch scripts/slurm/camels_oneshot.sh
#   sbatch scripts/slurm/camels_oneshot.sh                  # full
#   M_PROBE=18 sbatch scripts/slurm/camels_oneshot.sh       # if r_hat saturates

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
MODE="${MODE:-full}"
M_PROBE="${M_PROBE:-12}"
N_PCA="${N_PCA:-40}"
DATA_PATH="${DATA_PATH:-$REPO_DIR/data_scratch/data_L50_TNG_v3.hdf5}"
SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-0}}"
ALLOWED_SYMBOLS="${ALLOWED_SYMBOLS:-}"

# Experiment outputs must NOT land under $HOME: that filesystem is quota-capped
# (17.5G).  Hardcoded rather than relying on the caller exporting SCRATCH,
# because sbatch propagates the submitting environment and a non-interactive
# shell does not source ~/.bashrc.
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
OUT_BASE="${OUT_BASE:-$SCRATCH/camels_oneshot}"

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
  load_module_if_set "$CUDA_MODULE"
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Venv '$VENV_DIR' does not exist. Run scripts/setup_slurm_venv.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"
mkdir -p logs "$OUT_BASE"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Without this XLA cannot find libdevice.10.bc and every GPU kernel compile dies
# with "INTERNAL: libdevice not found at ./libdevice.10.bc" -- same line and the
# same fix as scripts/slurm/flattener_arch_sweep_gpu.sh.
export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
# pyoperon is the CPU-bound stage; the JAX fit is single-threaded anyway.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
if [[ -z "${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-}}" ]]; then
  # No GPU allocated: keep JAX from probing for a plugin and emitting CUDA 303.
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
fi

OUT_DIR="$OUT_BASE/${MODE}_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

MODE_FLAG=""
case "$MODE" in
  smoke) MODE_FLAG="--smoke" ;;
  full)  MODE_FLAG="" ;;
  *) echo "unknown MODE '$MODE' (expected smoke|full)"; exit 2 ;;
esac

EXTRA=()
if [[ -n "$ALLOWED_SYMBOLS" ]]; then
  EXTRA+=(--allowed-symbols "$ALLOWED_SYMBOLS")
fi

echo "node: $(hostname)"
echo "repo: $REPO_DIR"
echo "venv: $VENV_DIR"
echo "mode: $MODE   m_probe: $M_PROBE   n_pca: $N_PCA   seed: $SEED"
echo "out_dir: $OUT_DIR"
echo "gpus: ${SLURM_JOB_GPUS:-<none>}   OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "XLA_FLAGS: ${XLA_FLAGS:-<unset>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

# -u so the log streams; without it a crash loses every buffered line.
python -u new_idea/camels_oneshot.py \
  --data-path "$DATA_PATH" \
  --seed "$SEED" \
  --out "$OUT_DIR" \
  --m-probe "$M_PROBE" \
  --n-pca "$N_PCA" \
  $MODE_FLAG \
  "${EXTRA[@]}"

echo "done: $OUT_DIR"
