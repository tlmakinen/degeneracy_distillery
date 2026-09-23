#!/usr/bin/env bash
#SBATCH --job-name=heater1s
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Heater one-step discovery sweep. Each array task is one (dim, trial).
#
# TASK = DIM_IDX * N_TRIALS + TRIAL
#
# V100, large d (default DIMS below):
#   DIMS="6 8 10 12" sbatch --array=0 scripts/slurm/heater_oneshot_sweep_gpu.sh
#   DIMS="6 8 10 12" sbatch --array=0-39 scripts/slurm/heater_oneshot_sweep_gpu.sh
#
# CPU, small d (no GPU request; see heater_oneshot_sweep_cpu.sh):
#   DIMS="2 3 4" sbatch --array=0 scripts/slurm/heater_oneshot_sweep_cpu.sh
#   DIMS="2 3 4" sbatch --array=0-29 scripts/slurm/heater_oneshot_sweep_cpu.sh
#
# The torch build in degen is 2.11+cu128 (sm_75..sm_120). It has no sm_70
# image, so the MDN arms run on CPU on these V100s. JAX JIT-compiles for the
# card and keeps the ladder, ensemble, and frozen rescore.

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
OUT_BASE="${OUT_BASE:-$SCRATCH/heater_oneshot_scaling_v4}"
N_TRIALS="${N_TRIALS:-10}"
NSIMS="${NSIMS:-1000}"
N_TEST="${N_TEST:-1000}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-auto}"
read -r -a DIMS <<< "${DIMS:-6 8 10 12}"

TASK="${SLURM_ARRAY_TASK_ID:-${TASK:-0}}"
DIM_IDX=$(( TASK / N_TRIALS ))
TRIAL=$(( TASK % N_TRIALS ))
if (( DIM_IDX < 0 || DIM_IDX >= ${#DIMS[@]} )); then
  echo "array index $TASK maps to DIM_IDX=$DIM_IDX, out of range for DIMS=${DIMS[*]}"
  exit 2
fi
DIM="${DIMS[$DIM_IDX]}"
OUT_DIR="$OUT_BASE/d${DIM}/trial_${TRIAL}"

NTHREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NTHREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$NTHREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$NTHREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$NTHREADS}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

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
  if [[ -n "${CUDA_MODULE}" ]]; then
    module load "$CUDA_MODULE" 2>/dev/null || true
  fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Without this JAX fails at the first jitted op with
# "INTERNAL: libdevice not found at ./libdevice.10.bc".
export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"

if [[ "$DEVICE" == "auto" ]]; then
  # sm_70 (V100) is absent from the cu128 build, so fall back to CPU for the
  # NPE arms rather than dying mid-run. JAX keeps the GPU either way.
  DEVICE=$(python - <<'PY'
try:
    import torch
    if not torch.cuda.is_available():
        print("cpu")
    else:
        major, minor = torch.cuda.get_device_capability(0)
        archs = {a.split("_")[1] for a in torch.cuda.get_arch_list() if a.startswith("sm_")}
        print("cuda" if f"{major}{minor}" in archs else "cpu")
except Exception:
    print("cpu")
PY
)
fi

cd "$REPO_DIR"
mkdir -p logs "$OUT_DIR"

echo "node:      $(hostname)"
echo "dim:       $DIM"
echo "trial:     $TRIAL"
echo "out_dir:   $OUT_DIR"
echo "device:    $DEVICE"
echo "nthreads:  $NTHREADS"
echo "git head:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "XLA_FLAGS: $XLA_FLAGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

python scripts/heater_oneshot_discovery_sweep.py \
  --dims "$DIM" \
  --num-trials 1 \
  --trial-start "$TRIAL" \
  --nsims "$NSIMS" \
  --n-test "$N_TEST" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --resume \
  --out-dir "$OUT_DIR"
