#!/usr/bin/env bash
#SBATCH --job-name=scalar-oneshot-sr
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# One-step map + symbolic regression on the scalar Rosenbrock potential
# (new_idea/scalar_oneshot_sr.py).  This is the SR validation for the case
# Fishnets-then-flatten cannot do: x ~ N(f(theta), sigma^2) with a coupled
# scalar f, so every coordinate enters and the likelihood Fisher has rank 1.
#
# GPU vs CPU: the fit is small (64-64 MLPs, d<=4) but the volume term takes a
# per-sample jacfwd inside a vmap over 512 fresh theta draws every step, in
# float64 -- that is what makes it slow on a login node.  The SR stage is
# pyoperon and is CPU-bound either way, hence --cpus-per-task=16.
#
# No torch anywhere in this path, so the V100 sm_70 arch trap that bites the
# heater/sir wrappers does not apply here (see notes in
# scripts/slurm/heater_discovery_scaling_gpu.sh).
#
#   sbatch scripts/slurm/scalar_oneshot_sr.sh                       # quick smoke, V100
#   MODE=default sbatch scripts/slurm/scalar_oneshot_sr.sh
#   MODE=full PROBLEMS="scalar3 scalar4" sbatch scripts/slurm/scalar_oneshot_sr.sh
#
# CPU instead of GPU (drop the gres request on the command line):
#   sbatch --partition=comp --gres=NONE scripts/slurm/scalar_oneshot_sr.sh

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
MODE="${MODE:-quick}"
PROBLEMS="${PROBLEMS:-scalar3}"
SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-0}}"
SR_TIME_LIMIT="${SR_TIME_LIMIT:-}"
ALLOWED_SYMBOLS="${ALLOWED_SYMBOLS:-}"

# Experiment outputs must NOT land under $HOME: that filesystem is quota-capped
# (17.5G).  Hardcoded rather than relying on the caller exporting SCRATCH,
# because sbatch propagates the submitting environment and a non-interactive
# shell does not source ~/.bashrc.
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
OUT_BASE="${OUT_BASE:-$SCRATCH/scalar_oneshot_sr}"

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
  quick)   MODE_FLAG="--quick" ;;
  default) MODE_FLAG="" ;;
  full)    MODE_FLAG="--full" ;;
  *) echo "unknown MODE '$MODE' (expected quick|default|full)"; exit 2 ;;
esac

EXTRA=()
if [[ -n "$SR_TIME_LIMIT" ]]; then
  EXTRA+=(--sr-time-limit "$SR_TIME_LIMIT")
fi
if [[ -n "$ALLOWED_SYMBOLS" ]]; then
  EXTRA+=(--allowed-symbols "$ALLOWED_SYMBOLS")
fi

echo "node: $(hostname)"
echo "repo: $REPO_DIR"
echo "venv: $VENV_DIR"
echo "mode: $MODE   problems: $PROBLEMS   seed: $SEED"
echo "out_dir: $OUT_DIR"
echo "gpus: ${SLURM_JOB_GPUS:-<none>}   OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "XLA_FLAGS: ${XLA_FLAGS:-<unset>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

# -u so the log streams; without it a crash loses every buffered line.
python -u new_idea/scalar_oneshot_sr.py \
  --problems $PROBLEMS \
  --seed "$SEED" \
  --out "$OUT_DIR" \
  $MODE_FLAG \
  "${EXTRA[@]}"

echo "done: $OUT_DIR"
