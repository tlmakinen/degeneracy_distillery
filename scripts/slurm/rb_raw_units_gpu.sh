#!/usr/bin/env bash
#SBATCH --job-name=rb-raw-gpu
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# One-step Rayleigh-Benard sweep. One master-seed per array task.
#
# Nondimensional gate (n_params=3) at smoke budget:
#   MODE=smoke N_PARAMS=3 sbatch --array=0 scripts/slurm/rb_raw_units_gpu.sh
#
# Raw-units campaign (n_params=8), 10 seeds:
#   MODE=full N_PARAMS=8 sbatch scripts/slurm/rb_raw_units_gpu.sh
#
# Do not launch n_params=8 until the n_params=3 gate clears the Nusselt
# correlation and gradient-cosine thresholds.

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
MODE="${MODE:-full}"
N_PARAMS="${N_PARAMS:-8}"
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
OUT_BASE="${OUT_BASE:-${SCRATCH}/rebuttal_discovery/rayleigh_benard_oneshot_n${N_PARAMS}}"
NSIMS="${NSIMS:-}"
N_TEST="${N_TEST:-}"
MIN_NUSSELT_CORR="${MIN_NUSSELT_CORR:-0.9}"
MIN_NUSSELT_COSINE="${MIN_NUSSELT_COSINE:-0.9}"
SR_TIME_LIMIT="${SR_TIME_LIMIT:-}"

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
echo "mode: $MODE"
echo "n_params: $N_PARAMS"
echo "master seed: $SEED"
echo "out_dir: $OUT_DIR"
echo "XLA_FLAGS: $XLA_FLAGS"

CMD=(python scripts/oneshot_sweep.py
  --problem rayleigh_benard
  --problem-arg "n_params=${N_PARAMS}"
  --problem-arg "mode=${MODE}"
  --problem-arg "min_nusselt_corr=${MIN_NUSSELT_CORR}"
  --problem-arg "min_nusselt_cosine=${MIN_NUSSELT_COSINE}"
  --arms oneshot
  --num-trials 1
  --trial-start "$SEED"
  --seed 0
  --resume
  --out-dir "$OUT_DIR")

if [[ -n "$NSIMS" ]]; then
  CMD+=(--nsims "$NSIMS")
fi
if [[ -n "$N_TEST" ]]; then
  CMD+=(--n-test "$N_TEST")
fi
if [[ -n "$SR_TIME_LIMIT" ]]; then
  CMD+=(--sr-time-limit "$SR_TIME_LIMIT")
fi

echo "cmd: ${CMD[*]}"
"${CMD[@]}"
