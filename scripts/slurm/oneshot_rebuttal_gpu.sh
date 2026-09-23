#!/usr/bin/env bash
#SBATCH --job-name=1step-reb
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# One-step reruns of the NeurIPS rebuttal discovery experiments, one master
# seed per array task, written as seed_<k>/ so scripts/aggregate_seed_sweep.py
# reads them with no changes.
#
# The SR budget defaults below are copied from each experiment's own
# CONFIGS["rebuttal"] in scripts/*_notebook_run.py so the search budget matches
# the published run. Do not retune them per seed.
#
# Smoke (one seed, reduced budget, fits the 1h testing partition):
#   PROBLEM=sir QUICK=1 sbatch --partition=testing --time=00:55:00 \
#     --array=0 scripts/slurm/oneshot_rebuttal_gpu.sh
#
# Full rebuttal-matched array:
#   PROBLEM=sir sbatch --array=0-9 scripts/slurm/oneshot_rebuttal_gpu.sh
#   PROBLEM=gw_taylorf2 sbatch --array=0-9 scripts/slurm/oneshot_rebuttal_gpu.sh
#   PROBLEM=gw_imrphenomd sbatch --array=0-9 scripts/slurm/oneshot_rebuttal_gpu.sh
#
# Aggregate and compare:
#   python scripts/aggregate_seed_sweep.py \
#     $SCRATCH/oneshot_rebuttal/<problem>

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"

PROBLEM="${PROBLEM:-sir}"
QUICK="${QUICK:-0}"
OUT_BASE="${OUT_BASE:-${SCRATCH}/oneshot_rebuttal/${PROBLEM}}"
SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-0}}"

# Per-experiment budgets, matching CONFIGS["rebuttal"] in the published driver.
case "$PROBLEM" in
  sir)
    NSIMS="${NSIMS:-500}"; N_TEST="${N_TEST:-500}"
    SR_N_AUG="${SR_N_AUG:-10000}"; SR_TIME_LIMIT="${SR_TIME_LIMIT:-60}"
    SR_MAX_LENGTH="${SR_MAX_LENGTH:-25}"; SR_MAX_DEPTH="${SR_MAX_DEPTH:-10}"
    SR_SYMBOLS="${SR_SYMBOLS:-add,mul,div,pow,constant,variable}"
    ;;
  gw_taylorf2|gw_imrphenomd)
    NSIMS="${NSIMS:-500}"; N_TEST="${N_TEST:-500}"
    SR_N_AUG="${SR_N_AUG:-2000}"; SR_TIME_LIMIT="${SR_TIME_LIMIT:-120}"
    SR_MAX_LENGTH="${SR_MAX_LENGTH:-30}"; SR_MAX_DEPTH="${SR_MAX_DEPTH:-20}"
    SR_SYMBOLS="${SR_SYMBOLS:-add,mul,div,pow,constant,variable,exp}"
    ;;
  *)
    echo "unknown PROBLEM '$PROBLEM' (expected sir|gw_taylorf2|gw_imrphenomd)" >&2
    exit 2
    ;;
esac

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
  [[ -n "$CUDA_MODULE" ]] && module load "$CUDA_MODULE" 2>/dev/null || true
  module load "$PYTHON_MODULE" 2>/dev/null || true
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Venv '$VENV_DIR' does not exist. Run scripts/setup_slurm_venv.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"
OUT_DIR="$OUT_BASE/seed_${SEED}"
mkdir -p logs "$OUT_DIR"

export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "node:     $(hostname)"
echo "problem:  $PROBLEM"
echo "seed:     $SEED"
echo "out_dir:  $OUT_DIR"
echo "git head: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "quick:    $QUICK"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

CMD=(python scripts/oneshot_sweep.py
  --problem "$PROBLEM"
  --arms oneshot
  --num-trials 1
  --trial-start "$SEED"
  --seed 0
  --nsims "$NSIMS"
  --n-test "$N_TEST"
  --sr-n-aug "$SR_N_AUG"
  --sr-time-limit "$SR_TIME_LIMIT"
  --sr-max-length "$SR_MAX_LENGTH"
  --sr-max-depth "$SR_MAX_DEPTH"
  --sr-allowed-symbols "$SR_SYMBOLS"
  --rank-rule info
  --info-floor 1.0
  --keep-workdirs
  --resume
  --out-dir "$OUT_DIR")

# Optional overrides. The driver's 20000-step defaults were tuned at
# nsims=2000; at nsims=500 the volume term overfits well before then (probe
# val NLL rises while r_hat stays stable). Set these to something smaller for
# rebuttal-budget runs.
for opt_var in PROBE_STEPS RUNG_STEPS ENSEMBLE_STEPS FROZEN_STEPS RANK_MIN_GAP; do
  val="${!opt_var:-}"
  if [[ -n "$val" ]]; then
    flag="--$(echo "$opt_var" | tr '[:upper:]_' '[:lower:]-')"
    CMD+=("$flag" "$val")
  fi
done

# --quick shrinks every step count; it is for plumbing only and its r_hat must
# not be trusted (see notes/oneshot_sweep_pipeline.md).
if [[ "$QUICK" == "1" ]]; then
  CMD+=(--quick)
fi

echo "cmd: ${CMD[*]}"
"${CMD[@]}"
