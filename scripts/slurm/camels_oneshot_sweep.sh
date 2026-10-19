#!/usr/bin/env bash
#SBATCH --job-name=camels-1step
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# CAMELS SB35 through the one-step driver (scripts/oneshot_sweep.py), all 35
# parameters, no pre-selection. This replaces the standalone
# new_idea/camels_oneshot.py, which had its own screening statistic, its own
# ensemble and no architecture variation.
#
# MODE=screen   ladder only. Settles r_hat and writes the info-nats spectrum.
#               No ensemble, no SR. Check r_hat before spending on MODE=full.
# MODE=full     ladder + K-member varied-architecture ensemble + calibration
#               + SR. Pass M_FIT to pin the rank chosen from the screen run.
#
# Screening sweep over the info floor (the choice that sets r_hat):
#   for f in 0.10 0.25 0.50 1.00; do
#     MODE=screen INFO_FLOOR=$f sbatch scripts/slurm/camels_oneshot_sweep.sh
#   done
#
# Smoke (fits the 1h testing partition, plumbing only, r_hat not to be trusted):
#   MODE=full QUICK=1 sbatch --partition=testing --time=00:55:00 \
#     scripts/slurm/camels_oneshot_sweep.sh

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"

MODE="${MODE:-screen}"
QUICK="${QUICK:-0}"
SEED="${SEED:-0}"
N_PCA="${N_PCA:-40}"
M_PROBE="${M_PROBE:-12}"
INFO_FLOOR="${INFO_FLOOR:-0.5}"
if [[ "${MODE:-screen}" == "screen" ]]; then
  ENSEMBLE_K="${ENSEMBLE_K:-2}"
else
  ENSEMBLE_K="${ENSEMBLE_K:-10}"
fi
DATA_PATH="${DATA_PATH:-$REPO_DIR/data_scratch/data_L50_TNG_v3.hdf5}"
# Where the SR augmentation draws theta: uniform on the scaled box, or each
# parameter's own training marginal by inverse-CDF. Never a resample of the
# training rows -- that adds no coverage and leaks theta across the SR split.
PRIOR_DRAW="${PRIOR_DRAW:-uniform}"
# Flattener MLP-branch init. "zeros" starts eta as exactly the linear
# skip (a prior toward linear solutions); "glorot" starts it active.
DELTA_INIT="${DELTA_INIT:-zeros}"

# CAMELS is a fixed 1024-row dataset: 819 train / 205 held out. Asking for
# more than the split holds just returns the whole split.
NSIMS="${NSIMS:-819}"
N_TEST="${N_TEST:-205}"

SR_N_AUG="${SR_N_AUG:-20000}"
SR_TIME_LIMIT="${SR_TIME_LIMIT:-600}"
SR_MAX_LENGTH="${SR_MAX_LENGTH:-30}"
SR_MAX_DEPTH="${SR_MAX_DEPTH:-6}"
SR_SYMBOLS="${SR_SYMBOLS:-add,mul,div,pow,constant,variable,square,exp}"

TAG="${TAG:-${MODE}_floor${INFO_FLOOR}}"
OUT_BASE="${OUT_BASE:-${SCRATCH}/camels_oneshot_sweep}"
OUT_DIR="${OUT_DIR:-$OUT_BASE/${TAG}_seed${SEED}}"

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
mkdir -p logs "$OUT_DIR"

# Without this, JAX on the V100 nodes dies with
#   INTERNAL: libdevice not found at ./libdevice.10.bc
export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "node:       $(hostname)"
echo "mode:       $MODE"
echo "seed:       $SEED"
echo "info_floor: $INFO_FLOOR"
echo "m_probe:    $M_PROBE"
echo "n_pca:      $N_PCA"
echo "prior_draw: $PRIOR_DRAW"
echo "delta_init: $DELTA_INIT"
echo "out_dir:    $OUT_DIR"
echo "git head:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

CMD=(python -u scripts/oneshot_sweep.py
  --problem camels
  --problem-arg "data_path=$DATA_PATH"
  --problem-arg "n_pca=$N_PCA"
  --problem-arg "m_probe=$M_PROBE"
  --problem-arg "prior_draw=$PRIOR_DRAW"
  --delta-init "$DELTA_INIT"
  --arms oneshot
  --num-trials 1
  --seed "$SEED"
  --nsims "$NSIMS"
  --n-test "$N_TEST"
  --m-probe "$M_PROBE"
  --rank-rule "${RANK_RULE:-info}"
  --info-floor "$INFO_FLOOR"
  --refit-pad "${REFIT_PAD:-0}"
  --ensemble-k "$ENSEMBLE_K"
  --vary-architectures
  --keep-workdirs
  --out-dir "$OUT_DIR")

if [[ "$MODE" == "screen" ]]; then
  # The driver always fits an ensemble. In screen mode only r_hat matters,
  # so keep it at the 2-member minimum rather than paying for 10.
  CMD+=(--skip-sr)
else
  CMD+=(--sr-n-aug "$SR_N_AUG"
        --sr-time-limit "$SR_TIME_LIMIT"
        --sr-max-length "$SR_MAX_LENGTH"
        --sr-max-depth "$SR_MAX_DEPTH"
        --sr-allowed-symbols "$SR_SYMBOLS")
  # Hand SR only the parameters the screen ranks highest. Unset = all 35.
  [[ -n "${SR_TOP_K:-}" ]] && CMD+=(--sr-top-k "$SR_TOP_K" --sr-screen "${SR_SCREEN:-fisher}")
fi

for opt_var in PROBE_STEPS RUNG_STEPS ENSEMBLE_STEPS FROZEN_STEPS SCREEN_STEPS; do
  val="${!opt_var:-}"
  if [[ -n "$val" ]]; then
    flag="--$(echo "$opt_var" | tr '[:upper:]_' '[:lower:]-')"
    CMD+=("$flag" "$val")
  fi
done

[[ "$QUICK" == "1" ]] && CMD+=(--quick)

echo "cmd: ${CMD[*]}"
"${CMD[@]}"
