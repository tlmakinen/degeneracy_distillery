#!/usr/bin/env bash
#SBATCH --job-name=rbnpe
#SBATCH --partition=comp
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Rosenbrock NPE scaling. One array task per (d, trial). Task map is
# TASK = DIM_IDX * N_TRIALS + TRIAL, matching heater_oneshot_sweep_gpu.sh.
#
# Each task regenerates the discovery driver's simulations at the same seed,
# trains three MAF NPE arms (raw / oracle / discovered) at nsims=2000, and
# scores raw against oracle and discovered on their respective shared axes.
#
# Pilot (one task per d):
#   DIMS="4"  sbatch --array=0        scripts/slurm/rosenbrock_npe_scaling.sh
#   DIMS="16" sbatch --array=0        scripts/slurm/rosenbrock_npe_scaling.sh
#   DIMS="32" sbatch --array=1        scripts/slurm/rosenbrock_npe_scaling.sh   # trial 1
#
# Full CPU array over d in {2, 4, 8, 16} (10 trials each -> 40 tasks):
#   DIMS="2 4 8 16" sbatch --array=0-39 scripts/slurm/rosenbrock_npe_scaling.sh
#
# d=32 only, once all 10 discovery trials have landed. Route to comp by
# default; the H100 route below is opt-in via GPU=1.
#   DIMS="32" sbatch --array=0-9 scripts/slurm/rosenbrock_npe_scaling.sh
#   DIMS="32" GPU=1 sbatch --array=0-9 scripts/slurm/rosenbrock_npe_scaling.sh
#
# Each task runs from an exact git-archive snapshot so a subsequent
# rebase or partial edit in the working tree cannot alter what is running.

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-}"
MODULE_PURGE="${MODULE_PURGE:-1}"

SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
DISCOVERY_DIR="${DISCOVERY_DIR:-$SCRATCH/rosenbrock_oneshot_scaling_inforank/oneshot}"
OUT_BASE="${OUT_BASE:-$SCRATCH/rosenbrock_npe_scaling}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-$SCRATCH/code_snapshots}"
SNAPSHOT_TAG="${SNAPSHOT_TAG:-rosenbrock_npe}"

N_TRIALS="${N_TRIALS:-10}"
NSIMS="${NSIMS:-2000}"
N_TEST="${N_TEST:-1000}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-400}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
HIDDEN="${HIDDEN:-50}"
NTRANSFORMS="${NTRANSFORMS:-5}"
REPEATS="${REPEATS:-2}"
N_MARG_VAL="${N_MARG_VAL:-200}"
N_MARG_SAMPLES="${N_MARG_SAMPLES:-2000}"
GPU="${GPU:-0}"

read -r -a DIMS <<< "${DIMS:-2 4 8 16}"

TASK="${SLURM_ARRAY_TASK_ID:-${TASK:-0}}"
DIM_IDX=$(( TASK / N_TRIALS ))
TRIAL=$(( TASK % N_TRIALS ))
if (( DIM_IDX < 0 || DIM_IDX >= ${#DIMS[@]} )); then
  echo "array index $TASK maps to DIM_IDX=$DIM_IDX, out of range for DIMS=${DIMS[*]}" >&2
  exit 2
fi
DIM="${DIMS[$DIM_IDX]}"

NTHREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NTHREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$NTHREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$NTHREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$NTHREADS}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

if [[ "$GPU" == "1" ]]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  DEVICE="${DEVICE:-auto}"
else
  # Pure-CPU path. torch/JAX must not see the GPU or JAX can grab it lazily.
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
  export CUDA_VISIBLE_DEVICES=""
  DEVICE="${DEVICE:-cpu}"
fi

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
  if [[ -n "${CUDA_MODULE}" ]]; then
    module load "$CUDA_MODULE" 2>/dev/null || true
  fi
  module load "$PYTHON_MODULE" 2>/dev/null || true
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Venv '$VENV_DIR' does not exist. Run scripts/setup_slurm_venv.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Build an exact snapshot of the tree at HEAD, so a working-tree edit made
# after sbatch (or another agent's commit) cannot change what is running.
# The snapshot is tagged with git head + a short uid so concurrent submits
# from the same HEAD share one snapshot.
mkdir -p "$SNAPSHOT_ROOT" "$OUT_BASE" "$REPO_DIR/logs"
pushd "$REPO_DIR" >/dev/null
GIT_HEAD="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short HEAD)"
SNAPSHOT_DIR="$SNAPSHOT_ROOT/${SNAPSHOT_TAG}_${SHORT}"
# Concurrent array tasks land here at the same time. mkdir is atomic on
# posix; whichever task wins the mkdir race populates the snapshot and
# writes the marker file. Losers just wait for the marker.
if [[ ! -f "$SNAPSHOT_DIR/.git_head" ]]; then
  if mkdir "$SNAPSHOT_DIR" 2>/dev/null; then
    tmp="${SNAPSHOT_DIR}.tmp.$$"
    rm -rf "$tmp"
    mkdir -p "$tmp"
    git archive HEAD | tar -x -C "$tmp"
    (cd "$tmp" && tar -cf - .) | (cd "$SNAPSHOT_DIR" && tar -xf -)
    rm -rf "$tmp"
    echo "$GIT_HEAD" > "$SNAPSHOT_DIR/.git_head"
  else
    for _ in $(seq 1 120); do
      [[ -f "$SNAPSHOT_DIR/.git_head" ]] && break
      sleep 1
    done
    if [[ ! -f "$SNAPSHOT_DIR/.git_head" ]]; then
      echo "timed out waiting for snapshot at $SNAPSHOT_DIR" >&2
      exit 3
    fi
  fi
fi
popd >/dev/null

cd "$SNAPSHOT_DIR"

OUT_DIR="$OUT_BASE"
mkdir -p "$OUT_DIR/d${DIM}/trial_${TRIAL}"

echo "node:          $(hostname)"
echo "snapshot:      $SNAPSHOT_DIR"
echo "git head:      $GIT_HEAD"
echo "dim:           $DIM"
echo "trial:         $TRIAL"
echo "out_dir:       $OUT_DIR"
echo "discovery:     $DISCOVERY_DIR"
echo "device:        $DEVICE"
echo "nthreads:      $NTHREADS"
echo "CUDA_VISIBLE:  ${CUDA_VISIBLE_DEVICES-<unset>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

python scripts/rosenbrock_npe_scaling.py \
  --dims "$DIM" \
  --num-trials 1 \
  --trial-start "$TRIAL" \
  --discovery-dir "$DISCOVERY_DIR" \
  --out-dir "$OUT_DIR" \
  --nsims "$NSIMS" \
  --n-test "$N_TEST" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LR" \
  --hidden-features "$HIDDEN" \
  --num-transforms "$NTRANSFORMS" \
  --repeats "$REPEATS" \
  --n-marginal-val "$N_MARG_VAL" \
  --n-marginal-samples "$N_MARG_SAMPLES" \
  --num-threads "$NTHREADS" \
  --device "$DEVICE" \
  --resume
