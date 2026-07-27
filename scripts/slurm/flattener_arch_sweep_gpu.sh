#!/usr/bin/env bash
#SBATCH --job-name=flatarch
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Flattener architecture sweep for kuramoto / kolmogorov.
#
# Both problems inherited the same untuned 128x5 flattener default, and both sit
# at the worst neural flatness in the campaign (neural/raw 0.238 and 0.248,
# versus 0.04-0.12 for the tuned problems). Everything else in their
# fit_flattening call -- learning rates, beta_det, loss_type, log-Euclidean
# median F_avg, whitening, the 2000/2500/500 schedule -- is already identical to
# the hand-tuned GW recipe, so architecture is the only untested variable.
#
#   arm A  128 x 5   control (current default)
#   arm B  100 x 3   the hand-tuned GW config, which earned 10/10
#   arm C  256 x 7   rosenbrock's config, the "bigger network" hypothesis
#
# Note arm B is SMALLER than the control: GW's hand-tuning went down, not up.
#
# Each task reuses the fishnet ensemble already trained for that seed
# (--skip-fishnets --from-dir), so the Fisher stage is held fixed and the arms
# differ only in the flattener. The script verifies the reused artifacts match a
# fresh simulation before using them.
#
#   PROBLEM=kuramoto   sbatch --array=0-8 scripts/slurm/flattener_arch_sweep_gpu.sh
#   PROBLEM=kolmogorov sbatch --array=0-8 scripts/slurm/flattener_arch_sweep_gpu.sh
#
# Array index = arm * N_SEEDS + seed, with 3 arms x 3 seeds = 9 tasks.

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
PROBLEM="${PROBLEM:-kuramoto}"
MODE="${MODE:-rebuttal}"
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
FROM_BASE="${FROM_BASE:-$SCRATCH/follow_up_results/$PROBLEM/$MODE}"
OUT_BASE="${OUT_BASE:-$SCRATCH/follow_up_results/$PROBLEM/flatarch}"
N_SEEDS="${N_SEEDS:-3}"

ARMS_HID=(128 100 256)
ARMS_LAY=(5   3   7)
ARM_NAMES=(A_128x5 B_100x3 C_256x7)

TASK="${SLURM_ARRAY_TASK_ID:-${TASK:-0}}"
ARM=$(( TASK / N_SEEDS ))
SEED=$(( TASK % N_SEEDS ))
HID=${ARMS_HID[$ARM]}
LAY=${ARMS_LAY[$ARM]}
ARM_NAME=${ARM_NAMES[$ARM]}

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
  module load "$CUDA_MODULE" 2>/dev/null || true
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Without this JAX fails at the first jitted op with
# "INTERNAL: libdevice not found at ./libdevice.10.bc". Same line as the
# per-problem *_notebook_gpu.sh launchers; do not drop it.
export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"

cd "$REPO_DIR"
mkdir -p logs "$OUT_BASE"

FROM_DIR="$FROM_BASE/seed_${SEED}"
OUT_DIR="$OUT_BASE/${ARM_NAME}/seed_${SEED}"
mkdir -p "$OUT_DIR"

echo "problem:  $PROBLEM"
echo "arm:      $ARM_NAME (hidden=$HID layers=$LAY)"
echo "seed:     $SEED"
echo "from_dir: $FROM_DIR"
echo "out_dir:  $OUT_DIR"
echo "git head: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "XLA_FLAGS: $XLA_FLAGS"

if [[ ! -f "$FROM_DIR/fishnets-$PROBLEM/fishnets_outputs.npz" ]]; then
  echo "ERROR: no fishnet artifacts at $FROM_DIR/fishnets-$PROBLEM" >&2
  exit 2
fi

python "scripts/${PROBLEM}_notebook_run.py" \
  --mode "$MODE" \
  --master-seed "$SEED" \
  --out-dir "$OUT_DIR" \
  --skip-fishnets \
  --from-dir "$FROM_DIR" \
  --flatten-hidden-size "$HID" \
  --flatten-layers "$LAY"
