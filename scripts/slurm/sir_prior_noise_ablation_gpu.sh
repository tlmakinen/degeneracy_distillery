#!/usr/bin/env bash
#SBATCH --job-name=sir-ablate
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Prior-range and noise-model ablation for SIR, answering QfCk Q3.
#
# One variable at a time about the frozen configuration (noise sd 0.05, the
# tuned beta/gamma prior). The frozen setting itself is NOT re-run here: the
# existing follow_up_results/sir/rebuttal seeds 0-2 are the baseline arm, and
# they were produced by this same code path.
#
#   arm            noise_std   prior_width_scale
#   noise_lo       0.02        1.0
#   noise_hi       0.10        1.0
#   prior_narrow   0.05        0.5
#   prior_wide     0.05        2.0
#
# Prior widths scale in LOG space about the geometric midpoint, so beta and
# gamma stay strictly positive; a linear rescale at 2x would push the gamma
# lower edge negative. Widening also changes how many draws survive the
# supercriticality cut (beta/gamma >= 1 + DELTA), so the effective prior is not
# a simple rescale of the frozen one -- config_manifest.json records
# ablation.supercriticality_accept_rate for exactly this reason. Read it before
# comparing arms.
#
#   sbatch --array=0-11 scripts/slurm/sir_prior_noise_ablation_gpu.sh
#
# Array index = arm * N_SEEDS + seed, 4 arms x 3 seeds = 12 tasks.

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
MODE="${MODE:-rebuttal}"
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
OUT_BASE="${OUT_BASE:-$SCRATCH/follow_up_results/sir/ablation}"
N_SEEDS="${N_SEEDS:-3}"

ARM_NAMES=(noise_lo     noise_hi     prior_narrow prior_wide)
ARM_NOISE=(0.02         0.10         0.05         0.05)
ARM_WIDTH=(1.0          1.0          0.5          2.0)

TASK="${SLURM_ARRAY_TASK_ID:-${TASK:-0}}"
ARM=$(( TASK / N_SEEDS ))
SEED=$(( TASK % N_SEEDS ))
ARM_NAME=${ARM_NAMES[$ARM]}
NOISE=${ARM_NOISE[$ARM]}
WIDTH=${ARM_WIDTH[$ARM]}

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
# "INTERNAL: libdevice not found at ./libdevice.10.bc".
export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"

cd "$REPO_DIR"
mkdir -p logs "$OUT_BASE"

OUT_DIR="$OUT_BASE/${ARM_NAME}/seed_${SEED}"
mkdir -p "$OUT_DIR"

echo "arm:       $ARM_NAME (noise_std=$NOISE prior_width_scale=$WIDTH)"
echo "seed:      $SEED"
echo "out_dir:   $OUT_DIR"
echo "git head:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "XLA_FLAGS: $XLA_FLAGS"

python scripts/sir_notebook_run.py \
  --mode "$MODE" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR" \
  --noise-std "$NOISE" \
  --prior-width-scale "$WIDTH"
