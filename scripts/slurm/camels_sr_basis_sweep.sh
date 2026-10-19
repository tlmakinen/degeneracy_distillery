#!/usr/bin/env bash
#SBATCH --job-name=camels-srsweep
#SBATCH --partition=comp
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# SR basis sweeps over a finished one-step run. No neural refit, so these are
# CPU jobs: operon is CPU-bound and the only JAX use is small expression
# jacobians in analyze_equations.
#
# k-ladder (k = 12 down to 2; array index i gives k = 12 - i):
#   MODE=kladder sbatch --array=0-10 scripts/slurm/camels_sr_basis_sweep.sh
#
# per-component, top-k from each eta_i's own Jacobian row:
#   MODE=percomp SELECTOR=varj  K=4 sbatch --array=0 scripts/slurm/camels_sr_basis_sweep.sh
#   MODE=percomp SELECTOR=meanj K=4 sbatch --array=0 scripts/slurm/camels_sr_basis_sweep.sh

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/${ENV_NAME:-degen}}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"

MODE="${MODE:-kladder}"
SELECTOR="${SELECTOR:-fisher}"
FROM_RUN="${FROM_RUN:-$SCRATCH/camels_oneshot_sweep/aug20k_empirical_35_seed0}"
SR_TIME_LIMIT="${SR_TIME_LIMIT:-300}"
SR_MAX_LENGTH="${SR_MAX_LENGTH:-30}"
SEED="${SEED:-0}"

if [[ "$MODE" == "kladder" ]]; then
  K=$(( 12 - ${SLURM_ARRAY_TASK_ID:-0} ))
  TAG="kladder_k${K}"
else
  K="${K:-4}"
  TAG="percomp_${SELECTOR}_k${K}"
fi
OUT_DIR="${OUT_DIR:-$SCRATCH/camels_sr_basis_sweep/$TAG}"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load "$PYTHON_MODULE" 2>/dev/null || true
fi
[[ -f "$VENV_DIR/bin/activate" ]] || { echo "no venv at $VENV_DIR" >&2; exit 1; }
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"
mkdir -p logs "$OUT_DIR"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MPLBACKEND=Agg

echo "node:     $(hostname)"
echo "mode:     $MODE   selector: $SELECTOR   k: $K"
echo "from_run: $FROM_RUN"
echo "out_dir:  $OUT_DIR"
echo "git head: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

python -u scripts/camels_sr_basis_sweep.py \
  --from-run "$FROM_RUN" \
  --mode "$MODE" \
  --k "$K" \
  --selector "$SELECTOR" \
  --seed "$SEED" \
  --sr-time-limit "$SR_TIME_LIMIT" \
  --sr-max-length "$SR_MAX_LENGTH" \
  --out "$OUT_DIR"
