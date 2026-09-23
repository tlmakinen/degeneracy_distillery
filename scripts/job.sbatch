#!/usr/bin/env bash
#SBATCH --job-name=job
#SBATCH --account=iusers
#SBATCH --partition=comp
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Generic wrapper: runs whatever command follows the script name inside the
# degen venv on a compute node. Override any #SBATCH line on the command line.
#
#   CPU (comp, AMD 128 cores / 512G nodes):
#     sbatch --job-name=foo scripts/job.sbatch python scripts/foo.py --arg 1
#   GPU (pscomp; V100 by default, --gres=gpu:H100:1 for an H100):
#     sbatch --job-name=foo --partition=pscomp --gres=gpu:1 --cpus-per-task=8 \
#       scripts/job.sbatch python scripts/foo.py
#   Smoke test (testing, 1h max, usually idle):
#     sbatch --partition=testing --time=00:10:00 --cpus-per-task=2 --mem=8G \
#       scripts/job.sbatch python -c "import degeneracy_distillery"
#   Longer than 2 days: --partition=compl (CPU) or pscompl (GPU), max 4-04:00:00.
#
# logs/ must exist before submission: Slurm opens the log files before this
# script starts, and a missing directory makes the job fail with no output.

if [[ $# -eq 0 ]]; then
  echo "usage: sbatch [sbatch options] scripts/job.sbatch <command> [args...]" >&2
  exit 2
fi

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
# Hardcoded default because sbatch propagates the submitting environment, and
# non-interactive submitters (agents, cron) do not source ~/.bashrc.
export SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"

HAS_GPU=0
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && HAS_GPU=1

init_modules() {
  if command -v module >/dev/null 2>&1; then return; fi
  if [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
}

init_modules
if command -v module >/dev/null 2>&1; then
  if [[ "$MODULE_PURGE" == "1" ]]; then
    module purge || true
  fi
  module load "$PYTHON_MODULE"
  if [[ "$HAS_GPU" == "1" ]]; then
    module load "$CUDA_MODULE"
  fi
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Venv '$VENV_DIR' does not exist. Run scripts/setup_slurm_venv.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ "$HAS_GPU" == "1" ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"
else
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

cd "$REPO_DIR"

echo "job:      ${SLURM_JOB_NAME:-<none>} ${SLURM_JOB_ID:-<none>}"
echo "node:     $(hostname)"
echo "gpus:     ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "venv:     $VENV_DIR"
echo "scratch:  $SCRATCH"
echo "git head: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "cmd:      $*"
echo "start:    $(date -Is)"

"$@"
