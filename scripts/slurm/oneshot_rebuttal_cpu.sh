#!/usr/bin/env bash
#SBATCH --job-name=1step-reb-cpu
#SBATCH --partition=comp
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# CPU slice of the one-step rebuttal reruns. No GPU request; the GPU script
# body does the rest, exactly as scripts/slurm/oneshot_sweep_cpu.sh does.
#
# Plumbing smoke on the short testing partition (1h cap, usually idle):
#   PROBLEM=sir QUICK=1 sbatch --partition=testing --time=00:55:00 \
#     --array=0 scripts/slurm/oneshot_rebuttal_cpu.sh
#
# Note: the JAX CPU backend was seen to fail at d>=3 in the THREE-step
# flattening stage (notes + memory). The one-step path does not use
# fit_flattening and the Rosenbrock one-step CPU array runs fine at d=4..32,
# so CPU is expected to work here -- but check the log rather than assume.
#
# Slurm runs a spool copy of this file, so resolve the sibling from the
# submit directory.

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_MODULE="${CUDA_MODULE:-}"
NTHREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NTHREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$NTHREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$NTHREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$NTHREADS}"

echo "jax platforms: $JAX_PLATFORMS"
echo "cpu threads:   $NTHREADS"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$REPO_DIR/scripts/slurm/oneshot_rebuttal_gpu.sh"
