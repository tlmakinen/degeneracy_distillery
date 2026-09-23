#!/usr/bin/env bash
#SBATCH --job-name=oneshot-cpu
#SBATCH --partition=pscomp
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --exclude=h07,h08,h09,h10,h11,h12,h13,j01,j02
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# CPU slice of the one-step sweep. No GPU request, and the exclude list
# keeps these tasks off the V100/H100 nodes so they do not block the GPU
# array. The GPU script body does the rest.
#
#   PROBLEM=rosenbrock sbatch --array=1-29 scripts/slurm/oneshot_sweep_cpu.sh
#
# Slurm runs a spool copy of this file, so resolve the sibling from the
# submit directory.

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_MODULE="${CUDA_MODULE:-}"
NTHREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NTHREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$NTHREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$NTHREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$NTHREADS}"

echo "jax platforms: $JAX_PLATFORMS"
echo "cpu threads:   $NTHREADS"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$REPO_DIR/scripts/slurm/oneshot_sweep_gpu.sh"
