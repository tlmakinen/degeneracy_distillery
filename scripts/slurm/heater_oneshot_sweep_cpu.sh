#!/usr/bin/env bash
#SBATCH --job-name=heater1s-cpu
#SBATCH --partition=comp
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# CPU half of the heater one-step sweep. No #SBATCH --gres line, so this
# lands on comp. The GPU script body does the rest.
#
#   DIMS="2 3 4" sbatch --array=0 scripts/slurm/heater_oneshot_sweep_cpu.sh
#   DIMS="2 3 4" sbatch --array=0-29 scripts/slurm/heater_oneshot_sweep_cpu.sh
#
# Slurm runs a spool copy of this file, so BASH_SOURCE cannot find the
# sibling GPU script. Resolve it from the submit directory.

export DEVICE="${DEVICE:-cpu}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_MODULE="${CUDA_MODULE:-}"
export DIMS="${DIMS:-2 3 4}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$REPO_DIR/scripts/slurm/heater_oneshot_sweep_gpu.sh"
