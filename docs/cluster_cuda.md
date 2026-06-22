# CUDA Cluster Setup

This repo can be installed on a SLURM GPU cluster with the same basic pattern as
the Colab tutorial notebooks:

```bash
git clone https://github.com/tlmakinen/degeneracy_distillery.git
cd degeneracy_distillery
pip install -e .

git clone https://github.com/DeaglanBartlett/ESR.git
pip install -e ESR
```

The cluster scripts wrap that flow with Cursor CLI setup, Conda environment
creation, CUDA PyTorch, and SLURM submission.

## Login Node Install

Run this from the cluster login node:

```bash
export WORKDIR="$SCRATCH/degeneracy_distillery"
export CONDA_MODULE="miniforge"        # edit for your cluster
export CUDA_MODULE="cuda/12.1"         # edit for your cluster
export TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu121"

bash scripts/cluster_install_cuda.sh "$WORKDIR"
```

If the cluster does not provide modules, leave `CONDA_MODULE` and
`CUDA_MODULE` unset and make sure `conda` is already on `PATH`.

For Cursor CLI on a headless SSH session, prefer an API key:

```bash
export CURSOR_API_KEY="..."
agent status
agent --mode=ask "summarize this repo"
agent --plan "prepare a SLURM run for the SIR sweep"
```

Keep `CURSOR_API_KEY` out of SLURM job logs and shared shell profiles.

## GPU Smoke Test

Submit a tiny Rosenbrock run first:

```bash
cd "$SCRATCH/degeneracy_distillery"
mkdir -p logs

sbatch scripts/slurm_gpu_sweep.sh
```

Override site-specific settings at submission time:

```bash
CONDA_MODULE="miniforge" \
CUDA_MODULE="cuda/12.1" \
REPO_DIR="$SCRATCH/degeneracy_distillery" \
OUT_BASE="$SCRATCH/degen_runs" \
sbatch --partition=gpu --gres=gpu:1 scripts/slurm_gpu_sweep.sh
```

## Full Sweeps

Use `TARGET` to select an existing command-line sweep script and `MODE=full` to
run with the script defaults:

```bash
MODE=full TARGET=rosen sbatch --time=24:00:00 scripts/slurm_gpu_sweep.sh
MODE=full TARGET=sir sbatch --time=24:00:00 scripts/slurm_gpu_sweep.sh
MODE=full TARGET=gw_imr sbatch --time=24:00:00 scripts/slurm_gpu_sweep.sh
MODE=full TARGET=gw_waveform sbatch --time=24:00:00 scripts/slurm_gpu_sweep.sh
MODE=full TARGET=wl sbatch --time=24:00:00 scripts/slurm_gpu_sweep.sh
MODE=full TARGET=heater sbatch --time=24:00:00 scripts/slurm_gpu_sweep.sh
```

Extra arguments are forwarded to the selected Python script in full mode:

```bash
MODE=full TARGET=rosen sbatch --time=24:00:00 scripts/slurm_gpu_sweep.sh --nsims 100 500 1000 --epochs 500
```

Outputs are written under `$OUT_BASE`, defaulting to
`$SCRATCH/degen_runs` when `SCRATCH` is set.
