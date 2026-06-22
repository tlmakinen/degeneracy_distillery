#!/usr/bin/env bash
set -euo pipefail

# Standalone handoff script for installing Degeneracy Distillery on a SLURM CUDA
# cluster and creating a GPU job launcher.
#
# Copy this file to the cluster login node, then run:
#   bash cluster_cuda_setup.sh
#
# Common overrides:
#   WORKDIR=$SCRATCH/degeneracy_distillery \
#   CONDA_MODULE=miniforge \
#   CUDA_MODULE=cuda/12.1 \
#   TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu121 \
#   bash cluster_cuda_setup.sh
#
# Cursor CLI auth for headless SSH:
#   export CURSOR_API_KEY=...

WORKDIR="${WORKDIR:-${1:-$HOME/degeneracy_distillery}}"
REPO_URL="${REPO_URL:-https://github.com/tlmakinen/degeneracy_distillery.git}"
ENV_NAME="${ENV_NAME:-degen}"
CONDA_ENV_FILE="${CONDA_ENV_FILE:-degen_env_minimal.yml}"
CONDA_MODULE="${CONDA_MODULE:-}"
CUDA_MODULE="${CUDA_MODULE:-}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"
INSTALL_CURSOR="${INSTALL_CURSOR:-0}"
INSTALL_PYCBC="${INSTALL_PYCBC:-1}"
INSTALL_ILI="${INSTALL_ILI:-1}"

load_module_if_set() {
  local module_name="$1"
  if [[ -n "$module_name" ]]; then
    module load "$module_name"
  fi
}

if command -v module >/dev/null 2>&1; then
  module purge || true
  load_module_if_set "$CONDA_MODULE"
  load_module_if_set "$CUDA_MODULE"
fi

if [[ "$INSTALL_CURSOR" == "1" ]] && ! command -v agent >/dev/null 2>&1; then
  curl https://cursor.com/install -fsS | bash
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ "$INSTALL_CURSOR" == "1" ]] && command -v agent >/dev/null 2>&1; then
  agent status || true
fi

mkdir -p "$(dirname "$WORKDIR")"
if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone "$REPO_URL" "$WORKDIR"
else
  git -C "$WORKDIR" pull --ff-only
fi

cd "$WORKDIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available. Set CONDA_MODULE to your cluster's Miniforge/Anaconda module."
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if command -v mamba >/dev/null 2>&1; then
    mamba env create -n "$ENV_NAME" -f "$CONDA_ENV_FILE"
  else
    conda env create -n "$ENV_NAME" -f "$CONDA_ENV_FILE"
  fi
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX"
python -m pip install -e .

if [[ ! -d ESR ]]; then
  git clone https://github.com/DeaglanBartlett/ESR.git
fi
python -m pip install -e ESR

if [[ "$INSTALL_PYCBC" == "1" ]]; then
  python -m pip install pycbc
fi

if [[ "$INSTALL_ILI" == "1" ]]; then
  python -m pip install "ltu-ili[pytorch] @ git+https://github.com/maho3/ltu-ili"
fi

cat > run_gpu_sweep.sh <<'SLURM'
#!/usr/bin/env bash
#SBATCH --job-name=degen-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

MODE="${MODE:-smoke}"
TARGET="${TARGET:-rosen}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
CONDA_MODULE="${CONDA_MODULE:-}"
CUDA_MODULE="${CUDA_MODULE:-}"
OUT_BASE="${OUT_BASE:-${SCRATCH:-$REPO_DIR/results}/degen_runs}"

load_module_if_set() {
  local module_name="$1"
  if [[ -n "$module_name" ]]; then
    module load "$module_name"
  fi
}

if command -v module >/dev/null 2>&1; then
  module purge || true
  load_module_if_set "$CONDA_MODULE"
  load_module_if_set "$CUDA_MODULE"
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd "$REPO_DIR"
mkdir -p logs "$OUT_BASE"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

case "$TARGET" in
  rosen) SCRIPT="scripts/rosen_nsims_logprob_sweep.py" ;;
  sir) SCRIPT="scripts/sir_nsims_logprob_sweep.py" ;;
  wl) SCRIPT="scripts/wl_2d_nsims_logprob_sweep.py" ;;
  gw_imr) SCRIPT="scripts/gw_imr_nsims_logprob_sweep.py" ;;
  gw_waveform) SCRIPT="scripts/gw_waveform_nsims_logprob_sweep.py" ;;
  heater) SCRIPT="scripts/heater_dim_scaling_sweep.py" ;;
  *)
    echo "Unknown TARGET '$TARGET'. Expected: rosen, sir, wl, gw_imr, gw_waveform, heater."
    exit 1
    ;;
esac

python - <<'PY'
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_BASE/${TARGET}_${MODE}_${RUN_ID}"

if [[ "$MODE" == "smoke" ]]; then
  python "$SCRIPT" \
    --device cuda \
    --nsims 100 \
    --epochs 2 \
    --n-test 32 \
    --n-posterior-samples 128 \
    --out-dir "$OUT_DIR"
elif [[ "$MODE" == "full" ]]; then
  python "$SCRIPT" --device cuda --out-dir "$OUT_DIR" "$@"
else
  echo "Unknown MODE '$MODE'. Expected 'smoke' or 'full'."
  exit 1
fi

echo "Run complete: $OUT_DIR"
SLURM
chmod +x run_gpu_sweep.sh

python - <<'PY'
import jax
import torch
import degeneracy_distillery
import esr.generation.generator

try:
    import ili
except ImportError:
    ili = None

print("degeneracy_distillery import: OK")
print("ESR import: OK")
print(f"JAX: {jax.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"ltu-ili import: {'OK' if ili is not None else 'missing'}")
PY

echo
echo "Install complete."
echo "Repo: $WORKDIR"
echo "SLURM helper: $WORKDIR/run_gpu_sweep.sh"
echo
echo "Submit a GPU smoke test:"
echo "  cd \"$WORKDIR\""
echo "  mkdir -p logs"
echo "  sbatch run_gpu_sweep.sh"
echo
echo "Submit a full run:"
echo "  MODE=full TARGET=rosen sbatch --time=24:00:00 run_gpu_sweep.sh"
