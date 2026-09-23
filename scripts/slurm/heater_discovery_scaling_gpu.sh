#!/usr/bin/env bash
#SBATCH --job-name=heater-disc
#SBATCH --partition=pscomp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Wrapper for scripts/heater_discovery_dim_scaling_sweep.py -- the rerun that
# discovers the 1-D coordinate independently at every ambient dimension
# (notes/heater_discovery_scaling.md).
#
# Deliberately a new file: scripts/slurm_gpu_sweep.sh's `heater` target points
# at the *original* oracle-coordinate sweep and must keep doing so, since the
# submitted paper numbers are reproduced from it.
#
# RUN selects a preset; extra CLI args are forwarded to the python script.
#
#   RUN=singlecheck sbatch scripts/slurm/heater_discovery_scaling_gpu.sh
#   RUN=full        sbatch --time=24:00:00 scripts/slurm/heater_discovery_scaling_gpu.sh
#   RUN=rank        NOISE=1e-3 sbatch --partition=comp scripts/slurm/heater_discovery_scaling_gpu.sh
#
# RUN=rank stops after rank selection (--rank-only imports neither torch nor
# ltu-ili), so submit it to a CPU partition without --gres.
#
# Runs on any GPU node, including the V100s. The degen venv's torch is
# 2.11.0+cu128, whose arch list is sm_75..sm_120, so torch CUDA kernels cannot
# launch on the sm_70 V100s (h07-h13): they fail with "no kernel image is
# available for execution on the device". JAX is unaffected -- XLA JIT-compiles
# for the actual device -- so fishnets and flattening use the V100 normally and
# only the torch/ltu-ili NPE stage is affected.
#
# DEVICE=auto (the default) therefore probes torch's compiled arch list against
# the allocated GPU and passes --device cpu when they do not match, keeping the
# GPU for JAX. The NPE arms are small (MDN, <=d-dim target, 1000 sims), so the
# CPU fallback costs little. Set DEVICE=cuda or DEVICE=cpu to override.

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_NAME="${ENV_NAME:-degen}"
VENV_DIR="${VENV_DIR:-/home/makinen/venvs/$ENV_NAME}"
PYTHON_MODULE="${PYTHON_MODULE:-intelpython/3-2025.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
MODULE_PURGE="${MODULE_PURGE:-1}"
RUN="${RUN:-singlecheck}"
NOISE="${NOISE:-1e-3}"

# Experiment outputs must NOT land under $HOME (quota-capped at 17.5G).
# Hardcoded rather than relying on the caller exporting SCRATCH, because sbatch
# propagates the *submitting* environment and a non-interactive shell does not
# source ~/.bashrc.
SCRATCH="${SCRATCH:-/data103/makinen/degeneracy_experiments}"
OUT_BASE="${OUT_BASE:-$SCRATCH/heater_discovery}"

init_modules() {
  if command -v module >/dev/null 2>&1; then
    return
  fi

  if [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
}

load_module_if_set() {
  local module_name="$1"
  if [[ -n "$module_name" ]]; then
    module load "$module_name"
  fi
}

init_modules
if command -v module >/dev/null 2>&1; then
  if [[ "$MODULE_PURGE" == "1" ]]; then
    module purge || true
  fi
  load_module_if_set "$CUDA_MODULE"
  load_module_if_set "$PYTHON_MODULE"
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Venv '$VENV_DIR' does not exist. Run scripts/setup_slurm_venv.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"
mkdir -p logs "$OUT_BASE"

export XLA_FLAGS="${XLA_FLAGS:-"--xla_gpu_cuda_data_dir=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

case "$RUN" in
  singlecheck)
    OUT_DIR="${OUT_DIR:-$OUT_BASE/singlecheck}"
    PRESET=(--dims 2 --num-trials 1 --nsims 1000 --n-test 1000
            --sr-time-limit 120 --n-marginal-val 50 --keep-workdirs)
    ;;
  rank)
    OUT_DIR="${OUT_DIR:-$OUT_BASE/rank_ablation_noise$NOISE}"
    PRESET=(--dims 2 3 4 6 8 10 12 --num-trials 3 --nsims 1000 --n-test 1000
            --rank-only --flatten-noise "$NOISE")
    ;;
  full)
    OUT_DIR="${OUT_DIR:-$OUT_BASE/scaling_v1}"
    PRESET=(--dims 2 3 4 6 8 10 12 --num-trials 10 --nsims 1000 --n-test 1000
            --flatten-noise 1e-3 --sr-time-limit 300
            --fishnet-epochs 5000 --flatten-restarts 3 --resume)
    ;;
  full_array)
    # One array task per ambient dimension. Each task owns its own out-dir,
    # because the script rewrites metrics.csv after every run and concurrent
    # tasks sharing a directory would clobber each other. Aggregate across the
    # per-d directories afterwards.
    #   sbatch --array=0-6 RUN=full_array scripts/slurm/heater_discovery_scaling_gpu.sh
    DIMS=(2 3 4 6 8 10 12)
    IDX="${SLURM_ARRAY_TASK_ID:-0}"
    D="${DIMS[$IDX]}"
    if [[ -z "$D" ]]; then
      echo "array index $IDX out of range for dims ${DIMS[*]}"
      exit 2
    fi
    OUT_DIR="${OUT_DIR:-$OUT_BASE/scaling_v1/d$D}"
    PRESET=(--dims "$D" --num-trials 10 --nsims 1000 --n-test 1000
            --flatten-noise 1e-3 --sr-time-limit 300
            --fishnet-epochs 5000 --flatten-restarts 3 --resume)
    ;;
  bigd_array)
    # High-d capacity probe: does a bigger Fisher ensemble + a wider flattener +
    # more restarts lift recovery where the standard config collapses?
    #
    # Motivation, measured on Phase 3: recovery factorises into
    # P(some restart clears the 0.99 alignment gate) x P(recovery | cleared),
    # and BOTH fall with d. Per-restart hit rate goes 0.53 (d=2) -> 0.074 (d=6),
    # so K=3 is ~10x under-provisioned at high d; and the flattener's hidden
    # width is fixed at 256 regardless of d in the shared module.
    #
    # SKIP_NPE=1 (default) answers the recovery question only, which roughly
    # doubles the trials that fit in a given window. SKIP_NPE=0 runs the three
    # NPE arms too, for table rows at this configuration.
    #
    # The two variants MUST use different out-dirs. They produce the same
    # (nsims, d, trial) keys, so pointing the +NPE variant at the skip-npe
    # directory would make --resume treat every run as already done and skip
    # the whole sweep silently.
    DIMS=(8 10 12)
    IDX="${SLURM_ARRAY_TASK_ID:-0}"
    D="${DIMS[$IDX]}"
    if [[ -z "$D" ]]; then
      echo "array index $IDX out of range for dims ${DIMS[*]}"
      exit 2
    fi
    SKIP_NPE="${SKIP_NPE:-1}"
    if [[ "$SKIP_NPE" == "1" ]]; then
      OUT_DIR="${OUT_DIR:-$OUT_BASE/bigd_capacity/d$D}"
      NPE_FLAG=(--skip-npe)
    else
      OUT_DIR="${OUT_DIR:-$OUT_BASE/bigd_capacity_npe/d$D}"
      NPE_FLAG=()
    fi
    PRESET=(--dims "$D" --num-trials "${TRIALS:-5}" --nsims 1000 --n-test 1000
            "${NPE_FLAG[@]}" --sr-time-limit 300
            --fishnet-epochs 5000 --num-fishnets "${NFISH:-20}"
            --flatten-hidden-size "${FHID:-512}"
            --flatten-restarts "${KRESTART:-6}" --resume)
    ;;
  wideprior_check)
    # Cheap validation of the fallback prior before committing Phase 3 to it.
    OUT_DIR="${OUT_DIR:-$OUT_BASE/wideprior/check}"
    PRESET=(--dims 2 4 --num-trials 3 --nsims 1000 --n-test 1000 --skip-npe
            --theta-min 0.3 --theta-max 3 --sigma 0.02
            --sr-time-limit 120 --fishnet-epochs 5000 --flatten-restarts 3
            --keep-workdirs --resume)
    ;;
  wideprior_array)
    # Fallback Phase 3 on the wider prior, to be used only if recovery stays
    # poor at high d on [1,2].
    #
    # Identifiability is set by the ratio theta_max/theta_min -- that is what
    # makes log non-affine across the box, and it is why the additive surrogate
    # sum(X_i) is indistinguishable from the true product on [1,2] (|rho|=0.996,
    # level-set U=0.005). SNR, however, is set by the box's *geometric mean*:
    # U[0.1,2] has exp(E[log theta]) < 1, so P decays with d and the weakest 5%
    # of draws at d=12 sit ~23x below the observation noise, which would confound
    # the scaling claim with a vanishing-signal effect.
    #
    # [0.3,3] keeps ratio 10 (~83% of the identifiability gain of [0.1,2]:
    # U=0.043 vs 0.060 at d=2, rho(linear)=0.961 so even the Spearman criterion
    # discriminates again) while its geometric mean of 1.43 keeps P *growing*
    # with d, holding 5th-percentile SNR at ~381 -- on par with [1,2]'s 582.
    # sigma is dropped to 0.02 to buy headroom; per the notes' own algebra
    # F = (|k|^2/sigma^2) outer(P/theta, P/theta) is rank 1 for any sigma, so
    # sigma rescales the Fisher uniformly and changes neither the rank nor the
    # degeneracy structure.
    DIMS=(2 3 4 6 8 10 12)
    IDX="${SLURM_ARRAY_TASK_ID:-0}"
    D="${DIMS[$IDX]}"
    if [[ -z "$D" ]]; then
      echo "array index $IDX out of range for dims ${DIMS[*]}"
      exit 2
    fi
    OUT_DIR="${OUT_DIR:-$OUT_BASE/wideprior/scaling_v1/d$D}"
    PRESET=(--dims "$D" --num-trials 10 --nsims 1000 --n-test 1000
            --theta-min 0.3 --theta-max 3 --sigma 0.02
            --flatten-noise 1e-3 --sr-time-limit 300
            --fishnet-epochs 5000 --flatten-restarts 3 --resume)
    ;;
  custom)
    OUT_DIR="${OUT_DIR:-$OUT_BASE/custom}"
    PRESET=()
    ;;
  *)
    echo "Unknown RUN='$RUN' (expected singlecheck|rank|full|full_array|wideprior_check|wideprior_array|custom)"
    exit 2
    ;;
esac

DEVICE="${DEVICE:-auto}"
if [[ "$DEVICE" == "auto" && "$RUN" != "rank" ]]; then
  # Ask torch whether it holds kernels for the GPU we were actually allocated.
  # sm_70 (V100) is absent from the cu128 build, so fall back to CPU for the
  # NPE arms rather than dying mid-run. JAX keeps the GPU either way.
  DEVICE=$(python - <<'PY'
try:
    import torch
    if not torch.cuda.is_available():
        print("cpu")
    else:
        major, minor = torch.cuda.get_device_capability(0)
        archs = {a.split("_")[1] for a in torch.cuda.get_arch_list() if a.startswith("sm_")}
        print("cuda" if f"{major}{minor}" in archs else "cpu")
except Exception:
    print("cpu")
PY
)
fi

echo "node: $(hostname)"
echo "repo: $REPO_DIR"
echo "venv: $VENV_DIR"
echo "run: $RUN"
echo "out_dir: $OUT_DIR"
echo "CUDA_PATH: ${CUDA_PATH:-}"
echo "torch device: $DEVICE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

DEVICE_ARGS=()
if [[ "$RUN" != "rank" ]]; then
  DEVICE_ARGS=(--device "$DEVICE")
fi

python scripts/heater_discovery_dim_scaling_sweep.py \
  "${PRESET[@]}" \
  --out-dir "$OUT_DIR" \
  "${DEVICE_ARGS[@]}" \
  "$@"
