#!/usr/bin/env bash
# Generic master-seed sweep wrapper for the notebook_run.py pipelines.
#
# Usage:
#   scripts/run_seed_sweep.sh <target> [mode] [seed_start] [seed_end]
#
# Examples:
#   scripts/run_seed_sweep.sh rosenbrock smoke 0 2       # 3-seed local dry run
#   scripts/run_seed_sweep.sh rosenbrock rebuttal 0 9    # full 10-seed rebuttal sweep
#
# Unlike the per-experiment slurm/*_notebook_gpu.sh launchers (one job per
# script), this loops seeds within a single process/job so it can be run
# interactively or from inside one Slurm allocation. It does NOT abort on a
# per-seed failure -- every seed's outcome is recorded, and a final pass/fail
# tally is printed so failed runs stay in the denominator instead of being
# silently dropped.
set -uo pipefail

TARGET="${1:?usage: run_seed_sweep.sh <target> [mode] [seed_start] [seed_end]}"
MODE="${2:-smoke}"
SEED_START="${3:-0}"
SEED_END="${4:-9}"

REPO_DIR="${REPO_DIR:-$PWD}"
OUT_BASE="${OUT_BASE:-$REPO_DIR/follow_up_results}"
REQUIRE_GPU_FLAG="${REQUIRE_GPU_FLAG:-}"  # set to "--require-gpu" or "--no-require-gpu"

# target -> script path. Add an entry here once a script gets the same
# master-seed treatment as scripts/rosenbrock_notebook_run.py.
declare -A TARGET_SCRIPT=(
  [rosenbrock]="scripts/rosenbrock_notebook_run.py"
  [gw_taylorf2]="scripts/gw_notebook_run.py"
  [gw_imrphenomd]="scripts/imrphenomd_notebook_run.py"
  [qm7b]="scripts/qm7b_notebook_run.py"
  [sir]="scripts/sir_notebook_run.py"
  [ising]="scripts/ising_notebook_run.py"
  [kolmogorov]="scripts/kolmogorov_notebook_run.py"
  [kuramoto]="scripts/kuramoto_notebook_run.py"
)

# Most scripts take --seed; the intractable-likelihood family (ising/kolmogorov/
# kuramoto) takes --master-seed instead. Default to --seed, override per target.
declare -A TARGET_SEED_FLAG=(
  [ising]="--master-seed"
  [kolmogorov]="--master-seed"
  [kuramoto]="--master-seed"
)

SCRIPT="${TARGET_SCRIPT[$TARGET]:-}"
if [[ -z "$SCRIPT" ]]; then
  echo "Unknown target '$TARGET'. Known targets: ${!TARGET_SCRIPT[*]}" >&2
  exit 1
fi
SEED_FLAG="${TARGET_SEED_FLAG[$TARGET]:---seed}"

cd "$REPO_DIR"
mkdir -p logs "$OUT_BASE/$TARGET/$MODE"

declare -a FAILED_SEEDS=()
declare -a SUCCEEDED_SEEDS=()

for seed in $(seq "$SEED_START" "$SEED_END"); do
  out_dir="$OUT_BASE/$TARGET/$MODE/seed_${seed}"
  log_file="logs/${TARGET}_${MODE}_seed${seed}.log"
  echo "=== ${TARGET} mode=${MODE} seed=${seed} -> ${out_dir} (log: ${log_file}) ==="

  # shellcheck disable=SC2086
  python "$SCRIPT" \
    --mode "$MODE" \
    "$SEED_FLAG" "$seed" \
    --out-dir "$out_dir" \
    $REQUIRE_GPU_FLAG \
    >"$log_file" 2>&1
  status=$?

  if [[ $status -eq 0 ]]; then
    echo "  seed ${seed}: OK"
    SUCCEEDED_SEEDS+=("$seed")
  else
    echo "  seed ${seed}: FAILED (exit ${status}); see ${log_file}"
    FAILED_SEEDS+=("$seed")
  fi
done

total=$(( SEED_END - SEED_START + 1 ))
echo ""
echo "=== ${TARGET} mode=${MODE} sweep complete: ${#SUCCEEDED_SEEDS[@]}/${total} succeeded ==="
if [[ ${#FAILED_SEEDS[@]} -gt 0 ]]; then
  echo "Failed seeds: ${FAILED_SEEDS[*]}"
fi

exit 0
