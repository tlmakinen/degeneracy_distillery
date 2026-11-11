#!/usr/bin/env bash
# Discovery + NPE budget sweep with the discovery simulations inside the budget.
#
#   scripts/run_reuse_rerun.sh sir|taylorf2|imrphenomd [discovery|sweep|all]
#
# Run from the repository root with the project environment active. Outputs go
# to $OUT_BASE/<problem>/ (default results/reuse_rerun/<problem>/):
#   discovery/            discovery driver output (run_record.json, SR artefacts)
#   discovery_sims.npz    the discovery simulations (+ PCA basis for GW)
#   sweep_exprs.json      discovered expressions, rewritten for the sweep
#   sweep/                NPE sweep output + summary.csv / summary.json
#   sweep_train_only/     the same with POOL=train_only
#
# See notes/reuse_rerun_results.md for what this was run with and what came out.

set -euo pipefail

PROBLEM="${1:?usage: run_reuse_rerun.sh sir|taylorf2|imrphenomd [discovery|sweep|all]}"
STAGE="${2:-all}"
SEED="${SEED:-0}"
MODE="${MODE:-rebuttal}"
NSIMS="${NSIMS:-500 1000 2500 5000}"
DEVICE="${DEVICE:-cpu}"
OUT_BASE="${OUT_BASE:-results/reuse_rerun}"
# POOL=all (default): pool = discovery training sims, then discovery held-out
# sims, then fresh sims. POOL=train_only: discovery training sims, then fresh.
POOL="${POOL:-all}"
OUT="$OUT_BASE/$PROBLEM"
mkdir -p "$OUT"

# macOS only: System Integrity Protection drops DYLD_* variables when a script is
# started through /usr/bin/env, so a library path that pyoperon's wheel needs
# has to be set from inside the script. Unset (the default), nothing happens.
if [[ -n "${REUSE_DYLD_FALLBACK_LIBRARY_PATH:-}" ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="$REUSE_DYLD_FALLBACK_LIBRARY_PATH"
fi

case "$PROBLEM" in
  sir)
    DRIVER=scripts/sir_notebook_run.py
    SWEEP=scripts/sir_nsims_logprob_sweep.py
    # 0.01 is the noise level the paper states; both scripts default to 0.05.
    DRIVER_EXTRA=(--noise-std 0.01)
    SWEEP_EXTRA=(--noise-std 0.01)
    ;;
  taylorf2)
    DRIVER=scripts/gw_notebook_run.py
    SWEEP=scripts/gw_waveform_nsims_logprob_sweep.py
    DRIVER_EXTRA=()
    SWEEP_EXTRA=()
    ;;
  imrphenomd)
    DRIVER=scripts/imrphenomd_notebook_run.py
    SWEEP=scripts/gw_imr_nsims_logprob_sweep.py
    DRIVER_EXTRA=()
    SWEEP_EXTRA=()
    ;;
  *)
    echo "unknown problem '$PROBLEM'" >&2
    exit 1
    ;;
esac

if [[ "$STAGE" == "discovery" || "$STAGE" == "all" ]]; then
  python "$DRIVER" \
    --mode "$MODE" --seed "$SEED" --no-require-gpu \
    ${DRIVER_EXTRA[@]+"${DRIVER_EXTRA[@]}"} \
    --out-dir "$OUT/discovery" \
    --save-training-data "$OUT/discovery_sims.npz"
fi

SWEEP_DIR="$OUT/sweep"
if [[ "$POOL" == "train_only" ]]; then
  SWEEP_EXTRA+=(--training-data-train-only)
  SWEEP_DIR="$OUT/sweep_train_only"
elif [[ "$POOL" != "all" ]]; then
  echo "unknown POOL '$POOL' (expected all or train_only)" >&2
  exit 1
fi

if [[ "$STAGE" == "sweep" || "$STAGE" == "all" ]]; then
  python scripts/reuse_rerun_tools.py exprs \
    --problem "$PROBLEM" \
    --discovery-dir "$OUT/discovery" \
    --out "$OUT/sweep_exprs.json" > /dev/null
  # Read the expressions into an array, one per line (bash 3.2 has no mapfile).
  EXPRS=()
  while IFS= read -r line; do EXPRS+=("$line"); done < <(
    python -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["sweep_theta_to_eta"]))' \
      "$OUT/sweep_exprs.json"
  )

  # shellcheck disable=SC2086
  python "$SWEEP" \
    --nsims $NSIMS --seed "$SEED" --device "$DEVICE" \
    ${SWEEP_EXTRA[@]+"${SWEEP_EXTRA[@]}"} \
    --training-data "$OUT/discovery_sims.npz" \
    --theta-to-eta "${EXPRS[@]}" \
    --eta-inverse numeric \
    --paired-arm-seeds \
    --run-coverage --coverage-all-nsims \
    --eval-test-logprob \
    --out-dir "$SWEEP_DIR"

  python scripts/reuse_rerun_tools.py summarize --problem "$PROBLEM" --sweep-dir "$SWEEP_DIR"
fi
