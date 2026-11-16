#!/bin/bash
# CAMELS, two steps (see notes/whittle_procedure.md).
#   step 1  screen all 35 parameters, several seeds          scripts/camels_whittle_runs.sh screen
#   step 2  one-step refit on only the top-k parameters       scripts/camels_whittle_runs.sh refit [k ...]
# Laptop launcher by default; set RUN="python scripts/iterative_whittle.py" on the cluster.
RUN="${RUN:-scripts/rebuttal_checks/run_whittle_local.sh}"
PY="${PY:-.venv/bin/python}"
OUT="${OUT:-results/whittle_camels}"
# data vector: 0 = the raw 344 standardised features; 40 = PCA components (the adapter default)
NPCA="${NPCA:-0}"
mkdir -p "$OUT"
STEP="${1:-screen}"; shift

if [[ "$STEP" == "screen" ]]; then
  SCREEN="--problem camels --problem-arg n_pca=$NPCA --skip-discovery --k-target 35 --skip-fill none --score informative --probe-repeats 3 --probe-steps-a 1500"
  # default column order, five seeds (each seed is a different train/test split)
  $RUN $SCREEN --num-trials 5 --out-dir "$OUT/screen" > "$OUT/log_screen.txt" 2>&1 &
  # a second column order: the eight strongest parameters and four of the weakest sit in the
  # skip positions, so every borderline parameter is scored from outside them
  $RUN $SCREEN --num-trials 3 --out-dir "$OUT/screen_front" \
    --front "Omega_m,Omega_b,H_0,IMFslope,n_s,sigma_8,A_SN2,A_SN1,BH_FeedbackFactor,WindEnergyReductionExponent,SNIa_Rate_Norm,UVB_Hep_beta" \
    > "$OUT/log_screen_front.txt" 2>&1 &
  wait
  $PY scripts/whittle_aggregate.py "$OUT/screen" "$OUT/screen_front" --top-k "${TOPK:-14}" | tee "$OUT/screen_summary.txt"
fi

if [[ "$STEP" == "refit" ]]; then
  KS="${*:-14 12}"
  REFIT="--problem camels --problem-arg n_pca=$NPCA --skip-discovery --refit-steps ${STEPS:-6000} --refit-members ${MEMBERS:-4}"
  for K in $KS; do
    ONLY=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['table'][i]['name'] for i in range(int(sys.argv[2]))))" "$OUT/screen/screen_summary.json" "$K")
    # eta(theta) and the data summary both K-dimensional; the other parameters are marginalised
    $RUN $REFIT --only "$ONLY" --refit-m -1 --out-dir "$OUT/refit_top$K" > "$OUT/log_refit_top$K.txt" 2>&1 &
  done
  # reference: all 35 parameters with 12 latents, same budget
  ALL=$($PY -c "import json,sys; print(','.join(r['name'] for r in json.load(open(sys.argv[1]))['table']))" "$OUT/screen/screen_summary.json")
  $RUN $REFIT --only "$ALL" --refit-m 12 --out-dir "$OUT/refit_all35_m12" > "$OUT/log_refit_all35_m12.txt" 2>&1 &
  wait
  for d in "$OUT"/refit_*; do echo "== $d"; grep -h "^\[refit\]" "$OUT/log_$(basename $d).txt" | cut -c1-260; done
fi
