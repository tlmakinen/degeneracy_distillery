# One-step + Operon SR validation — handoff

Branch `new_idea`. Interpreter: `/home/makinen/venvs/degen/bin/python` (the
anaconda `pyoperon*` envs fail at `import esr.generation.generator` — no MPI).
Outputs: `/data103/makinen/degeneracy_experiments/` (never `$HOME`, 17.5G quota).

## What is being tested

Does Operon SR recover the true coordinates from the one-step neural map?

Pipeline: fit η(θ) and η̂(x) jointly under `½‖η−η̂‖² − ½log det(JJᵀ)` → read r̂
from the η spectrum → K bootstrap members → align → Operon SR on η(θ) → MDL
pick from the Pareto front → score against known ground truth.

Scripts:
- `new_idea/scalar_oneshot_sr.py` — Rosenbrock suite (problems from
  `new_idea/compare_rosen_ab.py`), `scripts/slurm/scalar_oneshot_sr.sh`
- `new_idea/camels_oneshot.py` — CAMELS-SB35, all 35 params,
  `scripts/slurm/camels_oneshot.sh`

## Pass criterion

For each problem, `coord_fn` gives the true coordinates. η is defined only up
to a rotation among informative axes, so each true coordinate is regressed on
**all** r̂ SR expressions jointly (cubic `poly_r2`). The headline is the
**worst** true coordinate. PASS = worst ≥0.99 AND r̂ == true rank. (A too-large r̂ still spans the true
coordinates, so R² alone reports success on a wrong-rank fit — banana4 does
exactly this.)

| problem | true coords | # | rank | mode |
|---|---|---|---|---|
| `banana{2,3,4}` | θ₀, θ₁−θ₀² | 2 | 2 | hard cut |
| `uncoupled4` | θ₀, θ₁−θ₀², θ₂, θ₃−θ₂² | 4 | 4 | hard cut |
| `coupled{3,4}` | θ₀…θ_{N−1} + links | 2N−1 | N | hard cut |
| `scalar{3,4}` | f=Σ(θ_{i+1}−θ_i²)²+(1−θ_i)² | 1 | 1 | rectangular |

## Status

**Passing (jobs 3828577 default, 3828859 full):** `scalar3` r̂=1
R²(f|SR)=0.9931/0.9957; `scalar4` r̂=1 0.9933/0.9713. Three-step on these gets
0.2094 / 0.0346. SR recovers the coupling structure explicitly — the d=4 winner
contains (θ₀²−θ₁)², (θ₁²−θ₂)², (θ₂²−θ₃)², all three links.

**Queued, never run before:** 3828886 (`banana3` quick smoke of the new
screening path) → 3828887 (`banana2 banana3 banana4`) and 3828888
(`uncoupled4 coupled3 coupled4`), both `afterok:3828886`.

**CAMELS 3828872 RUNNING** (h12): r̂=7 of 35 with no pre-selection, stable from
smoke to full. Three-step's Fisher floor reads rank 35 and needed a hand-built
heuristic to cut to 6. LOO calibration ratios [1.03 1.01 0.97 0.99 0.98 1.01
0.99] — well calibrated at K=8. Parameter usage puts all four feedback
amplitudes (A_SN1, A_SN2, A_AGN1, A_AGN2) in the top 7; the old Fisher
heuristic dropped every one of them. Awaiting SR + MDL.

## Two fixes made just before queueing (untested code is now in flight)

1. **Scorer read only `f_true_te[:, 0]`.** Correct for `scalar{3,4}` (1
   coordinate), wrong for the other six — `coupled3` has 5 true coordinates and
   it would have scored only θ₀, reporting a near-1 pass. Now scores every
   coordinate, rotation-invariantly, and reports the worst.
2. **No screening step existed.** `build_oneshot` hardcoded the active set to
   all d coordinates while still adding the `(d−m)·log w` hard-cut complement —
   the two treatments mixed. Ported `screen_inputs` (group-L1 probe) from
   `compare_rosen_ab`; `active_for(m)` now selects the log-det subset.
   **Job 3828886 is the first test of this path.** If it fails, the two
   dependent jobs never start.

## Known gotchas

- `filter_pareto_fronts` hangs unboundedly on nested-`exp` rows (one job sat 19h).
  Use `timed_predicate` in `scalar_oneshot_sr.py` (`--predicate-timeout`, 30s).
  SIGALRM does NOT work — sympy blocks in C. See memory `project_sr_predicate_hang`.
- The **Frobenius** criterion has failed on every run so far (0.03–0.50). Only
  the MDL pick works. Do not report Frobenius as a result.
- `pow` stays in `allowed_symbols` (user's call). Measured cost on `scalar3`:
  filter removed 7/27 including the MDL winner, replacement scored identically
  (0.9957) at lower complexity. Both filtered and unfiltered fronts are analysed
  and scored — keep it that way, there is no ground truth on CAMELS.
- GPU launchers need `XLA_FLAGS=--xla_gpu_cuda_data_dir=...` or every kernel
  compile dies with `libdevice not found`.
- Watcher: `scratchpad/watch_chain.sh <jobids>` exits on completion OR a 75-min
  log stall. Job state alone missed the 19h hang.
- `scalar4` scored *worse* at full budget than default (0.9713 vs 0.9933) —
  unexplained, more SR time bought a more contorted expression.
