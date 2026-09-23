# One-step reruns of SIR, GW TaylorF2 and GW IMRPhenomD (23 Sep 2026)

Ports of three NeurIPS rebuttal discovery experiments onto the `new_idea`
one-step joint objective, plus the diagnosis of the one place where the
one-step number came out well below the published one. Committed as `db4b146`
on `new_idea`. Weak lensing was in the original scope and was dropped: it has
no discovery script, no notebook and no dataset on this cluster.

## What was built

Two adapters, no new driver and no new sbatch file:

- `degeneracy_distillery/problems/sir.py` -> wraps `scripts/sir_notebook_run.py`
- `degeneracy_distillery/problems/gw.py` -> `GWTaylorF2Problem` wraps
  `scripts/gw_notebook_run.py`, `GWIMRPhenomDProblem` wraps
  `scripts/imrphenomd_notebook_run.py`
- registry clauses in `degeneracy_distillery/problems/base.py`
- `scripts/slurm/oneshot_rebuttal_{gpu,cpu}.sh`

Both adapters import the published driver by importlib rather than
reimplementing the simulator, hand `theta` to the one-step driver already
scaled to `(1.0, 2.0)` to match `fit_theta_scaler` in those scripts, and in
`gates()` convert expressions back to physical units with
`expressions_to_physical` before calling the published correlation function
unchanged. So `physics_alignment` is the identical statistic and
`scripts/aggregate_seed_sweep.py` reads the output with no modification.

`expected_rank` is `None` on both. The published records report
`rank_deficient=False`, so there is no ground-truth rank to score against; the
gate is the published correlation criterion.

**Do not edit `scripts/*_notebook_run.py`.** They hold the published record and
the adapters depend on their current behaviour.

## Headline comparison

Three-step numbers from `follow_up_results/<exp>/rebuttal/aggregate_summary.md`.
All 30 one-step seeds finished; every non-success is a clean threshold miss,
zero crashes.

| Experiment | Three-step (published) | One-step |
|---|---|---|
| SIR | 10/10, 0.674 [0.633, 0.745] | 10/10, 0.667 [0.635, 0.783] |
| GW TaylorF2 | 10/10, 0.972 [0.966, 0.981] | 8/10, 0.983 [0.980, 0.987] |
| GW IMRPhenomD | 7/10, 0.997 [0.986, 0.999] | 5/10, 0.991 [0.986, 0.994] |

Median [IQR] of physics alignment. Note the one-step alignment *medians* are at
or above the published ones in all three cases; where recovery is lower it is
lower for a structural reason, not because the maps are worse.

Artifacts: `$SCRATCH/oneshot_rebuttal/{sir,gw_taylorf2,gw_imrphenomd}/`
(`$SCRATCH` = `/data103/makinen/degeneracy_experiments`), per-seed
`run_record.json` in the published schema plus `aggregate_summary.{md,json}`.

Not comparable between the two arms, and worth saying so rather than papering
over: seeding (rebuttal uses master seeds 0-9 with a per-stage additive stride,
the one-step driver uses `seed + 7919*trial + 31*d`), float precision
(`oneshot.py` enables x64 at import, the notebook scripts do not), and the
`heldout_geometry` block, which is a Fisher-distance-from-identity the one-step
pipeline never builds. `aggregate_seed_sweep.py` prints `n/a` for those rows.

## Why IMRPhenomD was 5/10: rank selection, not fitting

The published criterion is a conjunction (`imrphenomd_notebook_run.py:843-845`):
total-mass `|r| >= 0.75` **and** mass-difference `|r| >= 0.5`, each a max over
*components*. `r_hat=1` on all ten seeds, and the info rule is reading the data
correctly -- the second probe axis carries **-0.06 to +0.002 nats** of held-out
information on every seed. What varies is `m_fit`, the width the ensemble and
SR actually run at, and it is set by `refit_rank` (`oneshot.py:245-266`), which
pads `r_hat` by one axis only when the probe spectrum ratio `lam[0]/lam[1]`
falls **below `min_gap=10`**.

That threshold splits the seeds exactly, 10/10 predicted:

| seed | `rank_margin` | ratio | `m_fit` | total | dm | |
|---|---|---|---|---|---|---|
| 2 | 0.839 | 5.4 | 2 | 0.701 | 0.958 | fail (total) |
| 6 | 0.834 | 5.3 | 2 | 0.991 | 0.973 | pass |
| 4 | 0.871 | 5.7 | 2 | 0.990 | 0.713 | pass |
| 9 | 0.931 | 6.4 | 2 | 0.982 | 0.948 | pass |
| 3 | 1.058 | 8.3 | 2 | 0.991 | 0.995 | pass |
| 7 | 1.057 | 8.3 | 2 | 0.998 | 0.979 | pass |
| **0** | **1.178** | **10.6** | **1** | 0.996 | 0.084 | fail |
| 1 | 1.824 | 38.4 | 1 | 0.991 | 0.016 | fail |
| 5 | 2.019 | 57.0 | 1 | 1.000 | 0.025 | fail |
| 8 | 2.138 | 72.1 | 1 | 0.981 | 0.014 | fail |

`rank_margin = 0.5 log(ratio)`, so `min_gap=10` is margin 1.151. Seed 0 misses
by 6% in a probe eigenvalue.

**At `m_fit=1` the criterion is unsatisfiable, not merely unmet.**
`mass_targets_correlations` draws `m1, m2` i.i.d. uniform on `[5,50]`, so
`cov(M, dm) = Var(m1) - Var(m2) = 0` (measured -0.0007). M and dm are exactly
orthogonal over the criterion's own sampling distribution, hence for any single
scalar `f`, `rho(f,M)^2 + rho(f,dm)^2 <= 1`:

| seed | rho(f,M) | max possible rho(f,dm) | observed |
|---|---|---|---|
| 0 | 0.996 | 0.089 | 0.084 |
| 1 | 0.991 | 0.134 | 0.016 |
| 5 | 1.000 | 0.000 | 0.025 |
| 8 | 0.981 | 0.194 | 0.014 |

Seed 0 saturates the bound. Those four maps are as good as one coordinate can
be. The fifth failure, seed 2, is unrelated: it had two coordinates and both
missed total mass (0.701 vs 0.75).

**Why the three-step got 7/10 on the same criterion.** At `d=2` it is a square
diffeomorphism, `m=d=2` by construction; it never chooses a rank, so it always
hands SR two coordinates however weakly the second is determined. The one-step's
extra capability -- deciding the rank from held-out information -- is precisely
what costs it here, because this criterion rewards reporting an axis the data
barely supports. That is a defensible result to report as-is, not a bug.

## Forced-`m=2` discovery run

Job `3833491`, seeds 0-9, `--rank-min-gap 1e9` (makes `refit_rank`'s pad always
fire, so `m_fit = min(r_hat+1, cap) = 2`). Every other flag identical to the
baseline array `3833465`. Output:
`$SCRATCH/oneshot_rebuttal/gw_imrphenomd_m2_discovery/`.

Recovery **9/10 on the frozen-NLL pick, 10/10 on the MDL pick** (baseline 5/10,
published three-step 7/10). All four `m_fit=1` seeds clear the dm gate:
0.014->0.929, 0.025->0.969, 0.084->0.932, 0.016->0.702. The diagnosis above is
confirmed end to end.

### The coordinates are (M, dm)-aligned, but not literally m1+m2 and m1-m2

Both theta axes span the same physical box `[5,50]`, so the fitted `(1,2)`
scalers are near-identical per axis and `X1 +/- X2` reads directly as
`m1 +/- m2`.

The sum coordinate always appears as a *monotone function of a near-equally
weighted sum*, never the bare sum:

```
seed 4:  (0.991*X1 + X2)^2.769
seed 9:  (X1 + 0.945*X2)^3.933
seed 5:  (0.500*X1 + 0.503*X2 - 1)^0.880
seed 6:  exp(0.870*X1) * exp(0.886*X2)   = exp(0.870*X1 + 0.886*X2)
```

Weight ratios 0.99, 0.95, 0.99, 0.98 -- the *direction* is `m1+m2` to within 6%,
wrapped in an arbitrary power or exponential. Expected: the one-step loss is
invariant under reparametrising each coordinate, so it has no reason to prefer
the linear representative, and SR takes whichever wrapper the Pareto front makes
cheapest.

The difference coordinate is consistently opposite-sign but **tilted**:

```
seed 5:  -1.838*X1 + 3.063*X2
seed 8:  -1.653*X1 + 3.836*X2
seed 9:  -2.007*X1 + 3.587*X2
seed 0:  -3.571*X1 + exp(0.606*X2)
```

Magnitude ratios ~1.7-2.3 rather than 1, which is what produces
`|r(dm)| ~ 0.93-0.97` with 0.24-0.38 of residual sum leakage instead of a clean
1.0. Seed 5's coefficients decompose as 0.970 along `(1,-1)` and 0.242 along
`(1,1)`, matching its measured 0.970 / 0.238.

Per-component correlations, MDL pick:

| seed | c0 \|r(M)\| / \|r(dm)\| | c1 \|r(M)\| / \|r(dm)\| | |
|---|---|---|---|
| 0 | 0.990 / 0.013 | 0.377 / 0.927 | clean split |
| 1 | 0.982 / 0.022 | 0.705 / 0.705 | c1 single-mass |
| 2 | 0.989 / 0.068 | 0.287 / 0.952 | clean split |
| 3 | 0.998 / 0.007 | 0.038 / 0.999 | clean split |
| 4 | 0.990 / 0.001 | 0.705 / 0.705 | c1 single-mass |
| 5 | 0.999 / 0.001 | 0.238 / 0.970 | clean split |
| 6 | 0.978 / 0.004 | 0.232 / 0.973 | clean split |
| 7 | 0.999 / 0.002 | 0.212 / 0.976 | clean split |
| 8 | 0.980 / 0.005 | 0.366 / 0.929 | clean split |
| 9 | 0.974 / 0.032 | 0.268 / 0.962 | clean split |

Seeds 1 and 4 do not find the symmetric basis: `c1` collapses to a function of
`m2` alone (`exp(-0.208*X2)`, `exp(0.357*X2)`), which sits at 45 degrees to both
targets at 0.705/0.705. They still clear the 0.5 gate, on a different coordinate
system. Whether that is SR budget or alignment is not settled.

### MDL vs frozen-NLL selection

`picks_agree` is False on 8 of 10 seeds, but it changes the outcome on exactly
one: seed 2, where the NLL pick takes the degenerate `X1*exp(0.591*X2)`
(0.708/0.709) and MDL takes a properly sum-like `c0` (0.989/0.068) with the same
dm component. This was invisible before `db4b146`: `row["expression_mdl"]` was
computed at `oneshot_sweep.py:598` and never copied into the record, so every
`run_record.json` written before that commit has `expression_mdl: null`. The
runs themselves were unaffected -- it was a recording gap, not a computation
gap. Records now also carry `m_fit`, `picks_agree`,
`physics_alignment_mdl` and `complementary_mass_diff_alignment_mdl`.

## Hazards for whoever picks this up

- `--rank-min-gap` is the same knob `jtj_eigengap` uses, so in the forced-`m=2`
  tree `jtj_eigengap_rank`, `jtj_eigengap_rank_screen` and
  `jtj_rank_agreement_k` all read 2 by construction. Vacuous, not evidence.
  It does not touch the fit, alignment, SR or criterion.
- `min_gap` is a frozen config value under the rule in
  `notes/neurips_discovery_reruns.md`. The forced-`m=2` run is a declared
  ablation in its own tree; the baseline 5/10 tree is untouched and is what
  should be quoted against the published table.
- Rectangular `m < d` is **not** a normalised NLL (`new_idea/oneshot_vs_threestep.md:75-86`):
  coarea leaves a dropped Hausdorff factor. Applies to SIR (d=3) and TaylorF2
  when `m_fit < d`. Never quote it as a posterior NLL.
- The driver's 20000-step defaults were tuned at `nsims=2000`. At the rebuttal
  budget of `nsims=500` the volume term overfits well before then -- probe
  validation NLL rises while `r_hat` stays stable. These runs used
  `PROBE_STEPS=2000 ENSEMBLE_STEPS=2000 FROZEN_STEPS=1500`, chosen from a
  4-point diagnostic (jobs 3833384-87) in which `r_hat=2` and
  `r2_true_min ~ 0.99` were stable throughout and `nll_neural` was *best* at
  500 steps.
- `filter_pareto_fronts` can hang unboundedly on nested-`exp` rows; SIGALRM does
  not work because sympy blocks in C. Use the subprocess `timed_predicate`
  pattern from `new_idea/scalar_oneshot_sr.py`.
- Frobenius selection has never worked on any one-step run (0.03-0.50). Report
  the MDL pick and the frozen-NLL pick only.
- No test covers any one-step code path; `tests/test_rosenbrock_pipeline.py` is
  three-step only.
- Cluster rules: no heavy compute on the login node, outputs to `$SCRATCH`
  (`$HOME` is quota-capped at ~17.5G), `mkdir -p logs` before `sbatch`, and only
  `scancel` jobs you submitted yourself.

## Open threads

1. **GW TaylorF2, two misses.** Seeds 15900 (0.690) and 55495 (0.681) sit well
   below the 0.75 gate and well below the rest of the distribution
   (IQR [0.980, 0.987]). Not threshold-boundary cases; worth a spot-check for a
   genuine training failure.
2. **Seeds 1 and 4 above.** At forced `m=2` their second coordinate is a
   single-mass function rather than a difference. Is that SR budget, alignment,
   or a real property of those datasets?
3. **The `min_gap` rule itself.** It is a hard threshold on a continuous random
   quantity, and IMR sits right on top of it. A margin-aware or
   information-aware pad would be a principled change, but it affects every
   one-step experiment, not just this one.
4. **Whether the criterion or the method is at fault for IMR.** A conjunction
   over components rewards reporting a weakly-determined axis. The one-step's
   rank selection is arguably doing the statistically honest thing and being
   penalised for it. Worth stating explicitly in any write-up.

## Job IDs

`3833363` SIR smoke, `3833366` SIR array, `3833380` GW smoke, `3833384-87` step
diagnostic, `3833444` cancelled (unplumbed env vars), `3833451` TaylorF2 array,
`3833462` IMR smoke, `3833465` IMR array, `3833489` cancelled (subsumed),
`3833491` forced-`m=2` array.
