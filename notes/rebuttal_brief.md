# Rebuttal discovery reruns: three-step vs one-step (25 Sep 2026)

Four NeurIPS rebuttal discovery experiments, each run as ten independent
end-to-end trials under two pipelines. The three-step arm is the published
rebuttal result. The one-step arm reruns the same experiments under the joint
objective on branch `new_idea`. The question is narrow: does the more stable
objective reproduce what the published pipeline found?

## The two arms

**Three-step** fits a Fishnets compressor, flattens the Fisher in `theta`,
aligns an ensemble, then runs symbolic regression on the aligned coordinates.
Drivers are `scripts/{rosenbrock,sir,gw,imrphenomd}_notebook_run.py`. These
files hold the published record and were not modified.

**One-step** fits the coordinate map `eta(theta)` and the estimator
`etahat(x)` together under

    L = E[ 0.5 ||eta(theta) - etahat(x)||^2 - 0.5 log det(J J^T) ]

and never builds a Fisher in `theta`. Driver is `scripts/oneshot_sweep.py`.
Each experiment reaches it through a `Problem` adapter
(`degeneracy_distillery/problems/{sir,gw,rosenbrock}.py`) that wraps the
published simulator by `importlib` rather than reimplementing it.

The `sir` and `gw` adapters hand `theta` to the driver already scaled to
`(1.0, 2.0)`, matching `fit_theta_scaler` in the published scripts, and in
`gates()` convert the discovered expressions back to physical units with
`expressions_to_physical` before calling the **published** correlation function
unchanged. Physics alignment is therefore the identical statistic in both arms.
Rosenbrock is the exception; see *What is not comparable*.

## Experiments

| Experiment | `d` | Parameters | Expected coordinate | Criterion | Threshold |
|---|---|---|---|---|---|
| Rosenbrock | 2 | `theta1, theta2` | valley `theta2 - theta1^2` | `physics_alignment` and `complementary_linear_alignment` | 0.5 and 0.5 |
| SIR | 3 | `beta, gamma, I0/10` | `R0 = beta / gamma` | `physics_alignment` | 0.5 |
| GW TaylorF2 | 2 | `m1, m2` | chirp mass `M_c` | `physics_alignment` | 0.75 |
| GW IMRPhenomD | 2 | `m1, m2` | total mass `M` and `dm = m1 - m2` | `physics_alignment` and `complementary_mass_diff_alignment` | 0.75 and 0.5 |

Alignment is `|Pearson r|` on held-out points against the expected coordinate.
Criteria are the pre-registered ones in
`scripts/recompute_success_at_threshold.py::EXPERIMENTS`; `make_rebuttal_tables.py`
recomputes recovery from them rather than trusting either driver's `success`
field, because the two arms set that field by different rules.

SIR varies all three parameters: `I0 = np.random.poisson(I0_MEAN)` per
simulation (`scripts/sir_notebook_run.py:282`), stacked into `theta` at `:309`.
`I0/10` reaches the distillery alongside `beta` and `gamma`.

## Budget

500 training simulations and 500 held-out simulations per trial, in both arms.
SR augmentation uses 2000 evaluations of the learned coordinate map (10000 for
SIR) — these are cheap network calls and are **not** simulator calls. For both
GW experiments the PCA basis is fitted inside the same `simulator_data` call
that produces train and test, so the basis is shared across the split exactly
as the published runs had it, and that waveform cost is reported separately in
`counts.n_pca_simulations`.

## Results

| Experiment | Train sims | Three-step recovered | Three-step alignment | One-step recovered | One-step alignment |
|---|---|---|---|---|---|
| Rosenbrock | 500 | 10/10 | 0.929 (0.814,0.960) | n/a * | 0.999 (0.999,1.000) * |
| SIR | 500 | 10/10 | 0.674 (0.636,0.741) | 10/10 | 0.667 (0.637,0.779) |
| GW TaylorF2 | 500 | 10/10 | 0.972 (0.966,0.979) | 8/10 | 0.983 (0.967,0.983) |
| GW IMRPhenomD | 500 | 7/10 | 0.997 (0.988,0.999) | 5/10 | 0.991 (0.984,0.995) |

`*` different statistic; see below.

All 40 one-step trials finished. There are **no crashes** in either arm — every
non-recovery is a threshold miss, which stays in the denominator.

Note that the one-step alignment *medians* are at or above the three-step ones
on SIR, TaylorF2 and IMRPhenomD. Where recovery is lower it is lower for a
structural reason, not because the maps are worse.

### Config variants

Deliberate config changes, kept out of the headline table so it stays
like-for-like.

| Variant | Change | Recovered (NLL pick) | Recovered (MDL pick) | Alignment |
|---|---|---|---|---|
| SIR, info-floor 0.5 | `--info-floor 0.5` | 10/10 | 10/10 | 0.770 (0.633,0.862) |
| GW IMRPhenomD, forced `m=2` | `--rank-min-gap 1e9` | 9/10 | 10/10 | 0.987 (0.981,0.996) |

The MDL column is `n/a` for the three baseline one-step trees: they were run
before `db4b146`, when `expression_mdl` was computed and then dropped before
the record was written, so those records carry no MDL pick to score.

## Why IMRPhenomD is 5/10

Not a fitting failure. `r_hat = 1` on all ten seeds and the information rule is
right: the second probe axis carries **-0.06 to +0.002 nats** of held-out
information on every seed. What varies is `m_fit`, the width the ensemble and
SR actually run at, set by `refit_rank`
(`degeneracy_distillery/oneshot.py:245-266`), which pads `r_hat` by one axis
only when the probe spectrum ratio `lam[0]/lam[1]` falls below `min_gap = 10`.
That threshold splits the seeds exactly, 10 of 10 predicted.

At `m_fit = 1` the conjunctive criterion is **unsatisfiable, not merely unmet**.
`mass_targets_correlations` draws `m1, m2` i.i.d. uniform on `[5,50]`, so
`cov(M, dm) = Var(m1) - Var(m2) = 0` (measured -0.0007). `M` and `dm` are
orthogonal over the criterion's own sampling distribution, hence for any single
scalar `f`, `rho(f,M)^2 + rho(f,dm)^2 <= 1`:

| seed | `rho(f,M)` | max possible `rho(f,dm)` | observed | gate |
|---|---|---|---|---|
| 0 | 0.996 | 0.089 | 0.084 | 0.5 |
| 1 | 0.991 | 0.134 | 0.016 | 0.5 |
| 5 | 1.000 | 0.000 | 0.025 | 0.5 |
| 8 | 0.981 | 0.194 | 0.014 | 0.5 |

Seed 0 saturates the bound. Those four maps are as good as one coordinate can
be. The fifth miss, seed 2, is unrelated: it had two coordinates and both missed
total mass (0.701 against a 0.75 gate).

Forcing `m = 2` confirms this end to end: all four capped seeds clear the `dm`
gate (0.014 -> 0.929, 0.025 -> 0.969, 0.084 -> 0.932, 0.016 -> 0.702), giving
9/10 on the frozen-NLL pick and 10/10 on the MDL pick.

**Why the three-step gets 7/10 on the same criterion.** At `d = 2` it is a
square diffeomorphism, `m = d = 2` by construction. It never chooses a rank, so
it always hands SR two coordinates however weakly the second is determined. The
one-step's extra capability — deciding the rank from held-out information — is
what costs it here, because this criterion rewards reporting an axis the data
barely supports.

## Discovered coordinates

The one-step coordinates are correct up to a monotone reparametrisation, which
the Pearson criterion penalises. The loss is invariant under reparametrising
each coordinate, so it has no reason to prefer the linear representative.

On IMRPhenomD at `m = 2`, the sum coordinate appears as a monotone function of
a near-equally weighted sum, never the bare sum (`X1 ~ m1`, `X2 ~ m2`):

    seed 4:  (0.991*X1 + X2)^2.769
    seed 9:  (X1 + 0.945*X2)^3.933
    seed 6:  exp(0.870*X1) * exp(0.886*X2)   = exp(0.870*X1 + 0.886*X2)

Weight ratios are within 6% of equal, so the direction is `m1 + m2` to high
precision. The difference coordinate is consistently opposite-sign but tilted
(magnitude ratios 1.7-2.3, not 1), which is what yields `|r(dm)| ~ 0.93-0.97`
with 0.24-0.38 of residual sum leakage rather than a clean 1.0.

On SIR the same pattern governs the score. Gradient cosine is 0.89-0.99 while
Pearson is 0.61-0.90, and Spearman lies between them on every seed — the
signature of a monotone wrapper around the right coordinate. In the
info-floor-0.5 rerun the score tracks which algebraic form SR chose:

| form | seeds | Pearson |
|---|---|---|
| ratio `X1/X2` (that is, `beta/gamma`) | 6, 8, 9 | 0.874-0.900 |
| linear `-a*X1 + b*X2` | 0, 3, 7 | 0.753-0.825 |
| `gamma` only, `~19*X2` | 1, 2, 4, 5 | 0.605-0.646 |

The four worst seeds carry `beta` and `gamma` in separate raw coordinates and
never combine them, yet their gradient cosine is still ~0.93. The map has `R0`;
SR expressed it in the raw parameter basis. Same failure mode as IMRPhenomD
seeds 1 and 4, whose second component collapses to a function of `m2` alone.

## What is not comparable

**Rosenbrock's alignment cells are different statistics.** The one-step
Rosenbrock adapter's `gates()` returns only rank and screen booleans
(`degeneracy_distillery/problems/rosenbrock.py:117`), so its
`physics_alignment` falls through the fallback chain in `oneshot_sweep.py` to
`r2_true_min` — an R^2 of the true coordinates on a cubic of the recovered ones.
The three-step reports `best_rosen_abs_corr`, a Pearson correlation
(`scripts/rosenbrock_notebook_run.py:731`). Hence 0.999 against 0.929: a
different measurement, not a better result. The one-step record also has no
`complementary_linear_alignment`, so its recovery cannot be scored on the
published conjunction at all. Both cells are marked `comparable=0` in
`metrics.csv` and `n/a` in the table.

This is fixable without a rerun. The Rosenbrock adapter uses raw
`theta ~ U[-3,3]^d` with no MinMax scaler, so the saved `expression` strings are
already in native `theta` units and could be fed directly to the published
correlation function, exactly as the SIR and GW adapters do. Not done here.

**Quartile convention.** The published table uses `numpy.percentile` (linear
interpolation) over all ten seeds. `scripts/aggregate_seed_sweep.py` instead
uses a Tukey split-halves rule over successes only, so the
`aggregate_summary.md` files copied into `records/` differ in the third decimal —
for example SIR reads `0.674 [0.633, 0.745]` there against `0.674 (0.636,0.741)`
published. IMRPhenomD differs on the denominator too: its published IQR is over
all ten seeds, not the seven successes. `make_rebuttal_tables.py` uses
`numpy.percentile` over all seeds for both arms, which reproduces every
published three-step row exactly; that equality is asserted on each build.

**Seeding.** The three-step arm uses master seeds 0-9 with a per-stage additive
stride (`seed = master + offset*10_000`). The one-step driver uses
`seed = args.seed + 7919*trial + 31*d`. Same number of independent trials,
different derivation — the datasets are not seed-matched.

**Float precision.** `degeneracy_distillery/oneshot.py` enables `jax_enable_x64`
at import; none of the notebook scripts do. The simulators therefore run in
float64 under the one-step driver and float32 under the published runs, so the
datasets are not bitwise identical even at an identical seed.

**Held-out geometry has no one-step analogue.** The rebuttal's
`heldout_geometry` block is the Frobenius distance of the transformed Fisher
from identity. The one-step pipeline never builds a Fisher, so that column
cannot be compared and is absent from `metrics.csv` for the one-step arm.

**MDL.** The `mdl_total` field is null in every one-step record. It was
previously being filled with `nll_symbolic`, a held-out NLL in nats and not a
description length; the driver computes no raw DL, so the slot is now null
rather than a wrong number. Three-step `mdl_total` is a real DL and is carried
through.

## Not ported

The three-step rebuttal also covered `qm7b`, `kolmogorov`, `kuramoto` and
`rayleigh_benard`. None has a one-step adapter, so none is in this comparison
and none of their data is copied here; they remain under
`$SCRATCH/follow_up_results/` and `$SCRATCH/rebuttal_discovery/`.

Weak lensing was in the original port scope and was dropped. It has no discovery
script, no notebook (`plots/wl_w0wa_Omega_c_sigma8_flattening.ipynb` referenced
in the notes does not exist), no dataset on this cluster, and no rebuttal
baseline to line up against.

## Open threads

1. **GW TaylorF2, two misses.** Seeds 15900 (0.690) and 55495 (0.681) sit well
   below the 0.75 gate and well below the rest of the distribution. Not
   threshold-boundary cases; worth a spot-check for a genuine training failure.
2. **The `min_gap` rule.** A hard threshold on a continuous random quantity, and
   IMRPhenomD sits on top of it — seed 0's ratio is 10.63 against a threshold of
   10. A margin-aware or information-aware pad would be principled, but it
   affects every one-step experiment.
3. **Criterion or method?** A conjunction over components rewards reporting a
   weakly determined axis. The one-step rank selection is arguably doing the
   statistically honest thing and being penalised for it.
4. **SR representation.** On both SIR and IMRPhenomD the score is set by which
   algebraic representative SR picks, not by map quality. Gradient cosine and
   Spearman are far more stable than Pearson across seeds.
