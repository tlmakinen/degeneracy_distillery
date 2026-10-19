# CAMELS one-step rebuild (ICML codebase)

Replaces the standalone `new_idea/camels_oneshot.py`, which predates the
one-step package changes and shared none of them: its own screening
statistic, its own ensemble, its own alignment, no architecture variation.

## What the old run actually measured

`r_hat = 7` came from thresholding `0.5 log lambda` of the **eta prior
covariance** at **0.10 nats**. That is prior *spread*, not predictability.
The package screens on `info_nats = 0.5 log(prior var / residual var)`
(`oneshot.py:413`), which is held-out information, and defaults to a
**0.5** floor; the heater sweep passes **1.0**. On the old spectrum
`[0.992, 0.686, 0.629, 0.576, 0.474, 0.361, 0.330 | 0.002, ...]` a 0.5
floor reads r_hat=4 and a 1.0 floor reads r_hat=0. So `m=7` was a floor
artefact on the wrong statistic and had to be redone, not re-confirmed.

The `info_probe` spectrum is computed once and the floor only thresholds
it, so **one** screening run answers the question at every floor. Do not
sweep the floor as separate jobs.

## Architecture variation (the y_std problem)

Fishnets draws per member (`training_loop_fishnets.py:140-190`):
activation from a 6-function bank, width uniform in `[hids_min, hids_max)`,
depth uniform in `[min_layers, max_layers]`.

The one-step ensemble did none of that:

- `oneshot.fit_ensemble` takes `estimator_factory` **singular**; all K
  members share an architecture *and* warm-start from the same ladder
  params including the `est` trunk. Diversity = bootstrap resample only.
- `oneshot_ensemble.fit_oneshot_ensemble` gained `estimator_factories`
  (plural), which cycles a bank and drops the `est` warm-start when the
  factory changes. Only `problems/mu_sigma.py` supplied a bank, of 4
  widths.
- `Estimator` hardcoded `nn.gelu`. Activation could not be varied at all.

Now: `Estimator.act` is a field, `EST_ACTS` holds the Fishnets bank, and
`estimator_bank(k, seed, hids, n_layers, acts)` samples width, depth and
activation per member. `scripts/oneshot_sweep.py --vary-architectures`
(default on) passes it whenever the problem offers `estimator_factories`.

Kinked activations are safe on the `eta(x)` side: the volume term
differentiates `eta_phi(theta)` only, so `eta_psi` never needs a smooth
Jacobian. Do **not** reuse this bank for the Flattener.

## Calibration

`oneshot_ensemble.ensemble_calibration` returns two z-scores.

- `z_in = (y_k - ybar)/s` is what SR weights by. Because `y_k` sits inside
  its own mean and spread, `sum_k z^2 = K-1` identically: its RMS is
  `sqrt((K-1)/K)` (0.935 at K=8), **not 1**, and the ratio to that
  reference is 1.000 by construction. It carries no information.
- `z_loo` leaves each member out, is exactly `t_{K-2}` under normality
  with reference width `sqrt((K-2)/(K-4))`, and needs `K > 4`. This is
  the real check. `ratio_loo > 1` means `y_std` is too small and SR will
  over-trust those coordinates.

Verified against synthetic calibrated ensembles at K=5,8,10: both ratios
return 1.000. The driver writes `calibration.npz` and `calibration.png`
(histogram vs `t(K-2)` plus Q-Q) per trial.

## SR in a restricted parameter basis

`--sr-top-k K` hands SR only the top-K parameters by screen score. `X`,
`dy` and `Fs` are sliced together so the Fisher stays conformable, and
`n_params` and the structure predicate follow the reduced dimension.

**The trap:** expression variables `X1..XK` then index the *kept* columns.
`rescore_frozen` compiled them against the full 35-column theta, so `X3`
in a reduced-basis expression would silently be read as parameter 3 of the
full space. It now takes `keep_idx` and composes the selection into the
coordinate function. Verified: a stack over kept indices `[2, 6, 30]`
evaluates on those columns, where the old path read `[0, 1, 2]`.

## Running it

    # stage 0 -- settle r_hat. One run covers every floor.
    MODE=screen sbatch scripts/slurm/camels_oneshot_sweep.sh

    # stage 1 -- retrain at the chosen rank, 10 varied architectures, SR.
    MODE=full ENSEMBLE_K=10 SR_TOP_K=7 \
      sbatch scripts/slurm/camels_oneshot_sweep.sh

CAMELS is a fixed 1024-row dataset: 819 train / 205 held out, disjoint.
`CamelsProblem.held_out()` serves the test split, and the driver uses it
instead of calling `sample` twice -- which on a dataset problem would have
handed back training rows and scored the map on its own fit set.

GPU only: the JAX CPU backend fails for this pipeline at d>=3.

## Stage 0 result (job 3838081, 20000 probe steps)

Held-out information per probe axis, and the eta prior spectrum beside it:

    info nats  1.285 1.318 1.185 0.982 0.802 0.554 0.470 | 0.015 0.017 -0.032 -0.076 -0.061
    lambda     19.37 15.21 12.88 9.71  6.71  5.53  2.47  | 1.54  1.39  1.25   1.14   1.08

Both break after axis 7, but the information gap is **31x** where the
lambda gap is **1.6x**. The 200-step smoke gave
`[1.222 1.370 1.152 0.972 0.760 0.590 0.464 | 0.022 ...]` -- the same
spectrum at 1/100th the budget, so the rank is not a budget artefact.

r_hat at each floor: 0.10 -> 7, 0.25 -> 7, 0.50 -> 6, 1.00 -> 3. The 0.5
default bisects a tight pair (0.554, 0.470) rather than the gap, so use
**0.25**, which sits inside the gap.

### The pad rule cancels the floor error

`refit_rank` pads by one axis whenever the **lambda** spectrum drops by
less than `min_gap=10` at the cut. CAMELS lambda decays smoothly from 19
to 1 with no 10x gap anywhere, so the pad fires unconditionally:

| floor | r_hat | lambda ratio at cut | pad | m_fit |
|-------|-------|---------------------|-----|-------|
| 0.50  | 6     | 5.53/2.47 = 2.2     | +1  | 7     |
| 0.25  | 7     | 2.47/1.54 = 1.6     | +1  | 8     |

So the default settings reach m_fit=7 by undercounting and then padding --
two errors cancelling -- and the "better" floor would have given 8. Use
`--info-floor 0.25 --refit-pad 0` so m=7 is a decision. `--refit-pad` is
now plumbed through the driver; the launcher defaults `REFIT_PAD=0`.

### Open: the probe and the refit disagree about the rank

    probe (m=12)  1.285 1.318 1.185 0.982 0.802 0.554 0.470
    refit (m=7)   1.424 1.010 0.818 0.687 0.336 | -0.013 -0.084

The production map carries *negative* information on two of its seven
axes. Reproduced identically at 200 and 20000 steps, so it is not an
optimisation accident. Either the probe is optimistic (only ~5 directions
are separable from 697 rows over 35 parameters) or the refit fails to use
its last two latents (genuine noise axes sit at 0.015, not 0.47).

Resolve it with the ensemble, not by argument: a dead axis shows up as
large member-to-member disagreement, so `ratio_loo` for axes 6-7 should
separate from axes 1-5 if they are artefacts.

## Screen ranking (no clean gap)

     1 Omega_m 0.4375   2 IMFslope 0.3637   3 Omega_b 0.3585
     4 H_0     0.3397   5 A_SN2    0.3137   6 SofteningLength 0.2756
     7 MinWindVel 0.2685  8 sigma_8 0.2675  9 WindFreeTravelDensFac 0.2646
    10 A_SN1   0.2634  11 n_s 0.2460  12 QuasarThresholdPower 0.2365

Scores decay smoothly 0.44 -> 0.16; the only sharp drop is after rank 1.
`--sr-top-k` is therefore a search-budget choice, not a discovery. k=12.

This ranking is closer to the old **Fisher** heuristic
`[Omega_m, sigma_8, Omega_b, H_0, n_s, IMFslope]` than to the old one-step
weight-norm "usage" ranking, which put all four feedback amplitudes in the
top 7. A_AGN1 and A_AGN2 are not in this top 20 at all. The group-L1 screen
trains a gated probe; the usage metric was a norm over theta-facing
weights. Trust the screen.

## Stage 1 result (job 3838094): m=7, K=10 varied architectures, SR top-12

`status=ok`, 46 min on a V100. Settings `--info-floor 0.25 --refit-pad 0
--ensemble-k 10 --vary-architectures --sr-top-k 12`, SR 300s/component.

### Rank: settled at 7

    jtj_rank_agreement_k = 10

All ten members -- five activations, widths 66-268, depths 2-4 -- give
JtJ rank 7 independently. This is the only rank test here that is
basis-independent and needs no floor, and it is unanimous.

### Calibration: works

    LOO ratio  0.973 0.982 1.013 0.960 1.004 1.031 1.127   ref t(8)=1.1547
    frac |z| > 3*ref = 1.2%

Six of seven axes within 4%. The varied-architecture ensemble produces an
honest y_std.

### SR: the open problem

    R2 per component  0.881 0.952 0.965 0.971 0.837 0.549 0.114
                      min 0.114  mean 0.753
    nll_neural -2.461   nll_symbolic 6.912   nll_gap 9.374

The symbolic stack is nowhere near the neural map.

## Three things that were wrong, and what they cost

**1. "Calibrated y_std will move the MDL pick to the sparse end of the
front."** It did not. Components 0-4 are still dense 9-10 term linear
sums. Calibration is necessary for MDL to mean anything but is not
sufficient to produce sparse expressions.

**2. "Component 6 is a genuine recovery."** Its expression is
`5.612 - 1.238*(0.565*A_SN2 + 0.988*A_SN1)` -- clean, two parameters, both
supernova amplitudes -- and its **R2 is 0.114**. It is sparse because SR
found nothing to fit. Never read recovery off the *form* of an expression
without checking the fit.

**3. "ratio_loo will identify dead axes", then "spread/signal will."**
Neither does. `ratio_loo` is scale-free: a dead axis gets a proportionally
large y_std and calibrates fine. `spread/signal` gave axis 7 the *best*
score (0.268) because members can agree tightly on a low-amplitude
direction that nothing can predict. The diagnostics that actually flagged
axis 7 were held-out information (-0.084) and SR R2 (0.114).

## Axis 7 is the weak one, on every measure that works

| measure | axis 7 |
|---------|--------|
| refit held-out info | -0.084 (negative) |
| calibration ratio_loo | 1.127 (the only miscalibrated axis) |
| signal amplitude (sd over theta) | 1.079 (vs 3.10 for axis 0) |
| SR R2 | 0.114 (worst) |

## The restriction test fired against the screen

Minimum R2 fell from **0.806** (earlier 35-param run) to **0.114** with
`--sr-top-k 12`. Dropping 23 parameters, including both AGN amplitudes,
cost an entire eta axis. Job 3838111 repeats 3838094 with the identical
seed, m, and ensemble and only `SR_TOP_K` unset, to confirm the
restriction is the cause rather than something else in that comparison.

If R2 recovers, `--sr-top-k` is unsafe here and the group-L1 screen drops
parameters that matter. If it stays at ~0.11, axis 7 is not expressible in
any parameter basis and m=6 is the honest rank despite the unanimous JtJ
rank of 7.

## Stage 2 (job 3838111): the restriction was the problem

Identical seed, m, and ensemble as 3838094; only `SR_TOP_K` unset.

                         R2 per component                      min    mean   nll_gap
    top-12  0.880 0.952 0.965 0.971 0.837 0.549 0.114         0.114  0.753   9.374
    all-35  0.973 0.981 0.977 0.976 0.935 0.923 0.961         0.923  0.961   5.638

Axis 7 recovers 0.114 -> 0.961. Every component improves.

### Why: the informative parameters are axis-specific

Parameters used by each MDL component in the all-35 run:

    [0-3]  cosmology + SN feedback + wind params -- no AGN
    [4]    + A_AGN1, A_AGN2
    [5]    + A_AGN1, A_AGN2, SeedBlackHoleMass, BH_Accretion/EddingtonFactor
    [6]    sigma_8, A_SN1/2, A_AGN1/2, UVB_*, SNIa_Rate_Norm

**AGN and black-hole physics lives entirely in the three weakest eta
axes.** The group-L1 screen ranks parameters by influence aggregated over
*all* axes, so a parameter that matters only to a low-amplitude axis
scores low and falls outside the top-12 -- while that axis still needs it.

Generalisable: **a global parameter screen is incompatible with
per-component SR when the informative parameters are axis-specific.** Do
not use `--sr-top-k` on CAMELS. The flag and the `keep_idx` basis fix in
`rescore_frozen` are still correct and worth keeping for problems where a
screen genuinely isolates a shared support.

This also settles the earlier disagreement: the AGN feedback amplitudes
**do** carry information, and "trust the screen, the AGN params are
correctly dropped" was wrong.

### The ensemble work paid off, by a different mechanism than predicted

    old standalone, seed-only ensemble, 600s SR    min 0.806  mean 0.880
    varied-architecture + calibrated, 300s SR      min 0.923  mean 0.961

Better on both at half the SR budget. The prediction was that calibrated
y_std would make the MDL pick *sparser*. It did not -- components are
still 12-13 variable expressions. It made them *fit better*. Right
intervention, wrong mechanism.

### m=7 is settled

Unanimous JtJ rank 7 across all ten architectures, and axis 7 fits at
R2=0.961 once it has the parameters it needs. The refit's -0.084 held-out
information was a basis artefact, not a dead axis.

### Still open

`nll_gap = 5.638`: the frozen symbolic stack is still well short of the
neural map (-2.461 vs 3.177). Mean R2 0.961 against eta is not the same as
reproducing the map's information content. That gap is the next target.

## Stage 3 (job 3838204): Fisher-based SR screen

`F = J^T J` was being built in `fit_oneshot_ensemble` for Procrustes
alignment and thrown away. Its diagonal is how much eta moves per unit
theta -- a parameter screen for free -- and its eigenvalue spectrum is an
independent rank check from the theta side.

Added `--sr-screen fisher|probe` (default fisher) and recorded
`fisher_sens`, `fisher_eigs`, `fisher_gap_rank`, `fisher_gap_ratio` on
every run.

### Three-way comparison, identical seed / m / ensemble

    basis          nv   R2 per component                            min    mean   nll_gap  time
    probe top-12   12   0.880 0.952 0.965 0.971 0.837 0.549 0.114   0.114  0.753   9.374   46m
    fisher top-12  12   0.931 0.968 0.967 0.947 0.871 0.878 0.890   0.871  0.922   5.457   47m
    all 35         35   0.973 0.981 0.977 0.976 0.935 0.923 0.961   0.923  0.961   5.638   64m

The ranking was the problem: same k, component 6 goes 0.114 -> 0.890.

But 12 is still too few -- all-35 wins on R2 (mean 0.961 vs 0.922), and SR
used a union of 28 of 35 parameters when given the choice. `nll_gap` is a
wash (5.457 vs 5.638, marginally favouring the restriction), and the
restriction saves 27% runtime.

**Use all 35 when affordable; `--sr-screen fisher --sr-top-k k` when not;
never the probe screen for this.**

### Why the probe screen fails here

The two screens rank theta differently (Spearman 0.68, top-12 overlap
8/12):

    rank  parameter    sqrt(F_ii)   probe rank
      7   A_AGN1          4.18          31
      8   A_AGN2          3.63          26

In correlation form `C_ij = F_ij / sqrt(F_ii F_jj)`, **A_AGN1 is the most
decoupled parameter in the top-12** (mean |C| = 0.064, vs 0.382 for
WindFreeTravelDensFac). The AGN amplitudes point almost orthogonally to
everything else, so they *are* their own eta axis and nothing can stand in
for them. The probe aggregates influence over all eta directions, so a
parameter that drives only a weak axis scores low -- exactly backwards for
picking an SR basis.

Strongest learned degeneracies: sigma_8 x n_s +0.964, A_SN1 x SNII_MinMass
-0.947, A_SN2 x WindFreeTravelDensFac -0.901.

### The Fisher also confirms rank 7 from the theta side

    eigs  165.4 118.9 94.5 68.7 38.8 21.2 18.5 | 0.598 0.378 ...   31x gap

Third independent route to 7. Note a *relative threshold* on the same
matrix reads full rank 35 -- the same pathology as the three-step Fisher.
Read the gap, not a floor.

### Fisher stability over the prior

`sqrt(var)/|mean|` on the diagonal is 0.098-0.228, uniform across all
twelve, so var(F) largely tracks mean(F)^2 and the Fisher is stable across
the prior volume. A single prior-averaged Fisher is a fair summary here --
not something to assume in general. Figure:
`scripts/plot_camels_fisher.py`, written to `<run>/fisher_top12.png`.

## Stage 4: the SR augmentation was leaking theta (my bug)

`CamelsProblem.sample_prior` originally resampled the 819 training rows
with replacement. 20000 "augmented" draws were therefore **819 unique
theta repeated ~24 times**, which

* adds **no** parameter-space coverage -- it only reweights existing
  points, defeating the purpose of augmentation, which is to sample the
  *fitted map* densely where the 1024 simulations do not reach; and
* leaks theta across the SR split. `fit_and_analyze_sr` shuffles rows, not
  theta, so **87.9% of unique held-out theta also appeared in SR
  training** (819 of 932).

All R2 values reported before this point are therefore substantially
in-sample at the theta level. The *comparisons* stand (identical flaw,
split and seed across probe-12 / fisher-12 / all-35, so 0.114 -> 0.890 is
real), but the absolute numbers are optimistic.

**Unaffected:** m=7, the calibration, the Fisher spectrum and screen.
`sample_prior` feeds only SR augmentation; the ladder and ensemble draw
from `problem.sample()`. Verified: the ladder reproduces byte-identically
after the fix.

### Correction: the old standalone did NOT have this bug

`new_idea/camels_oneshot.py:340` draws
`rng_sr.uniform(th_tr.min(0), th_tr.max(0), size=(sr_grid, d))` -- uniform
on the training box, with `y_std` from the aligned ensemble spread. It was
correct all along; the flaw was introduced in the port.

So the earlier claim that varied architectures improved SR from **0.880 ->
0.961** compared a clean number against a leaked one and is **not
supported**. It needs the reruns. (The old run also used a 3000-point grid
vs 20000 here, so even then the comparison is not exact.)

### Where to draw theta is a design choice, not a prior

SR approximates the already-fitted map eta(theta), so the draw just picks
where the expression should be accurate. Uniform on the scaled box is the
default. `prior_draw="empirical"` draws each parameter independently by
inverse-CDF of its training quantiles; the LH design is near-independent
(max |rho| 0.062) but 19 of 35 marginals are non-uniform on the scaled box
(KS to 0.57) because CAMELS samples several parameters log-uniformly
upstream, so uniform puts a median 11% of draws per coordinate (31% for
QuasarThreshold) where training density is low -- still inside the
support, just sparser. Both modes verified: 20000 unique rows, zero
overlap with training.

### The whole ensemble sees the new draws

`augment_aligned` evaluates **all K members** at `X_sr`, maps them into the
aligned frame with the stored rotations and member means, and takes the
weighted spread:

    etas = stack([f.eta(X_sr) for f in fits])     # all K
    jacs = stack([f.jac(X_sr) for f in fits])     # all K
    ys, js = apply_ensemble_alignment(etas, jacs, aligned)
    y, y_std = average(ys, w), weighted_std(ys, w, axis=0)

So `y_std` at the augmented points is genuine member disagreement, which
is what MDL weights by -- not a single member's output.

### Jobs 3838809-12: 2x2 over draw and basis

    3838809  uniform    all 35
    3838810  uniform    fisher top-12
    3838811  empirical  all 35
    3838812  empirical  fisher top-12

Same seed, so identical map / ensemble / alignment / eta. Rows compare the
draw at fixed basis; columns compare the basis at fixed draw.

## Stage 4 result: the 2x2 (jobs 3838809-12), all leak-free

    run               nv   R2 per component                            min    mean   nll_gap
    uniform x 35      35   0.968 0.940 0.974 0.964 0.900 0.930 0.962   0.900  0.948   5.846
    uniform x f12     12   0.929 0.961 0.961 0.940 0.839 0.888 0.897   0.839  0.917   5.954
    empirical x 35    35   0.973 0.980 0.977 0.975 0.934 0.930 0.961   0.930  0.962   5.203
    empirical x f12   12   0.932 0.968 0.966 0.946 0.862 0.889 0.882   0.862  0.920   5.082

Both effects replicate across the other factor:

    draw (empirical - uniform)   d(min)   d(mean)   d(nll_gap)
      at 35 vars                 +0.030   +0.013     -0.643
      at 12 vars                 +0.023   +0.004     -0.872
    basis (top-12 - all-35)
      at uniform                 -0.061   -0.032     +0.108
      at empirical               -0.069   -0.041     -0.121

**Use `prior_draw="empirical"` with the full 35-parameter basis.** min
0.930, mean 0.962, nll_gap 5.203.

The empirical draw wins on *both* metrics. The prediction was that it would
help nll_gap only -- nll_gap refits eta_psi against the real 819
simulations, which live in the dense region -- and that R2 could not
distinguish the draws since it is scored on the SR split, which follows
whatever draw was chosen. That was wrong: R2 improves too, in both bases.
Concentrating draws where the design lives is simply a smaller effective
volume for a fixed expression budget to approximate.

Restricting to 12 costs 0.03-0.04 mean R2 and 0.06-0.07 on the worst
component, replicated on both draws. nll_gap is a wash there (+0.11/-0.12,
opposite signs). Restrict only for the 27% speedup.

### Leakage mattered less than feared

Clean vs the earlier leaked runs: mean fell 0.961 -> 0.948 (all-35,
uniform) and 0.922 -> 0.917 (f12, uniform); worst component fell ~0.02-0.03.
A 25-30 token expression with ~12 free coefficients fitted against 10000
rows has almost no capacity to memorise individual theta, so theta-level
leakage barely moves the score. The real cost of the resampling bug was
**coverage**, not leakage: 819 unique theta carry far less information
about the map than 20000.

### The ensemble vs the old pipeline, finally on clean footing

    old standalone (clean, uniform, 3000-pt grid, 600s)   min 0.806  mean 0.880
    new, empirical x 35 (clean, 20000-pt, 300s)           min 0.930  mean 0.962

Not controlled -- different grid size and SR budget -- but the direction is
unambiguous and both numbers are now honest.

### The Fisher screen is invariant to the draw

The same top-12 was selected under three different theta distributions
(resampled, uniform, empirical), even though the screen averages F over the
augmented points. The ranking is a property of the map, not the sampling.
