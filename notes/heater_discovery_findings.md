# Heater Discovery Rerun: Measured Findings

Results log for the rerun planned in `notes/heater_discovery_scaling.md`. That file
is the design; this one records what was actually measured, including the places
where the plan's expectations did not survive contact with data.

Script: `scripts/heater_discovery_dim_scaling_sweep.py`.
Wrapper: `scripts/slurm/heater_discovery_scaling_gpu.sh` (new file;
`slurm_gpu_sweep.sh`'s `heater` target is untouched).
Aggregation: `scripts/aggregate_heater_discovery.py`.
Status as of 2026-07-28 12:00.

---

## 1. Headline: the paper's framing is wrong; the gap is the defensible claim

Common analytic axis, Phase 3, 10 trials per d (d=8 is n=6, still filling):

```
 d |  raw            |  analytic (oracle) |  gap = analytic - raw (paired)
 2 | +0.659 +- 0.018 | +0.628 +- 0.020    | -0.031 +- 0.017
 3 | +0.963 +- 0.016 | +0.955 +- 0.030    | -0.008 +- 0.031
 4 | +0.983 +- 0.022 | +1.143 +- 0.022    | +0.160 +- 0.015
 6 | +0.767 +- 0.031 | +1.582 +- 0.060    | +0.815 +- 0.057
 8 | +0.658 +- 0.073 | +1.241 +- 0.021    | +0.583 +- 0.094   (n=6, thin)
```

The raw arm is **non-monotone**: it *improves* to d=4, then falls (d=4->6 is
4.8 sigma). The oracle arm *rises*. So neither "raw degrades from d=2" nor "the
coordinate arm stays flat" describes this experiment.

Mechanism: two competing effects. On `[1,2]`, `E[log theta] = 0.386 > 0`, so
`P` grows like `e^{0.386 d}` and SNR *improves* with d -- this helps both arms.
Only the raw arm additionally pays the cost of an `R^d` target. Crossover near
d=4.

**Never quote a single slope for the raw arm.** Fitting one line across the rise
and the fall averages to ~+0.02 and prints "flat", which is how this was
initially misread. `aggregate_heater_discovery.py` now detects non-monotonicity
and refuses to summarise it; it reports a tail slope from the peak instead
(-0.092 +- 0.015 nats/dim from d=4).

Quote the **gap**: monotone, 10 sigma at d=4, 14 sigma at d=6. It cancels the SNR
trend common to both arms and isolates the dimensionality cost only raw pays.

Also note the prior box `[1,2]^d` has width exactly 1, so prior volume = 1 and
`log p_prior = 0` at every d. That makes the native `raw_log_prob` an information
gain in nats, readable *down a column* (2.0 -> 0.8 over d=2..8). It is still not
comparable *across arms*. A width-2 prior like `[1,3]` would inject `-d log 2`
and break even that.

## 2. The discovered arm is NOT yet demonstrated

Pooled d>=4, conditional on recovery:

```
recovered (n=3) : discovered - raw = +0.030 +- 0.061   (oracle on same runs: +0.352)
failed    (n=19): discovered - raw = +0.013 +- 0.022   (oracle on same runs: +0.519)
```

Under 10% of the available gain, indistinguishable from zero, and slightly
*negative* per-d at d=2,3,4. The oracle-free `alignment >= 0.99` gate does not
rescue it either, so this is not a selection artifact.

The honest statement: **n = 2-3 recovered runs at d>=4 is too few to measure
conversion at all.** Do not quote the 0.06 captured fraction as a property of the
method.

One known bias to remove first: `analytic_eta` is the *standardised log*-product,
near-Gaussian at every d (skew -0.06 to -0.17). SR frequently returns a
coordinate affine in `P` itself (`pearson_abs_product = 1.0000` on several d=2
runs), which is increasingly skewed and heavy-tailed with d (skew +0.40 at d=2,
+1.24 at d=6, +2.23 at d=12). Both are valid monotone representatives carrying
identical information, but the MDN and the Gaussian-matched evaluation both
prefer the former. **The comparison is currently tilted against the discovered
arm.** Fix: a monotone (rank-Gaussianising) warp fit on prior draws, replacing
the affine standardiser. Information-preserving, so free. NOT YET IMPLEMENTED.

## 3. Root cause of poor recovery: fishnet epochs

`--fishnet-epochs` defaulted to **300**; `heater_minimal_distillery.py`, the
reference that works at d=2, uses **5000**. The `train_fishnets` and
`fit_flattening` call sites are otherwise identical between the two scripts.

At 300 the logs end at "Epoch 299" and `patience=30` never fires -- training is
truncated at ~1/3 of convergence. At 5000 early stopping fires around epoch
700-3000, so the real cost is ~2.5x, not 16x. This produced the first exact
recoveries (`|rho| = 1.0000`, level-set `U ~ 1e-31`), never seen at 300.

Do **not** adopt the reference's other two settings: `noise=1e-2` +
`squared_frob_det` blew up with `All ensemble members have non-finite
eta_ensemble`. Keep `squared_frob` / `1e-3`.

## 4. Spearman is a broken recovery criterion here; use the level-set test

On `[1,2]`, `log` is near-affine, so an additive coordinate `sum X_i` -- no
multiplicative structure at all -- scores `|rho| = 0.996` at every d. The
pre-registered 0.99 gate therefore *passes non-recoveries*, and relaxing it to
0.9 makes it worse. Observed repeatedly: runs at rho = 0.99-0.996 with level-set
scores identical to the additive surrogate.

Replacement, implemented: `level_set_unexplained` + `product_preserving_partner`.
Since the data depend on theta only through `P`, any correct coordinate is
constant on level sets of `P`. Report

    U = E_P[Var(eta | P)] / Var(eta)        in [0, 1]

estimated from matched pairs built by `(t_i, t_j) -> (c t_i, t_j / c)` (exact,
in-box, works at any d). `U = 0` iff the coordinate is a function of `P` alone.
Shift/scale invariant, so SR's arbitrary units need no standardisation.
`levelset_recovered` is baseline-relative -- it must beat `sum X_i` measured at
the same d and prior -- so there is no threshold to tune and the `1/d` trend in
`U` cancels. Validated: analytic and `prod t` give ~1e-32; the in-pipeline
analytic reference reads ~5e-32 every run.

## 5. Restart selection works, and the alignment score is a trust gate

The flattening optimum is seed-dependent. `--flatten-restarts K` runs K
flattening fits against **one** fishnet ensemble and keeps the attempt whose
retained axes have the highest Fisher-alignment score. Oracle-free -- it reads
only the Fisher and the Jacobian. d=2 recovery went 1/6 -> 7/10 (now 8/10).

A retry is *only* flatten + align + rank (~109 s). Fishnets (~298 s) run once
before, SR/pyoperon (~302 s) and the NPE arms run once after, on the winner. So
K=15 costs ~2.4x a run, not 5x.

The score is also a **trust gate** -- arguably the more useful result, since it
says when the pipeline's own coordinate is trustworthy without any oracle:

```
alignment score < 0.90    : 0/6  recovered
              0.90-0.99   : 0/8  recovered
              0.99-0.999  : 5/11 recovered
              >= 0.999    : 7/8  recovered
```

Below 0.99: **0/14**. Bootstrap on 1000 real samples gives the score to
+-0.003, so a low score is a real measurement, not a noisy one -- augmenting the
scoring stage would not change any decision.

## 6. Why recovery collapses with d: two compounding factors

Recovery = P(some restart clears 0.99) x P(recovery | cleared). Both fall:

```
 d | per-restart P(score>=0.99) | P(>=1 of 3) pred/obs | recovery GIVEN cleared | K for 90%
 2 |   0.533 (16/30)            |   0.90 / 0.90        |  8/9                   |  4
 3 |   0.259  (7/27)            |   0.59 / 0.67        |  2/6                   |  8
 4 |   0.200  (6/30)            |   0.49 / 0.50        |  2/5                   | 11
 6 |   0.074  (2/27)            |   0.21 / 0.22        |  1/2                   | 30
```

The product reproduces observed recovery almost exactly (d=2 -> 0.80, saw 7/10;
d=6 -> 0.11, saw 1/10). **K=3 is ~10x under-provisioned at d=6.** Restarts do
explore independently (within-run variance 0.034 vs between-run 0.046; e.g. d=6
run 7 scored `[0.030, 0.999, 0.252]`), so more restarts genuinely buy chances.
The second factor -- a good axis no longer sufficing -- restarts cannot fix.

## 7. The Fisher noise plateau is an ARCHITECTURAL floor, not estimation noise

The design notes describe the ~1e-2 relative plateau as "fishnet estimation
noise". It is not. `fishnets.py:161`, inside
`construct_fisher_matrix_log_cholesky`:

    diag_elements = jax.nn.softplus(log_diag) + 1e-4

`F = L L^T` with a softplus-positive diagonal and a hard `+1e-4` offset, so the
network **cannot represent a singular Fisher** regardless of data. Measured, the
plateau is invariant to everything we varied:

```
d=6, plateau (median relative tail), 10 fishnets unless noted:
  baseline                  7.01e-03 / 7.10e-03   (two runs, reproducible)
  PCA to 5 comps            7.71e-03      <- slightly WORSE
  nsims 2000                7.69e-03
  nsims 4000                7.55e-03
  nsims 8000                7.30e-03      <- 8x data, no reduction
d=4:  baseline 7.45e-03,  PCA k=5 8.58e-03
d=8:  20 fishnets 8.17e-03               <- 2x ensemble, no reduction
```

So PCA on the data (tested at d=4 and d=6), **8x** the training simulations, and
2x the ensemble all fail to lower it: the plateau sits at 7.0-7.7e-03 across the
whole `nsims` scan with no trend. The `1/sqrt(N)` prediction (~2.5e-03 at
nsims=8000) is dead. Data-space PCA is well motivated -- PC1 explains 0.99973 of
variance and aligns with the thermal kernel to `|cos| = 0.999999`, since
`y = P k + noise` is exactly rank-1 in data space -- but the fishnets' embedding
net already handles that, and the floor is in theta-space parameterisation.

Two consequences:

**It justifies the eigengap rule.** A fixed absolute threshold would be
arbitrary; cutting at the largest multiplicative gap is exactly right for a
spectrum that is "one real eigenvalue + a hard floor". It also explains why rank
detection is so stable -- **42/42 correct across d=2..8 and four `--flatten-noise`
levels 1e-4..1e-1**, gaps of 100-126.

**It gives a mechanism for flattening degrading with d.** The flow is handed a
Fisher whose `d-1` null directions each carry a spurious ~7e-3, so spurious
energy grows linearly in d: 0.7% of the trace at d=2, 3.4% at d=6, 7.2% at d=12.
The flow must flatten `F` toward the identity while `d-1` floor-level directions
compete with the single informative one. That matches the measurements exactly:
rank detection unaffected (the gap stays huge), per-restart flattening success
collapsing 0.53 -> 0.074.

Implication: the remaining lever is on the **flattening** side (loss /
regulariser handling of floor-level directions), not more data, more ensemble
members, or PCA -- all three now ruled out empirically.

If a caveat about PCA is wanted: keep `k >= 4` if it is ever used, because `k=1`
compresses the data to a scalar and makes `F` rank-1 *by construction*, turning
the intrinsic-dimension result into an artifact of preprocessing.

## 8. Capacity probe (20 fishnets, 512-wide flattener, K=6) at d=8

```
d=8            | median alignment | spearman recovered | level-set recovered
standard       |      0.446       |       0/3          |      0/3
capacity probe |      0.981       |       2/5          |      0/5
```

Alignment roughly doubles and Spearman recovery leaves zero; level-set recovery
is still 0/5 (median `U` 0.0045 vs 0.0013 linear baseline). Since section 7 shows
the 20-member ensemble does *not* improve the Fisher, the gain must come from the
**wider flattener and/or the extra restarts**, not from a better Fisher.

## 9. Blocking bug found: exponential visualisation grid at d >= 10

`training_loop_flatten.fit_flattening` ends with a coordinate-visualisation block
(~lines 1448-1473) that builds a `num_pts ** n_params` grid and evaluates the
model on it. The `do_plot` guard covers only the *plotting*, so with
`do_plot=False` the grid and `jax.vmap` still run and the result is discarded:

```
d=8  -> 5^8  =     390,625   works (pure waste)
d=10 -> 5^10 =   9,765,625   JaxRuntimeError INTERNAL: Autotuning failed
d=12 -> 5^12 = 244,140,625   JaxRuntimeError RESOURCE_EXHAUSTED
```

So **every d>=10 run has died in the flattening stage for reasons unrelated to
the science** -- Phase 3 d=10 and d=12 are 0/10 and 0/10, and the enhanced-config
d=10/12 jobs are 0/4 each. The block sits after `np.savez`, so completed d<=8
results are valid, just slower than necessary.

Worked around locally (the shared module is deliberately not patched) by
`capped_visualisation_grid`, a scoped context manager that caps `jnp.meshgrid`
during the `fit_flattening` call. `jnp.meshgrid` is used nowhere else in that
module, so the patch is targeted; verified 244M rows -> 25 and restored on exit.
**Jobs launched before this fix will keep failing at d>=10 and must be
resubmitted.**

## 10. Environment traps

See `[[project-cluster-backend-pitfalls]]` in memory. Two items:

* The `degen` venv's torch is `2.11.0+cu128`, arch list `sm_75..sm_120`. The V100
  nodes are `sm_70`, so `torch.cuda.is_available()` returns True but any kernel
  launch dies. Only the NPE stage uses torch, so the wrapper probes the arch list
  and passes `--device cpu` on a mismatch, leaving the GPU to JAX. Verified ok on
  h13, and the V100 run was *faster* end-to-end than the H100 one (388s vs 448s).
* The JAX **CPU** backend fails for this pipeline at d>=3
  (`Failed to materialize symbols`). `--rank-only` therefore needs a GPU above
  d=2, contrary to the design note's claim that it runs in a minimal environment.
* If launching bare `sbatch --wrap`, remember
  `XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH`; without it JAX dies with
  `libdevice not found`.

## 10b. Can the flattener be replaced by the Fisher's own eigenstructure? No.

Worth recording because it is the obvious objection to the whole pipeline: if rank
detection already tells us the problem is 1-D, why train a flattening network?
Two shortcuts were tried and both fail, for instructive reasons.

**The learned Fisher's informative DIRECTION is constant in theta.** Measured
spread (mean |cos| of each per-sample top eigenvector against the mean direction;
1.0 = perfectly constant) is 0.9998-1.0000 in every run examined, against a true
`grad log P` spread of 0.9855 (`[1,2]`, d=4) down to 0.8933 (`[0.3,3]`, d=4) --
i.e. the learned field captures **0.3%** of the true rotation. Consistent with
the architecture: `model.apply(w, x)` is called on the **data only** (theta enters
solely via the loss residual `res = theta - mle`), and `y = P k + noise` carries
information about `P` but not about individual `theta_i`. Averaging the true
direction `(P/theta_i)` over a level set of `P` gives `~(1,...,1)`.

This is NOT a fundamental barrier -- runs achieving *exact* recovery
(`U ~ 1e-31`, `|rho| = 1.0000`) have equally constant direction fields. The
product information lives in the **eigenvalue magnitude**: analytically
`lambda_max = (|k|^2/sigma^2) P^2 sum(1/theta_i^2)`, and measured
`spearman(lambda_max, P)` is 0.989-0.997.

**Shortcut 1: integrate the eigenvector field.** With `g = sqrt(lambda) v` and `v`
constant, the field points always along `v` with varying magnitude, which is a
gradient field only if `lambda` depended on `v . theta` alone. It does not, so the
field is non-integrable and a least-squares potential returns the linear
compromise. Measured (kNN graph, MST sign propagation, sparse least squares):
`U = 0.0097` at d=4 against the additive surrogate's `0.0097`, and `0.0123` vs
`0.0122` at d=6 -- identical to three digits. On the wide `[0.3,3]` prior it does
separate at d=2 (`U` 0.017-0.025 vs additive 0.040-0.046, ~2x better) but ties
again by d=4.

**Shortcut 2: use `lambda_max` directly as the coordinate.** Fails by
construction, and gets worse with d, because `sum(1/theta_i^2)` varies freely at
fixed `P`:

```
 d | U(lambda_max) | U(logP) floor | U(sum theta) | spearman(lambda_max, P)
 2 |    0.0210     |    0.0013     |    0.0063    |   0.9854
 4 |    0.0127     |    0.0031     |    0.0099    |   0.9981
 6 |    0.0282     |    0.0038     |    0.0110    |   0.9993
 8 |    0.0579     |    0.0042     |    0.0115    |   0.9996
12 |    0.1554     |    0.0046     |    0.0122    |   0.9999
```

`lambda_max` is worse than the additive surrogate at every d. Measured from saved
fishnets: U = 0.0129/0.0131 at d=2, 0.0201 at d=6, all above the sum-theta
baseline. **Also the single best illustration of why the level-set metric was
needed**: at d=12 `lambda_max` is 0.9999 rank-correlated with `P` while being the
most contaminated coordinate measured anywhere in this project.

Conclusion: rank detection gives the *dimension*, `lambda_max` gives a scalar
0.9999-correlated with the truth, and **neither gives a coordinate that is a
function of `P`**. That requires solving the nonlinear `J^-T F J^-1 ~ I` problem,
which is exactly what the flattening does. The flattening stage is justified by
this failed attempt to remove it -- worth stating in the rebuttal, since "why not
just use the Fisher" is the natural reviewer question.

Not implemented as a full NPE arm: the level-set numbers above settle it
analytically and from saved Fishers, so an NPE arm would add a table row without
new information.

## 10c. Why `[1,2]` is a near-worst-case prior for this problem

The additive surrogate is hard to beat here because the true gradient field is
itself almost constant. `cos((1,...,1), grad log P)`:

```
 prior      d=4     d=6     d=8     d=12
 [1,2]     0.9856  0.9838  0.9830  0.9820
 [1,3]     0.9662  0.9611  0.9587  0.9562
 [0.3,3]   0.8945  0.8704  0.8562  0.8408
```

On `[1,2]` the true field deviates from constant by only ~1.6%, and the learned
Fisher's directional error is ~1.6% (measured cos 0.9854/0.9837/0.9835 at
d=4/6/8 -- a margin of **+-0.0002** against the constant-field baseline). So
distinguishing `log P` from `sum theta` requires resolving an effect the same size
as the estimation error. `[0.3,3]` gives 8x more room. This is the unifying
explanation for the Spearman saturation, the flattener's difficulty, and both
failed shortcuts above.

## 11. Dead ends (do not redo)

* **Relaxing the Spearman threshold to 0.9** -- below what a purely additive
  coordinate scores (0.996). Makes the criterion weaker, not better calibrated.
* **Prior widening.** `[1,3]` is indistinguishable from `[1,2]` (2/6 vs 2/6 at
  d=2, 0/6 at d=4 and d=8). `[0.3,3]` with `sigma=0.02` and the good settings gave
  1/4 at d=2 against 7/10 for `[1,2]`: widening raises the additive-surrogate bar
  as intended but makes flattening harder, net worse. `[0.1,2]` was rejected on
  analysis -- identifiability scales with the ratio `hi/lo`, but SNR scales with
  the box's geometric mean, and `U[0.1,2]` has geometric mean < 1 so `P` decays
  with d and the weakest 5% of draws at d=12 sit ~23x below the noise.
  Presets `wideprior_*` remain in the wrapper; do not rerun without a reason.
* **PCA on the data before fishnets** -- section 7.
* **More simulations to lower the Fisher plateau** -- section 7.
* **More fishnet ensemble members to lower the Fisher plateau** -- section 7.
* **Augmenting the axis-scoring stage.** Bootstrap gives the alignment score to
  +-0.003 on 1000 real samples; more draws change no decision. (SR augmentation
  itself *is* on and scales with d, `n_sr_samples = 1000 d`.)

## 12. Pipeline confirmed as intended

fishnets -> flatten -> Procrustes align -> select **informative** (not merely
non-constant) etas -> NPE at `d_eta = r < d`. Verified: `n_eta = 1` and
`retained_rank = 1` in every completed run. In a recovered d=4 run the kept axis
has ~10x the Jacobian energy, ~5x the spread, and `|rho| = 0.990` against the
truth, while the discarded three sit at ~0.1. In a *failed* run the least
constant axis (largest std, 43x the Jacobian energy) is *not* the informative
one -- so a non-constancy or nonlinearity-energy criterion would have picked it,
vindicating the design note's insistence on selecting via the Fisher.

Note SR fits the **post-Procrustes** coordinates. `keep_axes` indexes the aligned
frame, not `flattened.npz["eta"]`; comparing against the latter is misleading.
The aligned coordinates live in `sr_results/split_data.npz`.

## 13. What is solid vs not

**Solid**
* Rank rule: 42/42 correct, d=2..8, `--flatten-noise` 1e-4..1e-1, gaps 100-126.
* Oracle gap growing from d=4: +0.160 (10 sigma), +0.815 (14 sigma).
* Raw arm's post-peak decline: -0.092 +- 0.015 nats/dim from d=4.
* Fisher plateau is an architectural floor, invariant to data/ensemble/PCA.
* Alignment score as a trust gate: 0/14 recovered below 0.99.

**Thin**
* d=8 (n=6 of 10).

**Missing**
* d=10, d=12 entirely -- all runs hit the section-9 bug. Now unblocked.

**Not demonstrated**
* The discovered coordinate converting the oracle's advantage at any d.

## 14. Next steps, in priority order

1. Resubmit d=8/10/12 on the fixed code with the capacity config -- fills the
   thin/missing rows and tests recovery where it matters.
2. Implement the monotone (rank-Gaussianising) warp for the discovered
   coordinate, removing the section-2 bias.
3. Adaptive restarts: keep restarting until `score >= 0.999` or `K_max`, instead
   of a fixed K that overspends at d=2 and underspends 10x at d=6.
4. Only then, the flattening-side loss/regulariser question raised by section 7.
5. Still outstanding from the design note, never started: validating the rank
   rule on SIR, Rosenbrock, and GW.
