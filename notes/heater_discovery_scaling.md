# Heater Rerun: Discovery at Every Dimensionality

Plan for replacing the oracle-coordinate scaling sweep with one where the coordinate is discovered independently at each ambient dimension.

New script: `scripts/heater_discovery_dim_scaling_sweep.py`. Nothing else is modified; `scripts/heater_dim_scaling_sweep.py` and `scripts/heater_minimal_distillery.py` are left exactly as they are so the submitted numbers remain reproducible.

## What the current experiment actually shows

Two separate things, which the paper presented as one.

`scripts/heater_minimal_distillery.py` runs the full pipeline at d=2 and discovers a monotone function of `theta_1 * theta_2`. That is a genuine discovery result, at one dimension.

`scripts/heater_dim_scaling_sweep.py` sweeps d=2..12 and shows raw-theta NPE degrading while the 1-D distilled target stays flat. The distilled target there is supplied analytically, defaulting to `standardised_log_product`. The script says so itself at lines 61-66:

> Note: we use an *analytic* distilled coordinate rather than re-running the Distillery for every d.

So reviewer QfCk's Q2 is correct: the scaling claim is conditional on already knowing the answer. Answer this plainly in the rebuttal. The honest sentence is that the scaling sweep used an analytic coordinate, that discovery was demonstrated separately at d=2, and that the two were never composed.

## What the rerun adds

Discovery is run independently at every d, so the reported curve is end-to-end rather than oracle-conditioned. This is worth doing beyond just answering QfCk, because it is also the only evidence that would answer FoSE's objection that symbolic regression scales poorly in the number of input variables. Right now that objection is unanswered above d=2.

Three NPE arms per run:

1. `raw` -- target is theta in R^d.
2. `analytic` -- target is the 1-D `standardised_log_product`. This is the oracle ceiling. Keep it. Dropping it loses the ability to say what fraction of the achievable gain discovery captures.
3. `discovered` -- target is the symbolic coordinate the pipeline found at this d, standardised.

## Evaluation axes

Log-probabilities on different coordinates are not comparable, and the discovered coordinate differs across seeds and across d, so raw per-arm log-probs cannot be pooled. Evaluate on a common axis, and only where the push-forward is exact:

- Analytic axis: `raw` (push theta samples through the analytic projector) against `analytic` (identity). Reproduces the original comparison.
- Discovered axis: `raw` (push theta samples through the discovered symbolic expression) against `discovered` (identity). This is the new headline, and it uses no oracle knowledge.

The two axes are linked by reporting the rank correlation between the discovered coordinate and the analytic product axis on held-out draws.

Using the known coordinate to *score* while discovering the coordinate you *train* on is legitimate, and drawing that line explicitly is what answers QfCk. Say it in the rebuttal in those terms.

## Architecture: MDN everywhere

All three arms use an MDN. The existing script defaults the raw arm to a MAF and only the distilled arm to an MDN, which confounds the comparison the moment the discovered target is also low-dimensional: an autoregressive flow on a 1-D target degenerates to a stack of conditional scalar bijectors with nothing to autoregress over, and mis-localises sharp posteriors. Holding the density estimator fixed across arms means any remaining gap is attributable to the coordinate rather than to the architecture, which is a cleaner claim and removes an obvious reviewer objection.

## Stage settings

### Flattening (rank-deficient regime)

- `--no-invertibility-mlp` is mandatory, not optional. The default forward/backward penalty `mean((theta - theta_rec)^2)` pulls `J` toward the identity, which is the opposite of what a problem with `d-1` unidentifiable directions needs.
- `--loss-type squared_frob` as primary. `scripts/heater_minimal_distillery.py` is internally inconsistent here: line 29 suggests `squared_frob_det`, lines 46-47 call `squared_frob` cleaner for rank-1 because it drops the inverse term. Since the same docstring states the ideal is `|det J| -> 0`, a `(log det Q)^2` barrier pulling `det Q` toward 1 works against the structure. Keep `squared_frob_det` with small `beta_det` as the fallback if the flow collapses outright.
- Do not use `det F(eta) ~ 1` as a health check, and do not read the `log_frob` plateau near 0.7 as a training failure. Both are structural for rank-1.

### Fisher regulariser noise

`--noise` is the Cholesky noise added to `F` per sample, and it is doing two jobs at once: it makes the singular `F` invertible, and it sets the floor that the `d-1` null eigenvalues sit at. That makes it the yardstick for the rank rule, so resist raising it for stability -- doing so compresses the very gap being measured. Keep it as low as conditioning allows.

Note that raising the *simulator* noise `--sigma` does nothing for this. With `y = P*kernel + noise` we have `dmu/dtheta_i = (P/theta_i)*kernel`, hence

    F = (||kernel||^2 / sigma^2) * outer(P/theta, P/theta)

which is exactly rank 1 for any sigma. Larger observation noise rescales `F` uniformly and leaves the rank and the condition number untouched. The degeneracy is structural, not noise-induced.

Make the regulariser level an explicit ablation rather than a fixed choice: `--rank-only` runs the cheap prefix of the pipeline (simulate, fishnets, flatten, align, rank) with no SR and no NPE, so sweeping `--flatten-noise` over roughly 1e-4 to 1e-1 is affordable. Showing that the retained rank is invariant across that range turns the threshold from an arbitrary number into a robustness result, which is what stops a reviewer attacking it.

### Alignment

`align_mode="procrustes"` with `canonicalize="permute_and_sign"` and `separate_nonlinearity=True`.

Do not select axes by nonlinearity energy. The `d-1` junk axes have Jacobians driven by the regulariser rather than by the data, and nothing prevents that noise from carrying more apparent nonlinearity energy than the true product axis once d is large. Identifiability is a statement about the Fisher, so select on the Fisher.

Also do not assume a fixed axis position. `nonlinearity_rotation` orders by descending nonlinearity energy, `fisher_order_canonicalize(mode="permute_and_sign")` puts the largest Fisher eigenvalue *last*, and `mean_fisher_eigen` defaults to `ascending=True` so column 0 is the most degenerate direction. Assuming "axis 0 is the informative one" would silently fit SR to pure regulariser noise. The script therefore scores axes explicitly instead of relying on any ordering convention.

### Symbolic regression

Augmentation is on, via the built-in path in `fit_and_analyze_sr` (`flatten_model`, `ensemble_w`, `rotmats`, `ensemble_weights` must all be passed together or it raises). Two properties worth recording in the rebuttal: `X_sr` is drawn uniform in the bounding box of `X`, which coincides exactly with the prior here because the prior is a uniform box, so there is no covariate shift; and the fit uses augmented points while every reported geometric number is evaluated on the original simulations, because the Frobenius loss needs matched Fishers (`sr_utils.py` line 2186).

Two things must scale with d:

- `n_sr_samples`. A fixed augmentation pool gets exponentially sparser as d grows. The script defaults to `--sr-n-aug-per-dim * d`.
- `max_length`. A d-way product needs roughly `2d-1` tokens, so the default `max_length=25` makes a 12-way product literally unrepresentable and would guarantee failure for reasons that have nothing to do with SR capability. The script scales it as `max(25, 4d + 8)`.

The product form is more compact than the sum-of-logs form (`d-1` multiplications against `d` logs plus `d-1` additions), so `mul` is kept in `allowed_symbols` and `log` is not required.

A residual concern to watch: a product of d uniforms concentrates by CLT, so the extremes of the product range are starved of augmented samples exactly when d is large. If SR degrades at high d, check coverage before concluding SR failed. The oracle-free fix is to stratify the augmentation draws along the network's own first surviving eta output.

## The rank rule

Two steps. Both had to be corrected after testing against the analytic Fisher; the details below are the reason the rule works, not incidental implementation choices.

1. Compute the **per-sample** prior-normalised Fisher spectra, normalise each by its own leading eigenvalue, and take the median relative spectrum across samples. Cut at the largest multiplicative gap (`--rank-method eigengap`). For this simulator the correct answer is `r = 1` at every d.
2. Score each eta axis by the fraction of its Jacobian energy lying in the **per-sample** informative eigen-subspace, averaged over samples, and keep the `r` highest-scoring axes. Independent of any ordering convention.

### Why per-sample and not the mean Fisher

This matters more than it sounds. The Fisher here is exactly rank 1 at every theta, but the informative direction `g ~ P/theta` rotates across the prior, so the mean of those rank-1 outer products is **full rank**. Measured on the analytic Fisher:

```text
per-sample median relative spectrum : [1, 1e-8, 4e-10, ...]     <- unambiguously rank 1
mean-Fisher relative spectrum       : [1, 0.012, 0.009, 0.007]  <- looks full rank
```

An eigendecomposition of the mean would not merely be noisy, it would give a *d-dependent* answer for threshold reasons: at a 1e-2 cutoff the mean spectrum yields rank 2 at d=2 (second eigenvalue 0.0177) but rank 1 at d=12 (0.004), manufacturing a spurious trend in the headline result. Whitening and normalisation upstream are invertible congruences `F -> W^T F W`, which preserve per-sample rank exactly, so the per-sample diagnostic survives them.

### Why an eigengap and not a fixed floor

The analytic Fisher has a null-space floor around 1e-8, but the *fishnet-estimated* Fisher has a noise plateau near 1e-2 relative. Measured on real pipeline output:

```text
d=2 : median relative spectrum [1, 0.0162]                  -> floor rule says rank 2 (wrong)
d=4 : median relative spectrum [1, 0.0092, 0.0091, 0.0084]  -> floor rule says rank 1 (right, barely)
```

Any fixed floor near 1e-2 lands inside the plateau, so the retained rank becomes an artefact of the cutoff. The plateau is however very flat (0.0092, 0.0091, 0.0084 are nearly identical) and well separated from the leading eigenvalue, so cutting at the largest multiplicative gap gets every case right with no tuning: rank 1 for both spectra above, rank 2 for a genuine rank-2 Fisher, and full rank when there is no plateau at all (`--rank-min-gap` guards this, so the rule cannot under-report a genuinely non-degenerate model). This is also far easier to defend in a rebuttal than a hand-set threshold.

### The kept axis is not axis 0

Confirmed empirically: the surviving axis came out as index 1 at d=2 and index 3 at d=4, with unambiguous scores (`[0.548, 1.0]` and `[0.084, 0.014, 0.007, 0.989]`). A hardcoded index would have fitted SR to regulariser noise in both cases.

### Leakage is a diagnostic, not a gate

`linearity_residual` for the discarded axes came out at 0.9995, i.e. the discarded axes carry essentially *all* the Jacobian nonlinearity energy. That is the expected outcome here, because those axes are driven by the Fisher regulariser, and it is direct evidence that nonlinearity energy would be the wrong quantity to select axes by. Record it; do not treat a high value as failure. `nonlinearity_spectrum` is recorded alongside for the same reason.

SR is fitted only to the surviving axes, using `components_to_fit=keep_axes` with `slice_fisher=False`. `slice_fisher=True` would be wrong here: it "assumes components map to parameters 1-to-1" and would slice the theta inputs symmetrically with the eta outputs, whereas the surviving coordinate depends on all d thetas. All d inputs, `r` outputs.

Downstream NPE then trains on `n_eta = r < d` parameters.

The full eigenvalue spectrum is written to `rank_spectra.npz` for every run so the threshold can be revisited without rerunning anything.

Do not reach for `degeneracy_structure_scores` or `schur_marginalize_fisher` here. Those rank and marginalise *theta* parameters, and all d thetas are exchangeable in the product, so they will rank flat. The reduction here drops eta outputs, not thetas.

## Success criteria, defined before running

Two criteria, reported as separate columns, because they will separate:

- `rank_correct`: the rule returns `r = 1`.
- `symbolic_recovered`: Spearman `|rho|` between the discovered coordinate and the analytic log-product axis, on held-out draws, at or above `--recovery-corr-thresh` (default 0.99).

Spearman is primary because any monotone function of the product carries the same information; Pearson against both `P` and the standardised log-product is also recorded, since a high Spearman with low Pearson tells you SR found a monotone but differently-curved representative.

Expect rank detection to hold well past the d at which symbolic recovery breaks. That divergence point is not a failure to hide -- it is the quantified scaling limit FoSE asked for, offered voluntarily. Report failures in the denominator with the stage that failed.

## Validate the rule elsewhere

A rank detector whose only test case has the answer 1 is not convincing. Before the rule appears in the rebuttal as a general diagnostic, confirm it recovers the known `n_eta` on SIR, Rosenbrock, and the GW cases. That is what turns it from a heater-specific heuristic into a contribution, and it directly addresses the objection that the method is unproven outside low dimensions.

## Budget accounting

One training set of `--nsims` per run is shared by the fishnet ensemble and all three NPE arms, so the training budget is a single number `N`. Pass `--independent-npe-sims` for the conservative accounting where the NPE arms get their own fresh simulations.

Note that `--n-test` is not only an evaluation set: `train_fishnets` writes its outputs evaluated on the held-out set, so the flattening, alignment, and rank stages all operate on `--n-test` samples rather than on `--nsims`. Total simulator calls per run are therefore `--nsims + --n-test`, and quoting only `--nsims` would understate the cost. State both.

Augmented SR evaluations are network evaluations of a simulator-independent map and are counted in their own column, never folded into the simulation count.

## Two implementation traps

Both were hit during testing and are handled in the script, but they matter if the settings are changed.

`fit_flattening` truncates the sample axis to a multiple of `batch_size`, and when the sample count is *below* the batch size the truncation drops every sample and fails later with an opaque empty-median error from `compute_robust_norm_factor`. Since the samples reaching that stage are `--n-test` and the default batch size is 250, any `--n-test` below 250 would trigger it. The script clamps the batch size and reports when it does. The shared module is deliberately left unmodified so other experiments are unaffected.

`return_model=True` writes a cloudpickle bundle per call by default. The script passes `save_flatten_model_pickle=False`, since it holds the module in memory and a per-run pickle across a few hundred runs is pure I/O overhead plus an unnecessary dependency.

## Results tables

### Table 1 (headline): log-prob against ambient dimension

This is the table the whole rerun exists to produce -- raw-theta inference degrading from d=2 to d=12 while both the oracle and the newly discovered coordinates stay flat.

It has to be built from the common-axis columns, not from each arm's own training log-prob. The reason is a units problem that a reviewer will find: the raw arm's target is a density on R^d, so its log-prob carries d-dependent units and would fall with d even for a perfect estimator. A curve built that way does not isolate a statistical effect from a dimensional one. Every number below is instead a mean log-density on a single **scalar** axis, so all entries share units and "flat" is a meaningful claim.

Because there is no exact push-forward from the discovered coordinate back to the analytic one, the comparison splits into two exact blocks rather than one three-column table. Both blocks are evaluated on the same held-out observations.

```text
      analytic axis          |      discovered axis
d  |  raw      | analytic    |  raw      | discovered
---|-----------|-------------|-----------|------------
2  | [VAL]     | [VAL]       | [VAL]     | [VAL]
3  | [VAL]     | [VAL]       | [VAL]     | [VAL]
4  | [VAL]     | [VAL]       | [VAL]     | [VAL]
6  | [VAL]     | [VAL]       | [VAL]     | [VAL]
8  | [VAL]     | [VAL]       | [VAL]     | [VAL]
10 | [VAL]     | [VAL]       | [VAL]     | [VAL]
12 | [VAL]     | [VAL]       | [VAL]     | [VAL]
```

Columns come from `metrics_aggregate.csv`: `raw_on_analytic_marg_mean`, `analytic_on_analytic_marg_mean`, `raw_on_discovered_marg_mean`, `discovered_on_discovered_marg_mean`, each with its `_sem` partner. Quote as `mean +- sem` over trials.

The two `raw` columns are the same trained model scored on two different axes, so they should track each other closely. That agreement is a free consistency check: it confirms the discovered axis carries the same information as the analytic one, and a divergence between them at some d is a signal that discovery failed at that d rather than that inference did.

### Table 2: the same result as a gap, which is the punchier form

Differences taken within a single axis, so units cancel exactly and the dimensional objection cannot be raised at all. Both gaps should *grow* with d, because the raw arm degrades while the coordinate arms do not.

```text
d  | analytic - raw | discovered - raw | discovered / analytic
---|----------------|------------------|----------------------
2  | [VAL]+-[SEM]   | [VAL]+-[SEM]     | [VAL]
12 | [VAL]+-[SEM]   | [VAL]+-[SEM]     | [VAL]
```

The third column is the fraction of the oracle's advantage that discovery actually captures, which is the number that makes the case that the discovered coordinate is not merely better than nothing but close to the best available. This is the single most useful cell in either table for the rebuttal, and it is the reason the oracle arm is worth keeping.

### Quantify "relatively constant" rather than asserting it

Two of the reviews objected to unsubstantiated claims, so do not let "stays flat" rest on the reader's eye. Fit a straight line in d to each of the four columns over the full 2..12 range and report the slope in nats per dimension with a standard error:

```text
arm                    | slope (nats / dim) | flat?
-----------------------|--------------------|-------
raw (analytic axis)    | [VAL] +- [SEM]     | no
analytic               | [VAL] +- [SEM]     | yes
raw (discovered axis)  | [VAL] +- [SEM]     | no
discovered             | [VAL] +- [SEM]     | yes
```

The claim to make in the rebuttal is that the coordinate arms' slopes are statistically indistinguishable from zero while the raw arm's is not, with both numbers given. If the discovered arm's slope turns out to be small but significantly non-zero, say so and quote it -- a mild measured degradation is still a strong result against a raw arm that degrades an order of magnitude faster, and overclaiming flatness here is exactly the kind of thing that cost the paper credibility the first time.

### Table 3 (appendix only): native per-arm log-probs

`raw_log_prob`, `analytic_log_prob`, and `discovered_log_prob` are each arm's own best validation log-prob on its own target. **These are not comparable across arms**, because the targets have different dimensionality. They are readable down a column (one arm against d) but never across a row. Include them only as a supplementary record, with that caveat stated, and never as the headline. The existing `heater_dim_scaling_sweep.py` figure `log_prob_vs_d.pdf` is built from quantities of this kind, which is part of why the original claim was attackable.

### Table 4: recovery rates

`recovery_table.csv`, one row per d, carrying the two-part success criteria:

```text
d  | trials | rank 1/N | symbolic X/N | median |rho| | median complexity
---|--------|----------|--------------|--------------|------------------
2  | 10     | 10/10    | [X]/10       | [VAL]        | [VAL]
4  | 10     | [X]/10   | [X]/10       | [VAL]        | [VAL]
8  | 10     | [X]/10   | [X]/10       | [VAL]        | [VAL]
12 | 10     | [X]/10   | [X]/10       | [VAL]        | [VAL]
```

Table 1 should be read together with this one. A flat discovered-arm curve is only meaningful at dimensions where symbolic recovery actually succeeded, so if recovery breaks at some d, the discovered column above it is reporting inference in a coordinate that is not the intended one. Mark those cells rather than dropping them.

### Building the tables from the sweep output

```python
import pandas as pd, numpy as np

df = pd.read_csv("heater_discovery_scaling_v1/metrics.csv")
ok = df[df["status"] == "ok"]

cols = [
    "raw_on_analytic_marg", "analytic_on_analytic_marg",
    "raw_on_discovered_marg", "discovered_on_discovered_marg",
]

# Table 1: mean +- sem per d
t1 = ok.groupby("d")[cols].agg(["mean", "sem"]).round(3)

# Table 2: gaps, paired within trial so the sem reflects the paired comparison
g = ok.assign(
    gap_analytic=ok["analytic_on_analytic_marg"] - ok["raw_on_analytic_marg"],
    gap_discovered=ok["discovered_on_discovered_marg"] - ok["raw_on_discovered_marg"],
)
t2 = g.groupby("d")[["gap_analytic", "gap_discovered"]].agg(["mean", "sem"]).round(3)
t2[("captured_fraction", "")] = (
    t2[("gap_discovered", "mean")] / t2[("gap_analytic", "mean")]
).round(3)

# Slopes in nats per dimension, with standard errors
for c in cols:
    sub = ok.dropna(subset=[c])
    fit, cov = np.polyfit(sub["d"], sub[c], 1, cov=True)
    print(f"{c:32s} slope {fit[0]:+.4f} +- {np.sqrt(cov[0, 0]):.4f} nats/dim")
```

Take the gaps paired within a trial before aggregating, as above. Differencing the per-`d` means instead would throw away the seed pairing and inflate the error bars, since the raw and coordinate arms in a given trial share the same simulations and the same seed.

### Shape of a successful outcome

The following came from running the snippet above on **synthetic placeholder data**, purely to show the format and what a clean result would look like. These are not results and must not be quoted anywhere.

```text
d      raw (analytic axis)   analytic
2      -1.040 +- 0.020       -0.416 +- 0.013
12     -3.482 +- 0.017       -0.385 +- 0.025

d      analytic - raw        discovered - raw      captured fraction
2       0.624 +- 0.024        0.539 +- 0.013       0.864
12      3.097 +- 0.014        2.991 +- 0.041       0.966

raw (analytic axis)      slope -0.2471 +- 0.0022 nats/dim
analytic                 slope +0.0005 +- 0.0023 nats/dim
raw (discovered axis)    slope -0.2435 +- 0.0025 nats/dim
discovered               slope -0.0005 +- 0.0025 nats/dim
```

Read that way the claim becomes fully quantitative: the raw arm loses roughly a quarter of a nat per added dimension while both coordinate arms have slopes indistinguishable from zero, and the discovered coordinate captures an increasing fraction of the oracle's advantage as the problem gets harder. The captured fraction rising with d is the strongest single statement available, because it says the discovery matters more, not less, in exactly the regime the reviewers doubted.

## Run commands

Cheap rank-rule ablation over the regulariser, no SR and no NPE:

```bash
for nz in 1e-4 1e-3 1e-2 1e-1; do
  python scripts/heater_discovery_dim_scaling_sweep.py \
    --dims 2 3 4 6 8 10 12 --num-trials 3 --nsims 1000 --n-test 1000 \
    --rank-only --flatten-noise "$nz" \
    --out-dir heater_rank_ablation_noise"$nz"
done
```

Single-run check that the SR and NPE stages work before committing to the sweep:

```bash
python scripts/heater_discovery_dim_scaling_sweep.py \
    --dims 2 --num-trials 1 --nsims 1000 --n-test 1000 \
    --sr-time-limit 120 --n-marginal-val 50 \
    --keep-workdirs --out-dir heater_discovery_singlecheck
```

Full end-to-end sweep:

```bash
python scripts/heater_discovery_dim_scaling_sweep.py \
    --dims 2 3 4 6 8 10 12 --num-trials 10 --nsims 1000 \
    --flatten-noise 1e-3 --sr-time-limit 300 \
    --out-dir heater_discovery_scaling_v1
```

The script writes `metrics.csv` after every run and supports `--resume`, so a job that hits a wall-clock limit can be requeued without losing completed runs. Per-run failures are recorded with the failing stage and the sweep continues rather than aborting; this was verified by an induced failure.

`--rank-only` deliberately imports neither torch nor ltu-ili, so the ablation runs in a minimal environment.

On an Apple-silicon laptop the Metal JAX backend cannot legalise the `triangular_solve` in `training_loop_fishnets.py`. Prefix local runs with `JAX_PLATFORMS=cpu`. This does not affect the cluster.

## What has been verified so far

Smoke-tested end to end through rank selection at d=2 and d=4 with 1000 training simulations. Both returned rank 1 with clean axis scores. The rank rule itself was validated separately against the analytic Fisher at d=2,4,8,12, under an invertible congruence, on a genuine rank-2 Fisher, and on a genuinely full-rank Fisher.

Not yet exercised: the SR stage, the three NPE arms, and the common-axis evaluation. Those need ltu-ili and a GPU, so run a single `--dims 2 --num-trials 1` job on the cluster before launching the full sweep.

One caveat observed: at deliberately small settings (400 simulations, 60 fishnet epochs, 3 members) the estimated Fisher does not resolve the rank-1 structure at all, giving a median relative spectrum of `[1, 0.72, 0.58, 0.043]`. The rank rule needs a reasonably converged Fisher ensemble, so do not economise on `--num-fishnets` or `--fishnet-epochs` and check the saved spectra rather than trusting the retained rank blindly.

## Checklist before quoting numbers

- Confirm the retained rank is 1 at every d, and that the noise ablation leaves it unchanged.
- Confirm the median relative spectra show a real gap rather than a marginal one, using `rank_spectra.npz`.
- Confirm `max_length` was large enough at the highest d to represent a d-way product at all.
- Confirm the augmentation pool scaled with d, and check coverage of the product range before attributing any high-d failure to SR.
- Confirm failures are in the denominator with the failing stage named.
- Confirm all three arms used the same density estimator.
- Confirm the discovered-axis comparison nowhere used the analytic coordinate for training.
- Confirm the rank rule was also validated on SIR, Rosenbrock, and GW.
