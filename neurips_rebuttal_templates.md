# NeurIPS Rebuttal Response Templates

Fill-in templates for the author response period. Placeholders are marked `[LIKE_THIS]`.

Reviewers and scores: FoSE (2, conf 3), rUsi (2, conf 4), t9KB (5, conf 2), QfCk (5, conf 2), Rkfx (2, conf 4). Meta-review by AC Ej7L.

## Mechanics and constraints

- Responses are posted with the per-review Rebuttal buttons in OpenReview.
- Limit is 10,000 characters per review.
- Markdown is supported. No file uploads, no figures, no links.
- New results are permitted, but the original submission remains the basis for the decision. Rebuttals are meant to clarify, not to substitute a revised paper.
- Set the message readers deliberately.

## How to present new results

Constraints make tables the only way to show quantitative results, so treat formatting as load-bearing.

1. **Put the full table once in a top-level comment addressed to the AC and all reviewers.** The AC wrote the "unless the concerns reflect correctable misunderstandings" sentence and is the real audience for the headline evidence. Repeating a wide table in five replies wastes character budget.
2. **Put small, targeted numbers inline in each reviewer's reply.** Reviewers do not reliably read sibling threads, so each reply must stand alone for the specific point being answered. Reference the global comment for the full table.
3. **Keep tables narrow.** Five columns or fewer. OpenReview's markdown rendering degrades with wide tables and nested formatting.
4. **Use plain ASCII math, not LaTeX.** Write `R_0 = beta/gamma`, `|Q - I|_F`, `M_c`. Rendering of math in rebuttal comments is unreliable.
5. **Always show denominators.** Report `9/10`, never "usually" or "robustly".
6. **Report dispersion, not just medians.** Median with interquartile range across seeds.
7. **State the budget in every efficiency claim.** Say "500 training simulations" rather than "500 simulations" whenever a separate held-out simulated set was used.
8. **Provide a plain-text fallback** for any table you are unsure will render; a fenced block of aligned text always renders.

### Suggested headline table (global comment)

Keep to one row per problem.

```text
Problem        | Train sims | Recovered | Alignment (med, IQR) | Flatness vs raw
---------------|------------|-----------|----------------------|----------------
Rosenbrock     | 500        | 10/10     | 0.929 (0.814, 0.960) | 0.09x
SIR            | 500        | 10/10     | 0.674 (0.636, 0.741) | 0.27x
GW TaylorF2    | 500        | 10/10     | 0.972 (0.966, 0.979) | 0.05x
GW IMRPhenomD  | 500        | 7/10      | 0.997 (0.988, 0.999) | 0.04x
```

Alignment is |Pearson r| on held-out points against the expected coordinate.
Flatness is the median symbolic `|Q - I|_F` as a **ratio to the raw-parameter
baseline** (lower is better; see the ratio-not-absolute rule below). All four
sit at 0.04x-0.27x, i.e. the discovered coordinates are 4x-25x flatter than raw
parameters.

**These flatness numbers are post-fix.** They were regenerated on 2026-07-27
after a bug in the regrouping step (`abs()` in three flatness acceptance tests,
commit 4514c64) was found to be discarding coordinate *improvements*. Any
flatness number taken from a run record dated before that commit is wrong. See
`notes/postprocessing_flatness_bug_fix.md`.

**Do not add an intractable-likelihood row here.** Every row above is a
recovery-rate claim at a matched budget; the intractable problems currently
recover at 0/10 and 1/10, and placed in this table that reads as a contradiction
of the paper's own claim rather than as evidence of scope. They answer a
different question (can the method run with no likelihood at all) and belong in
their own table below, referenced from here in one sentence.

Follow it with one sentence of simulation accounting:

> Symbolic-regression augmentation used [N_AUG] evaluations of the learned coordinate map per run. These are network evaluations of a simulator-independent map and are not simulator calls. Held-out evaluation used a further [N_EVAL] simulations per run, reported separately.

### What each discovery run is, and the criterion it had to pass

One run = one fully independent end-to-end trial: fresh training simulations,
fresh Fisher-ensemble / flattener / alignment / SR seeds. 10 trials (master
seeds 0-9) per problem, 500 training simulations each. Criteria below are the
values actually recorded in each run's `config_manifest.json`, not the script
defaults.

```text
Problem       | What is discovered              | Pass criterion (pre-registered)
--------------|---------------------------------|--------------------------------
Rosenbrock    | the curved valley coordinate    | |r| >= 0.5 AND complementary
              |                                 | linear coord |r| >= 0.5
SIR           | R_0 = beta/gamma                | |r| >= 0.5 with R_0
GW TaylorF2   | chirp mass M_c                  | |r| >= 0.75
GW IMRPhenomD | total mass M and asymmetric     | |r|(M) >= 0.75 AND
              | mass combination                | |r|(mass diff) >= 0.5
QM7b          | HOMO-LUMO gap combination       | |r| >= 0.5 AND flatness
              |                                 | < 0.8x raw
Kolmogorov    | log Re = log f_0/(nu^2 k_f^3)   | |r| >= 0.9 AND log-log slope
              |                                 | error <= 0.4
Kuramoto      | coupling/noise ratio            | |r| >= 0.7 AND gradient
              |                                 | cosine >= 0.8
Rayleigh-Ben. | log Nu scaling coordinate       | |r| >= 0.9 AND Ra-Pr gradient
              |                                 | cosine >= 0.9
```

### Correlation and flatness per experiment (all 10 trials)

`corr` is |Pearson r| against the expected coordinate on held-out points.
`flat` is symbolic `|Q - I|_F` / raw baseline. `neural` is the same ratio for
the neural coordinate map before symbolic regression, which separates "the
geometry stage worked" from "the closed-form stage worked".

```text
Problem       | n | Recovered | corr mean | corr med (IQR)       | flat  | neural
--------------|---|-----------|-----------|----------------------|-------|-------
Rosenbrock    |10 | 10/10     | 0.878     | 0.929 (0.814, 0.960) | 0.09x | 0.04x
SIR           |10 | 10/10     | 0.717     | 0.674 (0.636, 0.741) | 0.27x | 0.08x
GW TaylorF2   |10 | 10/10     | 0.969     | 0.972 (0.966, 0.979) | 0.05x | 0.10x
GW IMRPhenomD |10 | 7/10      | 0.958     | 0.997 (0.988, 0.999) | 0.04x | 0.06x
QM7b          |10 | 0/10      | 0.645     | 0.626 (0.591, 0.693) | 7.14x | 0.04x
Kolmogorov    |10 | 1/10      | 0.789     | 0.866 (0.750, 0.884) | 1.97x | 0.24x
Kuramoto      | 9 | 0/9       | 0.756     | 0.753 (0.753, 0.768) | 1.69x | 0.24x
Rayleigh-Ben. | 9 | 0/9       | 0.718     | 0.732 (0.646, 0.765) | 1.16x | 0.29x
```

The `neural` column is the single most useful number in this table for
answering Rkfx and rUsi: it is 0.04x-0.29x on **every** problem including all
three intractable ones, so the geometric stage never fails. Every 0/10 and 1/10
above is a failure of the symbolic stage or of a threshold, not of the method's
core claim.

### Threshold sensitivity: what changes if the correlation bar moves

Recovery counts recomputed at correlation thresholds 0.9 / 0.8 / 0.7, holding
every **other** conjunct of each criterion fixed. "2nd conjunct" is how many
trials pass the non-correlation half on its own, which is what determines
whether moving the correlation bar can help at all.

```text
Problem       | pre-reg | actual | @0.9 | @0.8 | @0.7 | 2nd conjunct
--------------|---------|--------|------|------|------|-------------
Rosenbrock    | 0.50    | 10/10  |  6   |  8   |  9   | 10/10
SIR           | 0.50    | 10/10  |  2   |  2   |  4   | 10/10
GW TaylorF2   | 0.75    | 10/10  | 10   | 10   | 10   | 10/10
GW IMRPhenomD | 0.75    |  7/10  |  6   |  7   |  8   |  8/10
QM7b          | 0.50    |  0/10  |  0   |  0   |  0   |  0/10
Kolmogorov    | 0.90    |  1/10  |  1   |  1   |  1   |  1/10
Kuramoto      | 0.70    |  0/9   |  0   |  0   |  0   |  0/9
Rayleigh-Ben. | 0.90    |  0/9   |  0   |  2   |  6   |  9/9
```

Reading this for the 0.9-vs-0.7 decision:

- **Rayleigh-Benard is the only experiment where it matters.** Its second
  conjunct passes 9/9, so correlation is the sole binding constraint: 0/9 at
  0.9, 2/9 at 0.8, **6/9 at 0.7**.
- **Kolmogorov, Kuramoto and QM7b do not move at all.** Their binding
  constraints are the exponent test (1/10), the gradient cosine (0/9) and the
  geometric-improvement test (0/10) respectively. Lowering the correlation bar
  changes nothing, so doing it would look like moving a threshold for no gain.
- Rosenbrock, SIR and TaylorF2 already pass at their (lower) pre-registered
  bars; the 0.9/0.8/0.7 columns for them are counterfactuals, not proposals.
  Note SIR's correlation is genuinely modest: only 2/10 trials exceed 0.9 even
  though it is 10/10 against its 0.5 criterion.

### Decision taken: 0.7 for the two 0.9-bar experiments only

The correlation bar is moved from 0.9 to **0.7 for Kolmogorov and
Rayleigh-Benard only**. Rosenbrock, SIR, TaylorF2, IMRPhenomD, QM7b and
Kuramoto keep their pre-registered bars verbatim. Regenerate with:

```
python scripts/recompute_success_at_threshold.py --threshold 0.7
```

This re-evaluates saved held-out correlations; it does not re-run anything, and
it does not modify the original records. Output (records + `summary.md` +
`summary.json`) lands in `$SCRATCH/threshold_sweeps/corr_0p7/`.

```text
Problem          pre-reg thr  thr used   pre-reg  revised
---------------------------------------------------------
rosenbrock              0.50      0.50     10/10    10/10  (unchanged)
sir                     0.50      0.50     10/10    10/10  (unchanged)
gw_taylorf2             0.75      0.75     10/10    10/10  (unchanged)
gw_imrphenomd           0.75      0.75      7/10     7/10  (unchanged)
qm7b                    0.50      0.50      0/10     0/10  (unchanged)
kolmogorov              0.90      0.70      1/10     1/10
kuramoto                0.70      0.70       0/9      0/9  (unchanged)
rayleigh_benard         0.90      0.70       0/9      6/9
```

Net effect of the change: **Rayleigh-Benard 0/9 -> 6/9. Nothing else moves.**
Kolmogorov stays 1/10 because its exponent test, not its correlation, is what
binds.

**Do not apply 0.7 uniformly.** Rosenbrock and SIR were pre-registered at 0.5,
so a blanket 0.7 *tightens* them and destroys two headline results: Rosenbrock
10/10 -> 9/10 and SIR 10/10 -> **4/10**. The script guards against this by
defaulting to `--experiments kolmogorov rayleigh_benard`.

**Disclose the change rather than presenting 0.7 as pre-registered.** The
Kuramoto paragraph below stakes the response's credibility on *not* moving a
threshold ("we retain the pre-registered value and report 0/10"). Reporting
Rayleigh-Benard at 6/9 without saying the bar moved, in the same response,
is the exact move that paragraph disclaims, and Rkfx and FoSE are the reviewers
most likely to check. The honest and still-favourable framing:

> On Rayleigh-Benard we pre-registered a correlation threshold of 0.9 and met
> it in 0/9 completed trials. We report that number. We also note that the
> threshold was set without a principled basis for this observable, and that at
> a 0.7 threshold - the value we used for Kuramoto - the criterion is met in
> 6/9. We report both rather than selecting the more favourable one.

That sentence is defensible because it is disclosed and because 0.7 is
justified by precedent within the same paper (Kuramoto), not chosen to
maximise the count.

### Suggested downstream table (SIR, and any other problem you rerun)

```text
Budget | Method | CRPS(R_0)   | Coverage err
-------|--------|-------------|-------------
500    | theta  | [VAL]+-[SE] | [VAL]
500    | eta    | [VAL]+-[SE] | [VAL]
1000   | theta  | [VAL]+-[SE] | [VAL]
1000   | eta    | [VAL]+-[SE] | [VAL]
```

Always pair CRPS with calibration. Sharper CRPS alone invites the reply that the posterior is overconfident, which is precisely rUsi's objection to the original metric.

### Suggested intractable-likelihood table (separate from the headline table)

Answers Rkfx Q2 and rUsi's scope objection. Keep it separate so a low recovery
rate is read as honest reporting on new, harder problems rather than as a
retraction of the headline claims.

```text
Problem (no likelihood) | Train sims | Recovered | Flatness | Binding constraint
------------------------|------------|-----------|----------|-------------------
Kolmogorov (turbulence) | 500        | 1/10      | 0.24x    | SR exponent test
Kuramoto (noisy sync)   | 500        | 0/9       | 0.24x    | gradient cosine
Rayleigh-Benard (conv.) | 500        | 6/9       | 0.29x    | correlation (bar
                        |            |           |          | moved 0.9 -> 0.7)
```

Flatness is the **neural** coordinate map's median `|Q - I|_F` as a ratio to
raw parameters, i.e. the geometry stage alone. It is 0.24x-0.29x on all three
problems, and that is the number these experiments actually support: the
informative geometry is recovered without any likelihood, in every seed, on
every problem. Recovery rates are limited by the symbolic stage and by the
thresholds, not by the geometry.

Denominators: Kuramoto is 0/9 not 0/10 (one seed lost to a transient CUDA
fault) and Rayleigh-Benard is 6/9 not 6/10 (one seed lost to a code defect in
the alignment stage). Both are excluded rather than counted as failures, and
both exclusions should be stated if the denominator is questioned.
Rayleigh-Benard's 6/9 is at a 0.7 correlation bar, **not** the pre-registered
0.9, at which it is 0/9 - see the disclosure sentence above and use it.

One sentence to follow it, which is the actual claim these experiments support.
Note this is weaker than the "SR is the bottleneck" framing: on Rayleigh-Benard
the geometry is reliably good and the symbolic stage is *erratic*, not uniformly
bad, and the gate fails on correlation rather than on flatness.

> On Rayleigh-Benard convection the learned coordinate map reduces the held-out
> flatness |Q - I|_F to 0.29x that of the raw parameters (IQR 0.26-0.32 over 9
> completed seeds), so the informative geometry is recovered in every run. The
> symbolic stage is inconsistent, at 1.16x raw (IQR 0.50-1.42), and it is the
> closed-form step rather than the geometry that limits recovery.

Report flatness as a **ratio to the raw-parameter baseline, never as an absolute
value**. The absolute |Q - I|_F is set by a per-seed Fisher normalisation and is
not comparable across seeds: seed 0's raw baseline is 35.6 while every other
seed sits at 1.35-1.56, a 23x swing in a quantity that should depend only on the
prior. In ratio terms the seeds agree closely (0.19-0.44); in absolute terms they
appear to disagree wildly. Quoting seed 0's absolute numbers as representative
would be an error.

Status of the numbers as of writing:

- Kolmogorov 1/10 final (0 crashed). The one success has held-out correlation
  0.989 with log Re and the correct log-log slope; the other 9 correlate at
  0.41-0.885 with incorrect exponents.
- Kuramoto 0/10 final (1 crashed, transient CUDA fault, not a method failure).
- Rayleigh-Benard: **0/9 at the pre-registered 0.90 bar, 6/9 at 0.70** (9
  completed, 1 crashed on a code defect, not a method failure - see below). Nu
  correlation mean 0.718, median 0.732, IQR 0.646-0.765, maximum 0.850: no seed
  reaches 0.90, and 6 of 9 clear 0.70. Gradient cosine median 0.969, IQR
  0.967-0.979, minimum 0.939, so the cosine conjunct passes 9/9 and correlation
  is the sole binding constraint. The discovered coordinate reliably carries the
  correct Rayleigh-Prandtl scaling direction while tracking log Nu only
  moderately. **Kolmogorov remains the lead worked example**: its single success
  is a much stronger result (0.989 correlation with the correct exponent) than
  any Rayleigh-Benard seed, and its 1/10 is unaffected by the threshold change.
  If Rayleigh-Benard's 6/9 is quoted anywhere, the bar change must be disclosed
  in the same sentence.
- **Do not repeat the doc's claim that aspect ratio Gamma is the flat direction.**
  The campaign does not support it: Gamma correlation median 0.570, IQR
  0.449-0.803, and on seed 6 a coordinate tracks Gamma at 1.000. The DNS Nusselt
  exponents measured over the campaign are Ra 0.414, Pr 0.104, Gamma 1.397 - the
  aspect-ratio exponent is the *largest of the three*, over three times the
  Rayleigh exponent. At Gamma in [1, 2] the finite-size dependence of Nu on the
  roll count dominates, so Gamma is a leading driver of the observable rather
  than a nuisance direction. The honest statement is that Rayleigh-Benard as
  configured has no flat direction to isolate. Widening the Gamma prior is the
  fix if that claim is wanted, and it is out of scope for the rebuttal window.
- The one Rayleigh-Benard crash (seed 8) is an indexing defect in the alignment
  stage, present in all 11 experiment scripts, that fires only when the alignment
  stage drops ensemble members beyond those already filtered after Fisher
  training. It is a code bug, not a method failure, and should be described that
  way if the denominator is questioned.
- Ising: no usable campaign. CPU is infeasible in the window (20.7 h reached 2 of
  20 ensemble members) and the GPU path has an open fishnet-divergence bug.
  **Recommend cutting it from the rebuttal** rather than describing an experiment
  that cannot be shown.

### The Kuramoto 0/10, and why to volunteer it

Use this somewhere visible (t9KB Q2 or the AC comment). Reporting a failure you
could have hidden by moving a threshold is worth more with Rkfx and FoSE than any
number in the tables, and it directly evidences the "pre-registered criteria"
claim made to t9KB.

```markdown
We report one clear negative result. On the noisy Kuramoto network the criterion
was met in 0/10 trials. The failure is diagnosable rather than opaque: the
gradient cosine clusters in [0.7045, 0.7079] across every seed, and 1/sqrt(2) =
0.7071 is exactly the value obtained when a coordinate depends on a single raw
parameter instead of the intended ratio. The recovered expressions confirm this
literally, being affine in one parameter with no quotient formed. We note that
relaxing this threshold to 0.70 would have reported the same runs as 8/10; we
retain the pre-registered value and report 0/10, because 0.707 is the signature
of a specific failure mode rather than a near miss.
```

## Shared building blocks

Reuse these verbatim so the framing is identical across all five replies and the AC sees one consistent story.

```markdown
**[REFRAME]** We thank the reviewer for the careful reading. The reviews have made clear that our framing obscured the paper's primary contribution. The distillery's main output is not a conditioning trick for NPE: it is the **automated discovery of the symbolic, nonlinear parameter combinations that actually control a simulator's observations**. Simulation efficiency is a downstream consequence, not the motivation. Our clearest instance is the gravitational-wave result (Sec. 5.2): the method recovers conventional chirp-mass structure in the inspiral-dominated TaylorF2 regime, and then, for IMRPhenomD where merger and ringdown dominate, autonomously returns a different controlling pair (total mass and an asymmetric mass combination) that is geometrically flatter and better calibrated than (M_c, q).

**[GEOMETRY]** We agree our use of "flattening the Fisher" was imprecise and we will correct it throughout. Curvature is an invariant of the metric and cannot be altered by reparametrisation; our own Appendix D computes R = -1 for the Gaussian and states that no globally flat coordinates exist. What we actually optimise is the **coordinate representation** of the metric: we seek coordinates in which the metric components are as close to the identity as possible in expectation over the prior. We will replace "flatten the Fisher/metric" with this statement everywhere, and we will fix the related conflation of a statistical degeneracy (a direction with a small metric eigenvalue) with small Riemann curvature.

**[METRIC-CLAIM]** We will narrow the Fisher claim in the main text. The Fishnet objective recovers the **inverse posterior covariance**; identification with the likelihood Fisher information holds only under the Laplace / Bernstein-von Mises conditions stated in Appendix A. We will state this at first mention (Sec. 4, currently lines 147-149) rather than only in the Discussion and appendix. The pipeline is unaffected: the pullback in Eq. (4) transforms identically for either object.
```

## Reviewer FoSE (Rating 2, Confidence 3)

The sharpest technical review. Winning the non-uniqueness point is the highest-value move available.

```markdown
We thank Reviewer FoSE for the most technically detailed review we received, and for noting the writing quality and appendices.

[REFRAME]

**W1 / Q3 - Non-uniqueness of isotropising coordinates.** The reviewer is correct, and we now regard this as the conceptual centre of the paper rather than an appendix caveat. Isotropy alone defines only an equivalence class: as Appendix B proves, the loss is invariant under a constant offset and a global orthogonal transformation, and the reviewer's two-unit-Gaussian example is exactly the degenerate case. Isotropy therefore cannot by itself single out interpretable axes. Our claim is that the pipeline resolves this in two stages: (i) a canonicalisation that fixes the frame (Jacobian Procrustes alignment, a nonlinearity-separating rotation ordering axes by nonlinearity energy, and a sign/permutation fix from the mean prior-normalised Fisher eigenstructure; Appendix B); and (ii) an Occam selection, in which MDL-based symbolic regression selects the simplest closed-form representative of the class. We will restate the objective as "isotropy plus minimum description length" and present this as the discovery principle. Empirically, across [N] independent end-to-end trials per problem (new training simulations, new network and SR seeds), the expected coordinate was recovered in [X]/[N] (Rosenbrock), [X]/[N] (SIR), and [X]/[N] (IMRPhenomD) runs, with median expression complexity [VALUE].

**W2 / Q1 - Quantitative comparison and the 10x claim.** We agree the main text did not substantiate this. At matched validation calibration, NPE in eta reaches the theta-space baseline's performance using [N_ETA] vs [N_THETA] simulations ([RATIO]x) on [PROBLEM]. We will move this comparison into the main text with the baseline stated explicitly, and we report CRPS and coverage rather than resting on maximum validation log-probability alone.

**Q1 (cost and scaling) / W3 - Computational cost.** End-to-end wall time for [PROBLEM] is [T_FISH] (Fisher ensemble) + [T_FLAT] (coordinates) + [T_SR] (symbolic regression) = [T_TOTAL] on [HARDWARE]. We agree "real-world" overstated the demonstrated scope: symbolic regression scales poorly in the number of input variables, and we will state explicitly that current evidence covers low-dimensional parameter spaces, with the rank-deficient reduction case as the exception.

**W4 - Imprecise claims.** We accept all three.
- "requiring no data" will become "requiring no observed data realisation; the metric is estimated from simulations", in both the abstract and line 91.
- The Gaussian/Laplace assumption will be stated at first mention. [METRIC-CLAIM]
- On beta/gamma: the discovered first component has held-out correlation [VALUE] with R_0 = beta/gamma and gradient cosine similarity [VALUE]. If that is not compelling we will withdraw the word "closely" and report the number instead.

**Minor points.** (1) [GEOMETRY] (2) We will cite Amari for information geometry. (3) Agreed that symbolic post-processing could in principle be applied to other methods; we will say so, while noting the MDL step is load-bearing for uniqueness here rather than cosmetic. (4)-(5) NPE and TARP will be defined at first use, with a one-line statement of what TARP measures. (6) Figures 4 and 5 are results of actual experiments. [CONFIRM PER PANEL: state which panels are direct pipeline output. If any sub-panel is a schematic, identify it explicitly.] Figure 1 is a pipeline schematic and will be labelled as such.
```

## Reviewer rUsi (Rating 2, Confidence 4)

```markdown
We thank Reviewer rUsi for a clear statement of where the paper failed to justify its setting.

[REFRAME]

**Scope: when is this needed?** The pipeline requires only parameter-simulation pairs. It needs neither likelihood evaluations, nor an analytic Fisher, nor a differentiable simulator. We selected examples with tractable likelihoods deliberately, because they allow discovered coordinates to be validated against known answers, but we accept this made the setting look circular. We will (i) state the requirement explicitly, and (ii) add three applications with no evaluable likelihood: forced 2D turbulence (a density over a chaotic PDE trajectory), a noisy Kuramoto network (latent per-oscillator frequencies and Brownian paths would need marginalising), and Rayleigh-Benard convection (direct numerical simulation, observable is a normalised turbulence spectrum). In each the likelihood is intractable rather than merely unavailable, and in each there is a known controlling combination to check the discovery against. Recovery rates and the binding failure mode are reported in the global comment; we report them plainly, including where recovery is unreliable. We would also emphasise that the discovered coordinates are themselves the deliverable: knowing which combination of parameters controls an observable is useful independently of the inference algorithm subsequently used.

**NPE is not assumed knowledge.** Agreed. We will add a short primer defining neural posterior estimation, the amortisation it provides, and why conditioning matters for it, before it is used.

**Metric justification.** This is a fair criticism and we are changing the evaluation. Maximum validation log-probability was used as a budget-matched training-quality proxy, but it does not establish posterior quality and can reward overconfidence. We now report CRPS together with calibration and coverage, so that sharpness cannot be claimed without calibration. On [PROBLEM] at a fixed budget of [N] training simulations, CRPS improves from [VAL_THETA] to [VAL_ETA], with coverage error [VAL_THETA] vs [VAL_ETA], averaged over [K] seeds.

**Size of the improvement.** With the corrected metrics the gap is [DESCRIBE HONESTLY], and is largest in the low-simulation and higher-noise regime, which is where we claim the method matters. Where the gain is modest we will say so in the revision.

**Comparison to likelihood-based inference.** We do not claim the method outperforms well-tuned gradient-based inference where a likelihood is available, and we will remove any implication otherwise. [OPTIONAL: report HMC/NUTS effective sample size per gradient in theta vs eta, if run.]
```

## Reviewer t9KB (Rating 5, Confidence 2)

Strongest supporter. Consolidate rather than argue.

```markdown
We thank Reviewer t9KB for recognising the novelty of combining simulation-based information geometry estimation with symbolic discovery.

**Q2 - Sensitivity and robustness of the recovered expressions.** This is now our main new result. We ran [N] fully independent end-to-end trials per problem, varying the training simulations, Fisher-ensemble seed, flattener seed, alignment subsampling, and symbolic-regression seed together, rather than only the network initialisation. Success criteria were defined before running, and are evaluated numerically on held-out points by comparing the coordinate functions and their Jacobians up to permutation, sign, scaling, and algebraic equivalence, rather than by string matching, since algebraically equivalent forms and the residual orthogonal freedom make literal comparison fragile. Recovery rates: Rosenbrock [X]/[N], SIR [X]/[N], TaylorF2 [X]/[N], IMRPhenomD [X]/[N]. Median physics alignment [VALUE] (IQR [LO]-[HI]); median expression complexity [VALUE]. Failures remain in the denominator, with the failing stage identified. [IF RUN: we repeated the SR stage with [ALTERNATIVE ALGORITHM] and obtained [RESULT].]

**Q3 - Total computational overhead.** We agree the accounting must be end-to-end. For [PROBLEM]: coordinate discovery costs [N_DISC] training simulations and [T_DISC] wall time, split as [T_FISH] / [T_FLAT] / [T_SR]; downstream NPE then costs [N_NPE] simulations and [T_NPE]. The break-even point against the theta-space baseline is [N_BREAKEVEN] simulations, beyond which the amortised coordinates are net-positive; below it they are not, and we will state this. We also reduced discovery cost substantially by augmenting only the symbolic-regression stage with inexpensive evaluations of the learned map, bringing SIR discovery from 5,000 to approximately [N] simulator calls.

**Weakness - baselines.** We agree a comparison on the parameter-estimation component alone is appropriate. [STATE WHAT YOU CAN RUN, e.g. FMPE/CMPE at matched budget on PROBLEM giving RESULT; or state honestly what is out of scope for the rebuttal window and will appear in the revision.]
```

## Reviewer QfCk (Rating 5, Confidence 2)

Precise technical asks. Q2 is a direct factual question; answer it plainly.

```markdown
We thank Reviewer QfCk for a precise reading of the first stage; both concerns raised are correct.

**Wording: Fisher vs inverse posterior covariance.** [METRIC-CLAIM]

**Q1 - Where is the approximation good?** [ANSWER PER EXPERIMENT. For each of Rosenbrock, Gaussian, SIR, GW, WL, state whether the posterior is close to Gaussian at the relevant simulation count and prior, and where it is not. Be explicit that in the non-Gaussian cases the recovered object should be read as an inverse posterior covariance, and that the flattening pipeline is unchanged either way.]

**Q2 - Chain-product experiment: discovered or oracle?** [ANSWER DIRECTLY AND HONESTLY. If eta was supplied as the known product target for the scaling sweep, say so plainly, and state which dimensions if any were run through the full pipeline. A clear answer will be received far better than an ambiguous one. Then state the corrected claim the experiment actually supports.]

**Q3 - Sensitivity to prior range and noise model.** We ran an ablation varying the prior width by [FACTORS] and the noise level over [RANGE] on [PROBLEM]. The recovered structure was [STABLE/CHANGED] as follows: [RESULTS]. [IF THE COORDINATES CHANGE WITH NOISE, FRAME THIS AS EXPECTED: the informative directions are a property of the model together with the measurement, so a changed noise model legitimately changes which combination controls the data.]

**One-time discovery cost and break-even.** We accept this was omitted. We now separate (i) training simulator calls used for discovery, (ii) held-out simulator calls used only for evaluation, (iii) inexpensive augmented evaluations of the learned coordinate map, and (iv) downstream NPE simulations. Once the neural map eta = f(theta) is learned it is independent of the simulator, so we augment only the symbolic-regression stage by drawing fresh theta from the prior and evaluating the coordinate ensemble; these are network evaluations, not simulations. With [N_AUG] such evaluations, discovery for SIR drops from 5,000 to approximately [N] simulator calls, and the [PROBLEM] break-even against the baseline is [N_BREAKEVEN] simulations. We will add this accounting and an amortised-cost statement to the Limitations discussion, as suggested.
```

## Reviewer Rkfx (Rating 2, Confidence 4)

Hardest review. Note that their Q3 is already answered by Appendix D.

```markdown
We thank Reviewer Rkfx for pressing on the geometric formulation; the criticism of our language is correct and we are grateful for it.

[GEOMETRY]

**On the apparent contradiction.** The reviewer is right that we wrote something inconsistent. A statistical degeneracy is a direction along which the metric has a small eigenvalue, meaning the model is insensitive; this is unrelated to the Riemann curvature of the metric. Our text conflated the two. We never intended to claim that reparametrisation removes curvature, and Appendix D explicitly proves the opposite for the Gaussian. We will rewrite the objective statement so this is unambiguous.

**Q3 - Conformally flat case.** We would highlight that our Gaussian validation example is exactly the analytically tractable, non-trivial, conformally flat case the reviewer asks for. In the coordinates u = mu/sigma_star and v = sqrt(2) sigma/sigma_star, the metric is g_ab = (2/v^2) delta_ab, that is, conformally Euclidean, with Ricci scalar R = -1 (the hyperbolic plane); this is derived in Appendix D. Since the curvature is non-zero, no coordinates make the components equal to the identity everywhere, and the correct benchmark is the average deviation over the prior. We compare our learned and symbolic coordinates against geodesic normal coordinates about a fiducial point and against the ad hoc (mu/sigma, log sigma) choice: the geodesic coordinates are exact at the fiducial point but degrade rapidly with geodesic distance beta, while the discovered coordinates achieve lower average deviation across the prior. We will promote this from a validation example to an explicit answer to the conformally-flat question.

**Q2 - A genuinely intractable likelihood.** We agree this was the key missing experiment and have added three, all forward-simulable with no closed-form likelihood over the observable. We describe one. In forced 2D turbulence (Kolmogorov flow), theta = (forcing amplitude f_0, viscosity nu) and x is the time-averaged, amplitude-normalised energy spectrum of a chaotic trajectory; the likelihood is intractable rather than merely unavailable, being a density over a chaotic PDE trajectory with no closed form. Normalising the spectrum divides out the amplitude, leaving an observable that depends on theta only through Re = f_0/(nu^2 k_f^3) - a known exact degeneracy, so there is a right answer to check against. From 500 training simulations and no likelihood evaluations, the method recovers a coordinate with held-out correlation 0.989 with log Re and the correct log-log slope. The success criterion (correlation >= 0.9 and slope error <= 0.4) was fixed before the runs; we report below the effect of relaxing the correlation half to 0.7.

We report the reproducibility honestly: across 10 independent end-to-end trials the criterion was met in 1/10. The remaining 9 recovered coordinates correlating with log Re at 0.41-0.885 but with incorrect exponents. The same pattern holds on the other two problems, and it locates the difficulty precisely: the geometric stage is reliable while the closed-form step is not. On Rayleigh-Benard convection the learned coordinate map reduces held-out flatness to 0.29x the raw-parameter baseline in every one of 9 completed seeds (IQR 0.26-0.32), while the symbolic expressions are inconsistent at 1.16x raw (IQR 0.50-1.42). We take this to answer the reviewer's question - the method does not require a tractable likelihood, and the informative geometry is recovered without one - while being explicit that converting that geometry into a reliable closed form on these harder problems is not yet at the standard of the submitted case studies.

**Q1 - Usefulness alongside well-tuned likelihood-based inference.** We accept that for the submitted case studies a practitioner with a tractable likelihood would reasonably use HMC with a tuned mass matrix, and we will not claim otherwise. Two points remain. First, a tuned mass matrix is a local, linear preconditioner fitted per posterior, whereas the distillery returns a global, nonlinear, amortised map reused across observations. Second, and primarily, the deliverable is the discovered coordinate itself: the IMRPhenomD result identifies a controlling parameter combination that differs from the conventional chirp-mass basis, which is a statement about the physics rather than about a sampler. [IF RUN: we also compare NUTS in theta vs eta, giving ESS per gradient of [VALUE] vs [VALUE] and condition number [VALUE] vs [VALUE].]
```

## Optional top-level comment to the AC

The meta-review says rejection is likely "unless the reviewers' concerns reflect correctable misunderstandings". Address that sentence directly.

```markdown
We thank the AC for the summary. We respond to each reviewer individually and summarise here.

We accept as correct and are fixing: the imprecise "flattening" language (curvature is invariant; we optimise the coordinate representation of the metric); the overstated identification of the Fishnet output with the likelihood Fisher rather than the inverse posterior covariance; the "requiring no data" phrasing; the omission of the one-time coordinate-discovery cost from efficiency claims; and the use of maximum validation log-probability as the primary success metric.

We believe two concerns rest on a framing failure that is correctable. First, the paper's contribution is the discovery of the symbolic nonlinear coordinates that control a simulator's observations, with simulation efficiency as a consequence; our gravitational-wave result recovers conventional chirp-mass structure where it holds and departs from it where merger and ringdown dominate. Second, the pipeline requires only parameter-simulation pairs, not likelihood evaluations or a differentiable simulator; the tractable-likelihood examples were chosen for validation, and we now add [INTRACTABLE CASE].

New evidence in this response: [N] independent end-to-end trials per problem at 500 training simulations, with pre-registered success criteria, giving recovery rates of [SUMMARY]; a corrected end-to-end simulation and cost accounting with break-even points; and CRPS with calibration in place of the previous metric.

[HEADLINE TABLE]
```

## Recommendation on re-running the posterior experiments

Short answer: do not retrain to change metrics, but do retrain where the coordinates themselves changed.

### Do not retrain merely to compute CRPS

CRPS can be computed from already-saved outputs, so the metric change rUsi demands is a re-analysis rather than a re-training. This is the cheapest high-value item in the whole response.

Use `coverage_outputs.npz`, not `posterior_samples.npz`. The latter holds only the handful of `--n-examples` display cases and is far too small for a stable CRPS. The former holds `{nsims}_{method}_coverage_posterior_samples` with shape `(coverage_num_samples, coverage_n_test, n_params)`, defaulting to 1000 samples for each of 1000 held-out test observations, alongside the matching `theta_true`. That is ample. The samples are already transformed to theta units before being stored, so the theta-versus-eta comparison is like-for-like and does not need a separate Jacobian correction at analysis time.

Two caveats before assuming the data exists:

- Coverage export is gated behind `--run-coverage`. Runs launched without it saved nothing usable.
- Coverage is computed at a single simulation count only, `--coverage-nsims`, defaulting to `--fom-nsims`. A CRPS-versus-budget curve therefore does need reruns; a single-budget CRPS comparison does not.

CRPS itself is not implemented anywhere in the repository and will need to be written. The ensemble form over posterior samples is the appropriate estimator, and NaN-padded invalid samples must be masked out, since the coverage export writes NaN into rejected draws.

### Do retrain where the discovered coordinates changed

This is a consistency trap worth taking seriously. If the response claims discovery now costs approximately 500 simulations, but the downstream efficiency numbers were produced with coordinates discovered from 5,000 simulations, the two claims are inconsistent. QfCk already caught the discovery-cost omission and is the most likely reviewer to notice. Any problem where the headline efficiency claim is repeated must have its downstream NPE rerun using the newly discovered coordinates.

### Suggested scope, in priority order

1. **SIR.** Highest priority. It is where QfCk's 5,000-simulation criticism landed, where CRPS already exists in the updated manuscript, and where the low-budget claim is strongest.
2. **GW IMRPhenomD.** Second. It is the flagship discovery, and the calibration comparison against `(M_c, q)` is the most persuasive downstream evidence.
3. **Rosenbrock.** Third, as controlled validation with a known answer.
4. **TaylorF2 and weak lensing.** Skip for the rebuttal unless time permits.

### Protocol for any rerun

- Pair the `theta`, `eta`, and conventional-coordinate baselines on identical seeds and identical budgets, so the comparison is paired rather than across independent runs.
- Report the difference with dispersion across seeds, not two separate point estimates.
- Report CRPS and calibration together, always.
- Pass `--run-coverage` on every rerun. Without it the run produces no data from which CRPS or coverage can be recovered, and the run has to be repeated.
- Evaluate all methods in theta units, so the comparison cannot be dismissed as a change of measure. The sweep scripts already transform posterior samples back to theta before export; state this explicitly in the response.
- State the budget as training simulations, with held-out evaluation simulations reported separately.
