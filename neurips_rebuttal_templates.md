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
Problem        | Train sims | Recovered | Alignment (med, IQR) | Complexity
---------------|------------|-----------|----------------------|-----------
Rosenbrock     | 500        | [X]/10    | [VAL] ([LO],[HI])    | [VAL]
SIR            | 500        | [X]/10    | [VAL] ([LO],[HI])    | [VAL]
GW TaylorF2    | 500        | [X]/10    | [VAL] ([LO],[HI])    | [VAL]
GW IMRPhenomD  | 500        | [X]/10    | [VAL] ([LO],[HI])    | [VAL]
[INTRACTABLE]  | [N]        | [X]/10    | [VAL] ([LO],[HI])    | [VAL]
```

Follow it with one sentence of simulation accounting:

> Symbolic-regression augmentation used [N_AUG] evaluations of the learned coordinate map per run. These are network evaluations of a simulator-independent map and are not simulator calls. Held-out evaluation used a further [N_EVAL] simulations per run, reported separately.

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

**Scope: when is this needed?** The pipeline requires only parameter-simulation pairs. It needs neither likelihood evaluations, nor an analytic Fisher, nor a differentiable simulator. We selected examples with tractable likelihoods deliberately, because they allow discovered coordinates to be validated against known answers, but we accept this made the setting look circular. We will (i) state the requirement explicitly, and (ii) add an application with no evaluable likelihood [DESCRIBE; state precisely whether the likelihood is intractable or simply unavailable]. We would also emphasise that the discovered coordinates are themselves the deliverable: knowing which combination of parameters controls an observable is useful independently of the inference algorithm subsequently used.

**NPE is not assumed knowledge.** Agreed. We will add a short primer defining neural posterior estimation, the amortisation it provides, and why conditioning matters for it, before it is used.

**Metric justification.** This is a fair criticism and we are changing the evaluation. Maximum validation log-probability was used as a budget-matched training-quality proxy, but it does not establish posterior quality and can reward overconfidence. We now report CRPS together with calibration and coverage, so that sharpness cannot be claimed without calibration. On [PROBLEM] at a fixed budget of [N] training simulations, CRPS improves from [VAL_THETA] to [VAL_ETA], with coverage error [VAL_THETA] vs [VAL_ETA], averaged over [K] seeds.

**Size of the improvement.** With the corrected metrics the gap is [DESCRIBE HONESTLY], and is largest in the low-simulation and higher-noise regime, which is where we claim the method matters. Where the gain is modest we will say so in the revision.

**Comparison to likelihood-based inference.** We do not claim the method outperforms well-tuned gradient-based inference where a likelihood is available, and we will remove any implication otherwise. [OPTIONAL: report HMC/NUTS effective sample size per gradient in theta vs eta, if run.]
```

## Reviewer t9KB (Rating 5, Confidence 2)

Strongest supporter. Consolidate rather than argue.

```markdown
We thank Reviewer t9KB for recognising the novelty of combining simulation-based Fisher estimation with symbolic discovery.

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

**Q2 - A genuinely intractable likelihood.** We agree this was the key missing experiment. [DESCRIBE THE NEW APPLICATION: setup, what theta and x are, what was discovered, and the predefined success criterion. State precisely whether the likelihood is intractable or merely unavailable; do not overclaim.]

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
