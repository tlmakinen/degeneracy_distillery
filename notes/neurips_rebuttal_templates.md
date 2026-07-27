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

Each new result lives in the reply to the reviewer who asked for it, and nowhere else. The AC comment summarises what was added and points at the reply that contains it, rather than repeating the tables.

| Result | Belongs to | Because |
|---|---|---|
| Rosenbrock log-prob and CRPS | FoSE W2/Q1 | Asked what the 10x compares against |
| SIR CRPS with calibration | rUsi | Asked why max validation log-prob is a valid metric |
| Recovery rate out of 10 trials | t9KB Q2 | Asked about sensitivity and robustness |
| Chain-product scaling vs dimension | QfCk Q2 | Asked whether eta was discovered or supplied |
| Weak lensing / S_8 | Rkfx Q2, rUsi scope | Asked for a genuinely intractable likelihood |
| Prior and noise ablation | QfCk Q3 | Asked directly |

Formatting rules that matter inside the character limit:

1. **Keep tables narrow.** Five columns or fewer; OpenReview's markdown degrades on wide tables and nested formatting. Use a fenced plain-text block, which always renders.
2. **Use plain ASCII math.** Write `R_0 = beta/gamma`, `|Q - I|_F`, `M_c`, `Omega_m`.
3. **Always show denominators.** Report `9/10`, never "usually" or "robustly".
4. **Report dispersion.** Mean or median with a spread, across trials or observations.
5. **State the budget and the metric in every efficiency claim.** "500 training simulations, CRPS" rather than "500 simulations".
6. **Name the baseline once per table.** Same simulations, same architecture, same seeds, same held-out observations; only the coordinates differ.

## Shared building blocks

Reuse these verbatim so the framing is identical across all five replies and the AC sees one consistent story.

```markdown
**[REFRAME]** We thank the reviewer for the careful reading. The reviews have made clear that our framing obscured the paper's primary contribution. The distillery's main output is not a conditioning trick for NPE: it is the **automated discovery of the symbolic, nonlinear parameter combinations that actually control a simulator's observations**. Simulation efficiency is a downstream consequence, not the motivation. Our clearest instance is the gravitational-wave result (Sec. 5.2): the method recovers conventional chirp-mass structure in the inspiral-dominated TaylorF2 regime, and then, for IMRPhenomD where merger and ringdown dominate, autonomously returns a different controlling pair (total mass and an asymmetric mass combination) that is geometrically flatter and better calibrated than (M_c, q).

**[GEOMETRY]** We agree, and we will be precise about this throughout. Intrinsic curvature is measured by coordinate-independent invariants such as the Ricci scalar, and no reparametrisation can change it; Appendix D computes R = -1 for the Gaussian model and states that no globally flat coordinates exist. We do not claim to remove curvature. What we optimise is the **coordinate representation** of the metric: given a prior, we seek a single transformation, **global within that prior's support**, that minimises the expected distance between the Fisher matrix components and the identity. Two clarifications follow. First, "global" means one map covering the prior volume, as opposed to a local preconditioner fitted at a point; it does not mean globally flat. Second, a statistical degeneracy is a direction along which the metric has a small eigenvalue, which is a statement about sensitivity and not about Riemann curvature; our text conflated the two and we will fix it.

**[SCALING]** On dimensionality: symbolic regression is the **last** stage of the pipeline and does not run on the ambient parameter space. The Fisher spectrum is used first to identify the informative subspace, and symbolic recovery is attempted only inside it. Our chain-product example recovers a rank-1 manifold from ambient dimensions up to 12, and the symbolic step then acts on one coordinate rather than twelve. The demonstrated symbolic regime is therefore low-dimensional and we will state that plainly, but the reduction that precedes it is not limited in the same way.

**[METRIC-CLAIM]** We will narrow the Fisher claim in the main text. The Fishnet objective recovers the **inverse posterior covariance**; identification with the likelihood Fisher information holds only under the Laplace / Bernstein-von Mises conditions stated in Appendix A. We will state this at first mention (Sec. 4, currently lines 147-149) rather than only in the Discussion and appendix. The pipeline is unaffected: the pullback in Eq. (4) transforms identically for either object.
```

---

## Reviewer FoSE (Rating 2, Confidence 3)

The sharpest technical review. Winning the non-uniqueness point is the highest-value move available.

```markdown
We thank Reviewer FoSE for a thorough and technical review, and for noting the writing quality and appendices.

[REFRAME]

**W1 / Q3 - Non-uniqueness of isotropising coordinates.** We thank the reviewer for engaging with the non-uniqueness discussion in Appendix B, and we agree it belongs in the main text rather than an appendix; the revision will state it there. The reviewer's two-unit-Gaussian example is exactly the degenerate case Appendix B analyses: isotropy alone defines an equivalence class, since the loss is invariant under a constant offset and a global orthogonal transformation, so isotropy by itself cannot single out interpretable axes. Our claim is that the pipeline resolves this in two further stages, and both are load-bearing rather than cosmetic. (i) A canonicalisation fixes the frame: Jacobian Procrustes alignment, a nonlinearity-separating rotation that orders axes by nonlinearity energy, and a sign and permutation fix from the mean prior-normalised Fisher eigenstructure (Appendix B). (ii) An Occam selection: MDL-based symbolic regression returns the simplest closed-form representative of the class. We will restate the objective as "isotropy plus minimum description length" and present that pair as the discovery principle. Empirically the combination is reproducible: across 10 fully independent end-to-end trials per problem the expected coordinate is recovered in 10/10 (Rosenbrock), 10/10 (SIR), and 7/10 (IMRPhenomD) runs, evaluated numerically up to permutation, sign, and scaling rather than by string matching. The full table is in our reply to Reviewer t9KB.

**W2 / Q1 - What the 10x compares against.** The reviewer is right that the main text did not explicitly qualify the figure, and we will state it in the revision. The 10x is a **maximum validation log-probability** comparison on Rosenbrock, in matched theta-density units, so that both arms are densities over the same theta (the eta arm carries the log|det| correction). The baseline is NPE trained on the same simulations in the original parameters, with identical architecture, ensemble size, seeds, and held-out observations; only the coordinates the density is fitted in differ.

Best validation log-probability, theta-density units, mean +- sd over 2 ensemble members:

    N_sim | NPE in theta   | NPE in eta
    ------|----------------|---------------
    100   | -3.88 +- 0.15  | -2.18 +- 0.22
    1000  | -2.35 +- 0.09  | -1.07 +- 0.06
    5000  | -1.43 +- 0.08  | -0.51 +- 0.12
    10000 | -1.03 +- 0.00  | -0.21 +- 0.04

Read as a horizontal shift, eta at 100 simulations matches theta at approximately 1,360 (13.6x) and eta at 1,000 matches theta at approximately 9,260 (9.3x). Because a log-probability factor speaks to density fitting rather than to posterior quality, we now also report CRPS, and we will label every factor with its metric. Scored with CRPS against 1,000 held-out observations at a matched budget of 1,000 simulations, the same run gives 0.997 (theta) against 0.830 (eta) summed over parameters, a factor of 1.20, with better calibration in eta (PIT maximum deviation 0.038 and 0.070 against 0.064 and 0.114). On SIR the corresponding CRPS factor is approximately 1.9 (details in our reply to Reviewer rUsi). The revision will therefore quote approximately 2x for posterior quality and 9-14x for validation log-probability, each with its metric, budget, and baseline named.

**W3 / Q1 - Computational cost and scaling.** [SCALING] End-to-end discovery wall time is 9.3-21.0 min per trial on a single V100 across the four problems; for Rosenbrock, the most expensive, 262 s (Fisher ensemble) + 365 s (coordinates) + 654 s (symbolic regression) = 21.0 min. We accept that "real-world" overstated the demonstrated scope and will restrict the claim to the parameter dimensionalities we actually show.

**W4 - Imprecise claims.** We accept all three.
- "requiring no data" becomes "requiring no observed data realisation; the metric is estimated from simulations", in both the abstract and line 91.
- The Gaussian/Laplace assumption will be stated at first mention. [METRIC-CLAIM]
- On beta/gamma: the discovered first component has held-out correlation [VALUE] with R_0 = beta/gamma and gradient cosine similarity [VALUE]. If that is not compelling we will replace "closely" with the number itself.

**Minor points.** (1) [GEOMETRY] (2) We will cite Amari, *Information Geometry and Its Applications*, Applied Mathematical Sciences 194, Springer, 2016, for the information-geometric framing. (3) Agreed that symbolic post-processing could in principle be applied to other methods; we will say so, while noting that the MDL step is what resolves the non-uniqueness above rather than a presentational choice. (4)-(5) NPE and TARP will be defined at first use, with one line on what TARP measures. (6) Figures 4 and 5, all subpanels, are experimental results; Figure 1 is a pipeline schematic and will be labelled as such.

Thank you again for such a careful assessment. Please let us know if any of the above needs further clarification.
```

Reviewer's own words, for reference while drafting:

> Non-uniqueness of coordinates - To me, the main motivation for identifying coordinates in which the Fisher matrix is approximately isotropic is flawed. Such coordinates are not unique. The authors discuss non-uniqueness, but only in Appendix B in the context of the learning objective defined in Equation (6). To me, the problem is not only learning dynamics or convergence of the learning objective, but the existence of coordinates we wish to find. Suppose, our system is two 1D independent Gaussians with unit variance. The Fisher Matrix is an identity in this example. But it remains an identity for any orthogonal transformation of these coordinates. Since our objective is to "interpretable reparametrisations that align with the true directions of a sensitivity of the model", then I think that optimizing for an isotropic Fisher Matrix fundamentally does not solve this problem.

> Lack of clear comparison with prior work - Authors make claims that their method "requires 10x fewer simulations", but there are no quantitative comparisons in the main text. It is unclear what the authors are comparing to, at least not from the main text.

> Computational cost - this methods appears to be very expensive and is probably infeasible for high-dimensional systems, arguably systems, where it's the most desirable to identify interpretable directions. Therefore, I see this method as only applicable to very small systems. In this context, I think that saying that the method is applicable to "real-world" datasets and "real-world" problems can be misleading. I would clarify that this only holds (and has only been demonstrated) for problems of very low dimensionality.

---

## Reviewer rUsi (Rating 2, Confidence 4)

```markdown
We thank Reviewer rUsi for a clear statement of where the paper failed to justify its setting.

[REFRAME]

**Scope: when is this needed?** The pipeline requires only parameter-simulation pairs. It needs neither likelihood evaluations, nor an analytic Fisher, nor a differentiable simulator. Several of our examples were chosen with tractable likelihoods deliberately, because they let the discovered coordinates be checked against a known answer, and we accept that this made the setting look circular. We will state the requirement explicitly and lead with the case that does not have this property. Our weak-lensing example is simulation-based with no tractable likelihood: the data are two-point statistics of mock convergence images computed from expensive dark-matter simulations, varied over wide priors in (Omega_m, sigma_8) and in the initial conditions. There the method recovers

    eta_1 = Omega_m sigma_8 - 0.9 Omega_m^0.144
    eta_2 = 0.5 (Omega_m - 1.0)

which is close to the standard (Omega_m, S_8) parameterisation used in the field, with S_8 = sigma_8 (Omega_m/0.3)^0.5. The nonlinear exponent is found from simulations alone, without a likelihood and without being told that a damping of the Omega_m dependence is what resolves the degeneracy.

**NPE is not assumed knowledge.** Agreed. We will add a short primer defining neural posterior estimation, the amortisation it provides, and why conditioning matters for it, before it is used.

**Metric justification.** A fair criticism, and we are changing the evaluation. Maximum validation log-probability is a budget-matched training-quality proxy that can reward overconfidence, so it cannot carry the argument alone. We now report CRPS on a derived physical quantity together with calibration. On SIR, over 50 held-out observations with a 5-member ensemble, all arms evaluated in theta units on the same observations:

    N_sim | Method | CRPS(R_0), mean (95% CI) | PIT max dev | 68% cov
    ------|--------|--------------------------|-------------|--------
    500   | theta  | 0.228 (0.191, 0.270)     | 0.144       | 0.88
    500   | eta    | 0.176 (0.141, 0.216)     | 0.107       | 0.78
    1000  | theta  | 0.174 (0.144, 0.210)     | 0.216       | 0.86
    1000  | eta    | 0.168 (0.128, 0.214)     | 0.082       | 0.74
    5000  | theta  | 0.141 (0.104, 0.183)     | 0.091       | 0.72
    5000  | eta    | 0.142 (0.105, 0.184)     | 0.061       | 0.72

The comparison is paired: at 500 simulations eta is better on 42/50 observations (Wilcoxon p < 1e-4, mean paired difference 0.052 with 95% CI 0.019-0.086). Crucially the improvement is not bought with overconfidence, which is the failure mode the reviewer is right to suspect: at the same budget eta has the smaller PIT deviation and the 68% coverage nearer nominal, so it is the theta posteriors that are over-dispersed.

**Size of the improvement.** The range we quoted mixed two metrics, which we will separate. On validation log-probability the factor is 9-14x (Rosenbrock; see our reply to Reviewer FoSE). On CRPS it is approximately 2x in simulations at the low-budget end: eta's 500-simulation CRPS is matched by theta at approximately 970. We regard the CRPS figure as the one that answers the reviewer's question and will lead with it. We will also show that the advantage closes by 2,000 simulations (0.149 in both), rather than reporting only the regime where we win. That is the expected behaviour for a reparameterisation: it helps when simulations are scarce and the noise is high, and stops mattering once the baseline has enough data to learn the degeneracy itself.

**Comparison to likelihood-based inference.** We do not claim the method outperforms well-tuned gradient-based inference where a likelihood is available, and we will remove any implication otherwise.
```

Reviewer's own words, for reference while drafting:

> Strengths: The idea to reparameterize a model to make an approximate inference procedure more computationally efficient is sensible. The applications and some aspects of the figures illustrating results of them help to clarify the procedure and the success at improving the conditioning of the Fisher information, and the beneficial downstream impacts of this reparameterization.

> Weaknesses: The scope of applications on which the method may have utility is not made clear, and the overall inference pipeline is not clearly explained. Namely, the procedure seems not to be useful if the Fisher information is computable analytically, or even perhaps model likelihoods.

> Evaluations are compared in the setting of neural posterior estimation. This is not a standard statistical procedure with which I (or presumably a majority of relevant members of the NeurIPS community) will be closely familiar with. So this procedure and its scope should be explained rather than assumed. At present, it is unclear if the contribution is limited to providing improvements is settings when one would otherwise use a NPE, which presumably suffers more dramatically from poor conditioning than maximum likelihood inference by convex optimization (for example) when this is possible.

> The demonstrated improvement over NPE without reparamaterization seems small, and the validity of primary metric shown (max value of log p(x|theta)) as a proxy for better performance should be explained and defended. I do not see why this is a good measure for success

---

## Reviewer t9KB (Rating 5, Confidence 2)

Strongest supporter. Consolidate rather than argue. The 10-trial table is the centrepiece here.

```markdown
We thank Reviewer t9KB for recognising the novelty of combining simulation-based Fisher estimation with symbolic discovery.

**Q2 - Sensitivity and robustness of the recovered expressions.** This is our main new result. We ran 10 fully independent end-to-end trials per problem, varying the training simulations, Fisher-ensemble seed, flattener seed, alignment subsampling, and symbolic-regression seed together, rather than only the network initialisation. Success criteria were fixed before running and are evaluated numerically on held-out points, comparing the coordinate functions and their Jacobians up to permutation, sign, scaling, and algebraic equivalence; string matching is not used, because algebraically equivalent forms and the residual orthogonal freedom make literal comparison meaningless.

    Problem        | Train sims | Recovered | Alignment (med, IQR) | Complexity
    ---------------|------------|-----------|----------------------|-----------
    Rosenbrock     | 500        | 10/10     | 0.929 (0.814,0.960)  | 28
    SIR            | 500        | 10/10     | 0.674 (0.636,0.741)  | 17
    GW TaylorF2    | 500        | 10/10     | 0.972 (0.966,0.979)  | 24
    GW IMRPhenomD  | 500        | 7/10      | 0.997 (0.988,0.999)  | 25

Alignment is |Pearson r| on held-out points against the expected coordinate (the valley coordinate; R_0 = beta/gamma; chirp mass M_c; total mass M). All three IMRPhenomD non-recoveries are threshold misses on the second conjunct of its criterion, not crashes: the pipeline ran cleanly on all 10 trials and the median total-mass alignment is 0.997, while the complementary asymmetric mass combination is the harder and more variable direction. Failures stay in the denominator and we identify the stage that failed in each. Symbolic-regression augmentation used 2,000 evaluations of the learned coordinate map per run (7,509-7,632 for SIR, which augments after a validity cut); these are network evaluations of a simulator-independent map, not simulator calls, and are reported separately from the 500 held-out simulations used for evaluation. [IF RUN: we repeated the SR stage with [ALTERNATIVE ALGORITHM] and obtained [RESULT].]

**Q3 - Total computational overhead.** Agreed, and doing the accounting end-to-end changes our claim. For SIR, discovery costs 500 training simulations of the two-parameter (beta, gamma) system, plus 221 s (Fisher ensemble) / 174 s (coordinates) / 130 s (symbolic regression), 9.3 min end to end on one V100. Those simulations hold I_0 fixed, so they cannot double as NPE training data for the three-parameter inference problem: a single downstream analysis in eta costs 500 + 500 = 1,000 simulations to match a CRPS(R_0) that the theta baseline reaches with approximately 970. For one analysis the coordinates are therefore a wash, and we will say so. Their value is amortisation, since the map is learned once: for k analyses eta costs 500 + 500k against the baseline's 970k, which is net-positive from the second analysis (1,500 against 1,940) and approaches 1.9x. Separately, discovery cost itself fell from 5,000 to approximately 500 simulator calls per experiment (Rosenbrock, GW, SIR) by augmenting only the symbolic-regression stage with inexpensive evaluations of the learned map.

**Weakness - baselines.** We agree a comparison on the parameter-estimation component alone is appropriate. [STATE WHAT YOU CAN RUN, e.g. FMPE/CMPE at matched budget on PROBLEM giving RESULT; or state honestly what is out of scope for the rebuttal window and will appear in the revision.]

Thank you again for your thoughtful and thorough review.
```

---

## Reviewer QfCk (Rating 5, Confidence 2)

Precise technical asks. Q2 is a direct factual question; answer it plainly.

```markdown
We thank Reviewer QfCk for a precise reading of the first stage; both concerns raised are correct.

**Wording: Fisher vs inverse posterior covariance.** [METRIC-CLAIM]

**Q1 - Where is the approximation good?** [ANSWER PER EXPERIMENT. For each of Rosenbrock, Gaussian, SIR, GW, WL, state whether the posterior is close to Gaussian at the relevant simulation count and prior, and where it is not. Be explicit that in the non-Gaussian cases the recovered object should be read as an inverse posterior covariance, and that the flattening pipeline is unchanged either way.]

**Q2 - Chain-product experiment: discovered or supplied?** Supplied, and we should have said so. In the submitted scaling sweep the distilled coordinate is the analytic standardised log-product, not a discovered expression; only the d = 2 case was taken through the full fishnets-to-symbolic pipeline. The experiment as run therefore supports a narrower claim than we implied: **given** the correct one-dimensional coordinate, downstream inference is insensitive to ambient dimension, whereas inference in the original parameters degrades. That claim the sweep does support, on the common eta axis, with 1,000 training simulations and 3 trials per point:

    d  | NPE in theta   | NPE in eta (analytic)
    ---|----------------|----------------------
    2  | 0.69 +- 0.10   | 0.60 +- 0.13
    4  | 1.56 +- 0.17   | 1.33 +- 0.09
    6  | 1.76 +- 0.12   | 1.62 +- 0.15
    8  | 1.19 +- 0.26   | 1.66 +- 0.02
    10 | 0.90 +- 0.34   | 1.39 +- 0.34
    12 | 0.14 +- 0.94   | 1.54 +- 0.21

Both arms are scored as marginal log-probability on the same one-dimensional axis, so the columns are comparable within a row. The theta arm peaks near d = 6 and then collapses to 0.14 at d = 12, while the eta arm stays within 1.33-1.66 from d = 4 onward. Two caveats we will state in the revision: this run used a MAF for the theta arm and an MDN for the eta arm, so architecture is confounded with coordinates, and the eta coordinate is the analytic one. We are rerunning the sweep with discovery performed independently at every d and with the same density estimator in both arms, which is the experiment that would support the stronger claim; we will report it in the revision whether or not it succeeds at the larger d. What the pipeline does already demonstrate at every d is the reduction step itself: the Fisher spectrum identifies the rank-1 informative subspace, which is the prerequisite for the symbolic stage. [SCALING]

**Q3 - Sensitivity to prior range and noise model.** We ran an ablation varying the prior width by [FACTORS] and the noise level over [RANGE] on [PROBLEM]. The recovered structure was [STABLE/CHANGED] as follows: [RESULTS]. [IF THE COORDINATES CHANGE WITH NOISE, FRAME THIS AS EXPECTED: the informative directions are a property of the model together with the measurement, so a changed noise model legitimately changes which combination controls the data.]

**One-time discovery cost and break-even.** We accept this was omitted, and we now separate (i) discovery training calls, (ii) held-out evaluation calls, (iii) inexpensive evaluations of the learned map, and (iv) downstream NPE simulations. Because eta = f(theta) is simulator-independent once learned, we augment only the symbolic-regression stage with fresh theta drawn from the prior; those are network evaluations rather than simulations, and they bring SIR discovery from 5,000 to approximately 500 simulator calls. Charging discovery honestly, one downstream SIR analysis in eta costs about 1,000 simulations against the baseline's 970, which is neutral, and the coordinates pay for themselves only through reuse: net-positive from the second analysis, approaching 1.9x. This accounting, including the neutral single-analysis case, goes into the Limitations discussion. Our reply to Reviewer t9KB gives the same numbers in the context of total overhead.
```

---

## Reviewer Rkfx (Rating 2, Confidence 4)

Hardest review. Note that their Q3 is already answered by Appendix D.

```markdown
We thank Reviewer Rkfx for pressing on the geometric formulation; the criticism of our language is correct and we are grateful for it.

[GEOMETRY]

**On the apparent contradiction.** The reviewer is right that we wrote something inconsistent. We never intended to claim that reparametrisation removes curvature, and Appendix D proves the opposite for the Gaussian; the inconsistency is in our prose, and the objective statement will be rewritten so that it is unambiguous.

**Q3 - Conformally flat case.** Our Gaussian validation example is exactly the analytically tractable, non-trivial, conformally flat case the reviewer asks for. In the coordinates u = mu/sigma_star and v = sqrt(2) sigma/sigma_star the metric is g_ab = (2/v^2) delta_ab, that is, conformally Euclidean, with Ricci scalar R = -1, the hyperbolic plane; this is derived in Appendix D. Because the curvature is non-zero, no coordinates make the components equal to the identity everywhere, so the correct benchmark is the average deviation over the prior, which is what we optimise. We compare our learned and symbolic coordinates against geodesic normal coordinates about a fiducial point and against the ad hoc (mu/sigma, log sigma) choice: the geodesic coordinates are exact at the fiducial point but degrade with geodesic distance, while the discovered coordinates achieve lower average deviation across the prior. We will promote this from a validation example to the explicit answer to this question.

**Q2 - A genuinely intractable likelihood.** Our weak-lensing example is one, and we agree it was not presented as such. The data are two-point statistics of mock convergence images computed from expensive dark-matter simulations, varied over wide priors in (Omega_m, sigma_8) and in the initial conditions. No tractable likelihood is available for these maps; the pipeline sees only parameter-simulation pairs. It recovers

    eta_1 = Omega_m sigma_8 - 0.9 Omega_m^0.144
    eta_2 = 0.5 (Omega_m - 1.0)

against the field-standard S_8 = sigma_8 (Omega_m/0.3)^0.5, which is the combination cosmologists use precisely because it damps the Omega_m dependence of the two-point signal. The nonlinear exponent and the damping structure are recovered from simulations alone. We will be careful in the revision to distinguish "no tractable likelihood for the observable" from "no likelihood at all", and we will not overclaim beyond the former. [IF RUN: we additionally report [INTRACTABLE CASE], where [RESULT].]

**Q1 - Usefulness alongside well-tuned likelihood-based inference.** We accept that for the submitted case studies a practitioner with a tractable likelihood would reasonably use HMC with a tuned mass matrix, and we will not claim otherwise. Two points remain. First, a tuned mass matrix is a local, linear preconditioner fitted per posterior, whereas the distillery returns a map that is nonlinear, amortised, and global within the prior in the sense defined above, reusable across observations. Second, and primarily, the deliverable is the discovered coordinate itself: the IMRPhenomD result identifies a controlling parameter combination that differs from the conventional chirp-mass basis, which is a statement about the physics rather than about a sampler. [IF RUN: we also compare NUTS in theta vs eta, giving ESS per gradient of [VALUE] vs [VALUE] and condition number [VALUE] vs [VALUE].]
```

---

## Top-level comment to the AC

The meta-review says rejection is likely "unless the reviewers' concerns reflect correctable misunderstandings". Address that sentence directly, summarise, and point at the individual replies rather than repeating their tables.

```markdown
We thank the AC for the summary. We reply to each reviewer individually; this comment states what we accept, what we have added, and where each new result sits.

**What we accept and will implement.** The imprecise "flattening" language: intrinsic curvature is a coordinate-independent invariant and cannot be removed, and what we optimise is the coordinate representation of the metric, global within a specified prior. The overstated identification of the Fishnet output with the likelihood Fisher rather than the inverse posterior covariance. The "requiring no data" phrasing. The omission of the one-time coordinate-discovery cost from efficiency claims. And the use of maximum validation log-probability as the primary success metric. On this last point we owe an explanation rather than a retraction: the 10x figure is a validation log-probability factor measured on Rosenbrock, 9-14x in that metric, which the main text failed to attribute. Scored instead on posterior quality with CRPS against held-out observations, the same runs give approximately 2x. The revision reports both, each labelled with its metric, budget, and baseline.

**New results in this response, by reply.** To FoSE: the Rosenbrock log-probability and CRPS comparison that substantiates the efficiency claim, and the baseline definition. To rUsi: CRPS with PIT calibration and coverage for SIR at three budgets, paired across 50 held-out observations. To t9KB: recovery rates over 10 fully independent end-to-end trials per problem, with pre-registered numerical success criteria. To QfCk: a direct answer that the chain-product scaling sweep used the analytic coordinate rather than a discovered one, the corrected claim it supports, and the rerun now in progress. To Rkfx: the weak-lensing case presented properly as the intractable-likelihood example, and the conformally flat Gaussian as the explicit answer to their Q3.

**On two concerns we believe are framing failures rather than substance.** First, the paper's contribution is the discovery of the symbolic nonlinear coordinates that control a simulator's observations, with simulation efficiency as a consequence; our gravitational-wave result recovers conventional chirp-mass structure where it holds and departs from it where merger and ringdown dominate. Second, the pipeline requires only parameter-simulation pairs, not likelihood evaluations and not a differentiable simulator; the tractable-likelihood examples were chosen for validation against known answers, and the weak-lensing case has no tractable likelihood.

**On scope for this venue.** One review suggests that neural posterior estimation is unfamiliar to most of the NeurIPS community and that the setting should therefore be justified. We are happy to add the primer, and we will. We would gently note, though, that simulation-based inference is a long-established NeurIPS topic rather than an import from another field: the two papers that founded the modern neural-posterior line appeared at NIPS 2016 ("Fast epsilon-free Inference of Simulation Models with Bayesian Conditional Density Estimation") and NIPS 2017 ("Flexible statistical inference for mechanistic models of neural dynamics"), and the line is unbroken since, with five papers carrying simulation-based or likelihood-free inference in the title at NeurIPS 2023, two at 2024, and four at 2025, including work on simulation efficiency, calibration diagnostics, and misspecification. Counting titles undercounts the area substantially. We raise this only because the framing of the concern, that the topic itself may be out of scope, differs from the reviewer's substantive and fair request that we explain the pipeline more carefully.
```

---

# Internal notes: measurements, provenance, and open items

Everything below is for us, not for the response. Numbers quoted in the replies above are traceable to these sections.

## Measured SIR numbers (verified 27 Jul 2026)

Reproduced from the code path behind the revised paper's banner figure (`make_sir_banner_r0` in `plots/SIR_NEW_RESULTS.ipynb`, reading `results/sweep_results/sir_sweep_results_NEW_BATCH/posterior_samples.npz`, mean aggregator, 2,000-resample bootstrap), so the rebuttal and the figure cannot disagree.

### Setup as actually run

- Discovery: 500 simulations, two parameters (beta, gamma), noise sd 0.01, I_0 fixed at the simulator default. This is the run described in the paper; a separate noise-0.05 discovery run exists as a robustness check but did not produce the map used downstream.
- Inference: three parameters (beta, gamma, I_0/10), noise sd 0.05, fresh nested pool of up to 5,000 simulations (`training_data: null` in the manifest, budget N uses the first N rows), 5-member MAF ensemble, adaptive batch size, 50 held-out observations with 10,000 posterior draws each.
- Both arms are evaluated in theta units on the same observations. In the eta arm approximately 2% of draws land outside the theta prior after the inverse map and are dropped as NaN.

### CRPS(R_0) versus budget

```text
N_sim | CRPS theta | CRPS eta | ratio
------|------------|----------|------
100   | 0.262      | 0.257    | 1.02
200   | 0.388      | 0.171    | 2.27
500   | 0.228      | 0.176    | 1.29
1000  | 0.174      | 0.168    | 1.04
2000  | 0.149      | 0.149    | 1.00
5000  | 0.141      | 0.142    | 0.99
```

Read as a horizontal shift, eta's 500-simulation CRPS of 0.176 is matched by the theta curve at approximately 970 simulations (log-interpolated between the 500 and 1000 points), i.e. 1.9x; using medians instead of means gives 2.5x.

Paired per-observation differences (theta minus eta, 50 shared observations): at 500 simulations the mean difference is 0.052 (95% bootstrap CI 0.019-0.086) with eta better on 42/50 and Wilcoxon p < 1e-4; at 1000, eta is better on 35/50 (p = 0.032) but the CI on the mean includes zero; at 2000 and 5000 there is no detectable difference.

### Three things not to claim

1. **Do not read 10x off this plot.** The theta baseline reaches 0.141 at 5,000 simulations and eta does not beat that until 5,000 itself, so the "eta at 500 equals theta at 5,000" reading is false. On CRPS the supportable factor is approximately 2x; the 10x lives in the Rosenbrock log-probability sweep and must be attributed to that metric.
2. **Do not lean on the 200-simulation point.** The theta value there (0.388) is worse than at 100 simulations (0.262), which is an ensemble-training instability, not signal. It produces the largest apparent ratio on the plot (2.27x) and is the easiest number for a reviewer to discredit.
3. **The dashed line is not free.** It marks the discovery budget on the same axis, but those simulations are two-parameter with I_0 fixed and therefore cannot be reused as three-parameter NPE training data. Total-cost break-even for a single analysis is neutral; the claim must be amortisation across analyses.

### Calibration at the decisive budget

PIT maximum deviation and empirical 68% coverage of R_0, computed from the 50-observation export (nominal 0.68):

```text
N_sim | theta: PIT dev / cov68 | eta: PIT dev / cov68
------|------------------------|---------------------
500   | 0.144 / 0.88           | 0.107 / 0.78
1000  | 0.216 / 0.86           | 0.082 / 0.74
5000  | 0.091 / 0.72           | 0.061 / 0.72
```

Both arms over-cover, theta more so, so eta is sharper *and* nearer nominal. Weak point: `coverage_outputs.npz` holds only the 5,000-simulation case (`fom_nsims`), where the arms are already equivalent, so these 500-simulation calibration numbers rest on 50 observations rather than the 1,000 a coverage export would give.

### Contingency: making the reuse literal

Worth doing if the response needs a clean one-shot efficiency number rather than an amortised one. This is now automated in `scripts/sir_discovery_3d_rerun.py`, which is standalone: it does not import or modify either `sir_notebook_run.py` or `sir_nsims_logprob_sweep.py`.

What it does: runs the full fishnets -> flatten -> align -> augment -> SR pipeline over all three parameters `(beta, gamma, I_0/10)` at noise 0.05, writes the simulation pool it used with the discovery simulations as the first `--n-discovery` rows, and emits a ready-to-run sweep command with the matching scaler bounds and both coordinate directions. Sweep budgets at or above the discovery budget then consume nothing beyond what discovery already paid for, so the horizontal shift becomes a simulation-efficiency statement outright.

```bash
# cluster: discovery then sweep in one job
MODE=rebuttal SEED=0 RUN_SWEEP=1 DISJOINT_CHECK=1 \
  sbatch --time=24:00:00 scripts/slurm_sir_discovery_3d.sh

# ten independent discoveries for a recovery-rate table
for s in $(seq 0 9); do MODE=rebuttal SEED=$s sbatch scripts/slurm_sir_discovery_3d.sh; done
```

Four things it handles that matter for the rebuttal:

- **The inverse map.** The sweep needs closed-form `eta -> theta` expressions to push posterior samples back to theta units. The script tries `sympy.solve` under a timeout, scores the result by round-trip error both inside and outside the training box, and falls back to a verified polynomial surrogate when sympy fails or hangs. It also reports the fraction of out-of-box draws where the inverse goes non-finite, which is what silently shrinks the eta arm's effective posterior sample count.
- **The double-dipping control.** `--disjoint-check` rediscovers on the next block of the pool and reports agreement with the primary coordinates numerically, matched up to permutation and sign by absolute correlation on held-out theta, rather than by comparing expression strings. String comparison is the fragile criterion FoSE objected to and is not used.
- **Distributional compatibility.** The simulator is copied from the sweep verbatim, including RNG call order, and `--verify-simulator` cross-checks the two bitwise whenever ltu-ili is importable. If the copy ever drifts, the run aborts rather than quietly producing a pool the sweep would not have drawn.
- **The scaler contract.** The sweep rebuilds the affine theta map from `--scaler-data-min/--scaler-data-max`, so those bounds are emitted in the command, and the fraction of the full pool falling outside the resulting `[1, 2]` box is reported (a large excursion would handicap the theta baseline, whose MAF prior is that box).

Caveat to disclose either way: with literal reuse the same simulations train the coordinate map and the density estimator. The `--disjoint-check` agreement numbers are what to quote when a reviewer raises it.

## Measured Rosenbrock numbers (verified 27 Jul 2026)

From `results/sweep_results/rosen_sweep_results` (seed 0, 2-member MAF ensemble, 1,000 held-out test simulations, `fom_nsims` 1000, coverage exported at 1000 only). This is the origin of the "10x fewer simulations" claim, and it is a log-probability claim only.

### Best validation log-prob, theta-density units

The `_theta_density` columns are the comparable ones; the eta arm carries a constant log|det| correction of 0.0393.

```text
N_sim | theta            | eta              | delta (nats)
------|------------------|------------------|-------------
100   | -3.883 +- 0.150  | -2.175 +- 0.222  | +1.708
500   | -2.308 +- 0.056  | -1.971 +- 0.388  | +0.337
1000  | -2.351 +- 0.092  | -1.070 +- 0.063  | +1.281
5000  | -1.431 +- 0.080  | -0.506 +- 0.115  | +0.925
10000 | -1.025 +- 0.000  | -0.210 +- 0.036  | +0.815
```

Horizontal shifts: eta at 100 is matched by theta at approximately 1,360 (13.6x); eta at 1,000 by theta at approximately 9,260 (9.3x); eta at 5,000 and 10,000 is never matched within the swept budget. Caveats: only 2 ensemble members, and theta is non-monotonic between 500 and 1,000 (-2.308 to -2.351), the same instability the SIR sweep shows at 200.

### The same run scored with CRPS, at matched budget 1,000

Computed from `coverage_outputs.npz` (1,000 observations, 1,000 draws each), paired on identical observations:

```text
Quantity          | theta         | eta           | ratio
------------------|---------------|---------------|------------------
CRPS(mu_0)        | 0.253         | 0.245         | 1.03x (CI incl. 0)
CRPS(mu_1)        | 0.744         | 0.585         | 1.27x
CRPS summed       | 0.997         | 0.830         | 1.20x
PIT max dev       | 0.064 / 0.114 | 0.038 / 0.070 | eta better
Empirical 68% cov | 0.581 / 0.523 | 0.630 / 0.598 | eta nearer nominal
```

The mu_1 gain is paired and significant (mean difference +0.159, 95% CI 0.132-0.186, eta better on 640/997), and eta is better calibrated on both parameters, so the direction survives in both metrics. The magnitude is metric-dependent: a 1.28-nat log-probability gap at 1,000 simulations corresponds to a 1.20x CRPS improvement, not the 9.3x that the log-probability horizontal shift implies. This is rUsi's objection made concrete, so every efficiency factor must carry its metric: 9-14x is a log-probability number, approximately 2x is a posterior-quality number. A CRPS-based horizontal shift for Rosenbrock cannot be computed from this run at all, because coverage was exported at a single budget.

Also note the eta arm loses 15.7% of its draws to the theta prior support after the inverse map (against about 2% for SIR), which is high enough to check before quoting anything from this run.

### Analysis trap in the coverage archive

`coverage_outputs.npz` stores a top-level `theta_true` that is *already* `theta_test[coverage_indices]`, identical to the per-method `n{nsims}_{method}_coverage_theta_true`. Indexing it again with `coverage_indices` silently scrambles the pairing and produces plausible-looking nonsense (CRPS inflated roughly sevenfold, 68% coverage collapsing to 0.12). Always read the per-method `..._coverage_theta_true` array.

## Measured heater / chain-product numbers (verified 27 Jul 2026)

From `results/heater_dim_scaling_MDN` (`metrics_aggregate.csv` plus `manifest.json`): dims 2-12, `nsims` 1000, 3 trials per point, `raw_model: maf`, `distilled_model: mdn`, `distilled_coord: standardised_log_product`.

The manifest settles QfCk's Q2 unambiguously: `distilled_coord` is the analytic standardised log-product, so the swept eta is supplied, not discovered. Only the d = 2 case went through the full pipeline (`scripts/heater_minimal_distillery.py`).

Marginal log-probability on the common eta axis, which is the only cross-arm comparable column (`raw_marg_eta_mean` against `dist_marg_eta_mean`):

```text
d  | theta (MAF)   | eta analytic (MDN)
---|---------------|-------------------
2  | 0.687 +- 0.105| 0.596 +- 0.126
4  | 1.564 +- 0.175| 1.327 +- 0.087
6  | 1.760 +- 0.123| 1.623 +- 0.146
8  | 1.193 +- 0.262| 1.659 +- 0.022
10 | 0.897 +- 0.342| 1.392 +- 0.336
12 | 0.143 +- 0.944| 1.544 +- 0.212
```

Own-space columns (`raw_mean` 1.748 -> 0.203, `dist_mean` 0.738 -> 2.021 from d = 2 to 12) are not comparable across arms and should not be quoted as a head-to-head.

Two honest caveats to carry into any claim:

- **Architecture is confounded.** The theta arm is a MAF and the eta arm an MDN in this run. `scripts/heater_discovery_dim_scaling_sweep.py` uses MDNs in both arms specifically to remove this.
- **No discovery-at-each-d results exist yet.** There is no `results/*discovery*` directory; the rerun script is committed but has not produced output. Until it does, QfCk's answer must be the narrower claim in the reply above.

## Recommendation on re-running the posterior experiments

Short answer: do not retrain to change metrics, but do retrain where the coordinates themselves changed.

### Do not retrain merely to compute CRPS

CRPS can be computed from already-saved outputs, so the metric change rUsi demands is a re-analysis rather than a re-training. This is the cheapest high-value item in the whole response.

Prefer `coverage_outputs.npz` over `posterior_samples.npz` where both exist, but note that the SIR run actually used in the paper inverts this: its coverage export covers only the 5,000-simulation case, whereas `posterior_samples.npz` covers every budget with 50 observations and 10,000 draws each, which proved sufficient for a paired comparison at p < 1e-4. `coverage_outputs.npz` holds `{nsims}_{method}_coverage_posterior_samples` with shape `(coverage_num_samples, coverage_n_test, n_params)`, defaulting to 1000 samples for each of 1000 held-out test observations, alongside the matching `theta_true`. The samples are already transformed to theta units before being stored, so the theta-versus-eta comparison is like-for-like and needs no separate Jacobian correction at analysis time.

Two caveats before assuming the data exists:

- Coverage export is gated behind `--run-coverage`. Runs launched without it saved nothing usable.
- Coverage is computed at a single simulation count only, `--coverage-nsims`, defaulting to `--fom-nsims`. A *calibration*-versus-budget curve therefore needs reruns. A CRPS-versus-budget curve does not: `posterior_samples.npz` stores every budget, and that is how both the paper figure and the numbers above were produced.

CRPS exists in two places but not as a shared utility: `empirical_crps_batch` in `scripts/sir_notebook_run.py` (O(n log n) order-statistic form) and `_crps_1d` inside the plotting functions in `plots/SIR_NEW_RESULTS.ipynb` (pairwise form, used for the paper figure). Both use the ensemble estimator over posterior samples and both mask non-finite draws, which is required because the exports write NaN into rejected draws. Promoting one of them into the package would remove the risk of the rebuttal and the figure disagreeing.

### Do retrain where the discovered coordinates changed

If the response claims discovery now costs approximately 500 simulations, but the downstream efficiency numbers were produced with coordinates discovered from 5,000 simulations, the two claims are inconsistent. QfCk already caught the discovery-cost omission and is the most likely reviewer to notice.

### Suggested scope, in priority order

1. **Chain-product / heater.** Now the highest priority, because QfCk's Q2 has been answered with the narrower claim and the stronger one needs the discovery-at-each-d rerun. Same-architecture arms are required.
2. **SIR.** Largely settled: the downstream sweep already uses the 500-simulation coordinates, so it is consistent with the reduced discovery cost, and the CRPS numbers need no rerun. Remaining reasons to touch it are calibration at 500 simulations (currently 50 observations) and the literal-reuse contingency, both one sweep with `--run-coverage --coverage-nsims 500`.
3. **GW IMRPhenomD.** The flagship discovery; the calibration comparison against `(M_c, q)` is the most persuasive downstream evidence.
4. **Rosenbrock.** Controlled validation with a known answer. A second coverage budget would let us quote a CRPS-based horizontal shift instead of only a matched-budget ratio.
5. **TaylorF2 and weak lensing.** Skip unless time permits; the weak-lensing result is already quotable as-is from the current paper text.

### Protocol for any rerun

- Pair the `theta`, `eta`, and conventional-coordinate baselines on identical seeds and identical budgets, so the comparison is paired rather than across independent runs.
- Use the same density estimator in every arm. The heater run's MAF-versus-MDN asymmetry is exactly the kind of confound a reviewer will find.
- Report the difference with dispersion across seeds, not two separate point estimates.
- Report CRPS and calibration together, always.
- Pass `--run-coverage` on every rerun, and set `--coverage-nsims` to the budget the claim rests on rather than leaving it at `--fom-nsims`.
- Evaluate all methods in theta units, and say so in the response, so the comparison cannot be dismissed as a change of measure.
- State the budget as training simulations, with held-out evaluation simulations reported separately.

## Open items

- ~~10-trial recovery table~~ DONE (27 Jul 2026): filled under t9KB Q2 and FoSE W1.
  10/10 Rosenbrock, 10/10 SIR, 10/10 TaylorF2, 7/10 IMRPhenomD.
- ~~Wall-clock timings for the cost paragraphs~~ DONE (27 Jul 2026): filled under
  t9KB Q3 and FoSE W3, medians over the 10 rebuttal seeds, single V100.
  Fisher / coordinates / SR / total, in seconds unless noted:
  Rosenbrock 262 / 365 / 654 / 21.0 min; SIR 221 / 174 / 130 / 9.3 min;
  TaylorF2 276 / 227 / 263 / 12.9 min; IMRPhenomD 478 / 221 / 264 / 15.5 min.
  Simulation time is not separately recorded by these scripts (it is folded into
  the Fisher stage and is small), and these exclude queue wait.
- Flatness numbers anywhere in this file must be post-fix, i.e. regenerated after
  commit 4514c64, which corrected three flatness acceptance tests that were
  discarding coordinate *improvements*. See `notes/postprocessing_flatness_bug_fix.md`
  and `notes/fisher_variance_over_prior.md`.
- QfCk Q1 per-experiment Gaussianity statement, and Q3 prior/noise ablation.
- FoSE W4 beta/gamma correlation and gradient cosine similarity numbers.
- t9KB baselines: decide whether FMPE/CMPE at matched budget is feasible in the window.
- Verify the weak-lensing appendix reference exists in the version reviewers hold; the expressions quoted above are from the current `paper/degen_distillery_paper/main.tex` (Sec. applications, WL paragraph).

## References for the revision, not for the reply

```bibtex
@book{amari2016information,
  title={Information Geometry and Its Applications},
  author={Amari, Shun-ichi},
  series={Applied Mathematical Sciences},
  volume={194},
  year={2016},
  publisher={Springer},
  address={Tokyo},
  isbn={978-4-431-55977-1},
  doi={10.1007/978-4-431-55978-8}
}
```

SBI-at-NeurIPS evidence for the AC comment, counted from the proceedings indices at `papers.nips.cc` on 27 Jul 2026 by title match on "simulation-based inference" or "likelihood-free inference" (a lower bound, since many SBI papers do not say so in the title):

- NIPS 2016: Fast epsilon-free Inference of Simulation Models with Bayesian Conditional Density Estimation.
- NIPS 2017: Flexible statistical inference for mechanistic models of neural dynamics; Hierarchical Implicit Models and Likelihood-Free Variational Inference.
- NeurIPS 2023 (5): Flow Matching for Scalable Simulation-Based Inference; Calibrating Neural Simulation-Based Inference with Differentiable Coverage Probability; L-C2ST: Local Diagnostics for Posterior Approximations in SBI; Learning Robust Statistics for SBI under Model Misspecification; Meta-learning families of plasticity rules in recurrent spiking networks using SBI.
- NeurIPS 2024 (2): Consistency Models for Scalable and Fast Simulation-Based Inference; Active Sequential Posterior Estimation for Sample-Efficient SBI.
- NeurIPS 2025 (4): FNOPE (SBI on function spaces with Fourier Neural Operators); Multilevel neural simulation-based inference; Inductive Domain Transfer in Misspecified SBI; Simulation-Based Inference for Adaptive Experiments.
