# Square-completed rectangular loss for K < d shared parameter–data embeddings

*Handoff note from the `unified-sbi` project (5 Oct 2026). Self-contained: it needs no other file. The derivations and toy results are in `docs/shared_space.md` (§10.1, §11.6, §14) of that repo.*

## TL;DR

- **Setting.** The data inform only K of the d parameter directions; the other d − K directions are degeneracies and inert nuisances. We want a one-step loss that gives a proper posterior over all of θ and leaves interpretable K-dimensional "active" coordinates.
- **What fails.** Extending the rectangular loss by changing the volume term (½‖η − t‖² − log √det(JJᵀ) with η: ℝᵈ → ℝᴷ) is not a density with the right prior. Along each fibre of η it puts mass uniformly in arc length and ignores the prior.
- **The fix: square-complete η.** Use an invertible map η = (η_a, η_b): ℝᵈ → ℝᴷ × ℝᵈ⁻ᴷ. Only η_a sees the data:

  **ℒ(θ, x) = ½‖η_a(θ) − t(x)‖² + ½‖η_b(θ)‖² − log |det ∂η/∂θ|**   (+ (d/2) log 2π)

- **What you get:**
  - an exactly normalised posterior, as long as η is a bijection;
  - no normaliser Z;
  - exact sampling and densities;
  - the prior along the degeneracies, learned by η_b;
  - η_a as whitened informed combinations for the degeneracy distillery.
- **Evidence.** On a lensing toy with a nuisance exactly degenerate with the amplitude:
  - the √det(JJᵀ) version stalls at 1.05 nats excess KL, with σ₈ posteriors 3× too wide;
  - the square-completed loss reaches 0.14 nats, matches the normalised reference (0.13) and beats a MAF (0.21);
  - it recovers both the prior on the nuisance and the σ₈ width.

## 1. Notation

- θ ∈ ℝᵈ: parameters, with prior π(θ). For a bounded box, work in logit coordinates y(θ) so that η can be onto ℝᵈ.
- x: data. t(x) ∈ ℝᴷ: a learned data summary (the "statistics").
- η(θ): the learned parameter embedding. J = ∂η/∂θ.
- The shared-space likelihood model is t ~ N(η(θ), I).
- **Fibres:** η_a⁻¹(z), the (d − K)-dimensional sets of θ that the data cannot tell apart.

## 2. The rectangular loss at K = d is a proper density

The original loss is ℒ_rect = ½‖η(θ) − t(x)‖² − log vol J, with vol J = √det(JᵀJ).

- **Absolute value.** JᵀJ is positive semi-definite, so det ≥ 0 and the absolute value in "log |vol J|" is redundant.
- **At K = d,** vol J = |det J|. By change of variables,

  q(θ | x) = N(η(θ); t(x), I) · |det J(θ)|

  integrates to 1 if η is a bijection onto ℝᵈ. So ℒ_rect = −log q + const is a proper NPE loss: its expectation is E_x KL(p(·|x) ‖ q(·|x)) + const.
- **Interpretations.** It is a one-component, unit-covariance mixture density in η-space with mean t(x). Equivalently, it is a conditional normalising flow whose only conditioning is a shift of the latent.
- **The prior** is learned implicitly through η's distortion, as in any NPE.
- **Caveat.** If η is an MLP it can fold (not injective). The change-of-variables argument then fails and the loss can be pushed below any proper posterior. Use an invertible η. Empirically, an MLP at K = d stayed within ±0.008 nats of normalised on a 2D toy, but nothing guarantees this.

## 3. Why the naive K < d extension fails

- **The volume term breaks.** For K < d, JᵀJ (d × d) has rank K, so det(JᵀJ) = 0 and −log vol J = +∞. The natural generalisation is the product of J's K non-zero singular values: vol J = √det(JJᵀ).
- **What density the loss then implies.** By the coarea formula, for any g,

  ∫ g(θ) √det(JJᵀ) dθ = ∫_{ℝᴷ} dz ∫_{η⁻¹(z)} g dℋ^{d−K},

  where ℋ is Hausdorff (arc-length or area) measure on the fibre. So q(θ|x) ∝ N(η(θ); t, I) √det(JJᵀ) means:
  - **across fibres,** q(z|x) ∝ N(z; t, I) · ℋ(η⁻¹(z) ∩ Θ), weighted by fibre length;
  - **along a fibre,** the density is uniform in arc length in θ coordinates. **The prior never enters.**
- **It is also unnormalised.** Z(t) = ∫ N(z; t, I) ℋ(η⁻¹(z)) dz, and long or wiggly fibres make it large: the same coiling incentive as the K > d rectangular loss.
- **The correct posterior under the shared-space model** is p(θ|x) ∝ N(η(θ); t, I) π(θ). Along fibres it is ∝ π.

**What this is not about.** The failure is not a lack of sufficient statistics: K can be exactly the number of combinations the data inform. It is a missing parameter-side coordinate. Along the fibres there is no latent variable in which the prior could live.

## 4. The square completion

Take η = (η_a, η_b), a bijection ℝᵈ → ℝᵈ, and t̃(x) = (t(x), 0). The loss is the K = d rectangular loss with t padded by d − K zeros:

  q(θ | x) = N(η_a(θ); t(x), I_K) · N(η_b(θ); 0, I_{d−K}) · |det ∂η/∂θ|,
  ℒ = −log q = ½‖η_a − t‖² + ½‖η_b‖² − log|det J| + (d/2) log 2π.

**Why it works:**
- **Normalised by construction** (change of variables), so E[ℒ] = E_x KL(p ‖ q) + const. The minimiser is the true posterior whenever the family can represent it. That covers a likelihood depending on θ only through K combinations of shared-space form, with a prior that a flow can represent.
- **η_b carries the prior along the degeneracies.** The padded zeros are a dummy statistic: η_b never sees data. N(η_b; 0, I)·|det J| is a data-independent density along the fibres, learned from the prior draws in the training set. That is exactly the prior conditional the naive version could not express.
- **No normaliser.** The normalised alternative q ∝ N(η_a; t, I) π(θ)/Z(t) gets the prior analytically, but needs Z:
  - exact quadrature only for d ≲ 3;
  - beyond that a Monte Carlo estimate over prior draws, whose size grows like e^{I(θ;x)}, i.e. ≳ 10⁴–10⁵ draws or importance sampling once several combinations are well constrained.

  The square completion trades this for learning the prior inside a flow, at one flow pass per batch.
- **Exact sampling:** θ = η⁻¹(t(x) + ε_a, ε_b) with ε ~ N(0, I). No MCMC.

**Cost of learning the prior.** On the toy the square-completed loss was about 2× worse in KL than the normalised prior-base head at N = 250, and equal by N = 4000.
- An untested but cheap mitigation: prior draws cost no simulations, so pre-train or regularise the flow with t := 0 on extra prior samples. Note this only fixes the marginal, not the conditional, so treat it as a warm start.

## 5. Implementation notes

- **Make η invertible.** Use a coupling flow (affine or rational-quadratic spline coupling, e.g. RealNVP or NSF) on logit-box coordinates. log|det ∂η/∂θ| = flow log-det + Σᵢ log |dyᵢ/dθᵢ|. With a box prior, dyᵢ/dθᵢ = 1/((hiᵢ − loᵢ) uᵢ (1 − uᵢ)).
  - In zuko, an unconditional flow's transform exposes `call_and_ladj(y) -> (z, log|det|)`. Check against your version.
  - The first K outputs are η_a and the rest η_b. Permutation layers inside the flow are fine, since the split is defined at the output.
- **Data network** t(x) ∈ ℝᴷ, trained jointly with the flow on −log q.
- **Inert parameters** (no data dependence at all): draw them from the prior inside the simulator and leave them out of θ. That marginalises them exactly and keeps the flow small. Putting them in η_b also works, but wastes capacity.
- **Evaluation.**
  - The held-out NLL is a proper score, so it is comparable across K and architectures. Use it for the K-screen.
  - Also report joint-HPD and marginal coverage.
  - With an exact oracle, report E[log p − log q] = E KL.
- **Debugging check.** With the flow frozen and t fixed, sample q and compare the η_b-direction marginals with the prior. They should agree for directions the data cannot see.

## 6. Reading the active combinations (degeneracy distillery)

- **η_a is whitened.** At the optimum η_a(θ) | x ~ N(t(x), I_K), so the η_a are the informed combinations in flattening coordinates.
  - The Fisher matrix restricted to the informed subspace is J_aᵀJ_a (rank K), with J_a = ∂η_a/∂θ.
  - The fibres are the level sets of η_a. Their tangent space is the null space of J_a, which gives the degeneracy directions and their curvature.
- **Gauge freedom.** The loss is invariant under
  - η_a → R η_a + c with R ∈ O(K), applied together with t → R t + c;
  - any reparametrisation of η_b that preserves N(0, I) conditionally on η_a.

  Read only η_a, and fix its rotation.
- **Sparse rotation for interpretability.** Choose R ∈ O(K) to minimise E_π ‖R J_a(θ)‖₁, or a varimax-style criterion, so that each rotated η_a,k depends on as few parameters as possible. Then run symbolic regression on R η_a(θ) over the participating parameters. This usually reads much better than PCA axes.
- **Known-answer check.** On the toy, in (Ω_m, σ₈), the leading η coordinate was a function of ln σ₈ + α ln Ω_m with α = 0.817 ± 0.004 (3 seeds), against 0.807 exact. The local α error was 0.014 at N = 4000.

## 7. Choosing K and the participating set

- **Count active directions relative to the prior, not in raw units.** Solve F v = λ Σ_π⁻¹ v, with F averaged over the prior and Σ_π the prior covariance. A direction is informative if λ ≳ 1; it carries about ½ log(1 + λ) nats.
- **Participating parameters** are the support of the λ ≳ 1 eigenvectors in that basis. Screen directions, not individual parameters.
- **Check that the count is stable across the prior.** If it changes from point to point, the active subspace rotates, i.e. there is curvature.
- **K-screen with an exactly normalised loss,** such as this one: compare held-out NLL at K, K + 1, K + 3. Agreement with the Fisher count means the combinations behave close to an exponential family. A screen preferring larger K means curvature needs extra statistics.
  - InfoNCE-normalised screens can hide gains from larger K, because of the log B ceiling and estimator bias.

## 8. Evidence from the toy (`unified-sbi`, §14 of `docs/shared_space.md`)

**Setup.**
- θ = (Ω_m, σ₈, m); the observed convergence is (1 + m)κ, so m is exactly degenerate with the amplitude.
- Prior: m ~ N(0, 0.05²) truncated to ±3σ; the other parameters uniform on a box.
- The Gaussian-field likelihood is exact; the oracle is a 3D grid.
- Seed 0, 256 test maps, hybrid power-spectrum + CNN summary.

Excess KL in nats, and mean posterior std at N = 4000 (exact: σ₈ 0.087, m 0.048):

| loss | N = 250 | 1000 | 4000 | std σ₈ | std m |
|---|---|---|---|---|---|
| normalised, prior base, K = 2, Z on a 3D grid (reference) | 0.51 | 0.27 | 0.13 | 0.092 | 0.048 |
| **square-completed rectangular, η: ℝ³ → ℝ³, K = 2 coupled** | 1.05 | 0.43 | **0.14** | 0.092 | 0.048 |
| MAF (NPE) | 2.15 | 0.64 | 0.21 | 0.099 | 0.050 |
| rectangular with √det(JJᵀ), η: ℝ³ → ℝ², K = 2 | 2.21 | 1.34 | 1.05 | 0.262 | 0.069 |

- **Coverage.** Joint-HPD coverage error is 0.02–0.03 for all arms except √det(JJᵀ), which gets 0.09–0.14.
- **Without a nuisance** (2D, K = d = 2, 3 seeds), the rectangular loss reached excess KL 0.84 / 0.26 / 0.13 at N = 250 / 1000 / 4000, against 1.40 / 0.49 / 0.17 for a MAF.
- **Caveats:**
  - η in these runs was an MLP, not a flow;
  - one seed for the nuisance test;
  - linear-theory toy physics.

## 9. Pitfalls

1. **A non-invertible η** (MLP) makes the loss an upper bound at best and unbounded below at worst. Use a coupling flow.
2. **Forgetting the logit-box Jacobian** term for bounded priors.
3. **Reading η_b, or unrotated η_a,** as physical combinations. Only gauge-fixed η_a is meaningful.
4. **Comparing K values** with a non-normalised or InfoNCE-limited loss.
5. **Small N:** this loss must learn the prior along the fibres. If simulations are scarce and d is small enough for an exact Z, the normalised prior-base head is about 2× better at the lowest budgets.

## Reference code (`unified-sbi`)

- `src/unified_sbi/shared/nuisance.py`:
  - `RectHead(K, lo, hi, pad)`: rectangular loss with `pad` zero-padded statistics, i.e. the square completion when K + pad = d;
  - `EtaD.log_vol`: ½ log det of JᵀJ or JJᵀ;
  - `GridSharedHead`: the normalised reference;
  - `NuisanceData`: exact oracle.
- `scripts/nuisance_m_study.py`, `scripts/plot_nuisance.py`.
- Theory: `docs/shared_space.md`, §10.1 (rectangular loss = MDN in η), §11.6 (many parameters, K < d), §14 (this experiment).
