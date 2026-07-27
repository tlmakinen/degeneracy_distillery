# How much does the learned Fisher vary over the prior?

Status: **open, revisit when the QM7b procrustes arm lands** (job 3759243 and
the follow-up arrays, `follow_up_results/qm7b/rebuttal_procrustes/`).

Motivating question: if `Var_theta(F)` is small there is little nonlinear
parameter dependence for the flattener and symbolic regression to extract, and
a low recovery rate would be a property of the problem rather than a failure of
the pipeline.

Reproduce with `python scripts/fisher_variance_diagnostic.py`. It reads
`F_ensemble` / `ensemble_weights` / `theta` out of each run's saved
`*_flatten.npz`, so it needs no retraining and no GPU.

## Measures

`F(theta)` is the ensemble-averaged Fisher, shape `(500, p, p)`.

- `|dF|/|F|` — median over theta of `||F(theta) - Fbar||_F / ||Fbar||_F`.
  Overall relative variation across the prior.
- `shape` — the same after normalising each `F` to unit trace, so overall scale
  drops out and only the change in anisotropy / orientation survives.
- `R2lin` — fraction of `Var_theta(F)` captured by a linear-in-theta model.
  `1 - R2lin` is the genuinely nonlinear content, which is what SR has to find.
- `nonlin` = `|dF|/|F| * (1 - R2lin)`. Absolute nonlinear signal.

## Result (mean over all seeds of each rebuttal sweep)

```text
experiment            p  |dF|/|F|   shape   R2lin   nonlin  seeds
-----------------------------------------------------------------
Rosenbrock  (10/10)   2     0.364   0.231   0.692    0.112     10
SIR         (10/10)   2     0.775   0.374   0.549    0.349     10
TaylorF2    (10/10)   2     0.515   0.016   0.346    0.343     10
IMRPhenomD  ( 7/10)   2     0.194   0.012   0.226    0.151     10
Kolmogorov  ( 1/10)   2     0.199   0.166   0.662    0.068     10
Kuramoto    ( 0/9 )   3     0.709   0.502   0.775    0.159      9
Rayleigh-Ben( 0/9 )   3     0.388   0.106   0.441    0.228     10
QM7b        ( 0/10)   5     0.123   0.009   0.955    0.006     10
```

Seed scatter is small: Kuramoto `|dF|/|F| = 0.709 +- 0.030`, QM7b
`0.123 +- 0.021`, SIR `0.775 +- 0.038`.

## Kuramoto: the hypothesis does NOT hold

Kuramoto has the **largest** `Var_theta(F)` in the campaign (0.709, tied with
SIR) and by some margin the largest shape-only variation (0.502). Its Fisher is
strongly anisotropic and that anisotropy genuinely changes across the prior.
The flattener also demonstrably extracts it: neural flatness reaches 0.24x raw.

So Kuramoto's 0/9 is not a missing-signal problem. It stays where the cosine
analysis puts it: SR returns coordinates affine/log in a *single* parameter and
never forms the intended quotient, which caps `worst_of_best_cosine` at
`1/sqrt(2) = 0.7071` against a 0.8 bar. Measured values sit in
`[0.70453, 0.70793]` across every seed - an algebraic ceiling, not a near miss.

One point in the hypothesis's favour: Kuramoto's `R2lin = 0.775` is the highest
outside QM7b, so proportionally more of its Fisher variation is linear than in
SIR (0.549) or Rosenbrock (0.692). In absolute terms it is still mid-pack
(nonlin 0.159), above Rayleigh-Benard and more than double Kolmogorov, which
recovered 1/10. Not enough to explain a 0/9.

## QM7b: the hypothesis holds strongly, and this likely explains the 0/10

QM7b's Fisher is **near-constant over the prior**:

- overall variation 0.123, the lowest in the campaign
- shape-only **0.009** - orientation and anisotropy barely change at all
- **95.5%** of what little variation exists is linear in theta
- nonlinear signal 0.006, roughly **25x below the next lowest** (Kolmogorov 0.068)

If `F` is near-constant then the optimal coordinates are a fixed linear map,
raw theta is already close to flat - consistent with the observed very low
`frob_raw` baseline - and `geometric_improvement` asks the *symbolic*
coordinates to beat an already near-optimal baseline by 0.8x. Any SR
approximation error then makes things worse, which is exactly the observed
"symbolic flatness 7.14x worse than raw" behaviour.

This supersedes the previous "dataset-intrinsic difficulty" hand-wave in
`neurips_rebuttal_templates.md` and the campaign notes, and it is consistent
with the earlier finding that fixing the `feature_range=(0,1)` scaler bug
improved the ratios substantially without moving the bottom line: **there is
very little nonlinear geometry in QM7b to discover.**

## Caveats before this goes in a table

- Measures are computed on the ensemble-averaged Fisher in whatever units the
  npz stores. They are relative, hence scale-invariant, but prior-normalisation
  consistency across experiments has not been verified.
- QM7b is `p=5`, so it has more matrix entries than the `p=2` cases. That cuts
  *against* it showing artificially low variation, so the finding is
  conservative.
- `R2lin` regresses on raw theta. A problem whose Fisher is linear in *log*
  theta would score as nonlinear here. Worth checking for Kolmogorov and
  Kuramoto specifically, both of which are naturally log-parameterised.

## When the procrustes arm lands

QM7b's alignment is being changed from `kabsch`/`separate_nonlinearity=False`
to `procrustes`/`True` (`CONFIGS["rebuttal_procrustes"]`). Alignment sits
downstream of the Fisher, so `Var_theta(F)` itself should be **unchanged** -
the same fishnets are reused via `--skip-fishnets`. Re-running this diagnostic
on the procrustes outputs is therefore a consistency check, not a new
measurement: if the numbers move, something upstream changed that should not
have. The open question the procrustes arm actually answers is whether a
different alignment lets SR do better with the same (small) nonlinear signal.
