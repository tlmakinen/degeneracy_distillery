# Why Kuramoto / Kolmogorov / QM7b failed: the Pareto-filter + `pow` interaction

Investigation date: 2026-07-26/27. All numbers below are measured, not estimated.

## TL;DR

Symbolic regression was never the bottleneck, and neither was the network, the data,
the ensemble size, the SR time budget, nor the MDL selection criterion. The failures
come from an interaction between two things:

1. pyoperon's `pow` operator generates enormous numbers of **variable-in-exponent**
   expressions (`X3^(719.33/(0.20*X1)^...)`).
2. `sr_structure_predicate` has **`forbid_x_in_pow_exponent=True` by default**, so
   `filter_pareto_fronts` then deletes almost all of them, **in place**, leaving only
   trivial complexity-5/6 leftovers for `analyze_equations` to choose from.

Removing `pow` from `allowed_symbols` fixes it: the front survives filtering and
symbolic flatness goes from *worse than doing nothing* to *beating the raw baseline*.

## The decisive A/B (kuramoto rebuttal seed 0, saved aligned data, 45s/component)

| operator set | Pareto front | after filter | MDL held-out flatness |
|---|---|---|---|
| `add,mul,div,pow,constant,variable,square,sqrt,logabs` (current) | [26, 25, 28] | **[2, 1, 2]** | 2.476 |
| `add,mul,div,constant,variable,square,sqrt,logabs` (**no `pow`**) | [28, 28, 30] | **[28, 16, 23]** | **0.901** |

Baselines for that seed: `raw = 1.253`, `neural = 0.266`. So the current setting is
~2x **worse** than doing nothing, while dropping `pow` beats raw.

## Filter loss rates (rebuttal seed 0, unfiltered vs filtered pareto.csv)

| experiment | unfiltered | filtered | culled | outcome |
|---|---|---|---|---|
| rosenbrock | 21 | 21 | ~0% | 10/10 |
| gw_taylorf2 | 41 | 10 | 76% | 10/10 (survivors still good) |
| kolmogorov | 47 | 16 | 66% | 1/10 |
| qm7b | 127 | 18 | 86% | 0/10 |
| kuramoto | 78 | 6 | **92%** | 0/10 |

Cull rate alone is not destiny (GW loses 76% and still succeeds) -- what matters is
whether *good* expressions survive. They do not for kuramoto/kolmogorov/qm7b.

## Re-running analyze_equations on the saved `.unfiltered` fronts

`filter_pareto_fronts` writes a `pareto.csv.unfiltered` backup, so this costs nothing.

| | filtered (as run) | unfiltered | raw |
|---|---|---|---|
| kuramoto s0 | 2.545 | **0.911** | 1.253 |
| kuramoto s2 | 1.816 | 1.418 | 1.177 |
| kuramoto s5 | 2.797 | **0.752** | 1.140 |
| kolmogorov s1 | 2.467 | **0.884** | 0.997 |
| kolmogorov s5 | 1.268 | 1.019 | 0.997 |

**But do not simply disable the filter.** The unfiltered winners are things like
`X3^(719.334351/((0.204820*X1)^...))` -- excellent flatness, zero interpretability.
That is not discovery. Dropping `pow` is the better fix because it prevents those
forms from being generated at all, so what survives is interpretable.

Note the filter also deletes genuinely *good* expressions: kolmogorov s1's best
unfiltered coordinate is `(0.370502*X1 - 0.189171*X2) / (-0.150192*X2 + ...)` -- an
honest ratio, exactly the Re-like form the experiment is looking for.

## Hypotheses that were tested and REFUTED (don't re-litigate these)

- **"Observable too compressed / network needs richer data."** No: `frob_neural` is
  excellent everywhere (kuramoto 0.27, kolmogorov 0.24, qm7b 0.08 vs raw 1.0-1.6).
  qm7b's neural map is the *best* of any experiment while its symbolic is the worst.
- **"SR needs more time."** No: qm7b at 3x SR budget got *worse*
  (frob_symbolic 8.17 -> 35.97). Consistent with more time -> more complex ->
  more culled.
- **"MDL selection / length_penalty is mis-picking."** No: tested offline at
  2.0/3.0/5.0/10.0 -> **identical** expressions and flatness at every value. With only
  1-2 survivors there is nothing to select between.
- **"Ensemble spread is miscalibrated / too few fishnets."** No: measured
  signal/spread ratio is 2.6-6.6 on aligned data and 23-31 on the smoke grid; kuramoto,
  kolmogorov and ising were already at 20 fishnets.
- **"The augmented SR grid is degenerate."** No: measured against the aligned data it
  is structurally near-identical (SNR 23-31 both, linear R^2 within 0.04, augmented
  actually *higher* forest R^2).
- **"`objectives=['r2','length']` differs between scripts."** No: that is the
  function's default; all scripts use it identically.
- **"`sqrt`/`logabs` bloat the operator set."** Partially: they are not the main
  driver. `pow` is. (Both operator sets produced rich *unfiltered* fronts.)

## A methodological warning

An earlier version of this analysis claimed "the raw SR Pareto front has only 2
equations." That was **wrong**: `filter_pareto_fronts` rewrites `pareto.csv` **in
place**, so the file in the output directory is post-filter. The true raw output is in
`pareto.csv.unfiltered`. Always read the `.unfiltered` file when reasoning about what
SR actually produced.

## Recommended actions (not yet applied)

1. Remove `pow` from `allowed_symbols` in kuramoto / kolmogorov / ising / qm7b, then
   re-run the SR stage. Every target here (`K/sigma`, `D/sigma`, `J/T`, `h/T`,
   `f0/nu^2`, `lumo-homo`) is reachable with `div` + `square` alone.
2. Revert `forbid_self_transcendental=True` on qm7b (set 2026-07-26) -- it tightens
   the very filter that is doing the damage.
3. Consider whether `filter_pareto_fronts` should hard-delete at all, versus letting
   `analyze_equations` see the full front and rank by MDL (which already penalises
   complexity). Hard deletion before selection is what makes this failure silent.
