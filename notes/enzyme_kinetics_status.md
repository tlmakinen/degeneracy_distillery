# Enzyme kinetics: a non-power-law degeneracy, and where the pipeline stops

Script: [scripts/enzyme_notebook_run.py](../scripts/enzyme_notebook_run.py)

**Status: not ready to run as a benchmark.** The physics is exact and three of
the four pipeline stages work. The remaining gap is structural rather than a
budget or tuning problem, and is described at the bottom. Read that section
before spending cluster time on this.

## Why this problem was built

Every degeneracy currently in the paper is a product of powers: `J/T`, the
Reynolds number, `K/sigma`, the chirp mass, `S8`. All of them can be recovered
by fitting log-log exponents, which invites the objection that the method is
dimensional analysis with a neural network attached.

Enzyme kinetics breaks that. The mass-action scheme

    E + S  <->(k_f, k_r)  ES  ->(k_cat)  E + P

with a known enzyme concentration `E_0` reduces, under the quasi-steady-state
approximation, to two combinations:

    V_max = k_cat * E_0        (with E_0 known, this measures k_cat)
    K_M   = (k_r + k_cat)/k_f

`K_M` is a sum inside a quotient. No power law can express it. The best power
law in `(k_f, k_r, k_cat)` reaches only **R^2 = 0.9655** for `log K_M`, with a
median error of 10 percent and a tail out to 70 percent. Recovering `K_M`
would therefore be direct evidence that symbolic regression finds structure
exponent fitting provably cannot.

## Design

Observable: fractional product conversion `P(t)/S_0` on 16 log-spaced times
from t=2 to t=600, at three substrate concentrations `S_0 = {1, 3, 9}` that
bracket the `K_M` prior, with additive assay noise `sigma = 0.02`. 48 numbers
per simulation.

Priors are log-uniform: `k_f` in [0.5, 3.0], `k_r` in [0.3, 6.0], `k_cat` in
[0.5, 4.0], with `E_0 = 0.02` fixed and known.

Two design choices took several iterations and are worth keeping:

- **The measurement noise is load-bearing, not cosmetic.** With noiseless data
  the tiny non-QSSA signal makes every parameter identifiable and there is no
  degeneracy at all. The degeneracy exists only relative to a noise floor.
- **`k_r` and `k_cat` need comparable log-ranges.** If one dominates their sum,
  `(k_r + k_cat)` is well approximated by a power law and the whole point of
  the experiment evaporates.

An earlier four-parameter version that also inferred `E_0` was abandoned: two
null directions in 4D give a rank-2 Fisher, and the flattening objective cannot
map that to the identity. Fixing `E_0` gives 2 stiff directions and 1 null,
matching Ising. It is also the honest experimental setup, since you know the
enzyme concentration you pipetted.

## What works

**The physics is exact.** The observable is a deterministic ODE solve plus
Gaussian noise of known sigma, so the true Fisher is available analytically as
`J^T J / sigma^2`. Its null direction matches the Michaelis-Menten prediction
(the direction along which both `k_cat` and `K_M` are constant) to
`|cos| = 1.000000`, median over the prior. This is a verified degeneracy, not
an empirical correlation.

**The fishnets find it, above a sharp budget threshold.** Measured against the
analytic truth:

| sims | recovered condition number | null eigenvalue ratio | null direction error |
|---|---|---|---|
| 150 | 28 | 3.6e-2 | 14.0 deg |
| 500 | 364 | 2.7e-3 | 1.2 deg |
| 1500 | 882 | 1.1e-3 | 1.0 deg |
| 4000 | 978 | 1.0e-3 | 1.1 deg |
| truth | 6.9e5 | 1.4e-6 | 0 |

Two lessons. Below roughly 500 simulations the fishnets do not resolve this
degeneracy at all, so **never run this problem at smoke budgets** and expect
anything meaningful. And the recovered eigenvalue ratio plateaus at about
`1e-3` no matter how much data is added: the fishnet objective cannot represent
a direction carrying less information than the prior scale, because the loss is
flat there. That is the posterior-covariance-versus-Fisher distinction showing
up concretely, and it is worth knowing for any problem with a strong degeneracy.

**The flattening works once the Fisher is regularised.** See the next section.

**Symbolic regression is capable of the target.** Handed `K_M` directly on the
same input distribution with a 120 second budget, PyOperon returned

    (-125.22 + 84.97*X2 + 52.17*X3) / (37.27*X1 - 29.81)

which, dividing through by 14.908, is

    (-8.400 + 5.700*X2 + 3.499*X3) / (2.500*X1 - 2.000)

against a truth of `(-8.400 + 5.700*X2 + 3.500*X3) / (2.500*X1 - 2.000)`. Every
coefficient correct to four significant figures, `R^2 = 0.9999993`. SR is not
the bottleneck.

## The Fisher ridge, and why it is needed

The flattening loss contains a `||Q^-1 - I||` term. A direction the data cannot
see gives `Q` a near-zero eigenvalue, so `Q^-1` explodes and the loss is
dominated entirely by the degenerate direction. Satisfying it would need an
unbounded Jacobian stretch, which the invertibility penalty forbids, so the
optimiser makes no progress on the two directions that do carry information.

This is quantitatively out of range compared with the rest of the paper:

| Experiment | Fisher conditioning | flattener gain (raw -> nn) |
|---|---|---|
| Kuramoto | 5 | 1.20x |
| Ising | 21 | 1.47x |
| Kolmogorov | 48 | 5.09x |
| Enzyme, untreated | 455 | 1.28x |
| Enzyme, ridged | 38 | **4.0x** |

`apply_fisher_ridge` adds a constant `lambda * I` chosen so the median condition
number lands on `fisher_ridge_target_cond` (default 40). This is exactly a
Gaussian prior on the scaled parameters, so the flattened coordinates whiten the
*posterior* rather than the likelihood. Directions well above `lambda` are
untouched; the degenerate one is pinned at the prior scale instead of running
off to zero. `lambda` is global rather than per-sample so that it behaves as a
genuine prior and does not distort how the metric varies across parameter space.

With it, held-out flatness improves from 1.033 to **0.325** against 1.289 for
raw parameters, a 4x gain that matches the best in the paper.

Note this cannot be avoided by weakening the degeneracy physically. `E_0` sets
how exact the QSSA is, but sweeping it does not open a usable window: even at
`E_0/S_0 = 0.8`, where the approximation is thoroughly broken, conditioning is
still 5138. Progress curves pin `V_max` at SNR ~170 regardless, so the spread
between best- and worst-measured directions never closes.

## Where it stops

Everything above works, and the run still does not recover `K_M`. The best
single direction in the flat coordinate space predicts `log K_M` at
`R^2 = 0.81`, against `0.96` for the best power law.

The reason is that **flattening determines the metric, not the coordinates.**
On the 2D stiff manifold the observable depends only on `(k_cat, K_M)`, so the
flat coordinates are necessarily some diffeomorphism of that pair. But the
whitening is defined only up to an isometry, and the Fisher is not diagonal in
`(log k_cat, log K_M)`, so the flat axes come out as a *mixture* of the two
physical combinations. Nothing in the objective prefers the unmixed basis.

So symbolic regression is never handed `K_M` as a target. It is asked to
describe mixed coordinates, and it faithfully returns complicated expressions
for them.

This is the paper's coordinate-uniqueness issue in a sharp, isolated form. The
other experiments avoid it because their stiff coordinates are ratios whose
whitening happens to stay close to axis-aligned; here it does not.

## What would actually fix it

The disentangling has to come from a criterion outside the flattening
objective, since that objective is genuinely indifferent to the mixing. Options,
roughly in increasing order of effort:

1. **Search over rotations for the simplest description.** After flattening,
   sweep rotations of the stiff block and select the one minimising total MDL of
   the fitted expressions. `regroup_like_terms` already does something in this
   spirit but only over a restricted atom-based family, and it is applied after
   SR rather than before. Doing it as an explicit 1-parameter rotation search on
   the 2D stiff block, scoring each by SR description length, is cheap and is
   the most likely quick win.
2. **Penalise mixing during flattening,** for example an L1 term on the
   off-diagonal of the Jacobian in a candidate physical basis. This biases the
   result and needs care to avoid assuming the answer.
3. **Accept the pair and change the claim.** Report that the method recovers the
   2D reduction `(k_cat, K_M)` as a manifold, verified by showing that held-out
   progress curves collapse onto it, without claiming to name `K_M` itself. This
   is weaker but defensible and needs no new machinery.

For the rebuttal specifically, option 3 is the only one that is safe on the
timeline, and option 1 is the one worth trying if there is a spare afternoon.

## Reproducing the diagnostics

All of the numbers above come from analysis scripts written during development;
they are not committed, but each is a short standalone file. The load-bearing
ones are the analytic-Fisher comparison (true spectrum versus fishnet estimate,
plus the null-direction angle against Michaelis-Menten theory) and the
`E_0` conditioning sweep. Both need only `jax.jacfwd` of the simulator and no
training, so they run in seconds and are worth rebuilding if this is picked up
again.

CLI knobs added for this work: `--nsims`, `--num-fishnets`, and
`--fisher-ridge-target-cond` (pass 0 to disable the ridge and reproduce the
untreated failure).
