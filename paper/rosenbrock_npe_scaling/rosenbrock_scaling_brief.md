# Rosenbrock one-step screen, discovery, and NPE

A rank-2 banana is hidden in `d` unused coordinates. This note states the
screen, the symbolic search, and the posterior comparison used for the
N=2000 scaling figure. The numbers are means over scored trials in
`rosenbrock_npe_scaling`.

## Problem

Each trial draws `theta ~ U[-3, 3]^d`. The informative pair `(I0, I1)` is
drawn from the trial seed (`I0=0`, `I1=1` when `d=2`). One observation is
eight i.i.d. Gaussian replicates of

    (theta_I0,  theta_I1 - theta_I0^2)

with per-component noise `sigma = (0.25, 0.5)`. The other `d-2`
coordinates are nuisance.

Budget: 2000 training and 1000 test simulations per `(d, trial)`. Grid:
`d in {2, 4, 8, 16, 32}`, 10 trials. Seed `7919 * trial + 31 * d`.
Discovery and all three NPE arms regenerate those same draws.

## Screen

`eta(theta)` is a residual MLP with widths `(128, 128, 128)` and GELU.
Rosenbrock does not use log features. `eta(x)` mean-pools the eight
replicates, then a residual MLP of widths `(32, 32)`.

Each trial trains a probe one-step map with four latent axes (20k steps,
batch 512, Adam `1e-3`). The screen scores each probe axis on a held-out
slice of the training set (`max(20, 0.15 N) = 300` rows) by

    info_j = 0.5 log(lam_j / rho_j)     [nats]

in the prior principal-component basis of `eta`. `r_hat` is the number of
axes above 0.5 nats, at least 1 and at most `d`. The trial then trains a
new network from random initial weights at `m_fit` latents (`r_hat`, or
`r_hat + 1` when the probe spectrum has no clean cutoff), 20k steps. No
probe weights are reused.

## Symbolic search

An ensemble of four networks at `m_fit` (5k steps) is aligned to one
origin. Operon searches for a symbolic stack of the aligned coordinates
on the 1000 test points plus `1000 d` prior draws, mapped into the same
frame. Allowed operations include add, multiply, divide, power, square
root, a constant, and a variable. Up to eight Pareto candidates are
re-scored with the frozen one-step loss. The trial keeps the lowest-loss
stack.

A trial is symbolically recovered when a cubic polynomial of the stack
reaches `R^2 >= 0.95` on both true banana coordinates.

## Posterior comparison

Three masked autoregressive flows share the same estimator: 5 transforms,
50 hidden features, 2 ensemble members, 400 epochs, patience 20, batch
64, Adam `1e-4`. Raw trains on `theta` in `[-3, 3]^d`. Oracle trains on
the standardised true banana. Discovered trains on the standardised
symbolic stack.

The reported quantity is a 2D Gaussian-matched log density. For each of
200 validation observations the trial draws 2000 samples from the flow,
projects them onto a shared pair of axes, and scores `log N(true | mu,
Sigma)`. Raw and oracle share the banana axes. Raw and discovered share
the discovered axes. The figure puts discovered on the banana scale by

    discovered_on_oracle = raw_on_oracle + (discovered - raw)_on_discovered

A positive gap means the 2D network assigns a higher log density than
raw, on that shared axis.

## Results

Failed NPE rows stay out of the means. Discovery recovered rank 2 and the
symbolic banana in 10/10 trials at every `d`, including `d=32`. NPE is
missing trials 0 and 7 at `d=32` (aborted on the node). Those two are
absent from the figure (`n=8`).

Mean log density on the oracle-axis scale, in nats. The error is the
standard error of the scored trials.

| d | n | raw | discovered | oracle |
| --- | --- | --- | --- | --- |
| 2 | 10 | 2.887 ± 0.025 | 2.943 ± 0.027 | 2.937 ± 0.028 |
| 4 | 10 | 2.931 ± 0.033 | 2.936 ± 0.030 | 2.956 ± 0.024 |
| 8 | 10 | 2.970 ± 0.026 | 2.983 ± 0.019 | 3.016 ± 0.021 |
| 16 | 10 | 2.827 ± 0.038 | 2.906 ± 0.067 | 2.992 ± 0.027 |
| 32 | 8 | 2.613 ± 0.065 | 2.862 ± 0.040 | 2.964 ± 0.021 |

Oracle stays near 3 nats. Raw falls at `d=16` and `d=32`. Discovered
tracks oracle more closely than raw once `d` is large.
