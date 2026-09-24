# Heater one-step screen and discovery

The heater is a one-dimensional simulator. One observation equals the product of the d parameters, times (1 − exp(−t/τ)), plus noise. Each parameter has a uniform prior on [1, 2]. The Fisher information of one sample has rank 1 at every parameter value and every d.

This note states the screen, the symbolic search, and the posterior comparison. The numbers are means over finished trials in `heater_oneshot_scaling_v4` on 24 Sep 2026. Two trials remain.

## Screen

Each trial trains a probe one-step map with four latent axes. The map is a neural coordinate η of the parameters. A second network predicts η from the data. The training loss is the one-step negative log likelihood.

The screen then measures each probe axis on held-out data. The information of an axis is 0.5 log(prior variance / residual variance), in nats. Both variances use the prior principal-component basis of η. An axis that the data cannot predict has information near 0 nats.

The estimated rank r_hat is the number of axes with information above 1 nat. The screen limits r_hat to at most d. A spare axis does not reduce the information. The information is the gain over a prediction that ignores the data.

The trial then trains a new network from random initial weights. This network has m_fit latent axes. m_fit equals r_hat. If the probe spectrum falls by a factor below 10 before the next axis, m_fit equals r_hat + 1. m_fit is at most 4, and m_fit is at most d.

## Symbolic search

The trial fits an ensemble of eight networks at m_fit. It aligns the eight coordinates to one origin. The alignment centers each member, rotates it, scales it, and applies one shared shift.

Operon then searches for a symbolic expression of the aligned coordinates. The allowed operations are add, multiply, divide, a constant, a variable, and a square root. Pareto candidates are the short expressions on the Operon front. The trial evaluates those candidates with the frozen one-step loss. It keeps the candidate with the lowest loss.

A trial is correct only when both conditions hold. The screen returns r_hat = 1. The absolute Spearman correlation of the expression with the true product is at least 0.99.

## Posterior comparison

The trial trains three mixture density networks. One network takes the raw parameter vector. One network takes the analytic product. One network takes the discovered expression.

The reported quantity is a log density on one shared scalar axis. For each validation observation, the trial draws samples from the posterior. It projects the samples onto the axis and fits a Gaussian. The log density is the Gaussian density of the true axis value. The tables give the mean of that log density, in nats, and the standard error of the mean.

The raw network and the discovered network share the discovered axis. The analytic network uses the analytic product as its axis. A positive gap means the low-dimensional network assigns a higher log density than the raw network.

## Results

A failed posterior sample stays in the rank count and the product count. It does not enter the log-density means.

| d | trials | rank 1 | product | scored |
| --- | --- | --- | --- | --- |
| 2 | 10 | 9 | 9 | 10 |
| 3 | 10 | 10 | 10 | 10 |
| 4 | 10 | 10 | 10 | 10 |
| 6 | 10 | 10 | 10 | 10 |
| 8 | 10 | 10 | 10 | 8 |
| 10 | 10 | 10 | 10 | 10 |
| 12 | 10 | 9 | 9 | 9 |

The miss at d = 2 is trial 3. The screen returned rank 2. The expression did not match the product. At d = 8, trials 0 and 3 matched the product, and then the posterior sampler failed. Their log densities are absent. At d = 12, one trial remains.

Mean log density on the discovered axis, in nats. The error is the standard error of the scored trials. The gap is the paired difference on each trial.

| d | n | raw | discovered | discovered minus raw |
| --- | --- | --- | --- | --- |
| 2 | 10 | 0.47 ± 0.21 | 0.50 ± 0.20 | +0.03 ± 0.01 |
| 3 | 10 | 0.95 ± 0.03 | 1.07 ± 0.05 | +0.12 ± 0.04 |
| 4 | 10 | 1.00 ± 0.02 | 1.33 ± 0.06 | +0.34 ± 0.06 |
| 6 | 10 | 0.76 ± 0.04 | 1.44 ± 0.09 | +0.68 ± 0.09 |
| 8 | 8 | 0.54 ± 0.06 | 1.30 ± 0.05 | +0.76 ± 0.10 |
| 10 | 10 | 0.41 ± 0.04 | 1.11 ± 0.13 | +0.70 ± 0.12 |
| 12 | 9 | −0.06 ± 0.05 | 1.16 ± 0.10 | +1.21 ± 0.11 |

On the analytic axis, analytic minus raw is the oracle gap.

| d | n | oracle gap (nats) |
| --- | --- | --- |
| 2 | 10 | −0.00 ± 0.02 |
| 3 | 10 | +0.00 ± 0.03 |
| 4 | 10 | +0.19 ± 0.03 |
| 6 | 10 | +0.82 ± 0.06 |
| 8 | 8 | +0.96 ± 0.12 |
| 10 | 10 | +1.02 ± 0.13 |
| 12 | 9 | +1.57 ± 0.07 |

From d = 6, the discovered expression retains most of the oracle gap. The raw log density falls as d grows. The discovered log density stays near 1.1 to 1.4 nats.

## Still open

Two d = 8 trials still need a posterior log density. One d = 12 trial remains. The d = 2 rank error is a second axis that repeats the same product. The information test cannot separate that axis from a new direction.
