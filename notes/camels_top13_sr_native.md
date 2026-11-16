# CAMELS top 13: aligned coordinates, their nonlinear part, and native-MDL symbolic regression

16 Nov 2026. Laptop, on the outputs of the first Colab ensemble run
(`results/whittle_camels/colab_top13/`, data network 256-256-128 with a linear
shortcut). Background: `notes/whittle_procedure.md`.

A second Colab run with a deeper data network (512-512-256-128, no shortcut) was
started the same day. Everything here should be repeated on it; the commands are
at the end.

## Summary

- The data constrain **six directions** among the 13 parameters. After alignment
  the six coordinates agree across three splits at 0.99 or better on shared
  simulations.
- Each coordinate is **98 to 99% linear** in the 13 scaled parameters. The
  nonlinear remainder (0.7 to 2.1% of the variance) is partly real: it repeats
  across splits and is 30 to 70% predictable on held-out points.
- Native-MDL symbolic regression returns **short, near-linear expressions** (median
  10 nodes, 5 inputs) that match the aligned targets and transfer across splits
  (R² 0.97). It also finds the main bends (square roots in Ω_b and A_SN1).
- Searching on the description length alone was more reliable here than adding
  R² as an objective.

## 1. Alignment

The ensemble files hold the aligned coordinates in the frame of one member,
which is an arbitrary rotation and differs between splits. To compare splits and
to give symbolic regression simple targets, all runs are rotated to one common
basis:

1. Per run, take the six leading principal axes of the ensemble-mean coordinates
   (their variances track the information per axis) and their linear loadings on
   the standardised parameters.
2. Average the projectors onto the six-dimensional loading subspaces over all
   six runs (13 and 7 latents, three splits). Its six leading eigenvectors are
   the shared subspace (eigenvalues 1, 1, 1, 1, 0.99, 0.94; the seventh is 0.06).
3. Rotate that basis by varimax, which gives the sparsest loadings.
4. Per run, rotate the six principal axes so that their loadings match that
   basis (least squares). Rotate every member the same way; `y_std` is the member
   spread of the rotated coordinate.

Targets (loadings on standardised scaled parameters, 0.15 and above):

| | target |
|---|---|
| η₁ | Omega_b +1.00 |
| η₂ | H_0 +0.98 |
| η₃ | Omega_m +0.97 |
| η₄ | IMFslope +0.90, WindDumpFactor +0.26, sigma_8 −0.22, n_s −0.20 |
| η₅ | A_SN2 +0.80, WindFreeTravelDensFac −0.49, WindDumpFactor +0.26, MinWindVel +0.22 |
| η₆ | A_SN1 +0.70, n_s −0.43, sigma_8 −0.41, SNII_MinMass −0.33, WindDumpFactor −0.18 |

Checks:

- Loading mismatch after rotation: 0.04 to 0.19 (norm of the difference from
  the unit-norm target), largest for η₄.
- Simulations that sit in two splits' test sets (32 to 45 per pair; held out in
  both): the same coordinate from the two ensembles correlates at 0.988 to 0.997.
- The varimax basis is a choice. The six-dimensional subspace does not depend on it.

Figure: `results/whittle_camels/sr_native/aligned_coords_vs_theta.png` (every
coordinate against every parameter, three splits overlaid).

## 2. Linear part and nonlinear remainder

Per split and coordinate: weighted linear fit on the 13 scaled parameters
(weights from the member spread); the remainder is the coordinate minus that fit.

| | η₁ | η₂ | η₃ | η₄ | η₅ | η₆ |
|---|---|---|---|---|---|---|
| remainder, % of variance (three splits) | 0.9, 1.2, 0.8 | 0.8, 1.4, 1.1 | 1.2, 1.1, 1.3 | 0.8, 0.8, 0.7 | 1.5, 1.5, 1.5 | 1.1, 1.2, 2.1 |
| member-spread variance, % of variance | 2.8–4.9 | 4.0–5.7 | 1.7–5.9 | 1.5–4.1 | 1.1–2.6 | 1.9–5.0 |
| χ² per point of the linear fit | 0.25–0.67 | 0.28–0.47 | 0.29–0.72 | 0.31–0.68 | 0.64–1.70 | 0.42–0.82 |
| remainder, same simulations in two splits (correlation) | 0.40–0.59 | 0.32–0.75 | 0.60–0.69 | 0.43–0.53 | 0.63–0.76 | 0.55–0.75 |

The remainder is smaller than the member spread point by point, but it is not
noise: it repeats between independently trained ensembles.

What it consists of (share of the remainder explained on held-out points,
5-fold, ridge):

| model of the remainder | range over coordinates and splits |
|---|---|
| curves in the coordinate's own main parameters | −0.02 to 0.34 |
| products of two different parameters | 0.08 to 0.56 |
| curves and products together | 0.30 to 0.69 |

- **Bends in the coordinate's own main parameter**, same shape in all three
  splits: η₁ in Omega_b, η₃ in Omega_m, η₅ in A_SN2 and WindFreeTravelDensFac,
  η₆ in A_SN1. All flatten towards high values (quadratic coefficients −0.03 to
  −0.05 in units of the coordinate's standard deviation).
- **Products carry more of it than the bends**, but no single product dominates.
  Largest with the same sign in all splits: Omega_m × H_0, Omega_m × sigma_8,
  Omega_b × WindFreeTravelDensFac, A_SN2 × MinWindVel; each 0.01 to 0.05.
- A quadratic model fitted on one split predicts little of another split's
  remainder (R² −0.3 to 0.4). With 205 points the terms are not pinned down
  individually.

**Log-sampled parameters.** A_SN1, A_SN2, WindFreeTravelDensFac and
WindEnergyReductionFactor are log-uniform in the simulations (implied max/min
ratios about 21, 4.7, 100 and 15) but are scaled linearly to [1, 2]. The
coordinates are *not* linear in their logs: with log inputs the remainder of η₅
and η₆ rises from 1.5% to about 6%. The dependence lies between linear and log.

Figures: `remainder_vs_theta.png` (remainder against every parameter, with
binned means per split) and `linear_part_and_remainder.png`, same folder.

## 3. Symbolic regression with the native description length

Patched Operon of branch `operon-native-mdl` (`DistilleryMDLEvaluator`), run
from that worktree's `third_party/operon_mdl/venv`. Per split and coordinate:
155 points for the search, 50 held out; inputs are the 13 scaled parameters;
weights are the member spread; symbols `add, mul, pow, constant, variable, exp,
logabs, sqrt`; structure rules enforced in the search (no input in an exponent,
no self-transcendental term); 60 nodes at most; 45 s per search; length penalty
2. The pick is the smallest description length on the front, computed by Operon.

| | description length only | R² + description length |
|---|---|---|
| searches | 18 | 18 |
| median nodes | 10 | 15 |
| median inputs used | 5 | 5 |
| median description length | 159 | 178 |
| held-out R² (13-input linear fit: 0.987) | 0.979 | 0.981 |
| R² on the other two splits | 0.970 | 0.968 |
| picks more than 0.02 below the linear fit | 0 | 4 |
| share of the pick that is linear, over the prior box | 1.000 | 0.998 |
| generations per search | 3611 | 1307 |

- **The picks match the targets**: cosine between the pick's loadings and the
  target direction 0.98 to 0.997 (description-length search), and the same main
  inputs in all three splits for every coordinate.
- **Description length alone was the more reliable search.** Adding R² gave four
  poor picks (held-out R² 0.80 to 0.94). The branch's Rosenbrock tests found the
  opposite with hidden coordinates; here the targets are near-linear.
- **The bends are found.** Picks of the description-length search:

| | split 2 pick (X = scaled parameter) | in the other splits |
|---|---|---|
| η₁ | `sqrt(67.0·Omega_b) − 10.67 + 0.45·sigma_8` | sqrt in split 1, a product of two logs in split 0 |
| η₂ | `3.25·H_0 − 0.45·n_s − 0.26·A_SN1 + log|0.014·MinWindVel|` | split 0 drops A_SN1; split 1 adds WindDumpFactor |
| η₃ | `3.47·Omega_m + 0.59·sigma_8 + 0.45·A_SN1 + 0.29·MinWindVel + 0.22·IMFslope + log|0.0004·n_s|` | split 0 has SNII_MinMass in place of IMFslope; split 1 has both |
| η₄ | `3.33·IMFslope + 1.06·WindDumpFactor − 0.48·n_s − 0.44·sigma_8 − 5.22` | linear in all three; A_SN2 −0.2 added in splits 0 and 1, WindEnergyReductionFactor −0.2 in split 1 |
| η₅ | `2.86·A_SN2 − 1.92·WindFreeTravelDensFac + 0.90·WindDumpFactor + 0.35·sigma_8 + log|0.020·MinWindVel|` | split 0 has n_s in place of sigma_8; a product form in split 1 |
| η₆ | `sqrt(9.8 + 37.7·A_SN1) − 1.66·n_s − 1.51·sigma_8 − 1.17·SNII_MinMass − 0.71·WindDumpFactor` | sqrt in A_SN1 in all three |

- **Discount the `log|c·X|` terms with a tiny c.** Over 1 to 2 they are a
  constant plus a mild slope: the search writes an offset cheaply this way (the
  criterion is not invariant to how a constant is written; see
  `third_party/operon_mdl/README.md` on that branch). They are not evidence of a
  log dependence.
- The picks are worse than the 13-input linear fit by 0.008 in held-out R² and
  use 5 inputs instead of 13. χ² per held-out point is 1.03 against about 0.65
  for the linear fit.

All picks: `results/whittle_camels/sr_native/picks.json`; table:
`results/whittle_camels/sr_native/summary.txt`.

## 4. Search on the nonlinear remainder alone

Same search with the target replaced by the remainder of section 2
(`--residual`). Running at the time of writing; results to be added here
(`results/whittle_camels/sr_native_residual/`).

## Limits

- **205 points per split.** The run kept the coordinates on the test simulations
  only, so there is no augmentation with fresh prior draws through the networks.
  The second Colab run does not change this; it needs the fitted networks.
- **Informative axes identified through variance.** The first run did not save
  the data summaries, so the six leading axes are taken from the variance of the
  coordinates, not from the information itself. The second run saves them.
- **Member spread as the weight.** Members share a warm start, so the spread is
  a stability measure. It is slightly too small in a few coordinates
  (leave-one-out ratio up to 1.34).
- **The map starts linear on both sides.** Near-linear coordinates are what this
  map learned from 819 simulations, not proof that the sufficient coordinates
  are linear. The deeper data network without the shortcut tests the data side;
  the eta(theta) network is unchanged and still starts linear.
- One Operon seed per search, 45 s, on a laptop whose cores were shared with
  another session's runs.

## Reproduce, and repeat on the deeper-network run

```
PY=.claude/worktrees/operon-native-mdl/third_party/operon_mdl/venv/bin/python
RUN=results/whittle_camels/colab_top13          # or the unzipped camels_top13_deep
OUT=results/whittle_camels/sr_native

python scripts/camels_ensemble_summary.py $RUN $RUN/loadings.npz    # numpy + scipy only
$PY scripts/camels_sr_native_mdl.py --runs $RUN --out-dir $OUT --time-limit 45
$PY scripts/camels_sr_native_mdl.py --runs $RUN --out-dir ${OUT}_residual --residual --time-limit 45 --max-length 40
$PY scripts/camels_sr_native_summary.py $OUT
python scripts/camels_aligned_coords_plot.py $OUT                   # needs matplotlib
```

`camels_sr_native_mdl.py` writes `targets.npz` (aligned coordinates, spread,
parameters, split indices) before the first search; the plots and the
linear/nonlinear analysis only need that file.

For the comparison with the deeper network, the numbers to beat are: held-out
information 8.63 / 7.81 / 8.28 nats at 13 latents, remainder 0.7 to 2.1% of the
variance, member spread 12 to 24% of the range on the six leading axes.
