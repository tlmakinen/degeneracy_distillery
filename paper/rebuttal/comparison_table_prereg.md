# Rebuttal reruns at the pre-registered thresholds

The same records scored at each experiment's pre-registered correlation bar (Rosenbrock 0.5, SIR 0.5, GW TaylorF2 0.75, GW IMRPhenomD 0.75) rather than the uniform 0.6 used in `comparison_table.md`. This is the anchor: its three-step column reproduces the published rebuttal table exactly, and the build asserts that.

| Experiment | Train sims | Three-step recovered | Three-step alignment | One-step recovered | One-step alignment |
|---|---|---|---|---|---|
| Rosenbrock | 500 | 10/10 | 0.929 (0.814,0.960) | n/a * | 0.999 (0.999,1.000) * |
| SIR | 500 | 10/10 | 0.674 (0.636,0.741) | 10/10 | 0.667 (0.637,0.779) |
| GW TaylorF2 | 500 | 10/10 | 0.972 (0.966,0.979) | 8/10 | 0.983 (0.967,0.983) |
| GW IMRPhenomD | 500 | 7/10 | 0.997 (0.988,0.999) | 5/10 | 0.991 (0.984,0.995) |

`*` The one-step Rosenbrock arm reports `r2_true_min` (an R^2 of the true coordinates on a cubic of the recovered ones), not the Pearson correlation the three-step arm reports. The two alignment cells are different statistics and must not be compared. Its complementary conjunct `complementary_linear_alignment` is absent from the one-step record, so recovery cannot be scored on the published criterion either.

## Config variants

Deliberate config changes, not reruns of the frozen baseline. Listed separately so the headline table stays like-for-like.

| Variant | Change | Recovered (NLL pick) | Recovered (MDL pick) | Alignment |
|---|---|---|---|---|
| SIR, info-floor 0.5 | `--info-floor 0.5` | 10/10 | 10/10 | 0.770 (0.633,0.862) |
| GW IMRPhenomD, forced m=2 | `--rank-min-gap 1e9` | 9/10 | 10/10 | 0.987 (0.981,0.996) |

The MDL column reads `n/a` for the baseline one-step trees that predate `db4b146`, when `expression_mdl` was computed and then dropped before the record was written.
