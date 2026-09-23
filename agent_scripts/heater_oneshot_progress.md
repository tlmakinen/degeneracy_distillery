# Heater one-step sweep: progress (23 Sep 2026)

The live run is `heater_oneshot_scaling_v4` (V100 `3833220`, CPU `3833221`).
`heater_oneshot_scaling_info` was cancelled at 21/70: its SR augmentation
rows sat in a different origin from the aligned rows (see below). Earlier
trees used a rank rule that overcounts. Do not plot any of them as the result.

## SR augmentation offset (fixed for v4)

`augment_aligned` only rotated the augmented `eta`. Alignment also centres
each member, restores the reference mean and subtracts the floor shift, so
SR saw two copies of the curve offset by 22-161 `eta` units (10^3-10^4 x
`y_std`), with the augmented copy the majority. It now calls
`align_coords.apply_ensemble_alignment`. Each task checks that replaying the
alignment inputs reproduces `aligned["ys"]` (`align_replay_err`, must be 0)
and fails at `sr` if not. Spearman recovery and the frozen-NLL pick are
affine-invariant, so they were not biased; Operon's fit and the MDL ranking
were.

Do not edit `scripts/heater_dim_scaling_sweep.py`,
`scripts/heater_discovery_dim_scaling_sweep.py`, or
`scripts/heater_minimal_distillery.py`.

## Why the earlier sweeps failed

The heater is rank 1 at every `d` (`y` depends on `theta` only through
`prod_i theta_i`). Two bugs, not the one-step map:

1. **Rank rule.** The NLL drop test compared one-step NLL at `m` and `m-1`.
   Those are densities of different dimension. On the unit-width `U[1,2]` box
   a spare axis has negative entropy, so keeping it always lowers NLL and the
   ladder stopped at 2-4 (`r_hat=1` in 12/70). The probe spectrum already had
   the answer: leading axis 2.4-4.2 nats, spare axes <= 0.5 nats.
2. **SR search space.** With `pow` allowed, Operon spent its budget on
   `X_i ** (c X_j)`, which the structure predicate rejects. On `d=2` misses
   11 of 12 Pareto entries were filtered and only a single `X_1` survived.

## Fix (heater driver defaults; `scripts/oneshot_sweep.py` unchanged)

- `rank_rule="info"` in `whittle_ladder`: per-axis held-out information
  `0.5 log(prior var / residual var)` in the prior PCA basis
  (`oneshot.info_nats`). That is the one-step loss of each axis against the
  no-data null. `r_hat = #{info > 1 nat}`; no descent. Then a fresh network
  at `m_fit` (the current `oneshot.py` refit, from random init).
- SR: `--sr-allowed-symbols add,mul,div,constant,variable,sqrt`, complexity
  cap raised to `max_length`.
- New metrics columns: `rank_rule`, `info_probe_nats`, `info_final_nats`.

Re-scored on the 70 saved probe spectra, the info rule gives `r=1` in 70/70.

## Smokes (all passed)

| check | result |
|---|---|
| `d=2` full, CPU `3833111` | info `[2.25, 0.00, -0.04, -1.83]`, `r_hat=1`, `m_fit=1`, product recovered (`|rho|=1.0`), all NPE columns |
| rank only, `d=4,8,12` x 2 trials, CPU `3833113-5` | `r_hat=1` in 6/6; spare axes within 0.06 nats of 0; refit axis 3.2-4.5 nats |

`J^T J` on the `m=4` probe still reads 4 at `d=8,12`; on the `m=1` refit it
reads 1. It is a cross-check on the production map, not a screen.

## Live run

| Job | dims | where |
|---|---|---|
| `3833220` `--array=0-39` | `6 8 10 12` | V100, 18h wall |
| `3833221` `--array=0-29` | `2 3 4` | `comp` |

Outputs: `/data103/makinen/degeneracy_experiments/heater_oneshot_scaling_v4`

```bash
squeue -u "$USER" -j 3833220,3833221 -o '%.18i %.9P %.2t %.10M %R'
rg -n 'info rule|replay max|\[recovery\]|wrote ' logs/heater1s-3833220_*.out logs/heater1s-cpu-3833221_*.out
```

Known open issue: `d=2` trial 3 screens `r_hat=2` (probe info
`[2.30, -0.01, 1.23, -1.17]`). The extra axis is probably a second function
of the same product, which the data also predict. A conditional-information
rule would drop it. Left as is so all 70 rows share one rule.

After both finish:

```bash
MPLCONFIGDIR=/tmp/mplconfig python scripts/plot_heater_oneshot_scaling.py \
  --in-dir /data103/makinen/degeneracy_experiments/heater_oneshot_scaling_v4 \
  --out-dir data_scratch/heater_oneshot_plots_v4
```

Read `heater_oneshot_scaling_split.png`. Keep failures in the denominator.

Task map: `TASK = DIM_IDX * 10 + TRIAL`. CPU dims `2 3 4`. V100 dims `6 8 10 12`.
