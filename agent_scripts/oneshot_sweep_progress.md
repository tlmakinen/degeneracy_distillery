# One-step sweep: progress (23 Sep 2026, ~17:20 CEST)

Covers `scripts/oneshot_sweep.py`: Rosenbrock scaling and the
Rayleigh-Benard `n_params=3` gate. The heater sweep has its own note,
`agent_scripts/heater_oneshot_progress.md`.

## Protocol (commit `114db61`)

1. Input screen (group-L1 on theta). Diagnostic only. It does not choose
   the rank or remove coordinates.
2. Probe at `m_probe` latents (Rosenbrock 4, RB 5), 20k steps.
3. Rank, `--rank-rule info` (default). Per probe axis, on held-out data,
   `info_j = 0.5 log(prior var of eta_j / var of (eta_j - eta_hat_j(x)))`.
   `r_hat` counts axes above `--info-floor` (0.5 nats since 23 Sep 23:20;
   the first grid used 1 nat), capped at `d`.
   The old NLL drop test (`--rank-rule nll`) depends on the units of theta
   and read `r_hat=1` in 49/49 Rosenbrock trials.
4. `m_fit = r_hat`, plus one when the probe latent spectrum drops by less
   than 10x after axis `r_hat`. Capped at `min(m_probe, d)`.
5. Fresh refit from random init at `m = m0 = m_fit`, 20k steps. This map
   is scored (`r2_true_min`, RB Nusselt gate).
6. Ensemble, K=4 at `m_fit`, warm-started from the refit, bootstrap.
7. Alignment on the test set (`process_ensemble_rotation_v2`).
8. SR rows: aligned test points plus `1000 d` prior draws. The prior draws
   go through `apply_ensemble_alignment`, so both share one origin.
   Before `4c20bf8`/`61fcb30` they were only rotated, and sat a constant
   27-160 eta units away from the aligned rows.
9. Operon per component, `pow` kept, complexity cap `max(20, max_length)`.
10. Frozen rescore of up to 8 stacks on held-out one-step NLL. Before
    `61fcb30` every candidate failed (`th_fit[:1]` shape bug), so
    `expression` was always the MDL pick.

End-to-end check, Rosenbrock `d=4`, reduced budget: info
`[2.85, 2.62, 0.17, -0.01]` -> `r_hat=2`, `r2_true_min=0.9993`,
`r2_sr_min=0.99999`, frozen rescore ran.

## Live run

Output: `/data103/makinen/degeneracy_experiments/rosenbrock_oneshot_scaling_inforank`

| Job | Tasks | Code | Status at 17:20 |
|---|---|---|---|
| `3833302` CPU | oneshot `d=2` (0-9) | `code_snapshots/oneshot_828be86` (exact `git archive` of `114db61`; named before the trailer rewrite) | queued/running |
| `3833239` CPU | oneshot `d=4..32` (10-49) | `code_snapshots/oneshot_20260923_1701` | running |
| `3833240` GPU | threestep all dims (50-99) | `code_snapshots/oneshot_20260923_1701` | running; `d=32` skipped by budget |

`oneshot_20260923_1701` matches `114db61` in every code file except the
`r_hat <= d` cap, which cannot bind for `d >= 4` with `m_probe=4`.
The `d=2` tasks were resubmitted because, without the cap, the collapsed
spare probe axes read above 1 nat (`r_hat=3-4` in 8/10); the refit was
already capped at `m=2`.

`d=4` probe info so far: 10/10 give `r_hat=2`, spare axes 0.0-0.23 nats.

Plot when done:

```bash
python scripts/plot_oneshot_scaling.py \
  --in-dir /data103/makinen/degeneracy_experiments/rosenbrock_oneshot_scaling_inforank \
  --out-dir /data103/makinen/degeneracy_experiments/rosenbrock_oneshot_scaling_inforank/plots
```

Logs are under each snapshot's `logs/` directory.

## Rayleigh-Benard

Gate `3833105` (old NLL rank rule, alignment offset present, corr >= 0.8):
7/10 pass. Failures are all rank: seeds 2 and 6 `r_hat=2` (refit at 3,
fit fine), seed 3 `r_hat=1`. With the 1-nat info floor RB would likely read
`r_hat=1` (probe `0.5 log lam ~ [2.4, 0.75, 0.55]`), so the floor must be
relaxed for RB before any rerun. `n_params=8` not launched.

## Superseded outputs (contrast only)

- `rosenbrock_oneshot_scaling_projected_a41fd21`: projected probe map.
- `rosenbrock_oneshot_scaling`: fresh refit, NLL rank rule, alignment
  offset, frozen rescore broken. `threestep/d16` all failed on the
  `fit_flattening` grid overflow (fixed in `61fcb30`).
- `rebuttal_discovery/rayleigh_benard_oneshot_n3*`: pre-fix RB runs.
