# Heater sweep with the one-step loss

This note records the one-step heater scaling sweep. The script is
`scripts/heater_oneshot_discovery_sweep.py`. It does not edit
`scripts/heater_dim_scaling_sweep.py`,
`scripts/heater_discovery_dim_scaling_sweep.py`, or
`scripts/heater_minimal_distillery.py`. The submitted numbers stay
reproducible.

The sweep builds the scaling curve from the symbolic coordinate that
the pipeline finds at every ambient `d`. That is the answer to
reviewer QfCk.

## Pipeline

The simulator is the chain-product heater

    y(t) = (prod_i theta_i) * (1 - exp(-t / tau)) + noise(t)

with prior `U[1, 2]^d`. The Fisher is rank 1 at every `d`.

1. A warm-started descending ladder fits the one-step map and
   selects `r_hat`.
2. `K` one-step maps train at that `r_hat` (default `--n-ensemble 8`).
   Member 0 is the ladder map. The rest train on bootstrap resamples
   of the same pairs, warm-started from the ladder.
3. `process_ensemble_rotation_v2` aligns the `K` maps. The aligned
   mean is `y`. The aligned spread is `y_std`. MDL uses that spread.
4. pyoperon fits the kept axis on the aligned set plus self-drawn
   prior draws pushed through the same ensemble and rotations.
5. Frozen-candidate rescoring selects an expression by held-out
   one-step NLL.
6. Three MDN NPE arms train on raw theta, the analytic log-product,
   and the selected expression.

The one-step core lives in `degeneracy_distillery/oneshot.py`. The
sweep imports the simulator, the projector, the NPE arms, and the
aggregate writers from
`scripts/heater_discovery_dim_scaling_sweep.py`.

## One-step loss

The map is

    eta_m(theta) = s * (A_m g_phi(theta)),   A_m in R^{m x m0}

with Jacobian `J = d eta_m / d theta`. The loss is

    L = E[ 0.5 ||eta_phi(theta) - eta_psi(x)||^2 - 0.5 log det(J J^T) ]
      + (m / 2) log 2 pi

An informed axis has unit posterior variance under this loss, so
`y_std = 1` is the natural scale for symbolic regression.

## Ladder rank rule

`whittle_ladder` does not use the Fisher. That avoids the mean-Fisher
trap in `notes/heater_discovery_scaling.md`.

1. Fit at `m = M_PROBE` (default 4) for `--probe-steps` (default 20000).
2. Read the prior spectrum `lam, V = eigh(Cov_pi(eta))` and the
   held-out NLL.
3. Drop the lowest-lambda axis. Set
   `A_{m-1} = V[:, :m-1].T @ A_m`. Keep the trunk `g_phi`. Fine-tune
   for `--rung-steps` (default 5000).
4. Accept the drop when it is free. Let
   `dif = nll_{m-1} - nll_m` on held-out samples. Accept while
   `dif.mean() <= k * dif.std(ddof=1) / sqrt(n)` with `--ladder-k`
   (default 2.0). The first rejected drop fixes `r_hat`.

Each rung rotates into the canonical basis, so the kept axis is axis
0. The script does not need a separate axis scorer.

The per-sample `J^T J` eigengap rank is still computed and stored.
It is an independent cross-check. Both rules should give `r = 1` on
this simulator.

`--cold-rungs` refits each rung from scratch. Use this only as a
control.

Per-rung audit fields go to `ladder_spectra.npz`: `lam`, the nats
profile `0.5 log(lam)`, held-out NLL, `dif.mean()` and its SE,
residual variance per axis, and `median sqrt(det J J^T)`. Residual
variance on an informed axis should sit near 1.

## Symbolic regression

The built-in augmenter needs `flatten_model`, `ensemble_w`,
`rotmats`, and `ensemble_weights` together. This path has none of
those, so the script draws its own `X_sr` uniform on the prior box
and omits all four arguments.

- `X`: training thetas plus `n_aug = --sr-n-aug-per-dim * d` new
  draws.
- `y`: aligned ensemble mean of `eta`.
- `y_std`: aligned ensemble spread. This is the uncertainty MDL
  needs. A single map cannot supply it.
- `dy_sr`: aligned ensemble-mean Jacobian.
- `Fs = J^T J`, the implied Fisher. This is the stationarity claim
  of `new_idea/check_equivalence.py`.
- `max_length = max(25, 4d + 8)`. A `d`-way product needs about
  `2d - 1` tokens. The default 25 cannot represent a 12-way product.

`Fs = J^T J` has rank `r_hat = 1`, so it is singular.
`--sr-fisher {jtj,ridge,identity}` selects the matrix that
`analyze_equations` sees. `ridge` adds `eps I`. If the `jtj` path
raises, the script retries `ridge` once and records
`sr_fisher_used`.

## Frozen-candidate selector

The MDL pick is recorded. The NPE arm does not use it as the
primary expression.

For the top `--n-frozen-candidates` Pareto entries (default 8), the
script freezes `eta` to the candidate and refits only the estimator,
the scale `s`, and the mixing `A`. Selection uses held-out one-step
NLL. Complexity breaks ties.

Recorded fields:

- `expression`: the NLL pick, used by the discovered NPE arm.
- `expression_mdl`: the MDL pick.
- `picks_agree`: whether the two picks match.
- `nll_gap = nll_symbolic - nll_neural`: the price of
  symbolisation, in the same loss the map trained under.

## NPE arms and tables

All three arms use an MDN. The architecture stays fixed across arms.

Two exact common axes:

- Analytic: raw theta samples go through the log-product. The
  analytic arm is the identity on that axis.
- Discovered: raw theta samples go through the symbolic projector.
  The discovered arm is the identity on that axis. No oracle
  knowledge enters this block.

`scripts/plot_heater_oneshot_scaling.py` globs per-task
`metrics.csv` files and writes Tables 1 to 4. Gaps are paired
within trial before the script aggregates. Each of the four
common-axis columns gets a slope in nats per dimension with a
standard error. The headline curve is
`discovered_on_discovered_marg` against
`raw_on_discovered_marg`.

The two `raw` columns are one model scored on two axes. They should
track each other. Divergence at some `d` means discovery failed
there, not that inference failed.

## Success criteria

These are fixed before the sweep runs.

- `rank_correct`: the ladder returns `r_hat = 1`.
- `symbolic_recovered`: Spearman `|rho|` between the discovered
  coordinate and the analytic log-product, on held-out draws, is at
  least `--recovery-corr-thresh` (default 0.99).

Expect rank detection to hold past the `d` at which symbolic
recovery breaks. That split is the scaling limit, not a result to
hide. Report failures in the denominator with the failed stage.

## Budget

Simulator calls per run are `--nsims + --n-test + --n-marginal-val`.
The last term is a fresh set for the common-axis scores. Pass
`--independent-npe-sims` to give the NPE arms their own training
draws.

Augmented map evaluations are not simulator calls. They live in
`n_augmented_coordinate_evaluations`.

`--ladder-only` stops after the ladder. It does not import torch or
ltu-ili. `--skip-npe` runs the ladder, SR, and frozen rescoring.

## Cluster

`scripts/slurm/heater_oneshot_sweep_gpu.sh` maps
`SLURM_ARRAY_TASK_ID` to `(dim, trial)`:

    TASK = DIM_IDX * N_TRIALS + TRIAL

7 dims times 10 trials is `--array=0-69`. Outputs go under
`$SCRATCH`. After the array finishes:

    python scripts/plot_heater_oneshot_scaling.py \
      --in-dir "$SCRATCH/heater_oneshot_scaling"

## Local checks

Prefix local runs with `JAX_PLATFORMS=cpu`. The Metal backend cannot
legalise float64.

Ladder only, at two dimensions:

    JAX_PLATFORMS=cpu python scripts/heater_oneshot_discovery_sweep.py \
      --dims 2 4 --num-trials 1 --ladder-only \
      --out-dir heater_oneshot_ladder_smoke

Ladder, SR, and frozen rescoring, no NPE:

    JAX_PLATFORMS=cpu python scripts/heater_oneshot_discovery_sweep.py \
      --dims 2 --num-trials 1 --skip-npe \
      --out-dir heater_oneshot_sr_smoke

The NPE arms need ltu-ili, sbi, and lampe. Run one
`--dims 2 --num-trials 1` job on the cluster before the full array.

## What has been verified so far

`--ladder-only` at `d = 2` and `d = 4` on CPU returned `r_hat = 1`.
The `J^T J` eigengap rank was also 1. The accepted drops were free
on the paired held-out NLL test.

`--skip-npe` at `d = 2` first ran SR with `y_std = 1`. After the
ensemble was added, MDL reads `y_std` from the aligned maps. `Fs =
J^T J` did not break the Frobenius scorer, so the ridge fallback
was not used. The NLL pick was affine in `X1 X2`. Spearman `|rho|`
against the analytic log-product was 1.0 on held-out draws.

The three NPE arms have not been run locally. They need the cluster.
