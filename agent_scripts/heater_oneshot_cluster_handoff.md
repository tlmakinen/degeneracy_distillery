# Cluster handoff: heater one-step discovery sweep

Run the heater scaling sweep on the GPU cluster. Build the curve from the symbolic coordinate the pipeline finds at every `d`. This answers reviewer QfCk.

Do not edit `scripts/heater_dim_scaling_sweep.py`, `scripts/heater_discovery_dim_scaling_sweep.py`, or `scripts/heater_minimal_distillery.py`. Those files hold the submitted numbers.

Longer design notes live in `notes/heater_oneshot_scaling.md`.

## Goal

Submit the array. After it finishes, plot Tables 1 to 4 and the discovered-axis figure.

Success, fixed before the run:

- `rank_correct`: ladder `r_hat = 1`
- `symbolic_recovered`: Spearman `|rho|` vs the analytic log-product is at least 0.99 on held-out draws

Report failures in the denominator with the failed stage. Do not drop them.

## What the script does

`scripts/heater_oneshot_discovery_sweep.py` runs one `(d, trial)`:

1. Fit `whittle_ladder` and read `r_hat`.
2. Fit `K` one-step maps at that rank (`--n-ensemble`, default 8).
3. Align them with `process_ensemble_rotation_v2`. MDL uses that `y_std`.
4. Fit pyoperon. Rescore Pareto candidates by frozen one-step NLL.
5. Train three MDN NPE arms: raw, analytic, discovered.

The NPE arm uses the NLL pick. The MDL pick is a secondary column.

## Files

| Path | Role |
|---|---|
| `degeneracy_distillery/oneshot.py` | one-step loss, ladder, ensemble |
| `scripts/heater_oneshot_discovery_sweep.py` | per-task driver |
| `scripts/slurm/heater_oneshot_sweep_gpu.sh` | 2D job array |
| `scripts/plot_heater_oneshot_scaling.py` | tables and figure |
| `notes/heater_oneshot_scaling.md` | rank rule and budget |

Reuse only. Import helpers from `scripts/heater_discovery_dim_scaling_sweep.py`. Do not copy that file.

## Submit

Work from the repo root. Env is `/home/makinen/venvs/degen`. Outputs go under `$SCRATCH` (default `/data103/makinen/degeneracy_experiments/heater_oneshot_scaling`).

1. Run one task first. This checks the three NPE arms.

```bash
sbatch --array=0 scripts/slurm/heater_oneshot_sweep_gpu.sh
```

2. If that job writes `status=ok` and all three `*_log_prob` columns, submit the full array.

```bash
sbatch --array=0-69 scripts/slurm/heater_oneshot_sweep_gpu.sh
```

Index map: `TASK = DIM_IDX * N_TRIALS + TRIAL`. Default dims are `2 3 4 6 8 10 12`. Default trials are 10.

3. Requeue a killed job with the same command. The script has `--resume`.

4. After the array finishes, plot:

```bash
python scripts/plot_heater_oneshot_scaling.py \
  --in-dir "$SCRATCH/heater_oneshot_scaling"
```

## Per-task outputs

Each task writes `$OUT_BASE/d${DIM}/trial_${TRIAL}/`:

- `metrics.csv` (one row)
- `ladder_spectra.npz`
- `expressions.json`
- `manifest.json`

The plot script globs `**/metrics.csv` and concatenates them.

## Checks after one job

Open that task `metrics.csv`. Confirm:

- `status` is `ok`
- `r_hat` is 1
- `sr_fisher_used` is `jtj` (ridge is the fallback)
- `median_y_std` is finite and greater than 0
- `expression` is a product of the `X_i`, or monotone in the product
- `raw_log_prob`, `analytic_log_prob`, and `discovered_log_prob` are all present
- `raw_on_discovered_marg` and `discovered_on_discovered_marg` are present

The two `raw` columns are one model on two axes. They should track each other. If they diverge at some `d`, discovery failed there.

## Do not

- Do not change the three existing heater scripts named above.
- Do not fold augmented map evaluations into the simulator count.
- Do not quote native `*_log_prob` as the headline. Use the common-axis `*_marg` columns.
- Do not run the NPE arms on a laptop. Local `degen` lacks `ltu-ili`.
- Do not use Metal for float64. Local CPU smokes need `JAX_PLATFORMS=cpu`.

## If a task fails

The sweep records `failed_stage` and continues. Common stages: `ladder`, `ensemble`, `align`, `sr`, `frozen`, `npe`.

If `sr` fails on `jtj`, the script retries `ridge` once. If both fail, the row is `failed` and the array moves on.

If JAX cannot find `libdevice`, keep `XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH`. The slurm script already sets this.
