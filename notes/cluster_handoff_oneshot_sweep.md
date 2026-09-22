# Cluster handoff: one-step sweep

Read this first. Then read `notes/oneshot_sweep_pipeline.md` only if you
need the protocol or the mermaid diagram.

Do not edit `scripts/*_notebook_run.py`. Those files hold the published
three-step record. This sweep is a parallel track.

## Goal

1. Run the Rosenbrock scaling sweep on the GPU partition.
2. Run the Rayleigh-Benard `n_params=3` gate at smoke budget.
3. Stop. Do not start `n_params=8` until the gate clears.

## Site

| Item | Value |
| --- | --- |
| Branch | `new_idea` |
| Repo | `$SLURM_SUBMIT_DIR` or set `REPO_DIR` |
| Venv | `/home/makinen/venvs/degen` |
| Modules | `intelpython/3-2025.1.0`, `cuda/12.8` |
| Scratch | `/data103/makinen/degeneracy_experiments` |
| Driver | `scripts/oneshot_sweep.py` |

Each array task writes its own `metrics.csv`, `config_manifest.json`,
and `run_record.json` under `OUT_DIR`. Use `--resume`.

## 1. Rosenbrock scaling

Default grid: arms `oneshot threestep`, dims `2 4 8 16 32`, 10 trials.
That is `--array=0-99`. Three-step at `d>16` writes
`status=skipped_by_budget` and exits.

Index:

```
TASK = ARM_IDX * (N_DIMS * N_TRIALS) + DIM_IDX * N_TRIALS + TRIAL
```

One task first:

```
PROBLEM=rosenbrock sbatch --array=0 scripts/slurm/oneshot_sweep_gpu.sh
```

If that task writes `status=ok` or `skipped_by_budget`, launch the grid:

```
PROBLEM=rosenbrock sbatch --array=0-99 scripts/slurm/oneshot_sweep_gpu.sh
```

Output root:

```
$SCRATCH/rosenbrock_oneshot_scaling/${ARM}/d${DIM}/trial_${TRIAL}/
```

After the array finishes:

```
python scripts/plot_oneshot_scaling.py \
  --in-dir $SCRATCH/rosenbrock_oneshot_scaling \
  --out-dir $SCRATCH/rosenbrock_oneshot_scaling
```

Expect `summary.csv` and `oneshot_scaling.{pdf,png}`.

## 2. Rayleigh-Benard gate (`n_params=3` only)

Local laptop smoke (100+100 DNS, reduced steps, no SR) did not clear
the 0.9 / 0.9 pair. Rank was correct (`r_hat=3`). Neural `R^2` vs the
Pi groups was 0.996. Nusselt cosine was 0.977. Nusselt corr was 0.874.
Three-step still scores 0/9 on these thresholds. Rerun the gate on GPU
at the notebook smoke budget before any raw-units job.

One seed first:

```
MODE=smoke N_PARAMS=3 sbatch --array=0 scripts/slurm/rb_raw_units_gpu.sh
```

If `run_record.json` has `discovery.success=true`, launch the 10-seed
gate:

```
MODE=smoke N_PARAMS=3 sbatch --array=0-9 scripts/slurm/rb_raw_units_gpu.sh
```

Output root:

```
$SCRATCH/rebuttal_discovery/rayleigh_benard_oneshot_n3/seed_${SEED}/
```

A seed clears the gate when `metrics.csv` has `gate_ok=true`. That
requires `r_hat==3`, Nusselt corr `>=0.9`, and Nusselt cosine `>=0.9`.

## Do not do this

- Do not set `N_PARAMS=8` until at least 5 of 10 gate seeds pass.
- Do not write outputs under `$HOME`. Use `SCRATCH`.
- Do not change the DNS in `scripts/rayleigh_benard_notebook_run.py`.
- Do not rerun the ladder per ensemble member. `m` is fixed at `r_hat`.

## If a task fails

Read `failed_stage` in that task `metrics.csv`. Resubmit the same
array index. `--resume` skips rows with `status=ok` or
`skipped_by_budget`.

If RB DNS raises a `complex64` / `complex128` scan error, the process
enabled `jax_enable_x64` before `sample()`. The driver samples first.
Do not import `degeneracy_distillery.oneshot` before the DNS call.

## Optional knobs

Rosenbrock launcher: `NSIMS` (default 2000), `N_TEST` (1000),
`N_TRIALS` (10), `DIMS`, `ARMS`, `OUT_BASE`, `PROBLEM_ARGS`.

RB launcher: `MODE` (`smoke` or `full`), `N_PARAMS` (keep at 3),
`MIN_NUSSELT_CORR`, `MIN_NUSSELT_COSINE`, `SR_TIME_LIMIT`, `NSIMS`,
`N_TEST`.
