# Rosenbrock NPE scaling — progress

Companion to [plans/rosenbrock_npe_scaling_34a0c9f3.plan.md](../.cursor/plans/rosenbrock_npe_scaling_34a0c9f3.plan.md).

## Layout

- Discovery inputs (read only):
  `/data103/makinen/degeneracy_experiments/rosenbrock_oneshot_scaling_inforank/oneshot/d{d}/trial_{t}/metrics.csv`
- NPE outputs:
  `/data103/makinen/degeneracy_experiments/rosenbrock_npe_scaling/d{d}/trial_{t}/npe_metrics.csv`
- Driver: [scripts/rosenbrock_npe_scaling.py](../scripts/rosenbrock_npe_scaling.py)
- Slurm array wrapper: [scripts/slurm/rosenbrock_npe_scaling.sh](../scripts/slurm/rosenbrock_npe_scaling.sh)
- Plotter: [scripts/plot_rosenbrock_npe_scaling.py](../scripts/plot_rosenbrock_npe_scaling.py)

## Log

- Local smoke test on the login node passed on `d=4 trial=0` under
  `timeout 300` (reduced settings, ~10 s wall-clock). All numeric columns
  finite. Ordering `oracle > discovered > raw` as expected.
- Cluster pilots submitted on `comp`, one array task each, git-archive
  snapshot at commit `e4644be`:
  - `3833687` — `d=4` trial `0` (`DIMS=4 --array=0`)
  - `3833688` — `d=16` trial `0` (`DIMS=16 --array=0`)
  - `3833689` — `d=32` trial `1` (`DIMS=32 --array=1`, one of the five
    finished `d=32` discovery trials)
- First submission hit a snapshot-creation race on the shared
  `code_snapshots/` directory: two tasks tried to `mv -T tmp
  $SNAPSHOT_DIR` simultaneously and one lost. Fixed the launcher to use
  `mkdir` as an atomic lock (commit `9e1a708`) and re-submitted the two
  failed pilots as `3833694` (d=4) and `3833695` (d=32).
- Pilots landed clean, all numeric columns finite:

  | pilot | d | trial | m_fit | inbox raw | oracle-raw | disc-raw | runtime |
  |---|---|---|---|---|---|---|---|
  | 3833694 | 4  | 0 | 2 | 0.80 | -0.03 nats | -0.04 nats |  9m |
  | 3833688 | 16 | 0 | 2 | 0.28 | +0.51 nats | +0.52 nats |  8m |
  | 3833695 | 32 | 1 | 3 | 0.07 | +0.46 nats | +0.41 nats | 11m |

  Ordering matches the plan (oracle beats raw more strongly at higher
  `d`; raw's in-box fraction collapses from 0.80 → 0.07 as `d` grows;
  discovered on its `m_fit` axes tracks oracle). d=32 runtime is
  comfortable on 8 CPUs, so H100 route stays parked.
- Discovery inventory at submit time: `d=2..16` have 10/10 `status=ok`,
  `d=32` has 5/10 (trials 0, 2, 5, 7, 9 are still rerunning).
- Full CPU array for `d in {2, 4, 8, 16}` (40 tasks, 50 concurrent slots)
  submitted as `3833713` after the pilots cleared. `--resume` skips the
  two `d=4/d=16` trial-0 outputs already on disk.
- `d=32` full array deferred: launch `DIMS="32" sbatch --array=0-9
  scripts/slurm/rosenbrock_npe_scaling.sh` once the remaining five
  discovery trials (0, 2, 5, 7, 9) land in `status=ok`. Trial 1's pilot
  row on disk means it will be skipped by `--resume`.
