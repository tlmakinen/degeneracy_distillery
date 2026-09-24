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

- (pending) local smoke test, `d=4` trial `0`, reduced epochs.
- (pending) cluster pilots on `comp`: `d=4`, `d=16`, one finished `d=32` trial.
- (pending) full CPU array `d in {2, 4, 8, 16}`; `d=32` deferred until all
  10 discovery trials land.
