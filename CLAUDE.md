# Working on the Infinity cluster (IAP)

This session runs on the login node `inf02` (`infinity02.iap.fr`), shared by
all users. The user often steers it from a phone via Happy: keep status
messages short and put the key result first.

## Login node etiquette
- No heavy compute on the login node. Training, simulations, symbolic
  regression, sweeps, notebook execution and long test suites go through Slurm.
- Fine on the login node: git, editing, `python -m py_compile`, reading logs
  and results, quick single-core checks that finish in about a minute.
- Wrap anything you are unsure about in `timeout` or submit it to `testing`.

## Submitting jobs
- Generic wrapper: `scripts/job.sbatch` (loads modules, activates the venv,
  runs the command given after the script name). Usage examples are in its
  header. Per-experiment wrappers live in `scripts/slurm/`; reuse those when one fits.
- Run `mkdir -p logs` before `sbatch`; Slurm cannot create the log directory.
- Account `iusers` (the only one). Partitions:
  - `comp` (default): CPU, AMD 128 cores / 512G nodes, 2 days max.
  - `compl`: same nodes, 4 days max.
  - `pscomp`: mixed pool including GPUs (7 nodes with 1x V100, 2 nodes with
    2x H100; request `--gres=gpu:1` or `--gres=gpu:H100:1`), 2 days max.
  - `pscompl`: same pool, 4 days max.
  - `testing`: 2 Intel nodes, 1 hour max; use for smoke tests.
- Prefer CPU partitions when GPUs are not needed; the GPU pool is small.
- Test with one array task (`--array=0`) before launching a full array.

## Monitoring
- `squeue -u $USER` for queued/running jobs.
- `sacct -j <jobid> -X --format=JobID%16,JobName%24,State,ExitCode,Elapsed`
  for finished jobs.
- Logs: `logs/<job-name>-<jobid>.out` / `.err` (arrays use `%x-%A_%a`).
- The user usually has campaigns running. Only `scancel` jobs you submitted
  in this session; ask before touching anything else.

## Environment
- Modules: `intelpython/3-2025.1.0` (+ `cuda/12.8` for GPU jobs). The site uses
  Tcl Environment Modules: `module avail`, not `module spider`.
- Venv: `/home/makinen/venvs/degen` (built by `scripts/setup_slurm_venv.sh`).

## Storage
- `$HOME` has a ~17.5G quota and is nearly full. Do not write large outputs
  into the repo or install packages into home without asking.
- Experiment outputs go to `$SCRATCH`, default
  `/data103/makinen/degeneracy_experiments`. Job scripts hardcode that default
  because sbatch jobs submitted from non-interactive shells do not see `~/.bashrc`.
