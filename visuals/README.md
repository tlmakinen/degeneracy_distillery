# visuals/

Scripts for generating visualisations of distillery training dynamics for
the blog post / paper release.

The pipeline is two stages:

1. **Train + snapshot.** Run a modified copy of
   `degeneracy_distillery.training_loop_fishnets.train_fishnets` that
   writes per-model Fisher predictions on the validation set every
   `save_every` epochs. Because the ensemble is trained sequentially in
   the original code, we cannot ensemble-average at training time;
   instead we store every member's Fisher snapshot and let the GIF
   builder reconstruct the ensemble average on a common epoch axis.

2. **Build the GIF.** Load the snapshots, ensemble-average the Fisher at
   each saved epoch using the same weighted mean as
   `degeneracy_distillery.diagnostics._aggregate_fisher`, take the
   determinant, and animate a 2D scatter of `log10 det F` over two
   parameters in the style of the first panel of
   `diagnose_low_information`. The script also writes a portable
   `.npz` timeseries dump (alongside the `.gif`, same stem) containing
   everything needed to re-render the plots locally with custom font /
   styling.

## Quick start (toy mu/sigma problem, CLI)

The training script ships with a runnable `__main__` block that mirrors
the toy example at the bottom of `training_loop_fishnets.py`:

```bash
cd visuals
python training_loop_fishnets_snapshots.py
python make_detF_gif.py fishnets-log-snapshots/snapshots --out val_detF_mu_sigma.gif
```

## Notebook usage (drop-in for `train_fishnets`)

`visuals/` is a Python package with `__init__.py`, so it imports just
like `degeneracy_distillery`. From a notebook in `notebooks/` (sibling
of `visuals/`), add the repo root to `sys.path` once, then import:

```python
import sys, os
sys.path.insert(0, os.path.abspath(".."))   # repo root on sys.path

from visuals import (
    train_fishnets_with_snapshots,   # drop-in for train_fishnets
    make_gif,                        # build the .gif + .npz
    display_gif,                     # inline-display the .gif
)
```

`train_fishnets_with_snapshots` has the **same positional signature
and same return tuple** as `train_fishnets`, plus four snapshot-
related kwargs (`save_every`, `save_initial`, `save_final`,
`param_names`):

```python
ws, ensemble_weights, models, data_scaler, outputs = train_fishnets_with_snapshots(
    theta, data, theta_test, data_test,
    data_shape=n_d,
    num_models=20,
    train_epochs=4000,
    train_min_epochs=100,
    patience=20,
    lr=5e-5,
    outdir="runs/my_experiment",
    save_every=5,
    save_initial=True,                # snapshot the untrained init too
    param_names=["mu", "sigma^2"],
)
```

Then build the gif + portable `.npz` and display inline:

```python
gif_path, epochs, data_path = make_gif(
    "runs/my_experiment/snapshots",
    out_path="runs/my_experiment/val_detF.gif",
    fps=12,
    cmap="viridis",
)
display_gif(gif_path)   # last expression in cell → inline display
```

If you only want the data (e.g. you're on a server and want to
restyle locally), pass `save_gif=False` and copy just the `.npz`:

```python
_, epochs, data_path = make_gif(
    "runs/my_experiment/snapshots",
    save_gif=False,
    data_out_path="runs/my_experiment/val_detF.npz",
)
```

## Snapshot layout

```
outdir/
    fishnets_outputs.npz              # final ensemble predictions (same as train_fishnets)
    snapshots/
        metadata.json                 # num_models, ensemble_weights, save_every, per-model epoch lists, ...
        theta_val.npy                 # (n_val, n_params), saved once
        model_00/
            epoch_00000.npz           # F_val: (n_val, n_params, n_params), untrained
            epoch_00005.npz
            epoch_00010.npz
            ...
        model_01/
            ...
```

Each `epoch_{j:05d}.npz` is a compressed `npz` with a single key
`F_val`. Storage scales as
`num_models * (train_epochs / save_every) * n_val * n_params**2 * 4 B`.

## Notes

- Members that stop early (patience-based early stopping) keep
  contributing to later frames at their last snapshot, so the ensemble
  average is always well-defined.
- `metadata.json` is rewritten after every model finishes, so a killed
  run still leaves a usable snapshot directory: just point
  `make_detF_gif.py` at it.
- The GIF colormap is held fixed across frames using percentile clipping
  on the global stack of `log10 det F` (defaults to 2nd / 98th
  percentiles); override with `--norm-percentiles LO HI`.
- The GIF writer is `pillow` by default (no extra dependency); pass
  `--writer imagemagick` if you have it installed and prefer it.

## Portable `.npz` timeseries

Every call to `make_detF_gif.py` also writes a self-contained
`.npz` (default: same stem as the gif). Copy this single file to your
laptop to re-render plots / gifs with custom fonts, axis styling, etc.

Arrays / scalars inside the `.npz`:

| key                | shape                 | meaning                                                                 |
| ------------------ | --------------------- | ----------------------------------------------------------------------- |
| `epochs`           | `(n_frames,)` int     | epoch number for each frame                                             |
| `theta_val`        | `(n_val, n_params)`   | validation parameters (scatter support)                                 |
| `det_F`            | `(n_frames, n_val)`   | ensemble-averaged det F at each frame                                   |
| `log_det_F`        | `(n_frames, n_val)`   | log10 |det F| at each frame (what the GIF colours by)                   |
| `param_names`      | `(n_params,)` str     | display names for axis labels                                           |
| `ensemble_weights` | `(num_models,)`       | per-member weights used in the weighted Fisher average                  |
| `vmin`, `vmax`     | scalar                | the colour-scale limits chosen by the percentile rule                   |
| `snapshots_dir`    | str                   | absolute path the data was built from (provenance)                      |

Minimal local replotting example::

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    d = np.load("val_detF_mu_sigma.npz", allow_pickle=False)
    epochs, theta, log_det = d["epochs"], d["theta_val"], d["log_det_F"]
    norm = Normalize(vmin=float(d["vmin"]), vmax=float(d["vmax"]))

    plt.rcParams.update({"font.family": "serif", "font.size": 14})
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sc = ax.scatter(theta[:, 0], theta[:, 1], c=log_det[-1],
                    cmap="viridis", norm=norm, s=12)
    ax.set_xlabel(str(d["param_names"][0]))
    ax.set_ylabel(str(d["param_names"][1]))
    plt.colorbar(sc, ax=ax, label=r"$\log_{10}\,\det F_\theta$")
    plt.savefig("final_frame.pdf")

CLI flags controlling the data dump:

* `--data-out PATH` — explicit output path; defaults to the gif path
  with `.npz` extension.
* `--no-data` — skip the data dump (gif only).
* `--no-gif` — skip the gif (useful on a server: download just the
  small `.npz`).
