# visuals/

Scripts for generating visualisations of distillery training dynamics
for the blog post / paper release.

Two parallel pipelines, both two-stage (train + snapshot, then render):

**Fisher network — `val_detF` evolution**
- `training_loop_fishnets_snapshots.py` →
  `train_fishnets_with_snapshots`: drop-in for
  `degeneracy_distillery.training_loop_fishnets.train_fishnets`.
  Writes per-model Fisher predictions on the validation set every
  `save_every` epochs.
- `make_detF_gif.py` → `make_gif`: ensemble-averages the Fisher at each
  saved epoch (same recipe as
  `degeneracy_distillery.diagnostics._aggregate_fisher`), takes the
  determinant, and animates a 2D scatter of `log10 det F` over two
  parameters (style of the first panel of `diagnose_low_information`).
  Also writes a portable `.npz` timeseries for local restyling.

**Flattener network — learned-coordinates evolution**
- `training_loop_flatten_snapshots.py` →
  `fit_flattening_with_snapshots`: drop-in for
  `degeneracy_distillery.training_loop_flatten.fit_flattening`. Every
  `save_every` epochs (past a `burn_in`) it evaluates the **main
  model** `model.apply(w, X)` on a 2D grid in `(theta_1, theta_2)`
  (extra dims pinned to the midpoint of training range, exactly as the
  legacy `do_plot=True` block does) and writes `eta_grid` to disk.
  Ensemble fine-tuning is intentionally **not** snapshotted.
- `make_eta_grid_gif.py` → `make_eta_grid_gif`: animates a 2-panel
  contour gif of `eta_1` and `eta_2` over `(theta_1, theta_2)`. Also
  writes the portable `.npz`.

## Quick start (toy mu/sigma problem, CLI)

The training script ships with a runnable `__main__` block that mirrors
the toy example at the bottom of `training_loop_fishnets.py`:

```bash
cd degeneracy_distillery/visuals
python training_loop_fishnets_snapshots.py
python make_detF_gif.py fishnets-log-snapshots/snapshots --out val_detF_mu_sigma.gif
```

## Notebook usage (drop-in for `train_fishnets`)

`degeneracy_distillery.visuals` is a true subpackage of
`degeneracy_distillery`, so after `pip install -e .` you can import
from anywhere:

```python
from degeneracy_distillery.visuals import (
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

### Flattener-network notebook usage

`fit_flattening_with_snapshots` mirrors `fit_flattening`'s positional
signature and return tuple, plus a handful of snapshot-specific
keyword arguments (`snapshots_outdir`, `save_every=5`, `burn_in=500`,
`grid_num_pts=30`, `save_initial`, `param_names`). The Fisher inputs
typically come from the output of `train_fishnets_with_snapshots`:

```python
from degeneracy_distillery.visuals import fit_flattening_with_snapshots, make_eta_grid_gif

w, ensemble_ws, output_dict = fit_flattening_with_snapshots(
    outputs["Fs"],                 # (n_models, n_train, n_p, n_p)
    outputs["theta"],              # (n_train, n_p)
    ensemble_weights=ew,
    snapshots_outdir="runs/flat_exp/snapshots",
    save_every=5,
    burn_in=500,                   # discard wild early-training coords
    grid_num_pts=30,               # 30x30 grid in (theta_1, theta_2)
    param_names=["mu", "sigma^2"],
    output_prefix="runs/flat_exp/flattened_coords",
    # ...any other fit_flattening kwargs (hidden_size, n_layers, lr_*, etc.)
)

gif_path, epochs, data_path = make_eta_grid_gif(
    "runs/flat_exp/snapshots",
    out_path="runs/flat_exp/eta_grid.gif",
    fps=12,
    cmap="viridis",
    levels=20,
)
display_gif(gif_path)
```

The gif is a 2-panel `contourf` showing `eta_1(theta_1, theta_2)` on
the left and `eta_2(theta_1, theta_2)` on the right, with the
super-title updating to the current global epoch — the same layout as
the legacy `do_plot=True` block.

Why `burn_in`? The flattener starts very far from sensible coordinates;
the first few hundred epochs produce visually distracting frames. With
`burn_in=500` (default) snapshotting only kicks in once the network is
in a recognisable basin.

## Snapshot layouts

**Fisher side** (`train_fishnets_with_snapshots`):

```
outdir/
    fishnets_outputs.npz              # final ensemble predictions (same as train_fishnets)
    snapshots/
        metadata.json                 # num_models, ensemble_weights, save_every, per-model epoch lists, ...
        theta_val.npy                 # (n_val, n_params), saved once
        model_00/
            epoch_00000.npz           # F_val: (n_val, n_params, n_params), untrained
            epoch_00005.npz
            ...
        model_01/
            ...
```

Each `epoch_{j:05d}.npz` is a compressed `npz` with a single key
`F_val`. Storage scales as
`num_models * (train_epochs / save_every) * n_val * n_params**2 * 4 B`.

**Flattener side** (`fit_flattening_with_snapshots`):

```
snapshots_outdir/
    metadata.json                     # save_every, burn_in, param_names, grid_num_pts, saved_epochs, ...
    grid_axes.npz                     # xs_mesh, ys_mesh, X_grid (shape (G*G, n_params)), min_x, max_x
    epoch_00500.npz                   # eta_grid: (G*G, n_params)  -- first kept frame after burn-in
    epoch_00505.npz
    ...
```

Each `epoch_{j:05d}.npz` is a compressed `npz` with a single key
`eta_grid`. Storage scales as
`((total_epochs - burn_in) / save_every) * grid_num_pts**2 * n_params * 4 B`
— for the defaults (1500 effective epochs, save_every=5, 30x30 grid,
n_params=2) that's `300 * 900 * 2 * 4 B ≈ 2 MB`, trivial compared to
the Fisher snapshots.

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

Both `make_detF_gif.py` and `make_eta_grid_gif.py` write a
self-contained `.npz` alongside the gif (same stem). Copy a single
file to your laptop to re-render plots / gifs with custom fonts,
axis styling, etc.

### Fisher side (`make_detF_gif.py`)

| key                | shape                 | meaning                                                                 |
| ------------------ | --------------------- | ----------------------------------------------------------------------- |
| `epochs`           | `(n_frames,)` int     | epoch number for each frame                                             |
| `theta_val`        | `(n_val, n_params)`   | validation parameters (scatter support)                                 |
| `det_F`            | `(n_frames, n_val)`   | ensemble-averaged det F at each frame                                   |
| `log_det_F`        | `(n_frames, n_val)`   | log10 \|det F\| at each frame (what the GIF colours by)                 |
| `param_names`      | `(n_params,)` str     | display names for axis labels                                           |
| `ensemble_weights` | `(num_models,)`       | per-member weights used in the weighted Fisher average                  |
| `vmin`, `vmax`     | scalar                | the colour-scale limits chosen by the percentile rule                   |
| `snapshots_dir`    | str                   | absolute path the data was built from (provenance)                      |

### Flattener side (`make_eta_grid_gif.py`)

| key                | shape                                       | meaning                                                       |
| ------------------ | ------------------------------------------- | ------------------------------------------------------------- |
| `epochs`           | `(n_frames,)` int                           | global epoch number for each frame                            |
| `xs_mesh`          | `(G, G)`                                    | mesh of theta_1 values used for the contour grid              |
| `ys_mesh`          | `(G, G)`                                    | mesh of theta_2 values                                        |
| `X_grid`           | `(G*G, n_params)`                           | flat list of grid points used to evaluate the model           |
| `eta_stack`        | `(n_frames, G*G, n_params)`                 | raw model output per grid point per frame                     |
| `eta_grid_2d`      | `(n_frames, G, G, n_params)`                | reshaped to a 2D contour grid for direct plotting             |
| `eta_vmins`        | `(2,)`                                      | per-panel vmin (eta_1, eta_2) used for fixed colour scale     |
| `eta_vmaxs`        | `(2,)`                                      | per-panel vmax                                                |
| `param_names`      | `(n_params,)` str                           | axis labels                                                   |
| `min_x`, `max_x`   | `(n_params,)`                               | training-range bounds (for axis limits / extras)              |
| `snapshots_dir`    | str                                         | absolute path the data was built from (provenance)            |

When `make_eta_grid_gif(..., align_mode=...)` is non-default, the `.npz`
also carries the alignment metadata so a local replotter can faithfully
reproduce the gif:

| key                            | shape                            | meaning                                                                 |
| ------------------------------ | -------------------------------- | ----------------------------------------------------------------------- |
| `align_mode`                   | str                              | `"linear_residual" / "nonlinearity_rotation" / "both"`                  |
| `align_reference_frame_index`  | int                              | which frame of `eta_stack` defined the transform (default last)         |
| `affine_A`                     | `(n_params, 2)`                  | best affine fit `eta ≈ A @ theta + b` on the reference frame            |
| `affine_b`                     | `(n_params,)`                    | affine offset                                                           |
| `nonlin_R`                     | `(n_params, n_params)`           | orthogonal rotation that concentrates Jacobian variance up-front        |
| `nonlin_sigma`                 | `(n_params,)`                    | descending nonlinearity-energy spectrum (singular values of ΔJ)         |
| `eta_stack_raw`, `eta_grid_2d_raw` | same as `eta_stack` / `eta_grid_2d` | un-aligned snapshots, so you can switch alignment modes locally |

Minimal local replotting examples:

```python
# Fisher side
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
```

```python
# Flattener side
import numpy as np
import matplotlib.pyplot as plt

d = np.load("eta_grid.npz", allow_pickle=False)
xs, ys = d["xs_mesh"], d["ys_mesh"]
eta = d["eta_grid_2d"]                    # (n_frames, G, G, n_p)
vmins, vmaxs = d["eta_vmins"], d["eta_vmaxs"]

plt.rcParams.update({"font.family": "serif", "font.size": 14})
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for k, ax in enumerate(axes):
    cs = ax.contourf(xs, ys, eta[-1, :, :, k],
                     levels=np.linspace(vmins[k], vmaxs[k], 21),
                     cmap="viridis", extend="both")
    ax.set_xlabel(str(d["param_names"][0]))
    ax.set_ylabel(str(d["param_names"][1]))
    ax.set_title(rf"$\eta_{k + 1}$")
    plt.colorbar(cs, ax=ax)
plt.savefig("final_frame.pdf")
```

CLI flags controlling the data dump (both gif scripts):

* `--data-out PATH` — explicit output path; defaults to the gif path
  with `.npz` extension.
* `--no-data` — skip the data dump (gif only).
* `--no-gif` — skip the gif (useful on a server: download just the
  small `.npz`).

## Popping the nonlinear part of eta out (flattener gif only)

`make_eta_grid_gif` has an `align_mode` argument that transforms `eta`
so the nonlinear component dominates the contours. The transform is
computed once on `align_reference_frame` (default last) and applied
identically to every frame:

| `align_mode`              | what it does                                                                                              | panel label                            |
| ------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `"none"` *(default)*      | raw `eta`                                                                                                 | `η_k`                                  |
| `"linear_residual"`       | subtract best affine fit `eta ≈ A @ theta + b` (computed on the reference frame)                          | `η_k − (Aθ + b)_k`                    |
| `"nonlinearity_rotation"` | apply orthogonal rotation `R` that concentrates Jacobian variance into the leading η-axes                | `(R η)_k`                              |
| `"both"`                  | linear residual, then rotation                                                                            | `(R(η − Aθ − b))_k`                    |

CLI: `--align-mode {none,linear_residual,nonlinearity_rotation,both}`
and `--align-reference-frame <int>` (default `-1`).

For most quick visualisations `linear_residual` is the most directly
compelling: it strips the trivial linear gradient and leaves the
network's learned curvature. `nonlinearity_rotation` is the
algorithm from `degeneracy_distillery.align_coords.nonlinearity_rotation`,
vendored JAX-free into `make_eta_grid_gif`. `both` combines them.

## Showing the flatness Frobenius score in the gif title

`fit_flattening_with_snapshots` now also evaluates the validation
flatness score `mean_b ||Q_b − I||_F` (with `Q = J^{-T} F J^{-1}`) at
each snapshot and stores it as `frob_score` in the per-epoch `.npz`
and `saved_frob_scores` in `metadata.json`.

`make_eta_grid_gif` exposes a `show_loss=False` switch (CLI flag
`--show-loss`) to display the score next to the epoch number:

```python
gif_path, _, data_path = make_eta_grid_gif(
    "runs/flat_exp/snapshots",
    out_path="runs/flat_exp/eta_grid.gif",
    show_loss=True,        # append "||Q-I||_F = 0.43" to the title
    loss_fmt=".2f",        # one or two decimals (default ".2f")
)
```

Other knobs:

* `loss_label` — override the math-text label (default
  `r"\|Q-I\|_F"`).
* `loss_fmt` — Python format spec, e.g. `".1f"`, `".2e"`. Very large
  or very small values automatically fall back to scientific.

For snapshots written before this feature existed, `show_loss=True`
prints a one-time warning and falls back to epoch-only titles
(no crash). The portable `.npz` always includes a `frob_scores`
column (`NaN`-filled for missing values) so a local replotter can
draw a small loss inset alongside the contour gif.

Same transform is also exposed as a function so you can apply it
to an already-saved `.npz` locally:

```python
from degeneracy_distillery.visuals import align_eta_stack
import numpy as np

d = np.load("eta_grid.npz", allow_pickle=False)
eta_aligned, info = align_eta_stack(
    d["eta_stack_raw"] if "eta_stack_raw" in d.files else d["eta_stack"],
    X_grid=d["X_grid"],
    mode="linear_residual",
    grid_num_pts=int(d["xs_mesh"].shape[0]),
    xs_axis=d["xs_mesh"][0, :],
    ys_axis=d["ys_mesh"][:, 0],
)
```
