"""
Generate a .gif of the flattener's learned coordinates ``eta`` evolving
over training, plotted as two contour panels (``eta_1``, ``eta_2``) over
``(theta_1, theta_2)`` — exactly the layout of the legacy
``do_plot=True`` block in
``degeneracy_distillery.training_loop_flatten.fit_flattening``.

Consumes the per-epoch snapshots written by
:func:`visuals.training_loop_flatten_snapshots.fit_flattening_with_snapshots`.

By default also saves a portable ``.npz`` timeseries (same stem as the
``.gif``) containing everything needed to re-render plots locally with
custom styling, without access to the snapshot directory.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def _snapshot_files(snapshots_dir: str) -> List[Tuple[int, str]]:
    pairs: List[Tuple[int, str]] = []
    for path in glob.glob(os.path.join(snapshots_dir, "epoch_*.npz")):
        stem = os.path.basename(path).split("_")[1].split(".")[0]
        pairs.append((int(stem), path))
    pairs.sort(key=lambda t: t[0])
    return pairs


def load_eta_snapshots(snapshots_dir: str):
    """Load metadata, the eval grid, and the stack of ``eta_grid`` frames.

    Returns
    -------
    meta : dict
        Contents of ``metadata.json``.
    grid : dict with keys ``xs_mesh``, ``ys_mesh``, ``X_grid``, ``min_x``, ``max_x``
        The 2D evaluation grid used by the trainer.
    epochs : list of int
        Sorted snapshot epoch numbers.
    eta_stack : (n_frames, grid_num_pts**2, n_params) array
        Per-epoch ``eta`` values evaluated on the grid.
    """
    with open(os.path.join(snapshots_dir, "metadata.json")) as f:
        meta = json.load(f)
    grid_npz = np.load(os.path.join(snapshots_dir, "grid_axes.npz"))
    grid = {k: grid_npz[k] for k in grid_npz.files}

    pairs = _snapshot_files(snapshots_dir)
    if not pairs:
        raise RuntimeError(f"no epoch_*.npz snapshots found under {snapshots_dir}")
    epochs = [p[0] for p in pairs]
    eta_stack = np.stack(
        [np.load(p[1])["eta_grid"] for p in pairs], axis=0
    )  # (n_frames, num_pts**2, n_params)
    return meta, grid, epochs, eta_stack


# ----------------------------------------------------------------------------
# Portable npz dump
# ----------------------------------------------------------------------------

def save_eta_timeseries_npz(
    out_path: str,
    epochs: Sequence[int],
    grid: dict,
    eta_stack: np.ndarray,
    param_names: Sequence[str],
    eta_vmins: Sequence[float],
    eta_vmaxs: Sequence[float],
    snapshots_dir: str,
) -> str:
    """Save a self-contained ``.npz`` for local replotting.

    Keys: ``epochs``, ``xs_mesh``, ``ys_mesh``, ``X_grid``,
    ``eta_stack`` (n_frames, num_pts**2, n_params), ``eta_grid_2d``
    (n_frames, num_pts, num_pts, n_params), ``param_names``,
    ``eta_vmins``, ``eta_vmaxs``, ``min_x``, ``max_x``, ``snapshots_dir``.
    """
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    num_pts = int(grid["xs_mesh"].shape[0])
    n_p = int(eta_stack.shape[-1])
    eta_grid_2d = eta_stack.reshape(-1, num_pts, num_pts, n_p)
    np.savez_compressed(
        out_path,
        epochs=np.asarray(epochs, dtype=int),
        xs_mesh=np.asarray(grid["xs_mesh"]),
        ys_mesh=np.asarray(grid["ys_mesh"]),
        X_grid=np.asarray(grid["X_grid"]),
        eta_stack=np.asarray(eta_stack),
        eta_grid_2d=np.asarray(eta_grid_2d),
        param_names=np.asarray(list(param_names)),
        eta_vmins=np.asarray(eta_vmins, dtype=float),
        eta_vmaxs=np.asarray(eta_vmaxs, dtype=float),
        min_x=np.asarray(grid["min_x"]),
        max_x=np.asarray(grid["max_x"]),
        snapshots_dir=np.asarray(os.path.abspath(snapshots_dir)),
    )
    return out_path


# ----------------------------------------------------------------------------
# GIF rendering
# ----------------------------------------------------------------------------

def make_eta_grid_gif(
    snapshots_dir: str,
    out_path: Optional[str] = "eta_grid_evolution.gif",
    fps: int = 10,
    param_names: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (10.0, 4.0),
    cmap: str = "viridis",
    levels: int = 20,
    norm_percentiles: Tuple[float, float] = (2.0, 98.0),
    title_prefix: str = "learned coordinates",
    writer: str = "pillow",
    data_out_path: Optional[str] = "auto",
    save_gif: bool = True,
    burn_in: int = 0,
) -> Tuple[Optional[str], List[int], Optional[str]]:
    """Render a 2-panel contour gif of ``eta_1`` and ``eta_2`` vs
    ``(theta_1, theta_2)`` evolving over training.

    Parameters
    ----------
    snapshots_dir : str
        Directory written by
        :func:`visuals.training_loop_flatten_snapshots.fit_flattening_with_snapshots`.
    out_path : str or None
        Where to save the .gif. Ignored if ``save_gif=False``.
    fps : int
        Frames per second.
    param_names : sequence of str, optional
        Override the axis labels (need at least two). Defaults to the
        names recorded in ``metadata.json``.
    figsize : (w, h)
        Matplotlib figure size for the 2-panel layout.
    cmap : str
        Colormap (default ``"viridis"`` to match the legacy plot).
    levels : int
        Number of contour levels per panel.
    norm_percentiles : (lo, hi)
        Percentiles of each ``eta_k`` (across all frames) used to fix
        per-panel colour scales, so the colormap is consistent across
        frames.
    title_prefix : str
        Prepended to the per-frame super-title.
    writer : str
        matplotlib animation writer (``"pillow"`` or ``"imagemagick"``).
    data_out_path : str, None, or ``"auto"``
        Where to save the portable ``.npz`` timeseries. ``"auto"`` (the
        default) saves alongside the gif with the same stem; ``None``
        disables the dump; any other string is used verbatim.
    save_gif : bool
        If False, skip rendering the gif (useful when you only want the
        ``.npz`` to ship locally for replotting).
    burn_in : int, default 0
        Render-time burn-in: drop any snapshot whose epoch number is
        strictly less than this value before computing percentile
        colour limits, writing the ``.npz``, and animating. Use this to
        chop a few extra early epochs without re-running training
        (independent of, and stacks with, the training-time ``burn_in``
        in :func:`fit_flattening_with_snapshots`).

    Returns
    -------
    (gif_path_or_None, epochs, data_path_or_None)
    """
    meta, grid, epochs, eta_stack = load_eta_snapshots(snapshots_dir)
    if param_names is None:
        param_names = meta.get("param_names", ["theta_1", "theta_2"])
    if len(param_names) < 2:
        raise ValueError(
            f"need at least 2 param_names; got {list(param_names)!r}"
        )

    if burn_in and burn_in > 0:
        keep = np.asarray([e >= burn_in for e in epochs])
        n_dropped = int((~keep).sum())
        epochs = [e for e, k in zip(epochs, keep) if k]
        eta_stack = eta_stack[keep]
        if not epochs:
            raise RuntimeError(
                f"burn_in={burn_in} removed every snapshot; "
                f"max snapshot epoch was {int(meta.get('final_global_epoch', 0))}."
            )
        print(
            f"render-time burn_in={burn_in}: dropped {n_dropped} early "
            f"snapshot(s); {len(epochs)} frame(s) remaining "
            f"(first kept epoch: {epochs[0]})."
        )

    xs_mesh = np.asarray(grid["xs_mesh"])
    ys_mesh = np.asarray(grid["ys_mesh"])
    num_pts = xs_mesh.shape[0]
    n_p = eta_stack.shape[-1]
    # Reshape eta_stack -> (n_frames, num_pts, num_pts, n_params)
    eta_2d = eta_stack.reshape(-1, num_pts, num_pts, n_p)

    # Per-panel global colour limits via percentile clipping.
    eta_vmins, eta_vmaxs = [], []
    for k in range(min(2, n_p)):
        v = eta_2d[..., k]
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            eta_vmins.append(0.0)
            eta_vmaxs.append(1.0)
        else:
            lo, hi = np.percentile(finite, norm_percentiles)
            if lo == hi:
                hi = lo + 1.0
            eta_vmins.append(float(lo))
            eta_vmaxs.append(float(hi))

    # ----- portable npz dump (may run even without gif) -----
    resolved_data_path: Optional[str]
    if data_out_path == "auto":
        if out_path is None:
            resolved_data_path = os.path.join(
                snapshots_dir, "eta_grid_timeseries.npz"
            )
        else:
            base, _ = os.path.splitext(out_path)
            resolved_data_path = base + ".npz"
    else:
        resolved_data_path = data_out_path

    saved_data_path: Optional[str] = None
    if resolved_data_path is not None:
        saved_data_path = save_eta_timeseries_npz(
            resolved_data_path,
            epochs=epochs,
            grid=grid,
            eta_stack=eta_stack,
            param_names=param_names,
            eta_vmins=eta_vmins,
            eta_vmaxs=eta_vmaxs,
            snapshots_dir=snapshots_dir,
        )
        print(
            f"saved eta timeseries to {saved_data_path} "
            f"(epochs={len(epochs)}, grid_num_pts={num_pts})"
        )

    if not save_gif or out_path is None:
        return None, list(epochs), saved_data_path

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    def _draw(ax, k, frame_idx):
        ax.clear()
        Z = eta_2d[frame_idx, :, :, k]
        # Clip to per-panel fixed range so contour levels are stable.
        Z = np.clip(Z, eta_vmins[k], eta_vmaxs[k])
        cs = ax.contourf(
            xs_mesh, ys_mesh, Z,
            levels=np.linspace(eta_vmins[k], eta_vmaxs[k], levels + 1),
            cmap=cmap, extend="both",
        )
        ax.set_xlabel(rf"${param_names[0]}$" if "\\" not in param_names[0] else param_names[0])
        ax.set_ylabel(rf"${param_names[1]}$" if "\\" not in param_names[1] else param_names[1])
        ax.set_title(rf"$\eta_{k + 1}$")
        return cs

    cs0 = _draw(axes[0], 0, 0)
    cs1 = _draw(axes[1], 1, 0)
    cbar0 = plt.colorbar(cs0, ax=axes[0])
    cbar1 = plt.colorbar(cs1, ax=axes[1])
    suptitle = fig.suptitle(f"{title_prefix}  |  epoch {epochs[0]}")

    def update(i):
        # Contour plots can't be updated in-place; redraw and refresh colour bars.
        cs0_new = _draw(axes[0], 0, i)
        cs1_new = _draw(axes[1], 1, i)
        cbar0.update_normal(cs0_new)
        cbar1.update_normal(cs1_new)
        suptitle.set_text(f"{title_prefix}  |  epoch {epochs[i]}")
        return cs0_new, cs1_new, suptitle

    ani = animation.FuncAnimation(
        fig, update, frames=len(epochs), interval=1000.0 / max(1, fps), blit=False
    )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ani.save(out_path, writer=writer, fps=fps)
    plt.close(fig)
    print(f"saved gif to {out_path} ({len(epochs)} frames)")
    return out_path, list(epochs), saved_data_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Render a .gif of the flattener's learned eta coordinates evolving "
            "over training, as a 2-panel contour plot over (theta_1, theta_2)."
        )
    )
    p.add_argument(
        "snapshots_dir",
        help="Path to the snapshots directory (containing metadata.json + grid_axes.npz).",
    )
    p.add_argument("--out", default="eta_grid_evolution.gif", help="Output .gif path.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--levels", type=int, default=20)
    p.add_argument(
        "--norm-percentiles",
        nargs=2,
        type=float,
        default=(2.0, 98.0),
        metavar=("LO", "HI"),
        help="Percentiles of eta per panel used to fix per-panel colour scale.",
    )
    p.add_argument(
        "--param-names",
        nargs="+",
        default=None,
        help="Override axis labels (need at least two).",
    )
    p.add_argument(
        "--writer",
        default="pillow",
        choices=("pillow", "imagemagick"),
    )
    p.add_argument(
        "--data-out",
        default=None,
        help=(
            "Path for the portable .npz timeseries dump. Defaults to the gif "
            "path with .npz extension. Pass an empty string to disable."
        ),
    )
    p.add_argument(
        "--no-data",
        action="store_true",
        help="Skip writing the .npz timeseries dump.",
    )
    p.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip rendering the .gif (data dump only).",
    )
    p.add_argument(
        "--burn-in",
        type=int,
        default=0,
        help=(
            "Render-time burn-in: drop snapshots with epoch < this value "
            "before computing colour limits and animating. Stacks with "
            "the training-time burn_in. Default 0 (keep all)."
        ),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    if args.no_data:
        data_out: Optional[str] = None
    elif args.data_out is None:
        data_out = "auto"
    elif args.data_out == "":
        data_out = None
    else:
        data_out = args.data_out
    make_eta_grid_gif(
        snapshots_dir=args.snapshots_dir,
        out_path=args.out,
        fps=args.fps,
        cmap=args.cmap,
        levels=args.levels,
        norm_percentiles=tuple(args.norm_percentiles),
        param_names=args.param_names,
        writer=args.writer,
        data_out_path=data_out,
        save_gif=not args.no_gif,
        burn_in=args.burn_in,
    )


if __name__ == "__main__":
    main()
