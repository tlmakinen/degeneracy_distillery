"""
Generate a .gif of the ensemble-averaged val_detF evolving over training
epochs, as a 2D scatter colormap over two parameters.

Consumes the per-model Fisher snapshots written by
`training_loop_fishnets_snapshots.train_fishnets_with_snapshots`. At each
recorded epoch, the ensemble-averaged Fisher is reconstructed as in
`degeneracy_distillery.diagnostics._aggregate_fisher` and the determinant
is plotted in log10. Models that stopped early are held at their last
snapshot so the ensemble average is always defined.

Usage::

    python make_detF_gif.py path/to/outdir/snapshots --out detF.gif
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

def _safe_log10(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.clip(np.abs(x), floor, None))


def _model_snapshot_files(snapshots_dir: str, model_idx: int) -> List[Tuple[int, str]]:
    mdir = os.path.join(snapshots_dir, f"model_{model_idx:02d}")
    if not os.path.isdir(mdir):
        return []
    pairs: List[Tuple[int, str]] = []
    for path in glob.glob(os.path.join(mdir, "epoch_*.npz")):
        stem = os.path.basename(path).split("_")[1].split(".")[0]
        pairs.append((int(stem), path))
    pairs.sort(key=lambda t: t[0])
    return pairs


def load_snapshots(snapshots_dir: str):
    """Load metadata, theta_val, and per-model snapshot stacks.

    Returns
    -------
    meta : dict
        Contents of metadata.json.
    theta_val : (n_val, n_params) array
        Validation parameters used as the colormap support.
    per_model : list of (epochs, Fs)
        For each model, ``epochs`` is an int array of saved epoch numbers
        and ``Fs`` is a stack of Fisher matrices of shape
        ``(len(epochs), n_val, n_params, n_params)``.
    """
    with open(os.path.join(snapshots_dir, "metadata.json")) as f:
        meta = json.load(f)
    theta_val = np.load(os.path.join(snapshots_dir, "theta_val.npy"))

    per_model = []
    for i in range(meta["num_models"]):
        pairs = _model_snapshot_files(snapshots_dir, i)
        if not pairs:
            per_model.append((np.array([], dtype=int), np.zeros((0,))))
            continue
        epochs = np.array([p[0] for p in pairs], dtype=int)
        Fs = np.stack([np.load(p[1])["F_val"] for p in pairs], axis=0)
        per_model.append((epochs, Fs))
    return meta, theta_val, per_model


# ----------------------------------------------------------------------------
# Frame construction
# ----------------------------------------------------------------------------

def build_frames(
    meta: dict,
    per_model,
) -> Tuple[List[int], np.ndarray]:
    """Build the (n_frames, n_val) array of ensemble-averaged det F.

    The frame grid is the union of all per-model snapshot epochs. For each
    frame epoch ``k`` and each model, we pick the latest snapshot whose
    epoch is ``<= k``; if a model has no snapshot at or before ``k``, it
    is omitted from that frame's ensemble (and weights are renormalised).
    """
    snapshot_epochs = sorted(
        {int(e) for epochs, _ in per_model for e in epochs}
    )
    if not snapshot_epochs:
        raise RuntimeError(
            f"no snapshots found under {meta.get('num_models')} model dirs"
        )

    weights = meta.get("ensemble_weights")
    if weights is None:
        bvl = meta.get("best_val_losses")
        if bvl is None:
            weights = [1.0] * meta["num_models"]
        else:
            weights = [1.0 / float(np.exp(b)) for b in bvl]
    weights = np.asarray(weights, dtype=float)

    n_val = None
    n_p = None
    for _, Fs in per_model:
        if Fs.size:
            n_val = Fs.shape[1]
            n_p = Fs.shape[-1]
            break
    if n_val is None:
        raise RuntimeError("all per-model snapshot stacks are empty")

    det_frames = np.empty((len(snapshot_epochs), n_val), dtype=float)
    for fi, k in enumerate(snapshot_epochs):
        F_acc = np.zeros((n_val, n_p, n_p), dtype=float)
        w_acc = 0.0
        for (epochs, Fs), w in zip(per_model, weights):
            if epochs.size == 0:
                continue
            mask = epochs <= k
            if not mask.any():
                continue
            last_idx = int(np.where(mask)[0][-1])
            F_acc += w * Fs[last_idx]
            w_acc += w
        if w_acc <= 0:
            det_frames[fi] = np.nan
            continue
        F_bar = F_acc / w_acc
        F_bar = 0.5 * (F_bar + np.swapaxes(F_bar, -1, -2))
        det_frames[fi] = np.linalg.det(F_bar)

    return snapshot_epochs, det_frames


# ----------------------------------------------------------------------------
# Data export (portable npz)
# ----------------------------------------------------------------------------

def save_timeseries_npz(
    out_path: str,
    epochs,
    theta_val: np.ndarray,
    det_frames: np.ndarray,
    log_det_frames: np.ndarray,
    param_names: Sequence[str],
    ensemble_weights,
    vmin: float,
    vmax: float,
    snapshots_dir: str,
) -> str:
    """Save the ensemble-averaged val_detF timeseries to an ``.npz``.

    The resulting file is fully self-contained: it contains everything
    needed to reproduce the GIF locally with custom styling, without
    needing access to the snapshot directory.

    Arrays / scalars stored
    -----------------------
    epochs : (n_frames,) int
        Epoch number for each frame.
    theta_val : (n_val, n_params) float
        Validation parameters (the scatter support).
    det_F : (n_frames, n_val) float
        Ensemble-averaged det F at each frame.
    log_det_F : (n_frames, n_val) float
        log10 |det F| at each frame (matches what the GIF colours by).
    param_names : (n_params,) unicode
        Parameter display names.
    ensemble_weights : (num_models,) float
        Member weights used in the weighted Fisher average.
    vmin, vmax : float
        Colour-scale limits chosen by the GIF builder (percentile-based).
    snapshots_dir : unicode
        Path to the snapshots directory the data was built from.
    """
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        epochs=np.asarray(epochs, dtype=int),
        theta_val=np.asarray(theta_val),
        det_F=np.asarray(det_frames),
        log_det_F=np.asarray(log_det_frames),
        param_names=np.asarray(list(param_names)),
        ensemble_weights=np.asarray(ensemble_weights, dtype=float),
        vmin=np.float64(vmin),
        vmax=np.float64(vmax),
        snapshots_dir=np.asarray(os.path.abspath(snapshots_dir)),
    )
    return out_path


# ----------------------------------------------------------------------------
# GIF rendering
# ----------------------------------------------------------------------------

def make_gif(
    snapshots_dir: str,
    out_path: Optional[str] = "val_detF_evolution.gif",
    fps: int = 10,
    param_names: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
    cmap: str = "viridis",
    point_size: float = 12.0,
    norm_percentiles: Tuple[float, float] = (2.0, 98.0),
    title_prefix: str = "ensemble-averaged",
    writer: str = "pillow",
    data_out_path: Optional[str] = "auto",
    save_gif: bool = True,
) -> Tuple[Optional[str], List[int], Optional[str]]:
    """Render the evolution of ensemble-averaged log10(det F) as a .gif.

    Always returns the timeseries data path (when saved) alongside the
    gif path; either output can be disabled independently.

    Parameters
    ----------
    snapshots_dir : str
        Directory written by ``train_fishnets_with_snapshots``.
    out_path : str or None
        Where to save the .gif. Ignored if ``save_gif=False``.
    fps : int
        Frames per second.
    param_names : sequence of str, optional
        Override the axis labels from metadata.
    figsize : (w, h)
        Matplotlib figure size.
    cmap : str
        Colormap name (default ``"viridis"`` to match diagnostics.py).
    point_size : float
        Scatter marker size.
    norm_percentiles : (lo, hi)
        Percentiles of log10 det F (across all frames) used to fix the
        colour scale, so the colormap is consistent across frames.
    title_prefix : str
        Prepended to the per-frame title.
    writer : str
        matplotlib animation writer (``"pillow"`` or ``"imagemagick"``).
    data_out_path : str, None, or ``"auto"``
        Where to save the portable ``.npz`` timeseries. ``"auto"`` (the
        default) saves alongside the gif with the same stem; ``None``
        disables the data dump entirely; any other string is used
        verbatim as the path.
    save_gif : bool
        If False, skip rendering the gif entirely (useful when you only
        want the data dump, e.g. to ship locally for replotting).

    Returns
    -------
    (gif_path_or_None, epochs, data_path_or_None)
    """
    meta, theta_val, per_model = load_snapshots(snapshots_dir)
    epochs, det_frames = build_frames(meta, per_model)

    if param_names is None:
        param_names = meta.get("param_names", ["theta_1", "theta_2"])
    if theta_val.shape[1] < 2:
        raise ValueError(
            "Need at least 2 parameters to plot a 2D scatter; "
            f"got theta_val with shape {theta_val.shape}."
        )

    log_det = _safe_log10(det_frames)
    finite_log = log_det[np.isfinite(log_det)]
    if finite_log.size == 0:
        raise RuntimeError("no finite log10 det F values across frames")
    vmin, vmax = np.percentile(finite_log, norm_percentiles)

    # Resolve the data output path: "auto" means alongside the gif.
    resolved_data_path: Optional[str]
    if data_out_path == "auto":
        if out_path is None:
            resolved_data_path = os.path.join(
                snapshots_dir, "val_detF_timeseries.npz"
            )
        else:
            base, _ = os.path.splitext(out_path)
            resolved_data_path = base + ".npz"
    else:
        resolved_data_path = data_out_path

    ensemble_weights = meta.get(
        "ensemble_weights",
        [1.0 / float(np.exp(b)) for b in meta.get("best_val_losses", [])]
        or [1.0] * meta["num_models"],
    )

    saved_data_path: Optional[str] = None
    if resolved_data_path is not None:
        saved_data_path = save_timeseries_npz(
            out_path=resolved_data_path,
            epochs=epochs,
            theta_val=theta_val,
            det_frames=det_frames,
            log_det_frames=log_det,
            param_names=param_names,
            ensemble_weights=ensemble_weights,
            vmin=float(vmin),
            vmax=float(vmax),
            snapshots_dir=snapshots_dir,
        )
        print(
            f"saved timeseries to {saved_data_path} "
            f"(epochs={len(epochs)}, n_val={theta_val.shape[0]})"
        )

    if not save_gif or out_path is None:
        return None, list(epochs), saved_data_path

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sc = ax.scatter(
        theta_val[:, 0],
        theta_val[:, 1],
        c=log_det[0],
        cmap=cmap,
        norm=norm,
        s=point_size,
    )
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    title = ax.set_title(
        f"{title_prefix} $\\log_{{10}}\\,\\det F$  |  epoch {epochs[0]}"
    )
    plt.colorbar(sc, ax=ax, label=r"$\log_{10}\,\det F_\theta$")

    def update(i):
        sc.set_array(log_det[i])
        title.set_text(
            f"{title_prefix} $\\log_{{10}}\\,\\det F$  |  epoch {epochs[i]}"
        )
        return sc, title

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
            "Render a .gif of ensemble-averaged val_detF over training "
            "from per-model Fisher snapshots."
        )
    )
    p.add_argument(
        "snapshots_dir",
        help="Path to the snapshots directory (the one containing metadata.json).",
    )
    p.add_argument("--out", default="val_detF_evolution.gif", help="Output .gif path.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--point-size", type=float, default=12.0)
    p.add_argument(
        "--norm-percentiles",
        nargs=2,
        type=float,
        default=(2.0, 98.0),
        metavar=("LO", "HI"),
        help="Percentiles of log10 det F used to fix colour scale (default 2 98).",
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
            "Path for the portable .npz timeseries dump. Defaults to the "
            "gif path with .npz extension. Pass an empty string to disable."
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
        help=(
            "Skip rendering the .gif. Useful when you only want the .npz "
            "to ship locally for replotting."
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
    make_gif(
        snapshots_dir=args.snapshots_dir,
        out_path=args.out,
        fps=args.fps,
        cmap=args.cmap,
        point_size=args.point_size,
        norm_percentiles=tuple(args.norm_percentiles),
        param_names=args.param_names,
        writer=args.writer,
        data_out_path=data_out,
        save_gif=not args.no_gif,
    )


if __name__ == "__main__":
    main()
