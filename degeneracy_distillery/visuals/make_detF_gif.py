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
from typing import List, Literal, Optional, Sequence, Tuple, Union

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
    return_geometry: bool = False,
) -> Union[
    Tuple[List[int], np.ndarray],
    Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Build per-frame ensemble-averaged Fisher statistics.

    The frame grid is the union of all per-model snapshot epochs. For each
    frame epoch ``k`` and each model, we pick the latest snapshot whose
    epoch is ``<= k``; if a model has no snapshot at or before ``k``, it
    is omitted from that frame's ensemble (and weights are renormalised).

    Parameters
    ----------
    meta : dict
        Contents of ``metadata.json`` (from :func:`load_snapshots`).
    per_model : list of (epochs, Fs)
        Per-model snapshot stacks (from :func:`load_snapshots`).
    return_geometry : bool, default False
        When True, also return per-frame Fisher geometry arrays.
        Setting this to True does not affect the determinant arrays in
        the first two return values.

    Returns
    -------
    epochs : list of int
    det_frames : (n_frames, n_val) array
    lmax_frames : (n_frames, n_val) — only when ``return_geometry=True``
        Largest eigenvalue of the full ``n_params × n_params`` Fisher.
    vmax_frames : (n_frames, n_val, n_params) — only when ``return_geometry=True``
        Corresponding hardest-direction eigenvector.
    evals_2d_frames : (n_frames, n_val, 2) — only when ``return_geometry=True``
        Both eigenvalues (ascending) of the ``2 × 2`` displayed subblock
        ``F_bar[:, :2, :2]``.  Used to draw the 1-σ Fisher ellipses.
    vmin_2d_frames : (n_frames, n_val, 2) — only when ``return_geometry=True``
        Eigenvector of ``λ_min`` of the 2 × 2 subblock (semi-major axis
        direction of the 1-σ ellipse, i.e. the degeneracy direction in the
        displayed parameter plane).
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
    lmax_frames  = np.empty((len(snapshot_epochs), n_val), dtype=float)       if return_geometry else None
    vmax_frames  = np.empty((len(snapshot_epochs), n_val, n_p), dtype=float)   if return_geometry else None
    evals_2d_frames = np.empty((len(snapshot_epochs), n_val, 2), dtype=float)  if return_geometry else None
    vmin_2d_frames  = np.empty((len(snapshot_epochs), n_val, 2), dtype=float)  if return_geometry else None

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
            if return_geometry:
                lmax_frames[fi] = np.nan
                vmax_frames[fi] = np.nan
                evals_2d_frames[fi] = np.nan
                vmin_2d_frames[fi] = np.nan
            continue
        F_bar = F_acc / w_acc
        F_bar = 0.5 * (F_bar + np.swapaxes(F_bar, -1, -2))
        det_frames[fi] = np.linalg.det(F_bar)
        if return_geometry:
            # Full-matrix: largest eigenvalue + eigenvector (for quiver if needed).
            evals, evecs = np.linalg.eigh(F_bar)   # ascending; last = largest
            lmax_frames[fi] = evals[:, -1]
            vmax_frames[fi] = evecs[:, :, -1]
            # 2×2 displayed subblock: both eigenvalues + v_min for ellipse drawing.
            F_2d = F_bar[:, :2, :2]
            evals_2d, evecs_2d = np.linalg.eigh(F_2d)   # ascending: 0=min, 1=max
            evals_2d_frames[fi] = evals_2d
            vmin_2d_frames[fi]  = evecs_2d[:, :, 0]     # semi-major axis direction

    if return_geometry:
        return snapshot_epochs, det_frames, lmax_frames, vmax_frames, evals_2d_frames, vmin_2d_frames
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
    lmax_frames: Optional[np.ndarray] = None,
    vmax_frames: Optional[np.ndarray] = None,
    log_lmax_frames: Optional[np.ndarray] = None,
    lmax_vmin: Optional[float] = None,
    lmax_vmax: Optional[float] = None,
) -> str:
    """Save the ensemble-averaged Fisher timeseries to an ``.npz``.

    The resulting file is fully self-contained: it contains everything
    needed to reproduce the GIF locally with custom styling, without
    needing access to the snapshot directory.

    Arrays / scalars stored
    -----------------------
    epochs : (n_frames,) int
    theta_val : (n_val, n_params) float
    det_F : (n_frames, n_val) float
    log_det_F : (n_frames, n_val) float
    param_names : (n_params,) unicode
    ensemble_weights : (num_models,) float
    vmin, vmax : float  — colour-scale limits for log det F
    snapshots_dir : unicode

    Additional (geometry mode only)
    --------------------------------
    lmax_F : (n_frames, n_val) float        — largest eigenvalue per sample
    log_lmax_F : (n_frames, n_val) float    — log10(lambda_max)
    vmax_F : (n_frames, n_val, n_params)    — hardest-direction eigenvector
    lmax_vmin, lmax_vmax : float            — colour limits for log lmax
    """
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = dict(
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
    if lmax_frames is not None:
        payload["lmax_F"] = np.asarray(lmax_frames)
    if log_lmax_frames is not None:
        payload["log_lmax_F"] = np.asarray(log_lmax_frames)
    if vmax_frames is not None:
        payload["vmax_F"] = np.asarray(vmax_frames)
    if lmax_vmin is not None:
        payload["lmax_vmin"] = np.float64(lmax_vmin)
    if lmax_vmax is not None:
        payload["lmax_vmax"] = np.float64(lmax_vmax)
    np.savez_compressed(out_path, **payload)
    return out_path


# ----------------------------------------------------------------------------
# GIF rendering
# ----------------------------------------------------------------------------

def make_gif(
    snapshots_dir: str,
    out_path: Optional[str] = "val_detF_evolution.gif",
    fps: int = 10,
    param_names: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
    norm_percentiles: Tuple[float, float] = (2.0, 98.0),
    title_prefix: str = "ensemble-averaged",
    title: Optional[str] = None,
    show_epoch: bool = True,
    writer: str = "pillow",
    data_out_path: Optional[str] = "auto",
    save_gif: bool = True,
    burn_in: int = 0,
    viz_mode: Literal["det", "geometry"] = "det",
    ellipse_scale: float = 0.05,
    ellipse_n_sub: Optional[int] = None,
    ellipse_alpha: float = 0.75,
    ellipse_lw: float = 0.4,
    # Kept for backwards compatibility; no longer used for rendering.
    point_size: float = 12.0,
    quiver_direction: Literal["hardest", "softest"] = "hardest",
    quiver_n_sub: int = 300,
    quiver_scale: float = 25.0,
    quiver_color: str = "k",
    lmax_cmap: str = "plasma",
    quiver_scale_by_magnitude: bool = False,
) -> Tuple[Optional[str], List[int], Optional[str]]:
    """Render the evolution of the ensemble-averaged Fisher geometry as a .gif.

    Both ``viz_mode="det"`` and ``viz_mode="geometry"`` now show a single
    panel of **1-σ Fisher ellipses** drawn at each validation parameter
    location, coloured by ``log10(det F)``. The ellipses encode the full
    local Fisher geometry: their orientation shows the degeneracy direction
    and their aspect ratio shows how anisotropic the constraint is. As
    training progresses the ellipses shrink, rotate, and sharpen, making
    the emerging "mountain range" of Fisher information viscerally visible.

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
    figsize : (w, h), optional
        Matplotlib figure size. Default ``(7, 6)``.
    cmap : str
        Colormap for ``log10(det F)`` (default ``"viridis"``).
    norm_percentiles : (lo, hi)
        Percentiles of ``log10 det F`` across all frames used to fix the
        colour scale so it is consistent across frames.
    title_prefix : str
        Prepended to the per-frame super-title.
    title : str or None, default None
        If set, replaces ``title_prefix`` as the label part of the
        per-frame title. The epoch counter is still appended (see
        ``show_epoch``).  Pass an empty string (``""``) for a blank
        label with only the epoch number.
    show_epoch : bool, default True
        If False, suppress the epoch counter from the title entirely —
        useful for blog-post gifs where technical details should stay
        off the frame.
    writer : str
        Matplotlib animation writer (``"pillow"`` or ``"imagemagick"``).
    data_out_path : str, None, or ``"auto"``
        Where to save the portable ``.npz`` timeseries. ``"auto"``
        (default) saves alongside the gif with the same stem; ``None``
        disables the dump; any other string is used verbatim.
    save_gif : bool
        If False, skip rendering the gif (data dump only).
    burn_in : int, default 0
        Render-time burn-in: drop frames with epoch < this value before
        computing colour limits, writing the ``.npz``, and animating.
    viz_mode : {"det", "geometry"}, default ``"det"``
        Kept for API compatibility. Both modes now render the same
        single-panel 1-σ Fisher ellipse visualisation.
    ellipse_scale : float, default 0.05
        Controls the overall size of the ellipses. The semi-major axis
        of the *median* ellipse at the **last** (most-trained) frame is
        set to ``ellipse_scale × data_range`` (where ``data_range`` is
        the larger of the ``theta_1`` / ``theta_2`` ranges). All
        earlier frames use the same absolute normalisation, so you can
        watch ellipses grow/shrink/sharpen as training progresses.
        Increase this value if ellipses are too small to see; decrease
        if they overlap too much.
    ellipse_n_sub : int or None, default None
        Subsample the validation set to this many ellipses for
        rendering performance. ``None`` draws all validation points.
        Recommended: 300–800 for a clean visual.
    ellipse_alpha : float, default 0.75
        Opacity of the ellipse faces.
    ellipse_lw : float, default 0.4
        Ellipse edge linewidth (``0`` for no edge).

    Returns
    -------
    (gif_path_or_None, epochs, data_path_or_None)
    """
    meta, theta_val, per_model = load_snapshots(snapshots_dir)
    epochs, det_frames, lmax_frames, vmax_frames, evals_2d_frames, vmin_2d_frames = (
        build_frames(meta, per_model, return_geometry=True)
    )

    if burn_in and burn_in > 0:
        keep = np.asarray([e >= burn_in for e in epochs])
        n_dropped = int((~keep).sum())
        if not keep.any():
            raise RuntimeError(
                f"burn_in={burn_in} removed every frame; "
                f"max epoch was {epochs[-1] if epochs else 0}."
            )
        epochs        = [e for e, k in zip(epochs, keep) if k]
        det_frames    = det_frames[keep]
        lmax_frames   = lmax_frames[keep]
        vmax_frames   = vmax_frames[keep]
        evals_2d_frames = evals_2d_frames[keep]
        vmin_2d_frames  = vmin_2d_frames[keep]
        print(
            f"render-time burn_in={burn_in}: dropped {n_dropped} early "
            f"frame(s); {len(epochs)} frame(s) remaining "
            f"(first kept epoch: {epochs[0]})."
        )

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
    vmin_det, vmax_det = np.percentile(finite_log, norm_percentiles)

    log_lmax = _safe_log10(lmax_frames)
    _lmax_finite = log_lmax[np.isfinite(log_lmax)]
    lmax_vmin_v = float(np.percentile(_lmax_finite, norm_percentiles[0])) if _lmax_finite.size else 0.0
    lmax_vmax_v = float(np.percentile(_lmax_finite, norm_percentiles[1])) if _lmax_finite.size else 1.0

    # Resolve the data output path.
    resolved_data_path: Optional[str]
    if data_out_path == "auto":
        if out_path is None:
            resolved_data_path = os.path.join(snapshots_dir, "val_detF_timeseries.npz")
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
            vmin=float(vmin_det),
            vmax=float(vmax_det),
            snapshots_dir=snapshots_dir,
            lmax_frames=lmax_frames,
            log_lmax_frames=log_lmax,
            vmax_frames=vmax_frames,
            lmax_vmin=lmax_vmin_v,
            lmax_vmax=lmax_vmax_v,
        )
        print(
            f"saved timeseries to {saved_data_path} "
            f"(epochs={len(epochs)}, n_val={theta_val.shape[0]})"
        )

    if not save_gif or out_path is None:
        return None, list(epochs), saved_data_path

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.collections import EllipseCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    m1 = theta_val[:, 0]
    m2 = theta_val[:, 1]

    # ------------------------------------------------------------------
    # Ellipse normalisation: scale so the median semi-major axis at the
    # last (most-trained) frame equals ellipse_scale × data_range.
    # This normalisation is frozen for all frames, so early frames show
    # large uncertain ellipses that shrink and sharpen as training proceeds.
    # ------------------------------------------------------------------
    data_range = max(float(m1.max() - m1.min()), float(m2.max() - m2.min()), 1e-8)
    _lmin_last = np.clip(evals_2d_frames[-1, :, 0], 1e-30, None)
    med_sigma_last = float(np.nanmedian(1.0 / np.sqrt(_lmin_last)))
    _norm_factor = (ellipse_scale * data_range / med_sigma_last) if med_sigma_last > 0 else (ellipse_scale * data_range)

    # Subsample (fixed seed → arrows/ellipses stay at same locations each frame).
    rng = np.random.RandomState(0)
    n_sub = min(ellipse_n_sub or len(m1), len(m1))
    sub_idx = rng.choice(len(m1), size=n_sub, replace=False)
    m1_sub, m2_sub = m1[sub_idx], m2[sub_idx]

    def _ellipse_arrays(fi):
        """Return (widths, heights, angles_deg, colors) for frame fi."""
        lmin_i = np.clip(evals_2d_frames[fi, sub_idx, 0], 1e-30, None)
        lmax_i = np.clip(evals_2d_frames[fi, sub_idx, 1], 1e-30, None)
        semi_major = _norm_factor / np.sqrt(lmin_i)   # 1/√λ_min — degeneracy direction
        semi_minor = _norm_factor / np.sqrt(lmax_i)   # 1/√λ_max — constrained direction
        angle_deg = np.degrees(
            np.arctan2(vmin_2d_frames[fi, sub_idx, 1],
                       vmin_2d_frames[fi, sub_idx, 0])
        )
        colors = log_det[fi, sub_idx]
        return 2 * semi_major, 2 * semi_minor, angle_deg, colors  # full diameters

    _figsize = figsize if figsize is not None else (7.0, 6.0)
    norm = Normalize(vmin=vmin_det, vmax=vmax_det)
    pad = 0.03 * data_range

    fig, ax = plt.subplots(figsize=_figsize, constrained_layout=True)

    # Colorbar via an independent ScalarMappable so it survives ax.cla().
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=r"$\log_{10}\,\det F_\theta$")

    def _draw_frame(fi):
        ax.cla()
        w, h, ang, col = _ellipse_arrays(fi)
        ec = EllipseCollection(
            widths=w, heights=h, angles=ang,
            units="x",
            array=col, cmap=cmap, norm=norm,
            offsets=np.column_stack([m1_sub, m2_sub]),
            transOffset=ax.transData,
            alpha=ellipse_alpha, linewidths=ellipse_lw,
            edgecolors="face",
        )
        ax.add_collection(ec)
        ax.set_xlim(m1.min() - pad, m1.max() + pad)
        ax.set_ylim(m2.min() - pad, m2.max() + pad)
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        _label = title if title is not None else f"{title_prefix} $\\log_{{10}}\\,\\det F$"
        ax.set_title(
            f"{_label}  |  epoch {epochs[fi]}" if show_epoch else _label
        )
        return (ec,)

    _draw_frame(0)
    ani = animation.FuncAnimation(
        fig, _draw_frame, frames=len(epochs),
        interval=1000.0 / max(1, fps), blit=False,
    )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ani.save(out_path, writer=writer, fps=fps)
    plt.close(fig)
    print(f"saved gif to {out_path} ({len(epochs)} frames, 1-σ Fisher ellipses)")
    return out_path, list(epochs), saved_data_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Render a .gif of the ensemble-averaged Fisher geometry evolving "
            "over training, as 1-σ ellipses at each validation parameter "
            "location coloured by log10(det F)."
        )
    )
    p.add_argument(
        "snapshots_dir",
        help="Path to the snapshots directory (the one containing metadata.json).",
    )
    p.add_argument("--out", default="val_detF_evolution.gif", help="Output .gif path.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--cmap", default="viridis")
    p.add_argument(
        "--norm-percentiles",
        nargs=2, type=float, default=(2.0, 98.0), metavar=("LO", "HI"),
        help="Percentiles of log10 det F used to fix colour scale (default 2 98).",
    )
    p.add_argument(
        "--param-names", nargs="+", default=None,
        help="Override axis labels (need at least two).",
    )
    p.add_argument(
        "--writer", default="pillow", choices=("pillow", "imagemagick"),
    )
    p.add_argument(
        "--data-out", default=None,
        help=(
            "Path for the portable .npz timeseries dump. Defaults to the "
            "gif path with .npz extension. Pass an empty string to disable."
        ),
    )
    p.add_argument("--no-data", action="store_true",
                   help="Skip writing the .npz timeseries dump.")
    p.add_argument("--no-gif", action="store_true",
                   help="Skip rendering the .gif (data dump only).")
    p.add_argument(
        "--burn-in", type=int, default=0,
        help="Drop frames with epoch < this value. Default 0 (keep all).",
    )
    p.add_argument(
        "--viz-mode", default="det", choices=("det", "geometry"),
        help="Kept for compatibility; both modes render 1-σ ellipses.",
    )
    p.add_argument(
        "--ellipse-scale", type=float, default=0.05,
        help=(
            "Scales ellipse sizes: the median semi-major axis at the last "
            "frame = ellipse_scale × data_range. Default 0.05."
        ),
    )
    p.add_argument(
        "--ellipse-n-sub", type=int, default=None,
        help="Subsample validation set to this many ellipses. Default: all.",
    )
    p.add_argument(
        "--ellipse-alpha", type=float, default=0.75,
        help="Ellipse face opacity. Default 0.75.",
    )
    p.add_argument(
        "--ellipse-lw", type=float, default=0.4,
        help="Ellipse edge linewidth (0 = no edge). Default 0.4.",
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
        norm_percentiles=tuple(args.norm_percentiles),
        param_names=args.param_names,
        writer=args.writer,
        data_out_path=data_out,
        save_gif=not args.no_gif,
        burn_in=args.burn_in,
        viz_mode=args.viz_mode,
        ellipse_scale=args.ellipse_scale,
        ellipse_n_sub=args.ellipse_n_sub,
        ellipse_alpha=args.ellipse_alpha,
        ellipse_lw=args.ellipse_lw,
    )


if __name__ == "__main__":
    main()
