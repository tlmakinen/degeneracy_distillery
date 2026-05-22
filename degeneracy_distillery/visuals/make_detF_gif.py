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
    Tuple[List[int], np.ndarray, np.ndarray, np.ndarray],
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
        When True, also return ``lmax_frames`` (largest eigenvalue) and
        ``vmax_frames`` (corresponding eigenvector — the "hardest
        direction") computed from the ensemble-averaged Fisher at each
        frame. Setting this to True does not affect the determinant
        arrays in the first two return values.

    Returns
    -------
    epochs : list of int
    det_frames : (n_frames, n_val) array
    lmax_frames : (n_frames, n_val) array  — only when ``return_geometry=True``
    vmax_frames : (n_frames, n_val, n_params) array — only when ``return_geometry=True``
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
    lmax_frames = np.empty((len(snapshot_epochs), n_val), dtype=float) if return_geometry else None
    vmax_frames = np.empty((len(snapshot_epochs), n_val, n_p), dtype=float) if return_geometry else None

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
            continue
        F_bar = F_acc / w_acc
        F_bar = 0.5 * (F_bar + np.swapaxes(F_bar, -1, -2))
        det_frames[fi] = np.linalg.det(F_bar)
        if return_geometry:
            evals, evecs = np.linalg.eigh(F_bar)   # ascending; last = largest
            lmax_frames[fi] = evals[:, -1]
            vmax_frames[fi] = evecs[:, :, -1]

    if return_geometry:
        return snapshot_epochs, det_frames, lmax_frames, vmax_frames
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
    point_size: float = 12.0,
    norm_percentiles: Tuple[float, float] = (2.0, 98.0),
    title_prefix: str = "ensemble-averaged",
    writer: str = "pillow",
    data_out_path: Optional[str] = "auto",
    save_gif: bool = True,
    burn_in: int = 0,
    viz_mode: Literal["det", "geometry"] = "det",
    quiver_direction: Literal["hardest", "softest"] = "hardest",
    quiver_n_sub: int = 300,
    quiver_scale: float = 25.0,
    quiver_color: str = "k",
    lmax_cmap: str = "plasma",
    quiver_scale_by_magnitude: bool = False,
) -> Tuple[Optional[str], List[int], Optional[str]]:
    """Render the evolution of the ensemble-averaged Fisher geometry as a .gif.

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
        Matplotlib figure size. Defaults to ``(7, 6)`` for ``"det"``
        mode and ``(16, 5)`` for ``"geometry"`` mode.
    cmap : str
        Colormap for the log det F panel (default ``"viridis"``).
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
        Where to save the portable ``.npz`` timeseries. ``"auto"``
        (default) saves alongside the gif with the same stem; ``None``
        disables the data dump; any other string is used verbatim.
    save_gif : bool
        If False, skip rendering the gif (useful when you only want the
        ``.npz`` for shipping locally).
    burn_in : int, default 0
        Render-time burn-in: drop frames with epoch < this value before
        computing colour limits, writing the ``.npz``, and animating.
    viz_mode : {"det", "geometry"}, default ``"det"``
        ``"det"`` — the original single-panel ``log10(det F)`` scatter
        (unchanged legacy behaviour).

        ``"geometry"`` — a 3-panel layout that captures the full
        Fisher "mountain range" as it emerges over training:

        * **Panel 1** — ``log10(det F)`` scatter (same as ``"det"``
          mode; coloured by ``cmap``).
        * **Panel 2** — ``log10(λ_max)`` scatter coloured by
          ``lmax_cmap``, showing *where* the Fisher peaks (the
          mountain tops). This is the most visceral view of the
          degeneracy structure.
        * **Panel 3** — ``log10(λ_max)`` background with a quiver
          overlay of the hardest (or softest, see
          ``quiver_direction``) eigenvector, showing the *orientation*
          of the dominant Fisher sensitivity.
    quiver_direction : {"hardest", "softest"}, default ``"hardest"``
        Which eigenvector to overlay in panel 3 of ``"geometry"`` mode.
        ``"hardest"`` = eigenvector of λ_max (where the posterior is
        most constrained); ``"softest"`` = eigenvector of λ_min (the
        degeneracy direction).
    quiver_n_sub : int, default 300
        Number of scatter points subsampled for the quiver arrows.
        A fixed random seed (0) is used so arrows don't jump between
        frames.
    quiver_scale : float, default 25.0
        Passed directly to ``matplotlib.axes.Axes.quiver`` as
        ``scale``. Larger values → shorter arrows.
    quiver_color : str, default ``"k"``
        Arrow colour for the quiver overlay.
    lmax_cmap : str, default ``"plasma"``
        Colormap for the λ_max panels in ``"geometry"`` mode.
    quiver_scale_by_magnitude : bool, default False
        If True, scale each arrow's length by the corresponding
        eigenvalue (λ_max for ``"hardest"``, λ_min for ``"softest"``),
        so strong Fisher peaks produce long arrows and flat regions
        produce short ones. The lengths are normalised to the 98th
        percentile of the eigenvalue distribution across **all frames**
        before being multiplied by the unit eigenvectors, so the
        scaling is consistent across the animation (strong peaks keep
        growing visibly as the network trains). If False (default),
        all arrows are unit length — only direction is shown.

    Returns
    -------
    (gif_path_or_None, epochs, data_path_or_None)
    """
    _geometry = viz_mode == "geometry"

    meta, theta_val, per_model = load_snapshots(snapshots_dir)
    if _geometry:
        epochs, det_frames, lmax_frames, vmax_frames_arr = build_frames(
            meta, per_model, return_geometry=True
        )
    else:
        epochs, det_frames = build_frames(meta, per_model)
        lmax_frames = vmax_frames_arr = None

    if burn_in and burn_in > 0:
        keep = np.asarray([e >= burn_in for e in epochs])
        n_dropped = int((~keep).sum())
        if not keep.any():
            raise RuntimeError(
                f"burn_in={burn_in} removed every frame; "
                f"max epoch was {epochs[-1] if epochs else 0}."
            )
        epochs = [e for e, k in zip(epochs, keep) if k]
        det_frames = det_frames[keep]
        if lmax_frames is not None:
            lmax_frames = lmax_frames[keep]
            vmax_frames_arr = vmax_frames_arr[keep]
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

    log_lmax: Optional[np.ndarray] = None
    lmax_vmin_v = lmax_vmax_v = None
    if _geometry and lmax_frames is not None:
        log_lmax = _safe_log10(lmax_frames)
        finite_lmax = log_lmax[np.isfinite(log_lmax)]
        if finite_lmax.size:
            lmax_vmin_v, lmax_vmax_v = np.percentile(finite_lmax, norm_percentiles)
        else:
            lmax_vmin_v, lmax_vmax_v = 0.0, 1.0

    # Resolve the data output path.
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
            vmin=float(vmin_det),
            vmax=float(vmax_det),
            snapshots_dir=snapshots_dir,
            lmax_frames=lmax_frames,
            log_lmax_frames=log_lmax,
            vmax_frames=vmax_frames_arr,
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
    from matplotlib.colors import Normalize

    m1 = theta_val[:, 0]
    m2 = theta_val[:, 1]

    # ------------------------------------------------------------------
    # Single-panel "det" mode (legacy, unchanged)
    # ------------------------------------------------------------------
    if not _geometry:
        _figsize = figsize if figsize is not None else (7.0, 6.0)
        norm = Normalize(vmin=vmin_det, vmax=vmax_det)
        fig, ax = plt.subplots(figsize=_figsize, constrained_layout=True)
        sc = ax.scatter(m1, m2, c=log_det[0], cmap=cmap, norm=norm, s=point_size)
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

    # ------------------------------------------------------------------
    # 3-panel "geometry" mode: log_det | log_lmax | quiver(v_max/v_min)
    # ------------------------------------------------------------------
    _figsize = figsize if figsize is not None else (16.0, 5.0)
    norm_det = Normalize(vmin=vmin_det, vmax=vmax_det)
    norm_lmax = Normalize(vmin=lmax_vmin_v, vmax=lmax_vmax_v)

    # Fixed subsample index for quiver so arrows stay put across frames.
    rng = np.random.RandomState(0)
    n_sub = min(quiver_n_sub, len(m1))
    sub_idx = rng.choice(len(m1), size=n_sub, replace=False)

    # Which eigenvector to display in panel 3.
    _use_hardest = quiver_direction == "hardest"
    if _use_hardest:
        _vec_frames = vmax_frames_arr          # hardest direction (lambda_max)
        _mag_frames = lmax_frames              # magnitudes for optional scaling
        _quiver_title = r"hardest direction $v_{\max}(\lambda_{\max})$"
    else:
        if theta_val.shape[1] == 2:
            # Rotate v_max by 90 degrees to get v_min for the 2-param case.
            _vec_frames = vmax_frames_arr[..., ::-1] * np.array([-1.0, 1.0])
        else:
            print(
                "WARNING: quiver_direction='softest' with n_params>2 requires "
                "v_min which is not stored in the snapshot. Falling back to "
                "the hardest direction. Re-run with quiver_direction='hardest' "
                "to suppress this warning."
            )
            _vec_frames = vmax_frames_arr
        # lambda_min is not stored; lambda_max is still a useful magnitude proxy.
        _mag_frames = lmax_frames
        _quiver_title = r"softest direction $v_{\min}(\lambda_{\min})$"

    # Optionally weight each arrow by its eigenvalue magnitude, normalised
    # to the 98th-percentile across all frames so the scale is stable
    # across the whole animation (peaks grow visibly as the network trains).
    if quiver_scale_by_magnitude and _mag_frames is not None:
        _mag_ref = float(np.nanpercentile(_mag_frames, 98))
        _mag_ref = _mag_ref if _mag_ref > 0 else 1.0
        _scaled_vec_frames = _vec_frames * (_mag_frames[..., None] / _mag_ref)
    else:
        _scaled_vec_frames = _vec_frames

    fig, axes = plt.subplots(1, 3, figsize=_figsize, constrained_layout=True)

    sc0 = axes[0].scatter(m1, m2, c=log_det[0], s=point_size, cmap=cmap, norm=norm_det)
    axes[0].set_xlabel(param_names[0]); axes[0].set_ylabel(param_names[1])
    axes[0].set_title(r"$\log_{10}\,\det F_\theta$")
    plt.colorbar(sc0, ax=axes[0])

    sc1 = axes[1].scatter(m1, m2, c=log_lmax[0], s=point_size, cmap=lmax_cmap, norm=norm_lmax)
    axes[1].set_xlabel(param_names[0]); axes[1].set_ylabel(param_names[1])
    axes[1].set_title(r"$\log_{10}\,\lambda_{\max}(F_\theta)$")
    plt.colorbar(sc1, ax=axes[1])

    sc2 = axes[2].scatter(m1, m2, c=log_lmax[0], s=point_size * 0.4,
                          cmap=lmax_cmap, norm=norm_lmax, alpha=0.35)
    _U0 = _scaled_vec_frames[0][sub_idx, 0]
    _V0 = _scaled_vec_frames[0][sub_idx, 1] if theta_val.shape[1] >= 2 else np.zeros(n_sub)
    Q = axes[2].quiver(
        m1[sub_idx], m2[sub_idx], _U0, _V0,
        angles="xy", pivot="middle",
        scale=quiver_scale, width=0.003,
        color=quiver_color, alpha=0.75,
    )
    axes[2].set_xlabel(param_names[0]); axes[2].set_ylabel(param_names[1])
    axes[2].set_title(_quiver_title)

    suptitle = fig.suptitle(f"{title_prefix}  |  epoch {epochs[0]}")

    def update_geometry(i):
        sc0.set_array(log_det[i])
        sc1.set_array(log_lmax[i])
        sc2.set_array(log_lmax[i])
        U = _scaled_vec_frames[i][sub_idx, 0]
        V = _scaled_vec_frames[i][sub_idx, 1] if theta_val.shape[1] >= 2 else np.zeros(n_sub)
        Q.set_UVC(U, V)
        suptitle.set_text(f"{title_prefix}  |  epoch {epochs[i]}")
        return sc0, sc1, sc2, Q, suptitle

    ani = animation.FuncAnimation(
        fig, update_geometry, frames=len(epochs),
        interval=1000.0 / max(1, fps), blit=False,
    )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ani.save(out_path, writer=writer, fps=fps)
    plt.close(fig)
    print(f"saved gif to {out_path} ({len(epochs)} frames, viz_mode='geometry')")
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
    p.add_argument(
        "--burn-in",
        type=int,
        default=0,
        help=(
            "Render-time burn-in: drop frames with epoch < this value "
            "before computing colour limits and animating. Default 0 "
            "(keep all)."
        ),
    )
    p.add_argument(
        "--viz-mode",
        default="det",
        choices=("det", "geometry"),
        help=(
            "'det' (default): single-panel log10(det F) scatter. "
            "'geometry': 3-panel layout showing log_det, log_lmax, and "
            "a quiver of the hardest (or softest) Fisher eigenvector."
        ),
    )
    p.add_argument(
        "--quiver-direction",
        default="hardest",
        choices=("hardest", "softest"),
        help=(
            "Which eigenvector to overlay in geometry mode. "
            "'hardest' (default) = v_max of lambda_max; "
            "'softest' = v_min of lambda_min."
        ),
    )
    p.add_argument(
        "--quiver-n-sub",
        type=int,
        default=300,
        help="Number of subsampled points for the quiver overlay. Default 300.",
    )
    p.add_argument(
        "--quiver-scale",
        type=float,
        default=25.0,
        help="Quiver scale (larger = shorter arrows). Default 25.",
    )
    p.add_argument(
        "--quiver-color",
        default="k",
        help="Quiver arrow colour. Default 'k' (black).",
    )
    p.add_argument(
        "--lmax-cmap",
        default="plasma",
        help="Colormap for the lambda_max panels in geometry mode. Default 'plasma'.",
    )
    p.add_argument(
        "--scale-by-magnitude",
        dest="quiver_scale_by_magnitude",
        action="store_true",
        help=(
            "Scale each quiver arrow by the corresponding eigenvalue magnitude "
            "(lambda_max for 'hardest', lambda_max proxy for 'softest'), "
            "normalised to the 98th-percentile across all frames. "
            "Makes Fisher peaks visibly grow taller as training progresses. "
            "Default: unit-length arrows (direction only)."
        ),
    )
    p.set_defaults(quiver_scale_by_magnitude=False)
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
        burn_in=args.burn_in,
        viz_mode=args.viz_mode,
        quiver_direction=args.quiver_direction,
        quiver_n_sub=args.quiver_n_sub,
        quiver_scale=args.quiver_scale,
        quiver_color=args.quiver_color,
        lmax_cmap=args.lmax_cmap,
        quiver_scale_by_magnitude=args.quiver_scale_by_magnitude,
    )


if __name__ == "__main__":
    main()
