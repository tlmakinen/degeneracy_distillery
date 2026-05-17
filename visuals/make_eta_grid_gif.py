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
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Alignment helpers
# ----------------------------------------------------------------------------

AlignMode = Literal["none", "linear_residual", "nonlinearity_rotation", "both"]


def _fit_affine(
    eta: np.ndarray,
    theta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares affine fit ``eta ≈ theta @ A.T + b``.

    Parameters
    ----------
    eta : (N, n_eta) array
    theta : (N, n_in) array

    Returns
    -------
    A : (n_eta, n_in)
    b : (n_eta,)
    """
    eta = np.asarray(eta, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    X_aug = np.column_stack([theta, np.ones(theta.shape[0])])
    sol, *_ = np.linalg.lstsq(X_aug, eta, rcond=None)
    A = sol[:-1, :].T
    b = sol[-1, :]
    return A, b


def _nonlinearity_rotation(
    dy: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    prior_scales: Optional[np.ndarray] = None,
    regularization: float = 1e-12,
    enforce_proper: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """JAX-free vendored copy of
    :func:`degeneracy_distillery.align_coords.nonlinearity_rotation`.

    Returns an orthogonal rotation ``R`` (and a descending vector of
    nonlinearity "energies" ``sigma``) so that ``R @ J[b]`` has its
    rows ordered by descending sample-variance of the Jacobian. A row
    with ``sigma_i ≈ 0`` corresponds to an η component that is linear
    in θ.

    See ``align_coords.nonlinearity_rotation`` for the full docstring;
    this duplicate exists so the gif builder stays free of the JAX
    import chain (needed because the project's top-level
    ``align_coords`` does ``import jax``).
    """
    dy = np.asarray(dy, dtype=np.float64)
    if dy.ndim != 3:
        raise ValueError(f"dy must be 3-D (B, D_out, D_in); got {dy.shape}")
    B, D_out, D_in = dy.shape

    if sample_weights is None:
        w = np.full(B, 1.0 / B)
    else:
        w = np.asarray(sample_weights, dtype=np.float64)
        w = w / np.maximum(w.sum(), 1e-30)

    J_mean = np.einsum("b,bij->ij", w, dy)
    dJ = dy - J_mean[None]
    if prior_scales is not None:
        dJ = dJ * np.asarray(prior_scales, dtype=np.float64).reshape(1, 1, D_in)

    sqrt_w = np.sqrt(w).reshape(B, 1, 1)
    Omega = (sqrt_w * dJ).transpose(1, 0, 2).reshape(D_out, B * D_in)

    U, sigma, _ = np.linalg.svd(Omega, full_matrices=False)

    # Deterministic sign per column.
    dom_row = np.argmax(np.abs(U), axis=0)
    sign = np.sign(U[dom_row, np.arange(U.shape[1])])
    sign = np.where(sign == 0, 1.0, sign)
    U = U * sign[None, :]

    R = U.T
    if enforce_proper and np.linalg.det(R) < 0:
        R = R.copy()
        R[-1] *= -1.0
    return R, sigma + regularization


def _finite_diff_jacobian(
    eta_grid: np.ndarray,
    xs_axis: np.ndarray,
    ys_axis: np.ndarray,
    grid_num_pts: int,
) -> np.ndarray:
    """Finite-difference Jacobian on a regular 2D mesh in (theta_1, theta_2).

    Assumes ``eta_grid`` is the flattened output of the standard
    ``meshgrid('xy')`` build used by
    :func:`visuals.training_loop_flatten_snapshots._build_eval_grid`,
    i.e. ``eta_2d[i, j, k] = eta_k at (theta_1=xs[j], theta_2=ys[i])``.

    Returns
    -------
    J : (grid_num_pts**2, n_eta, 2) array
        ``J[:, :, 0] = ∂eta/∂theta_1``, ``J[:, :, 1] = ∂eta/∂theta_2``.
    """
    n_eta = eta_grid.shape[-1]
    eta_2d = eta_grid.reshape(grid_num_pts, grid_num_pts, n_eta)
    d_dtheta1 = np.gradient(eta_2d, xs_axis, axis=1)
    d_dtheta2 = np.gradient(eta_2d, ys_axis, axis=0)
    J = np.stack(
        [
            d_dtheta1.reshape(grid_num_pts * grid_num_pts, n_eta),
            d_dtheta2.reshape(grid_num_pts * grid_num_pts, n_eta),
        ],
        axis=-1,
    )
    return J


def align_eta_stack(
    eta_stack: np.ndarray,
    X_grid: np.ndarray,
    *,
    mode: AlignMode = "none",
    grid_num_pts: Optional[int] = None,
    xs_axis: Optional[np.ndarray] = None,
    ys_axis: Optional[np.ndarray] = None,
    reference_frame: int = -1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply an alignment transform to every frame so the nonlinear part
    of eta visually "pops out".

    The transform is computed **once** on the reference frame (default
    last, i.e. most-trained) and applied identically to every frame,
    so the gif keeps a consistent coordinate system as the network
    learns.

    Modes
    -----
    ``"none"`` :
        Return the stack unchanged.
    ``"linear_residual"`` :
        Subtract the best affine fit ``eta ≈ A @ theta + b`` (computed
        on the reference frame, over the ``(theta_1, theta_2)`` plane
        of the grid). The dominant linear gradient is removed; what
        remains is the pure nonlinear residual. **Most directly
        visually compelling.**
    ``"nonlinearity_rotation"`` :
        Apply the orthogonal rotation from
        :func:`degeneracy_distillery.align_coords.nonlinearity_rotation`,
        computed on a finite-difference Jacobian of the reference
        frame. After rotation, ``eta_1`` is the eta combination with
        the most sample-dependent Jacobian (= most nonlinear); the
        last component is closest to linear in theta.
    ``"both"`` :
        Subtract the affine fit, then rotate by ``nonlinearity_rotation``.
        Cleanest separation of nonlinear content.

    Parameters
    ----------
    eta_stack : (n_frames, G*G, n_eta) array
        Per-frame eta evaluated on a 2D grid.
    X_grid : (G*G, n_in) array
        Grid in theta space. Only the first two columns
        (``theta_1, theta_2``) are used for the affine fit / Jacobian.
    grid_num_pts, xs_axis, ys_axis :
        Mesh-axis information needed for finite-difference Jacobian.
        Required for ``"nonlinearity_rotation"`` / ``"both"`` only.
    reference_frame : int, default -1
        Frame index used to compute the alignment (negative indices
        allowed). Default is the last (most-trained) frame.

    Returns
    -------
    aligned : (n_frames, G*G, n_eta)
        Transformed eta stack.
    info : dict
        Recorded alignment metadata (``mode``, ``reference_frame_index``,
        ``affine_A``, ``affine_b``, ``nonlin_R``, ``nonlin_sigma``)
        depending on ``mode``. Always JSON-serialisable-ish (arrays
        are kept as ``np.ndarray``).
    """
    eta_stack = np.asarray(eta_stack, dtype=np.float64)
    X_grid = np.asarray(X_grid, dtype=np.float64)
    if mode == "none":
        return eta_stack, {"mode": "none"}
    if X_grid.shape[1] < 2:
        raise ValueError(
            "align_eta_stack needs at least 2 columns of X_grid; "
            f"got X_grid with shape {X_grid.shape}."
        )

    ref_idx = int(reference_frame % eta_stack.shape[0])
    eta_ref = eta_stack[ref_idx]
    theta_used = X_grid[:, :2]
    aligned = eta_stack.copy()
    info: Dict[str, Any] = {"mode": mode, "reference_frame_index": ref_idx}

    if mode in ("linear_residual", "both"):
        A, b = _fit_affine(eta_ref, theta_used)
        eta_linear = theta_used @ A.T + b[None, :]
        aligned = aligned - eta_linear[None, ...]
        info["affine_A"] = A
        info["affine_b"] = b

    if mode in ("nonlinearity_rotation", "both"):
        if grid_num_pts is None or xs_axis is None or ys_axis is None:
            raise ValueError(
                "mode requires `grid_num_pts`, `xs_axis`, `ys_axis`."
            )
        # Use the (possibly residual-subtracted) reference frame for J.
        # The rotation is invariant to affine offsets, but referring to
        # `aligned[ref_idx]` keeps the output deterministic.
        eta_for_J = aligned[ref_idx]
        J_ref = _finite_diff_jacobian(
            eta_for_J,
            xs_axis=np.asarray(xs_axis),
            ys_axis=np.asarray(ys_axis),
            grid_num_pts=int(grid_num_pts),
        )
        R, sigma = _nonlinearity_rotation(J_ref)
        aligned = np.einsum("ij,fpj->fpi", R, aligned)
        info["nonlin_R"] = R
        info["nonlin_sigma"] = sigma

    return aligned, info


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
    frob_scores : (n_frames,) np.ndarray of float
        Per-epoch validation flatness score
        ``mean_b ||Q_b - I||_F``. ``np.nan`` for frames that did not
        save a score (older snapshots written before the
        ``frob_score`` field existed).
    """
    with open(os.path.join(snapshots_dir, "metadata.json")) as f:
        meta = json.load(f)
    grid_npz = np.load(os.path.join(snapshots_dir, "grid_axes.npz"))
    grid = {k: grid_npz[k] for k in grid_npz.files}

    pairs = _snapshot_files(snapshots_dir)
    if not pairs:
        raise RuntimeError(f"no epoch_*.npz snapshots found under {snapshots_dir}")
    epochs = [p[0] for p in pairs]
    eta_stack_list: List[np.ndarray] = []
    frob_scores_list: List[float] = []
    for _, p in pairs:
        d = np.load(p)
        eta_stack_list.append(d["eta_grid"])
        if "frob_score" in d.files:
            frob_scores_list.append(float(d["frob_score"]))
        else:
            frob_scores_list.append(float("nan"))
    eta_stack = np.stack(eta_stack_list, axis=0)
    frob_scores = np.asarray(frob_scores_list, dtype=np.float64)
    return meta, grid, epochs, eta_stack, frob_scores


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
    align_info: Optional[Dict[str, Any]] = None,
    eta_stack_raw: Optional[np.ndarray] = None,
    frob_scores: Optional[np.ndarray] = None,
) -> str:
    """Save a self-contained ``.npz`` for local replotting.

    Required keys: ``epochs``, ``xs_mesh``, ``ys_mesh``, ``X_grid``,
    ``eta_stack`` (n_frames, num_pts**2, n_params), ``eta_grid_2d``
    (n_frames, num_pts, num_pts, n_params), ``param_names``,
    ``eta_vmins``, ``eta_vmaxs``, ``min_x``, ``max_x``,
    ``snapshots_dir``.

    Alignment metadata (only present when ``align_info`` is provided):
    ``align_mode``, ``align_reference_frame_index``, plus
    ``affine_A``, ``affine_b``, ``nonlin_R``, ``nonlin_sigma`` as
    applicable. The pre-alignment stack is stored under
    ``eta_stack_raw`` when ``eta_stack_raw`` is supplied, so a local
    replotter can switch alignment modes without re-running training.
    """
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    num_pts = int(grid["xs_mesh"].shape[0])
    n_p = int(eta_stack.shape[-1])
    eta_grid_2d = eta_stack.reshape(-1, num_pts, num_pts, n_p)
    payload: Dict[str, Any] = dict(
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
    if align_info is not None:
        payload["align_mode"] = np.asarray(str(align_info.get("mode", "none")))
        if "reference_frame_index" in align_info:
            payload["align_reference_frame_index"] = np.int64(
                align_info["reference_frame_index"]
            )
        for key in ("affine_A", "affine_b", "nonlin_R", "nonlin_sigma"):
            if key in align_info:
                payload[key] = np.asarray(align_info[key])
    if eta_stack_raw is not None:
        payload["eta_stack_raw"] = np.asarray(eta_stack_raw)
        payload["eta_grid_2d_raw"] = np.asarray(eta_stack_raw).reshape(
            -1, num_pts, num_pts, n_p
        )
    if frob_scores is not None:
        payload["frob_scores"] = np.asarray(frob_scores, dtype=np.float64)
    np.savez_compressed(out_path, **payload)
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
    align_mode: AlignMode = "none",
    align_reference_frame: int = -1,
    keep_raw_in_npz: bool = True,
    show_loss: bool = False,
    loss_label: str = r"\|Q-I\|_F",
    loss_fmt: str = ".2f",
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
    align_mode : {"none", "linear_residual", "nonlinearity_rotation", "both"}, default ``"none"``
        If not ``"none"``, transform ``eta`` so the **nonlinear**
        component pops out in the contour plots. The transform is
        computed once on ``align_reference_frame`` (default: last,
        i.e. most-trained frame) and applied identically to every
        frame so the gif stays consistent across training. See
        :func:`align_eta_stack` for the algorithms.
    align_reference_frame : int, default -1
        Frame index used to derive the alignment transform (negative
        indices supported, e.g. ``-1`` = last frame).
    keep_raw_in_npz : bool, default True
        When alignment is on, also store the un-aligned stack as
        ``eta_stack_raw`` / ``eta_grid_2d_raw`` in the portable
        ``.npz`` so a local replotter can switch alignment modes
        without re-running anything.
    show_loss : bool, default False
        If True, append the validation flatness Frobenius score
        (``mean_b ||Q_b - I||_F``) to the per-frame super-title,
        e.g. ``"epoch 1500  |  ||Q-I||_F = 0.43"``. Scores are
        read from ``frob_score`` in each ``epoch_*.npz`` (and
        ``saved_frob_scores`` in ``metadata.json``); they are written
        by the updated
        :func:`visuals.training_loop_flatten_snapshots.fit_flattening_with_snapshots`.
        Snapshots produced before that change silently fall back to
        showing only the epoch number.
    loss_label : str, default :code:`r"\\|Q-I\\|_F"`
        Math-text label used in front of the numeric score. Override
        to use a different name (e.g. :code:`r"\\mathcal{L}_\\mathrm{val}"`).
    loss_fmt : str, default ``".2f"``
        Python format spec for the numeric part of the loss readout
        (e.g. ``".1f"`` for one decimal, ``".2e"`` for scientific).
        For very large / very small values the function falls back to
        scientific notation automatically; this argument controls the
        format in the normal-magnitude regime.

    Returns
    -------
    (gif_path_or_None, epochs, data_path_or_None)
    """
    meta, grid, epochs, eta_stack, frob_scores = load_eta_snapshots(snapshots_dir)
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
        frob_scores = frob_scores[keep]
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

    # ----- alignment: pop the nonlinear component out of eta -----
    eta_stack_raw = eta_stack
    align_info: Optional[Dict[str, Any]] = None
    if align_mode != "none":
        # Mesh axes (1D linspaces) reconstructed from the 2D mesh.
        xs_axis = xs_mesh[0, :]
        ys_axis = ys_mesh[:, 0]
        eta_stack, align_info = align_eta_stack(
            eta_stack,
            X_grid=np.asarray(grid["X_grid"]),
            mode=align_mode,
            grid_num_pts=num_pts,
            xs_axis=xs_axis,
            ys_axis=ys_axis,
            reference_frame=align_reference_frame,
        )
        ref_idx = align_info.get("reference_frame_index", align_reference_frame)
        print(
            f"alignment: mode={align_mode!r}, reference_frame_index={ref_idx} "
            f"(epoch={epochs[int(ref_idx)] if epochs else '?'})"
        )
        if "nonlin_sigma" in align_info:
            sig = np.asarray(align_info["nonlin_sigma"])
            print(
                "  nonlinearity spectrum (descending): "
                + ", ".join(f"{s:.3g}" for s in sig)
            )

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
            align_info=align_info,
            eta_stack_raw=eta_stack_raw if (align_info is not None and keep_raw_in_npz) else None,
            frob_scores=frob_scores,
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

    def _mathwrap(label: str) -> str:
        """Return a matplotlib-safe mathtext label.

        Accepts any of:
        - ``"theta_1"`` (plain text)            → ``"$theta_1$"``
        - ``r"\theta_1"`` (TeX, no delimiters)  → ``"$\theta_1$"``
        - ``r"$\theta_1$"`` (already wrapped)   → unchanged

        The previous heuristic (``"\\" not in s``) failed because Python
        string literals like ``"$\theta_1$"`` (non-raw) interpret ``\t``
        as a tab, so the check picks the wrong branch and double-wraps
        ``$ ... $`` into a broken ``$$ ... $$``.
        """
        s = str(label)
        if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
            return s
        return rf"${s}$"

    # Panel titles reflect the alignment state so the viewer knows
    # what the contour values represent.
    if align_mode == "linear_residual":
        _panel_label = lambda k: rf"$\eta_{k + 1} - (A\theta + b)_{k + 1}$"
        _align_blurb = "linear residual"
    elif align_mode == "nonlinearity_rotation":
        _panel_label = lambda k: rf"$(R\,\eta)_{k + 1}$"
        _align_blurb = "nonlinearity rotation"
    elif align_mode == "both":
        _panel_label = lambda k: rf"$(R\,(\eta - A\theta - b))_{k + 1}$"
        _align_blurb = "residual + rotation"
    else:
        _panel_label = lambda k: rf"$\eta_{k + 1}$"
        _align_blurb = None

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
        ax.set_xlabel(_mathwrap(param_names[0]))
        ax.set_ylabel(_mathwrap(param_names[1]))
        ax.set_title(_panel_label(k))
        return cs

    def _format_score(val: float) -> str:
        """Render the per-frame Frobenius score for the super-title."""
        if not np.isfinite(val):
            return ""
        a = abs(val)
        if a != 0 and (a >= 1e3 or a < 1e-2):
            return f"${loss_label} = {val:.2e}$"
        return f"${loss_label} = {format(val, loss_fmt)}$"

    _scores_available = show_loss and np.any(np.isfinite(frob_scores))
    if show_loss and not _scores_available:
        print(
            "show_loss=True but no `frob_score` fields were found in the "
            "snapshots; rerun training with the updated "
            "`fit_flattening_with_snapshots` to capture the validation "
            "Frobenius score per snapshot. Titles will fall back to "
            "epoch-only."
        )

    def _title_text(i: int) -> str:
        base = f"{_suptitle_prefix}  |  epoch {epochs[i]}"
        if _scores_available:
            score_str = _format_score(float(frob_scores[i]))
            if score_str:
                base = base + "  |  " + score_str
        return base

    cs0 = _draw(axes[0], 0, 0)
    cs1 = _draw(axes[1], 1, 0)
    cbar0 = plt.colorbar(cs0, ax=axes[0])
    cbar1 = plt.colorbar(cs1, ax=axes[1])
    _suptitle_prefix = title_prefix + (f"  [{_align_blurb}]" if _align_blurb else "")
    suptitle = fig.suptitle(_title_text(0))

    def update(i):
        # Contour plots can't be updated in-place; redraw and refresh colour bars.
        cs0_new = _draw(axes[0], 0, i)
        cs1_new = _draw(axes[1], 1, i)
        cbar0.update_normal(cs0_new)
        cbar1.update_normal(cs1_new)
        suptitle.set_text(_title_text(i))
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
    p.add_argument(
        "--align-mode",
        default="none",
        choices=("none", "linear_residual", "nonlinearity_rotation", "both"),
        help=(
            "Pop the nonlinear part of eta out of the contours. "
            "'linear_residual': subtract best affine fit. "
            "'nonlinearity_rotation': rotate eta by the orthogonal map "
            "from align_coords.nonlinearity_rotation. "
            "'both': linear residual + rotation. Transform computed once "
            "on the reference frame (default: last) and applied to all "
            "frames. Default: none."
        ),
    )
    p.add_argument(
        "--align-reference-frame",
        type=int,
        default=-1,
        help=(
            "Frame index used to derive the alignment transform "
            "(negative indices supported). Default -1 (last frame)."
        ),
    )
    p.add_argument(
        "--show-loss",
        action="store_true",
        help=(
            "Display the validation flatness Frobenius score "
            "(||Q-I||_F) next to the epoch number in the gif's "
            "super-title. Requires snapshots written by the updated "
            "fit_flattening_with_snapshots."
        ),
    )
    p.add_argument(
        "--loss-fmt",
        default=".2f",
        help=(
            "Format spec for the loss readout (e.g. '.2f', '.1f', "
            "'.2e'). Default '.2f'."
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
        align_mode=args.align_mode,
        align_reference_frame=args.align_reference_frame,
        show_loss=args.show_loss,
        loss_fmt=args.loss_fmt,
    )


if __name__ == "__main__":
    main()
