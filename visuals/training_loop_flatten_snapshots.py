"""
Modified version of `degeneracy_distillery.training_loop_flatten.fit_flattening`
that saves grid-based snapshots of the learned coordinates ``eta`` from the
**main model only** (phase 1 + phase 2; ensemble fine-tuning is skipped for
snapshots, by design).

What gets saved
---------------
For each saved global epoch ``j``, we evaluate the current ``model.apply(w, X)``
on a 2D grid in ``(theta_1, theta_2)`` (extra dimensions, if present, are
held at the midpoint of their training range, matching the legacy
``do_plot=True`` block) and write ``eta_grid`` of shape
``(grid_num_pts * grid_num_pts, n_params)`` to disk.

The companion script :mod:`visuals.make_eta_grid_gif` consumes these
snapshots and renders a 2-panel contour ``.gif`` of ``eta_1`` and
``eta_2`` over ``(theta_1, theta_2)`` evolving with training, plus a
portable ``.npz`` of the same timeseries for local restyling.

Burn-in
-------
The flattener starts very far from sensible coordinates; the first few
hundred epochs produce visually distracting snapshots. Use ``burn_in``
(default 500) to discard everything earlier than that global epoch. The
first kept snapshot is at ``burn_in`` (if ``save_initial`` is True) or
at the next multiple of ``save_every`` after ``burn_in``.

Snapshot layout under ``snapshots_outdir``::

    metadata.json         # save_every, burn_in, param_names, grid_num_pts, ...
    grid_axes.npz         # xs, ys, X_grid (theta grid used for evaluation)
    epoch_00500.npz       # eta_grid: (grid_num_pts**2, n_params)
    epoch_00505.npz
    ...
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Any, Callable, Literal, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import scipy
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from degeneracy_distillery.io_utils import create_results_dict  # noqa: E402
from degeneracy_distillery.training_loop_flatten import (  # noqa: E402
    ForwardBackwardMLP,
    RealNVPWrapper,
    WhitenedForwardBackwardMLP,
    WhitenedMLP,
    WhitenedRealNVP,
    compute_robust_norm_factor,
    compute_whitening_transform,
    custom_MLP,
    stable_sin_swish,
    weighted_std,
)


# ----------------------------------------------------------------------------
# Snapshot helpers
# ----------------------------------------------------------------------------

def _clear_or_make(snap_dir: str) -> None:
    if os.path.exists(snap_dir):
        for filename in os.listdir(snap_dir):
            p = os.path.join(snap_dir, filename)
            try:
                if os.path.isfile(p) or os.path.islink(p):
                    os.unlink(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
            except Exception as e:
                print(f"Failed to delete {p}. Reason: {e}")
    else:
        os.makedirs(snap_dir, exist_ok=True)


def _save_eta_snapshot(
    snap_dir: str,
    epoch: int,
    eta_grid: np.ndarray,
    frob_score: Optional[float] = None,
) -> None:
    """Write a per-epoch snapshot.

    Always saves ``eta_grid``. When ``frob_score`` is supplied, also
    writes the scalar ``frob_score`` (the validation
    ``mean_b ||Q_b - I||_F`` flatness metric) so the gif builder can
    display it next to the epoch number.
    """
    payload: dict[str, np.ndarray] = {"eta_grid": np.asarray(eta_grid)}
    if frob_score is not None:
        payload["frob_score"] = np.asarray(float(frob_score), dtype=np.float64)
    np.savez_compressed(
        os.path.join(snap_dir, f"epoch_{epoch:05d}.npz"),
        **payload,
    )


def _build_eval_grid(
    min_x: np.ndarray,
    max_x: np.ndarray,
    n_params: int,
    grid_num_pts: int,
):
    """Mirror the legacy ``do_plot=True`` grid construction.

    For ``n_params == 2`` returns a meshgrid in ``(theta_1, theta_2)``.
    For ``n_params > 2``, extra dimensions are held at the midpoint of
    their training range so the plotting plane stays 2D.
    """
    xs = jnp.linspace(min_x[0], max_x[0], grid_num_pts)
    ys = jnp.linspace(min_x[1], max_x[1], grid_num_pts)
    if n_params > 2:
        extra = []
        for j in range(n_params - 2):
            zs = jnp.ones(grid_num_pts) * (
                (max_x[2 + j : 3 + j] - min_x[2 + j : 3 + j]) / 2.0
            )
            extra.append(zs)
        grds = jnp.meshgrid(xs, ys, *extra)
        X = jnp.stack([g.flatten() for g in grds], axis=-1)
        xs_mesh, ys_mesh = grds[0], grds[1]
    else:
        xs_mesh, ys_mesh = jnp.meshgrid(xs, ys)
        X = jnp.stack([xs_mesh.flatten(), ys_mesh.flatten()], axis=-1)
    return xs_mesh, ys_mesh, X


def _write_metadata(
    snap_dir: str,
    *,
    save_every: int,
    burn_in: int,
    save_initial: bool,
    grid_num_pts: int,
    n_params: int,
    param_names: Sequence[str],
    saved_epochs: Sequence[int],
    final_global_epoch: int,
    notes: str = "",
    saved_frob_scores: Optional[Sequence[float]] = None,
) -> None:
    meta = {
        "save_every": int(save_every),
        "burn_in": int(burn_in),
        "save_initial": bool(save_initial),
        "grid_num_pts": int(grid_num_pts),
        "n_params": int(n_params),
        "param_names": list(param_names),
        "saved_epochs": [int(e) for e in saved_epochs],
        "final_global_epoch": int(final_global_epoch),
        "notes": notes,
    }
    if saved_frob_scores is not None:
        # JSON cannot store NaN; encode missing values as None.
        meta["saved_frob_scores"] = [
            (None if (isinstance(s, float) and not np.isfinite(s)) else float(s))
            for s in saved_frob_scores
        ]
    with open(os.path.join(snap_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ----------------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------------

def fit_flattening_with_snapshots(
    F_network_ensemble,
    θs,
    ensemble_weights,
    *,
    snapshots_outdir: str = "flatten-log-snapshots/snapshots",
    save_every: int = 5,
    burn_in: int = 500,
    grid_num_pts: int = 30,
    save_initial: bool = True,
    param_names: Optional[Sequence[str]] = None,
    enable_snapshots: bool = True,
    # ----- below: identical to fit_flattening -----
    hidden_size: int = 256,
    n_layers: int = 3,
    batch_size: int = 250,
    epochs_phase1: int = 1000,
    epochs_phase2: int = 1000,
    finetune_epochs: int = 400,
    min_epochs: int = 1200,
    patience: int = 100,
    lr_phase1: float = 2e-6,
    lr_schedule_initial: float = 7e-5,
    lr_decay: float = 0.3,
    lr_finetune: float = 4e-6,
    l1_alpha: float = 0.0,
    noise: float = 1e-6,
    seed: int = 0,
    output_prefix: str = "flattened_coords_sr",
    SCALE_THETA: bool = False,
    do_average: bool = True,
    Fisher_to_flatten: Literal["average", "best"] = "average",
    F_avg: Any = None,
    norm_factor: Any = None,
    norm_method: str = "median_max_eig",
    use_whitening: bool = True,
    minmax_scale_inputs: bool = True,
    offset: float = 0.1,
    augment_log_inputs: bool = False,
    nn_inv: bool = False,
    forward_backward_mlp: bool = False,
    forward_backward_invertibility_weight: float = 1.0,
    flattener_activation: Union[
        Literal["sin_swish", "softplus"], Callable[[Any], Any]
    ] = "sin_swish",
    loss_type: Literal[
        "log_frob", "frob", "squared_frob", "squared_frob_det"
    ] = "log_frob",
    loss_reweight_lambda: float = 100.0,
    loss_reweight_epsilon: float = 1e-7,
    loss_log_epsilon: float = 1e-12,
    loss_log_tau: float = 0.1,
    q_inv_jitter: float = 1e-8,
    beta_det: float = 0.1,
    grad_clip_norm: Optional[float] = None,
    lr_schedule_phase1: Optional[Any] = None,
    lr_schedule_phase2: Optional[Any] = None,
    lr_schedule_finetune: Optional[Any] = None,
    update_pbar_every: int = 10,
    return_model: bool = False,
    save_flatten_model_pickle: bool = False,
):
    """Same signature and return contract as
    :func:`degeneracy_distillery.training_loop_flatten.fit_flattening`,
    plus six snapshot-specific keyword-only arguments at the start.

    Snapshot kwargs
    ---------------
    snapshots_outdir : str
        Directory to write the snapshot ``epoch_*.npz`` files,
        ``grid_axes.npz``, and ``metadata.json`` into. Cleared on entry.
    save_every : int, default 5
        Save a snapshot every this many global training epochs (phase 1
        + phase 2 combined).
    burn_in : int, default 500
        Discard snapshots before this global epoch. The flattener is
        typically still in nonsense-coordinate territory until ~500
        epochs.
    grid_num_pts : int, default 30
        Number of points along each axis of the 2D evaluation grid
        (matches the legacy ``do_plot=True`` block).
    save_initial : bool, default True
        If True and ``burn_in == 0``, save a snapshot at epoch 0
        (untrained init).
    param_names : sequence of str, optional
        Display names recorded in ``metadata.json``; defaults to
        ``["theta_1", "theta_2", ...]``.
    enable_snapshots : bool, default True
        Master switch (set False to make the function a no-op
        snapshot-wise; useful for A/B comparisons).
    """
    # ---------------------- SNAPSHOT SETUP -----------------------
    if enable_snapshots:
        _clear_or_make(snapshots_outdir)
    n_params = θs.shape[-1]
    if param_names is None:
        param_names = [f"theta_{i + 1}" for i in range(n_params)]
    elif len(param_names) != n_params:
        raise ValueError(
            f"param_names has length {len(param_names)}; expected {n_params}"
        )

    # ---------------------- CONSTANTS & SETUP -----------------------
    key = jr.PRNGKey(seed)

    # Truncate to a multiple of batch_size (identical to upstream).
    _n_in = int(θs.shape[0])
    _n_drop = _n_in % int(batch_size)
    if _n_drop:
        key, _trunc_key = jr.split(key)
        _trunc_idx = np.asarray(
            jr.permutation(_trunc_key, _n_in)[: _n_in - _n_drop]
        )
        θs = θs[_trunc_idx]
        F_network_ensemble = F_network_ensemble[:, _trunc_idx]
        if F_avg is not None:
            F_avg = F_avg[_trunc_idx]
        print(
            f"WARNING: n_samples ({_n_in}) is not divisible by batch_size "
            f"({batch_size}); randomly dropping {_n_drop} samples "
            f"({100.0 * _n_drop / _n_in:.2f}%)."
        )

    if Fisher_to_flatten not in ("average", "best"):
        raise ValueError(
            f"Fisher_to_flatten must be 'average' or 'best', got {Fisher_to_flatten!r}"
        )
    _fisher_mode: Literal["average", "best"] = (
        "best" if (Fisher_to_flatten == "best" or not do_average) else "average"
    )
    best_idx = int(jnp.argmax(jnp.asarray(ensemble_weights)))

    if F_avg is None:
        if _fisher_mode == "average":
            print("AVERAGING FISHERS (weighted ensemble)")
            F_fishnets = jnp.average(
                F_network_ensemble, axis=0, weights=ensemble_weights
            )
        else:
            print(f"USING BEST FISHER ENSEMBLE MEMBER (index {best_idx})")
            F_fishnets = F_network_ensemble[best_idx]
    else:
        F_fishnets = F_avg

    if norm_factor is None:
        print(f"COMPUTING ROBUST NORM FACTOR (method: {norm_method})")
        norm_factor = compute_robust_norm_factor(F_network_ensemble, method=norm_method)
    print(f"norm_factor = {norm_factor:.6g}")
    F_fishnets = F_fishnets / norm_factor

    W = None
    W_inv = None
    F_mean = None
    if use_whitening:
        print("COMPUTING WHITENING TRANSFORM")
        F_ensemble_normalized = F_network_ensemble / norm_factor
        if _fisher_mode == "average":
            W, W_inv, F_mean = compute_whitening_transform(
                F_ensemble_normalized, ensemble_weights
            )
        else:
            W, W_inv, F_mean = compute_whitening_transform(
                F_ensemble_normalized[best_idx : best_idx + 1],
                jnp.ones((1,), dtype=ensemble_weights.dtype),
            )

    max_x = θs.max(0) + 1e-3
    min_x = θs.min(0) - 1e-3

    if nn_inv and forward_backward_mlp:
        raise ValueError("nn_inv and forward_backward_mlp cannot both be True.")

    _string_activations = {"sin_swish": stable_sin_swish, "softplus": nn.softplus}
    if isinstance(flattener_activation, str):
        if flattener_activation not in _string_activations:
            raise ValueError(
                f"flattener_activation string must be one of "
                f"{sorted(_string_activations)}, got {flattener_activation!r}"
            )
        _flattener_act = _string_activations[flattener_activation]
        _flattener_act_name = flattener_activation
    elif callable(flattener_activation):
        _flattener_act = flattener_activation
        _flattener_act_name = getattr(
            flattener_activation, "__name__", repr(flattener_activation)
        )
    else:
        raise TypeError(
            "flattener_activation must be a string or callable, got "
            f"{type(flattener_activation).__name__}"
        )
    if loss_type not in ("log_frob", "frob", "squared_frob", "squared_frob_det"):
        raise ValueError(f"unknown loss_type {loss_type!r}")
    if augment_log_inputs and nn_inv:
        print("WARNING: augment_log_inputs is not supported with RealNVP; ignoring.")
        augment_log_inputs = False

    _feat = [hidden_size] * n_layers + [n_params]

    # ---------------------- MODEL SELECTION (identical to upstream) ---
    if forward_backward_mlp and use_whitening:
        model = WhitenedForwardBackwardMLP(
            features=_feat, max_x=max_x, min_x=min_x, W_inv=W_inv,
            minmax_scale_inputs=minmax_scale_inputs,
            augment_log_inputs=augment_log_inputs,
            act=_flattener_act, apply_inverse_whitening=True, offset=offset,
        )
    elif forward_backward_mlp:
        model = ForwardBackwardMLP(
            features=_feat, max_x=max_x, min_x=min_x,
            minmax_scale_inputs=minmax_scale_inputs,
            augment_log_inputs=augment_log_inputs,
            act=_flattener_act, offset=offset,
        )
    elif nn_inv and use_whitening:
        model = WhitenedRealNVP(
            num_layers=n_layers, hidden_dims=hidden_size, input_dim=n_params,
            max_x=max_x, min_x=min_x, W_inv=W_inv,
            minmax_scale_inputs=minmax_scale_inputs,
            act=_flattener_act, apply_inverse_whitening=True, offset=offset,
        )
    elif nn_inv:
        model = RealNVPWrapper(
            num_layers=n_layers, hidden_dims=hidden_size, input_dim=n_params,
            max_x=max_x, min_x=min_x,
            minmax_scale_inputs=minmax_scale_inputs,
            act=_flattener_act, offset=offset,
        )
    elif use_whitening:
        model = WhitenedMLP(
            features=_feat, max_x=max_x, min_x=min_x, W_inv=W_inv,
            minmax_scale_inputs=minmax_scale_inputs,
            augment_log_inputs=augment_log_inputs,
            act=_flattener_act, apply_inverse_whitening=True, offset=offset,
        )
    else:
        model = custom_MLP(
            features=_feat, max_x=max_x, min_x=min_x,
            minmax_scale_inputs=minmax_scale_inputs,
            augment_log_inputs=augment_log_inputs,
            act=_flattener_act, offset=offset,
        )
    print(f"Using model: {type(model).__name__}")

    # ---------------------- BUILD EVAL GRID + WRITE METADATA ----------
    xs_mesh, ys_mesh, X_grid = _build_eval_grid(
        np.asarray(min_x), np.asarray(max_x), n_params, grid_num_pts
    )
    if enable_snapshots:
        np.savez_compressed(
            os.path.join(snapshots_outdir, "grid_axes.npz"),
            xs_mesh=np.asarray(xs_mesh),
            ys_mesh=np.asarray(ys_mesh),
            X_grid=np.asarray(X_grid),
            min_x=np.asarray(min_x),
            max_x=np.asarray(max_x),
        )

    # JIT the grid evaluation; recompiles once per `w` pytree shape.
    @jax.jit
    def _eta_on_grid(w):
        return jax.vmap(lambda d: model.apply(w, d))(X_grid)

    # JIT'd validation Frobenius flatness score, independent of the
    # training `loss_type`: returns the mean over a batch of theta of
    # ``||Q - I||_F`` where ``Q = (J^-1).T @ F @ J^-1``. This is what we
    # display next to the epoch number when ``show_loss=True`` in the
    # gif builder.
    @jax.jit
    def _frob_score_on_batch(w, theta_batch, F_batch):
        def _per_sample(theta_one, F_one):
            apply = lambda d: model.apply(w, d)
            J = jax.jacrev(apply)(theta_one).squeeze()
            J_inv = jnp.linalg.pinv(J)
            Q = J_inv.T @ F_one @ J_inv
            eye = jnp.eye(n_params)
            return jnp.sqrt(jnp.sum((Q - eye) ** 2))
        return jnp.mean(jax.vmap(_per_sample)(theta_batch, F_batch))

    saved_epochs: list[int] = []
    saved_frob_scores: list[float] = []
    # Snapshot-time validation slice — frozen at the same shape as the
    # training-loop's val slice (last 5 batches, in theta/F space).
    # Populated once below, after the data is reshaped.
    _val_theta_for_score: Optional[jnp.ndarray] = None
    _val_F_for_score: Optional[jnp.ndarray] = None

    def _maybe_save(j_global: int, w) -> None:
        if not enable_snapshots:
            return
        if j_global < burn_in:
            return
        if (j_global - burn_in) % save_every != 0:
            return
        eta_grid = np.asarray(_eta_on_grid(w))
        score: Optional[float] = None
        if _val_theta_for_score is not None and _val_F_for_score is not None:
            try:
                s = float(np.asarray(
                    _frob_score_on_batch(w, _val_theta_for_score, _val_F_for_score)
                ))
                # Guard against the occasional inf / nan early in training.
                score = s if np.isfinite(s) else None
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: frob score failed at epoch {j_global}: {exc}")
                score = None
        _save_eta_snapshot(snapshots_outdir, j_global, eta_grid, frob_score=score)
        saved_epochs.append(j_global)
        saved_frob_scores.append(float("nan") if score is None else score)

    # ---------------------- LOSS DEFINITION (identical to upstream) ---
    @jax.jit
    def norm(A):
        return jnp.sqrt(jnp.einsum("ij,ij->", A, A))

    _loss_lam = loss_reweight_lambda
    _loss_eps = loss_reweight_epsilon
    _loss_alpha = float(
        -np.log(_loss_eps * (_loss_lam - 1.0) + _loss_eps ** 2.0 / (1.0 + _loss_eps))
        / _loss_eps
    )
    _log_tau = loss_log_tau
    _q_jitter = q_inv_jitter
    _inv_pen_w = forward_backward_invertibility_weight
    _l1_alpha = l1_alpha
    _loss_type = loss_type
    _beta_det = float(beta_det)

    if forward_backward_mlp:
        @jax.jit
        def info_loss(w, theta_batched, F_batched):
            def fn(theta, F):
                mymodel = lambda d: model.apply(w, d)
                eta = mymodel(theta)
                theta_rec = model.apply(w, eta, method="inverse_path")
                inv_pen = jnp.mean((theta - theta_rec) ** 2)

                J_eta = jax.jacrev(mymodel)(theta).squeeze()
                jac_l1 = _l1_alpha * jnp.mean(jnp.abs(J_eta))
                jac_l1 = jnp.nan_to_num(jac_l1, nan=0.0, posinf=0.0, neginf=0.0)
                Jeta_inv = jnp.linalg.pinv(J_eta)
                Q = Jeta_inv.T @ F @ Jeta_inv
                eye = jnp.eye(n_params)

                det_q = jnp.linalg.det(Q)
                det_q = jnp.nan_to_num(det_q, nan=0.0, posinf=0.0, neginf=0.0)

                if _loss_type in ("squared_frob", "squared_frob_det"):
                    loss = jnp.sum((Q - eye) ** 2)
                    loss = jnp.where(jnp.isfinite(loss), loss, jnp.asarray(1e6, dtype=loss.dtype))
                    if _loss_type == "squared_frob_det":
                        Q_reg = Q + _q_jitter * eye
                        _, logabsdet_q = jnp.linalg.slogdet(Q_reg)
                        det_pen = logabsdet_q ** 2
                        det_pen = jnp.nan_to_num(det_pen, nan=0.0, posinf=0.0, neginf=0.0)
                        loss = loss + _beta_det * det_pen
                else:
                    Q_reg = Q + _q_jitter * eye
                    inv_term = jnp.linalg.inv(Q_reg)
                    loss = norm(Q - eye) + norm(inv_term - eye)
                    loss = jnp.where(jnp.isfinite(loss), loss, jnp.asarray(1e6, dtype=loss.dtype))
                    r = _loss_lam * loss / (loss + jnp.exp(-_loss_alpha * loss))
                    loss *= r

                loss = loss + _inv_pen_w * inv_pen
                if _loss_type == "log_frob":
                    loss = _log_tau * jnp.log1p(loss / _log_tau)
                loss = jnp.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                return loss, det_q, jac_l1

            log_losses, dets, l1_terms = jax.vmap(fn)(theta_batched, F_batched)
            return jnp.mean(log_losses) + l1_terms.mean(), jnp.mean(dets)
    else:
        @jax.jit
        def info_loss(w, theta_batched, F_batched):
            def fn(theta, F):
                mymodel = lambda d: model.apply(w, d)
                J_eta = jax.jacrev(mymodel)(theta).squeeze()
                jac_l1 = _l1_alpha * jnp.mean(jnp.abs(J_eta))
                jac_l1 = jnp.nan_to_num(jac_l1, nan=0.0, posinf=0.0, neginf=0.0)
                Jeta_inv = jnp.linalg.pinv(J_eta)
                Q = Jeta_inv.T @ F @ Jeta_inv
                eye = jnp.eye(n_params)

                det_q = jnp.linalg.det(Q)
                det_q = jnp.nan_to_num(det_q, nan=0.0, posinf=0.0, neginf=0.0)

                if _loss_type in ("squared_frob", "squared_frob_det"):
                    loss = jnp.sum((Q - eye) ** 2)
                    loss = jnp.where(jnp.isfinite(loss), loss, jnp.asarray(1e6, dtype=loss.dtype))
                    if _loss_type == "squared_frob_det":
                        Q_reg = Q + _q_jitter * eye
                        _, logabsdet_q = jnp.linalg.slogdet(Q_reg)
                        det_pen = logabsdet_q ** 2
                        det_pen = jnp.nan_to_num(det_pen, nan=0.0, posinf=0.0, neginf=0.0)
                        loss = loss + _beta_det * det_pen
                else:
                    Q_reg = Q + _q_jitter * eye
                    inv_term = jnp.linalg.inv(Q_reg)
                    loss = norm(Q - eye) + norm(inv_term - eye)
                    loss = jnp.where(jnp.isfinite(loss), loss, jnp.asarray(1e6, dtype=loss.dtype))
                    r = _loss_lam * loss / (loss + jnp.exp(-_loss_alpha * loss))
                    loss *= r

                if _loss_type == "log_frob":
                    loss = _log_tau * jnp.log1p(loss / _log_tau)
                loss = jnp.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                return loss, det_q, jac_l1

            log_losses, dets, l1_terms = jax.vmap(fn)(theta_batched, F_batched)
            return jnp.mean(log_losses) + l1_terms.mean(), jnp.mean(dets)

    # ---------------------- PREPARE TRAINING DATA -----------------------
    key, shuffle_key = jr.split(key)
    n_samples = θs.shape[0]
    shuffle_idx = jr.permutation(shuffle_key, jnp.arange(n_samples))
    θs_shuffled = θs[shuffle_idx]
    F_fishnets_shuffled = F_fishnets[shuffle_idx]
    theta_true = θs_shuffled.reshape(-1, batch_size, n_params)
    F_fishnets = F_fishnets_shuffled.reshape(-1, batch_size, n_params, n_params)

    # Fixed snapshot-time validation slice for the Frobenius score:
    # matches the training-loop convention (last 5 batches, flattened).
    _val_size_for_score = 5
    if theta_true.shape[0] >= _val_size_for_score:
        _val_theta_for_score = theta_true[-_val_size_for_score:].reshape(
            -1, n_params
        )
        _val_F_for_score = F_fishnets[-_val_size_for_score:].reshape(
            -1, n_params, n_params
        )

    # ---------------------- TRAINING LOOP -----------------------
    _pbar_stride = max(1, int(update_pbar_every))

    def training_loop(
        key, w, theta_true, F_fishnets,
        val_size: int = 5,
        lr=1e-5,
        batch_size: int = batch_size,
        patience: int = patience,
        epochs: int = epochs_phase1,
        min_epochs: int = min_epochs,
        opt_type=None,
        epoch_offset: int = 0,
        do_snapshots: bool = True,
    ):
        best_w = w
        best_loss = jnp.inf
        base_opt = (
            optax.adam(learning_rate=lr) if opt_type is None else opt_type(learning_rate=lr)
        )
        if grad_clip_norm is not None:
            tx = optax.chain(optax.clip_by_global_norm(grad_clip_norm), base_opt)
        else:
            tx = base_opt
        opt_state = tx.init(w)
        loss_grad_fn = jax.value_and_grad(info_loss, has_aux=True)

        def body_fun(i, inputs):
            w, loss_val, opt_state, detFeta, key, theta_true, F_fishnets = inputs
            theta_samples = theta_true[i]
            F_samples = F_fishnets[i]
            key, sk = jr.split(key)
            L = jax.scipy.linalg.cholesky(F_samples, lower=True)
            n_p = L.shape[-1]
            mask = jnp.tril(jnp.ones((n_p, n_p), dtype=L.dtype), k=-1)
            Z = jr.normal(sk, shape=L.shape)
            L_noisy = L + noise * Z * mask
            F_samples = jnp.einsum("bij,bkj->bik", L_noisy, L_noisy)
            (loss_val, detFeta), grads = loss_grad_fn(w, theta_samples, F_samples)
            updates, opt_state = tx.update(grads, opt_state)
            w = optax.apply_updates(w, updates)
            return w, loss_val, opt_state, detFeta, key, theta_true, F_fishnets

        num_sims = theta_true.reshape(-1, n_params).shape[0]
        lower = 0
        upper = theta_true.shape[0]
        losses = jnp.zeros(epochs)
        detFetas = jnp.zeros(epochs)
        val_losses = jnp.zeros(epochs)
        val_detFetas = jnp.zeros(epochs)
        loss = 0.0
        detFeta = 0.0
        val_detFeta = 0.0
        counter = 0

        # Optional snapshot at the very start of the very first phase.
        if (
            do_snapshots
            and save_initial
            and epoch_offset == 0
            and burn_in == 0
        ):
            _maybe_save(0, w)

        pbar = tqdm(range(epochs), leave=True, position=0, miniters=_pbar_stride)
        j = 0
        for j in pbar:
            if (counter > patience) and (j + 1 > min_epochs):
                print("\n patience reached. stopping training.")
                losses = losses[:j]
                detFetas = detFetas[:j]
                val_losses = val_losses[:j]
                val_detFetas = val_detFetas[:j]
                pbar.set_description(
                    "epoch %d loss: %.4f, det F(η): %.4f, val det F(η): %.4f"
                    % (j, loss, detFeta, val_detFeta)
                )
                break
            else:
                key, rng = jr.split(key)
                randidx = jr.permutation(key, jnp.arange(num_sims), independent=True)
                theta_train = theta_true.reshape(-1, n_params)[randidx].reshape(
                    -1, batch_size, n_params
                )
                F_train = F_fishnets.reshape(-1, n_params, n_params)[randidx].reshape(
                    -1, batch_size, n_params, n_params
                )
                init_vals = (w, loss, opt_state, detFeta, key, theta_train, F_train)
                w, loss, opt_state, detFeta, key, theta_train, F_train = jax.lax.fori_loop(
                    lower, upper, body_fun, init_vals
                )
                theta_val = theta_true[-val_size:].reshape(-1, n_params)
                F_val = F_fishnets[-val_size:].reshape(-1, n_params, n_params)
                (val_loss, val_detFeta), _ = loss_grad_fn(w, theta_val, F_val)
                losses = losses.at[j].set(loss)
                detFetas = detFetas.at[j].set(detFeta)
                val_losses = val_losses.at[j].set(val_loss)
                val_detFetas = val_detFetas.at[j].set(val_detFeta)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_w = w
                    counter = 0
                else:
                    counter += 1

                if do_snapshots:
                    j_global = (j + 1) + epoch_offset  # 1-indexed global epoch
                    _maybe_save(j_global, w)

            if (j + 1) % _pbar_stride == 0 or j == epochs - 1:
                pbar.set_description(
                    "epoch %d loss: %.4f, det F(η): %.4f, val det F(η): %.4f"
                    % (j, loss, detFeta, val_detFeta)
                )

        epochs_done = j + 1
        return best_w, (losses, val_losses), (detFetas, val_detFetas), epochs_done

    # ---------------------- TRAINING PHASE 1 -----------------------
    print("TRAINING FLATTENER NET (phase 1)")
    key, rng = jr.split(key)
    x_init = jnp.ones((n_params,))
    if forward_backward_mlp:
        w = model.init(key, x_init, method="init_forward_and_reverse")
    else:
        w = model.init(key, x_init)
    lr1 = lr_schedule_phase1 if lr_schedule_phase1 is not None else lr_phase1
    w, all_loss, all_dets, epochs_done_p1 = training_loop(
        key, w, theta_true, F_fishnets,
        lr=lr1, opt_type=optax.adam,
        epochs=epochs_phase1,
        epoch_offset=0,
        do_snapshots=True,
    )

    # ---------------------- TRAINING PHASE 2 -----------------------
    print("FINE-TUNING FLATTENER NET (phase 2)")
    if lr_schedule_phase2 is not None:
        lr2 = lr_schedule_phase2
    else:
        total_steps = epochs_phase2 * (F_fishnets.shape[0]) + epochs_phase2
        lr2 = optax.schedules.exponential_decay(
            init_value=lr_schedule_initial, transition_begin=0,
            transition_steps=total_steps, decay_rate=lr_decay,
        )
    w, all_loss, all_dets, epochs_done_p2 = training_loop(
        key, w, theta_true, F_fishnets,
        lr=lr2, opt_type=optax.adam,
        epochs=epochs_phase2,
        epoch_offset=epochs_done_p1,
        do_snapshots=True,
    )
    final_global_epoch = epochs_done_p1 + epochs_done_p2

    # ---------------------- ENSEMBLE FINE-TUNING (NO SNAPSHOTS) ----------
    F_ensemble = jnp.array(F_network_ensemble) / norm_factor
    F_ensemble_for_training = F_ensemble  # whitening already implicit via model
    theta_true = θs.reshape(-1, batch_size, n_params)
    F_fishnets_ensemble = [
        f.reshape(-1, batch_size, n_params, n_params) for f in F_ensemble_for_training
    ]

    print("FINE-TUNING EACH ENSEMBLE MEMBER (snapshots disabled by design)")
    ensemble_ws = []
    init_anew = False
    for k, f in enumerate(F_fishnets_ensemble):
        print("fine-tuning for ensemble member %d" % (k))
        key, rng = jr.split(key)
        if init_anew:
            key, rng = jr.split(key)
            if forward_backward_mlp:
                _w = model.init(key, x_init, method="init_forward_and_reverse")
            else:
                _w = model.init(key, x_init)
        else:
            _w = w
        lr_ft = lr_schedule_finetune if lr_schedule_finetune is not None else lr_finetune
        _w, all_loss, all_dets, _ = training_loop(
            key, _w, theta_true, f,
            lr=lr_ft, opt_type=optax.adam,
            epochs=finetune_epochs, patience=20,
            do_snapshots=False,
        )
        ensemble_ws.append(_w)

    # ---------------------- EVALUATION -----------------------
    @jax.jit
    def get_jacobian(θ, w=w):
        mymodel = lambda d: model.apply(w, d)
        return jax.jacobian(mymodel)(θ)

    η_ensemble = []
    Jbar_ensemble = []
    mymodel = lambda d: model.apply(w, d)
    for k, _w in enumerate(ensemble_ws):
        print("applying model to ensemble member %d" % (k))
        current_model = lambda d: model.apply(_w, d)
        ηs_k = jax.vmap(current_model)(θs)
        getjac = lambda d: get_jacobian(d, w=_w)
        η_ensemble.append(ηs_k)
        Jbar_ensemble.append(
            jnp.concatenate(
                jnp.array(
                    [
                        jax.vmap(getjac)(t)
                        for t in θs.reshape(-1, batch_size, n_params)
                    ]
                )
            )
        )

    ηs = jax.vmap(mymodel)(θs)
    Jbar = jnp.concatenate(
        jnp.array(
            [jax.vmap(get_jacobian)(t) for t in θs.reshape(-1, batch_size, n_params)]
        )
    )

    allFs = jnp.array(F_ensemble)
    dFs = weighted_std(allFs, jnp.ones(allFs.shape), axis=0)

    def get_δJ(F, δF, Jbar):
        J = np.linalg.inv(Jbar)
        Q = -np.einsum("bik,bkj,blj->bil", J, δF, J)
        X = J @ F
        A = np.einsum("bij,bkj->bik", X, X)
        S = jnp.array(
            [
                scipy.linalg.solve_sylvester(a=A[i], b=A[i], q=Q[i])
                for i in range(Q.shape[0])
            ]
        )
        δJ = S @ X
        return np.linalg.inv(J + δJ) - Jbar, δJ

    print("CALCULATING JACOBIAN ERROR")
    δJs, δinvJ = get_δJ(allFs.mean(0), dFs, Jbar)

    print("(skipping global rotation correction; visualisation context)")
    ys = []
    dys = []
    F_ensemble_out = []
    weights = []
    for i, y in enumerate(η_ensemble):
        try:
            ys.append(y)
            dy = Jbar_ensemble[i]
            dys.append(dy)
            weights.append(ensemble_weights[i])
            F_ensemble_out.append(allFs[i])
        except Exception:
            pass

    outname = output_prefix
    if SCALE_THETA:
        outname += "_scaled"

    output_dict = create_results_dict(
        theta=np.array(θs),
        eta=np.array(ηs),
        Jacobians=np.array(Jbar),
        deltaJ=np.array(δJs),
        delta_invJ=np.array(δinvJ),
        meanF=np.array(F_ensemble_out),
        dFs=np.array(dFs),
        F_ensemble=np.array(allFs),
        norm_factor=norm_factor,
        ensemble_weights=weights,
        eta_ensemble=np.array(ys),
        Jbar_ensemble=np.array(dys),
        use_whitening=use_whitening,
        nn_inv=nn_inv,
        forward_backward_mlp=np.array(forward_backward_mlp),
        fisher_to_flatten=np.array(_fisher_mode),
        best_ensemble_member_index=np.array(best_idx),
        flattener_activation=np.array(_flattener_act_name),
    )
    if use_whitening:
        output_dict["W"] = np.array(W)
        output_dict["W_inv"] = np.array(W_inv)
        output_dict["F_mean"] = np.array(F_mean)

    np.savez(outname, **dict(output_dict))
    print("EXPERIMENT COMPLETED & RESULTS SAVED TO:", outname + ".npz")

    # ---------------------- WRITE SNAPSHOT METADATA -----------------------
    if enable_snapshots:
        _write_metadata(
            snapshots_outdir,
            save_every=save_every,
            burn_in=burn_in,
            save_initial=save_initial,
            grid_num_pts=grid_num_pts,
            n_params=n_params,
            param_names=param_names,
            saved_epochs=saved_epochs,
            final_global_epoch=final_global_epoch,
            saved_frob_scores=saved_frob_scores,
            notes=(
                f"phase 1 ran {epochs_done_p1} epochs; phase 2 ran "
                f"{epochs_done_p2} epochs. Snapshots are from the main "
                f"`w` (phase 1 + phase 2 combined); ensemble fine-tuning "
                f"is intentionally not snapshotted."
            ),
        )
        print(
            f"SNAPSHOTS: wrote {len(saved_epochs)} epoch snapshots "
            f"to {snapshots_outdir}"
        )

    if return_model:
        if save_flatten_model_pickle:
            import cloudpickle as pickle  # noqa: F401

            flatten_model_path = outname + "_flatten_model.pkl"
            with open(flatten_model_path, "wb") as f:
                pickle.dump(
                    {
                        "flatten_model": model,
                        "w": w,
                        "ensemble_ws": ensemble_ws,
                        "output_dict": output_dict,
                    },
                    f,
                )
            print("Saved flattener module + weights to:", flatten_model_path)
        return w, ensemble_ws, output_dict, model
    return w, ensemble_ws, output_dict
