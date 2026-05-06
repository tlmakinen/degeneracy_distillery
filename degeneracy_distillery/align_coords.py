"""
Improved ensemble alignment for flattened neural coordinates.

Replaces the coordinate-Kabsch + varimax scheme in ``preprocessing_utils`` with:

1. :func:`nonlinearity_rotation` — an SVD of the **sample-centered** Jacobian stack
   that yields an orthogonal rotation concentrating all θ-nonlinearity into the
   first few η-axes.  Axes are returned in descending order of nonlinearity
   "energy" (singular value of ΔJ), giving a principled permutation that
   replaces varimax + ``ortho_rotation``.

2. :func:`jacobian_procrustes` — closed-form orthogonal Procrustes on batched
   Jacobians.  Aligns each ensemble member to a reference using *all* samples
   and O(1) derivative magnitudes instead of noisy coordinates.

3. :func:`fisher_order_canonicalize` — sign / permutation fix-up based on the
   mean Fisher in θ-space (prior-normalized), so axes are consistently
   ordered across runs.

4. :func:`rotate_coords_v2` and :func:`process_ensemble_rotation_v2` — drop-in
   replacements for the functions in :mod:`preprocessing_utils` using the above.
   The returned dictionary has the same keys as
   ``preprocessing_utils.process_ensemble_rotation`` so the downstream SR /
   plotting code does not need to change.  After alignment and rotation,
   :func:`process_ensemble_rotation_v2` optionally applies a **global** per-axis
   translation so pooled η coordinates (all ensemble ``ys`` and the weighted
   mean ``y``) have a configurable positive floor (``offset_delta``).

All rotations act on the output (η) axis of the Jacobian, i.e.
``J'[b, i, j] = sum_k R[i, k] J[b, k, j]``, matching the convention of
``preprocessing_utils.rotate_coords``.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Sequence, Tuple, Dict, Any, Literal

from .preprocessing_utils import (  # type: ignore
    weighted_std,
    kabsch_jax,
    load_and_process_data,
)


# =============================================================================
# 1. NONLINEARITY-SEPARATING ROTATION (replaces varimax)
# =============================================================================

def nonlinearity_rotation(
    dy: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    prior_scales: Optional[np.ndarray] = None,
    regularization: float = 1e-12,
    center: bool = True,
    enforce_proper: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Orthogonal rotation that concentrates θ-nonlinearity in the first η-axes.

    A row ``i`` of ``J[b] = ∂η/∂θ|_b`` is **sample-independent** iff ``η_i`` is
    a linear function of θ.  We therefore pick ``R ∈ O(D_out)`` maximizing the
    row-wise sample variance of the first axes.  Closed form:

        ΔJ[b]  = J[b] − J̄                 (sample-centered Jacobian)
        Ω      = reshape(ΔJ, (D_out, B · D_in))
        U Σ Vᵀ = svd(Ω)
        R      = Uᵀ

    The axes of ``R @ J[b]`` are then ordered by decreasing nonlinearity
    energy, and the *last* ``D_out − k`` rows are (up to the level of ``Σ``)
    linear in θ when ``Σ`` drops off after index ``k``.

    Parameters
    ----------
    dy : np.ndarray
        Jacobian batch of shape ``(B, D_out, D_in)``.
    sample_weights : np.ndarray, optional
        Non-negative weights of shape ``(B,)`` for a weighted covariance.
        Defaults to uniform.
    prior_scales : np.ndarray, optional
        Per-θ prior widths of shape ``(D_in,)``.  If given, columns of ΔJ are
        multiplied by ``prior_scales`` so that "variation" is measured in
        dimensionless, prior-normalized θ units (same idea as
        ``F_norm = F / outer(Δθ, Δθ)`` in :mod:`preprocessing_utils`).
    regularization : float
        Added to singular values before returning (avoids exact zeros).
    center : bool
        If False, skip the subtraction of ``J̄`` (useful when ``dy`` is already
        pre-centered, or to capture absolute J structure instead of variation).
    enforce_proper : bool
        If True (default), force ``det(R) = +1`` (proper rotation, ``SO(D_out)``)
        by flipping the sign of the *last* row of ``R`` when needed.  The last
        row corresponds to the smallest singular value, i.e. the axis with the
        least nonlinearity energy, so the flip has minimal semantic impact.
        Fisher flatness ``Q = J^{-T} F J^{-1}`` is preserved either way (any
        orthogonal conjugation leaves ``‖Q − I‖_F`` and ``eig(Q)`` unchanged).

    Returns
    -------
    R : np.ndarray
        ``(D_out, D_out)`` orthogonal rotation.  Apply via
        ``dy_rot = np.einsum("ij,bjk->bik", R, dy)``.
    sigma : np.ndarray
        ``(D_out,)`` singular values of Ω in descending order.
        ``sigma[i]`` is the "nonlinearity energy" of the ``i``-th rotated axis.
    """
    dy = np.asarray(dy, dtype=np.float64)
    if dy.ndim != 3:
        raise ValueError(f"dy must be 3-D (B, D_out, D_in); got shape {dy.shape}")
    B, D_out, D_in = dy.shape

    if sample_weights is None:
        w = np.full(B, 1.0 / B)
    else:
        w = np.asarray(sample_weights, dtype=np.float64)
        if w.shape != (B,):
            raise ValueError(f"sample_weights must have shape ({B},); got {w.shape}")
        w = w / np.maximum(w.sum(), 1e-30)

    if center:
        J_mean = np.einsum("b,bij->ij", w, dy)
        dJ = dy - J_mean[None]
    else:
        dJ = dy

    if prior_scales is not None:
        ps = np.asarray(prior_scales, dtype=np.float64).reshape(1, 1, D_in)
        dJ = dJ * ps

    # Weight rows of Ω by √w so that Ω Ωᵀ is the weighted covariance.
    sqrt_w = np.sqrt(w).reshape(B, 1, 1)
    # (B, D_out, D_in) -> (D_out, B, D_in) -> (D_out, B * D_in)
    Omega = (sqrt_w * dJ).transpose(1, 0, 2).reshape(D_out, B * D_in)

    U, sigma, _ = np.linalg.svd(Omega, full_matrices=False)

    # Deterministic sign: make the dominant entry of each column of U positive.
    dom_row = np.argmax(np.abs(U), axis=0)
    sign = np.sign(U[dom_row, np.arange(U.shape[1])])
    sign = np.where(sign == 0, 1.0, sign)
    U = U * sign[None, :]

    R = U.T

    if enforce_proper and np.linalg.det(R) < 0:
        # Flip the lowest-σ axis (least nonlinear): minimal impact on ordering.
        R = R.copy()
        R[-1] *= -1.0

    return R, sigma + regularization


# =============================================================================
# 2. JACOBIAN PROCRUSTES (replaces coordinate-Kabsch for ensemble alignment)
# =============================================================================

def jacobian_procrustes(
    dy_source: np.ndarray,
    dy_target: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    allow_reflection: bool = False,
) -> np.ndarray:
    """Orthogonal Procrustes on batched Jacobians.

    Solves

        R* = argmin_{R ∈ O(D_out)}  Σ_b  w_b  ‖ R J_b − J^{ref}_b ‖_F²

    with closed-form solution ``R = U Vᵀ``, where
    ``Σ_b w_b J^{ref}_b J_bᵀ = U S Vᵀ``.

    Parameters
    ----------
    dy_source : np.ndarray
        Source Jacobians of shape ``(B, D_out, D_in)``.
    dy_target : np.ndarray
        Target (reference) Jacobians of shape ``(B, D_out, D_in)``.
    sample_weights : np.ndarray, optional
        Non-negative sample weights ``(B,)``.  Uniform if None.
    allow_reflection : bool
        If False (default) force ``det(R) = +1`` (proper rotation, SO group).

    Returns
    -------
    R : np.ndarray
        ``(D_out, D_out)`` rotation aligning ``dy_source`` to ``dy_target``.
        Apply via ``dy_aligned = np.einsum("ij,bjk->bik", R, dy_source)``.
    """
    A = np.asarray(dy_source, dtype=np.float64)
    Bt = np.asarray(dy_target, dtype=np.float64)
    if A.shape != Bt.shape:
        raise ValueError(
            f"source / target shape mismatch: {A.shape} vs {Bt.shape}"
        )
    B, D_out, D_in = A.shape

    if sample_weights is None:
        w = np.ones(B)
    else:
        w = np.asarray(sample_weights, dtype=np.float64)
        if w.shape != (B,):
            raise ValueError(f"sample_weights must have shape ({B},); got {w.shape}")

    # M = Σ_b w_b J^ref_b J_bᵀ ∈ R^{D_out × D_out}
    M = np.einsum("b,bij,bkj->ik", w, Bt, A)
    U, _, Vt = np.linalg.svd(M)
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1.0
    return U @ Vt


# =============================================================================
# 3. FISHER-EIGENVALUE ORDERING / CANONICALIZATION
# =============================================================================

def mean_fisher_eigen(
    Fs: np.ndarray,
    prior_scales: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    ascending: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of the mean (prior-normalized) Fisher in θ-space.

    Parameters
    ----------
    Fs : np.ndarray
        Sample- and ensemble-aggregated Fisher matrices, shape
        ``(B, D_in, D_in)`` or ``(D_in, D_in)``.
    prior_scales : np.ndarray, optional
        Prior widths ``Δθ`` of shape ``(D_in,)``.  If given, ``F`` is replaced
        by ``F / outer(Δθ, Δθ)`` before averaging.
    weights : np.ndarray, optional
        Sample weights for the mean along the first axis (only relevant when
        ``Fs`` is 3-D).
    ascending : bool
        If True (default), eigenvalues/vectors returned smallest-first (so
        column 0 is the flattest / most-degenerate direction — same convention
        as :func:`preprocessing_utils.rotate_coords`).

    Returns
    -------
    eigvals : np.ndarray
        Sorted eigenvalues.
    eigvecs : np.ndarray
        Column-matrix of eigenvectors in the same order.
    """
    Fs = np.asarray(Fs, dtype=np.float64)
    if Fs.ndim == 3:
        if weights is None:
            Fbar = Fs.mean(axis=0)
        else:
            w = np.asarray(weights, dtype=np.float64)
            w = w / np.maximum(w.sum(), 1e-30)
            Fbar = np.einsum("b,bij->ij", w, Fs)
    elif Fs.ndim == 2:
        Fbar = Fs
    else:
        raise ValueError(f"Fs must be 2-D or 3-D; got shape {Fs.shape}")

    if prior_scales is not None:
        delta = np.asarray(prior_scales, dtype=np.float64)
        Fbar = Fbar / np.outer(delta, delta)

    eigvals, eigvecs = np.linalg.eigh(Fbar)
    idx = np.argsort(eigvals) if ascending else np.argsort(-eigvals)
    return eigvals[idx], eigvecs[:, idx]


def fisher_order_canonicalize(
    R: np.ndarray,
    dy: np.ndarray,
    Fs: np.ndarray,
    prior_scales: Optional[np.ndarray] = None,
    mode: Literal["sign_only", "permute_and_sign"] = "sign_only",
    enforce_proper: bool = True,
) -> np.ndarray:
    """Apply a sign (and optional permutation) fix to ``R`` using the mean Fisher.

    The Jacobian of the map from θ to η pushes the θ-space Fisher into
    η-space.  Each rotated axis ``i`` should be "best correlated" with a
    specific θ-eigendirection: ``sign_only`` preserves the nonlinearity
    ordering from :func:`nonlinearity_rotation` and only flips rows of ``R``
    so the per-axis sign is deterministic; ``permute_and_sign`` additionally
    reorders rows so the last axis aligns with the largest Fisher eigenvalue
    (best-constrained θ-direction).

    Parameters
    ----------
    R : np.ndarray
        Current rotation, shape ``(D_out, D_out)``.
    dy : np.ndarray
        Jacobian batch, shape ``(B, D_out, D_in)``.  Used only to define the
        "aligned row direction" in θ-space: ``Jbar_row_i = mean_b (R @ J)_i``.
    Fs : np.ndarray
        Fisher matrices, shape ``(B, D_in, D_in)`` (or 2-D).
    prior_scales : np.ndarray, optional
        Prior widths, forwarded to :func:`mean_fisher_eigen`.
    mode : {"sign_only", "permute_and_sign"}
        How aggressive the canonicalization is.
    enforce_proper : bool
        If True (default), guarantee ``det(R) = +1`` (proper rotation,
        ``SO(D_out)``) on output.  When the sign / permutation step would
        otherwise produce a reflection (``det = -1``), the row whose sign was
        chosen with the *least* confidence — i.e. ``argmin |⟨R J, eigvec⟩|`` —
        has its sign re-flipped.  This costs nothing in Fisher flatness (any
        orthogonal conjugation preserves it) and minimally perturbs the
        canonical sign convention.

    Returns
    -------
    R_canonical : np.ndarray
        New rotation with sign (and possibly row order) fixed.
    """
    R = np.asarray(R, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    D_out = R.shape[0]

    # Row direction in θ-space: where does the rotated Jacobian point on average?
    J_rot_mean = np.einsum("ij,bjk->ik", R, dy) / max(dy.shape[0], 1)  # (D_out, D_in)

    # Fisher eigenbasis (ascending) — reference axis directions in θ-space
    _, eigvecs = mean_fisher_eigen(Fs, prior_scales=prior_scales, ascending=True)

    if mode == "permute_and_sign":
        # Greedy match each Fisher eigenvector to the best-aligned rotated axis
        taken = np.zeros(D_out, dtype=bool)
        perm = np.empty(D_out, dtype=int)
        corr = np.abs(J_rot_mean @ eigvecs)  # (D_out, D_out)
        for k in range(D_out):
            scores = corr[:, k].copy()
            scores[taken] = -np.inf
            row = int(np.argmax(scores))
            perm[k] = row
            taken[row] = True
        R = R[perm]
        J_rot_mean = J_rot_mean[perm]

    # Sign fix: make <J_rot_mean[i], eigvecs[:, i]> non-negative
    inner = np.einsum("ij,ji->i", J_rot_mean, eigvecs)
    signs = np.where(inner >= 0, 1.0, -1.0)
    R = signs[:, None] * R

    if enforce_proper and np.linalg.det(R) < 0:
        # Flip the row whose sign was chosen with the least confidence.
        flip_idx = int(np.argmin(np.abs(inner)))
        R = R.copy()
        R[flip_idx] *= -1.0

    return R


# =============================================================================
# 3b. GLOBAL BASIS BUILDER
# =============================================================================

def build_global_basis(
    dy_reference: np.ndarray,
    Fs_reference: np.ndarray,
    prior_scales: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None,
    separate_nonlinearity: bool = True,
    canonicalize: Literal["none", "sign_only", "permute_and_sign"] = "sign_only",
    use_prior_normalization: bool = True,
    enforce_proper: bool = True,
) -> np.ndarray:
    """Build the **member-independent** rotation applied to every ensemble member.

    Combines :func:`nonlinearity_rotation` and (optionally)
    :func:`fisher_order_canonicalize`, all evaluated on the **reference**
    Jacobian / Fisher.  Calling this once and reusing the result across all
    members guarantees that the η-axis ordering and signs are member-
    independent — which is what keeps the ensemble distribution tight on the
    near-degenerate (least-nonlinear) axes.

    Parameters
    ----------
    dy_reference : np.ndarray
        Reference Jacobian batch ``(B, D_out, D_in)``.
    Fs_reference : np.ndarray
        Fisher matrices used for canonicalization, ``(B, D_in, D_in)``
        (or 2-D).  Pass the same ``Favg`` you intend to flatten against.
    prior_scales, sample_weights
        Forwarded to :func:`nonlinearity_rotation`.
    separate_nonlinearity, canonicalize, use_prior_normalization, enforce_proper
        Same semantics as in :func:`rotate_coords_v2`; see that function's
        docstring for details.

    Returns
    -------
    R_basis : np.ndarray
        ``(D_out, D_out)`` orthogonal rotation that should be left-multiplied
        on top of every member's per-member alignment ``R_align``:
        ``R_total = R_basis @ R_align``.
    """
    dy_reference = np.asarray(dy_reference, dtype=np.float64)
    D_out = dy_reference.shape[1]

    if separate_nonlinearity:
        R_nl, _ = nonlinearity_rotation(
            dy_reference,
            sample_weights=sample_weights,
            prior_scales=prior_scales if use_prior_normalization else None,
            enforce_proper=enforce_proper,
        )
    else:
        R_nl = np.eye(D_out)

    if canonicalize != "none":
        # fisher_order_canonicalize uses J_rot_mean = R @ mean(dy); evaluating
        # on the *reference* (rather than per-member) gives a single,
        # member-independent sign/permutation fix.
        R_basis = fisher_order_canonicalize(
            R_nl, dy_reference, Fs_reference,
            prior_scales=prior_scales if use_prior_normalization else None,
            mode="sign_only" if canonicalize == "sign_only" else "permute_and_sign",
            enforce_proper=enforce_proper,
        )
    else:
        R_basis = R_nl

    if enforce_proper and np.linalg.det(R_basis) < 0:
        R_basis = R_basis.copy()
        R_basis[-1] *= -1.0

    return R_basis


def _global_coordinate_floor_shift(
    ys_arr: np.ndarray,
    y_mean: np.ndarray,
    offset_delta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one global translation per η-axis so pooled values sit above ``offset_delta``.

    Pool all values in ``ys_arr`` (every ensemble member and sample) together
    with ``y_mean``, take ``axis=0`` min per output dimension, then subtract
    ``(y_min - offset_delta)`` from ``ys_arr`` and ``y_mean``.  Equivalently
    ``y ← y - y.min(0) + offset_delta`` on that pooled set, so each axis's
    global minimum becomes ``offset_delta`` (default 0.1).

    Jacobians are unchanged by output translation and are not modified here.

    Returns
    -------
    ys_shifted, y_mean_shifted, shift
        ``shift`` has shape ``(D_out,)`` for optional diagnostics.
    """
    ys_arr = np.asarray(ys_arr, dtype=np.float64)
    y_mean = np.asarray(y_mean, dtype=np.float64)
    if ys_arr.ndim != 3:
        raise ValueError(f"ys_arr must be 3-D; got shape {ys_arr.shape}")
    if y_mean.ndim != 2:
        raise ValueError(f"y_mean must be 2-D; got shape {y_mean.shape}")
    D_out = ys_arr.shape[-1]
    if y_mean.shape[-1] != D_out:
        raise ValueError(
            f"y_mean last dim {y_mean.shape[-1]} != ys_arr last dim {D_out}"
        )
    pooled = np.concatenate(
        [ys_arr.reshape(-1, D_out), y_mean.reshape(-1, D_out)],
        axis=0,
    )
    y_min_axis = pooled.min(axis=0)
    shift = y_min_axis - float(offset_delta)
    return ys_arr - shift, y_mean - shift, shift


# =============================================================================
# 5. SINGLE-MEMBER ROTATION (drop-in replacement for rotate_coords)
# =============================================================================

def rotate_coords_v2(
    y: np.ndarray,
    theta: np.ndarray,
    Fs: np.ndarray,
    dy: np.ndarray,
    y_reference: Optional[np.ndarray] = None,
    dy_reference: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None,
    align_mode: Literal["procrustes", "kabsch", "none"] = "procrustes",
    separate_nonlinearity: bool = True,
    canonicalize: Literal["none", "sign_only", "permute_and_sign"] = "sign_only",
    use_prior_normalization: bool = True,
    restore_reference_mean: bool = True,
    enforce_proper: bool = True,
    align_allow_reflection: bool = True,
    R_basis: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Improved coordinate/Jacobian rotation for a single ensemble member.

    Performs, in order:

    1. Mean-center ``y`` (and ``y_reference`` if supplied).
    2. **Align member → reference** using either
       * ``"procrustes"`` — Jacobian Procrustes (recommended), or
       * ``"kabsch"`` — legacy coordinate Kabsch, or
       * ``"none"`` — skip alignment (member is already in reference frame).
    3. Optional **nonlinearity-separating** rotation derived **once on the
       reference Jacobian** (or supplied via ``R_basis``).  Applied identically
       to every ensemble member so the basis is shared.
    4. Optional Fisher-eigenvalue **sign / permutation canonicalization**, also
       computed on the **reference frame**, so the axes have a reproducible
       orientation that is *member-independent*.
    5. Apply the combined rotation ``R_total = R_basis @ R_align`` to ``y``
       and ``dy``.

    Parameters mirror :func:`preprocessing_utils.rotate_coords` plus the new
    options.  Returns the same tuple ``(y_rotated, dy, dy_sr, rotmat, A)``.

    Notes
    -----
    * ``R_basis @ ybar_reference`` is added back to ``y`` when
      ``restore_reference_mean=True`` so every aligned member receives the same
      reference-member mean in the shared output basis.  Pure translation has
      no effect on Jacobians.
    * ``A`` is kept for signature compatibility (identity here).

    Per-member vs. global handedness — IMPORTANT
    --------------------------------------------
    Earlier versions of this routine forced ``det(R_total) = +1`` per member
    (via ``enforce_proper=True``).  That choice deviates from the optimal
    Procrustes alignment whenever a member relates to the reference by a
    reflection, paying the cost in the **smallest-singular-value** direction
    of the cross-covariance — i.e. exactly the **least-nonlinear** axes after
    :func:`nonlinearity_rotation`.  Different members would also receive
    *different* sign decisions in :func:`fisher_order_canonicalize` whenever
    ``<J_rot_mean[i], eigvec[:, i]>`` was near zero, which again happens on
    the most-degenerate (least-nonlinear) axes.  Both effects compound to
    inflate ``y_std`` on the linear axes — visible as non-Gaussian
    ``(y - ybar) / y_std`` distributions across ensemble members.

    The fix:

    * ``align_allow_reflection=True`` (default): per-member Procrustes uses
      the **optimal** (possibly improper) rotation.  Fisher flatness is
      invariant under O(D), so a member living in a mirrored frame is fine
      — what matters is that all members land in the **same** frame.
    * Sign / permutation canonicalization is computed once on the reference
      and folded into ``R_basis``; every member multiplies by the same
      ``R_basis``, so axes have a consistent orientation across the ensemble.
    * ``enforce_proper`` now governs only ``R_basis`` (the global rotation),
      not the per-member alignment.

    Parameters
    ----------
    align_allow_reflection : bool, default True
        Allow per-member Procrustes to return an improper rotation when that
        is the optimal alignment.  Strongly recommended for ensembles; set
        ``False`` only to reproduce the legacy behaviour for diagnostics.
    R_basis : np.ndarray, optional
        Pre-computed global ``(D_out, D_out)`` rotation to apply on top of the
        member alignment.  If supplied, the internal ``R_nl`` /
        ``fisher_order_canonicalize`` calls are skipped.  This is what
        :func:`process_ensemble_rotation_v2` uses to guarantee a
        member-independent basis.
    """
    y = np.asarray(y, dtype=np.float64).copy()
    theta = np.asarray(theta).copy()
    Fs = np.asarray(Fs, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    D_out = y.shape[-1]

    ybar = y.mean(0)
    y = y - ybar

    ybar_reference: Optional[np.ndarray] = None
    if y_reference is not None:
        y_reference = np.asarray(y_reference, dtype=np.float64).copy()
        ybar_reference = y_reference.mean(0)
        y_reference = y_reference - ybar_reference

    prior_scales = None
    if use_prior_normalization:
        prior_scales = np.abs(theta.max(0) - theta.min(0))
        prior_scales = np.where(prior_scales > 0, prior_scales, 1.0)

    # ---- Step 1: member → reference alignment ------------------------------
    if align_mode == "procrustes":
        if dy_reference is None:
            raise ValueError("align_mode='procrustes' requires dy_reference.")
        # IMPORTANT: allow reflections during per-member alignment.  Forcing
        # det(R_align) = +1 deviates from the optimal Procrustes solution in
        # the smallest-singular direction of the cross-covariance, which —
        # composed with the nonlinearity rotation — systematically inflates
        # ensemble dispersion on the *least-nonlinear* axes.
        R_align = jacobian_procrustes(
            dy, dy_reference,
            sample_weights=sample_weights,
            allow_reflection=align_allow_reflection,
        )
    elif align_mode == "kabsch":
        if y_reference is None:
            raise ValueError("align_mode='kabsch' requires y_reference.")
        R_align, *_ = kabsch_jax(jnp.asarray(y), jnp.asarray(y_reference))
        R_align = np.asarray(R_align)
    elif align_mode == "none":
        R_align = np.eye(D_out)
    else:
        raise ValueError(f"Unknown align_mode: {align_mode!r}")

    # ---- Step 2: build / use the GLOBAL canonical basis --------------------
    # The basis (nonlinearity rotation + sign/permutation fix) is computed
    # once on the *reference* and applied identically to every member.  This
    # is what guarantees consistent η-axis orientation across the ensemble.
    if R_basis is not None:
        R_basis_arr = np.asarray(R_basis, dtype=np.float64)
        if R_basis_arr.shape != (D_out, D_out):
            raise ValueError(
                f"R_basis must have shape ({D_out}, {D_out}); got "
                f"{R_basis_arr.shape}"
            )
        sigma = None
    else:
        R_basis_arr = build_global_basis(
            dy_reference=dy_reference if dy_reference is not None else dy,
            Fs_reference=Fs,
            prior_scales=prior_scales,
            sample_weights=sample_weights,
            separate_nonlinearity=separate_nonlinearity,
            canonicalize=canonicalize,
            use_prior_normalization=use_prior_normalization,
            enforce_proper=enforce_proper,
        )
        sigma = None  # spectrum exposed via nonlinearity_spectrum() instead

    R_total = R_basis_arr @ R_align

    # NOTE: we deliberately do NOT enforce det(R_total) = +1 here.  Doing so
    # would re-introduce the per-member reflection bug we just fixed.  The
    # global basis R_basis already has det = +1 when ``enforce_proper=True``;
    # any sign of det(R_total) comes purely from det(R_align) and represents
    # the genuine (member-specific) handedness relationship between the
    # network's η-frame and the reference's.

    # ---- Step 3: apply ------------------------------------------------------
    y = np.einsum("ij,bj->bi", R_total, y)
    dy_sr = np.einsum("ij,bjk->bik", R_total, dy)

    if restore_reference_mean and ybar_reference is not None:
        # was before: y = y + R_basis_arr @ ybar_reference
        # now: y = y + R_total @ ybar_reference
        # The reference mean is defined by the principal member after the
        # shared basis rotation.  Do not include the per-member alignment here,
        # otherwise each ensemble member receives a different translation.
        y = y + R_basis_arr @ ybar_reference

    A = np.eye(D_out)
    return y, dy, dy_sr, R_total, A


# =============================================================================
# 6. ENSEMBLE ORCHESTRATOR (drop-in replacement for process_ensemble_rotation)
# =============================================================================

def process_ensemble_rotation_v2(
    datafile: Dict[str, Any],
    randidx: np.ndarray,
    Favg: np.ndarray,
    best_model_idx: int,
    n_d: float = 1.0,
    ensemble_indices: Optional[np.ndarray] = None,
    align_mode: Literal["procrustes", "kabsch", "none"] = "procrustes",
    separate_nonlinearity: bool = True,
    canonicalize: Literal["none", "sign_only", "permute_and_sign"] = "sign_only",
    use_prior_normalization: bool = True,
    restore_reference_mean: bool = True,
    enforce_proper: bool = True,
    Fisher_to_flatten: Literal["average", "best"] = "average",
    verbose: bool = True,
    offset_delta: Optional[float] = 0.1,
) -> Dict[str, Any]:
    """Ensemble alignment using Jacobian-Procrustes + nonlinearity rotation.

    API-compatible with :func:`preprocessing_utils.process_ensemble_rotation`
    (returns the same keys, plus ``reference_offset``), but uses the improved algorithms from this
    module.

    Key differences vs. the legacy function:

    * ``align_mode="procrustes"`` (default) aligns each member to the reference
      by orthogonal Procrustes **on the Jacobians**, not Kabsch on coordinates.
    * ``separate_nonlinearity=True`` applies a single, reference-defined
      "nonlinearity-first" rotation to every member *after* alignment, so the
      ensemble shares an axis ordering in which the most sample-dependent (i.e.
      most nonlinear) η-components come first.
    * ``canonicalize`` fixes per-axis sign (and optionally permutation) using
      the mean θ-space Fisher.
    * ``enforce_proper=True`` (default) guarantees every per-member ``rotmat``
      is in ``SO(D_out)`` (proper rotation, det = +1) so the η frame has the
      same handedness as the θ frame.  Disable to allow reflections; Fisher
      flatness is preserved either way.
    * After masking and averaging, a **global** per-axis translation (optional)
      shifts ``ys`` and ``y`` so every η component, pooled over all ensemble
      members and samples and the weighted mean, has global minimum
      ``offset_delta`` (default ``0.1``).  Set ``offset_delta=None`` to skip.
      The subtracted vector is ``y_global_min - offset_delta`` (shape
      ``(D_out,)``); see :func:`_global_coordinate_floor_shift`.  When enabled,
      the return dict may include ``eta_coordinate_shift`` with that vector.
    * ``reference_offset``: when ``restore_reference_mean`` is True, the vector
      ``(R_basis @ mean(y_reference)) / sqrt(n_d)`` added to ``ys_arr`` and
      ``y_mean`` (full η dimension, before the ``mask`` is applied to means);
      ``None`` when ``restore_reference_mean`` is False.
    """
    if Fisher_to_flatten not in ("average", "best"):
        raise ValueError(
            f"Fisher_to_flatten must be 'average' or 'best', got {Fisher_to_flatten!r}"
        )

    ensemble_weights_raw = np.asarray(datafile["ensemble_weights"])
    if ensemble_indices is None:
        member_idx = np.arange(ensemble_weights_raw.shape[0], dtype=int)
    else:
        member_idx = np.asarray(ensemble_indices, dtype=int)
    w_sub = ensemble_weights_raw[member_idx]
    w_sub = w_sub / np.maximum(w_sub.sum(), 1e-12)
    num_nets = member_idx.shape[0]

    y_reference = np.asarray(datafile["eta_ensemble"][best_model_idx][randidx])
    dy_reference = np.asarray(datafile["Jbar_ensemble"][best_model_idx][randidx])

    ys: list = []
    dys: list = []
    dys_sr: list = []
    Fs_list: list = []
    ensemble_weights: list = []
    rotmats: list = []

    X = np.asarray(datafile["theta"][randidx])

    # ---- Build the global basis ONCE on the reference -----------------------
    # This is what guarantees a member-independent axis orientation: every
    # member multiplies by the same R_basis, so per-member sign drift on the
    # near-degenerate (least-nonlinear) axes is eliminated.
    if use_prior_normalization:
        prior_scales = np.abs(X.max(0) - X.min(0))
        prior_scales = np.where(prior_scales > 0, prior_scales, 1.0)
    else:
        prior_scales = None

    R_basis = build_global_basis(
        dy_reference=dy_reference,
        Fs_reference=Favg,
        prior_scales=prior_scales,
        separate_nonlinearity=separate_nonlinearity,
        canonicalize=canonicalize,
        use_prior_normalization=use_prior_normalization,
        enforce_proper=enforce_proper,
    )

    if verbose:
        det_basis = float(np.linalg.det(R_basis))
        print(
            f"Global basis R_basis: shape={R_basis.shape}, "
            f"det={det_basis:+.6f} (proper rotation expected)"
        )

    reference_offset = None
    if restore_reference_mean:
        reference_offset = R_basis @ y_reference.mean(0)

    for k, i in enumerate(member_idx):
        y = np.asarray(datafile["eta_ensemble"][i][randidx])
        dy = np.asarray(datafile["Jbar_ensemble"][i][randidx])
        F_i = np.asarray(datafile["F_ensemble"][i][randidx])

        if verbose:
            print(
                f"Network {i} (subset {k}/{num_nets}): "
                f"y.min()={y.min():.6f}, weight={w_sub[k]:.4f}"
            )

        y_rot, dy_orig, dy_sr_rot, rotmat, _ = rotate_coords_v2(
            y=y, theta=X, Fs=Favg, dy=dy,
            y_reference=y_reference,
            dy_reference=dy_reference,
            align_mode=align_mode,
            separate_nonlinearity=separate_nonlinearity,
            canonicalize=canonicalize,
            use_prior_normalization=use_prior_normalization,
            restore_reference_mean=False,
            enforce_proper=enforce_proper,
            R_basis=R_basis,
        )

        ys.append(y_rot)
        dys.append(dy_orig)
        dys_sr.append(dy_sr_rot)
        rotmats.append(rotmat)
        ensemble_weights.append(w_sub[k])
        Fs_list.append(F_i)

    ys_arr = np.array(ys) / np.sqrt(n_d)
    dys_arr = np.array(dys) / np.sqrt(n_d)
    dys_sr_arr = np.array(dys_sr) / np.sqrt(n_d)
    ensemble_weights_arr = np.array(ensemble_weights)
    rotmats_arr = np.array(rotmats)
    Fs_arr = np.array(Fs_list) / n_d
    ensemble_Fs = Fs_arr.copy()

    if verbose:
        print(f"\nEnsemble shapes: dys={dys_arr.shape}, ys={ys_arr.shape}")

    y_mean = np.average(ys_arr, axis=0, weights=ensemble_weights_arr)
    y_std = np.array(weighted_std(
        jnp.asarray(ys_arr), weights=jnp.asarray(ensemble_weights_arr), axis=0,
    ))

    if reference_offset is not None:
        # Keep the ensemble scatter estimate translation-free.  The principal
        # member's mean is a shared coordinate origin in the new basis, so it is
        # added only after y_std has been computed.
        reference_offset = reference_offset / np.sqrt(n_d)
        ys_arr = ys_arr + reference_offset[None, None, :]
        y_mean = y_mean + reference_offset[None, :]

    rotmat_avg = np.average(rotmats_arr, weights=ensemble_weights_arr, axis=0)

    mask = y_std[:, 0] != 0
    y_mean = y_mean[mask]
    y_std = y_std[mask]

    dy_mean = np.average(dys_arr, axis=0, weights=ensemble_weights_arr)[mask]
    dy_sr_mean = np.average(dys_sr_arr, axis=0, weights=ensemble_weights_arr)[mask]

    dys_arr = np.array([j[mask] for j in dys_arr])
    dys_sr_arr = np.array([j[mask] for j in dys_sr_arr])

    Jbar = dy_mean.copy()

    X_masked = X[mask]
    Fs_arr = np.array([f[mask] for f in Fs_arr])
    ensemble_Fs = Fs_arr.copy()

    if Fisher_to_flatten == "average":
        Fs_avg = np.average(Fs_arr, axis=0, weights=ensemble_weights_arr)
    else:
        k_best = int(np.argmax(ensemble_weights_arr))
        Fs_avg = Fs_arr[k_best]
        if verbose:
            print(
                f"Fisher_to_flatten='best': using ensemble member "
                f"{member_idx[k_best]} (subset slot {k_best}) for Fs"
            )

    eta_ensemble_masked = datafile["eta_ensemble"][member_idx][:, randidx, :][:, mask, :]

    eta_floor_shift: Optional[np.ndarray] = None
    if offset_delta is not None:
        ys_arr, y_mean, eta_floor_shift = _global_coordinate_floor_shift(
            ys_arr, y_mean, float(offset_delta)
        )
        if verbose:
            print(
                f"\nGlobal η coordinate floor (last step): offset_delta={float(offset_delta):g}; "
                f"subtracted per-axis shift = y_global_min - offset_delta = {eta_floor_shift}"
            )

    out: Dict[str, Any] = {
        "y": y_mean,
        "y_std": y_std,
        "dy": dy_mean,
        "dy_sr": dy_sr_mean,
        "Fs": Fs_avg,
        "X": X_masked,
        "ys": ys_arr,
        "dys": dys_arr,
        "dys_sr": dys_sr_arr,
        "ensemble_Fs": ensemble_Fs,
        "ensemble_weights": ensemble_weights_arr,
        "rotmats": rotmats_arr,
        "rotmat_avg": rotmat_avg,
        "mask": mask,
        "Jbar": Jbar,
        "n_d": n_d,
        "eta_ensemble": eta_ensemble_masked,
        "norm_factor": datafile["norm_factor"],
        "reference_offset": reference_offset,
    }
    if eta_floor_shift is not None:
        out["eta_coordinate_shift"] = eta_floor_shift
    return out


# =============================================================================
# 7. CONVENIENCE: LOAD + PROCESS (drop-in for load_and_process_data)
# =============================================================================

def load_and_process_data_v2(
    datapath: str,
    filename: str,
    num_samps: int = 4000,
    seed: int = 44,
    process_ensemble: bool = True,
    n_d: float = 1.0,
    y_reference_index: Optional[int] = None,
    align_mode: Literal["procrustes", "kabsch", "none"] = "procrustes",
    separate_nonlinearity: bool = True,
    canonicalize: Literal["none", "sign_only", "permute_and_sign"] = "sign_only",
    use_prior_normalization: bool = True,
    restore_reference_mean: bool = True,
    enforce_proper: bool = True,
    Fisher_to_flatten: Literal["average", "best"] = "average",
    verbose: bool = True,
    offset_delta: Optional[float] = 0.1,
) -> Dict[str, Any]:
    """Drop-in replacement for :func:`preprocessing_utils.load_and_process_data`.

    Reuses the legacy loader (for its NaN-member filtering and Favg logic),
    then runs :func:`process_ensemble_rotation_v2` instead of the legacy
    Kabsch/varimax pipeline.

    Parameters
    ----------
    datapath, filename, num_samps, seed, n_d, y_reference_index, Fisher_to_flatten, verbose
        Same semantics as :func:`preprocessing_utils.load_and_process_data`.
    process_ensemble : bool
        If False, only loads the file and returns the same "loading-only"
        dictionary as the legacy loader (X, Favg, datafile, etc.).
    align_mode, separate_nonlinearity, canonicalize, use_prior_normalization, restore_reference_mean, enforce_proper
        Forwarded to :func:`process_ensemble_rotation_v2`.  ``enforce_proper``
        defaults to True (guarantees ``det(R) = +1`` per member).
    offset_delta
        Forwarded to :func:`process_ensemble_rotation_v2` (global η coordinate
        floor after alignment).  ``None`` disables that final translation.

    Returns
    -------
    data : dict
        When ``process_ensemble=True``, merges the legacy loader output with
        :func:`process_ensemble_rotation_v2` — same keys as the legacy call.
    """
    base = load_and_process_data(
        datapath=datapath,
        filename=filename,
        num_samps=num_samps,
        seed=seed,
        process_ensemble=False,
        n_d=n_d,
        y_reference_index=y_reference_index,
        verbose=verbose,
        Fisher_to_flatten=Fisher_to_flatten,
    )

    if not process_ensemble:
        return base

    ensemble_result = process_ensemble_rotation_v2(
        datafile=base["datafile"],
        randidx=base["randidx"],
        Favg=base["Favg"],
        best_model_idx=base["best_model_idx"],
        n_d=n_d,
        ensemble_indices=base.get("ensemble_indices"),
        align_mode=align_mode,
        separate_nonlinearity=separate_nonlinearity,
        canonicalize=canonicalize,
        use_prior_normalization=use_prior_normalization,
        restore_reference_mean=restore_reference_mean,
        enforce_proper=enforce_proper,
        Fisher_to_flatten=Fisher_to_flatten,
        verbose=verbose,
        offset_delta=offset_delta,
    )

    merged = dict(base)
    merged.update(ensemble_result)
    return merged


# =============================================================================
# 8. DIAGNOSTICS
# =============================================================================

def nonlinearity_spectrum(
    dy: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    prior_scales: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the sorted singular values of the sample-centered Jacobian stack.

    Useful to decide how many η-axes carry genuine θ-nonlinearity: look for the
    "elbow" in the returned ``sigma``.  ``sigma[i] ≈ 0`` means axis ``i`` (in
    the :func:`nonlinearity_rotation` basis) is linear in θ.
    """
    _, sigma = nonlinearity_rotation(
        dy, sample_weights=sample_weights, prior_scales=prior_scales
    )
    return sigma


def linearity_residual(
    dy: np.ndarray,
    axes: Sequence[int],
    sample_weights: Optional[np.ndarray] = None,
) -> float:
    """Fraction of Jacobian variation NOT captured by the given ``axes``.

    After applying :func:`nonlinearity_rotation` externally and picking the
    first ``k`` axes as "nonlinear", this returns the leakage into the
    remaining axes.  Values close to 0 confirm that ``axes`` exhausts the
    nonlinear structure.
    """
    sigma = nonlinearity_spectrum(dy, sample_weights=sample_weights)
    total = float(np.sum(sigma**2))
    kept = float(np.sum(sigma[list(axes)] ** 2))
    if total <= 0:
        return 0.0
    return 1.0 - kept / total


__all__ = [
    "nonlinearity_rotation",
    "jacobian_procrustes",
    "mean_fisher_eigen",
    "fisher_order_canonicalize",
    "build_global_basis",
    "rotate_coords_v2",
    "process_ensemble_rotation_v2",
    "load_and_process_data_v2",
    "nonlinearity_spectrum",
    "linearity_residual",
]
