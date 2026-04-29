"""
Fisher marginalisation and parameter-of-interest selection.

Two utilities used to slim down a large parameter space (e.g. 7-parameter weak
lensing) to a few "parameters of interest" before training the flattening /
running symbolic regression:

1. :func:`schur_marginalize_fisher` — closed-form Schur complement marginalising
   over nuisance parameters, batched over arbitrary leading dimensions so it
   can be applied to a per-sample Fisher stack ``(B, P, P)`` *or* an ensemble
   ``(N, B, P, P)`` directly.

2. :func:`degeneracy_structure_scores` (and :func:`recommend_parameters_of_interest`)
   — diagnostic scores ranking each parameter by how strongly it participates
   in the **sample-dependent** part of the Fisher (the user's
   ``Fs.std(0) / Fs.mean(0)`` intuition), to guide the choice of which
   parameters to keep.

Typical workflow for the WL example
-----------------------------------

>>> from degeneracy_distillery.fisher_marginalization import (
...     schur_marginalize_fisher,
...     degeneracy_structure_scores,
...     recommend_parameters_of_interest,
... )
>>> # 1) Look at the data-driven score and confirm the cosmological choice
>>> param_names = ["Omega_m", "sigma_8", "w0", "wa", "h", "n_s", "A_IA"]
>>> scores = degeneracy_structure_scores(
...     Fs=data["ensemble_Fs"].mean(0),    # mean over ensemble: (B, 7, 7)
...     prior_scales=np.abs(theta.max(0) - theta.min(0)),
...     names=param_names,
... )
>>> print(scores["ranking_table"])
>>> # 2) Marginalise Fisher down to the parameters of interest
>>> idx_of_interest = [0, 1, 2, 3]   # Omega_m, sigma_8, w0, wa
>>> Fs_marg = schur_marginalize_fisher(data["Fs"], idx_of_interest)   # (B, 4, 4)
>>> ensemble_Fs_marg = schur_marginalize_fisher(
...     data["ensemble_Fs"], idx_of_interest                           # (N, B, 4, 4)
... )
>>> # 3) Trim θ and J to the same indices and pass downstream
>>> theta_marg = data["X"][:, idx_of_interest]
>>> J_marg     = data["dy_sr"][:, :, idx_of_interest]
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Sequence, Dict, Any, List, Literal


# =============================================================================
# 1. SCHUR-COMPLEMENT MARGINALISATION
# =============================================================================

def schur_marginalize_fisher(
    Fs: np.ndarray,
    indices_of_interest: Sequence[int],
    regularization: float = 0.0,
    method: Literal["solve", "pinv"] = "solve",
) -> np.ndarray:
    r"""Marginalise nuisance parameters out of a Fisher matrix via Schur complement.

    Partitioning ``F`` into block form

    .. math::
        F = \begin{pmatrix} F_{aa} & F_{ab} \\ F_{ba} & F_{bb} \end{pmatrix}

    where ``a`` indexes the parameters of interest and ``b`` indexes the
    nuisance parameters, the marginalised Fisher on ``a`` is

    .. math::
        F^{\text{marg}}_{aa} = F_{aa} - F_{ab}\, F_{bb}^{-1}\, F_{ba}.

    Equivalent (and how this is justified): inverting the full :math:`F` yields
    the joint covariance, restricting to the ``a`` block, and inverting that
    block back is exactly the Schur complement of :math:`F_{bb}`.  No extra
    information is lost beyond what is gained by *integrating out* the
    nuisance posterior (under a Gaussian likelihood).

    Parameters
    ----------
    Fs : np.ndarray
        Fisher matrices of shape ``(..., P, P)``.  Leading dimensions are
        treated as independent batches and preserved on output.  E.g.
        ``(B, P, P)`` for a per-sample stack, ``(N, B, P, P)`` for an
        ensemble of stacks.
    indices_of_interest : Sequence[int]
        Indices ``a`` of the parameters to keep, in the desired output order.
        ``len(indices_of_interest)`` is the dimension of the output Fisher.
        Must be a (proper) subset of ``range(P)``.
    regularization : float, default 0.0
        If ``> 0``, add ``regularization * I`` to ``F_bb`` before inverting.
        Useful when nuisance parameters are highly degenerate and ``F_bb`` is
        ill-conditioned.  The effect on the marginalised Fisher is to *under*-
        marginalise (treat very degenerate nuisance directions as effectively
        prior-fixed), which is usually the desired numerical safety net.
    method : {"solve", "pinv"}, default "solve"
        How to apply ``F_bb^{-1}`` to ``F_ba``.

        * ``"solve"`` uses :func:`numpy.linalg.solve` — fast, stable when
          ``F_bb`` is well-conditioned.  Falls back to ``"pinv"`` automatically
          if a ``LinAlgError`` is raised.
        * ``"pinv"`` uses :func:`numpy.linalg.pinv` — slower but robust to
          singular ``F_bb``.

    Returns
    -------
    F_marg : np.ndarray
        Marginalised Fisher matrices of shape ``(..., K, K)`` with
        ``K = len(indices_of_interest)``.  Symmetry is enforced after the
        Schur step (``F_marg = (F_marg + F_marg.T) / 2``) since round-off can
        otherwise leave ``F_marg`` non-symmetric at the level of ULPs.

    Notes
    -----
    Use the *same* index list to also restrict ``θ`` and ``J = ∂η/∂θ`` so the
    downstream flattening sees a consistent (a-only) parametrisation::

        Fs_marg    = schur_marginalize_fisher(Fs, idx)         # (B, K, K)
        theta_marg = theta[..., idx]                           # (B, K)
        J_marg     = J[..., idx]                               # (B, D_out, K)

    The mean-Fisher Schur complement of a *prior-normalised* Fisher
    (``F̃ = F / outer(Δθ, Δθ)``) equals ``Δθ_a Δθ_a^T`` element-wise times the
    Schur complement of ``F`` and then divided by ``outer(Δθ_a, Δθ_a)``, so
    you can apply prior normalisation either before *or* after the Schur step
    — they commute.
    """
    F = np.asarray(Fs, dtype=np.float64)
    if F.ndim < 2 or F.shape[-1] != F.shape[-2]:
        raise ValueError(
            f"Fs must end with a square (P, P) block; got shape {F.shape}"
        )
    P = F.shape[-1]

    a = np.asarray(indices_of_interest, dtype=int).ravel()
    if a.size == 0:
        raise ValueError("indices_of_interest is empty.")
    if a.min() < 0 or a.max() >= P:
        raise ValueError(
            f"indices_of_interest contain out-of-range entry: {a.tolist()} "
            f"vs. P={P}"
        )
    if np.unique(a).size != a.size:
        raise ValueError(
            f"indices_of_interest must be unique; got {a.tolist()}"
        )

    b = np.array([j for j in range(P) if j not in set(a.tolist())], dtype=int)

    F_aa = F[..., a[:, None], a[None, :]]
    if b.size == 0:
        # Nothing to marginalise — return F restricted to a (with symmetrised
        # output for consistency).
        F_aa = 0.5 * (F_aa + np.swapaxes(F_aa, -1, -2))
        return F_aa

    F_ab = F[..., a[:, None], b[None, :]]
    F_bb = F[..., b[:, None], b[None, :]]

    if regularization > 0.0:
        eye_b = np.eye(b.size)
        F_bb = F_bb + regularization * eye_b

    F_ba = np.swapaxes(F_ab, -1, -2)

    if method == "solve":
        try:
            X = np.linalg.solve(F_bb, F_ba)            # F_bb @ X = F_ba
        except np.linalg.LinAlgError:
            X = np.linalg.pinv(F_bb) @ F_ba
    elif method == "pinv":
        X = np.linalg.pinv(F_bb) @ F_ba
    else:
        raise ValueError(f"Unknown method: {method!r}")

    F_marg = F_aa - F_ab @ X
    F_marg = 0.5 * (F_marg + np.swapaxes(F_marg, -1, -2))
    return F_marg


# =============================================================================
# 2. DEGENERACY-STRUCTURE DIAGNOSTICS
# =============================================================================

def fisher_correlation(Fs: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    r"""Per-sample correlation matrix derived from the Fisher.

    .. math::
        \rho[b, i, j] = F[b, i, j] / \sqrt{F[b, i, i] \cdot F[b, j, j]}

    Diagonal is exactly 1.  Off-diagonal entries are unit-free, in
    :math:`[-1, 1]` for any positive-definite ``F[b]``, and capture the
    *degeneracy structure* (correlation pattern) independently of the
    overall constraint strength.

    Parameters
    ----------
    Fs : np.ndarray
        Fisher matrices of shape ``(..., P, P)``.
    eps : float
        Floor on the diagonal-product before taking the square root.

    Returns
    -------
    rho : np.ndarray
        Same shape as ``Fs``.
    """
    F = np.asarray(Fs, dtype=np.float64)
    diag = np.diagonal(F, axis1=-2, axis2=-1)            # (..., P)
    denom = np.sqrt(np.maximum(diag[..., :, None] * diag[..., None, :], eps))
    return F / denom


def degeneracy_structure_scores(
    Fs: np.ndarray,
    weights: Optional[np.ndarray] = None,
    prior_scales: Optional[np.ndarray] = None,
    prior_fisher: Optional[np.ndarray] = None,
    names: Optional[Sequence[str]] = None,
    eps: float = 1e-30,
) -> Dict[str, Any]:
    r"""Score each parameter by its participation in the *sample-dependent*
    part of the Fisher.

    The intuition the user proposed — ``Fs.std(0) / Fs.mean(0)`` — is the
    per-element coefficient of variation (CV) of the Fisher across the
    sample axis.  Large CV at element ``(i, j)`` means the constraint
    coupling between ``θ_i`` and ``θ_j`` *changes* with the sample, i.e. is
    nonlinear in ``θ``.  This routine returns several aggregations of that
    matrix into per-parameter scores plus a composite ranking.

    Parameters
    ----------
    Fs : np.ndarray
        Fisher matrices of shape ``(B, P, P)`` (a per-sample stack).  If you
        have an ensemble ``(N, B, P, P)``, average over members first
        (``Fs.mean(0)``) — these scores characterise the *population*
        Fisher, not the per-network noise.
    weights : np.ndarray, optional
        Sample weights ``(B,)``.  Uniform if ``None``.
    prior_scales : np.ndarray, optional
        Prior widths ``Δθ`` of shape ``(P,)``.  When supplied, the Fisher is
        prior-normalised (``F̃ = F / outer(Δθ, Δθ)``, matching the convention
        in :mod:`preprocessing_utils`) before scoring, so different
        parameters are compared on a common dimensionless footing.  If
        ``None``, raw ``F`` is used.  Also used to put ``prior_fisher`` in the
        same prior-normalised basis for ``info_strength``.  When
        ``prior_fisher`` is not supplied, the prior reference is the identity
        matrix in this basis.
    prior_fisher : np.ndarray, optional
        Prior Fisher matrix of shape ``(P, P)`` or prior Fisher stack
        ``(B, P, P)``.  When supplied, ``info_strength`` compares the mean
        Fisher row norm to the prior row norm in the same basis used for the
        structural scores.  If ``prior_scales`` is supplied, both ``Fs`` and
        ``prior_fisher`` are divided by ``outer(Δθ, Δθ)`` first; if
        ``prior_fisher`` is omitted, the prior reference is the identity in
        that dimensionless basis.  This prevents weakly constrained /
        prior-limited parameters from ranking highly just because their
        relative CV is large.
    names : Sequence[str], optional
        Parameter labels for the returned ``ranking_table`` string.
    eps : float
        Floor for the relative-CV denominator.

    Returns
    -------
    scores : dict with keys
        - ``"cv_matrix"`` : ``(P, P)`` element-wise CV
          ``std_b(F[b, i, j]) / max(|mean_b F[b, i, j]|, eps)``.  This is the
          most direct expression of the user's ``std/mean`` idea.
        - ``"diag_cv"`` : ``(P,)`` diagonal of ``cv_matrix`` — relative
          variation of each parameter's *own* Fisher information.
        - ``"row_l2_cv"`` : ``(P,)`` row L2 relative variation,
          ``sqrt(sum_j var_b(F_ij)) / sqrt(sum_j (mean_b F_ij)^2)``.  More
          robust than ``diag_cv`` because it folds in coupling variation.
        - ``"corr_var"`` : ``(P,)`` per-parameter variability of the
          Fisher *correlation* matrix
          ``mean_{j != i} std_b(rho[b, i, j])``.  Pure degeneracy-structure
          variation, free of constraint-strength units.
        - ``"eig_participation_top_k"`` : ``(P, k)`` participation of each
          parameter in the top-``k`` eigenvectors of ``mean_b F`` (where
          ``k = min(4, P)``); larger means the parameter is dominant in
          well-constrained directions.
        - ``"info_strength"`` : ``(P,)`` row-L2 information strength relative
          to the prior Fisher.  Values near 1 mean the parameter is only
          prior-level constrained; values below 1 mean weaker than the prior;
          values much larger than 1 mean data-dominated.  ``NaN`` if no prior
          reference is available.
        - ``"composite"`` : ``(P,)`` rank-averaged composite of
          ``row_l2_cv``, ``corr_var``, and, when available,
          ``info_strength`` (all descending).  Lower is better.
        - ``"ranking"`` : ``(P,)`` ``argsort`` of ``composite`` ascending —
          first entry is the *most* nonlinearly-constrained parameter.
        - ``"ranking_table"`` : pre-formatted string for printing.
    """
    F = np.asarray(Fs, dtype=np.float64)
    if F.ndim != 3 or F.shape[-1] != F.shape[-2]:
        raise ValueError(
            f"Fs must have shape (B, P, P); got {F.shape}.  Average over "
            f"any extra leading dims first."
        )
    B, P, _ = F.shape

    if weights is None:
        w = np.full(B, 1.0 / B)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (B,):
            raise ValueError(f"weights must have shape ({B},); got {w.shape}")
        w = w / np.maximum(w.sum(), eps)

    F_use = F.copy()
    d: Optional[np.ndarray] = None
    prior_norm: Optional[np.ndarray] = None
    if prior_scales is not None:
        d = np.asarray(prior_scales, dtype=np.float64).ravel()
        if d.shape != (P,):
            raise ValueError(f"prior_scales must have shape ({P},); got {d.shape}")
        prior_norm = np.outer(d, d)
        F_use = F_use / prior_norm[None, :, :]

    # Weighted mean / std along the sample axis.
    F_mean = np.einsum("b,bij->ij", w, F_use)
    F_var  = np.einsum("b,bij->ij", w, (F_use - F_mean[None]) ** 2)
    F_std  = np.sqrt(np.maximum(F_var, 0.0))

    # ---- 1. Element-wise CV (the user's std/mean idea) ----------------------
    cv_matrix = F_std / np.maximum(np.abs(F_mean), eps)

    # ---- 2. Diagonal CV ----------------------------------------------------
    diag_cv = np.diagonal(cv_matrix)

    # ---- 3. Row L2 CV ------------------------------------------------------
    row_var_l2  = np.sum(F_var, axis=1)
    row_mean_l2 = np.sum(F_mean ** 2, axis=1)
    row_l2_cv = np.sqrt(row_var_l2) / np.sqrt(np.maximum(row_mean_l2, eps))

    # ---- 4. Correlation-structure variation --------------------------------
    rho = fisher_correlation(F_use, eps=eps)              # (B, P, P)
    rho_mean = np.einsum("b,bij->ij", w, rho)
    rho_var  = np.einsum("b,bij->ij", w, (rho - rho_mean[None]) ** 2)
    rho_std  = np.sqrt(np.maximum(rho_var, 0.0))
    off_diag = ~np.eye(P, dtype=bool)
    if P > 1:
        # Average over off-diagonal columns per row.
        corr_var = (rho_std * off_diag).sum(axis=1) / max(P - 1, 1)
    else:
        corr_var = np.zeros(P)

    # ---- 5. Mean-Fisher eigenvector participation --------------------------
    # Eigenvectors of mean F descending in eigenvalue (best-constrained first)
    eigvals, eigvecs = np.linalg.eigh(F_mean)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    k_top = min(4, P)
    eig_participation = (eigvecs[:, :k_top] ** 2)        # (P, k_top)

    # ---- 6. Data/prior information strength --------------------------------
    prior_reference = None
    if prior_fisher is not None:
        PF = np.asarray(prior_fisher, dtype=np.float64)
        if PF.shape == (P, P):
            prior_reference = PF
        elif PF.shape == (B, P, P):
            prior_reference = np.einsum("b,bij->ij", w, PF)
        else:
            raise ValueError(
                f"prior_fisher must have shape ({P}, {P}) or "
                f"({B}, {P}, {P}); got {PF.shape}"
            )
        if prior_norm is not None:
            prior_reference = prior_reference / prior_norm
    elif d is not None:
        prior_reference = np.eye(P)

    if prior_reference is None:
        info_strength = np.full(P, np.nan)
        diag_info_strength = np.full(P, np.nan)
    else:
        mean_row_l2 = np.sqrt(np.sum(F_mean ** 2, axis=1))
        prior_row_l2 = np.sqrt(np.sum(prior_reference ** 2, axis=1))
        info_strength = mean_row_l2 / np.maximum(prior_row_l2, eps)

        mean_diag = np.diagonal(F_mean)
        prior_diag = np.abs(np.diagonal(prior_reference))
        diag_info_strength = mean_diag / np.maximum(prior_diag, eps)

    # ---- 7. Composite ranking (lower = more nonlinear / more interesting) --
    rank_row = (-row_l2_cv).argsort().argsort()          # high row_l2_cv -> low rank
    rank_corr = (-corr_var).argsort().argsort()
    rank_terms = [rank_row, rank_corr]
    if prior_reference is not None:
        rank_info = (-info_strength).argsort().argsort()
        rank_terms.append(rank_info)
    composite = np.mean(np.stack(rank_terms, axis=0), axis=0)
    ranking = np.argsort(composite)

    # ---- 8. Pretty table ---------------------------------------------------
    nm = list(names) if names is not None else [f"theta_{i}" for i in range(P)]
    if len(nm) != P:
        raise ValueError(f"names length {len(nm)} != P={P}")
    rows: List[str] = [
        f"{'rank':>4}  {'idx':>4}  {'name':<12s}  "
        f"{'diag_cv':>10s}  {'row_l2_cv':>10s}  {'corr_var':>10s}  "
        f"{'info':>10s}  "
        + "  ".join([f"|v_{k+1}|^2" for k in range(k_top)])
    ]
    rows.append("-" * len(rows[0]))
    for r, i in enumerate(ranking):
        evec_part = "  ".join(f"{eig_participation[i, k]:7.3f}" for k in range(k_top))
        info = "       nan" if np.isnan(info_strength[i]) else f"{info_strength[i]:>10.3e}"
        rows.append(
            f"{r:>4d}  {i:>4d}  {nm[i]:<12s}  "
            f"{diag_cv[i]:>10.3e}  {row_l2_cv[i]:>10.3e}  {corr_var[i]:>10.3e}  "
            f"{info}  "
            + evec_part
        )
    table = "\n".join(rows)

    return {
        "cv_matrix": cv_matrix,
        "diag_cv": diag_cv,
        "row_l2_cv": row_l2_cv,
        "corr_var": corr_var,
        "info_strength": info_strength,
        "diag_info_strength": diag_info_strength,
        "eig_participation_top_k": eig_participation,
        "eigvals_meanF": eigvals,
        "composite": composite,
        "ranking": ranking,
        "ranking_table": table,
        "names": nm,
        "F_mean": F_mean,
        "F_std": F_std,
    }


def recommend_parameters_of_interest(
    Fs: np.ndarray,
    k: int = 4,
    weights: Optional[np.ndarray] = None,
    prior_scales: Optional[np.ndarray] = None,
    prior_fisher: Optional[np.ndarray] = None,
    names: Optional[Sequence[str]] = None,
    return_scores: bool = False,
) -> Any:
    """Top-``k`` parameters ranked by :func:`degeneracy_structure_scores` composite.

    Parameters
    ----------
    Fs, weights, prior_scales, prior_fisher, names : see
        :func:`degeneracy_structure_scores`
    k : int, default 4
        Number of parameters to recommend.
    return_scores : bool, default False
        If True, also return the full score dictionary.

    Returns
    -------
    indices : np.ndarray, shape ``(k,)``
        Recommended indices in descending order of composite score.
    scores : dict (optional)
        Same as :func:`degeneracy_structure_scores` output.
    """
    scores = degeneracy_structure_scores(
        Fs, weights=weights, prior_scales=prior_scales,
        prior_fisher=prior_fisher, names=names
    )
    indices = np.array(scores["ranking"][:k], dtype=int)
    if return_scores:
        return indices, scores
    return indices


__all__ = [
    "schur_marginalize_fisher",
    "fisher_correlation",
    "degeneracy_structure_scores",
    "recommend_parameters_of_interest",
]
