"""
Diagnostics for low-information regions in Fisher-based degeneracy distillation.

When training a flattener with target ``F_eta = I``, the network folds
(``det J -> 0``) in regions where ``det(F_theta)`` collapses by orders of
magnitude relative to the bulk. Low-information regions can have several
distinct origins, and the right action depends on which:

    1. SNR / detection boundary
       --> trim by lambda_min percentile; report domain-of-validity.
    2. Phase boundary / critical line
       --> restrict to one side of the transition; the boundary itself is a
           physical feature, not an artefact.
    3. Coordinate singularity
       --> reparameterise inputs.
    4. Gauge / label symmetry
       --> enforce canonical ordering of inputs.

This module provides a single entry point, :func:`diagnose_low_information`,
that computes per-sample Fisher diagnostics, plots three diagnostic panels for
2D parameter spaces, prints a stratified summary, and emits a heuristic
classification of the failure mode plus a recommended action. The
classification is a heuristic, not a proof; always sanity-check against the
plot.

Typical usage::

    from degeneracy_distillery.diagnostics import diagnose_low_information

    out = diagnose_low_information(
        F_network_ensemble, thetas, ensemble_weights,
        param_names=['m1', 'm2'],
    )
    print(out['classification'], out['recommendation'])
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


__all__ = ["diagnose_low_information"]


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

def _safe_log10(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.clip(x, floor, None))


def _aggregate_fisher(
    F_ensemble: np.ndarray,
    ensemble_weights: Optional[np.ndarray],
) -> np.ndarray:
    """Weighted-mean over leading ensemble axis if present, then symmetrise."""
    F_np = np.asarray(F_ensemble)
    if F_np.ndim == 4:
        if ensemble_weights is None:
            w = np.full(F_np.shape[0], 1.0 / F_np.shape[0])
        else:
            w = np.asarray(ensemble_weights, dtype=float)
            if w.shape != (F_np.shape[0],):
                raise ValueError(
                    f"ensemble_weights shape {w.shape} does not match "
                    f"leading axis of F_ensemble {F_np.shape}"
                )
            w = w / w.sum()
        F_bar = np.tensordot(w, F_np, axes=(0, 0))
    elif F_np.ndim == 3:
        F_bar = F_np
    else:
        raise ValueError(
            "F_ensemble must have shape (E, N, n, n) or (N, n, n); "
            f"got {F_np.shape}"
        )
    return 0.5 * (F_bar + np.swapaxes(F_bar, -1, -2))


def _classify(
    theta: np.ndarray,
    log_det: np.ndarray,
    percentile: float,
    spread_dex_threshold: float,
    centroid_offset_threshold: float,
    anisotropy_threshold: float,
) -> Tuple[str, Dict[str, float], str]:
    """
    Heuristic classification of the low-information region in (theta, log_det).

    The classifier is intentionally conservative: it returns
    ``"mixed_or_unclear"`` whenever the data does not unambiguously match one
    of the canonical patterns. Treat the verdict as a starting point for
    inspection, not a final answer.

    Statistics computed:

    * ``spread_dex`` = ``median(log_det) - min(log_det)``: the *downward*
      dynamic range from the bulk to the worst-conditioned sample. The
      upward tail above the median is rarely a problem for the flattener,
      so this one-sided measure is the right scale for "is there a
      low-information region?".
    * ``centroid_offset_max``: max single-axis standardised distance
      between the low-info-subset centroid and the full-sample centroid.
      Dimension-invariant by construction (a corner pattern, a 1D edge,
      and a multi-axis corner all have max-component ~1; an interior
      ridge has max-component ~0).
    * ``centroid_offset_l2``: L2 norm of the same standardised offset.
      Reported for inspection but NOT used in the classification rules
      (its scale grows with n, which is what made the earlier version
      drift in higher dimensions).
    * ``anisotropy`` = ``sqrt(eigval_max / eigval_min)`` of the low-info
      subset's covariance: ratio of the longest to shortest principal
      axis of the low-info cluster. High = low-rank submanifold (1D
      ridge in 2D, k-D facet in n-D); low = isotropic blob.

    Decision tree (using ``centroid_offset_max`` only, so the same
    thresholds apply for any n):

    * spread_dex < spread_dex_threshold              -> well_conditioned
    * max-axis offset > thr AND anisotropy <= thr    -> snr_boundary
    * max-axis offset > thr AND anisotropy >  thr    -> edge_or_singularity
    * max-axis offset <= thr AND anisotropy >  thr   -> phase_boundary
    * otherwise                                       -> mixed_or_unclear
    """
    spread_dex = float(np.median(log_det) - log_det.min())

    if spread_dex < spread_dex_threshold:
        return (
            "well_conditioned",
            dict(spread_dex=spread_dex, n_low=0,
                 centroid_offset_max=float("nan"),
                 centroid_offset_l2=float("nan"),
                 anisotropy=float("nan")),
            (
                f"log10 det F varies by less than {spread_dex_threshold:.1f} "
                f"dex across the dataset (spread = {spread_dex:.2f}); the "
                f"Fisher is approximately uniform. No trimming or "
                f"reparameterisation is needed."
            ),
        )

    thr = float(np.percentile(log_det, percentile))
    low_mask = log_det <= thr
    n_low = int(low_mask.sum())
    if n_low < 5:
        return (
            "well_conditioned",
            dict(spread_dex=spread_dex, n_low=n_low,
                 centroid_offset_max=float("nan"),
                 centroid_offset_l2=float("nan"),
                 anisotropy=float("nan")),
            (
                "Fewer than 5 samples flagged as low-information; the "
                "Fisher is effectively uniform across the input domain."
            ),
        )

    n_params = theta.shape[1]
    all_centroid = theta.mean(0)
    all_std = theta.std(0)
    safe_std = np.where(all_std > 0, all_std, 1.0)
    low_centroid = theta[low_mask].mean(0)
    standardised_offset = (low_centroid - all_centroid) / safe_std
    centroid_offset_max = float(np.max(np.abs(standardised_offset)))
    centroid_offset_l2 = float(np.linalg.norm(standardised_offset))

    if n_low >= 3 and n_params >= 2:
        cov_low = np.cov(theta[low_mask].T)
        cov_low_eigs = np.linalg.eigvalsh(cov_low)
        cov_low_eigs = np.clip(cov_low_eigs, 1e-30, None)
        anisotropy = float(np.sqrt(cov_low_eigs[-1] / cov_low_eigs[0]))
    else:
        anisotropy = 1.0

    stats = dict(
        spread_dex=spread_dex,
        n_low=n_low,
        threshold_log10_det_F=thr,
        centroid_offset_max=centroid_offset_max,
        centroid_offset_l2=centroid_offset_l2,
        anisotropy=anisotropy,
    )

    if anisotropy > anisotropy_threshold and centroid_offset_max <= centroid_offset_threshold:
        ridge_word = "1D ridge" if n_params == 2 else "low-rank submanifold"
        return (
            "phase_boundary",
            stats,
            (
                f"Low-information samples form an elongated {ridge_word} "
                f"passing through the interior of the parameter domain "
                f"(anisotropy = {anisotropy:.1f}, max-axis offset = "
                f"{centroid_offset_max:.2f} sigma). This is the signature "
                f"of a critical line or phase boundary. Restrict the analysis "
                f"to one side of the transition (e.g. select samples above/"
                f"below the boundary) rather than trimming by det F. The "
                f"boundary itself is a physical feature, not an artefact, "
                f"and a single flattener cannot smoothly span both sides "
                f"of it."
            ),
        )

    if centroid_offset_max > centroid_offset_threshold and anisotropy <= anisotropy_threshold:
        return (
            "snr_boundary",
            stats,
            (
                "Low-information samples are concentrated in a corner / edge "
                "of the parameter domain, away from the bulk centroid "
                f"(max-axis offset = {centroid_offset_max:.2f} sigma, "
                f"anisotropy = {anisotropy:.1f}). This is the standard SNR / "
                "detection-boundary case (e.g. low chirp mass, low photon "
                "count). Trim by 'lambda_min' percentile (or 'det F' "
                "percentile) and report the resulting domain-of-validity in "
                "the analysis."
            ),
        )

    if centroid_offset_max > centroid_offset_threshold and anisotropy > anisotropy_threshold:
        return (
            "edge_or_singularity",
            stats,
            (
                "Low-information samples are simultaneously off-centre and "
                f"highly anisotropic (max-axis offset = "
                f"{centroid_offset_max:.2f} sigma, anisotropy = "
                f"{anisotropy:.1f}). This pattern is typical of a coordinate "
                "singularity along a domain edge (e.g. equal-mass boundary "
                "in a triangle, or a logarithmic singularity at zero). "
                "Consider reparameterising the inputs (e.g. log-transform) "
                "before trimming."
            ),
        )

    return (
        "mixed_or_unclear",
        stats,
        (
            "Low-information samples do not cleanly form a corner or a "
            f"low-rank ridge (max-axis offset = {centroid_offset_max:.2f} "
            f"sigma, anisotropy = {anisotropy:.1f}). Inspect the diagnostic "
            "plot manually. Possible causes: multiple disjoint low-info "
            "regions, a label-symmetry that has not been canonicalised, "
            "or noise in the Fisher predictions."
        ),
    )


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def diagnose_low_information(
    F_ensemble: np.ndarray,
    theta: np.ndarray,
    ensemble_weights: Optional[np.ndarray] = None,
    *,
    param_names: Optional[Sequence[str]] = None,
    classification_percentile: float = 10.0,
    spread_dex_threshold: float = 1.5,
    centroid_offset_threshold: float = 1.0,
    anisotropy_threshold: float = 4.0,
    plot: bool = True,
    figsize: Tuple[float, float] = (16.0, 4.6),
    n_quiver: int = 400,
    rng_seed: int = 0,
    title: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Diagnose low-information regions of a Fisher field over the input grid.

    Computes per-sample diagnostics from the (optionally ensemble-averaged)
    Fisher matrix, produces a 3-panel scatter plot for 2D parameter spaces
    (``log10 det F``, ``log10 lambda_min(F)``, softest-eigenvector quiver),
    prints a stratified per-parameter summary, and emits a heuristic
    classification of any low-information pattern (``well_conditioned``,
    ``snr_boundary``, ``phase_boundary``, ``edge_or_singularity``, or
    ``mixed_or_unclear``) together with a recommended next action.

    Parameters
    ----------
    F_ensemble : array_like, shape (E, N, n, n) or (N, n, n)
        Fisher matrices per sample. If 4D, the leading axis is averaged using
        ``ensemble_weights``. If 3D, treated as already aggregated.
    theta : array_like, shape (N, n)
        Parameter samples. ``n`` must equal the trailing dimension of
        ``F_ensemble``. The plot panel is only rendered for ``n == 2``;
        the classification and summary work for any ``n >= 1``.
    ensemble_weights : array_like, shape (E,), optional
        Weights for ensemble members. Uniform if None. Ignored for 3D
        ``F_ensemble``.
    param_names : sequence of str, optional
        Display names for the parameters. Defaults to ``theta_1, theta_2,
        ...``.
    classification_percentile : float, default 10.0
        Percentile of ``log10 det F`` defining the "low-information"
        subset for the classification heuristic.
    spread_dex_threshold : float, default 1.5
        If ``log10 det F`` varies by less than this many dex across the
        dataset, the Fisher is declared ``well_conditioned`` regardless of
        the spatial pattern.
    centroid_offset_threshold : float, default 1.0
        Threshold on the **max single-axis** standardised offset between the
        low-info-subset centroid and the full-sample centroid (units of
        sigma) used to distinguish patterns whose low-info samples sit at
        a domain corner/edge (high) from patterns whose low-info samples
        pass through the interior (low). Using the max-component rather
        than the L2 norm makes the threshold dimension-invariant.
    anisotropy_threshold : float, default 4.0
        Threshold on the square root of the eigenvalue ratio of the
        low-info-subset covariance, used to distinguish "ridge" patterns
        (high anisotropy) from "corner" patterns (low anisotropy).
    plot : bool, default True
        Whether to render the matplotlib diagnostic figure.
    figsize : (width, height), default (16, 4.6)
        Matplotlib figure size.
    n_quiver : int, default 400
        Maximum number of softest-eigenvector arrows drawn in the third
        panel; subsampled if the dataset is larger.
    rng_seed : int, default 0
        Seed for the deterministic quiver subsampling.
    title : str, optional
        Suptitle for the figure. Default includes the verdict.
    verbose : bool, default True
        Print the stratified summary and classification verdict to stdout.

    Returns
    -------
    dict
        Keys: ``F_bar`` (N, n, n), ``eigvals`` (N, n) ascending,
        ``eigvecs`` (N, n, n), ``det_F`` (N,), ``lam_min`` (N,),
        ``lam_max`` (N,), ``cond_F`` (N,), ``classification`` (str),
        ``classification_stats`` (dict), ``recommendation`` (str),
        ``fig`` (matplotlib Figure or None).
    """
    theta = np.asarray(theta)
    if theta.ndim != 2:
        raise ValueError(
            f"theta must be 2D with shape (N, n_params); got {theta.shape}"
        )
    n_samples, n_params = theta.shape

    if param_names is None:
        param_names = [f"theta_{i + 1}" for i in range(n_params)]
    elif len(param_names) != n_params:
        raise ValueError(
            f"param_names has length {len(param_names)}; expected {n_params}"
        )

    F_bar = _aggregate_fisher(F_ensemble, ensemble_weights)
    if F_bar.shape != (n_samples, n_params, n_params):
        raise ValueError(
            f"F_ensemble aggregates to shape {F_bar.shape}; expected "
            f"({n_samples}, {n_params}, {n_params}) to match theta."
        )

    evals, evecs = np.linalg.eigh(F_bar)
    n_neg = int((evals[:, 0] < 0).sum())
    if n_neg > 0 and verbose:
        print(
            f"WARNING: {n_neg}/{n_samples} samples have a negative smallest "
            f"eigenvalue (min = {evals[:, 0].min():.2e}); clipping for log "
            "plots. This usually indicates non-PSD Fisher predictions and is "
            "worth investigating upstream."
        )

    det_F = evals.prod(axis=-1)
    lam_min = evals[:, 0]
    lam_max = evals[:, -1]
    v_min = evecs[:, :, 0]
    cond_F = lam_max / np.clip(lam_min, 1e-30, None)

    log_det = _safe_log10(det_F)
    log_lmin = _safe_log10(lam_min)

    cls, cls_stats, recommendation = _classify(
        theta=theta,
        log_det=log_det,
        percentile=classification_percentile,
        spread_dex_threshold=spread_dex_threshold,
        centroid_offset_threshold=centroid_offset_threshold,
        anisotropy_threshold=anisotropy_threshold,
    )

    fig = None
    if plot:
        if n_params == 2:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(
                1, 3, figsize=figsize, constrained_layout=True
            )

            sc0 = axes[0].scatter(
                theta[:, 0], theta[:, 1], c=log_det, s=8, cmap="viridis"
            )
            axes[0].set_title(r"$\log_{10}\,\det F_\theta$")
            axes[0].set_xlabel(param_names[0])
            axes[0].set_ylabel(param_names[1])
            plt.colorbar(sc0, ax=axes[0])

            sc1 = axes[1].scatter(
                theta[:, 0], theta[:, 1], c=log_lmin, s=8, cmap="plasma"
            )
            axes[1].set_title(r"$\log_{10}\,\lambda_{\min}(F_\theta)$")
            axes[1].set_xlabel(param_names[0])
            axes[1].set_ylabel(param_names[1])
            plt.colorbar(sc1, ax=axes[1])

            axes[2].scatter(
                theta[:, 0], theta[:, 1], c=log_lmin, s=4, cmap="plasma",
                alpha=0.35,
            )
            n_sub = min(n_quiver, n_samples)
            sub_idx = np.random.RandomState(rng_seed).choice(
                n_samples, size=n_sub, replace=False
            )
            axes[2].quiver(
                theta[sub_idx, 0], theta[sub_idx, 1],
                v_min[sub_idx, 0], v_min[sub_idx, 1],
                angles="xy", pivot="middle",
                scale=25.0, width=0.003, color="k", alpha=0.75,
            )
            axes[2].set_title(r"softest direction $v_{\min}$")
            axes[2].set_xlabel(param_names[0])
            axes[2].set_ylabel(param_names[1])

            fig.suptitle(
                title if title is not None
                else f"Fisher diagnostic [verdict: {cls}]",
                y=1.04,
            )
            plt.show()
        elif verbose:
            print(
                f"NOTE: plot skipped because n_params={n_params} (only 2D "
                "is currently supported). Classification and summary below "
                "still apply."
            )

    if verbose:
        print(
            f"\n=== Fisher diagnostic summary "
            f"(N={n_samples}, n_params={n_params}) ==="
        )
        print(
            f"  log10 det F     : "
            f"min={log_det.min():.2f}, "
            f"med={np.median(log_det):.2f}, "
            f"max={log_det.max():.2f}, "
            f"spread={log_det.max() - log_det.min():.2f} dex"
        )
        print(
            f"  log10 lambda_min: "
            f"min={log_lmin.min():.2f}, "
            f"med={np.median(log_lmin):.2f}, "
            f"max={log_lmin.max():.2f}"
        )
        print(
            f"  cond(F)         : "
            f"min={cond_F.min():.2e}, "
            f"med={np.median(cond_F):.2e}, "
            f"max={cond_F.max():.2e}"
        )
        print("\n  Per-parameter quantile stratification:")
        for i in range(n_params):
            ti = theta[:, i]
            print(f"    {param_names[i]}:")
            for ql, qh, label in [
                (0.00, 0.05, " bottom 5%"),
                (0.45, 0.55, " middle 10%"),
                (0.95, 1.00, "    top 5%"),
            ]:
                lo, hi = np.quantile(ti, [ql, qh])
                mask = (ti >= lo) & (ti <= hi)
                if mask.sum() == 0:
                    continue
                print(
                    f"      {label} ({param_names[i]} in "
                    f"[{lo:.3g}, {hi:.3g}], n={mask.sum():4d}): "
                    f"med log10 det F = {np.median(log_det[mask]): .2f}"
                )
        print(f"\n  Verdict: {cls}")
        for k in ("spread_dex", "centroid_offset_max",
                  "centroid_offset_l2", "anisotropy"):
            v = cls_stats.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                print(f"    {k:21s}= {v:.2f}")
        n_low = cls_stats.get("n_low", 0)
        if n_low:
            print(f"    n_low samples flagged = {n_low}")
        print(f"\n  Recommendation: {recommendation}\n")

    return dict(
        F_bar=F_bar,
        eigvals=evals,
        eigvecs=evecs,
        det_F=det_F,
        lam_min=lam_min,
        lam_max=lam_max,
        cond_F=cond_F,
        classification=cls,
        classification_stats=cls_stats,
        recommendation=recommendation,
        fig=fig,
    )
