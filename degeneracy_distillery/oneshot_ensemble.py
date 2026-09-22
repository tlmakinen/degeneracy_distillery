"""Bootstrap one-step ensemble at a fixed rank, then align for SR.

Fits ``K`` members with ``train_oneshot`` at a shared ``m``. Does not rerun
the rank ladder. The reference member is the best out-of-bag NLL. Alignment
uses ``process_ensemble_rotation_v2`` with ``F = J^T J`` and
``canonicalize="none"`` so rectangular ``m < d`` maps stay valid.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np

from degeneracy_distillery.align_coords import process_ensemble_rotation_v2
from degeneracy_distillery.oneshot import OneShotFit, jtj_eigengap, train_oneshot


def fit_oneshot_ensemble(
    theta,
    x,
    m: int,
    K: int,
    *,
    lo,
    hi,
    th_te,
    x_te,
    steps: int,
    seed: int,
    use_log_features: bool = True,
    estimator_factory: Optional[Callable] = None,
    bootstrap: bool = True,
    batch: int = 512,
    lr: float = 1e-3,
    min_gap: float = 10.0,
    r_hat: Optional[int] = None,
    align_subsample: Optional[int] = None,
    canonicalize: str = "none",
    align_mode: str = "procrustes",
    verbose: bool = True,
    init_fit: Optional[OneShotFit] = None,
) -> dict[str, Any]:
    """Fit ``K`` one-step maps at fixed ``m`` and align them.

    Members train on bootstrap resamples of ``(theta, x)``. The reference
    for Procrustes is the member with the lowest out-of-bag NLL. Each
    member also reports ``jtj_eigengap`` on ``th_te`` so rank agreement
    with the ladder can be scored without extra fits.

    Returns
    -------
    dict
        ``fits``, ``oob_nlls``, ``best_model_idx``, ``weights``,
        ``jtj_ranks``, ``jtj_rank_agreement_k``, plus the aligned
        ``y``, ``y_std``, ``dy_sr``, ``rotmats``, ``X``, ``Fs``.
    """
    theta = np.asarray(theta)
    x = np.asarray(x)
    th_te = np.asarray(th_te)
    x_te = np.asarray(x_te)
    K = max(2, int(K))
    m = int(m)
    n = int(theta.shape[0])
    rng = np.random.default_rng(int(seed) + 41)
    t_fit = time.time()

    fits: list[OneShotFit] = []
    oob_nlls: list[float] = []
    init_params = None if init_fit is None else init_fit.params
    m0 = int(m if init_fit is None else init_fit.m0)

    start = 0
    if init_fit is not None:
        if verbose:
            print(f"[ensemble] member 0 = ladder map (m={m})", flush=True)
        fits.append(init_fit)
        oob_nlls.append(float(np.mean(init_fit.nll_vec(th_te, x_te))))
        start = 1

    for k in range(start, K):
        if bootstrap:
            idx = rng.integers(0, n, size=n)
            oob = np.setdiff1d(np.arange(n), np.unique(idx))
            if oob.size < 20:
                oob = np.arange(min(50, n))
            th_k = theta[idx]
            x_k = x[idx]
            th_oob, x_oob = theta[oob], x[oob]
        else:
            th_k, x_k = theta, x
            th_oob, x_oob = th_te, x_te
        if verbose:
            print(f"[ensemble] member {k + 1}/{K}  steps={steps}", flush=True)
        fit = train_oneshot(
            th_k, x_k, th_oob, x_oob,
            m=m, m0=m0, lo=lo, hi=hi,
            steps=int(steps), seed=int(seed) + 1009 * (k + 1),
            init_params=init_params,
            batch=batch, lr=lr, verbose=verbose,
            use_log_features=use_log_features,
            estimator_factory=estimator_factory,
        )
        fits.append(fit)
        oob_nlls.append(float(np.mean(fit.nll_vec(th_oob, x_oob))))
        if verbose:
            print(f"  member {k}: oob NLL {oob_nlls[-1]:.4f}", flush=True)

    runtime_fit_s = time.time() - t_fit
    t_align = time.time()
    etas_te = [np.asarray(f.eta(th_te)) for f in fits]
    jacs_te = [np.asarray(f.jac(th_te)) for f in fits]
    jacs_stack = np.stack(jacs_te)
    F_ensemble = np.einsum("knmi,knmj->knij", jacs_stack, jacs_stack)

    ranks = []
    for J in jacs_te:
        ranks.append(int(jtj_eigengap(J, min_gap=min_gap)["rank"]))
    r_ref = int(m if r_hat is None else r_hat)
    agreement = int(sum(r == r_ref for r in ranks))

    n_te = int(th_te.shape[0])
    n_align = n_te if align_subsample is None else min(int(align_subsample), n_te)
    randidx = np.arange(n_te)
    if n_align < n_te:
        randidx = np.sort(rng.choice(n_te, n_align, replace=False))

    K_eff = len(fits)
    weights = np.ones(K_eff) / K_eff
    datafile = {
        "theta": th_te,
        "eta_ensemble": np.stack(etas_te),
        "Jbar_ensemble": jacs_stack,
        "F_ensemble": F_ensemble,
        "ensemble_weights": weights,
        "norm_factor": 1.0,
    }
    best_model_idx = int(np.argmin(np.asarray(oob_nlls)))
    Favg = F_ensemble[best_model_idx][randidx]
    if verbose:
        print(
            f"[ensemble] align reference=member {best_model_idx}  "
            f"n_align={n_align}  canonicalize={canonicalize}",
            flush=True,
        )
    aligned = process_ensemble_rotation_v2(
        datafile=datafile,
        randidx=randidx,
        Favg=Favg,
        best_model_idx=best_model_idx,
        n_d=1.0,
        align_mode=align_mode,
        separate_nonlinearity=True,
        canonicalize=canonicalize,
        use_prior_normalization=True,
        restore_reference_mean=True,
        Fisher_to_flatten="best",
        verbose=verbose,
        offset_delta=0.1,
    )
    return {
        "fits": fits,
        "oob_nlls": np.asarray(oob_nlls, dtype=np.float64),
        "best_model_idx": best_model_idx,
        "weights": weights,
        "jtj_ranks": np.asarray(ranks, dtype=int),
        "jtj_rank_agreement_k": agreement,
        "y": np.asarray(aligned["y"]),
        "y_std": np.asarray(aligned["y_std"]),
        "dy_sr": np.asarray(aligned["dy_sr"]),
        "rotmats": np.asarray(aligned["rotmats"]),
        "X": np.asarray(aligned["X"]),
        "Fs": np.asarray(aligned["Fs"]),
        "aligned": aligned,
        "eta_ensemble_raw": np.stack(etas_te),
        "jac_ensemble_raw": jacs_stack,
        "runtime_fit_s": float(runtime_fit_s),
        "runtime_align_s": float(time.time() - t_align),
    }
