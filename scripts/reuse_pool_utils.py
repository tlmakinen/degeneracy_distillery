"""Shared helpers for NPE budget sweeps that reuse the discovery simulations.

Used by ``sir_nsims_logprob_sweep.py``, ``gw_waveform_nsims_logprob_sweep.py`` and
``gw_imr_nsims_logprob_sweep.py`` when they are given ``--training-data`` (a pool
that begins with the discovery driver's simulations) and/or
``--eta-inverse numeric`` (no closed-form inverse of the discovered coordinates).

Nothing here is imported on the sweeps' legacy code paths.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree


_ALLOWED_NAMES: dict[str, Any] = {
    "E": math.e,
    "abs": np.abs,
    "arccos": np.arccos,
    "arcsin": np.arcsin,
    "arctan": np.arctan,
    "cos": np.cos,
    "e": math.e,
    "exp": np.exp,
    "log": np.log,
    "logAbs": lambda x: np.log(np.abs(x)),
    "maximum": np.maximum,
    "minimum": np.minimum,
    "pi": np.pi,
    "sin": np.sin,
    "sqrt": np.sqrt,
    "tan": np.tan,
}


def make_expression_transform64(
    expressions: tuple[str, ...] | list[str],
    aliases: tuple[str, ...] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Float64 twin of the sweeps' ``make_expression_transform``.

    The sweeps' own transform casts to float32, which is too coarse for
    finite-difference Jacobians and Newton iterations. Column ``k`` of the input
    is bound to ``X{k+1}`` and, when given, to ``aliases[k]`` (``m1``/``m2`` in
    the GW sweeps).
    """
    compiled = [
        compile(expr.replace("^", "**"), f"<coordinate:{idx}>", "eval")
        for idx, expr in enumerate(expressions)
    ]

    def transform(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        local_names = {f"X{k + 1}": x[:, k] for k in range(x.shape[1])}
        if aliases is not None:
            local_names |= {alias: x[:, k] for k, alias in enumerate(aliases)}
        with np.errstate(all="ignore"):
            cols = [
                np.broadcast_to(
                    np.asarray(
                        eval(code, {"__builtins__": {}}, _ALLOWED_NAMES | local_names),
                        dtype=np.float64,
                    ),
                    (x.shape[0],),
                )
                for code in compiled
            ]
        return np.column_stack(cols)

    return transform


def forward_jacobian(
    forward_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    rel_step: float = 1e-6,
) -> np.ndarray:
    """Central finite-difference Jacobian d forward / d x, shape (N, d_out, d_in)."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    cols = []
    for dim in range(d):
        step = rel_step * np.maximum(1.0, np.abs(x[:, dim]))
        plus = x.copy()
        minus = x.copy()
        plus[:, dim] += step
        minus[:, dim] -= step
        cols.append((forward_fn(plus) - forward_fn(minus)) / (2.0 * step[:, None]))
    return np.stack(cols, axis=2)


def forward_logabsdet(
    forward_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Per-point log|det(d forward / d x)|; NaN where the Jacobian is not finite."""
    jac = forward_jacobian(forward_fn, x)
    out = np.full(jac.shape[0], np.nan)
    finite = np.isfinite(jac).all(axis=(1, 2))
    if finite.any():
        sign, logabs = np.linalg.slogdet(jac[finite])
        logabs = np.where(sign == 0.0, np.nan, logabs)
        out[finite] = logabs
    return out


def mean_inverse_logdet_from_forward(
    forward_fn: Callable[[np.ndarray], np.ndarray],
    x_train: np.ndarray,
    max_points: int,
    seed: int,
) -> float:
    """Mean log|det(dx/deta)| over (a subsample of) the training points.

    Same quantity as the sweeps' ``inverse_logdet_jacobian`` -- which
    differentiates an explicit eta -> theta expression at the training eta --
    obtained from the forward map through log|det dx/deta| = -log|det deta/dx|
    at the matching x. The subsample is drawn the same way (same rng, same
    ``max_points``), so the two agree when a closed-form inverse exists.
    """
    rng = np.random.default_rng(seed)
    x_train = np.asarray(x_train, dtype=np.float64)
    if x_train.shape[0] > max_points:
        x_train = x_train[rng.choice(x_train.shape[0], size=max_points, replace=False)]
    logdet = forward_logabsdet(forward_fn, x_train)
    logdet = logdet[np.isfinite(logdet)]
    if logdet.size == 0:
        raise RuntimeError("Could not compute any finite forward Jacobian determinants.")
    return float(-np.mean(logdet))


class NumericInverse:
    """Vectorised numerical inverse of a coordinate map on a box-shaped domain.

    ``forward_fn`` maps (N, d) points x to (N, d) coordinates eta. The inverse
    starts from the nearest of ``n_ref`` reference points (uniform in the box,
    optionally restricted by ``domain_fn``) and runs damped Gauss-Newton steps.
    Points that do not converge, or whose solution leaves the padded box, come
    back as NaN, which the sweeps already treat as "outside the theta prior".
    The map has to be injective on the domain; ``round_trip_error`` checks that.
    """

    def __init__(
        self,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        low: np.ndarray,
        high: np.ndarray,
        domain_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        n_ref: int = 50_000,
        seed: int = 0,
        tol: float = 1e-7,
        max_iter: int = 60,
        pad_fraction: float = 0.05,
    ) -> None:
        self.forward_fn = forward_fn
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        pad = pad_fraction * (self.high - self.low)
        self.clip_low = self.low - pad
        self.clip_high = self.high + pad

        rng = np.random.default_rng(seed)
        ref = rng.uniform(self.low, self.high, size=(n_ref, self.low.size))
        if domain_fn is not None:
            ref = ref[np.asarray(domain_fn(ref), dtype=bool)]
        eta_ref = forward_fn(ref)
        finite = np.isfinite(eta_ref).all(axis=1)
        ref, eta_ref = ref[finite], eta_ref[finite]
        if ref.shape[0] < 16:
            raise RuntimeError("NumericInverse: too few finite reference points in the domain.")
        self.ref = ref
        self.eta_scale = np.maximum(eta_ref.std(axis=0), 1e-12)
        self.tree = cKDTree(eta_ref / self.eta_scale)

    def _residual(self, x: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        res = (self.forward_fn(x) - target) / self.eta_scale
        norm = np.max(np.abs(res), axis=1)
        norm = np.where(np.isfinite(norm), norm, np.inf)
        return res, norm

    def __call__(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        n, d = eta.shape
        out = np.full((n, d), np.nan)
        finite = np.isfinite(eta).all(axis=1)
        if not finite.any():
            return out.astype(np.float32)

        target = eta[finite]
        _, idx = self.tree.query(target / self.eta_scale)
        x = self.ref[idx].copy()
        res, norm = self._residual(x, target)
        active = np.flatnonzero(norm > self.tol)

        for _ in range(self.max_iter):
            if active.size == 0:
                break
            xa, ta, ra, na = x[active], target[active], res[active], norm[active]
            jac = forward_jacobian(self.forward_fn, xa) / self.eta_scale[None, :, None]
            jac = np.where(np.isfinite(jac), jac, 0.0)
            jtj = np.einsum("nki,nkj->nij", jac, jac)
            ridge = 1e-12 * (np.trace(jtj, axis1=1, axis2=2) + 1e-300)
            jtj = jtj + ridge[:, None, None] * np.eye(d)[None]
            rhs = -np.einsum("nki,nk->ni", jac, np.where(np.isfinite(ra), ra, 0.0))
            step = np.linalg.solve(jtj, rhs[..., None])[..., 0]
            step = np.where(np.isfinite(step), step, 0.0)

            best_x, best_res, best_norm = xa, ra, na
            improved = np.zeros(active.size, dtype=bool)
            for damping in (1.0, 0.5, 0.25, 0.1, 0.02):
                todo = ~improved
                if not todo.any():
                    break
                trial = np.clip(xa[todo] + damping * step[todo], self.clip_low, self.clip_high)
                trial_res, trial_norm = self._residual(trial, ta[todo])
                better = trial_norm < na[todo]
                rows = np.flatnonzero(todo)[better]
                if rows.size:
                    best_x = best_x.copy() if best_x is xa else best_x
                    best_res = best_res.copy() if best_res is ra else best_res
                    best_norm = best_norm.copy() if best_norm is na else best_norm
                    best_x[rows] = trial[better]
                    best_res[rows] = trial_res[better]
                    best_norm[rows] = trial_norm[better]
                    improved[rows] = True

            x[active], res[active], norm[active] = best_x, best_res, best_norm
            # Stalled points (no damping helped) are dropped: they stay unconverged.
            active = active[improved & (best_norm > self.tol)]

        solved = norm <= self.tol
        x[~solved] = np.nan
        out[finite] = x
        return out.astype(np.float32)

    def round_trip_error(self, x: np.ndarray) -> tuple[float, float]:
        """(fraction not recovered, max abs error relative to the box width)."""
        x = np.asarray(x, dtype=np.float64)
        back = np.asarray(self(self.forward_fn(x)), dtype=np.float64)
        failed = ~np.isfinite(back).all(axis=1)
        err = np.abs(back - x) / (self.high - self.low)
        max_err = float(np.nanmax(err[~failed])) if (~failed).any() else float("nan")
        return float(failed.mean()), max_err


class FrozenPCA:
    """The slice of sklearn's PCA interface the GW sweeps use, with a fixed basis."""

    def __init__(
        self,
        mean: np.ndarray,
        components: np.ndarray,
        explained_variance_ratio: np.ndarray,
    ) -> None:
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.components_ = np.asarray(components, dtype=np.float64)
        self.explained_variance_ratio_ = np.asarray(explained_variance_ratio, dtype=np.float64)
        self.n_components = int(self.components_.shape[0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean_) @ self.components_.T


def load_gw_discovery_pool(
    path: Path,
    freqs: np.ndarray,
    whiten_factor: np.ndarray,
    n_pca: int,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    waveform: str,
) -> dict[str, Any]:
    """Load a GW discovery archive written by ``*_notebook_run.py --save-training-data``.

    Returns the pool (discovery training simulations first, then the discovery
    held-out simulations), the frozen PCA basis, and bookkeeping. Raises if the
    sweep's simulator settings differ from the ones the archive was made with.

    The discovery drivers draw (m1, m2) on the full square, the sweeps on the
    triangle m1 >= m2. Both waveform models depend on the masses only through
    symmetric combinations, so a discovery simulation at (m1, m2) with m1 < m2
    *is* the simulation at (m2, m1): its label is swapped, its data untouched.
    A uniform square folded this way is uniform on the triangle, i.e. the
    sweep's own prior.
    """
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "theta_train", "data_train", "theta_test", "data_test",
            "pca_mean", "pca_components", "pca_explained_variance_ratio",
            "freqs", "whiten", "sim_mass_range", "sim_n_pca", "sim_waveform",
        )
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError(f"{path} is missing keys {missing}; expected a discovery archive.")
        arrays = {key: np.asarray(archive[key]) for key in archive.files}

    problems = []
    if str(arrays["sim_waveform"]) != waveform:
        problems.append(f"waveform: archive={arrays['sim_waveform']} sweep={waveform}")
    if int(arrays["sim_n_pca"]) != int(n_pca) or arrays["pca_components"].shape[0] != int(n_pca):
        problems.append(f"n_pca: archive={int(arrays['sim_n_pca'])} sweep={n_pca}")
    if arrays["freqs"].shape != freqs.shape or not np.allclose(arrays["freqs"], freqs, rtol=0, atol=1e-9):
        problems.append("frequency grid differs")
    elif not np.allclose(arrays["whiten"], whiten_factor, rtol=1e-10, atol=0):
        problems.append("whitening vector differs")
    lo1, hi1, lo2, hi2 = (float(v) for v in arrays["sim_mass_range"])
    if not np.allclose([lo1, lo2], prior_low) or not np.allclose([hi1, hi2], prior_high):
        problems.append(
            f"mass range: archive=[{lo1},{hi1}]x[{lo2},{hi2}] "
            f"sweep={prior_low.tolist()}..{prior_high.tolist()}"
        )
    if problems:
        raise ValueError(
            "Discovery archive and sweep simulator settings differ: " + "; ".join(problems)
        )

    theta = np.concatenate([arrays["theta_train"], arrays["theta_test"]], axis=0).astype(np.float32)
    data = np.concatenate([arrays["data_train"], arrays["data_test"]], axis=0).astype(np.float32)
    swapped = theta[:, 0] < theta[:, 1]
    theta_folded = np.where(swapped[:, None], theta[:, ::-1], theta).astype(np.float32)

    return {
        "theta": theta_folded,
        "data": data,
        "n_train": int(arrays["theta_train"].shape[0]),
        "n_heldout": int(arrays["theta_test"].shape[0]),
        "n_label_swapped": int(swapped.sum()),
        "pca": FrozenPCA(
            arrays["pca_mean"], arrays["pca_components"], arrays["pca_explained_variance_ratio"]
        ),
        "n_pca_basis_waveforms": int(arrays.get("sim_n_pca_basis_waveforms", -1)),
    }


def posterior_log_prob(posterior: Any, params: np.ndarray, x: np.ndarray, device: str) -> np.ndarray:
    """log q(params_i | x_i) for paired rows, from the trained flow itself.

    Each ensemble member is an ltu-ili ``LampeNPE``; calling it evaluates the
    flow with its input/parameter normalisation and *without* the prior box,
    i.e. exactly the quantity the training and validation log-probs are made
    of -- here on a set that neither arm chose its stopping epoch on. Members
    are combined as a mixture with the ensemble weights (the sweeps train a
    single member, so this is that member's log-prob).
    """
    import torch

    theta_t = torch.as_tensor(np.asarray(params, dtype=np.float32), device=device)
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    members = list(getattr(posterior, "posteriors", [posterior]))
    weights = getattr(posterior, "weights", None)
    if weights is None:
        weights = torch.ones(len(members), device=device) / len(members)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=device)

    member_logps = []
    with torch.no_grad():
        for member in members:
            member.eval()
            member_logps.append(member(theta_t, x_t))
    stacked = torch.stack(member_logps, dim=0)
    logp = torch.logsumexp(stacked + torch.log(weights)[:, None], dim=0)
    return logp.detach().cpu().numpy().astype(np.float64)
