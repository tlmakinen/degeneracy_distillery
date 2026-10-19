"""Gaussian (mu, sigma) adapter for the one-step flattening panel.

Draws match ``mu_sig_degen_install.py``: mu ~ U(-5, 5), sigma ~ U(0.2, 7),
x = mu + sigma Z with Z ~ N(0, I_50). The analytic Fisher is
50 * diag(1/sigma^2, 2/sigma^2). Geometry helpers are copied here so the
install script is never imported (it trains on import).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

N_D = 50
MIN_MU = -5.0
MAX_MU = 5.0
MIN_SIGMA = 0.2
MAX_SIGMA = 7.0
MU_STAR = 0.0
SIGMA_STAR = 1.0
V_STAR = float(np.sqrt(2.0))


# eta(theta): deeper than the stock 3x128 flattener. The flattening
# score is a property of J_eta(theta), so this is the lever for the
# left-hand panel. eta(x) default is a mild funnel on the 50-vector.
FLAT_FEATURES = (256, 256, 128)
Z_RANGE = (1.0, 2.0)
EST_FEATURES = (256, 256, 128)
# One trunk per ensemble member. Member 0 uses EST_FEATURES; the rest
# cycle this list so y_std sees real data-path variation.
EST_FEATURE_BANK: tuple[tuple[int, ...], ...] = (
    (256, 256, 128),
    (128, 64),
    (256, 128),
    (128, 128, 64),
    (64, 64, 64),
    (256, 64),
    (192, 96, 48),
    (128, 128),
    (256, 256),
    (64, 128, 64),
)


def _estimator(m: int, features: Sequence[int] = EST_FEATURES):
    from degeneracy_distillery.oneshot import Estimator

    return Estimator(m=int(m), features=tuple(features))


def estimator_factory_for(features: Sequence[int]):
    feats = tuple(int(w) for w in features)

    def factory(m: int):
        return _estimator(m, feats)

    factory.features = feats  # type: ignore[attr-defined]
    return factory


class MuSigmaProblem:
    """Rank-2 location-scale Gaussian. ``use_log_features=False`` (mu can change sign)."""

    def __init__(self, n_d: int = N_D, **_ignored):
        self.d = 2
        self.n_d = int(n_d)
        self.name = "mu_sigma"
        self.param_names = ("mu", "sigma")
        self.theta_lo = np.array([MIN_MU, MIN_SIGMA], dtype=np.float64)
        self.theta_hi = np.array([MAX_MU, MAX_SIGMA], dtype=np.float64)
        self.use_log_features = False
        self.m_probe = 4
        self.expected_rank = 2

    def sample(self, n: int, rng: np.random.Generator):
        n = int(n)
        mu = rng.uniform(MIN_MU, MAX_MU, size=n)
        sigma = rng.uniform(MIN_SIGMA, MAX_SIGMA, size=n)
        th = np.stack([mu, sigma], axis=-1)
        x = mu[:, None] + sigma[:, None] * rng.normal(size=(n, self.n_d))
        return th, x

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        n = int(n)
        mu = rng.uniform(MIN_MU, MAX_MU, size=n)
        sigma = rng.uniform(MIN_SIGMA, MAX_SIGMA, size=n)
        return np.stack([mu, sigma], axis=-1)

    def estimator_factory(self, m: int):
        return _estimator(m, EST_FEATURES)

    def estimator_factories(self, k: int):
        """``k`` eta(x) factories with distinct widths for the ensemble."""
        bank = EST_FEATURE_BANK
        return [estimator_factory_for(bank[i % len(bank)]) for i in range(int(k))]

    def truth_coords(self, theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=np.float64)
        return np.stack(
            [
                0.5 * (th[:, 0] - MU_STAR) ** 2 + (th[:, 1] - SIGMA_STAR),
                np.sqrt(2.0) * (th[:, 0] - MU_STAR),
            ],
            axis=1,
        )

    def gates(self, payload: dict) -> dict[str, bool]:
        r_hat = int(payload.get("r_hat", -1))
        return {"rank_correct": bool(r_hat == 2)}

    def sr_hints(self) -> dict:
        return {
            "allowed_symbols": "add,mul,div,pow,constant,variable,sqrt",
            "max_length": 25,
            "max_depth": 10,
        }


def true_fisher(theta, n_d: int = N_D):
    """Analytic Fisher of n_d i.i.d. N(mu, sigma^2) draws. ``theta`` is (mu, sigma)."""
    import jax.numpy as jnp

    sig = theta[1]
    return n_d * jnp.array(
        [[1.0 / sig**2, 0.0], [0.0, 2.0 / sig**2]],
        dtype=theta.dtype,
    )


def x_to_sinhbeta(x):
    import jax.numpy as jnp

    return (
        jnp.sqrt(
            (0.5 * (x[0] - MU_STAR) ** 2 + (x[1] - SIGMA_STAR) ** 2)
            * (0.5 * (x[0] - MU_STAR) ** 2 + (x[1] + SIGMA_STAR) ** 2)
        )
        / x[1]
        / SIGMA_STAR
        / 2.0
    )


def x_to_xprime(x):
    """Analytic geodesic-normal coordinates about (mu*, sigma*) = (0, 1)."""
    import jax.numpy as jnp

    sinh_beta = x_to_sinhbeta(x)
    beta = jnp.arcsinh(sinh_beta)
    xi = jnp.where(
        jnp.abs(beta) > 1e-4,
        beta * (x[0] - MU_STAR) / x[1] / sinh_beta,
        0.0,
    )
    eta = jnp.where(
        jnp.abs(beta) > 1e-4,
        xi
        * (0.5 * (x[0] - MU_STAR) ** 2 + x[1] ** 2 - SIGMA_STAR**2)
        / (SIGMA_STAR * (x[0] - MU_STAR) * jnp.sqrt(2.0)),
        0.0,
    )
    return jnp.array([xi, eta])


def xp_to_beta(xp):
    import jax.numpy as jnp

    return jnp.sqrt(xp[0] ** 2 + xp[1] ** 2) / V_STAR


def hyperbolic_beta(theta):
    """Hyperbolic radius beta of the analytic Gaussian geometry."""
    return xp_to_beta(x_to_xprime(theta))


def get_beta(x, return_beta: bool = False):
    """Intrinsic flattening residual of the geodesic coordinates, and optionally beta."""
    import jax.numpy as jnp

    beta = hyperbolic_beta(x)
    residual = (jnp.sinh(beta) ** 2.0) / beta**2.0 - 1.0
    if return_beta:
        return residual, beta
    return residual


def _frobenius_to_eye(q):
    from degeneracy_distillery.sr_utils import norm
    import jax.numpy as jnp

    return norm(q - jnp.eye(q.shape[-1]))


def flatten_losses_from_jacobian(jacs, fishers):
    """Per-row Frobenius ||(J^+)^T F J^+ - I|| from Jacobians and Fishers."""
    import jax
    import jax.numpy as jnp
    from degeneracy_distillery.preprocessing_utils import flatten_with_numerical_jacobian

    j = jnp.asarray(jacs)
    f = jnp.asarray(fishers)
    flats = jax.vmap(flatten_with_numerical_jacobian)(j, f)
    losses = jax.vmap(_frobenius_to_eye)(flats)
    return np.asarray(flats), np.asarray(losses)


def flatten_losses_from_fn(fn, theta, fishers):
    import jax
    import jax.numpy as jnp
    from degeneracy_distillery.preprocessing_utils import flatten_with_numerical_jacobian

    def one(t, f):
        j = jax.jacfwd(fn)(t)
        return flatten_with_numerical_jacobian(j, f)

    flats = jax.vmap(one)(jnp.asarray(theta), jnp.asarray(fishers))
    losses = jax.vmap(_frobenius_to_eye)(flats)
    return np.asarray(flats), np.asarray(losses)


def _bin_median(x, y, bins=35, min_count=5):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return np.zeros(0), np.zeros(0)
    edges = np.linspace(np.nanmin(x), np.nanmax(x), int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(x, edges) - 1
    xb, yb = [], []
    for i in range(len(edges) - 1):
        m = idx == i
        if int(m.sum()) < int(min_count):
            continue
        xb.append(centers[i])
        yb.append(float(np.median(y[m])))
    return np.asarray(xb), np.asarray(yb)


def plot_flattening_comparison_analytic_binned(
    betas,
    X_test,
    true_fs,
    mdl_coords,
    frob_coords,
    dy_sr_test,
    x_to_xprime_fn=None,
    ad_hoc_coords=None,
    beta_bins=35,
    statistic="median",
    spread=None,
    min_count=5,
    aggregate_in_log_space=False,
    plot_vanilla=True,
    plot_adhoc=False,
    xlabel=r"$\beta$",
    ylabel=r"flatness of coordinates",
    figwidth=3.41,
    filename=None,
    ax=None,
    show=False,
    show_legend=True,
    n_d: int = N_D,
):
    """Left-panel flattening vs true beta. Scores use the analytic Fisher.

    Geodesic coordinates flatten the per-observation metric, so they are
    scored with ``true_fs / n_d``. Neural, SR, and raw theta use ``true_fs``.
    """
    import matplotlib.pyplot as plt

    del statistic, spread, aggregate_in_log_space
    x_to_xprime_fn = x_to_xprime if x_to_xprime_fn is None else x_to_xprime_fn
    betas = np.asarray(betas, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    true_fs = np.asarray(true_fs, dtype=np.float64)
    dy_sr_test = np.asarray(dy_sr_test, dtype=np.float64)
    n = X_test.shape[0]
    j_eye = np.broadcast_to(np.eye(2), (n, 2, 2))

    style = {
        "vanilla": {"label": r"$\theta$", "color": "#7C2A2A", "marker": "o"},
        "mdl": {"label": "MDL SR", "color": "#2A9D8F", "marker": "s"},
        "frob": {"label": "Frob loss SR", "color": "#0F766E", "marker": "D"},
        "adhoc": {"label": "ad-hoc", "color": "#94A3B8", "marker": "o"},
        "nn": {"label": "NN", "color": "#E76F51", "marker": "^"},
        "geo": {"label": "geodesic normal", "color": "#264653", "marker": "P"},
    }

    _, vanilla_losses = flatten_losses_from_jacobian(j_eye, true_fs)
    _, nn_losses = flatten_losses_from_jacobian(dy_sr_test, true_fs)
    _, geo_losses = flatten_losses_from_fn(
        x_to_xprime_fn, X_test, true_fs / float(n_d),
    )
    scores = {
        "vanilla": float(np.mean(vanilla_losses)),
        "nn": float(np.mean(nn_losses)),
        "geo": float(np.mean(geo_losses)),
    }

    series = []
    if plot_vanilla:
        series.append(("vanilla", vanilla_losses))
    series.append(("nn", nn_losses))

    for key, coords in (("mdl", mdl_coords), ("frob", frob_coords)):
        if not coords:
            continue
        try:
            _, losses = flatten_losses_from_fn(
                _compile_coord_stack(list(coords)), X_test, true_fs,
            )
        except Exception as exc:
            print(f"[plot] {key} flatten failed: {exc}", flush=True)
            continue
        scores[key] = float(np.mean(losses))
        series.append((key, losses))
    if plot_adhoc and ad_hoc_coords:
        try:
            _, adhoc_losses = flatten_losses_from_fn(
                _compile_coord_stack(list(ad_hoc_coords)), X_test, true_fs,
            )
            scores["adhoc"] = float(np.mean(adhoc_losses))
            series.append(("adhoc", adhoc_losses))
        except Exception as exc:
            print(f"[plot] adhoc flatten failed: {exc}", flush=True)
    series.append(("geo", geo_losses))

    print("\nchecking against analytic Fisher\n---", flush=True)
    print(f"vanilla coords (theta) flattening score: {scores['vanilla']}", flush=True)
    print(f"neural network flattening score: {scores['nn']}", flush=True)
    if "mdl" in scores:
        print(f"MDL SR flattening score: {scores['mdl']}", flush=True)
    if "frob" in scores:
        print(f"Frob SR flattening score: {scores['frob']}", flush=True)
    print(f"numerical geo flattening score: {scores['geo']}", flush=True)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(figwidth, figwidth * 0.82))
    else:
        fig = ax.figure

    handles = []
    for key, losses in series:
        xb, yb = _bin_median(betas, losses, bins=beta_bins, min_count=min_count)
        if xb.size == 0:
            continue
        s = style[key]
        h = ax.scatter(
            xb, yb, label=s["label"], c=s["color"], marker=s["marker"],
            s=30, alpha=0.95, edgecolors="white", linewidths=0.4, zorder=3,
        )
        ax.plot(xb, yb, color=s["color"], linewidth=1.2, alpha=0.9, zorder=2)
        handles.append(h)

    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if show_legend and handles:
        ax.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28),
            ncol=min(3, len(handles)), frameon=False, fontsize=8,
            handlelength=1.2, columnspacing=1.0,
        )
    if created:
        fig.subplots_adjust(left=0.17, right=0.98, top=0.90, bottom=0.28)
    if filename is not None:
        fig.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"saved {filename}", flush=True)
    if show:
        plt.show()
    return fig, ax, scores


def _compile_coord_stack(exprs: list[str]):
    import jax.numpy as jnp
    import sympy
    from degeneracy_distillery.sr_utils import _parse_sr_equation, safe_lambdify

    fns = []
    n_params = 2
    for expr in exprs:
        expr_s, _, _, pars = _parse_sr_equation(expr)
        xs = list(sympy.symbols(" ".join(f"X{i + 1}" for i in range(n_params)), real=True))
        bs = list(sympy.symbols([f"b{i}" for i in range(len(pars))], real=True))
        fn = safe_lambdify(bs + xs, expr_s, ["jax"])

        def _one(t, _fn=fn, _pars=pars):
            val = _fn(*_pars, *t)
            return jnp.atleast_1d(jnp.asarray(val)).reshape(-1)

        fns.append(_one)

    def coord_fn(t):
        t = jnp.asarray(t)
        return jnp.concatenate([fn(t) for fn in fns])

    return coord_fn
