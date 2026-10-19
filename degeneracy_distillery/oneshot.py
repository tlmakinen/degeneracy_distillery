"""One-step map: joint eta_phi / eta_psi fit and a warm-started rank ladder.

The loss is

    L = E[ 0.5 ||eta_phi(theta) - eta_psi(x)||^2 - 0.5 log det(J J^T) ]
      + (m/2) log 2 pi

with J = d eta_phi / d theta.  The map carries an explicit mixing matrix A so
an axis can be projected out without re-initialising the trunk::

    eta_m(theta) = s * (A_m @ g_phi(theta)),   A_m in R^{m x m0}.

``whittle_ladder`` is two stages. The probe is low-rank discovery: it only
chooses ``r_hat``, either by the NLL drop test or (``rank_rule="info"``) by
counting probe axes with held-out information above a floor. The production
map is then a fresh network at
``m_fit`` latents (``r_hat``, or ``r_hat + 1`` when the probe spectrum has no
clean cutoff), trained from random init with ``m0 = m_fit``. No projected
probe weights survive into it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax
import numpy as np

# Apple Metal does not compile float64. CUDA and CPU do, and the paper
# uses x64. Skip the flag when Metal is the only device.
if not any(d.platform.upper() == "METAL" for d in jax.devices()):
    jax.config.update("jax_enable_x64", True)


SCALE_BOOST = 8.0
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 512


def _bound_hyper(v):
    """Flax hyperparam form of a scalar or per-coordinate bound."""
    a = np.asarray(v, dtype=float)
    if a.size == 1:
        return float(a.reshape(()))
    return tuple(float(x) for x in np.ravel(a))


class Flattener(nn.Module):
    """theta -> g in R^{m0}. Skip is the first m0 standardised coords of theta.

    ``use_log_features`` concatenates ``log(theta)``. Set it false when any
    coordinate can be non-positive (Rosenbrock lives in ``[-3, 3]``).
    """

    m: int
    lo: Union[float, tuple]
    hi: Union[float, tuple]
    features: Sequence[int] = (128, 128, 128)
    use_log_features: bool = True
    z_lo: float = -1.0
    z_hi: float = 1.0
    #: Init for the MLP branch. "zeros" starts eta as *exactly* the linear
    #: skip, so the map begins life linear and only departs as the loss
    #: pushes it. "glorot" starts the branch active. The usual argument for
    #: zeros -- a well-conditioned Jacobian for the -0.5 logdet(JJ^T) term --
    #: is carried by the skip, not the zeros: measured at init, cond(J) is
    #: 1.000 with zeros and 1.179 with glorot, and logdet is 9.704 vs 9.901.
    #: So zeros is a prior toward linear solutions, not a numerical need.
    delta_init: str = "zeros"

    @nn.compact
    def __call__(self, theta):
        lo = jnp.asarray(self.lo)
        hi = jnp.asarray(self.hi)
        u = (theta - lo) / (hi - lo)
        z = self.z_lo + (self.z_hi - self.z_lo) * u
        h = jnp.concatenate([z, jnp.log(theta)]) if self.use_log_features else z
        for w in self.features:
            h = nn.gelu(nn.Dense(w)(h))
        if self.delta_init == "zeros":
            d_kernel = nn.initializers.zeros
        elif self.delta_init == "glorot":
            d_kernel = nn.initializers.glorot_uniform()
        else:
            raise ValueError(f"delta_init must be zeros|glorot, got {self.delta_init!r}")
        delta = nn.Dense(
            self.m,
            kernel_init=d_kernel,
            bias_init=nn.initializers.zeros,
        )(h)
        skip = nn.Dense(
            self.m,
            use_bias=False,
            kernel_init=lambda k, s, dt: jnp.eye(s[0], s[1], dtype=dt),
        )(z)
        return skip + delta


class Estimator(nn.Module):
    """x -> hat_g in R^{g_dim}. Output lives in the same space as g_phi.

    ``act`` is a field so an ensemble can draw a different nonlinearity per
    member. Fishnets varies width, depth and activation together; varying
    only width leaves members far more correlated than the Fishnets
    ensemble whose spread ``y_std`` is meant to match.
    """

    m: int
    features: Sequence[int] = (256, 256, 128)
    act: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        h = x
        for w in self.features:
            h = self.act(nn.Dense(w)(h))
        out = nn.Dense(
            self.m,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h)
        return out + nn.Dense(self.m, kernel_init=nn.initializers.normal(1e-2))(x)


def _mish(x):
    return x * jnp.tanh(nn.softplus(x))


def _smooth_leaky(x, alpha: float = 0.2):
    """C-infinity leaky ReLU; the ``optimized_smooth_leaky`` of ``fishnets``.

    Not in the default bank. As ``x -> -inf``, ``log_sigmoid(x) -> x``, so this
    grows like ``(1 - alpha) x**2``. Stacked four deep that compounds to about
    ``x**16`` and diverges. It survives a bounded loss but not the unbounded
    ``-log|det J|`` of the square-completed objective.
    """
    return alpha * x + (1.0 - alpha) * jax.nn.log_sigmoid(x) * x


#: Activation bank for the eta(x) trunk, mirroring the Fishnets ensemble.
#: Safe to include kinked functions here: the volume term differentiates
#: eta_phi(theta) only, so eta_psi(x) never needs a smooth Jacobian.
EST_ACTS: dict[str, Callable] = {
    "gelu": nn.gelu,
    "swish": nn.swish,
    "relu": nn.relu,
    "leaky_relu": nn.leaky_relu,
    "mish": _mish,
    "smooth_leaky": _smooth_leaky,
    "tanh": nn.tanh,
}


def estimator_factory_for(features: Sequence[int], act: str = "gelu"):
    """One eta(x) factory with a fixed architecture, tagged for logging."""
    feats = tuple(int(w) for w in features)
    if act not in EST_ACTS:
        raise KeyError(f"unknown activation {act!r}; have {sorted(EST_ACTS)}")
    fn = EST_ACTS[act]

    def factory(m: int):
        return Estimator(m=int(m), features=feats, act=fn)

    factory.features = feats  # type: ignore[attr-defined]
    factory.act_name = act  # type: ignore[attr-defined]
    factory.tag = f"{act}:{'x'.join(str(w) for w in feats)}"  # type: ignore[attr-defined]
    return factory


def estimator_bank(
    k: int,
    seed: int = 0,
    *,
    hids: tuple[int, int] = (64, 320),
    n_layers: tuple[int, int] = (2, 4),
    acts: Optional[Sequence[str]] = None,
    first: Optional[Sequence[int]] = None,
) -> list:
    """``k`` eta(x) factories with sampled width, depth and activation.

    This is the one-step analogue of the Fishnets ensemble draw in
    ``training_loop_fishnets``: width uniform in ``hids``, depth uniform in
    ``n_layers``, activation uniform over ``acts``. Diversity in the data
    map is what makes ``y_std`` an honest spread rather than seed noise, so
    the SR step's MDL weights mean something.

    ``first`` pins member 0 to a given width tuple (with the first
    activation) so the ladder map can stay warm-started into the ensemble.
    """
    k = int(k)
    if acts is None:
        acts = ("gelu", "swish", "relu", "leaky_relu", "mish", "tanh")
    acts = tuple(acts)
    rng = np.random.default_rng(int(seed) + 7717)
    out = []
    for i in range(k):
        if i == 0 and first is not None:
            out.append(estimator_factory_for(first, acts[0]))
            continue
        depth = int(rng.integers(n_layers[0], n_layers[1] + 1))
        width = int(rng.integers(hids[0], hids[1]))
        a = str(acts[int(rng.integers(0, len(acts)))])
        out.append(estimator_factory_for((width,) * depth, a))
    return out


def _core(p, theta, *, flat_mod, coord_fn, scale_boost):
    g = coord_fn(theta) if coord_fn is not None else flat_mod.apply(p["flat"], theta)
    mixed = p["A"] @ g
    return jnp.exp(scale_boost * p["raw_log_scale"]) * mixed


def _hat(p, x, *, est_mod, scale_boost):
    h = est_mod.apply(p["est"], x)
    mixed = p["A"] @ h
    return jnp.exp(scale_boost * p["raw_log_scale"]) * mixed


@dataclass
class OneShotFit:
    params: Any
    flat_mod: Optional[Flattener]
    est_mod: Estimator
    m: int
    m0: int
    g_dim: int
    const: float
    best_val: float
    scale_boost: float = SCALE_BOOST
    coord_fn: Optional[Callable] = None
    lo: float = 1.0
    hi: float = 2.0
    use_log_features: bool = True

    def _core(self, theta):
        return _core(
            self.params, theta,
            flat_mod=self.flat_mod, coord_fn=self.coord_fn,
            scale_boost=self.scale_boost,
        )

    def _hat(self, x):
        return _hat(
            self.params, x, est_mod=self.est_mod, scale_boost=self.scale_boost,
        )

    def eta(self, theta):
        fn = jax.jit(lambda t: jax.vmap(self._core)(t))
        return np.asarray(fn(jnp.asarray(theta)))

    def eta_hat(self, x):
        fn = jax.jit(lambda y: jax.vmap(self._hat)(y))
        return np.asarray(fn(jnp.asarray(x)))

    def jac(self, theta):
        fn = jax.jit(lambda t: jax.vmap(jax.jacfwd(self._core))(t))
        return np.asarray(fn(jnp.asarray(theta)))

    def nll_vec(self, theta, x):
        eye = jnp.eye(self.m)

        def one(t, y):
            r = self._core(t) - self._hat(y)
            J = jax.jacfwd(self._core)(t)
            vol = 0.5 * jnp.linalg.slogdet(J @ J.T + 1e-10 * eye)[1]
            return 0.5 * jnp.sum(r ** 2) - vol + self.const

        fn = jax.jit(lambda T, Y: jax.vmap(one)(T, Y))
        return np.asarray(fn(jnp.asarray(theta), jnp.asarray(x)))

    def z_vec(self, theta, x):
        fn = jax.jit(lambda T, Y: jax.vmap(lambda t, y: self._core(t) - self._hat(y))(T, Y))
        return np.asarray(fn(jnp.asarray(theta), jnp.asarray(x)))


def spectrum(eta: np.ndarray):
    """Prior covariance spectrum of eta. Returns (rotated, lam, V) descending."""
    e = np.asarray(eta, dtype=np.float64)
    e = e - e.mean(0)
    m = e.shape[1]
    C = np.cov(e, rowvar=False).reshape(m, m)
    lam, V = np.linalg.eigh(C)
    o = np.argsort(-lam)
    return e @ V[:, o], np.maximum(lam[o], 1e-12), V[:, o]


def r2(a, b) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size < 2 or a.std() < 1e-15 or b.std() < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def spearman(a, b) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size < 2:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def jtj_eigengap(J: np.ndarray, min_gap: float = 10.0) -> dict:
    """Per-sample eigengap rank of F = J^T J. Independent of the ladder rule."""
    J = np.asarray(J, dtype=np.float64)
    F = np.einsum("nmi,nmj->nij", J, J)
    F = 0.5 * (F + np.swapaxes(F, -1, -2))
    eigvals = np.linalg.eigvalsh(F)[:, ::-1]
    lead = eigvals[:, :1]
    usable = np.isfinite(eigvals).all(axis=1) & (lead[:, 0] > 0.0)
    if not usable.any():
        raise RuntimeError("no sample has a positive finite J^T J eigenvalue")
    relative = np.abs(eigvals[usable]) / np.maximum(lead[usable], 1e-300)
    median_relative = np.median(relative, axis=0)
    n_dim = int(median_relative.size)
    gap_ratios = (
        median_relative[:-1] / np.maximum(median_relative[1:], 1e-300)
        if n_dim > 1 else np.zeros(0)
    )
    if gap_ratios.size == 0:
        rank = n_dim
    else:
        idx = int(np.argmax(gap_ratios))
        rank = idx + 1 if float(gap_ratios[idx]) >= min_gap else n_dim
    return {
        "rank": int(max(1, min(rank, n_dim))),
        "median_relative_spectrum": median_relative,
        "gap_ratios": gap_ratios,
    }


def project_params(params: Any, n_keep: int, eta: np.ndarray, scale_boost: float = SCALE_BOOST):
    """Rotate A into the canonical basis and drop the lowest-lambda axes."""
    _, lam, V = spectrum(eta)
    A_old = np.asarray(params["A"], dtype=np.float64)
    s = np.exp(scale_boost * np.asarray(params["raw_log_scale"], dtype=np.float64))
    A_eff = s[:, None] * A_old
    n_keep = max(1, min(int(n_keep), A_eff.shape[0]))
    A_new = V[:, :n_keep].T @ A_eff
    new_p = dict(params)
    new_p["A"] = jnp.asarray(A_new)
    new_p["raw_log_scale"] = jnp.zeros(n_keep)
    return new_p, lam, V


def refit_rank(
    r_hat: int,
    lam_probe: np.ndarray,
    cap: int,
    min_gap: float = 10.0,
    pad: Optional[int] = None,
) -> int:
    """Latent width for the fresh refit.

    ``pad=None`` adds one axis when the probe latent spectrum drops by less
    than ``min_gap`` between axis ``r_hat`` and ``r_hat + 1``. An integer
    ``pad`` is used as given. The result never exceeds ``cap``.
    """
    r_hat = int(r_hat)
    lam = np.asarray(lam_probe, dtype=np.float64)
    if pad is None:
        pad = 0
        if r_hat < lam.size:
            ratio = lam[r_hat - 1] / max(lam[r_hat], 1e-300)
            if ratio < float(min_gap):
                pad = 1
    return int(max(1, min(r_hat + int(pad), int(cap))))


def train_oneshot(
    th_fit,
    x_fit,
    th_val,
    x_val,
    m: int,
    m0: int,
    lo: float,
    hi: float,
    steps: int,
    seed: int,
    init_params: Any = None,
    coord_fn: Optional[Callable] = None,
    batch: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    scale_boost: float = SCALE_BOOST,
    verbose: bool = True,
    flat_features: Sequence[int] = (128, 128, 128),
    est_features: Sequence[int] = (256, 256, 128),
    use_log_features: bool = True,
    estimator_factory: Optional[Callable] = None,
    z_range: tuple[float, float] = (-1.0, 1.0),
    delta_init: str = "zeros",
) -> OneShotFit:
    """Joint fit of eta_phi and eta_psi. Warm-start via ``init_params``."""
    th_fit = jnp.asarray(th_fit)
    x_fit = jnp.asarray(x_fit)
    th_val = jnp.asarray(th_val)
    x_val = jnp.asarray(x_val)
    n_fit = int(th_fit.shape[0])
    bp = min(int(batch), n_fit)
    frozen = coord_fn is not None
    lo_h = _bound_hyper(lo)
    hi_h = _bound_hyper(hi)

    if frozen:
        g0 = coord_fn(th_fit[0])
        g_dim = int(np.asarray(g0).reshape(-1).size)
        flat_mod = None
    else:
        g_dim = int(m0)
        flat_mod = Flattener(
            m=g_dim, lo=lo_h, hi=hi_h, features=tuple(flat_features),
            use_log_features=bool(use_log_features),
            z_lo=float(z_range[0]), z_hi=float(z_range[1]),
            delta_init=str(delta_init),
        )

    if estimator_factory is not None:
        est_mod = estimator_factory(g_dim)
    else:
        est_mod = Estimator(m=g_dim, features=tuple(est_features))
    k0, k1 = jr.split(jr.PRNGKey(int(seed)), 2)

    if init_params is not None:
        p = dict(init_params)
        if "A" not in p or tuple(np.asarray(p["A"]).shape) != (m, g_dim):
            p["A"] = jnp.eye(m, g_dim)
        if "raw_log_scale" not in p or int(np.asarray(p["raw_log_scale"]).size) != m:
            p["raw_log_scale"] = jnp.zeros(m)
        if "est" not in p:
            p["est"] = est_mod.init(k1, x_fit[0])
        if (not frozen) and "flat" not in p:
            p["flat"] = flat_mod.init(k0, th_fit[0])
    else:
        p = {
            "est": est_mod.init(k1, x_fit[0]),
            "raw_log_scale": jnp.zeros(m),
            "A": jnp.eye(m, g_dim),
        }
        if not frozen:
            p["flat"] = flat_mod.init(k0, th_fit[0])

    const = 0.5 * m * np.log(2.0 * np.pi)
    eyeM = jnp.eye(m)

    def core(p, theta):
        return _core(p, theta, flat_mod=flat_mod, coord_fn=coord_fn, scale_boost=scale_boost)

    def hat(p, x):
        return _hat(p, x, est_mod=est_mod, scale_boost=scale_boost)

    def nll_vec(p, th_b, x_b):
        def one(theta, x):
            r = core(p, theta) - hat(p, x)
            J = jax.jacfwd(lambda t: core(p, t))(theta)
            vol = 0.5 * jnp.linalg.slogdet(J @ J.T + 1e-10 * eyeM)[1]
            return 0.5 * jnp.sum(r ** 2) - vol + const
        return jax.vmap(one)(th_b, x_b)

    nll_mean = jax.jit(lambda p, t, y: jnp.mean(nll_vec(p, t, y)))
    warm = max(1, min(500, max(steps, 1) // 10))
    sched = optax.warmup_cosine_decay_schedule(
        1e-5, lr, warm, max(steps, warm + 1), lr * 1e-2,
    )
    tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(sched))
    st = tx.init(p)

    @jax.jit
    def step(p, st, key):
        idx = jr.randint(key, (bp,), 0, n_fit)

        def obj(p):
            return jnp.mean(nll_vec(p, th_fit[idx], x_fit[idx]))

        l, g = jax.value_and_grad(obj)(p)
        upd, st = tx.update(g, st, p)
        new = optax.apply_updates(p, upd)
        ok = jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(z)) for z in jax.tree_util.tree_leaves(g)
        ]))
        return jax.tree_util.tree_map(lambda o, n: jnp.where(ok, n, o), p, new), st, l, ok

    key = jr.PRNGKey(int(seed) + 7)
    ev = max(1, min(200, max(steps, 1) // 50))
    best, best_val, skipped = p, np.inf, 0
    for i in range(int(steps)):
        key, sk = jr.split(key)
        p, st, l, ok = step(p, st, sk)
        skipped += int(not bool(ok))
        if i % ev == 0 or i == steps - 1:
            v = float(nll_mean(p, th_val, x_val))
            if verbose:
                ls = np.round(np.asarray(scale_boost * p["raw_log_scale"]), 2)
                print(
                    f"    step {i:6d}  train {float(l): .4f}  val {v: .4f}  "
                    f"log_scale {ls}",
                    flush=True,
                )
            if np.isfinite(v) and v < best_val:
                best_val, best = v, p
    if skipped and verbose:
        print(f"    (skipped {skipped}/{steps} non-finite updates)", flush=True)

    return OneShotFit(
        params=best, flat_mod=flat_mod, est_mod=est_mod,
        m=int(m), m0=int(m0), g_dim=int(g_dim), const=float(const),
        best_val=float(best_val), scale_boost=float(scale_boost),
        coord_fn=coord_fn, lo=lo_h, hi=hi_h,
        use_log_features=bool(use_log_features),
    )


def info_nats(eta: np.ndarray, eta_hat: np.ndarray) -> np.ndarray:
    """Per-axis information, ``0.5 log(prior var / residual var)``, descending.

    Both variances are taken in the prior PCA basis of ``eta``. This is the
    one-step loss of each axis measured against the no-data null (``eta_hat``
    replaced by the prior mean), so an axis the data cannot predict scores 0.
    """
    rot, lam, V = spectrum(eta)
    hat_rot = (np.asarray(eta_hat, dtype=np.float64) - np.asarray(eta).mean(0)) @ V
    resid = np.mean((rot - hat_rot) ** 2, axis=0)
    return 0.5 * np.log(lam / np.maximum(resid, 1e-12))


def _rung_record(fit: OneShotFit, th, x, extra: Optional[dict] = None) -> dict:
    eta = fit.eta(th)
    hat = fit.eta_hat(x)
    nll = fit.nll_vec(th, x)
    J = fit.jac(th)
    _, lam, _ = spectrum(eta)
    vol = 0.5 * np.asarray([
        np.linalg.slogdet(j @ j.T + 1e-12 * np.eye(j.shape[0]))[1] for j in J
    ])
    rec = {
        "m": int(fit.m),
        "best_val": float(fit.best_val),
        "nll_mean": float(nll.mean()),
        "nll_vec": nll,
        "lam": lam,
        "nats": 0.5 * np.log(np.maximum(lam, 1e-12)),
        "info_nats": info_nats(eta, hat),
        "residual_var": np.asarray(((eta - hat) ** 2).mean(0)),
        "median_sqrt_det_JJT": float(np.exp(np.median(vol))),
        "eta": eta,
        "J": J,
        "fit": fit,
    }
    if extra:
        rec.update(extra)
    return rec


def whittle_ladder(
    th_fit,
    x_fit,
    th_val,
    x_val,
    lo: float,
    hi: float,
    m_probe: int = 4,
    probe_steps: int = 20000,
    rung_steps: int = 5000,
    seed: int = 0,
    ladder_k: float = 2.0,
    cold_rungs: bool = False,
    refit_at_rhat: bool = True,
    refit_steps: Optional[int] = None,
    refit_pad: Optional[int] = None,
    rank_rule: str = "nll",
    info_floor: float = 0.5,
    batch: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    scale_boost: float = SCALE_BOOST,
    verbose: bool = True,
    min_gap: float = 10.0,
    use_log_features: bool = True,
    estimator_factory: Optional[Callable] = None,
    flat_features: Sequence[int] = (128, 128, 128),
    est_features: Sequence[int] = (256, 256, 128),
    z_range: tuple[float, float] = (-1.0, 1.0),
    delta_init: str = "zeros",
) -> dict:
    """Discover ``r_hat``, then refit a fresh network at ``m_fit`` latents.

    The descent (warm projection, or ``cold_rungs``) only chooses ``r_hat``.
    Unless ``refit_at_rhat`` is off, the production ``fit`` is trained from
    random init with ``m = m0 = m_fit``, where ``m_fit`` comes from
    ``refit_rank`` on the probe latent spectrum. ``refit_at_rhat=False``
    returns the projected discovery map and exists only for ablation.
    ``jtj_gap_screen`` is the discovery map; ``jtj_gap`` is the returned map.

    ``rank_rule="nll"`` drops axes while held-out NLL does not rise by more
    than ``ladder_k`` standard errors. That compares densities of different
    dimension, so a spare axis on a narrow prior box still lowers NLL and the
    rule overcounts. ``rank_rule="info"`` skips the descent and counts probe
    axes whose held-out information (:func:`info_nats`) exceeds ``info_floor``.
    """
    if rank_rule not in ("nll", "info"):
        raise ValueError(f"rank_rule must be 'nll' or 'info', got {rank_rule!r}")
    if rank_rule == "info" and not refit_at_rhat:
        raise ValueError("rank_rule='info' has no descent map; it needs refit_at_rhat=True")
    m_probe = int(m_probe)
    train_kw = dict(
        lo=lo, hi=hi, batch=batch, lr=lr, scale_boost=scale_boost,
        verbose=verbose, use_log_features=use_log_features,
        estimator_factory=estimator_factory,
        flat_features=flat_features, est_features=est_features,
        z_range=z_range, delta_init=delta_init,
    )
    if verbose:
        print(f"[ladder] probe m={m_probe}  steps={probe_steps}", flush=True)
    current = train_oneshot(
        th_fit, x_fit, th_val, x_val,
        m=m_probe, m0=m_probe, steps=probe_steps, seed=seed, **train_kw,
    )
    rungs = [_rung_record(current, th_val, x_val)]
    if verbose:
        print(
            f"  m={m_probe}  nll={rungs[-1]['nll_mean']:.4f}  "
            f"lam={np.round(rungs[-1]['lam'], 3)}",
            flush=True,
        )

    r_hat = m_probe
    if rank_rule == "info":
        info = rungs[0]["info_nats"]
        # When m_probe > d the spare axes collapse (lam ~ 0) and their info
        # is a ratio of two near-zero variances. eta has at most d
        # independent directions, so cap there.
        d_theta = int(np.asarray(th_fit).shape[-1])
        r_hat = int(max(1, min(np.sum(info > float(info_floor)), d_theta)))
        if verbose:
            print(
                f"[ladder] info rule  nats={np.round(info, 3)}  "
                f"floor={info_floor}  -> r_hat={r_hat}",
                flush=True,
            )
            if r_hat >= m_probe:
                print("[ladder] WARNING: r_hat reached m_probe; raise --m-probe", flush=True)
    for m in range(m_probe - 1, 0, -1):
        if rank_rule == "info":
            break
        nll_m = rungs[-1]["nll_vec"]
        if cold_rungs:
            if verbose:
                print(f"[ladder] cold refit m={m}  steps={rung_steps}", flush=True)
            nxt = train_oneshot(
                th_fit, x_fit, th_val, x_val,
                m=m, m0=m, steps=rung_steps, seed=seed + 11 + m, **train_kw,
            )
            lam, V = rungs[-1]["lam"], None
        else:
            eta_prior = current.eta(th_fit)
            p_new, lam, V = project_params(current.params, m, eta_prior, scale_boost)
            if verbose:
                print(
                    f"[ladder] drop lowest axis -> m={m}  "
                    f"lam={np.round(lam, 3)}  steps={rung_steps}",
                    flush=True,
                )
            nxt = train_oneshot(
                th_fit, x_fit, th_val, x_val,
                m=m, m0=current.m0, steps=rung_steps, seed=seed + 11 + m,
                init_params=p_new, **train_kw,
            )

        rec = _rung_record(nxt, th_val, x_val)
        nll_mm1 = rec["nll_vec"]
        dif = nll_mm1 - nll_m
        mean_dif = float(dif.mean())
        se = float(dif.std(ddof=1) / np.sqrt(len(dif))) if len(dif) > 1 else np.inf
        accept = bool(np.isfinite(mean_dif) and mean_dif <= ladder_k * se)
        rec["dif_mean"] = mean_dif
        rec["dif_se"] = se
        rec["accepted"] = accept
        rec["parent_lam"] = lam
        rungs.append(rec)
        if verbose:
            print(
                f"  m={m}  nll={rec['nll_mean']:.4f}  "
                f"dif={mean_dif:+.4f} +- {se:.4f}  "
                f"accept={accept}",
                flush=True,
            )
        if not accept:
            r_hat = current.m
            break
        current = nxt
        r_hat = m

    gap_screen = jtj_eigengap(current.jac(th_val), min_gap=min_gap)
    n_refit = int(probe_steps if refit_steps is None else refit_steps)
    m_fit = int(r_hat)
    if refit_at_rhat:
        d = int(np.asarray(th_fit).shape[-1])
        m_fit = refit_rank(
            r_hat, rungs[0]["lam"], cap=min(m_probe, d),
            min_gap=min_gap, pad=refit_pad,
        )
        if verbose:
            print(
                f"[ladder] fresh refit at m={m_fit} (r_hat={r_hat})  "
                f"steps={n_refit}  probe lam={np.round(rungs[0]['lam'], 3)}  "
                f"(screen J^T J rank={gap_screen['rank']})",
                flush=True,
            )
        current = train_oneshot(
            th_fit, x_fit, th_val, x_val,
            m=m_fit, m0=m_fit, steps=n_refit, seed=int(seed) + 97,
            **train_kw,
        )
        rungs.append(_rung_record(
            current, th_val, x_val, extra={"refit": True, "m_fit": m_fit},
        ))

    gap = jtj_eigengap(current.jac(th_val), min_gap=min_gap)
    if verbose:
        print(
            f"[ladder] r_hat={r_hat}  J^T J eigengap rank={gap['rank']}"
            + (
                f"  (screen {gap_screen['rank']})"
                if refit_at_rhat else ""
            )
            + f"  refit info={np.round(rungs[-1]['info_nats'], 3)}",
            flush=True,
        )
    return {
        "r_hat": int(r_hat),
        "m_fit": int(current.m),
        "fit": current,
        "rungs": rungs,
        "jtj_gap": gap,
        "jtj_gap_screen": gap_screen,
        "refit_at_rhat": bool(refit_at_rhat),
        "rank_rule": rank_rule,
        "info_probe": np.asarray(rungs[0]["info_nats"]),
        "info_final": np.asarray(rungs[-1]["info_nats"]),
    }


def fit_ensemble(
    th_fit,
    x_fit,
    th_val,
    x_val,
    m: int,
    lo: float,
    hi: float,
    n_members: int,
    steps: int,
    seed: int,
    init_fit: Optional[OneShotFit] = None,
    bootstrap: bool = True,
    batch: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    verbose: bool = True,
    use_log_features: bool = True,
    estimator_factory: Optional[Callable] = None,
) -> dict:
    """Fit ``K`` one-step maps at fixed ``m`` for an aligned ensemble.

    The ladder map can be member 0 via ``init_fit``. The rest train on
    bootstrap resamples of ``(th_fit, x_fit)``, warm-started from that map
    when it is given. Weights are ``1 / exp(held-out NLL)``.
    """
    n_members = max(2, int(n_members))
    if init_fit is not None and int(init_fit.m) != int(m):
        raise ValueError(
            f"init_fit has m={init_fit.m} but the ensemble asks for m={m}; "
            "pass m=init_fit.m so members are not projected"
        )
    rng = np.random.default_rng(int(seed) + 41)
    n = int(np.asarray(th_fit).shape[0])
    fits: list[OneShotFit] = []
    nlls: list[float] = []
    init_params = None if init_fit is None else init_fit.params
    m0 = int(m if init_fit is None else init_fit.m0)

    start = 0
    if init_fit is not None:
        if verbose:
            print(f"[ensemble] member 0 = ladder map (m={m})", flush=True)
        fits.append(init_fit)
        nlls.append(float(np.mean(init_fit.nll_vec(th_val, x_val))))
        start = 1

    for k in range(start, n_members):
        if bootstrap:
            idx = rng.integers(0, n, size=n)
            th_k = np.asarray(th_fit)[idx]
            x_k = np.asarray(x_fit)[idx]
        else:
            th_k, x_k = th_fit, x_fit
        if verbose:
            print(f"[ensemble] member {k + 1}/{n_members}  steps={steps}", flush=True)
        fit = train_oneshot(
            th_k, x_k, th_val, x_val,
            m=m, m0=m0, lo=lo, hi=hi,
            steps=steps, seed=int(seed) + 1009 * (k + 1),
            init_params=init_params,
            batch=batch, lr=lr, verbose=verbose,
            use_log_features=use_log_features,
            estimator_factory=estimator_factory,
        )
        fits.append(fit)
        nlls.append(float(np.mean(fit.nll_vec(th_val, x_val))))

    nlls_a = np.asarray(nlls, dtype=np.float64)
    w = np.exp(-np.clip(nlls_a, -50.0, 50.0))
    w = w / np.maximum(w.sum(), 1e-12)
    if verbose:
        print(
            f"[ensemble] held-out NLL {np.round(nlls_a, 3)}  "
            f"weights {np.round(w, 3)}",
            flush=True,
        )
    return {"fits": fits, "weights": w, "nlls": nlls_a}
