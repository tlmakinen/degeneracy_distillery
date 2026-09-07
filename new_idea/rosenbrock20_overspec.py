"""Does an over-specified m recover the true coordinates in one shot?

Fits m = 2, 3, 5, 8 on the d=20 uniform-prior Rosenbrock + 18 nuisance problem
and asks, for each: (a) what does it cost in NLL, (b) does the information
spectrum lambda_i = Var_pi(eta_i) isolate r = 2 with lambda = 1 on the rest,
(c) do the top-2 spectral directions span the true coordinates, (d) are the
discarded directions transports of the spurious screened coordinates.
"""
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import flax.linen as nn
import optax
from typing import Sequence, Callable, Optional

jax.config.update("jax_enable_x64", True)

D, NSIMS, NTEST = 20, 32000, 20000
SD, N_REP = (0.25, 0.5), 8
STEPS, SCREEN_STEPS = 8000, 3000
BATCH_PAIR, BATCH_AUG = 256, 512
LR, SCALE_BOOST, SEED = 1e-3, 2.0, 0
L1_BASE, L1_SCREEN = 2.0, 1e-2
M_TEST = (2, 3, 5, 8)

A_BOX = 3.0
WIDTH = 2.0 * A_BOX
LOGW = float(np.log(WIDTH))
sig = np.asarray(SD) / np.sqrt(N_REP)
gain = np.sort(LOGW - 0.5 * np.log(2 * np.pi * sig ** 2) - 0.5)[::-1]
def oracle(m, d=D):
    return d * LOGW - gain[:min(m, len(gain))].sum()

_r = np.random.default_rng(1000 + D)
I0, I1 = (int(v) for v in _r.choice(D, size=2, replace=False))
print(f"hidden informative coordinates: theta_{I0}, theta_{I1}")
print(f"floor at m>=2: {oracle(2):.4f}   (m=1: {oracle(1):.4f})")
# true information spectrum of the exact solution
lam_true = np.array([3.0 / sig[0] ** 2, (3.0 + 7.2) / sig[1] ** 2])
print(f"predicted lambda for the informative pair: {np.round(lam_true, 1)}"
      f"  -> 0.5 log lambda = {np.round(0.5 * np.log(lam_true), 3)} nats")

def sim(n, rng):
    th = rng.uniform(-A_BOX, A_BOX, size=(n, D))
    mu = np.stack([th[:, I0], th[:, I1] - th[:, I0] ** 2], axis=1)
    return th, mu[:, None, :] + rng.normal(size=(n, N_REP, 2)) * np.asarray(SD)

rng = np.random.default_rng(SEED)
th_np, x_np = sim(NSIMS, rng)
th_te_np, x_te_np = sim(NTEST, np.random.default_rng(7777))
mu_x, sd_x = x_np.reshape(-1, 2).mean(0), x_np.reshape(-1, 2).std(0) + 1e-12
TH = jnp.asarray(th_np);      X = jnp.asarray((x_np - mu_x) / sd_x)
TH_TE = jnp.asarray(th_te_np); X_TE = jnp.asarray((x_te_np - mu_x) / sd_x)
XB = jnp.asarray((x_np.mean(1) - mu_x) / sd_x)


class Flattener(nn.Module):
    m: int
    skip_init: np.ndarray
    hidden: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, theta):
        h = nn.gelu(nn.Dense(self.hidden[0], name="in_proj")(theta))
        for k, w in enumerate(self.hidden[1:]):
            h = nn.gelu(nn.Dense(w, name=f"h{k}")(h))
        delta = nn.Dense(self.m, name="out", kernel_init=nn.initializers.zeros,
                         bias_init=nn.initializers.zeros)(h)
        skip = nn.Dense(self.m, use_bias=False, name="skip",
                        kernel_init=lambda k, s, dt: jnp.asarray(self.skip_init, dt))(theta)
        return skip + delta


class Estimator(nn.Module):
    m: int
    hidden: Sequence[int] = (32, 32)

    @nn.compact
    def __call__(self, x):
        h = x.mean(0)
        for w in self.hidden:
            h = nn.gelu(nn.Dense(w)(h))
        return (nn.Dense(self.m, kernel_init=nn.initializers.zeros,
                         bias_init=nn.initializers.zeros)(h)
                + nn.Dense(self.m)(x.mean(0)))


class Screen(nn.Module):
    hidden: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, theta):
        h = nn.gelu(nn.Dense(self.hidden[0], name="in_proj")(theta))
        for k, w in enumerate(self.hidden[1:]):
            h = nn.gelu(nn.Dense(w, name=f"h{k}")(h))
        return nn.Dense(2, name="out")(h)


def screen_inputs(th, xb, lam=L1_SCREEN, steps=SCREEN_STEPS, seed=0):
    mod = Screen()
    ps = mod.init(jr.PRNGKey(seed), th[0])
    tx = optax.adam(3e-3); st = tx.init(ps); n = th.shape[0]
    b = min(512, n)

    @jax.jit
    def one(ps, st, key):
        i = jr.randint(key, (b,), 0, n)
        def obj(ps):
            pr = jax.vmap(lambda t: mod.apply(ps, t))(th[i])
            K = ps["params"]["in_proj"]["kernel"]
            return (jnp.mean((pr - xb[i]) ** 2)
                    + lam * jnp.sum(jnp.sqrt(jnp.sum(K ** 2, 1) + 1e-12)))
        l, g = jax.value_and_grad(obj)(ps)
        u, st = tx.update(g, st, ps)
        return optax.apply_updates(ps, u), st, l

    key = jr.PRNGKey(seed + 1)
    for _ in range(steps):
        key, sk = jr.split(key)
        ps, st, l = one(ps, st, sk)
    return np.sqrt((np.asarray(ps["params"]["in_proj"]["kernel"]) ** 2).sum(1))


def fit_joint(M, th_fit, x_fit, th_val, x_val, active, steps=STEPS, seed=SEED,
              coord_fn: Optional[Callable] = None, d=D):
    n_fit = th_fit.shape[0]
    bp = min(BATCH_PAIR, n_fit)
    act = jnp.asarray(np.asarray(active, dtype=np.int32))
    Q0 = np.zeros((d, M)); Q0[np.asarray(active, dtype=int), np.arange(M)] = 1.0
    frozen = coord_fn is not None
    flat_mod = None if frozen else Flattener(m=M, skip_init=Q0)
    est_mod = Estimator(m=M)
    k0, k1 = jr.split(jr.PRNGKey(seed), 2)
    p = {"est": est_mod.init(k1, x_fit[0]), "raw_log_scale": jnp.zeros(M)}
    if frozen:
        p["A"] = jnp.eye(M)
    else:
        p["flat"] = flat_mod.init(k0, th_fit[0])
    const = 0.5 * M * np.log(2 * np.pi) + (d - M) * LOGW
    eyeM = jnp.eye(M); l1_w = L1_BASE / n_fit

    def core(p, theta):
        g = coord_fn(theta) if frozen else flat_mod.apply(p["flat"], theta)
        if frozen:
            g = p["A"] @ g
        return jnp.exp(SCALE_BOOST * p["raw_log_scale"]) * g

    def logdet_active(p, theta):
        J = jax.jacfwd(lambda a: core(p, theta.at[act].set(a)))(theta[act])
        return 0.5 * jnp.linalg.slogdet(J.T @ J + 1e-10 * eyeM)[1]

    def zvec(p, th_b, x_b):
        s = jnp.exp(SCALE_BOOST * p["raw_log_scale"])
        return jax.vmap(lambda t, y: core(p, t) - s * est_mod.apply(p["est"], y))(th_b, x_b)

    def nll_vec(p, th_b, x_b):
        s = jnp.exp(SCALE_BOOST * p["raw_log_scale"])
        def one(theta, x):
            r = core(p, theta) - s * est_mod.apply(p["est"], x)
            return 0.5 * jnp.sum(r ** 2) - logdet_active(p, theta) + const
        return jax.vmap(one)(th_b, x_b)

    def geom(p, th_b):
        return -jnp.mean(jax.vmap(lambda t: logdet_active(p, t))(th_b))

    def group_l1(p):
        if frozen:
            return 0.0
        K1 = p["flat"]["params"]["in_proj"]["kernel"]
        Ks = p["flat"]["params"]["skip"]["kernel"]
        return jnp.sum(jnp.sqrt(jnp.sum(K1 ** 2, 1) + jnp.sum(Ks ** 2, 1) + 1e-12))

    nll_mean = jax.jit(lambda p, t, y: jnp.mean(nll_vec(p, t, y)))
    warm = max(1, min(300, steps // 10))
    sched = optax.warmup_cosine_decay_schedule(1e-5, LR, warm, max(steps, warm + 1), LR * 1e-2)
    tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adamw(sched, weight_decay=1e-4))
    st = tx.init(p)

    @jax.jit
    def step(p, st, key):
        kp, ka = jr.split(key)
        idx = jr.randint(kp, (bp,), 0, n_fit)
        th_aug = jr.uniform(ka, (BATCH_AUG, d), minval=-A_BOX, maxval=A_BOX)
        def obj(p):
            z = zvec(p, th_fit[idx], x_fit[idx])
            return (0.5 * jnp.mean(jnp.sum(z ** 2, 1)) + geom(p, th_aug)
                    + const + l1_w * group_l1(p))
        l, g = jax.value_and_grad(obj)(p)
        upd, st = tx.update(g, st, p)
        new = optax.apply_updates(p, upd)
        ok = jnp.all(jnp.stack([jnp.all(jnp.isfinite(z))
                                for z in jax.tree_util.tree_leaves(g)]))
        return jax.tree_util.tree_map(lambda o, n: jnp.where(ok, n, o), p, new), st, l, ok

    key = jr.PRNGKey(seed + 7); ev = max(50, steps // 100)
    best, best_val = p, np.inf
    for i in range(steps):
        key, sk = jr.split(key)
        p, st, l, ok = step(p, st, sk)
        if i % ev == 0 or i == steps - 1:
            v = float(nll_mean(p, th_val, x_val))
            if np.isfinite(v) and v < best_val:
                best_val, best = v, p
    p = best
    return dict(m=M, active=tuple(int(v) for v in np.asarray(active)), best_val=best_val,
                nll_vec=jax.jit(lambda t, y: nll_vec(p, t, y)),
                eta=jax.jit(lambda t: jax.vmap(lambda u: core(p, u))(t)),
                jac=jax.jit(lambda t: jax.vmap(jax.jacrev(lambda u: core(p, u)))(t)))


def poly_r2(y, feats, deg=3):
    """R^2 of y regressed on polynomial features of `feats` up to total degree deg."""
    k = feats.shape[1]
    f = (feats - feats.mean(0)) / (feats.std(0) + 1e-12)
    cols = [f[:, j] for j in range(k)]
    if deg >= 2:
        cols += [f[:, i] * f[:, j] for i in range(k) for j in range(i, k)]
    if deg >= 3:
        cols += [f[:, i] * f[:, j] * f[:, l]
                 for i in range(k) for j in range(i, k) for l in range(j, k)]
    Xf = np.column_stack(cols + [np.ones(len(y))])      # intercept last, unscaled
    beta, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    return 1.0 - ((y - Xf @ beta) ** 2).mean() / y.var()


NVAL = max(50, int(0.15 * NSIMS))
th_fit, x_fit, th_val, x_val = TH[:-NVAL], X[:-NVAL], TH[-NVAL:], X[-NVAL:]

print("\n#### screen ####")
en = screen_inputs(TH, XB)
order = np.argsort(-en)
print(f"  ranked coords: {list(int(v) for v in order[:8])}  (truth {I0}, {I1})")
print(f"  norms:         {np.round(en[order][:8], 4)}")

# true coordinates on the test set
c_true = np.c_[th_te_np[:, I0], th_te_np[:, I1] - th_te_np[:, I0] ** 2]

for M in M_TEST:
    A = tuple(sorted(int(v) for v in order[:M]))
    mo = fit_joint(M, th_fit, x_fit, th_val, x_val, A)
    nll = float(np.mean(np.asarray(mo["nll_vec"](TH_TE, X_TE))))
    eta = np.asarray(mo["eta"](TH_TE))

    # information spectrum: eigenvalues of the prior covariance of eta
    C = np.cov(eta, rowvar=False).reshape(M, M)
    lam, V = np.linalg.eigh(C)
    o = np.argsort(-lam); lam, V = lam[o], V[:, o]
    e_rot = (eta - eta.mean(0)) @ V          # canonical axes

    print(f"\n#### m={M}  S={A} ####")
    print(f"  test NLL {nll:9.4f}   (floor {oracle(M):.4f}, excess {nll - oracle(M):+.4f})")
    print(f"  spectrum lambda      {np.round(lam, 2)}")
    print(f"  0.5 log lambda (nats) {np.round(0.5 * np.log(np.maximum(lam, 1e-12)), 3)}")
    r_spec = int(np.sum(lam > 3.0))
    print(f"  directions with lambda > 3: {r_spec}   (truth 2)")

    # do the top-2 canonical axes span the true coordinates?
    top2 = e_rot[:, :2]
    print(f"  R^2(true coord | poly3 of top-2 axes): "
          f"{np.round([poly_r2(c_true[:, k], top2) for k in (0, 1)], 5)}")
    print(f"  R^2(top-2 axis  | poly3 of true coords): "
          f"{np.round([poly_r2(top2[:, k], c_true) for k in (0, 1)], 5)}")
    if M > 2:
        junk = e_rot[:, 2:]
        print(f"  R^2(discarded axis | poly3 of true coords): "
              f"{np.round([poly_r2(junk[:, k], c_true) for k in range(min(3, M - 2))], 5)}")
        spur = [j for j in A if j not in (I0, I1)]
        th_spur = th_te_np[:, spur]
        print(f"  R^2(discarded axis | poly3 of spurious theta {spur}): "
              f"{np.round([poly_r2(junk[:, k], th_spur) for k in range(min(3, M - 2))], 5)}")
        print(f"  var of discarded axes: {np.round(lam[2:], 3)}  (want 1.0 = held at prior)")
print("\ndone")
