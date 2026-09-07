# =============================================================================
# 20D Rosenbrock + 18 nuisance, UNIFORM box prior, NO SWEEP OVER m.
#
# Protocol:
#   0. screen once            -> ranking of all d coordinates
#   1. ONE probe fit at m = M_PROBE (generous)
#      -> information spectrum lambda_i = Var_pi(eta_i), null value 1 exactly
#      -> r_hat = #{ i : 0.5 log lambda_i > NATS_FLOOR }
#   2. refit at r_hat         -> production map; check the spare-dimension cost
#   3. calibration at r_hat   -> z ~ N(0,I), complement uniform, leakage R^2
#   4. bootstrap at M_PROBE   -> error bars on lambda, screen recovery, map spread
#   5. candidate scoring      -> two-part DL + pairwise paired tests
#
# Cost: 2 + K fits.  No value of m is ever swept.
# =============================================================================
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import flax.linen as nn
import optax
import matplotlib.pyplot as plt
from typing import Sequence, Callable, Optional

jax.config.update("jax_enable_x64", True)

D            = 20
NSIMS        = 32000
NTEST        = 20000
M_PROBE      = 6         # generous: ~1.5-3x the rank you expect.  Never swept.
NATS_FLOOR   = 0.10      # keep axes carrying more than this many nats
SD           = (0.25, 0.5)
N_REP        = 8
STEPS        = 8000
SCREEN_STEPS = 3000
BATCH_PAIR   = 256
BATCH_AUG    = 512
LR, SCALE_BOOST, SEED = 1e-3, 2.0, 0
L1_BASE      = 2.0
L1_SCREEN    = 1e-2
K_BOOT       = 8
# The bootstrap serves two different purposes and they want different m.
#   True  -> fit members at M_PROBE: error bars on lambda, confirms r_hat.
#   False -> fit members at r_hat:   the y_std handed to symbolic regression.
# A probe-m ensemble has spare dimensions, so it reports a wider coordinate
# spread than the production map at r_hat does.  Do not quote it as y_std.
BOOT_AT_PROBE = True

# ------------------------------------------------------------------ prior
A_BOX = 3.0
WIDTH = 2.0 * A_BOX
LOGW  = float(np.log(WIDTH))

sig  = np.asarray(SD) / np.sqrt(N_REP)
gain = np.sort(LOGW - 0.5 * np.log(2 * np.pi * sig ** 2) - 0.5)[::-1]
def floor_nll(m, d=D):
    """Reference value, assuming the m active coords are the informative ones."""
    return d * LOGW - gain[:min(m, len(gain))].sum()

_r = np.random.default_rng(1000 + D)
I0, I1 = (int(v) for v in _r.choice(D, size=2, replace=False))
print(f"truth (hidden from the method): informative pair theta_{I0}, theta_{I1}")
print(f"prior U(-{A_BOX},{A_BOX})^{D};  reference floor at the true rank: {floor_nll(2):.4f}")

def sim(n, rng):
    th = rng.uniform(-A_BOX, A_BOX, size=(n, D))
    mu = np.stack([th[:, I0], th[:, I1] - th[:, I0] ** 2], axis=1)
    return th, mu[:, None, :] + rng.normal(size=(n, N_REP, 2)) * np.asarray(SD)

rng = np.random.default_rng(SEED)
th_np, x_np = sim(NSIMS, rng)
th_te_np, x_te_np = sim(NTEST, np.random.default_rng(7777))
mu_x, sd_x = x_np.reshape(-1, 2).mean(0), x_np.reshape(-1, 2).std(0) + 1e-12
TH   = jnp.asarray(th_np);     X    = jnp.asarray((x_np - mu_x) / sd_x)
TH_TE = jnp.asarray(th_te_np); X_TE = jnp.asarray((x_te_np - mu_x) / sd_x)
XB    = jnp.asarray((x_np.mean(1) - mu_x) / sd_x)

# ------------------------------------------------------------------ networks
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
    def __call__(self, x):                    # x: (N_REP, 2)
        h = x.mean(0)                         # exactly sufficient for mu(theta)
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


# ----------------------------------------------------- 0. coordinate screen
def screen_inputs(th, xb, lam=L1_SCREEN, steps=SCREEN_STEPS, seed=0):
    """Prior-free: regress the sufficient data summary on theta with a group-L1
    penalty on the input columns.  Returns per-coordinate group norms."""
    mod = Screen()
    ps = mod.init(jr.PRNGKey(seed), th[0])
    tx = optax.adam(3e-3); st = tx.init(ps)
    n = th.shape[0]; b = min(512, n)

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


# ------------------------------------------------------------------ fit
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
    best, best_val, skipped = p, np.inf, 0
    for i in range(steps):
        key, sk = jr.split(key)
        p, st, l, ok = step(p, st, sk)
        skipped += int(not bool(ok))
        if i % ev == 0 or i == steps - 1:
            v = float(nll_mean(p, th_val, x_val))
            if np.isfinite(v) and v < best_val:
                best_val, best = v, p
    p = best
    if skipped:
        print(f"    (skipped {skipped}/{steps} non-finite updates)")
    return dict(m=M, active=tuple(int(v) for v in np.asarray(active)), best_val=best_val,
                nll_vec=jax.jit(lambda t, y: nll_vec(p, t, y)),
                z_vec=jax.jit(lambda t, y: zvec(p, t, y)),
                eta=jax.jit(lambda t: jax.vmap(lambda u: core(p, u))(t)))


# --------------------------------------------------------- spectral helpers
def spectrum(eta):
    """Canonical axes: eigenbasis of the prior covariance of eta, ordered by
    information.  lambda_i = Var_pi(eta_i) has null value 1 (Proposition 3)."""
    e = eta - eta.mean(0)
    m = e.shape[1]
    C = np.cov(e, rowvar=False).reshape(m, m)
    lam, V = np.linalg.eigh(C)
    o = np.argsort(-lam)
    return e @ V[:, o], np.maximum(lam[o], 1e-12), V[:, o]


def procrustes_align(E, E_ref):
    """Remove the O(r) gauge between two sets of canonical axes."""
    U, _, Wt = np.linalg.svd(E.T @ E_ref)
    return E @ (U @ Wt)


def poly_r2(y, feats, deg=3):
    k = feats.shape[1]
    f = (feats - feats.mean(0)) / (feats.std(0) + 1e-12)
    cols = [f[:, j] for j in range(k)]
    if deg >= 2:
        cols += [f[:, i] * f[:, j] for i in range(k) for j in range(i, k)]
    if deg >= 3:
        cols += [f[:, i] * f[:, j] * f[:, l]
                 for i in range(k) for j in range(i, k) for l in range(j, k)]
    Xf = np.column_stack(cols + [np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    return 1.0 - ((y - Xf @ beta) ** 2).mean() / y.var()


NVAL = max(50, int(0.15 * NSIMS))
th_fit, x_fit, th_val, x_val = TH[:-NVAL], X[:-NVAL], TH[-NVAL:], X[-NVAL:]
c_true = np.c_[th_te_np[:, I0], th_te_np[:, I1] - th_te_np[:, I0] ** 2]

# ------------------------------------------------------- 0. screen (one fit)
print("\n######## 0. coordinate screen (one regression, prior-free) ########")
en = screen_inputs(TH, XB)
order = np.argsort(-en)
print(f"  group norms (sorted): {np.round(en[order][:8], 4)} ... {np.round(en[order][-2:], 4)}")
print(f"  ranking:              {[int(v) for v in order[:8]]}")
A_probe = tuple(sorted(int(v) for v in order[:M_PROBE]))
print(f"  active set at m={M_PROBE}: {A_probe}")

# --------------------------------------- 1. ONE probe fit -> spectrum -> r_hat
print(f"\n######## 1. single probe fit at m={M_PROBE} (no sweep) ########")
probe = fit_joint(M_PROBE, th_fit, x_fit, th_val, x_val, A_probe)
nll_probe = float(np.mean(np.asarray(probe["nll_vec"](TH_TE, X_TE))))
E_probe, lam, V_probe = spectrum(np.asarray(probe["eta"](TH_TE)))
nats = 0.5 * np.log(lam)
band_samp = (1.0 + np.sqrt(M_PROBE / NTEST)) ** 2 - 1.0

lam_null = float(np.median(lam[M_PROBE // 2:]))   # robust null level of the spare block
nats_n = 0.5 * np.log(lam / lam_null)             # profile against the measured null
print(f"  test NLL {nll_probe:.4f}")
print(f"  information spectrum lambda_i : {np.round(lam, 3)}")
print(f"  measured null level           : {lam_null:.4f}  (exactly 1 at the optimum)")
print(f"  profile 0.5 log(lambda/null)  : {np.round(nats_n, 4)} nats")
print(f"  sampling-only band on lambda  : 1 + {band_samp:.4f}"
      f"  ({0.5 * np.log1p(band_samp):.4f} nats)")

# r_hat should be stable over a wide range of the floor.  That stability, not any
# single threshold, is the evidence that the rank is well determined.
print("  sensitivity of r_hat to the floor:")
for f in (0.02, 0.05, 0.10, 0.25, 0.50, 1.00):
    print(f"    floor {f:4.2f} nats -> r_hat = {int(np.sum(nats_n > f))}")
r_hat = int(np.sum(nats_n > NATS_FLOOR))
print(f"  adopt floor {NATS_FLOOR} nats  ->  r_hat = {r_hat}")
if r_hat >= M_PROBE:
    print(f"  WARNING: r_hat reached the probe dimension. Increase M_PROBE and refit.")
gaps = lam[:r_hat] / np.maximum(lam[1:r_hat + 1], 1e-12)
if r_hat > 1 and np.min(gaps[:-1] if len(gaps) > 1 else gaps) < 1.2:
    print(f"  WARNING: near-degenerate retained eigenvalues (ratios {np.round(gaps, 3)});"
          f" that plane is not individually identified.")
print(f"  discarded block: lambda in [{lam[r_hat:].min():.3f}, {lam[r_hat:].max():.3f}]"
      f"  (null value 1)")
print(f"  [truth check] R^2(true coord | cubic in top-{r_hat} axes): "
      f"{np.round([poly_r2(c_true[:, k], E_probe[:, :r_hat]) for k in (0, 1)], 5)}")

# ------------------------------------------- 2. refit at r_hat -> production
print(f"\n######## 2. refit at r_hat={r_hat} ########")
A_hat = tuple(sorted(int(v) for v in order[:r_hat]))
ref = fit_joint(r_hat, th_fit, x_fit, th_val, x_val, A_hat)
nll_hat = float(np.mean(np.asarray(ref["nll_vec"](TH_TE, X_TE))))
per_spare = (nll_probe - nll_hat) / max(M_PROBE - r_hat, 1)
print(f"  active set {A_hat}   (truth {tuple(sorted((I0, I1)))})"
      f"  -> {'MATCH' if set(A_hat) == {I0, I1} else 'MISMATCH'}")
print(f"  test NLL {nll_hat:.4f}   (reference floor {floor_nll(r_hat):.4f},"
      f" excess {nll_hat - floor_nll(r_hat):+.4f})")
print(f"  cost of a spare dimension: {per_spare:.4f} nats"
      f"  ({M_PROBE - r_hat} spare, total {nll_probe - nll_hat:.4f})")
E_hat, lam_hat, _ = spectrum(np.asarray(ref["eta"](TH_TE)))
print(f"  spectrum at r_hat: {np.round(lam_hat, 2)}   profile {np.round(0.5 * np.log(lam_hat), 4)} nats")
print(f"  [truth check] R^2(true coord | cubic in axes): "
      f"{np.round([poly_r2(c_true[:, k], E_hat) for k in (0, 1)], 5)}")
print(f"  [truth check] R^2(axis | cubic in true coords): "
      f"{np.round([poly_r2(E_hat[:, k], c_true) for k in range(r_hat)], 5)}")

# ------------------------------------------------------- 3. calibration
print(f"\n######## 3. calibration at r_hat={r_hat} ########")
z = np.asarray(ref["z_vec"](TH_TE, X_TE))
comp = np.setdiff1d(np.arange(D), np.asarray(A_hat))
tc = th_te_np[:, comp]
xbar = x_te_np.mean(1)
print(f"  active-block z: mean {np.round(z.mean(0), 4)}  std {np.round(z.std(0), 4)}  (want 0, 1)")
for lev, nom in ((1.0, 0.6827), (1.96, 0.9500), (2.58, 0.9901)):
    print(f"    |z| < {lev:4.2f}: {np.round((np.abs(z) < lev).mean(0), 4)}  (nominal {nom:.4f})")
u = (tc + A_BOX) / WIDTH
print(f"  complement PIT: mean {u.mean():.4f} (want 0.5)  std {u.std():.4f} "
      f"(want {1/np.sqrt(12):.4f})")
Xa = np.c_[xbar, xbar ** 2, np.ones(len(xbar))]
leak = [1.0 - ((tc[:, j] - Xa @ np.linalg.lstsq(Xa, tc[:, j], rcond=None)[0]) ** 2).mean()
        / tc[:, j].var() for j in range(tc.shape[1])]
print(f"  complement leakage R^2: max {max(leak):.5f}  mean {np.mean(leak):.5f}"
      f"   (null {Xa.shape[1] - 1}/{NTEST} = {(Xa.shape[1] - 1) / NTEST:.5f})")

# ------------------------- 4. bootstrap at M_PROBE: lambda bars + map spread
M_BOOT = M_PROBE if BOOT_AT_PROBE else r_hat
print(f"\n######## 4. bootstrap at m={M_BOOT} (K={K_BOOT}) ########")
print(f"  purpose: {'error bars on lambda, confirmation of r_hat' if BOOT_AT_PROBE else 'y_std for symbolic regression'}")
boot_rng = np.random.default_rng(SEED + 31)
n = TH.shape[0]
lams, Es, in_net, exact, r_members = [], [], [], [], []
for k in range(K_BOOT):
    idx = boot_rng.integers(0, n, n)
    oob = np.setdiff1d(np.arange(n), np.unique(idx))
    en_k = screen_inputs(TH[idx], XB[idx], seed=100 + k)
    ok_k = np.argsort(-en_k)
    A_k = tuple(sorted(int(v) for v in ok_k[:M_BOOT]))
    mo = fit_joint(M_BOOT, TH[idx], X[idx], TH[oob], X[oob], A_k, seed=SEED + 500 + k)
    Ek, lk, _ = spectrum(np.asarray(mo["eta"](TH_TE)))
    rk = int(np.sum(0.5 * np.log(lk / np.median(lk[M_BOOT // 2:])) > NATS_FLOOR)) if M_BOOT > r_hat else r_hat
    lams.append(lk); Es.append(Ek[:, :r_hat]); r_members.append(rk)
    in_net.append({I0, I1} <= set(A_k))
    exact.append(set(int(v) for v in ok_k[:r_hat]) == {I0, I1})
    print(f"  member {k}: oob NLL {mo['best_val']:.4f}  r_hat={rk}"
          f"  lambda[:3] {np.round(lk[:3], 1)}  pair in top-{M_BOOT}: {in_net[-1]}")

L = np.stack(lams)
print(f"\n  lambda across members (mean +/- sd):")
for i in range(M_BOOT):
    tag = "kept" if i < r_hat else "prior"
    print(f"    axis {i} [{tag:5s}]: {L[:, i].mean():9.3f} +/- {L[:, i].std(ddof=1):7.3f}"
          f"   -> {0.5*np.log(L[:, i].mean()):+.4f} nats")
uniq = sorted(set(r_members))
print(f"  r_hat across members: {r_members}  ->  {'UNANIMOUS' if len(uniq) == 1 else 'DISAGREEMENT'}")
if len(uniq) > 1:
    print("    Members disagree, so the probe fits have not converged. The spare-axis")
    print("    transport is unlearned, which inflates the null level. Raise STEPS or")
    print("    NSIMS before reading the rank. Disagreement is not rank ambiguity.")
print(f"  true pair inside the top-{M_BOOT} net: {np.mean(in_net):.2f} of members")
print(f"  screen ranks the true pair first: {np.mean(exact):.2f} of members")

Ens = np.stack([procrustes_align(E, Es[0]) for E in Es])     # remove O(r) gauge
eta_mu, eta_sd = Ens.mean(0), Ens.std(0, ddof=1)
tag = "(probe-m: upper bound, NOT y_std)" if BOOT_AT_PROBE else "(y_std for SR)"
print(f"  map spread {tag}: {np.round(np.median(eta_sd, 0), 4)}"
      f"   -> variance inflation {np.round(1 + np.median(eta_sd, 0) ** 2, 4)}")
K = K_BOOT
mu_loo = (K * eta_mu[None] - Ens) / (K - 1)
sd_loo = np.sqrt(np.clip((K * (Ens.var(0, ddof=0)[None] + eta_mu[None] ** 2)
                          - Ens ** 2) / (K - 1) - mu_loo ** 2, 1e-24, None))
sd_loo *= np.sqrt((K - 1) / (K - 2))
z_loo = (Ens - mu_loo) / (sd_loo * np.sqrt(K / (K - 1)))
tref = np.sqrt((K - 2) / (K - 4))
zstd = z_loo.reshape(-1, r_hat).std(0)
print(f"  LOO ensemble z: std {np.round(zstd, 3)}  reference std(t_{K-2})={tref:.4f}"
      f"  ratio {np.round(zstd / tref, 3)}")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].errorbar(np.arange(M_BOOT), L.mean(0), yerr=L.std(0, ddof=1), fmt="o", capsize=3)
ax[0].axhline(1.0, color="k", ls="--", lw=1, label=r"null $\lambda=1$")
ax[0].set_yscale("log"); ax[0].set_xlabel("canonical axis")
ax[0].set_ylabel(r"$\lambda_i=\mathrm{Var}_\pi(\eta_i)$")
ax[0].set_title(f"information spectrum at m={M_BOOT}"); ax[0].legend()
gx = np.linspace(-4, 4, 200)
for j in range(r_hat):
    ax[1].hist(z[:, j], bins=80, density=True, histtype="step", label=f"$z_{j}$")
ax[1].plot(gx, np.exp(-gx ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1)
ax[1].set_title(f"calibration at $\\hat r$={r_hat}"); ax[1].legend()
ax[2].hist(u.ravel(), bins=60, density=True, histtype="step")
ax[2].axhline(1.0, color="k", ls="--", lw=1); ax[2].set_ylim(0, 2)
ax[2].set_title(f"complement ({D - r_hat} dims at prior)")
plt.tight_layout(); plt.show()

# ------------------------------------------------------- 5. candidate scoring
print("\n######## 5. candidate scoring (DL = DL_model + heldout NLL, nats) ########")
CANDIDATES = [
    ("theta_i0, theta_i1 - theta_i0^2",
     lambda t: jnp.stack([t[I0], t[I1] - t[I0] ** 2]), 9.0),
    ("theta_i0, theta_i1                (misses the quadratic)",
     lambda t: jnp.stack([t[I0], t[I1]]), 4.0),
    ("theta_i0, theta_i1 - theta_i0^1.9 (wrong exponent)",
     lambda t: jnp.stack([t[I0], t[I1] - jnp.abs(t[I0]) ** 1.9]), 14.0),
    ("theta_i0, theta_i1 - theta_i0^2 - 0.3 theta_j (spurious nuisance)",
     lambda t: jnp.stack([t[I0], t[I1] - t[I0] ** 2 - 0.3 * t[(I0 + 1) % D]]), 15.0),
]
ref_per = np.asarray(ref["nll_vec"](TH_TE, X_TE))
print(f"  neural map (reference, m={r_hat}): {ref_per.mean():.4f} nats/sample")
rows, per_of = [], {}
for name, fn, dl in CANDIDATES:
    m_c = int(jax.eval_shape(fn, TH[0]).shape[0])
    A_c = tuple(sorted(int(v) for v in order[:m_c]))
    mo = fit_joint(m_c, th_fit, x_fit, th_val, x_val, A_c, coord_fn=fn)
    per = np.asarray(mo["nll_vec"](TH_TE, X_TE)); per_of[name] = per
    dif = per - ref_per; se = dif.std(ddof=1) / np.sqrt(len(dif))
    rows.append((name, per.mean(), dl, dl + per.sum()))
    print(f"  {name}  [m={m_c}]")
    print(f"      heldout NLL {per.mean():9.4f}   vs neural {dif.mean():+8.4f} +/- {se:.4f}")

print("\n  --- ranked by two-part description length ---")
for name, nll, dl, tot in sorted(rows, key=lambda r: r[3]):
    print(f"   DL {tot:13.2f} = model {dl:5.1f} + data {tot - dl:12.2f}   {name}")

print("\n  --- pairwise paired tests (negative = row is better) ---")
nm = list(per_of)
for i, name in enumerate(nm):
    print(f"   c{i} = {name}")
for a in range(len(nm)):
    for b in range(a + 1, len(nm)):
        dd = per_of[nm[a]] - per_of[nm[b]]
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        print(f"   c{a} - c{b}: {dd.mean():+9.5f} +/- {se:.5f} ({abs(dd.mean()/se):7.1f} sigma)")
print("\ndone")
