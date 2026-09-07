# =============================================================================
# Joint objective, calibration, bootstrap UQ and two-part MDL
# with a UNIFORM BOX prior  theta ~ U(-A_BOX, A_BOX)^D.
#
#   -log q(theta|x) = 0.5||eta(theta) - etahat(x)||^2          active block
#                     - log|det d eta / d theta_S|             change of vars
#                     + 0.5 m log 2pi + (d-m) log W            normalisation
#
# The d-m coordinates outside S are held at the prior, so their contribution is
# the CONSTANT (d-m) log W.  This requires S to be a coordinate subset: a box
# prior is not rotationally invariant, so a general learned subspace has no
# closed-form complement density (see notes at the bottom of the file).
# S is chosen by a prior-free group-L1 screen, and m by the NLL plateau.
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
M_LIST       = (1, 2, 3)
SD           = (0.25, 0.5)
N_REP        = 8
STEPS        = 8000
SCREEN_STEPS = 3000
BATCH_PAIR   = 256
BATCH_AUG    = 512
LR, SCALE_BOOST, SEED = 1e-3, 2.0, 0
L1_BASE      = 2.0
L1_SCREEN    = 1e-2
PLATEAU_TOL  = 0.05
K_BOOT       = 8

# ------------------------------------------------------------------ prior
A_BOX = 3.0
WIDTH = 2.0 * A_BOX
LOGW  = float(np.log(WIDTH))

# gain from promoting an informative axis out of the prior into the active block
sig  = np.asarray(SD) / np.sqrt(N_REP)
gain = np.sort(LOGW - 0.5 * np.log(2 * np.pi * sig ** 2) - 0.5)[::-1]
def oracle(m, d=D):
    """Floor assuming the m selected coords are the informative ones, best first."""
    return d * LOGW - gain[:min(m, len(gain))].sum()

_r = np.random.default_rng(1000 + D)
I0, I1 = (int(v) for v in _r.choice(D, size=2, replace=False))
print(f"hidden informative coordinates: theta_{I0}, theta_{I1}")
print(f"prior U(-{A_BOX}, {A_BOX});  per-axis gains {np.round(gain, 4)};  "
      f"floors m=0..3 {np.round([oracle(k) for k in range(4)], 4)}")

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
XB    = jnp.asarray((x_np.mean(1) - mu_x) / sd_x)          # sufficient summary

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

# ------------------------------------------------------- 0. coordinate screen
def screen_inputs(th, xb, lam=L1_SCREEN, steps=SCREEN_STEPS, seed=0):
    """Prior-free: regress the sufficient data summary on theta with a group-L1
    penalty on the input columns.  Returns per-coordinate group norms."""
    mod = Screen()
    ps  = mod.init(jr.PRNGKey(seed), th[0])
    tx  = optax.adam(3e-3)
    st  = tx.init(ps)
    n   = th.shape[0]
    b   = min(512, n)

    @jax.jit
    def one(ps, st, key):
        i = jr.randint(key, (b,), 0, n)
        def obj(ps):
            pr = jax.vmap(lambda t: mod.apply(ps, t))(th[i])
            K  = ps["params"]["in_proj"]["kernel"]
            return (jnp.mean((pr - xb[i]) ** 2)
                    + lam * jnp.sum(jnp.sqrt(jnp.sum(K ** 2, 1) + 1e-12)))
        l, g = jax.value_and_grad(obj)(ps)
        u, st = tx.update(g, st, ps)
        return optax.apply_updates(ps, u), st, l

    key = jr.PRNGKey(seed + 1)
    for _ in range(steps):
        key, sk = jr.split(key)
        ps, st, l = one(ps, st, sk)
    K = np.asarray(ps["params"]["in_proj"]["kernel"])
    return np.sqrt((K ** 2).sum(1))

# ------------------------------------------------------------------ fit
def fit_joint(M, th_fit, x_fit, th_val, x_val, active, steps=STEPS, seed=SEED,
              coord_fn: Optional[Callable] = None, d=D):
    """coord_fn None -> learn a Flattener.  Otherwise freeze coord_fn (a pure jnp
    callable theta -> (M,)) and fit only A, scale, estimator.  A is a full m x m
    matrix so a frozen candidate is judged up to a linear map, matching the gauge.
    `active` is the coordinate subset S whose sub-Jacobian sets the volume term."""
    n_fit = th_fit.shape[0]
    bp = min(BATCH_PAIR, n_fit)
    act = jnp.asarray(np.asarray(active, dtype=np.int32))
    Q0 = np.zeros((d, M))
    Q0[np.asarray(active, dtype=int), np.arange(M)] = 1.0   # det = 1 at step 0
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
    eyeM  = jnp.eye(M)
    l1_w  = L1_BASE / n_fit

    def core(p, theta):
        g = coord_fn(theta) if frozen else flat_mod.apply(p["flat"], theta)
        if frozen:
            g = p["A"] @ g
        return jnp.exp(SCALE_BOOST * p["raw_log_scale"]) * g

    def logdet_active(p, theta):
        J = jax.jacfwd(lambda a: core(p, theta.at[act].set(a)))(theta[act])   # (M,M)
        return 0.5 * jnp.linalg.slogdet(J.T @ J + 1e-10 * eyeM)[1]

    def zvec(p, th_b, x_b):                                    # (B, M), claimed N(0,I)
        s = jnp.exp(SCALE_BOOST * p["raw_log_scale"])
        return jax.vmap(lambda t, y: core(p, t) - s * est_mod.apply(p["est"], y))(th_b, x_b)

    def nll_vec(p, th_b, x_b):
        s = jnp.exp(SCALE_BOOST * p["raw_log_scale"])
        def one(theta, x):
            r = core(p, theta) - s * est_mod.apply(p["est"], x)
            return 0.5 * jnp.sum(r ** 2) - logdet_active(p, theta) + const
        return jax.vmap(one)(th_b, x_b)

    def geom(p, th_b):                                         # theta-only: free
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

    key = jr.PRNGKey(seed + 7)
    ev = max(50, steps // 100)
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

    R1T = np.eye(d)[:, np.asarray(active, dtype=int)]
    return dict(m=M, params=p, frozen=frozen, best_val=best_val, skipped=skipped,
                active=tuple(int(v) for v in np.asarray(active)),
                nll_vec=jax.jit(lambda t, y: nll_vec(p, t, y)),
                z_vec=jax.jit(lambda t, y: zvec(p, t, y)),
                eta=jax.jit(lambda t: jax.vmap(lambda u: core(p, u))(t)),
                jac=jax.jit(lambda t: jax.vmap(jax.jacrev(lambda u: core(p, u)))(t)),
                R1T=R1T)

# ------------------------------------------------------------------ gauge fixing
def gauge_fix(eta, theta, n_grid=721, sweeps=3):
    e = eta - eta.mean(0)
    m = e.shape[1]
    Xa = np.c_[theta - theta.mean(0), np.ones(len(theta))]
    B, *_ = np.linalg.lstsq(Xa, e, rcond=None)
    C  = np.cov(e, rowvar=False).reshape(m, m)
    Cr = np.cov(e - Xa @ B, rowvar=False).reshape(m, m)
    W = np.eye(m)
    r2 = lambda a: 1.0 - (a @ Cr @ a) / (a @ C @ a)
    if m > 1:
        angles = np.linspace(0.0, np.pi, n_grid, endpoint=False)
        for _ in range(sweeps):
            for i in range(m):
                for j in range(i + 1, m):
                    vals = [r2(np.cos(f) * W[:, i] + np.sin(f) * W[:, j]) for f in angles]
                    f = angles[int(np.argmax(vals))]
                    c, s = np.cos(f), np.sin(f)
                    wi, wj = W[:, i].copy(), W[:, j].copy()
                    W[:, i], W[:, j] = c * wi + s * wj, -s * wi + c * wj
        W = W[:, np.argsort([-r2(W[:, j]) for j in range(m)])]
    e_rot = e @ W
    for j in range(m):
        cc = np.corrcoef(np.c_[e_rot[:, j], theta], rowvar=False)[0, 1:]
        e_rot[:, j] *= np.sign(cc[int(np.argmax(np.abs(cc)))])
    return e_rot, W, [r2(W[:, j]) for j in range(m)]

# ------------------------------------------------------------- 0. screen
NVAL = max(50, int(0.15 * NSIMS))
th_fit, x_fit, th_val, x_val = TH[:-NVAL], X[:-NVAL], TH[-NVAL:], X[-NVAL:]
E = np.zeros((D, 2)); E[I0, 0] = 1.0; E[I1, 1] = 1.0

print("\n######## 0. prior-free coordinate screen ########")
en = screen_inputs(TH, XB)
order = np.argsort(-en)
print(f"  input-group norms, sorted: {np.round(en[order][:6], 4)} ... "
      f"{np.round(en[order][-3:], 4)}")
print(f"  ranked coords: {list(order[:6])}   (truth {{{I0}, {I1}}})")
active_of = {M: tuple(sorted(int(v) for v in order[:M])) for M in M_LIST}
for M in M_LIST:
    print(f"    m={M} -> S = {active_of[M]}")

# ------------------------------------------------------------- 1. rank sweep
print("\n######## 1. rank sweep ########")
mods = []
for M in M_LIST:
    mo = fit_joint(M, th_fit, x_fit, th_val, x_val, active_of[M])
    v = float(np.mean(np.asarray(mo["nll_vec"](TH_TE, X_TE))))
    mo["val"] = v; mods.append(mo)
    print(f"  m={M}: test NLL {v:9.4f}  (oracle {oracle(M):8.4f})   S={mo['active']}")
g = [a["val"] - b["val"] for a, b in zip(mods, mods[1:])]
hit = [i for i, q in enumerate(g) if q < PLATEAU_TOL]
r_hat = M_LIST[hit[0]] if hit else M_LIST[-1]
print(f"  gains {np.round(g, 4)}  (oracle {np.round(np.diff([-oracle(m) for m in M_LIST]), 4)})"
      f"  ->  r_hat = {r_hat}")
ref = mods[M_LIST.index(r_hat)]
print(f"  selected set {ref['active']}  vs truth {tuple(sorted((I0, I1)))}  "
      f"-> {'MATCH' if set(ref['active']) == {I0, I1} else 'MISMATCH'}")

# ------------------------------------------------------------- 2. calibration
print("\n######## 2. calibration of the fitted model ########")
z = np.asarray(ref["z_vec"](TH_TE, X_TE))
pit = 0.5 * (1.0 + jax.scipy.special.erf(np.asarray(z) / np.sqrt(2.0)))
comp = np.setdiff1d(np.arange(D), np.asarray(ref["active"]))
tc = th_te_np[:, comp]                       # held at prior: claimed U(-A, A)
xbar = x_te_np.mean(1)

print(f"  active-block z: mean {np.round(z.mean(0), 4)}   std {np.round(z.std(0), 4)}"
      f"   (want 0 and 1)")
for lev, nom in ((1.0, 0.6827), (1.96, 0.9500), (2.58, 0.9901)):
    print(f"    |z| < {lev:4.2f}: {np.round((np.abs(z) < lev).mean(0), 4)}  (nominal {nom:.4f})")
u = (tc + A_BOX) / WIDTH                     # claimed U(0,1)
print(f"  complement PIT: mean {u.mean():.4f} (want 0.5)  std {u.std():.4f} "
      f"(want {1/np.sqrt(12):.4f})")
Xa = np.c_[xbar, xbar ** 2, np.ones(len(xbar))]
leak = [1.0 - ((tc[:, j] - Xa @ np.linalg.lstsq(Xa, tc[:, j], rcond=None)[0]) ** 2).mean()
        / tc[:, j].var() for j in range(tc.shape[1])]
print(f"  complement leakage R^2 vs data: max {max(leak):.5f}  mean {np.mean(leak):.5f}"
      f"   (null expectation {Xa.shape[1] - 1} / {len(xbar)} = "
      f"{(Xa.shape[1] - 1) / len(xbar):.5f})")

gx = np.linspace(-4, 4, 200)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for j in range(r_hat):
    ax[0].hist(z[:, j], bins=80, density=True, histtype="step", label=f"$z_{j}$")
ax[0].plot(gx, np.exp(-gx ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="N(0,1)")
ax[0].set_title(r"active block  $z=\eta(\theta)-\hat\eta(x)$"); ax[0].legend()
ax[1].hist(pit.ravel(), bins=50, density=True, histtype="step")
ax[1].axhline(1.0, color="k", ls="--", lw=1)
ax[1].set_title("PIT $\\Phi(z)$  (want uniform)"); ax[1].set_ylim(0, 2)
ax[2].hist(u.ravel(), bins=60, density=True, histtype="step")
ax[2].axhline(1.0, color="k", ls="--", lw=1)
ax[2].set_title(f"complement ({D - r_hat} dims held at prior, want uniform)")
ax[2].set_ylim(0, 2)
plt.tight_layout(); plt.show()

# ------------------------------------------------------- 3. bootstrap ensemble
print(f"\n######## 3. bootstrap ensemble (K={K_BOOT}) ########")
boot_rng = np.random.default_rng(SEED + 31)
n = TH.shape[0]
etas, sets = [], []
for k in range(K_BOOT):
    idx = boot_rng.integers(0, n, n)
    oob = np.setdiff1d(np.arange(n), np.unique(idx))
    en_k = screen_inputs(TH[idx], XB[idx], seed=100 + k)          # re-screen: selection UQ
    act_k = tuple(sorted(int(v) for v in np.argsort(-en_k)[:r_hat]))
    mo = fit_joint(r_hat, TH[idx], X[idx], TH[oob], X[oob], act_k, seed=SEED + 500 + k)
    e = np.asarray(mo["eta"](TH_TE))
    e, _, _ = gauge_fix(e, th_te_np)
    etas.append(e); sets.append(act_k)
    print(f"  member {k}: oob NLL {mo['best_val']:.4f}   S={act_k}"
          f"   {'ok' if set(act_k) == {I0, I1} else 'DIFFERENT'}")

Ens = np.stack(etas)
eta_mu, eta_sd = Ens.mean(0), Ens.std(0, ddof=1)
print(f"  ensemble spread (median over test points): {np.round(np.median(eta_sd, 0), 4)}")
print(f"  spread / |eta| (median):                   "
      f"{np.round(np.median(eta_sd / (np.abs(eta_mu) + 1e-12), 0), 4)}")

K = K_BOOT
mu_loo = (K * eta_mu[None] - Ens) / (K - 1)
sd_loo = np.sqrt(np.clip((K * (Ens.var(0, ddof=0)[None] + eta_mu[None] ** 2)
                          - Ens ** 2) / (K - 1) - mu_loo ** 2, 1e-24, None))
sd_loo *= np.sqrt((K - 1) / (K - 2))                      # population -> sample sd
z_loo = (Ens - mu_loo) / (sd_loo * np.sqrt(K / (K - 1)))
zstd = z_loo.reshape(-1, r_hat).std(0)
tref = np.sqrt((K - 2) / (K - 4))                         # std of t_{K-2}, needs K>4
print(f"  LOO ensemble z: std {np.round(zstd, 3)}   reference std(t_{K-2}) = {tref:.4f}"
      f"   ratio {np.round(zstd / tref, 3)}  (>1 = spread too small)")
frq = np.mean([set(s) == {I0, I1} for s in sets])
print(f"  selection agreement: {frq:.2f} of members recover the true coordinate set")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for j in range(r_hat):
    ax[0].hist(z_loo[..., j].ravel() / tref, bins=80, density=True,
               histtype="step", label=f"axis {j}")
ax[0].plot(gx, np.exp(-gx ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="N(0,1)")
ax[0].set_title("leave-one-out ensemble $z$ / std$(t_{K-2})$"); ax[0].legend()
ax[1].hist(eta_sd.ravel(), bins=60, histtype="step")
ax[1].set_title(r"ensemble spread of $\eta$ (SR $y_{std}$)")
plt.tight_layout(); plt.show()

# ------------------------------------------------------------- 4. two-part MDL
print("\n######## 4. candidate scoring: DL = DL_model + heldout NLL (nats) ########")
CANDIDATES = [
    ("theta_i0, theta_i1 - theta_i0^2",
     lambda t: jnp.stack([t[I0], t[I1] - t[I0] ** 2]), 9.0),
    ("theta_i0, theta_i1                (misses the quadratic)",
     lambda t: jnp.stack([t[I0], t[I1]]), 4.0),
    ("theta_i0, theta_i1 - theta_i0^1.9 (wrong exponent)",
     lambda t: jnp.stack([t[I0], t[I1] - jnp.abs(t[I0]) ** 1.9]), 14.0),
    ("theta_i0, theta_i1 - theta_i0^2 - 0.3 theta_j (spurious nuisance)",
     lambda t: jnp.stack([t[I0], t[I1] - t[I0] ** 2 - 0.3 * t[(I0 + 1) % D]]), 15.0),
    ("theta_i0                          (rank 1)",
     lambda t: jnp.stack([t[I0]]), 2.0),
]

ref_per = np.asarray(ref["nll_vec"](TH_TE, X_TE))
print(f"  neural map (reference, m={r_hat}): heldout NLL {ref_per.mean():.4f} nats/sample")
rows, per_of = [], {}
for name, fn, dl in CANDIDATES:
    m_c = int(np.asarray(jax.eval_shape(fn, TH[0]).shape)[0])   # candidate's own m
    act_c = active_of.get(m_c, tuple(sorted(int(v) for v in order[:m_c])))
    mo = fit_joint(m_c, th_fit, x_fit, th_val, x_val, act_c, coord_fn=fn)
    per = np.asarray(mo["nll_vec"](TH_TE, X_TE))
    per_of[name] = per
    dif = per - ref_per
    se = dif.std(ddof=1) / np.sqrt(len(dif))
    rows.append((name, per.mean(), dl, dl + per.sum(), dif.mean(), se))
    print(f"  {name}   [m={m_c}, S={act_c}]")
    print(f"      heldout NLL {per.mean():9.4f}   vs neural {dif.mean():+8.4f} "
          f"+/- {se:.4f} nats/sample")

print("\n  --- ranked by two-part description length ---")
for name, nll, dl, tot, dif, se in sorted(rows, key=lambda r: r[3]):
    print(f"   DL {tot:14.2f}  = model {dl:6.1f} + data {tot - dl:13.2f}   {name}")

print("\n  --- pairwise paired tests (nats/sample, negative = row is better) ---")
nm = list(per_of)
for i, name in enumerate(nm):
    print(f"   c{i} = {name}")
for a in range(len(nm)):
    for b in range(a + 1, len(nm)):
        dd = per_of[nm[a]] - per_of[nm[b]]
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        print(f"   c{a} - c{b}: {dd.mean():+9.5f} +/- {se:.5f} "
              f"({abs(dd.mean() / se):7.1f} sigma)")
print("\ndone")
