# =============================================================================
# 20D Rosenbrock + 18 nuisance -- NO coordinate screen, all theta enter.
#
# Contrast with rosenbrock20_oneshot.py: that script hard-cuts an active set A
# so the box-prior complement (d-m) log W is well-defined.  Here we ask whether
# the latents are recoverable from the geometry alone:
#
#   L = E[ 0.5 ||eta(theta) - etahat(x)||^2 - 0.5 log det(J J^T) ]
#       with J = d eta / d theta  in R^{m x d},  ALL d inputs.
#
# Protocol (same one-shot rank rule, no m-sweep):
#   1. ONE probe fit at m = M_PROBE
#   2. spectrum -> r_hat; refit at r_hat
#   3. diagnose which theta matter via Jacobian column norms (soft screen)
#   4. light bootstrap for r_hat agreement + map spread
#   5. score the frozen true product candidate
#
# Absolute NLL is NOT comparable to the screened box oracle.  Compare maps by
# R^2 / Spearman / column norms / paired NLL between candidates.
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
M_PROBE      = 6
NATS_FLOOR   = 0.10
SD           = (0.25, 0.5)
N_REP        = 8
STEPS        = 8000
BATCH_PAIR   = 256
BATCH_AUG    = 512
LR, SCALE_BOOST, SEED = 1e-3, 2.0, 0
L1_BASE      = 2.0          # soft sparsity on input columns (not a hard cut)
K_BOOT       = 8
BOOT_AT_PROBE = True
A_BOX = 3.0

_r = np.random.default_rng(1000 + D)
I0, I1 = (int(v) for v in _r.choice(D, size=2, replace=False))
print(f"truth (hidden from the method): informative pair theta_{I0}, theta_{I1}")
print(f"prior U(-{A_BOX},{A_BOX})^{D};  ALL {D} inputs enter; no coordinate screen")


def sim(n, rng):
    th = rng.uniform(-A_BOX, A_BOX, size=(n, D))
    mu = np.stack([th[:, I0], th[:, I1] - th[:, I0] ** 2], axis=1)
    return th, mu[:, None, :] + rng.normal(size=(n, N_REP, 2)) * np.asarray(SD)


rng = np.random.default_rng(SEED)
th_np, x_np = sim(NSIMS, rng)
th_te_np, x_te_np = sim(NTEST, np.random.default_rng(7777))
mu_x, sd_x = x_np.reshape(-1, 2).mean(0), x_np.reshape(-1, 2).std(0) + 1e-12
TH = jnp.asarray(th_np)
X = jnp.asarray((x_np - mu_x) / sd_x)
TH_TE = jnp.asarray(th_te_np)
X_TE = jnp.asarray((x_te_np - mu_x) / sd_x)

NVAL = max(50, int(0.15 * NSIMS))
th_fit, x_fit = TH[:-NVAL], X[:-NVAL]
th_val, x_val = TH[-NVAL:], X[-NVAL:]
c_true = np.c_[th_te_np[:, I0], th_te_np[:, I1] - th_te_np[:, I0] ** 2]


# ------------------------------------------------------------------ networks
class Flattener(nn.Module):
    m: int
    hidden: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, theta):
        h = nn.gelu(nn.Dense(self.hidden[0], name="in_proj")(theta))
        for k, w in enumerate(self.hidden[1:]):
            h = nn.gelu(nn.Dense(w, name=f"h{k}")(h))
        delta = nn.Dense(self.m, name="out", kernel_init=nn.initializers.zeros,
                         bias_init=nn.initializers.zeros)(h)
        # skip: first m ambient coordinates (eye(d, m))
        skip = nn.Dense(
            self.m, use_bias=False, name="skip",
            kernel_init=lambda k, s, dt: jnp.eye(s[0], s[1], dtype=dt),
        )(theta)
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


def fit_joint(M, th_fit, x_fit, th_val, x_val, steps=STEPS, seed=SEED,
              coord_fn: Optional[Callable] = None, d=D):
    n_fit = th_fit.shape[0]
    bp = min(BATCH_PAIR, n_fit)
    frozen = coord_fn is not None
    flat_mod = None if frozen else Flattener(m=M)
    est_mod = Estimator(m=M)
    k0, k1 = jr.split(jr.PRNGKey(seed), 2)
    p = {"est": est_mod.init(k1, x_fit[0]), "raw_log_scale": jnp.zeros(M)}
    if frozen:
        p["A"] = jnp.eye(M)
    else:
        p["flat"] = flat_mod.init(k0, th_fit[0])
    const = 0.5 * M * np.log(2 * np.pi)   # no (d-M)*log W -- not a box complement model
    eyeM = jnp.eye(M)
    l1_w = L1_BASE / n_fit

    def core(p, theta):
        g = coord_fn(theta) if frozen else flat_mod.apply(p["flat"], theta)
        if frozen:
            g = p["A"] @ g
        return jnp.exp(SCALE_BOOST * p["raw_log_scale"]) * g

    def logdet(p, theta):
        J = jax.jacfwd(lambda t: core(p, t))(theta)          # (m, d)
        return 0.5 * jnp.linalg.slogdet(J @ J.T + 1e-10 * eyeM)[1]

    def zvec(p, th_b, x_b):
        s = jnp.exp(SCALE_BOOST * p["raw_log_scale"])
        return jax.vmap(lambda t, y: core(p, t) - s * est_mod.apply(p["est"], y))(th_b, x_b)

    def nll_vec(p, th_b, x_b):
        def one(theta, x):
            s = jnp.exp(SCALE_BOOST * p["raw_log_scale"])
            r = core(p, theta) - s * est_mod.apply(p["est"], x)
            return 0.5 * jnp.sum(r ** 2) - logdet(p, theta) + const
        return jax.vmap(one)(th_b, x_b)

    def geom(p, th_b):
        return -jnp.mean(jax.vmap(lambda t: logdet(p, t))(th_b))

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
    return dict(
        m=M, best_val=best_val, params=p,
        nll_vec=jax.jit(lambda t, y: nll_vec(p, t, y)),
        z_vec=jax.jit(lambda t, y: zvec(p, t, y)),
        eta=jax.jit(lambda t: jax.vmap(lambda u: core(p, u))(t)),
        jac=jax.jit(lambda t: jax.vmap(lambda u: jax.jacfwd(lambda v: core(p, v))(u))(t)),
    )


def spectrum(eta):
    e = eta - eta.mean(0)
    m = e.shape[1]
    C = np.cov(e, rowvar=False).reshape(m, m)
    lam, V = np.linalg.eigh(C)
    o = np.argsort(-lam)
    return e @ V[:, o], np.maximum(lam[o], 1e-12), V[:, o]


def procrustes_align(E, E_ref):
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


def column_norms(jac):
    """RMS sensitivity of eta to each theta_j, averaged over the test set."""
    # jac: (N, m, d) -> (d,)
    return np.sqrt((np.asarray(jac) ** 2).mean(axis=(0, 1)))


# ------------------------------------------------------- 1. probe -> r_hat
print(f"\n######## 1. single probe fit at m={M_PROBE} (all {D} inputs, no screen) ########")
probe = fit_joint(M_PROBE, th_fit, x_fit, th_val, x_val, seed=SEED)
nll_probe = float(np.mean(np.asarray(probe["nll_vec"](TH_TE, X_TE))))
E_p, lam, _ = spectrum(np.asarray(probe["eta"](TH_TE)))
jac_p = probe["jac"](TH_TE)
col_p = column_norms(jac_p)

lam_null = float(np.median(lam[M_PROBE // 2:]))
nats_n = 0.5 * np.log(lam / lam_null)
band_samp = (1.0 + np.sqrt(M_PROBE / NTEST)) ** 2 - 1.0

print(f"  test NLL {nll_probe:.4f}   (geometry objective; not the screened box NLL)")
print(f"  information spectrum lambda_i : {np.round(lam, 3)}")
print(f"  measured null level           : {lam_null:.4f}")
print(f"  profile 0.5 log(lambda/null)  : {np.round(nats_n, 4)} nats")
print(f"  sampling-only band on lambda  : 1 + {band_samp:.4f}"
      f"  ({0.5 * np.log1p(band_samp):.4f} nats)")
print("  sensitivity of r_hat to the floor:")
for f in (0.02, 0.05, 0.10, 0.25, 0.50, 1.00):
    print(f"    floor {f:4.2f} nats -> r_hat = {int(np.sum(nats_n > f))}")
r_hat = int(np.sum(nats_n > NATS_FLOOR))
print(f"  adopt floor {NATS_FLOOR} nats  ->  r_hat = {r_hat}")
if r_hat >= M_PROBE:
    print("  WARNING: r_hat reached the probe dimension. Increase M_PROBE and refit.")
if r_hat < M_PROBE:
    print(f"  discarded block: lambda in [{lam[r_hat:].min():.3f}, {lam[r_hat:].max():.3f}]")

print(f"  [truth] R^2(true coord | cubic in top-{r_hat} axes): "
      f"{np.round([poly_r2(c_true[:, k], E_p[:, :r_hat]) for k in (0, 1)], 5)}")

order_col = np.argsort(-col_p)
print(f"  soft screen (Jacobian column norms): {np.round(col_p[order_col][:8], 4)} ...")
print(f"  ranking by ||J_:j||: {[int(v) for v in order_col[:8]]}")
print(f"  true pair ranks: theta_{I0} at {int(np.where(order_col == I0)[0][0]) + 1}, "
      f"theta_{I1} at {int(np.where(order_col == I1)[0][0]) + 1}"
      f"  -> {'TOP-2' if set(order_col[:2]) == {I0, I1} else 'NOT top-2'}")


# ------------------------------------------------------- 2. refit at r_hat
print(f"\n######## 2. refit at r_hat={r_hat} (still all {D} inputs) ########")
ref = fit_joint(r_hat, th_fit, x_fit, th_val, x_val, seed=SEED + 11)
nll_hat = float(np.mean(np.asarray(ref["nll_vec"](TH_TE, X_TE))))
E, lam_hat, _ = spectrum(np.asarray(ref["eta"](TH_TE)))
jac = ref["jac"](TH_TE)
col = column_norms(jac)
order_col = np.argsort(-col)

print(f"  test NLL {nll_hat:.4f}   (probe {nll_probe:.4f},"
      f" spare cost {(nll_probe - nll_hat) / max(M_PROBE - r_hat, 1):.4f} nats/axis)")
print(f"  spectrum at r_hat: {np.round(lam_hat, 2)}"
      f"   profile {np.round(0.5 * np.log(lam_hat), 4)} nats")
print(f"  [truth] R^2(true coord | cubic in axes): "
      f"{np.round([poly_r2(c_true[:, k], E) for k in (0, 1)], 5)}")
print(f"  [truth] R^2(axis | cubic in true coords): "
      f"{np.round([poly_r2(E[:, k], c_true) for k in range(r_hat)], 5)}")
print(f"  Jacobian column norms (sorted): {np.round(col[order_col][:8], 4)}")
print(f"  ranking: {[int(v) for v in order_col[:8]]}")
print(f"  true pair ranks: theta_{I0} at {int(np.where(order_col == I0)[0][0]) + 1}, "
      f"theta_{I1} at {int(np.where(order_col == I1)[0][0]) + 1}"
      f"  -> {'TOP-2' if set(order_col[:2]) == {I0, I1} else 'NOT top-2'}")

# nuisance leakage: max column norm outside the true pair, relative to the pair
pair_n = sorted([float(col[I0]), float(col[I1])])
nuis = np.delete(col, [I0, I1])
print(f"  nuisance max ||J|| / min(true pair): {nuis.max() / max(pair_n[0], 1e-12):.4f}"
      f"   (want << 1)")


# ------------------------------------------------------- 3. calibration
print(f"\n######## 3. calibration at r_hat={r_hat} ########")
z = np.asarray(ref["z_vec"](TH_TE, X_TE))
print(f"  z: mean {np.round(z.mean(0), 4)}  std {np.round(z.std(0), 4)}  (want 0, 1)")
for lev, nom in ((1.0, 0.6827), (1.96, 0.9500), (2.58, 0.9901)):
    print(f"    |z| < {lev:4.2f}: {np.round((np.abs(z) < lev).mean(0), 4)}  (nominal {nom:.4f})")


# ------------------------------------------------------- 4. bootstrap
M_BOOT = M_PROBE if BOOT_AT_PROBE else r_hat
print(f"\n######## 4. bootstrap at m={M_BOOT} (K={K_BOOT}) ########")
boot_rng = np.random.default_rng(SEED + 31)
n = TH.shape[0]
lams, Es, r_members, top2 = [], [], [], []
for k in range(K_BOOT):
    idx = boot_rng.integers(0, n, n)
    oob = np.setdiff1d(np.arange(n), np.unique(idx))
    mo = fit_joint(M_BOOT, TH[idx], X[idx], TH[oob], X[oob], seed=SEED + 500 + k)
    Ek, lk, _ = spectrum(np.asarray(mo["eta"](TH_TE)))
    rk = (int(np.sum(0.5 * np.log(lk / np.median(lk[M_BOOT // 2:])) > NATS_FLOOR))
          if M_BOOT > r_hat else r_hat)
    cj = column_norms(mo["jac"](TH_TE))
    oj = np.argsort(-cj)
    lams.append(lk); Es.append(Ek[:, :r_hat]); r_members.append(rk)
    top2.append(set(oj[:2]) == {I0, I1})
    print(f"  member {k}: oob NLL {mo['best_val']:.4f}  r_hat={rk}"
          f"  lambda[:3] {np.round(lk[:3], 1)}  top-2 cols match truth: {top2[-1]}")

L = np.stack(lams)
print("\n  lambda across members (mean +/- sd):")
for i in range(M_BOOT):
    tag = "kept" if i < r_hat else "prior"
    print(f"    axis {i} [{tag:5s}]: {L[:, i].mean():9.3f} +/- {L[:, i].std(ddof=1):7.3f}"
          f"   -> {0.5 * np.log(L[:, i].mean()):+.4f} nats")
uniq = sorted(set(r_members))
print(f"  r_hat across members: {r_members}  ->  "
      f"{'UNANIMOUS' if len(uniq) == 1 else 'DISAGREEMENT'}")
print(f"  soft-screen top-2 = true pair: {np.mean(top2):.2f} of members")

Ens = np.stack([procrustes_align(E, Es[0]) for E in Es])
eta_sd = Ens.std(0, ddof=1)
tag = "(probe-m: upper bound, NOT y_std)" if BOOT_AT_PROBE else "(y_std for SR)"
print(f"  map spread {tag}: {np.round(np.median(eta_sd, 0), 4)}")


# ------------------------------------------------------- 5. candidates
print("\n######## 5. candidate scoring (paired NLL; geometry objective) ########")
CANDIDATES = [
    ("theta_i0, theta_i1 - theta_i0^2",
     lambda t: jnp.stack([t[I0], t[I1] - t[I0] ** 2])),
    ("theta_i0, theta_i1                (misses the quadratic)",
     lambda t: jnp.stack([t[I0], t[I1]])),
    ("theta_i0, theta_i1 - theta_i0^1.9 (wrong exponent)",
     lambda t: jnp.stack([t[I0], t[I1] - jnp.abs(t[I0]) ** 1.9])),
    ("theta_i0, theta_i1 - theta_i0^2 - 0.3 theta_j (spurious nuisance)",
     lambda t: jnp.stack([t[I0], t[I1] - t[I0] ** 2 - 0.3 * t[(I0 + 1) % D]])),
]
ref_per = np.asarray(ref["nll_vec"](TH_TE, X_TE))
print(f"  neural map (reference, m={r_hat}): {ref_per.mean():.4f}")
per_of = {}
for name, fn in CANDIDATES:
    m_c = int(jax.eval_shape(fn, TH[0]).shape[0])
    mo = fit_joint(m_c, th_fit, x_fit, th_val, x_val, coord_fn=fn, seed=SEED + 33)
    per = np.asarray(mo["nll_vec"](TH_TE, X_TE)); per_of[name] = per
    dif = per - ref_per; se = dif.std(ddof=1) / np.sqrt(len(dif))
    print(f"  {name}")
    print(f"      heldout NLL {per.mean():9.4f}   vs neural {dif.mean():+8.4f} +/- {se:.4f}")

print("\n  --- pairwise paired tests (negative = row is better) ---")
nm = list(per_of)
for i, name in enumerate(nm):
    print(f"   c{i} = {name}")
for a in range(len(nm)):
    for b in range(a + 1, len(nm)):
        dd = per_of[nm[a]] - per_of[nm[b]]
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        print(f"   c{a} - c{b}: {dd.mean():+9.5f} +/- {se:.5f} ({abs(dd.mean() / se):7.1f} sigma)")


# ------------------------------------------------------- plots
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].errorbar(np.arange(M_BOOT), L.mean(0), yerr=L.std(0, ddof=1), fmt="o", capsize=3)
ax[0].axhline(1.0, color="k", ls="--", lw=1, label=r"null $\lambda=1$")
ax[0].set_yscale("log")
ax[0].set_xlabel("canonical axis")
ax[0].set_ylabel(r"$\lambda_i=\mathrm{Var}_\pi(\eta_i)$")
ax[0].set_title(f"spectrum at m={M_BOOT} (no screen)")
ax[0].legend()

ax[1].bar(np.arange(D), col, color=["C1" if j in (I0, I1) else "C0" for j in range(D)])
ax[1].set_xlabel(r"parameter index $j$")
ax[1].set_ylabel(r"RMS $\|\partial\eta/\partial\theta_j\|$")
ax[1].set_title("soft screen (true pair in orange)")

gx = np.linspace(-4, 4, 200)
for j in range(r_hat):
    ax[2].hist(z[:, j], bins=80, density=True, histtype="step", label=fr"$z_{j}$")
ax[2].plot(gx, np.exp(-gx ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1)
ax[2].set_title(f"calibration at $\\hat r$={r_hat}")
ax[2].legend()
plt.tight_layout()
plt.show()
print("\ndone")
