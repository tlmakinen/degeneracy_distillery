# =============================================================================
# Chain heater, one-shot rank protocol -- m unknown, no Fisher, no screen.
#
# All d factors enter the product, so there is no nuisance *parameter* and no
# coordinate screen.  The degeneracy is a nuisance *direction*.  Protocol:
#
#   1. ONE probe fit at m = M_PROBE (generous, rectangular flattener R^d -> R^m)
#      -> information spectrum lambda_i = Var_pi(eta_i), null value 1
#      -> r_hat = #{ i : 0.5 log(lambda_i / null) > NATS_FLOOR }
#   2. refit at r_hat
#   3. diagnose: residual var, R^2 vs product, Jacobian health
#
# Cost: 2 fits.  No value of m is swept.
# =============================================================================
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import flax.linen as nn
import optax
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Sequence, Optional, Callable

jax.config.update("jax_enable_x64", True)

D            = 8
NSIMS        = 4000
NTEST        = 4000
M_PROBE      = 4          # generous over-specification; true r = 1.  Never swept.
NATS_FLOOR   = 0.10
STEPS        = 20000
BATCH        = 512
LR           = 1e-3
SCALE_BOOST  = 8.0
SEED         = 0
NVAL_FRAC    = 0.15


@dataclass
class Cfg:
    theta_min: float = 1.0
    theta_max: float = 2.0
    tau: float = 1.0
    t_max: float = 4.0
    n_t: int = 20
    sigma: float = 0.2

    @property
    def thermal_kernel(self):
        t = np.linspace(0.0, self.t_max * self.tau, self.n_t, dtype=np.float64)
        return 1.0 - np.exp(-t / self.tau)


cfg = Cfg()
LO, HI = cfg.theta_min, cfg.theta_max


def sim(n, d, cfg, rng):
    th = rng.uniform(cfg.theta_min, cfg.theta_max, size=(n, d))
    mean = np.prod(th, axis=1)[:, None] * cfg.thermal_kernel
    return th, mean + rng.normal(scale=cfg.sigma, size=mean.shape)


rng = np.random.default_rng(SEED)
th_np, x_np = sim(NSIMS, D, cfg, rng)
th_te_np, x_te_np = sim(NTEST, D, cfg, np.random.default_rng(SEED + 1))
x_mu, x_sd = x_np.mean(0), x_np.std(0) + 1e-12
TH = jnp.asarray(th_np)
X = jnp.asarray((x_np - x_mu) / x_sd)
TH_TE = jnp.asarray(th_te_np)
X_TE = jnp.asarray((x_te_np - x_mu) / x_sd)

NVAL = max(50, int(NVAL_FRAC * NSIMS))
th_fit, x_fit = TH[:-NVAL], X[:-NVAL]
th_val, x_val = TH[-NVAL:], X[-NVAL:]
print(f"chain heater D={D}, prior U({LO},{HI})^{D}, probe m={M_PROBE}")
print(f"n_train={NSIMS - NVAL}, n_val={NVAL}, n_test={NTEST}, n_t={cfg.n_t}")


# ------------------------------------------------------------------ networks
class Flattener(nn.Module):
    """theta -> eta in R^m.  Skip is the first m standardised coords of theta."""
    m: int
    lo: float
    hi: float
    features: Sequence[int] = (128, 128, 128)

    @nn.compact
    def __call__(self, theta):
        z = 2.0 * (theta - self.lo) / (self.hi - self.lo) - 1.0
        h = jnp.concatenate([z, jnp.log(theta)])
        for w in self.features:
            h = nn.gelu(nn.Dense(w)(h))
        delta = nn.Dense(self.m, kernel_init=nn.initializers.zeros,
                         bias_init=nn.initializers.zeros)(h)
        skip = nn.Dense(
            self.m, use_bias=False,
            kernel_init=lambda k, s, dt: jnp.eye(s[0], s[1], dtype=dt),
        )(z)
        return skip + delta


class Estimator(nn.Module):
    m: int
    features: Sequence[int] = (256, 256, 128)

    @nn.compact
    def __call__(self, x):
        h = x
        for w in self.features:
            h = nn.gelu(nn.Dense(w)(h))
        out = nn.Dense(self.m, kernel_init=nn.initializers.zeros,
                       bias_init=nn.initializers.zeros)(h)
        return out + nn.Dense(self.m, kernel_init=nn.initializers.normal(1e-2))(x)


# ------------------------------------------------------------------ fit
def fit_joint(M, th_fit, x_fit, th_val, x_val, steps=STEPS, seed=SEED,
              coord_fn: Optional[Callable] = None):
    """Rectangular joint fit: all d parameters enter, m output axes."""
    n_fit = th_fit.shape[0]
    bp = min(BATCH, n_fit)
    frozen = coord_fn is not None
    flat_mod = None if frozen else Flattener(m=M, lo=LO, hi=HI)
    est_mod = Estimator(m=M)
    k0, k1 = jr.split(jr.PRNGKey(seed), 2)
    p = {"est": est_mod.init(k1, x_fit[0]), "raw_log_scale": jnp.zeros(M)}
    if frozen:
        p["A"] = jnp.eye(M)
    else:
        p["flat"] = flat_mod.init(k0, th_fit[0])
    const = 0.5 * M * np.log(2 * np.pi)
    eyeM = jnp.eye(M)

    def core(p, theta):
        g = coord_fn(theta) if frozen else flat_mod.apply(p["flat"], theta)
        if frozen:
            g = p["A"] @ g
        return jnp.exp(SCALE_BOOST * p["raw_log_scale"]) * g

    def logdet(p, theta):
        # J: (m, d); volume of the m-dimensional image
        J = jax.jacfwd(lambda t: core(p, t))(theta)
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

    nll_mean = jax.jit(lambda p, t, y: jnp.mean(nll_vec(p, t, y)))
    warm = max(1, min(500, steps // 10))
    sched = optax.warmup_cosine_decay_schedule(1e-5, LR, warm, max(steps, warm + 1), LR * 1e-2)
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
        ok = jnp.all(jnp.stack([jnp.all(jnp.isfinite(z))
                                for z in jax.tree_util.tree_leaves(g)]))
        return jax.tree_util.tree_map(lambda o, n: jnp.where(ok, n, o), p, new), st, l, ok

    key = jr.PRNGKey(seed + 7)
    ev = max(200, steps // 50)
    best, best_val, skipped = p, np.inf, 0
    for i in range(steps):
        key, sk = jr.split(key)
        p, st, l, ok = step(p, st, sk)
        skipped += int(not bool(ok))
        if i % ev == 0 or i == steps - 1:
            v = float(nll_mean(p, th_val, x_val))
            ls = np.round(np.asarray(SCALE_BOOST * p["raw_log_scale"]), 2)
            print(f"    step {i:6d}  train {float(l): .4f}  val {v: .4f}  log_scale {ls}")
            if np.isfinite(v) and v < best_val:
                best_val, best = v, p
    p = best
    if skipped:
        print(f"    (skipped {skipped}/{steps} non-finite updates)")

    def eta_hat_fn(x):
        s = jnp.exp(SCALE_BOOST * p["raw_log_scale"])
        return s * est_mod.apply(p["est"], x)

    return dict(
        m=M, best_val=best_val,
        nll_vec=jax.jit(lambda t, y: nll_vec(p, t, y)),
        z_vec=jax.jit(lambda t, y: zvec(p, t, y)),
        eta=jax.jit(lambda t: jax.vmap(lambda u: core(p, u))(t)),
        eta_hat=jax.jit(lambda x: jax.vmap(eta_hat_fn)(x)),
        jac=jax.jit(lambda t: jax.vmap(lambda u: jax.jacfwd(lambda v: core(p, v))(u))(t)),
    )


def spectrum(eta):
    e = eta - eta.mean(0)
    m = e.shape[1]
    C = np.cov(e, rowvar=False).reshape(m, m)
    lam, V = np.linalg.eigh(C)
    o = np.argsort(-lam)
    return e @ V[:, o], np.maximum(lam[o], 1e-12), V[:, o]


def r2(a, b):
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------- 1. probe -> r_hat
print(f"\n######## 1. single probe fit at m={M_PROBE} (no sweep) ########")
probe = fit_joint(M_PROBE, th_fit, x_fit, th_val, x_val, seed=SEED)
nll_probe = float(np.mean(np.asarray(probe["nll_vec"](TH_TE, X_TE))))
eta_p = np.asarray(probe["eta"](TH_TE))
eta_hat_p = np.asarray(probe["eta_hat"](X_TE))
E_p, lam, _ = spectrum(eta_p)

lam_null = float(np.median(lam[M_PROBE // 2:])) if M_PROBE >= 2 else 1.0
nats_n = 0.5 * np.log(lam / lam_null)
band_samp = (1.0 + np.sqrt(M_PROBE / NTEST)) ** 2 - 1.0

print(f"  test NLL {nll_probe:.4f}")
print(f"  residual var per axis (target ~1): {np.round(((eta_p - eta_hat_p) ** 2).mean(0), 3)}")
print(f"  information spectrum lambda_i : {np.round(lam, 3)}")
print(f"  measured null level           : {lam_null:.4f}  (exactly 1 at the optimum)")
print(f"  profile 0.5 log(lambda/null)  : {np.round(nats_n, 4)} nats")
print(f"  sampling-only band on lambda  : 1 + {band_samp:.4f}"
      f"  ({0.5 * np.log1p(band_samp):.4f} nats)")

print("  sensitivity of r_hat to the floor:")
for f in (0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00):
    print(f"    floor {f:4.2f} nats -> r_hat = {int(np.sum(nats_n > f))}")
r_hat = int(np.sum(nats_n > NATS_FLOOR))
print(f"  adopt floor {NATS_FLOOR} nats  ->  r_hat = {r_hat}")
if r_hat >= M_PROBE:
    print("  WARNING: r_hat reached the probe dimension. Increase M_PROBE and refit.")
if r_hat < M_PROBE:
    print(f"  discarded block: lambda in [{lam[r_hat:].min():.3f}, {lam[r_hat:].max():.3f}]"
          f"  (null value 1)")

P = np.prod(th_te_np, axis=1)
knorm = float(np.linalg.norm(cfg.thermal_kernel))
pred = (knorm ** 2 / cfg.sigma ** 2) * P.var()
print(f"  analytic prediction for leading eigenvalue: {pred:.4g}  (measured {lam[0]:.4g})")
print(f"  [truth] R^2(leading axis vs product P): {r2(E_p[:, 0], P):.6f}")
print(f"  [truth] R^2(leading axis vs log P)   : {r2(E_p[:, 0], np.log(P)):.6f}")
print(f"  [truth] Spearman vs P                : {spearman(E_p[:, 0], P):.6f}")

# ------------------------------------------------------- 2. refit at r_hat
print(f"\n######## 2. refit at r_hat={r_hat} ########")
ref = fit_joint(r_hat, th_fit, x_fit, th_val, x_val, seed=SEED + 11)
nll_hat = float(np.mean(np.asarray(ref["nll_vec"](TH_TE, X_TE))))
eta = np.asarray(ref["eta"](TH_TE))
eta_hat = np.asarray(ref["eta_hat"](X_TE))
E, lam_hat, _ = spectrum(eta)
Js = np.asarray(ref["jac"](TH_TE))

print(f"  test NLL {nll_hat:.4f}   (probe was {nll_probe:.4f},"
      f" spare cost {(nll_probe - nll_hat) / max(M_PROBE - r_hat, 1):.4f} nats/axis)")
print(f"  residual var per axis (target ~1): {np.round(((eta - eta_hat) ** 2).mean(0), 3)}")
print(f"  spectrum at r_hat: {np.round(lam_hat, 3)}"
      f"   profile {np.round(0.5 * np.log(lam_hat), 4)} nats")
print(f"  analytic leading eigenvalue: {pred:.4g}  (measured {lam_hat[0]:.4g})")

print("\n--- what the retained axes are functions of ---")
for j in range(r_hat):
    print(f"  axis {j}: R^2 vs P {r2(E[:, j], P):.6f}"
          f"   R^2 vs log P {r2(E[:, j], np.log(P)):.6f}"
          f"   Spearman {spearman(E[:, j], P):.6f}")
print("  (R^2 vs P ~ 1 on axis 0 => the SR target is X1*X2*...*Xd)")

print("\n--- Jacobian health ---")
# For rectangular J (m, d), report volume factors of J J^T
vol = np.asarray(jax.vmap(
    lambda J: 0.5 * jnp.linalg.slogdet(J @ J.T + 1e-12 * jnp.eye(J.shape[0]))[1]
)(jnp.asarray(Js)))
print(f"  median sqrt(det JJ^T) {np.exp(np.median(vol)):.4g}"
      f"   10-90 pct [{np.exp(np.percentile(vol, 10)):.3g},"
      f" {np.exp(np.percentile(vol, 90)):.3g}]")

print("\n--- calibration at r_hat ---")
z = np.asarray(ref["z_vec"](TH_TE, X_TE))
print(f"  z: mean {np.round(z.mean(0), 4)}  std {np.round(z.std(0), 4)}  (want 0, 1)")
for lev, nom in ((1.0, 0.6827), (1.96, 0.9500), (2.58, 0.9901)):
    print(f"    |z| < {lev:4.2f}: {np.round((np.abs(z) < lev).mean(0), 4)}  (nominal {nom:.4f})")

# Probe residuals: same z = eta - etahat, but in the M_PROBE-dimensional probe map.
# Expectation: still ~N(0,1) on EVERY axis (including spares).  The loss trains
# that directly.  Overspecification shows up in the spectrum / NLL, not as broken z.
print("\n--- calibration of the m=M_PROBE probe (for comparison) ---")
z_probe = np.asarray(probe["z_vec"](TH_TE, X_TE))
# Rotate probe residuals into the canonical (spectrum) basis so axis 0 is the
# informative one and axes 1.. are the near-null block.
_, _, V_p = spectrum(eta_p)
z_probe_c = (z_probe - z_probe.mean(0)) @ V_p   # same gauge as lambda_i
# re-center after rotation is already mean-zero; restore raw mean in print only
print(f"  probe z (canonical): mean {np.round(z_probe_c.mean(0), 4)}"
      f"  std {np.round(z_probe_c.std(0), 4)}  (want 0, 1 on all {M_PROBE} axes)")
for lev, nom in ((1.0, 0.6827), (1.96, 0.9500)):
    print(f"    |z| < {lev:4.2f}: {np.round((np.abs(z_probe_c) < lev).mean(0), 4)}"
          f"  (nominal {nom:.4f})")

# ------------------------------------------------------- 3. frozen product candidate
print("\n######## 3. frozen true candidate (product) vs neural map ########")
cand = fit_joint(
    1, th_fit, x_fit, th_val, x_val, seed=SEED + 22,
    coord_fn=lambda t: jnp.stack([jnp.prod(t)]),
)
per_ref = np.asarray(ref["nll_vec"](TH_TE, X_TE))
per_c = np.asarray(cand["nll_vec"](TH_TE, X_TE))
dif = per_c - per_ref
se = dif.std(ddof=1) / np.sqrt(len(dif))
print(f"  neural (m={r_hat}): {per_ref.mean():.4f}")
print(f"  product (frozen)  : {per_c.mean():.4f}   delta {dif.mean():+.4f} +/- {se:.4f}"
      f"  ({abs(dif.mean()) / se:.1f} sigma)")

# ------------------------------------------------------- plots
fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
ax[0].semilogy(np.arange(M_PROBE), lam, "o-", label="probe")
if r_hat < M_PROBE:
    ax[0].axhline(lam_null, color="k", ls="--", lw=1, label=fr"null $\lambda={lam_null:.2f}$")
ax[0].axhline(1.0, color="0.5", ls=":", lw=1, label=r"exact null $\lambda=1$")
ax[0].set_xlabel("canonical axis")
ax[0].set_ylabel(r"$\lambda_i=\mathrm{Var}_\pi(\eta_i)$")
ax[0].set_title(f"information spectrum at m={M_PROBE}")
ax[0].legend()

gx = np.linspace(-4, 4, 200)
gauss = np.exp(-gx ** 2 / 2) / np.sqrt(2 * np.pi)
for j in range(M_PROBE):
    ax[1].hist(z_probe_c[:, j], bins=80, density=True, histtype="step",
               label=fr"$z_{j}$" + (" (kept)" if j < r_hat else " (spare)"))
ax[1].plot(gx, gauss, "k--", lw=1)
ax[1].set_title(f"probe residuals at m={M_PROBE}")
ax[1].legend(fontsize=8)

for j in range(r_hat):
    ax[2].hist(z[:, j], bins=80, density=True, histtype="step", label=fr"$z_{j}$")
ax[2].plot(gx, gauss, "k--", lw=1)
ax[2].set_title(f"calibration at $\\hat r$={r_hat}")
ax[2].legend()

ax[3].scatter(P, E[:, 0], s=4, alpha=0.35, edgecolors="none")
ax[3].set_xlabel(r"product $P=\prod_i\theta_i$")
ax[3].set_ylabel(r"leading canonical axis $\eta_0$")
ax[3].set_title(f"$R^2$={r2(E[:, 0], P):.4f}")
plt.tight_layout()
plt.show()
print("\ndone")
