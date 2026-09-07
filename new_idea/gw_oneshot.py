# =============================================================================
# GW TaylorF2 joint oneshot (m=2) + Operon SR
#
# Data generation / PCA / theta scaling match scripts/gw_notebook_run.py.
# Joint objective (no Fishnets, no separate flattener stage):
#
#   L = E[ 0.5 ||eta(theta) - etahat(x)||^2 - 0.5 log det(J J^T) ]
#
# with eta: R^2 -> R^2 (m=2).  Bootstrap ensemble -> (X, y, y_std) in the
# same convention as gw_notebook_run.align_and_sample_sr_grid, then
# fit_symbolic_regression(...).
#
# Outputs handed to Operon:
#   X      <-> theta_scaled   (variables X1=m1_s, X2=m2_s)
#   y      <-> eta            (ensemble-mean, min-shifted)
#   y_std  <-> deta           (ensemble std after Procrustes)
# =============================================================================
from __future__ import annotations

# Must be set before NumPy/sklearn/JAX import.  On Colab, OpenBLAS×JAX often
# deadlocks inside sklearn.PCA.fit (looks like a hang at "building PCA basis").
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathlib import Path
from typing import Sequence, Optional, Callable

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from degeneracy_distillery.align_coords import (
    process_ensemble_rotation_v2,
    nonlinearity_spectrum,
)
from degeneracy_distillery.postprocessing_utils import weighted_std
from degeneracy_distillery.sr_utils import (
    fit_theta_scaler,
    fit_symbolic_regression,
    filter_pareto_fronts,
    analyze_equations,
    sr_structure_predicate,
)

# ------------------------------------------------------------------ config
# Match gw_notebook_run CONFIGS["full"] data settings; SR knobs match that file too.
NSIMS        = 1000
N_PCA        = 40
PCA_BASIS_MAX = 5000
M            = 2              # fixed output dim (d = 2 as well)
STEPS        = 8000
BATCH        = 128
BATCH_AUG    = 256            # theta-only volume samples (no simulator)
LR           = 5e-4
SCALE_BOOST  = 1.0
WEIGHT_DECAY = 1e-3
PATIENCE     = 20             # early-stop evals without val improvement
SEED         = 0
K_BOOT       = 8
ALIGN_SUBSAMPLE = 1000        # points used for Jacobian-Procrustes / nonlinearity basis
SR_GRID_SIZE = 2000
SR_TIME_LIMIT = 120
SR_MAX_LENGTH = 30
SR_MAX_DEPTH  = 20
SR_LENGTH_PENALTY = 2.0
RUN_SR       = True           # set False to stop after dumping (X,y,y_std)
OUTDIR       = Path("new_idea/gw_oneshot_out")

M_SUN_SEC = 4.925491025543576e-6
MPC_SEC = 1.0292712503e14
M1_MIN, M1_MAX = 5.0, 50.0
M2_MIN, M2_MAX = 5.0, 50.0
D_L_MPC = 200.0
F_LOW, DF = 20.0, 0.5

OUTDIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------- GW data (gw_notebook_run)
def chirp_mass(m1, m2):
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def symmetric_mass_ratio(m1, m2):
    return (m1 * m2) / (m1 + m2) ** 2


def a_ligo_psd(f):
    f = np.asarray(f, dtype=float)
    x = f / 215.0
    psd = 1e-49 * (x**-4.14 + 2.0 + 2.0 * x**2)
    return np.where(f >= 10.0, psd, np.inf)


def taylor_f2_waveform(m1, m2, freqs, d_l_mpc=D_L_MPC):
    m_sec = (m1 + m2) * M_SUN_SEC
    eta = m1 * m2 / (m1 + m2) ** 2
    mc_sec = chirp_mass(m1, m2) * M_SUN_SEC
    d_l_sec = d_l_mpc * MPC_SEC

    f_isco = 1.0 / (6**1.5 * np.pi * m_sec)
    active = (freqs > 0) & (freqs < f_isco)
    v = np.zeros_like(freqs)
    v[active] = (np.pi * m_sec * freqs[active]) ** (1.0 / 3)

    amp = np.zeros_like(freqs)
    amp[active] = (
        np.sqrt(5.0 / 24)
        * np.pi ** (-2.0 / 3)
        * mc_sec ** (5.0 / 6)
        / d_l_sec
        * freqs[active] ** (-7.0 / 6)
    )

    c1 = 3715.0 / 756 + 55.0 / 9 * eta
    c15 = -16.0 * np.pi
    c2 = 15379365.0 / 508032 + 27145.0 / 504 * eta + 3085.0 / 72 * eta**2

    phase = np.zeros_like(freqs)
    vm = v[active]
    phase[active] = 3.0 / (128 * eta) * vm**-5 * (1 + c1 * vm**2 + c15 * vm**3 + c2 * vm**4)
    return amp * np.exp(1j * phase)


def simulator_data(nsims: int, seed: int) -> dict:
    """Same pipeline as scripts/gw_notebook_run.simulator_data (train+test split).

    Noise + PCA transform are vectorised (the per-sample jax.random loop in the
    notebook looks hung on Colab; GPU does not accelerate this stage anyway).
    """
    import time as _time
    rng = np.random.default_rng(seed)

    f_isco_lightest = 1.0 / (6**1.5 * np.pi * (M1_MIN + M2_MIN) * M_SUN_SEC)
    freqs = np.arange(F_LOW, min(f_isco_lightest, 1024.0), DF)
    whiten = np.sqrt(4.0 * DF) / np.sqrt(a_ligo_psd(freqs))
    n_freq = len(freqs)
    n_tot = 2 * nsims
    log(f"frequency grid: {F_LOW}-{freqs[-1]:.0f} Hz, {n_freq} bins  ({n_tot} waveforms)")

    m1_all = rng.uniform(M1_MIN, M1_MAX, n_tot)
    m2_all = rng.uniform(M2_MIN, M2_MAX, n_tot)
    theta_all = np.stack([m1_all, m2_all], axis=1).astype(np.float32)

    log("computing noiseless waveforms (shared by PCA basis fit + noisy data)")
    t0 = _time.time()
    hvecs_all = np.empty((n_tot, 2 * n_freq), dtype=np.float64)
    for i in tqdm(range(n_tot), desc="waveforms"):
        h = taylor_f2_waveform(theta_all[i, 0], theta_all[i, 1], freqs) * whiten
        hvecs_all[i] = np.concatenate([h.real, h.imag])
    log(f"  waveforms done in {_time.time() - t0:.1f}s")

    n_basis = min(n_tot, PCA_BASIS_MAX)
    log(f"building PCA basis (n_components={N_PCA}, n_basis={n_basis}, "
        f"svd_solver=randomized) -- if this stalls, restart the runtime so "
        f"OMP_NUM_THREADS=1 is set before NumPy loads")
    t0 = _time.time()
    basis_idx = rng.choice(n_tot, n_basis, replace=False)
    # randomized SVD avoids the full LAPACK path that deadlocks with JAX on Colab
    pca = PCA(
        n_components=N_PCA,
        svd_solver="randomized",
        random_state=seed,
        iterated_power=2,
    ).fit(hvecs_all[basis_idx])
    cumvar = pca.explained_variance_ratio_.cumsum()
    log(f"  PCA done in {_time.time() - t0:.1f}s; "
        f"{N_PCA} components capture {cumvar[-1] * 100:.1f}% variance")

    log("generating noisy PCA coefficients (vectorised)")
    t0 = _time.time()
    snr_all = np.linalg.norm(hvecs_all, axis=1)
    # Same N(0,1) whitened noise as jr.normal in gw_notebook_run, via numpy RNG
    noise = rng.normal(size=hvecs_all.shape)
    data_all = pca.transform(hvecs_all + noise).astype(np.float32)
    log(f"  noisy data done in {_time.time() - t0:.1f}s")

    return {
        "theta_train": theta_all[:nsims],
        "data_train": data_all[:nsims],
        "theta_test": theta_all[nsims:],
        "data_test": data_all[nsims:],
        "snr_train": snr_all[:nsims],
        "cumvar": cumvar,
    }


# ------------------------------------------------------------------ networks
class Flattener(nn.Module):
    """theta_scaled in [lo,hi]^2 -> eta in R^m.  log features help chirp-like products."""
    m: int
    lo: float = 1.0
    hi: float = 2.0
    # Small nets: n~1000 + 40-D PCA estimator overfits a (128,128,128) stack easily.
    features: Sequence[int] = (64, 64)

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
    features: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, x):
        h = x
        for w in self.features:
            h = nn.gelu(nn.Dense(w)(h))
        out = nn.Dense(self.m, kernel_init=nn.initializers.zeros,
                       bias_init=nn.initializers.zeros)(h)
        return out + nn.Dense(self.m, kernel_init=nn.initializers.normal(1e-2))(x)


def fit_joint(M, th_fit, x_fit, th_val, x_val, steps=STEPS, seed=SEED,
              coord_fn: Optional[Callable] = None):
    n_fit = th_fit.shape[0]
    bp = min(BATCH, n_fit)
    frozen = coord_fn is not None
    flat_mod = None if frozen else Flattener(m=M, lo=1.0, hi=2.0)
    est_mod = Estimator(m=M)
    k0, k1 = jr.split(jr.PRNGKey(seed), 2)
    p = {"est": est_mod.init(k1, x_fit[0]), "raw_log_scale": jnp.zeros(M)}
    if frozen:
        p["A"] = jnp.eye(M)
    else:
        p["flat"] = flat_mod.init(k0, th_fit[0])
    const = 0.5 * M * np.log(2 * np.pi)
    eyeM = jnp.eye(M)
    # Scaled-theta box for free volume samples (matches MinMax feature_range).
    th_lo = jnp.asarray(th_fit.min(0))
    th_hi = jnp.asarray(th_fit.max(0))

    def core(p, theta):
        g = coord_fn(theta) if frozen else flat_mod.apply(p["flat"], theta)
        if frozen:
            g = p["A"] @ g
        return jnp.exp(SCALE_BOOST * p["raw_log_scale"]) * g

    def logdet(p, theta):
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

    def geom(p, th_b):
        return -jnp.mean(jax.vmap(lambda t: logdet(p, t))(th_b))

    nll_mean = jax.jit(lambda p, t, y: jnp.mean(nll_vec(p, t, y)))
    warm = max(1, min(300, steps // 10))
    sched = optax.warmup_cosine_decay_schedule(1e-5, LR, warm, max(steps, warm + 1), LR * 1e-2)
    tx = optax.chain(optax.clip_by_global_norm(10.0),
                     optax.adamw(sched, weight_decay=WEIGHT_DECAY))
    st = tx.init(p)

    @jax.jit
    def step(p, st, key):
        # Residuals on simulated pairs; volume term on fresh theta draws (no x).
        # Coupling both to the same tiny train batch is what drove val NLL to 1e3.
        kp, ka = jr.split(key)
        idx = jr.randint(kp, (bp,), 0, n_fit)
        u = jr.uniform(ka, (BATCH_AUG, th_fit.shape[1]))
        th_aug = th_lo + u * (th_hi - th_lo)
        def obj(p):
            z = zvec(p, th_fit[idx], x_fit[idx])
            return 0.5 * jnp.mean(jnp.sum(z ** 2, 1)) + geom(p, th_aug) + const
        l, g = jax.value_and_grad(obj)(p)
        upd, st = tx.update(g, st, p)
        new = optax.apply_updates(p, upd)
        ok = jnp.all(jnp.stack([jnp.all(jnp.isfinite(z))
                                for z in jax.tree_util.tree_leaves(g)]))
        return jax.tree_util.tree_map(lambda o, n: jnp.where(ok, n, o), p, new), st, l, ok

    key = jr.PRNGKey(seed + 7)
    ev = max(50, steps // 100)
    best, best_val, best_step, skipped = p, np.inf, -1, 0
    bad_evals = 0
    for i in range(steps):
        key, sk = jr.split(key)
        p, st, l, ok = step(p, st, sk)
        skipped += int(not bool(ok))
        if i % ev == 0 or i == steps - 1:
            v = float(nll_mean(p, th_val, x_val))
            ls = np.round(np.asarray(SCALE_BOOST * p["raw_log_scale"]), 2)
            marker = ""
            if np.isfinite(v) and v < best_val:
                best_val, best, best_step = v, p, i
                bad_evals = 0
                marker = "  *"
            else:
                bad_evals += 1
            log(f"    step {i:6d}  train {float(l): .4f}  val {v: .4f}  log_scale {ls}{marker}")
            if bad_evals >= PATIENCE and best_step >= 0:
                log(f"    early stop at step {i} (best val {best_val:.4f} at step {best_step})")
                break
    if not np.isfinite(best_val):
        best, best_val, best_step = p, float(nll_mean(p, th_val, x_val)), steps - 1
    p = best
    log(f"    using checkpoint step {best_step}  val={best_val:.4f}")
    if skipped:
        log(f"    (skipped {skipped}/{steps} non-finite updates)")

    eta_fn = jax.jit(lambda t: jax.vmap(lambda u: core(p, u))(t))
    jac_fn = jax.jit(lambda t: jax.vmap(lambda u: jax.jacfwd(lambda v: core(p, v))(u))(t))
    return dict(
        m=M, best_val=best_val, best_step=best_step, params=p,
        nll_vec=jax.jit(lambda t, y: nll_vec(p, t, y)),
        z_vec=jax.jit(lambda t, y: zvec(p, t, y)),
        eta=eta_fn,
        jac=jac_fn,
    )


def spectrum(eta):
    e = eta - eta.mean(0)
    m = e.shape[1]
    C = np.cov(e, rowvar=False).reshape(m, m)
    lam, V = np.linalg.eigh(C)
    o = np.argsort(-lam)
    return e @ V[:, o], np.maximum(lam[o], 1e-12), V[:, o]


def r2(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.corrcoef(a, b)[0, 1] ** 2)


# ================================================================ main
log("######## 0. GW TaylorF2 data (gw_notebook_run conventions) ########")
data = simulator_data(NSIMS, seed=SEED)

# MinMax scale theta -> [1, 2], same as train_fishnet_ensemble in gw_notebook_run
scaler = fit_theta_scaler(data["theta_train"], feature_range=(1.0, 2.0))
th_tr = scaler.transform(data["theta_train"]).astype(np.float64)
th_te = scaler.transform(data["theta_test"]).astype(np.float64)
log(f"scaled theta range: {th_tr.min(0)} to {th_tr.max(0)}")

# Standardise PCA coefficients (estimator inputs)
x_mu = data["data_train"].mean(0)
x_sd = data["data_train"].std(0) + 1e-12
x_tr = ((data["data_train"] - x_mu) / x_sd).astype(np.float64)
x_te = ((data["data_test"] - x_mu) / x_sd).astype(np.float64)

NVAL = max(50, int(0.15 * NSIMS))
TH = jnp.asarray(th_tr); X = jnp.asarray(x_tr)
TH_TE = jnp.asarray(th_te); X_TE = jnp.asarray(x_te)
th_fit, x_fit = TH[:-NVAL], X[:-NVAL]
th_val, x_val = TH[-NVAL:], X[-NVAL:]

mc_te = chirp_mass(data["theta_test"][:, 0], data["theta_test"][:, 1])
q_te = data["theta_test"][:, 0] / data["theta_test"][:, 1]
eta_sym_te = symmetric_mass_ratio(data["theta_test"][:, 0], data["theta_test"][:, 1])


# ------------------------------------------------------- 1. joint fit m=2
log(f"\n######## 1. joint fit at m={M} ########")
ref = fit_joint(M, th_fit, x_fit, th_val, x_val, seed=SEED)
nll = float(np.mean(np.asarray(ref["nll_vec"](TH_TE, X_TE))))
eta = np.asarray(ref["eta"](TH_TE))
E, lam, V = spectrum(eta)
z = np.asarray(ref["z_vec"](TH_TE, X_TE))
z_c = (z - z.mean(0)) @ V

log(f"  test NLL {nll:.4f}")
log(f"  information spectrum lambda: {np.round(lam, 3)}")
log(f"  residual std (canonical): {np.round(z_c.std(0), 4)}  (want ~1)")
log(f"  R^2(eta_c0 vs Mc):     {r2(E[:, 0], mc_te):.4f}")
log(f"  R^2(eta_c1 vs Mc):     {r2(E[:, 1], mc_te):.4f}")
log(f"  R^2(eta_c0 vs m1/m2):  {r2(E[:, 0], q_te):.4f}")
log(f"  R^2(eta_c1 vs m1/m2):  {r2(E[:, 1], q_te):.4f}")
log(f"  R^2(eta_c0 vs eta_sym):{r2(E[:, 0], eta_sym_te):.4f}")
log(f"  R^2(eta_c1 vs eta_sym):{r2(E[:, 1], eta_sym_te):.4f}")


# ------------------------------------------------------- 2. bootstrap + align_coords
log(f"\n######## 2. bootstrap + align_coords Procrustes / nonlinearity ########")
boot_rng = np.random.default_rng(SEED + 31)
n = TH.shape[0]
n_te = TH_TE.shape[0]
etas_te, jacs_te, etas_grid_members, oob_nlls = [], [], [], []

# SR grid in scaled-theta box (same construction as gw_notebook_run)
key_sr = jr.PRNGKey(SEED + 5 * 10_000)  # STAGE_OFFSETS["sr_grid"] * stride
X_sr = np.asarray(jr.uniform(
    key_sr, minval=TH.min(0), maxval=TH.max(0), shape=(SR_GRID_SIZE, M),
))

for k in range(K_BOOT):
    idx = boot_rng.integers(0, n, n)
    oob = np.setdiff1d(np.arange(n), np.unique(idx))
    if len(oob) < 20:
        oob = np.arange(min(50, n))
    mo = fit_joint(M, TH[idx], X[idx], TH[oob], X[oob], seed=SEED + 500 + k)
    etas_te.append(np.asarray(mo["eta"](TH_TE)))
    jacs_te.append(np.asarray(mo["jac"](TH_TE)))
    etas_grid_members.append(np.asarray(mo["eta"](jnp.asarray(X_sr))))
    oob_nlls.append(float(mo["best_val"]))
    log(f"  member {k}: oob NLL {mo['best_val']:.4f}")

# Subsample for alignment (Jacobian Procrustes + nonlinearity basis)
n_align = min(ALIGN_SUBSAMPLE, n_te)
randidx = boot_rng.choice(n_te, n_align, replace=False)
randidx.sort()
# align_coords expects θ-space Fishers (n, d, d).  No Fishnets here, so use
# the pullback F ≈ J^T J from each member's Jacobian (n, m, d).
jacs_stack = np.stack(jacs_te)  # (K, n_te, m, d)
F_ensemble = np.einsum("knmi,knmj->knij", jacs_stack, jacs_stack)
datafile = {
    "theta": np.asarray(th_te),
    "eta_ensemble": np.stack(etas_te),          # (K, n_te, m)
    "Jbar_ensemble": jacs_stack,                # (K, n_te, m, d)
    "F_ensemble": F_ensemble,                   # (K, n_te, d, d)
    "ensemble_weights": np.ones(K_BOOT) / K_BOOT,
    "norm_factor": 1.0,
}
best_model_idx = int(np.argmin(oob_nlls))
Favg = F_ensemble[best_model_idx][randidx]  # (n_align, d, d)
log(f"  align_coords: reference=member {best_model_idx} (best oob), "
    f"n_align={n_align}, F shape={Favg.shape} (θ-space JᵀJ), "
    f"separate_nonlinearity=True, canonicalize=permute_and_sign")
aligned = process_ensemble_rotation_v2(
    datafile=datafile,
    randidx=randidx,
    Favg=Favg,
    best_model_idx=best_model_idx,
    n_d=1.0,
    align_mode="procrustes",
    separate_nonlinearity=True,
    canonicalize="permute_and_sign",
    use_prior_normalization=True,
    restore_reference_mean=True,
    Fisher_to_flatten="best",
    verbose=True,
    offset_delta=0.1,
)

# Axis 0 = most nonlinear after nonlinearity_rotation (align_coords convention).
y_te = np.asarray(aligned["y"])
y_std_te_raw = np.asarray(aligned["y_std"])
X_aligned = np.asarray(aligned["X"])
dy_sr_te = np.asarray(aligned["dy_sr"])
Ens_te = np.asarray(aligned["ys"])          # (K, n_align_masked, m)
rotmats = np.asarray(aligned["rotmats"])
w_ens = np.asarray(aligned["ensemble_weights"])
nl_sigma = nonlinearity_spectrum(np.stack(jacs_te)[best_model_idx][randidx])
log(f"  nonlinearity spectrum sigma (ref member): {np.round(nl_sigma, 4)}")
log(f"  -> axis 0 is most nonlinear; axis {M - 1} is most linear")

# Physics checks in the aligned frame (X is scaled theta on the align subsample)
mc_al = chirp_mass(
    scaler.inverse_transform(X_aligned)[:, 0],
    scaler.inverse_transform(X_aligned)[:, 1],
)
q_al = scaler.inverse_transform(X_aligned)[:, 0] / scaler.inverse_transform(X_aligned)[:, 1]
for j in range(M):
    tag = "nonlinear" if j == 0 else ("linear" if j == M - 1 else f"axis{j}")
    log(f"  aligned eta_{j} ({tag}): R^2 vs Mc={r2(y_te[:, j], mc_al):.4f}  "
        f"vs m1/m2={r2(y_te[:, j], q_al):.4f}")

# SR grid: rotate each member with its rotmat (gw_notebook_run convention)
ys_sr = np.stack(etas_grid_members)                         # (K, n_sr, m)
ys_sr_rot = np.array([
    np.einsum("ij,bj->bi", rotmats[i], ys_sr[i] - ys_sr[i].mean(0))
    for i in range(K_BOOT)
])
y_std_sr_raw = np.asarray(weighted_std(jnp.asarray(ys_sr_rot), weights=jnp.asarray(w_ens)))
y_sr = np.average(ys_sr_rot, 0, w_ens)
ys_sr_rot -= y_sr.min(0)
y_sr -= y_sr.min(0)

# Leave-one-out calibration of bootstrap y_std on the aligned test ensemble.
K = K_BOOT
eta_mu = Ens_te.mean(0)
mu_loo = (K * eta_mu[None] - Ens_te) / (K - 1)
sd_loo = np.sqrt(np.clip(
    (K * (Ens_te.var(0, ddof=0)[None] + eta_mu[None] ** 2) - Ens_te ** 2) / (K - 1)
    - mu_loo ** 2, 1e-24, None))
sd_loo *= np.sqrt((K - 1) / (K - 2))
z_loo = (Ens_te - mu_loo) / (sd_loo * np.sqrt(K / (K - 1)))
zstd = z_loo.reshape(-1, M).std(0)
tref = np.sqrt((K - 2) / (K - 4)) if K > 4 else np.nan
log(f"  map spread (aligned test, median raw): {np.round(np.median(y_std_te_raw, 0), 4)}")
if K > 4:
    log(f"  LOO ensemble z: std {np.round(zstd, 3)}  "
        f"reference std(t_{K-2})={tref:.4f}  "
        f"ratio {np.round(zstd / tref, 3)}  (>1 = y_std too small)")
else:
    log(f"  LOO ensemble z: std {np.round(zstd, 3)}  (need K>4 for t reference)")

Y_STD_FLOOR = 1e-3
y_std_te = np.maximum(y_std_te_raw, Y_STD_FLOOR)
y_std_sr = np.maximum(y_std_sr_raw, Y_STD_FLOOR)
frac_floored = (y_std_te_raw < Y_STD_FLOOR).mean(0)
log(f"  fraction of align points hitting y_std floor {Y_STD_FLOOR}: "
    f"{np.round(frac_floored, 3)}")
log(f"  y_sr shape {y_sr.shape}, y_std_sr median {np.median(y_std_sr, 0)}")

# Pack Operon inputs under the gw names
#   (eta, deta, theta) = (y, y_std, X)  — X/y on the align subsample after mask
X = X_aligned
y = y_te
y_std = y_std_te
dy_sr = dy_sr_te

np.savez(
    OUTDIR / "gw_joint_sr_inputs.npz",
    X=X, y=y, y_std=y_std, dy_sr=dy_sr,
    X_sr=X_sr, y_sr=y_sr, y_std_sr=y_std_sr,
    theta_test_physical=data["theta_test"],
    scaler_scale=scaler.scale_,
    scaler_min=scaler.min_,
    rotmats=rotmats,
    nonlinearity_sigma=nl_sigma,
    best_model_idx=best_model_idx,
)
log(f"  wrote {OUTDIR / 'gw_joint_sr_inputs.npz'}")
log("  Operon mapping: X <-> theta_scaled, y <-> eta (nonlinear-first), "
    "y_std <-> ensemble deta")


# ------------------------------------------------------- 3. Operon SR (gw conventions)
if RUN_SR:
    sr_dir = OUTDIR / "sr_results_gw"
    sr_dir.mkdir(exist_ok=True)
    log(f"\n######## 3. Operon SR into {sr_dir} ########")
    fit_symbolic_regression(
        X_sr,
        y_sr,
        y_std_sr,
        parent_dir=str(sr_dir) + os.sep,
        random_state=SEED + 6 * 10_000,  # STAGE_OFFSETS["sr_fit"]
        time_limit=SR_TIME_LIMIT,
        max_length=SR_MAX_LENGTH,
        max_depth=SR_MAX_DEPTH,
        allowed_symbols="add,mul,div,pow,constant,variable,exp",
        objectives=["r2", "length"],
    )

    n_params = X.shape[1]
    equation_predicate = sr_structure_predicate(
        n_params=n_params,
        forbid_self_transcendental=True,
    )
    filter_summaries = filter_pareto_fronts(
        str(sr_dir), n_params, equation_predicate,
    )
    removed = sum(int(s["removed"]) for s in filter_summaries)
    log(f"  removed {removed} self-transcendental/invalid equations from Pareto fronts")

    # Fs placeholder: joint method has no Fishnets F.  Use identity so
    # analyze_equations' Frobenius term is well-defined but not physics-meaningful.
    Fs_id = np.repeat(np.eye(n_params)[None, ...], len(X), axis=0)
    mdl_coords, frob_coords, analysis = analyze_equations(
        X, y, y_std, dy_sr, Fs_id,
        parent_dir=str(sr_dir) + os.sep,
        n_params=n_params,
        equation_set="pareto",
        max_complexity_thresh=20,
        length_penalty=SR_LENGTH_PENALTY,
        equation_predicate=equation_predicate,
    )
    log(f"  MDL coords:  {mdl_coords}")
    log(f"  Frob coords: {frob_coords}")
else:
    log("\n######## 3. RUN_SR=False; skipping Operon ########")
    log("  load new_idea/gw_oneshot_out/gw_joint_sr_inputs.npz for external SR")


# ------------------------------------------------------- plots
gx = np.linspace(-4, 4, 200)
gauss = np.exp(-gx ** 2 / 2) / np.sqrt(2 * np.pi)

fig, ax = plt.subplots(2, 3, figsize=(14, 7.2))

ax[0, 0].semilogy(np.arange(M), lam, "o-")
ax[0, 0].axhline(1.0, color="k", ls="--", lw=1, label=r"null $\lambda=1$")
ax[0, 0].set_xlabel("canonical axis")
ax[0, 0].set_ylabel(r"$\lambda_i$")
ax[0, 0].set_title(f"spectrum at m={M}")
ax[0, 0].legend()

for j in range(M):
    ax[0, 1].hist(z_c[:, j], bins=60, density=True, histtype="step", label=fr"$z_{j}$")
ax[0, 1].plot(gx, gauss, "k--", lw=1)
ax[0, 1].set_title(r"joint residuals $z=\eta-\hat\eta$")
ax[0, 1].legend()

# Aligned frame: axis 0 = most nonlinear (align_coords convention)
th_al_phys = scaler.inverse_transform(X_aligned)
sc = ax[0, 2].scatter(th_al_phys[:, 0], th_al_phys[:, 1],
                      c=y_te[:, 0], s=6, cmap="viridis", alpha=0.7)
plt.colorbar(sc, ax=ax[0, 2], label=r"$\eta_0$ (most nonlinear)")
ax[0, 2].set_xlabel(r"$m_1$"); ax[0, 2].set_ylabel(r"$m_2$")
ax[0, 2].set_title(rf"aligned $\eta_0$ ($R^2$ Mc={r2(y_te[:,0], mc_al):.2f})")

# Bootstrap y_std calibration (LOO z / std(t_{K-2}) should look N(0,1))
scale = tref if (K > 4 and np.isfinite(tref)) else 1.0
for j in range(M):
    ax[1, 0].hist((z_loo[..., j] / scale).ravel(), bins=80, density=True,
                  histtype="step",
                  label=fr"axis {j}" + (" (nl)" if j == 0 else " (lin)"))
ax[1, 0].plot(gx, gauss, "k--", lw=1, label="N(0,1)")
ax[1, 0].set_title(r"bootstrap LOO $z$ / std$(t_{K-2})$" if K > 4
                   else r"bootstrap LOO $z$")
ax[1, 0].legend(fontsize=8)

for j in range(M):
    ax[1, 1].hist(y_std_te_raw[:, j], bins=60, histtype="step",
                  label=fr"$y_{{\mathrm{{std}},{j}}}$")
ax[1, 1].axvline(Y_STD_FLOOR, color="k", ls="--", lw=1, label="floor")
ax[1, 1].set_xlabel(r"ensemble $y_{\mathrm{std}}$ (raw, aligned)")
ax[1, 1].set_title("bootstrap spread after align_coords")
ax[1, 1].legend(fontsize=8)

ax[1, 2].scatter(mc_al, y_te[:, 0], s=6, alpha=0.5, edgecolors="none", label=r"$\eta_0$ (nl)")
ax[1, 2].scatter(mc_al, y_te[:, 1], s=6, alpha=0.35, edgecolors="none", label=r"$\eta_1$ (lin)")
ax[1, 2].set_xlabel(r"$\mathcal{M}_c$")
ax[1, 2].set_ylabel(r"aligned $\eta$")
ax[1, 2].set_title(rf"$R^2(\eta_0,M_c)$={r2(y_te[:,0], mc_al):.3f}, "
                   rf"$R^2(\eta_1,M_c)$={r2(y_te[:,1], mc_al):.3f}")
ax[1, 2].legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUTDIR / "gw_oneshot_diag.png", dpi=160)
plt.show()
log(f"\nwrote {OUTDIR / 'gw_oneshot_diag.png'}")
log("done")
