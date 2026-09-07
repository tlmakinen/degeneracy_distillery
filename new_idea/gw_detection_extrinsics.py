# =============================================================================
# GW TaylorF2 rank-detection with weakly informed extrinsics
#
# Harder analogue of the hidden-Rosenbrock oneshot: true intrinsic rank is
# still dominated by (m1, m2) / chirp structure, but theta also carries
# physics parameters that *do* enter the waveform (so they are not pure
# prior-transport dummies):
#
#   theta = (m1, m2, phi_c, t_c, d_L)          d = 5
#   h(f)  = A(m1,m2,d_L; f) exp(i (psi(m1,m2;f) - phi_c + 2 pi f t_c))
#
# Extrinsics are typically closer to linear / gauge-like than Mc, and often
# sit nearer the information-spectrum null than the mass sector — that is
# the point of the experiment.  Labeled d_L is further softened by a small
# Gaussian jitter on the realized distance in h(f) (stand-in for a
# hierarchical distance model).
#
# Protocol (heater / Rosenbrock oneshot; no m-sweep):
#   1. ONE probe fit at m = M_PROBE  (require M_PROBE <= d)
#   2. spectrum -> r_hat; refit at r_hat
#   3. soft screen via Jacobian column norms
#   4. ONE bootstrap at m = M_BOOT (M_PROBE if BOOT_AT_PROBE else r_hat):
#        detection stats (lambda / r_hat votes / column-norm top-2)
#        AND the ensemble maps (eta, J) for align_coords
#      -> Procrustes + nonlinearity rotation on that same ensemble
#      -> keep the SR_ETA_AXES most nonlinear eta axes
#   5. Operon SR on those eta axes (X = all scaled theta)
#
# Contrast with gw_oneshot.py (fixed m=d=2, no detection) and with a pure
# dummy-nuisance variant (extras never enter h).
# =============================================================================
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathlib import Path
from typing import Sequence

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
NSIMS         = 2000
N_PCA         = 40
PCA_BASIS_MAX = 5000
D             = 5
M_PROBE       = 4              # overspec; keep <= D so JJ^T can be full rank
NATS_FLOOR    = 0.10
STEPS         = 8000
BATCH         = 128
BATCH_AUG     = 256
LR            = 5e-4
SCALE_BOOST   = 1.0
WEIGHT_DECAY  = 1e-3
PATIENCE      = 20
SEED          = 0
K_BOOT        = 8
BOOT_AT_PROBE = True
SR_ETA_AXES   = 2              # keep this many most-nonlinear eta axes for Operon
ALIGN_SUBSAMPLE = 1000
SR_GRID_SIZE  = 2000
SR_TIME_LIMIT = 120
SR_MAX_LENGTH = 30
SR_MAX_DEPTH  = 20
SR_LENGTH_PENALTY = 2.0
RUN_SR        = True
OUTDIR        = Path("new_idea/gw_detection_extrinsics_out")

PARAM_NAMES = ("m1", "m2", "phi_c", "t_c", "d_L")
# Soft truth labels for ranking diagnostics (not used by the fit).
# Masses dominate; extrinsics are expected weaker / closer to null.
ACTIVE_HINT = (0, 1)           # m1, m2 indices

M_SUN_SEC = 4.925491025543576e-6
MPC_SEC = 1.0292712503e14
M1_MIN, M1_MAX = 5.0, 50.0
M2_MIN, M2_MAX = 5.0, 50.0
PHI_MIN, PHI_MAX = 0.0, 2.0 * np.pi
TC_MIN, TC_MAX = -0.02, 0.02          # seconds
DL_MIN, DL_MAX = 100.0, 400.0         # Mpc (was fixed at 200 in gw_oneshot)
F_LOW, DF = 20.0, 0.5
# Stand-in for a hierarchical d_L: jitter the realized distance used in h(f)
# so the labeled d_L coordinate is only weakly informative vs (m1, m2).
DL_JITTER_MPC = 5.0

OUTDIR.mkdir(parents=True, exist_ok=True)
assert M_PROBE <= D, "probe m > d makes JJ^T singular; raise D or lower M_PROBE"


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------- physics
def chirp_mass(m1, m2):
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def symmetric_mass_ratio(m1, m2):
    return (m1 * m2) / (m1 + m2) ** 2


def a_ligo_psd(f):
    f = np.asarray(f, dtype=float)
    x = f / 215.0
    psd = 1e-49 * (x**-4.14 + 2.0 + 2.0 * x**2)
    return np.where(f >= 10.0, psd, np.inf)


def taylor_f2_waveform(m1, m2, phi_c, t_c, d_l_mpc, freqs):
    """TaylorF2 with coalescence phase/time and luminosity distance."""
    m_sec = (m1 + m2) * M_SUN_SEC
    eta = m1 * m2 / (m1 + m2) ** 2
    mc_sec = chirp_mass(m1, m2) * M_SUN_SEC
    d_l_sec = float(d_l_mpc) * MPC_SEC

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
    fa = freqs[active]
    psi = 3.0 / (128 * eta) * vm**-5 * (1 + c1 * vm**2 + c15 * vm**3 + c2 * vm**4)
    phase[active] = psi - phi_c + 2.0 * np.pi * fa * t_c
    return amp * np.exp(1j * phase)


def sample_theta(n: int, rng: np.random.Generator) -> np.ndarray:
    """Physical theta; columns match PARAM_NAMES."""
    return np.stack([
        rng.uniform(M1_MIN, M1_MAX, n),
        rng.uniform(M2_MIN, M2_MAX, n),
        rng.uniform(PHI_MIN, PHI_MAX, n),
        rng.uniform(TC_MIN, TC_MAX, n),
        rng.uniform(DL_MIN, DL_MAX, n),
    ], axis=1).astype(np.float64)


def simulator_data(nsims: int, seed: int) -> dict:
    import time as _time
    rng = np.random.default_rng(seed)

    f_isco_lightest = 1.0 / (6**1.5 * np.pi * (M1_MIN + M2_MIN) * M_SUN_SEC)
    freqs = np.arange(F_LOW, min(f_isco_lightest, 1024.0), DF)
    whiten = np.sqrt(4.0 * DF) / np.sqrt(a_ligo_psd(freqs))
    n_freq = len(freqs)
    n_tot = 2 * nsims
    log(f"frequency grid: {F_LOW}-{freqs[-1]:.0f} Hz, {n_freq} bins  ({n_tot} waveforms)")
    log(f"theta = {PARAM_NAMES}  (extrinsics enter the waveform; "
        f"d_L jitter N(0,{DL_JITTER_MPC}) Mpc)")

    theta_all = sample_theta(n_tot, rng).astype(np.float32)

    log("computing waveforms (d_L jittered; detector noise added later)")
    t0 = _time.time()
    hvecs_all = np.empty((n_tot, 2 * n_freq), dtype=np.float64)
    # Realized distance = labeled d_L + seedable Gaussian jitter (clipped > 0).
    # theta still stores the unlabeled coordinate; only h sees the jitter.
    d_l_realized = np.maximum(
        theta_all[:, 4] + rng.normal(0.0, DL_JITTER_MPC, n_tot),
        1.0,
    )
    for i in tqdm(range(n_tot), desc="waveforms"):
        th = theta_all[i]
        h = taylor_f2_waveform(
            th[0], th[1], th[2], th[3], d_l_realized[i], freqs,
        ) * whiten
        hvecs_all[i] = np.concatenate([h.real, h.imag])
    log(f"  waveforms done in {_time.time() - t0:.1f}s")

    n_basis = min(n_tot, PCA_BASIS_MAX)
    log(f"building PCA basis (n_components={N_PCA}, n_basis={n_basis})")
    t0 = _time.time()
    basis_idx = rng.choice(n_tot, n_basis, replace=False)
    pca = PCA(
        n_components=N_PCA,
        svd_solver="randomized",
        random_state=seed,
        iterated_power=2,
    ).fit(hvecs_all[basis_idx])
    cumvar = pca.explained_variance_ratio_.cumsum()
    log(f"  PCA done in {_time.time() - t0:.1f}s; "
        f"{N_PCA} components capture {cumvar[-1] * 100:.1f}% variance")

    log("generating noisy PCA coefficients")
    t0 = _time.time()
    snr_all = np.linalg.norm(hvecs_all, axis=1)
    noise = rng.normal(size=hvecs_all.shape)
    data_all = pca.transform(hvecs_all + noise).astype(np.float32)
    log(f"  noisy data done in {_time.time() - t0:.1f}s")
    log(f"  SNR (whitened norm): median {np.median(snr_all):.1f}, "
        f"IQR [{np.percentile(snr_all, 25):.1f}, {np.percentile(snr_all, 75):.1f}]")

    return {
        "theta_train": theta_all[:nsims],
        "data_train": data_all[:nsims],
        "theta_test": theta_all[nsims:],
        "data_test": data_all[nsims:],
        "snr_train": snr_all[:nsims],
        "snr_test": snr_all[nsims:],
        "cumvar": cumvar,
    }


# ------------------------------------------------------------------ networks / fit
class Flattener(nn.Module):
    m: int
    lo: float = 1.0
    hi: float = 2.0
    features: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, theta):
        z = 2.0 * (theta - self.lo) / (self.hi - self.lo) - 1.0
        # theta is MinMax-scaled to [lo,hi], so log is safe for all coordinates
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


def fit_joint(M, th_fit, x_fit, th_val, x_val, steps=STEPS, seed=SEED):
    n_fit = th_fit.shape[0]
    bp = min(BATCH, n_fit)
    flat_mod = Flattener(m=M, lo=1.0, hi=2.0)
    est_mod = Estimator(m=M)
    k0, k1 = jr.split(jr.PRNGKey(seed), 2)
    p = {
        "flat": flat_mod.init(k0, th_fit[0]),
        "est": est_mod.init(k1, x_fit[0]),
        "raw_log_scale": jnp.zeros(M),
    }
    const = 0.5 * M * np.log(2 * np.pi)
    eyeM = jnp.eye(M)
    th_lo = jnp.asarray(th_fit.min(0))
    th_hi = jnp.asarray(th_fit.max(0))

    def core(p, theta):
        g = flat_mod.apply(p["flat"], theta)
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
    if a.std() < 1e-15 or b.std() < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def column_norms(jac):
    """RMS |d eta / d theta_j| averaged over samples and eta axes."""
    return np.sqrt((np.asarray(jac) ** 2).mean(axis=(0, 1)))


def rank_report(col, title=""):
    order = np.argsort(-col)
    log(f"  {title}Jacobian column norms:")
    for rank, j in enumerate(order):
        mark = "  <-- mass" if j in ACTIVE_HINT else "  <-- extrinsic"
        log(f"    rank {rank + 1}: {PARAM_NAMES[j]:6s}  ||J||={col[j]:.4f}{mark}")
    top2 = set(int(v) for v in order[:2])
    log(f"  top-2 = {[PARAM_NAMES[j] for j in order[:2]]}"
        f"  -> {'MASS PAIR' if top2 == set(ACTIVE_HINT) else 'NOT pure mass pair'}")
    return order


# ================================================================ main
log("######## 0. GW TaylorF2 + extrinsics ########")
log(f"d={D}, M_PROBE={M_PROBE}, NATS_FLOOR={NATS_FLOOR}, nsims={NSIMS}")
data = simulator_data(NSIMS, seed=SEED)
th_phys_te = data["theta_test"]

scaler = fit_theta_scaler(data["theta_train"], feature_range=(1.0, 2.0))
th_tr = scaler.transform(data["theta_train"]).astype(np.float64)
th_te = scaler.transform(data["theta_test"]).astype(np.float64)
log(f"scaled theta range: {th_tr.min(0)} to {th_tr.max(0)}")

x_mu = data["data_train"].mean(0)
x_sd = data["data_train"].std(0) + 1e-12
x_tr = ((data["data_train"] - x_mu) / x_sd).astype(np.float64)
x_te = ((data["data_test"] - x_mu) / x_sd).astype(np.float64)

NVAL = max(50, int(0.15 * NSIMS))
NTEST = th_te.shape[0]
TH = jnp.asarray(th_tr); X = jnp.asarray(x_tr)
TH_TE = jnp.asarray(th_te); X_TE = jnp.asarray(x_te)
th_fit, x_fit = TH[:-NVAL], X[:-NVAL]
th_val, x_val = TH[-NVAL:], X[-NVAL:]

mc_te = chirp_mass(th_phys_te[:, 0], th_phys_te[:, 1])
q_te = th_phys_te[:, 0] / th_phys_te[:, 1]
eta_sym_te = symmetric_mass_ratio(th_phys_te[:, 0], th_phys_te[:, 1])
targets = {
    "Mc": mc_te,
    "m1/m2": q_te,
    "eta_sym": eta_sym_te,
    "phi_c": th_phys_te[:, 2],
    "t_c": th_phys_te[:, 3],
    "d_L": th_phys_te[:, 4],
    "log_d_L": np.log(th_phys_te[:, 4]),
}


# ------------------------------------------------------- 1. probe
log(f"\n######## 1. probe fit at m={M_PROBE} (all {D} inputs) ########")
probe = fit_joint(M_PROBE, th_fit, x_fit, th_val, x_val, seed=SEED)
nll_probe = float(np.mean(np.asarray(probe["nll_vec"](TH_TE, X_TE))))
eta_p = np.asarray(probe["eta"](TH_TE))
E_p, lam, V_p = spectrum(eta_p)
jac_p = probe["jac"](TH_TE)
col_p = column_norms(jac_p)
z_p = np.asarray(probe["z_vec"](TH_TE, X_TE))
z_p_c = (z_p - z_p.mean(0)) @ V_p

lam_null = float(np.median(lam[M_PROBE // 2:])) if M_PROBE >= 2 else 1.0
nats_n = 0.5 * np.log(lam / lam_null)
band_samp = (1.0 + np.sqrt(M_PROBE / max(NTEST, 1))) ** 2 - 1.0

log(f"  test NLL {nll_probe:.4f}")
log(f"  information spectrum lambda_i : {np.round(lam, 3)}")
log(f"  measured null level           : {lam_null:.4f}  (theory: 1 at optimum)")
log(f"  profile 0.5 log(lambda/null)  : {np.round(nats_n, 4)} nats")
log(f"  sampling-only band on lambda  : 1 + {band_samp:.4f}"
    f"  ({0.5 * np.log1p(band_samp):.4f} nats)")
log(f"  residual std (canonical): {np.round(z_p_c.std(0), 4)}  (want ~1)")

log("  sensitivity of r_hat to the floor:")
for f in (0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00):
    log(f"    floor {f:4.2f} nats -> r_hat = {int(np.sum(nats_n > f))}")
r_hat = int(np.sum(nats_n > NATS_FLOOR))
log(f"  adopt floor {NATS_FLOOR} nats  ->  r_hat = {r_hat}")
if r_hat >= M_PROBE:
    log("  WARNING: r_hat reached the probe dimension. Increase M_PROBE (and D if needed).")
if r_hat < M_PROBE:
    log(f"  discarded block: lambda in [{lam[r_hat:].min():.3f}, {lam[r_hat:].max():.3f}]")
if r_hat < 1:
    log("  WARNING: r_hat < 1; forcing r_hat = 1 for the refit.")
    r_hat = 1

log("  [truth] R^2(probe axis vs physics):")
for j in range(M_PROBE):
    bits = ", ".join(f"{name}={r2(E_p[:, j], t):.3f}" for name, t in targets.items())
    log(f"    axis {j}: {bits}")

rank_report(col_p, title="probe ")


# ------------------------------------------------------- 2. refit
log(f"\n######## 2. refit at r_hat={r_hat} ########")
ref = fit_joint(r_hat, th_fit, x_fit, th_val, x_val, seed=SEED + 11)
nll_hat = float(np.mean(np.asarray(ref["nll_vec"](TH_TE, X_TE))))
eta = np.asarray(ref["eta"](TH_TE))
E, lam_hat, V = spectrum(eta)
jac = ref["jac"](TH_TE)
col = column_norms(jac)
z = np.asarray(ref["z_vec"](TH_TE, X_TE))
z_c = (z - z.mean(0)) @ V

spare = max(M_PROBE - r_hat, 1)
log(f"  test NLL {nll_hat:.4f}   (probe was {nll_probe:.4f},"
    f" spare cost {(nll_probe - nll_hat) / spare:.4f} nats/axis)")
log(f"  spectrum at r_hat: {np.round(lam_hat, 3)}"
    f"   profile {np.round(0.5 * np.log(lam_hat), 4)} nats")
log(f"  residual std (canonical): {np.round(z_c.std(0), 4)}  (want ~1)")

log("  [truth] R^2(refit axis vs physics):")
for j in range(r_hat):
    bits = ", ".join(f"{name}={r2(E[:, j], t):.3f}" for name, t in targets.items())
    log(f"    axis {j}: {bits}")

order_col = rank_report(col, title="refit ")
mass_n = np.array([col[i] for i in ACTIVE_HINT])
ext_n = np.delete(col, list(ACTIVE_HINT))
log(f"  extrinsic max ||J|| / min(mass ||J||): "
    f"{ext_n.max() / max(mass_n.min(), 1e-12):.4f}"
    f"  (>>1 means an extrinsic rivals the masses)")


# ------------------------------------------------------- 3. bootstrap (detection + align ensemble)
# One ensemble at M_BOOT serves both r_hat UQ and align_coords / Operon.
# Align at the same m; nonlinearity rotation + SR_ETA_AXES slice picks the
# axes for SR (no second refit at max(r_hat, 2)).
M_BOOT = M_PROBE if BOOT_AT_PROBE else max(int(r_hat), SR_ETA_AXES)
assert M_BOOT <= D
assert M_BOOT >= SR_ETA_AXES
log(f"\n######## 3. bootstrap at m={M_BOOT} (K={K_BOOT}): "
    f"detection stats + align ensemble ########")

boot_rng = np.random.default_rng(SEED + 31)
n = TH.shape[0]
n_te = TH_TE.shape[0]
key_sr = jr.PRNGKey(SEED + 5 * 10_000)
X_sr = np.asarray(jr.uniform(
    key_sr, minval=TH.min(0), maxval=TH.max(0), shape=(SR_GRID_SIZE, D),
))

etas_te, jacs_te, etas_grid_members, oob_nlls = [], [], [], []
r_members, lam_members, top2_members = [], [], []
for k in range(K_BOOT):
    idx = boot_rng.integers(0, n, n)
    oob = np.setdiff1d(np.arange(n), np.unique(idx))
    if len(oob) < 20:
        oob = np.arange(min(50, n))
    mo = fit_joint(M_BOOT, TH[idx], X[idx], TH[oob], X[oob], seed=SEED + 500 + k)
    eta_k = np.asarray(mo["eta"](TH_TE))
    jac_k = np.asarray(mo["jac"](TH_TE))
    etas_te.append(eta_k)
    jacs_te.append(jac_k)
    etas_grid_members.append(np.asarray(mo["eta"](jnp.asarray(X_sr))))
    oob_nlls.append(float(mo["best_val"]))

    _, lam_k, _ = spectrum(eta_k)
    null_k = float(np.median(lam_k[M_BOOT // 2:])) if M_BOOT >= 2 else 1.0
    nats_k = 0.5 * np.log(lam_k / null_k)
    rk = int(np.sum(nats_k > NATS_FLOOR))
    col_k = column_norms(jac_k)
    top2 = tuple(int(v) for v in np.argsort(-col_k)[:2])
    r_members.append(rk)
    lam_members.append(lam_k)
    top2_members.append(top2)
    log(f"  member {k}: oob NLL {mo['best_val']:.4f}  r_hat={rk}  "
        f"lambda={np.round(lam_k, 2)}  top2={[PARAM_NAMES[j] for j in top2]}")

lam_stack = np.stack(lam_members)
log(f"  r_hat votes: {dict(zip(*np.unique(r_members, return_counts=True)))}")
log(f"  lambda mean ± std: {np.round(lam_stack.mean(0), 3)} ± {np.round(lam_stack.std(0), 3)}")
mass_top2_frac = np.mean([set(t) == set(ACTIVE_HINT) for t in top2_members])
log(f"  fraction with top-2 = {{m1,m2}}: {mass_top2_frac:.2f}")


# ------------------------------------------------------- 4. align_coords on that ensemble
log(f"\n######## 4. align_coords Procrustes on bootstrap (m={M_BOOT}, "
    f"keep top {SR_ETA_AXES} nonlinear eta) ########")
n_align = min(ALIGN_SUBSAMPLE, n_te)
randidx = boot_rng.choice(n_te, n_align, replace=False)
randidx.sort()
# align_coords expects θ-space Fishers (n, d, d).  No Fishnets here, so use
# the pullback F ≈ J^T J from each member's Jacobian (n, m, d).
jacs_stack = np.stack(jacs_te)  # (K, n_te, m, d)
F_ensemble = np.einsum("knmi,knmj->knij", jacs_stack, jacs_stack)
datafile = {
    "theta": np.asarray(th_te),
    "eta_ensemble": np.stack(etas_te),
    "Jbar_ensemble": jacs_stack,
    "F_ensemble": F_ensemble,
    "ensemble_weights": np.ones(K_BOOT) / K_BOOT,
    "norm_factor": 1.0,
}
best_model_idx = int(np.argmin(oob_nlls))
Favg = F_ensemble[best_model_idx][randidx]  # (n_align, d, d)
log(f"  align_coords: reference=member {best_model_idx} (best oob), "
    f"n_align={n_align}, F shape={Favg.shape} (θ-space JᵀJ), "
    f"separate_nonlinearity=True")
aligned = process_ensemble_rotation_v2(
    datafile=datafile,
    randidx=randidx,
    Favg=Favg,
    best_model_idx=best_model_idx,
    n_d=1.0,
    align_mode="procrustes",
    separate_nonlinearity=True,
    # none: keep nonlinearity-first order for the SR_ETA_AXES slice, and avoid
    # fisher_order_canonicalize (old package builds assume m==d and crash when
    # m_align < d).  With a patched align_coords, "sign_only" is also fine.
    canonicalize="none",
    use_prior_normalization=True,
    restore_reference_mean=True,
    Fisher_to_flatten="best",
    verbose=True,
    offset_delta=0.1,
)

y_full = np.asarray(aligned["y"])
y_std_full = np.asarray(aligned["y_std"])
X_aligned = np.asarray(aligned["X"])
dy_sr_full = np.asarray(aligned["dy_sr"])
Ens_full = np.asarray(aligned["ys"])
rotmats = np.asarray(aligned["rotmats"])
w_ens = np.asarray(aligned["ensemble_weights"])
nl_sigma = nonlinearity_spectrum(np.stack(jacs_te)[best_model_idx][randidx])
log(f"  nonlinearity spectrum sigma (ref member): {np.round(nl_sigma, 4)}")
log(f"  -> keeping axes 0..{SR_ETA_AXES - 1} (most nonlinear)")

# Restrict to the most nonlinear SR_ETA_AXES coordinates
m_sr = SR_ETA_AXES
y_te = y_full[:, :m_sr]
y_std_te_raw = y_std_full[:, :m_sr]
dy_sr_te = dy_sr_full[:, :m_sr, :]
Ens_te = Ens_full[:, :, :m_sr]

th_al_phys = scaler.inverse_transform(X_aligned)
mc_al = chirp_mass(th_al_phys[:, 0], th_al_phys[:, 1])
q_al = th_al_phys[:, 0] / th_al_phys[:, 1]
for j in range(m_sr):
    tag = "most nonlinear" if j == 0 else f"nl-rank {j}"
    log(f"  aligned eta_{j} ({tag}): R^2 vs Mc={r2(y_te[:, j], mc_al):.4f}  "
        f"vs m1/m2={r2(y_te[:, j], q_al):.4f}  "
        f"vs phi_c={r2(y_te[:, j], th_al_phys[:, 2]):.4f}  "
        f"vs t_c={r2(y_te[:, j], th_al_phys[:, 3]):.4f}  "
        f"vs log d_L={r2(y_te[:, j], np.log(th_al_phys[:, 4])):.4f}")

# SR grid: rotate full m_align, then keep nonlinear-first axes
ys_sr = np.stack(etas_grid_members)  # (K, n_sr, M_BOOT)
ys_sr_rot = np.array([
    np.einsum("ij,bj->bi", rotmats[i], ys_sr[i] - ys_sr[i].mean(0))
    for i in range(K_BOOT)
])
ys_sr_rot = ys_sr_rot[:, :, :m_sr]
y_std_sr_raw = np.asarray(weighted_std(jnp.asarray(ys_sr_rot), weights=jnp.asarray(w_ens)))
y_sr = np.average(ys_sr_rot, 0, w_ens)
ys_sr_rot -= y_sr.min(0)
y_sr -= y_sr.min(0)

# LOO calibration on the kept nonlinear axes
K = K_BOOT
eta_mu = Ens_te.mean(0)
mu_loo = (K * eta_mu[None] - Ens_te) / (K - 1)
sd_loo = np.sqrt(np.clip(
    (K * (Ens_te.var(0, ddof=0)[None] + eta_mu[None] ** 2) - Ens_te ** 2) / (K - 1)
    - mu_loo ** 2, 1e-24, None))
sd_loo *= np.sqrt((K - 1) / (K - 2))
z_loo = (Ens_te - mu_loo) / (sd_loo * np.sqrt(K / (K - 1)))
zstd = z_loo.reshape(-1, m_sr).std(0)
tref = np.sqrt((K - 2) / (K - 4)) if K > 4 else np.nan
log(f"  map spread (aligned test, median raw): {np.round(np.median(y_std_te_raw, 0), 4)}")
if K > 4:
    log(f"  LOO ensemble z: std {np.round(zstd, 3)}  "
        f"reference std(t_{K-2})={tref:.4f}  "
        f"ratio {np.round(zstd / tref, 3)}")
else:
    log(f"  LOO ensemble z: std {np.round(zstd, 3)}  (need K>4 for t reference)")

Y_STD_FLOOR = 1e-3
y_std_te = np.maximum(y_std_te_raw, Y_STD_FLOOR)
y_std_sr = np.maximum(y_std_sr_raw, Y_STD_FLOOR)
frac_floored = (y_std_te_raw < Y_STD_FLOOR).mean(0)
log(f"  fraction of align points hitting y_std floor {Y_STD_FLOOR}: "
    f"{np.round(frac_floored, 3)}")
log(f"  y_sr shape {y_sr.shape} (eta axes={m_sr}), "
    f"X_sr shape {X_sr.shape} (theta d={D})")

# Operon pack: X = all scaled theta; y = top nonlinear eta axes
X_sr_pack = X_aligned
y_pack = y_te
y_std_pack = y_std_te
dy_sr_pack = dy_sr_te

np.savez(
    OUTDIR / "gw_detection_sr_inputs.npz",
    X=X_sr_pack, y=y_pack, y_std=y_std_pack, dy_sr=dy_sr_pack,
    X_sr=X_sr, y_sr=y_sr, y_std_sr=y_std_sr,
    theta_test_physical=data["theta_test"],
    scaler_scale=scaler.scale_,
    scaler_min=scaler.min_,
    rotmats=rotmats,
    nonlinearity_sigma=nl_sigma,
    best_model_idx=best_model_idx,
    m_boot=M_BOOT,
    sr_eta_axes=m_sr,
    param_names=np.asarray(PARAM_NAMES),
    r_hat=r_hat,
)
log(f"  wrote {OUTDIR / 'gw_detection_sr_inputs.npz'}")
log("  Operon mapping: X <-> theta_scaled (all d), "
    f"y <-> top-{m_sr} nonlinear eta, y_std <-> ensemble deta")


# ------------------------------------------------------- 5. Operon SR
if RUN_SR:
    sr_dir = OUTDIR / "sr_results"
    sr_dir.mkdir(exist_ok=True)
    log(f"\n######## 5. Operon SR into {sr_dir} "
        f"(y has {m_sr} nonlinear eta axes, X has {D} theta) ########")
    fit_symbolic_regression(
        X_sr,
        y_sr,
        y_std_sr,
        parent_dir=str(sr_dir) + os.sep,
        random_state=SEED + 6 * 10_000,
        time_limit=SR_TIME_LIMIT,
        max_length=SR_MAX_LENGTH,
        max_depth=SR_MAX_DEPTH,
        allowed_symbols="add,mul,div,pow,constant,variable,exp",
        objectives=["r2", "length"],
    )

    n_params = X_sr_pack.shape[1]       # theta dim (X columns)
    n_components = y_pack.shape[1]      # eta axes fitted by Operon (= SR_ETA_AXES)
    equation_predicate = sr_structure_predicate(
        n_params=n_params,
        forbid_self_transcendental=True,
    )
    # filter_pareto_fronts second arg is n_components, not n_params
    # (equal only when m == d; here m_sr=2, d=5).
    filter_summaries = filter_pareto_fronts(
        str(sr_dir) + os.sep, n_components, equation_predicate,
    )
    removed = sum(int(s["removed"]) for s in filter_summaries)
    log(f"  removed {removed} self-transcendental/invalid equations from Pareto fronts")

    Fs_id = np.repeat(np.eye(n_params)[None, ...], len(X_sr_pack), axis=0)
    mdl_coords, frob_coords, analysis = analyze_equations(
        X_sr_pack, y_pack, y_std_pack, dy_sr_pack, Fs_id,
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
    log("\n######## 5. RUN_SR=False; skipping Operon ########")
    log(f"  load {OUTDIR / 'gw_detection_sr_inputs.npz'} for external SR")


# ------------------------------------------------------- persist + plots
np.savez(
    OUTDIR / "gw_detection_extrinsics.npz",
    lam_probe=lam,
    nats_probe=nats_n,
    r_hat=r_hat,
    lam_hat=lam_hat,
    col_probe=col_p,
    col_refit=col,
    r_members=np.asarray(r_members),
    lam_members=lam_stack,
    param_names=np.asarray(PARAM_NAMES),
    snr_test=data["snr_test"],
    nonlinearity_sigma=nl_sigma,
    m_boot=M_BOOT,
)
log(f"\nwrote {OUTDIR / 'gw_detection_extrinsics.npz'}")

gx = np.linspace(-4, 4, 200)
gauss = np.exp(-gx ** 2 / 2) / np.sqrt(2 * np.pi)
fig, ax = plt.subplots(2, 3, figsize=(14, 7.2))

ax[0, 0].semilogy(np.arange(M_PROBE), lam, "o-", label="probe")
if r_hat <= len(lam_hat):
    ax[0, 0].semilogy(np.arange(r_hat), lam_hat, "s--", label=rf"refit $\hat r$={r_hat}")
ax[0, 0].axhline(1.0, color="k", ls="--", lw=1, label=r"null $\lambda=1$")
ax[0, 0].axhline(lam_null, color="0.5", ls=":", lw=1, label=rf"meas. null={lam_null:.2f}")
ax[0, 0].set_xlabel("canonical axis")
ax[0, 0].set_ylabel(r"$\lambda_i$")
ax[0, 0].set_title(f"information spectrum (probe m={M_PROBE})")
ax[0, 0].legend(fontsize=8)

ax[0, 1].bar(np.arange(D), col[np.argsort(-col)], color="C0")
ax[0, 1].set_xticks(np.arange(D))
ax[0, 1].set_xticklabels([PARAM_NAMES[j] for j in np.argsort(-col)], rotation=30, ha="right")
ax[0, 1].set_ylabel(r"RMS $\|\partial\eta/\partial\theta_j\|$")
ax[0, 1].set_title("soft screen at " + rf"$\hat r$={r_hat}")

for j in range(r_hat):
    ax[0, 2].hist(z_c[:, j], bins=60, density=True, histtype="step", label=fr"$z_{j}$")
ax[0, 2].plot(gx, gauss, "k--", lw=1)
ax[0, 2].set_title(r"refit residuals $z$ (canonical)")
ax[0, 2].legend(fontsize=8)

sc = ax[1, 0].scatter(th_al_phys[:, 0], th_al_phys[:, 1], c=y_te[:, 0], s=6,
                      cmap="viridis", alpha=0.7)
plt.colorbar(sc, ax=ax[1, 0], label=r"$\eta_0$ (most nl)")
ax[1, 0].set_xlabel(r"$m_1$"); ax[1, 0].set_ylabel(r"$m_2$")
ax[1, 0].set_title(rf"aligned $\eta_0$ ($R^2$ Mc={r2(y_te[:,0], mc_al):.2f})")

ax[1, 1].scatter(mc_al, y_te[:, 0], s=6, alpha=0.45, edgecolors="none", label=r"$\eta_0$ (nl)")
if m_sr > 1:
    ax[1, 1].scatter(mc_al, y_te[:, 1], s=6, alpha=0.35, edgecolors="none",
                     label=r"$\eta_1$ (nl-rank1)")
ax[1, 1].set_xlabel(r"$\mathcal{M}_c$")
ax[1, 1].set_ylabel(r"aligned nonlinear $\eta$")
ttl = rf"$R^2(\eta_0,M_c)$={r2(y_te[:,0], mc_al):.3f}"
if m_sr > 1:
    ttl += rf", $R^2(\eta_1,M_c)$={r2(y_te[:,1], mc_al):.3f}"
ax[1, 1].set_title(ttl)
ax[1, 1].legend(fontsize=8)

scale = tref if (K > 4 and np.isfinite(tref)) else 1.0
for j in range(m_sr):
    ax[1, 2].hist((z_loo[..., j] / scale).ravel(), bins=80, density=True,
                  histtype="step", label=fr"axis {j}")
ax[1, 2].plot(gx, gauss, "k--", lw=1, label="N(0,1)")
ax[1, 2].set_title(r"bootstrap LOO $z$ (aligned nl axes)")
ax[1, 2].legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUTDIR / "gw_detection_extrinsics_diag.png", dpi=160)
plt.show()
log(f"wrote {OUTDIR / 'gw_detection_extrinsics_diag.png'}")
log("done")
