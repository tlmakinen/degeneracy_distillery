#!/usr/bin/env python
"""One-step map + SR on CAMELS-SB35, all 35 parameters, no pre-selection.

The three-step run (scripts/camels_notebook_run.py, job 3689657) could not use
the 35D space directly.  Its mean Fisher is

    141.6  95.7  52.4  34.5  26.4  17.7 | 7.1  5.7  4.9 | 4.5 ... 2.3

-- six large eigenvalues, a gap, then a flat band of ~26 between 2.3 and 4.5
with condition number 61.  A relative floor of 1e-2 reads rank 35, so the
pipeline needed a hand-built Fisher heuristic (diag informativeness +
off-diagonal variance) to cut to 6 parameters before flattening.  That flat
band is the tightened-prior signature described in
new_idea/oneshot_vs_threestep.md: a Cholesky floor keeping every eigenvalue
away from zero when the real rank is much lower.

This is the same trigger as scalar{3,4} -- every theta enters, rank r << d --
at d = 35, so the one-step map should read r_hat off the eta spectrum without
any pre-selection.  There is no ground truth here, so the checks are:
r_hat vs the Fisher spectrum, held-out R^2 of eta on the SR expressions, and
which parameters actually carry the discovered axes.

Rectangular m < d with no fibre measure is NOT an ELBO (oneshot_vs_threestep.md):
the coarea Hausdorff factor is dropped, so the score is a bound on the
pushforward u = eta(theta), not on theta.  That is the intended use -- read
r_hat and the active set -- not a posterior NLL.  Do not quote it as one.

    python new_idea/camels_oneshot.py --smoke
    python new_idea/camels_oneshot.py --m-probe 12
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from scalar_oneshot_sr import (  # noqa: E402
    align_members,
    apply_rotations,
    log,
    r2,
    timed_predicate,
)

OBSERVABLES = [
    "MBH_Mh_s61", "MBH_Mh_s90", "Mg_Mh_s61", "Mg_Mh_s90",
    "Ms_Mh_s61", "Ms_Mh_s90", "Rs_Ms_s61", "Rs_Ms_s90",
    "SFRH", "SFRH_100Myr", "SFR_Ms_s61", "SFR_Ms_s90",
    "Zs_Ms_s61", "Zs_Ms_s90",
]
PARAM_NAMES = [
    "Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2", "Omega_b",
    "H_0", "n_s", "MaxSfrTimescale", "FactorForSofterEQS", "IMFslope",
    "SNII_MinMass", "ThermalWindFraction", "VariableWindSpecMomentum",
    "WindFreeTravelDensFac", "MinWindVel", "WindEnergyReductionFactor",
    "WindEnergyReductionMetallicity", "WindEnergyReductionExponent",
    "WindDumpFactor", "SeedBlackHoleMass", "BH_AccretionFactor",
    "BH_EddingtonFactor", "BH_FeedbackFactor", "BH_RadiativeEfficiency",
    "QuasarThreshold", "QuasarThresholdPower", "UVB_H0_beta", "UVB_H0_Deltaz",
    "UVB_Hep_beta", "UVB_Hep_Deltaz", "SNIa_Rate_Norm", "SNIa_Rate_DTD_power",
    "SofteningLength",
]
N_SIMS = 1024
N_PARAMS = 35


def load_camels(data_path: Path, n_pca: int, seed: int):
    """theta (n,35) scaled to [1,2]; x = PCA coefficients of the 344 features."""
    import h5py
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from degeneracy_distillery.sr_utils import fit_theta_scaler

    with h5py.File(data_path, "r") as f:
        P = np.asarray(f["Parameters"][...], dtype=np.float32)
        blocks = []
        for name in OBSERVABLES:
            a = np.asarray(f[name][...], dtype=np.float32)
            blocks.append(a[:, :N_SIMS].T if a.shape[0] != N_SIMS else a[:N_SIMS])
    theta_raw = P[:, :N_SIMS].T if P.shape[0] == N_PARAMS else P[:N_SIMS]
    data_raw = np.nan_to_num(np.concatenate(blocks, axis=1), nan=0.0,
                             posinf=0.0, neginf=0.0)
    log(f"  theta {theta_raw.shape}  data {data_raw.shape}")

    th_tr, th_te, x_tr, x_te = train_test_split(
        theta_raw, data_raw, test_size=0.2, random_state=seed
    )
    # Features standardised before PCA: the 344 columns span ~2 orders of
    # magnitude in scale (0.007 to 1.1), so raw PCA would track the loudest
    # scaling relation rather than the informative directions.
    fs = StandardScaler().fit(x_tr)
    n_pca = int(min(n_pca, x_tr.shape[0] - 1, x_tr.shape[1]))
    pca = PCA(n_components=n_pca, svd_solver="randomized",
              random_state=seed, iterated_power=4).fit(fs.transform(x_tr))
    cum = pca.explained_variance_ratio_.cumsum()[-1]
    log(f"  PCA {n_pca} components capture {cum * 100:.1f}% of feature variance")

    scaler = fit_theta_scaler(th_tr, feature_range=(1.0, 2.0))
    return {
        "th_tr": scaler.transform(th_tr).astype(np.float64),
        "th_te": scaler.transform(th_te).astype(np.float64),
        "x_tr": pca.transform(fs.transform(x_tr)).astype(np.float64),
        "x_te": pca.transform(fs.transform(x_te)).astype(np.float64),
        "scaler": scaler,
        "cumvar": float(cum),
    }


def build_oneshot(th, x, cfg):
    """Joint (eta, etahat) fit.  Rectangular m < d: logdet uses J J^T."""
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import optax

    jax.config.update("jax_enable_x64", True)

    d = th.shape[1]
    th_j, x_j = jnp.asarray(th), jnp.asarray(x)
    nval = max(40, int(0.15 * th.shape[0]))
    th_hold, x_hold = th_j[-nval:], x_j[-nval:]
    th_pool, x_pool = th_j[:-nval], x_j[:-nval]
    th_lo = jnp.asarray(th.min(0))
    th_hi = jnp.asarray(th.max(0))

    class Flattener(nn.Module):
        m: int
        skip_init: np.ndarray
        hidden: Sequence[int] = (128, 128)

        @nn.compact
        def __call__(self, theta):
            h = nn.gelu(nn.Dense(self.hidden[0], name="in_proj")(theta))
            for k, w in enumerate(self.hidden[1:]):
                h = nn.gelu(nn.Dense(w, name=f"h{k}")(h))
            delta = nn.Dense(self.m, name="out",
                             kernel_init=nn.initializers.zeros,
                             bias_init=nn.initializers.zeros)(h)
            skip = nn.Dense(self.m, use_bias=False, name="skip",
                            kernel_init=lambda k, s, dt: jnp.asarray(
                                self.skip_init, dt))(theta)
            return skip + delta

    class Estimator(nn.Module):
        """x is one PCA coefficient vector per simulation -- no repeat axis."""
        m: int
        hidden: Sequence[int] = (128, 128)

        @nn.compact
        def __call__(self, xx):
            h = xx
            for w in self.hidden:
                h = nn.gelu(nn.Dense(w)(h))
            return (nn.Dense(self.m, kernel_init=nn.initializers.zeros,
                             bias_init=nn.initializers.zeros)(h)
                    + nn.Dense(self.m,
                               kernel_init=nn.initializers.normal(1e-2))(xx))

    def fit_joint(m, steps, seed, boot=False):
        if boot:
            idx_b = jr.randint(jr.PRNGKey(seed + 31337),
                               (th_pool.shape[0],), 0, th_pool.shape[0])
            th_fit, x_fit = th_pool[idx_b], x_pool[idx_b]
        else:
            th_fit, x_fit = th_pool, x_pool
        n_fit = th_fit.shape[0]
        bp = min(cfg.batch, n_fit)

        q0 = np.zeros((d, m))
        q0[np.arange(min(m, d)), np.arange(min(m, d))] = 1.0
        flat_mod = Flattener(m=m, skip_init=q0)
        est_mod = Estimator(m=m)
        k0, k1 = jr.split(jr.PRNGKey(seed), 2)
        p = {"est": est_mod.init(k1, x_fit[0]),
             "raw_log_scale": jnp.zeros(m),
             "flat": flat_mod.init(k0, th_fit[0])}
        const = 0.5 * m * np.log(2 * np.pi)   # no complement: rectangular, no hard cut
        eye_m = jnp.eye(m)
        l1_w = cfg.l1 / n_fit

        def core(p, theta):
            return jnp.exp(2.0 * p["raw_log_scale"]) * flat_mod.apply(p["flat"], theta)

        def logdet(p, theta):
            j = jax.jacfwd(lambda t: core(p, t))(theta)      # (m, d), m < d
            return 0.5 * jnp.linalg.slogdet(j @ j.T + 1e-10 * eye_m)[1]

        def nll_vec(p, th_b, x_b):
            s = jnp.exp(2.0 * p["raw_log_scale"])

            def one(theta, xx):
                r = core(p, theta) - s * est_mod.apply(p["est"], xx)
                return 0.5 * jnp.sum(r ** 2) - logdet(p, theta) + const

            return jax.vmap(one)(th_b, x_b)

        nll_mean = jax.jit(lambda p, t, y: jnp.mean(nll_vec(p, t, y)))
        warm = max(1, min(300, steps // 10))
        sched = optax.warmup_cosine_decay_schedule(
            1e-5, cfg.lr, warm, max(steps, warm + 1), cfg.lr * 1e-2)
        tx = optax.chain(optax.clip_by_global_norm(10.0),
                         optax.adamw(sched, weight_decay=cfg.weight_decay))
        st = tx.init(p)

        @jax.jit
        def step(p, st, key):
            kp, ka = jr.split(key)
            idx = jr.randint(kp, (bp,), 0, n_fit)
            u = jr.uniform(ka, (cfg.batch_aug, d))
            th_aug = th_lo + u * (th_hi - th_lo)

            def obj(p):
                s = jnp.exp(2.0 * p["raw_log_scale"])
                z = jax.vmap(lambda t, y: core(p, t) - s * est_mod.apply(p["est"], y))(
                    th_fit[idx], x_fit[idx])
                geom = -jnp.mean(jax.vmap(lambda t: logdet(p, t))(th_aug))
                k1 = p["flat"]["params"]["in_proj"]["kernel"]
                ks = p["flat"]["params"]["skip"]["kernel"]
                pen = jnp.sum(jnp.sqrt(jnp.sum(k1 ** 2, 1) + jnp.sum(ks ** 2, 1) + 1e-12))
                return 0.5 * jnp.mean(jnp.sum(z ** 2, 1)) + geom + const + l1_w * pen

            l, g = jax.value_and_grad(obj)(p)
            upd, st = tx.update(g, st, p)
            new = optax.apply_updates(p, upd)
            ok = jnp.all(jnp.stack([jnp.all(jnp.isfinite(z))
                                    for z in jax.tree_util.tree_leaves(g)]))
            return jax.tree_util.tree_map(lambda o, n: jnp.where(ok, n, o), p, new), st, l

        key = jr.PRNGKey(seed + 7)
        ev = max(40, steps // 50)
        best, best_val, since = p, np.inf, 0
        for i in range(steps):
            key, sk = jr.split(key)
            p, st, _ = step(p, st, sk)
            if i % ev == 0 or i == steps - 1:
                v = float(nll_mean(p, th_hold, x_hold))
                if np.isfinite(v) and v < best_val:
                    best_val, best, since = v, p, 0
                else:
                    since += 1
                    if since >= cfg.patience:
                        break
        p = best
        # L1 group norms on the theta-facing layers: which parameters the map uses.
        kin = np.asarray(p["flat"]["params"]["in_proj"]["kernel"])
        ksk = np.asarray(p["flat"]["params"]["skip"]["kernel"])
        usage = np.sqrt((kin ** 2).sum(1) + (ksk ** 2).sum(1))
        return (jax.jit(lambda t: jax.vmap(lambda u: core(p, u))(t)),
                jax.jit(lambda t: jax.vmap(jax.jacfwd(lambda u: core(p, u)))(t)),
                best_val, usage)

    return fit_joint


def spectrum_of(eta: np.ndarray):
    e = eta - eta.mean(0)
    c = np.cov(e, rowvar=False).reshape(e.shape[1], e.shape[1])
    lam, v = np.linalg.eigh(c)
    o = np.argsort(-lam)
    return e @ v[:, o], np.maximum(lam[o], 1e-12)


def fisher_reference(fish_npz: Path):
    """Three-step's 35D mean Fisher spectrum, for comparison against r_hat."""
    with np.load(fish_npz) as z:
        F = np.asarray(z["Fs"])
    F = F[np.isfinite(F).all(axis=(1, 2, 3))]
    mF = F.mean(axis=(0, 1))
    ev = np.sort(np.linalg.eigvalsh(0.5 * (mF + mF.T)))[::-1]
    ev = np.maximum(ev, 0.0)
    rel = ev / (ev[0] + 1e-12)
    return ev, rel


def run(cfg, outdir: Path) -> dict:
    from degeneracy_distillery.sr_utils import (
        analyze_equations,
        filter_pareto_fronts,
        fit_symbolic_regression,
        get_y_sr,
    )

    t0 = time.time()
    log("\n######## 0. data ########")
    data = load_camels(cfg.data_path, cfg.n_pca, cfg.seed)
    th_tr, x_tr = data["th_tr"], data["x_tr"]
    th_te, x_te = data["th_te"], data["x_te"]
    d = th_tr.shape[1]
    log(f"  train {th_tr.shape} / {x_tr.shape}   test {th_te.shape} / {x_te.shape}")

    fisher_ev, fisher_rel = None, None
    if cfg.fishnets_npz and Path(cfg.fishnets_npz).exists():
        fisher_ev, fisher_rel = fisher_reference(Path(cfg.fishnets_npz))
        log(f"  three-step mean Fisher eigs (top 10): {np.round(fisher_ev[:10], 2)}")
        log(f"  three-step rank at rel>1e-2: {int((fisher_rel > 1e-2).sum())} of {d}")

    fit_joint = build_oneshot(th_tr, x_tr, cfg)

    # ---- 1. probe.  m_probe well above the expected rank, not d: a 35-wide
    # Jacobian per sample is expensive and 819 pairs cannot support it.
    log(f"\n######## 1. probe fit at m = {cfg.m_probe} (d = {d}) ########")
    eta_fn, _, val_probe, usage = fit_joint(cfg.m_probe, cfg.steps, cfg.seed)
    _, lam = spectrum_of(np.asarray(eta_fn(th_te)))
    nats = 0.5 * np.log(np.maximum(lam, 1e-12))
    r_hat = max(1, min(int(np.sum(nats > cfg.nats_floor)), cfg.m_probe))
    log(f"  probe lambda {np.round(lam, 4)}")
    log(f"  nats         {np.round(nats, 4)}   floor {cfg.nats_floor}")
    log(f"  r_hat = {r_hat}   val score {val_probe:.4f}")
    if r_hat == cfg.m_probe:
        log("  WARNING: r_hat saturates m_probe; rerun with a larger --m-probe")

    order = np.argsort(-usage)
    log("  parameter usage (top 12 by input-layer group norm):")
    for i in order[:12]:
        log(f"    {PARAM_NAMES[i]:32s} {usage[i]:.4f}")

    # ---- 2. bootstrap ensemble at m = r_hat
    log(f"\n######## 2. {cfg.k_boot} bootstrap members at m = r_hat = {r_hat} ########")
    rng_sr = np.random.default_rng(cfg.seed + 4242)
    th_sr = rng_sr.uniform(th_tr.min(0), th_tr.max(0), size=(cfg.sr_grid, d))
    members, members_sr, jacs, usages = [], [], [], []
    for k in range(cfg.k_boot):
        e_fn, j_fn, v, u = fit_joint(r_hat, cfg.steps, cfg.seed + 101 + k, boot=True)
        members.append(np.asarray(e_fn(th_te)))
        members_sr.append(np.asarray(e_fn(th_sr)))
        jacs.append(np.asarray(j_fn(th_te)))
        usages.append(u)
        log(f"  member {k}: val score {v:.4f}")
    ens = np.stack(members)
    ens_sr = np.stack(members_sr)
    usage_ens = np.stack(usages).mean(0)

    ens_al, rots = align_members(ens)
    eta = ens_al.mean(0)
    y_std_raw = ens_al.std(0, ddof=1)
    ens_sr_al = apply_rotations(ens_sr - ens_sr.mean(1, keepdims=True), rots)
    eta_sr = ens_sr_al.mean(0)
    y_std_sr_raw = ens_sr_al.std(0, ddof=1)
    log(f"  aligned eta: median |y_std| {np.round(np.median(y_std_raw, 0), 4)}")

    K = cfg.k_boot
    mu = ens_al.mean(0)
    mu_loo = (K * mu[None] - ens_al) / (K - 1)
    sd_loo = np.sqrt(np.clip(
        (K * (ens_al.var(0, ddof=0)[None] + mu[None] ** 2) - ens_al ** 2) / (K - 1)
        - mu_loo ** 2, 1e-24, None))
    sd_loo *= np.sqrt((K - 1) / (K - 2))
    zstd = ((ens_al - mu_loo) / (sd_loo * np.sqrt(K / (K - 1)))).reshape(-1, r_hat).std(0)
    tref = float(np.sqrt((K - 2) / (K - 4))) if K > 4 else float("nan")
    log(f"  LOO ensemble z std {np.round(zstd, 3)}  vs t_{K-2} ref {tref:.4f}  "
        f"ratio {np.round(zstd / tref, 3)}  (>1 = y_std too small)")

    dy_sr = apply_rotations(np.stack(jacs), rots, jacobian=True).mean(0)

    log("  ensemble parameter usage (top 12):")
    ord_e = np.argsort(-usage_ens)
    for i in ord_e[:12]:
        log(f"    {PARAM_NAMES[i]:32s} {usage_ens[i]:.4f}")

    # ---- 3. SR
    log("\n######## 3. symbolic regression ########")
    offset = eta_sr.min(0)
    X, y, y_std = th_te, eta - offset, np.maximum(y_std_raw, cfg.y_std_floor)
    X_sr, y_sr = th_sr, eta_sr - offset
    y_std_sr = np.maximum(y_std_sr_raw, cfg.y_std_floor)
    log(f"  SR grid {X_sr.shape}, analyse on test {X.shape}, {r_hat} components")

    sr_dir = outdir / "sr_camels"
    sr_dir.mkdir(parents=True, exist_ok=True)
    fit_symbolic_regression(
        X_sr, y_sr, y_std_sr,
        parent_dir=str(sr_dir) + os.sep,
        random_state=cfg.seed + 60_000,
        time_limit=cfg.sr_time_limit,
        max_length=cfg.sr_max_length,
        max_depth=cfg.sr_max_depth,
        allowed_symbols=cfg.allowed_symbols,
        objectives=["r2", "length"],
    )

    Fs_id = np.repeat(np.eye(d)[None, ...], len(X), axis=0)
    kw = dict(parent_dir=str(sr_dir) + os.sep, n_params=d, equation_set="pareto",
              max_complexity_thresh=cfg.sr_max_length, length_penalty=2.0)

    coord_sets: dict[str, list] = {}
    mdl_raw, frob_raw, _ = analyze_equations(
        X, y, y_std, dy_sr, Fs_id, equation_predicate=None, **kw)
    coord_sets["mdl_unfiltered"] = list(mdl_raw)
    coord_sets["frob_unfiltered"] = list(frob_raw)

    predicate = timed_predicate(d, cfg.predicate_timeout, log)
    summaries = filter_pareto_fronts(str(sr_dir), r_hat, predicate)
    kept = [int(s.get("kept", -1)) for s in summaries]
    original = [int(s.get("original", -1)) for s in summaries]
    removed = sum(int(s["removed"]) for s in summaries)
    log(f"  predicate: {predicate.state['timeouts']} timeouts, "
        f"{predicate.state['spent']:.1f}s total")
    log(f"  Pareto filter: removed {removed}, kept {kept} of {original}")

    if min(kept) > 0:
        try:
            mdl_f, frob_f, _ = analyze_equations(
                X, y, y_std, dy_sr, Fs_id, equation_predicate=predicate, **kw)
            coord_sets["mdl_filtered"] = list(mdl_f)
            coord_sets["frob_filtered"] = list(frob_f)
        except Exception as exc:                          # noqa: BLE001
            log(f"  filtered-front analysis failed: {exc!r}")
    else:
        log("  filter emptied a component's front")

    # ---- 4. held-out score.  No ground truth here, so this is R^2 of the
    # expressions against eta on the test split, per component.
    log("\n######## 4. held-out agreement with eta ########")
    scores = {}
    for tag, coords in coord_sets.items():
        try:
            pred = np.asarray(get_y_sr(list(coords), X)).reshape(len(X), -1)
        except Exception as exc:                          # noqa: BLE001
            log(f"  {tag}: could not evaluate ({exc!r})")
            scores[tag] = {"error": repr(exc)}
            continue
        per = [r2(y[:, j], pred[:, j]) for j in range(r_hat)]
        scores[tag] = {"expr": list(coords), "r2_vs_eta": per,
                       "r2_min": float(np.min(per)), "r2_mean": float(np.mean(per))}
        log(f"  {tag}: R^2 per component {np.round(per, 4)}  "
            f"min {np.min(per):.4f}  mean {np.mean(per):.4f}")

    np.savez(outdir / "camels_oneshot.npz",
             X=X, y=y, y_std=y_std, dy_sr=dy_sr, ens=ens_al,
             X_sr=X_sr, y_sr=y_sr, y_std_sr=y_std_sr,
             lam_probe=lam, usage_probe=usage, usage_ens=usage_ens,
             param_names=np.array(PARAM_NAMES),
             fisher_eigs=(fisher_ev if fisher_ev is not None else np.zeros(0)))

    return {
        "d": d, "m_probe": cfg.m_probe, "r_hat": r_hat,
        "probe_lambda": [float(v) for v in lam],
        "probe_nats": [float(v) for v in nats],
        "fisher_eigs": ([float(v) for v in fisher_ev]
                        if fisher_ev is not None else None),
        "fisher_rank_1e-2": (int((fisher_rel > 1e-2).sum())
                             if fisher_rel is not None else None),
        "usage_ranked": [PARAM_NAMES[i] for i in np.argsort(-usage_ens)[:15]],
        "loo_z_std": [float(v) for v in zstd], "loo_t_ref": tref,
        "pca_cumvar": data["cumvar"],
        "predicate_timeouts": int(predicate.state["timeouts"]),
        "pareto_original": original, "pareto_kept": kept,
        "sr": scores, "seconds": time.time() - t0,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", type=Path,
                   default=Path("data_scratch/data_L50_TNG_v3.hdf5"))
    p.add_argument("--fishnets-npz", type=str,
                   default="results/camels_notebook/full_3656499_20260628_202314/"
                           "fishnets-camels/fishnets_outputs.npz")
    p.add_argument("--out", type=Path, default=Path("new_idea/camels_oneshot_out"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m-probe", type=int, default=12)
    p.add_argument("--n-pca", type=int, default=40)
    p.add_argument("--nats-floor", type=float, default=0.10)
    p.add_argument("--predicate-timeout", type=float, default=30.0)
    p.add_argument("--allowed-symbols", type=str,
                   default="add,mul,div,pow,constant,variable,square,exp")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    smoke = args.smoke
    cfg = argparse.Namespace(
        seed=args.seed, data_path=args.data_path, fishnets_npz=args.fishnets_npz,
        m_probe=args.m_probe, n_pca=args.n_pca, nats_floor=args.nats_floor,
        steps=600 if smoke else 6000,
        k_boot=3 if smoke else 8,
        batch=64, batch_aug=256, lr=5e-4, weight_decay=1e-3, l1=2.0,
        patience=20,
        sr_grid=500 if smoke else 3000,
        sr_time_limit=30 if smoke else 600,
        sr_max_length=30, sr_max_depth=20,
        y_std_floor=1e-3,
        predicate_timeout=args.predicate_timeout,
        allowed_symbols=args.allowed_symbols,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    log(f"CAMELS-SB35 one-step, all {N_PARAMS} parameters, no pre-selection")
    log(f"mode={'smoke' if smoke else 'full'}  m_probe={cfg.m_probe}  "
        f"n_pca={cfg.n_pca}  steps={cfg.steps}  K={cfg.k_boot}")
    log(f"allowed_symbols={cfg.allowed_symbols}")

    row = run(cfg, args.out)
    out_json = args.out / "camels_oneshot.json"
    out_json.write_text(json.dumps(row, indent=2))
    log("\n================ summary ================")
    log(f"  r_hat = {row['r_hat']} of d = {row['d']}  "
        f"(three-step Fisher rank at rel>1e-2: {row['fisher_rank_1e-2']})")
    log(f"  top parameters: {', '.join(row['usage_ranked'][:8])}")
    best = row["sr"].get("mdl_filtered") or row["sr"].get("mdl_unfiltered", {})
    if "r2_mean" in best:
        log(f"  SR vs eta: mean R^2 {best['r2_mean']:.4f}  min {best['r2_min']:.4f}")
    log(f"  wrote {out_json}  ({row['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
