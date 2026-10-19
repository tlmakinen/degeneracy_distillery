"""CAMELS with the square-completed loss: d=12 participating, K=7 active.

The data-side ensemble is preserved. Each member draws its own eta(x) width,
depth and activation from ``estimator_bank``; the eta(theta) flow has one
fixed architecture. ``y_std`` therefore stays defined on the augmented draws,
which a single shared eta could not give (no x exists there).

Only ``eta_a`` is read. ``eta_b`` carries the prior along the fibres and is
gauge-dependent (see square_completion_loss.md, section 6).
"""
from __future__ import annotations

import argparse, json, math, os, time
from pathlib import Path

import numpy as np


def log(*a):
    print(*a, flush=True)


def participating_set(F, Sp, k):
    """Support of the lambda >= 1 eigenvectors of F v = lam Sigma_pi^-1 v."""
    w, V = np.linalg.eigh(Sp)
    H = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
    lam, U = np.linalg.eigh(H @ F @ H)
    o = np.argsort(-lam)
    lam = lam[o]
    n_act = int((lam >= 1.0).sum())
    support = np.abs(U[:, o[:n_act]]).max(1)
    keep = np.sort(np.argsort(-support)[:k])
    return keep, lam, n_act, support


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=Path, default=Path("data_scratch/data_L50_TNG_v3.hdf5"))
    p.add_argument("--n-pca", type=int, default=0, help="0 = raw 344 standardised features")
    p.add_argument("--d-keep", type=int, default=12)
    p.add_argument("--k-active", type=int, default=7)
    p.add_argument("--box-margin", type=float, default=0.05,
                   help="Widen the logit box beyond the theta range by this "
                        "fraction. fit_theta_scaler maps the extreme simulation "
                        "to exactly the box edge, where the logit coordinate "
                        "diverges; the box must strictly contain the data.")
    # 819 training rows against a flow whose -log|det J| is unbounded below.
    # Keep the flow small and select on held-out NLL.
    p.add_argument("--flow-layers", type=int, default=4)
    p.add_argument("--flow-width", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--prior-reg", type=float, default=1.0,
                   help="Weight on a prior-draw term that scores the flow with "
                        "t := 0, i.e. 0.5||eta||^2 - log|det J|. Prior draws "
                        "cost no simulations and cannot be memorised, so this "
                        "stops the flow inflating the volume on the 819 "
                        "training rows. Fixes the marginal, not the "
                        "conditional (square_completion_loss.md section 5).")
    p.add_argument("--prior-batch", type=int, default=256)
    p.add_argument("--drop-tol", type=float, default=5.0,
                   help="Drop ensemble members whose best held-out NLL exceeds "
                        "the median by more than this.")
    p.add_argument("--ensemble-k", type=int, default=10)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sr-n-aug", type=int, default=20000)
    p.add_argument("--sr-time-limit", type=int, default=300)
    p.add_argument("--skip-sr", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import jax, jax.numpy as jnp, optax
    jax.config.update("jax_enable_x64", True)
    from degeneracy_distillery.problems.base import get_problem
    from degeneracy_distillery.problems.camels import PARAM_NAMES
    from degeneracy_distillery.oneshot import estimator_bank
    from degeneracy_distillery.square_completion import SquareEta, square_loss

    a.out.mkdir(parents=True, exist_ok=True)
    prob = get_problem("camels", data_path=str(a.data_path), n_pca=a.n_pca,
                       seed=a.seed, prior_draw="empirical")
    th_tr, x_tr = prob.th_tr, prob.x_tr
    th_te, x_te = prob.held_out()
    log(f"[data] train {th_tr.shape} {x_tr.shape}   test {th_te.shape} {x_te.shape}")

    # ---- participating set, from the prior-relative Fisher of a pilot map ----
    # Use the saved 35-parameter Fisher when available, else the design itself.
    pilot = Path(os.environ.get("PILOT_FISHER", ""))
    if pilot.is_file():
        F = np.load(pilot)["F"]
    else:
        from sklearn.linear_model import Ridge
        # cheap surrogate Fisher: J of a ridge fit of x onto theta
        W = Ridge(alpha=1.0).fit(th_tr, x_tr).coef_          # (n_feat, 35)
        F = W.T @ W
    keep, lam, n_act, support = participating_set(F, np.cov(th_tr, rowvar=False), a.d_keep)
    names = [PARAM_NAMES[i] for i in keep]
    log(f"[screen] lambda top-10 {np.round(lam[:10], 3)}   lam>=1 -> K {n_act}")
    log(f"[screen] keeping d={len(keep)}: {names}")

    th_tr, th_te = th_tr[:, keep], th_te[:, keep]
    d, K = len(keep), a.k_active
    mg = float(a.box_margin)
    lo = jnp.full(d, 1.0 - mg); hi = jnp.full(d, 2.0 + mg)
    u_edge = (np.array([th_tr.min(), th_tr.max()]) - (1.0 - mg)) / (1.0 + 2 * mg)
    log(f"[box] margin {mg}  box [{1-mg:.3f}, {2+mg:.3f}]  "
        f"u range [{u_edge[0]:.4f}, {u_edge[1]:.4f}]  "
        f"max |logit/s| {abs(np.log(u_edge[0]/(1-u_edge[0])))/ (np.pi/np.sqrt(3)):.3f}")

    # ---- train K members: shared eta architecture, varied data network ----
    bank = estimator_bank(a.ensemble_k, seed=a.seed, hids=(64, 320), n_layers=(2, 4))
    log(f"[ensemble] eta(x) bank: {[f.tag for f in bank]}")
    THtr, Xtr = jnp.asarray(th_tr), jnp.asarray(x_tr)
    THte, Xte = jnp.asarray(th_te), jnp.asarray(x_te)

    # Prior pool for the flow regulariser. Drawn once: a numpy call inside the
    # step would cross the jit boundary every iteration.
    pool = jnp.asarray(prob.sample_prior(50000, np.random.default_rng(a.seed + 5))[:, keep])
    log(f"[prior] regulariser pool {tuple(pool.shape)}  weight {a.prior_reg}")

    members = []
    for m_i, fac in enumerate(bank):
        eta_mod = SquareEta(d=d, lo=lo, hi=hi, n_layers=a.flow_layers,
                            features=(a.flow_width, a.flow_width))
        key = jax.random.PRNGKey(a.seed + 977 * m_i)
        k1, k2 = jax.random.split(key)
        pe = eta_mod.init(k1, THtr[0])
        if K > 0:
            est_mod = fac(K)
            par = {"eta": pe, "est": est_mod.init(k2, Xtr[0])}
        else:
            # K = 0 is the no-information baseline: every coordinate is pulled
            # to N(0, I) and no statistic enters. Its held-out NLL is the
            # prior-only score, so (NLL_0 - NLL_K) is the information the data
            # carry about the kept parameters, in nats.
            est_mod = None
            par = {"eta": pe}

        def nll(par, th, x):
            eta, ladj = eta_mod.apply(par["eta"], th)
            t = (est_mod.apply(par["est"], x) if K > 0
                 else jnp.zeros((0,), dtype=eta.dtype))
            return square_loss(eta, ladj, t, K, d)

        def batch_loss(par, th, x):
            return jnp.mean(jax.vmap(lambda u, v: nll(par, u, v))(th, x))

        def prior_nll(par, th):
            """Flow score with the statistic set to zero: all d coords to N(0,I)."""
            def one(t):
                eta, ladj = eta_mod.apply(par["eta"], t)
                return 0.5 * jnp.sum(eta ** 2) - ladj + 0.5 * d * math.log(2 * math.pi)
            return jnp.mean(jax.vmap(one)(th))

        def train_obj(par, th, x, thp):
            return batch_loss(par, th, x) + a.prior_reg * prior_nll(par, thp)

        opt = optax.adam(a.lr)
        st = opt.init(par)

        @jax.jit
        def step(par, st, key):
            k1, k2 = jax.random.split(key)
            idx = jax.random.randint(k1, (a.batch,), 0, THtr.shape[0])
            pid = jax.random.randint(k2, (a.prior_batch,), 0, pool.shape[0])
            g = jax.grad(train_obj)(par, THtr[idx], Xtr[idx], pool[pid])
            upd, st = opt.update(g, st)
            return optax.apply_updates(par, upd), st

        t0 = time.time()
        kk = jax.random.PRNGKey(a.seed + 31 * m_i)
        best_val, best_par, best_step = np.inf, par, 0
        hist = []
        for s_i in range(a.steps):
            kk, sub = jax.random.split(kk)
            par, st = step(par, st, sub)
            if (s_i + 1) % a.eval_every == 0:
                v = float(batch_loss(par, THte, Xte))
                tr = float(batch_loss(par, THtr, Xtr))
                hist.append((s_i + 1, tr, v))
                if np.isfinite(v) and v < best_val:
                    best_val, best_par, best_step = v, par, s_i + 1
        par = best_par
        val = best_val
        tr0 = hist[0][1] if hist else float("nan")
        trl, vl = (hist[-1][1], hist[-1][2]) if hist else (float("nan"),) * 2
        log(f"[member {m_i}] {fac.tag:26s} best heldout {val:+.4f} @ step {best_step}"
            f"   final train {trl:+.3f} / heldout {vl:+.3f}   {time.time()-t0:.0f}s")
        members.append({"par": par, "eta_mod": eta_mod, "est_mod": est_mod, "nll": val})

    # ---- evaluate eta_a and its Jacobian on the test split ----
    def eta_a_of(mem, TH):
        f = lambda t: mem["eta_mod"].apply(mem["par"]["eta"], t)[0][:K]
        return np.asarray(jax.vmap(f)(TH)), np.asarray(jax.vmap(jax.jacfwd(f))(TH))

    if K == 0:
        nlls = np.array([m["nll"] for m in members])
        log(f"[ensemble] K=0 prior-only held-out NLL  mean {nlls.mean():+.4f}  "
            f"best {nlls.min():+.4f}")
        json.dump({"d": d, "K": 0, "keep": [int(v) for v in keep], "names": names,
                   "nll_mean": float(nlls.mean()), "nll_best": float(nlls.min())},
                  open(a.out / "summary.json", "w"), indent=2)
        log(f"[done] wrote {a.out}")
        return

    # One diverged member poisons the ensemble mean and inflates every
    # calibration ratio, so drop members far from the median held-out NLL.
    allnll = np.array([m["nll"] for m in members], dtype=float)
    med = float(np.median(allnll[np.isfinite(allnll)]))
    good = [i for i, v in enumerate(allnll)
            if np.isfinite(v) and v < med + a.drop_tol]
    if len(good) < len(members):
        drop = [(i, float(allnll[i])) for i in range(len(members)) if i not in good]
        log(f"[ensemble] dropped {len(drop)} diverged member(s): "
            + ", ".join(f"{i} (NLL {v:+.3g})" for i, v in drop))
    if len(good) < 3:
        raise RuntimeError(f"only {len(good)} healthy members; cannot calibrate")
    members = [members[i] for i in good]

    # Per-direction held-out information of the fitted model:
    #   0.5 log( Var_prior[eta_a_i] / Var[eta_a_i - t_i] )
    # At the optimum eta_a|x ~ N(t, I), so a direction the data cannot predict
    # has residual variance equal to its prior variance and scores 0. This asks
    # the fitted model how many directions it actually uses, which the nesting
    # K-screen cannot: held-out NLL is non-increasing in K by construction.
    def info_dirs(mem):
        ea = np.asarray(jax.vmap(
            lambda t: mem["eta_mod"].apply(mem["par"]["eta"], t)[0][:K])(THte))
        tt = np.asarray(jax.vmap(
            lambda v: mem["est_mod"].apply(mem["par"]["est"], v))(Xte))
        rot, lamv, V = _spectrum(ea)
        hat = (tt - ea.mean(0)) @ V
        resid = np.mean((rot - hat) ** 2, axis=0)
        return 0.5 * np.log(lamv / np.maximum(resid, 1e-12))

    from degeneracy_distillery.oneshot import spectrum as _spectrum
    inf_dirs = np.stack([info_dirs(m) for m in members])
    log("[info] per-direction held-out nats (median over members): "
        + np.array2string(np.median(inf_dirs, 0), precision=3))
    log(f"[info] directions above 0.1 nats: "
        f"{int((np.median(inf_dirs,0) > 0.1).sum())} of {K}")

    etas, jacs = zip(*[eta_a_of(m, THte) for m in members])
    etas, jacs = np.stack(etas), np.stack(jacs)
    nlls = np.array([m["nll"] for m in members])

    # Gauge-invariant rank of the fitted map. eta_a is defined only up to
    # R in O(K), and (R J)^T (R J) = J^T J, so the prior-relative spectrum of
    # F_a = J_a^T J_a does not depend on the rotation. Counting per-direction
    # information does, which is why that count drifts with K.
    Fa = np.einsum("knmi,knmj->kij", jacs, jacs) / jacs.shape[1]
    w_, V_ = np.linalg.eigh(np.cov(th_te, rowvar=False))
    H_ = V_ @ np.diag(np.sqrt(np.clip(w_, 0, None))) @ V_.T
    lam_a = np.stack([np.linalg.eigvalsh(H_ @ F @ H_)[::-1] for F in Fa])
    med = np.median(lam_a, 0)
    log("[rank] prior-relative eig of J_a^T J_a (median over members): "
        + np.array2string(med, precision=3))
    log(f"[rank] lam >= 1: {int((med >= 1).sum())}   "
        f"lam >= 0.5: {int((med >= 0.5).sum())}   of d={d}")
    log(f"[ensemble] held-out NLL  mean {nlls.mean():+.4f}  best {nlls.min():+.4f}")

    from degeneracy_distillery.align_coords import process_ensemble_rotation_v2
    Fe = np.einsum("knmi,knmj->knij", jacs, jacs)
    best = int(np.argmin(nlls))
    w = np.ones(len(members)) / len(members)
    aligned = process_ensemble_rotation_v2(
        datafile={"theta": th_te, "eta_ensemble": etas, "Jbar_ensemble": jacs,
                  "F_ensemble": Fe, "ensemble_weights": w, "norm_factor": 1.0},
        randidx=np.arange(th_te.shape[0]), Favg=Fe[best], best_model_idx=best,
        n_d=1.0, align_mode="procrustes", separate_nonlinearity=True,
        canonicalize="none", use_prior_normalization=True,
        restore_reference_mean=True, Fisher_to_flatten="best",
        verbose=True, offset_delta=0.1,
    )
    from degeneracy_distillery.oneshot_ensemble import ensemble_calibration
    calib = ensemble_calibration(np.asarray(aligned["ys"]))
    log(f"[calib] K={calib['K']}  LOO ratio {np.round(calib['ratio_loo'],3)}  "
        f"ref {calib['ref_loo']:.4f}")

    np.savez_compressed(a.out / "ensemble.npz",
                        ys=np.asarray(aligned["ys"]), y=np.asarray(aligned["y"]),
                        y_std=np.asarray(aligned["y_std"]), X=th_te,
                        keep=keep, nlls=nlls, lam=lam,
                        ratio_loo=calib["ratio_loo"], z_loo=calib["z_loo"],
                        info_dirs=inf_dirs, lam_a=lam_a)
    json.dump({"keep": [int(v) for v in keep], "names": names,
               "lam": [float(v) for v in lam[:12]], "n_active": int(n_act),
               "nll_mean": float(nlls.mean()), "nll_best": float(nlls.min()),
               "ratio_loo": [float(v) for v in np.atleast_1d(calib["ratio_loo"])],
               "d": d, "K": K,
               "info_dirs_median": [float(v) for v in np.median(inf_dirs, 0)],
               "n_dirs_above_0p1": int((np.median(inf_dirs, 0) > 0.1).sum()),
               "lam_a_median": [float(v) for v in np.median(lam_a, 0)],
               "rank_lam_ge_1": int((np.median(lam_a, 0) >= 1).sum())},
              open(a.out / "summary.json", "w"), indent=2)
    log(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()
