#!/usr/bin/env python
"""Test A: one-step joint map vs three-step Fishnets-then-flatten.

Rosenbrock in d = 2, 3, 4. The first two coordinates generate the banana.
Any leftover coordinates are unused (held at the prior by a correct method).

    mu(x) = (theta_0, theta_1 - theta_0^2)

Same (theta, x) pairs go to both methods. No symbolic regression. The score
is cubic R^2 of the true coordinates against the learned eta axes, plus
whether the method reports rank 2.

    python new_idea/compare_rosen_ab.py --quick
    python new_idea/compare_rosen_ab.py --dims 2 3 4
    python new_idea/compare_rosen_ab.py --full --dims 2 3 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Shared problem
# ---------------------------------------------------------------------------
A_BOX = 3.0
WIDTH = 2.0 * A_BOX
LOGW = float(np.log(WIDTH))
SD = (0.25, 0.5)
N_REP = 8
TRUE_RANK = 2
I0, I1 = 0, 1


def true_coords(th: np.ndarray) -> np.ndarray:
    return np.column_stack([th[:, I0], th[:, I1] - th[:, I0] ** 2])


def simulate(n: int, d: int, rng: np.random.Generator):
    th = rng.uniform(-A_BOX, A_BOX, size=(n, d))
    mu = np.stack([th[:, I0], th[:, I1] - th[:, I0] ** 2], axis=1)
    x = mu[:, None, :] + rng.normal(size=(n, N_REP, 2)) * np.asarray(SD)
    return th, x


def poly_r2(y: np.ndarray, feats: np.ndarray, deg: int = 3) -> float:
    k = feats.shape[1]
    f = (feats - feats.mean(0)) / (feats.std(0) + 1e-12)
    cols = [f[:, j] for j in range(k)]
    if deg >= 2:
        cols += [f[:, i] * f[:, j] for i in range(k) for j in range(i, k)]
    if deg >= 3:
        cols += [
            f[:, i] * f[:, j] * f[:, l]
            for i in range(k)
            for j in range(i, k)
            for l in range(j, k)
        ]
    xf = np.column_stack(cols + [np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(xf, y, rcond=None)
    return float(1.0 - ((y - xf @ beta) ** 2).mean() / (y.var() + 1e-12))


def spectrum(eta: np.ndarray):
    e = eta - eta.mean(0)
    m = e.shape[1]
    c = np.cov(e, rowvar=False).reshape(m, m)
    lam, v = np.linalg.eigh(c)
    o = np.argsort(-lam)
    return e @ v[:, o], np.maximum(lam[o], 1e-12)


def recovery(c_true: np.ndarray, eta: np.ndarray, r_keep: int) -> dict:
    e, lam = spectrum(eta)
    r_keep = max(1, min(r_keep, e.shape[1]))
    return {
        "lambda": [float(v) for v in lam],
        "r2_true_from_topr": [poly_r2(c_true[:, k], e[:, :r_keep]) for k in (0, 1)],
        "r2_true_from_all": [poly_r2(c_true[:, k], e) for k in (0, 1)],
        "r2_axis_from_true": [poly_r2(e[:, k], c_true) for k in range(r_keep)],
    }


# ---------------------------------------------------------------------------
# One-step (hard-cut rectangular protocol, no bootstrap / MDL)
# ---------------------------------------------------------------------------
def run_oneshot(th, x, th_te, x_te, cfg) -> dict:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import optax

    jax.config.update("jax_enable_x64", True)

    d = th.shape[1]
    mu_x = x.reshape(-1, 2).mean(0)
    sd_x = x.reshape(-1, 2).std(0) + 1e-12
    th_j = jnp.asarray(th)
    x_j = jnp.asarray((x - mu_x) / sd_x)
    th_te_j = jnp.asarray(th_te)
    x_te_j = jnp.asarray((x_te - mu_x) / sd_x)
    xb = jnp.asarray((x.mean(1) - mu_x) / sd_x)

    nval = max(40, int(0.15 * th.shape[0]))
    th_fit, x_fit = th_j[:-nval], x_j[:-nval]
    th_val, x_val = th_j[-nval:], x_j[-nval:]

    class Flattener(nn.Module):
        m: int
        skip_init: np.ndarray
        hidden: Sequence[int] = (64, 64)

        @nn.compact
        def __call__(self, theta):
            h = nn.gelu(nn.Dense(self.hidden[0], name="in_proj")(theta))
            for k, w in enumerate(self.hidden[1:]):
                h = nn.gelu(nn.Dense(w, name=f"h{k}")(h))
            delta = nn.Dense(
                self.m, name="out",
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(h)
            skip = nn.Dense(
                self.m, use_bias=False, name="skip",
                kernel_init=lambda k, s, dt: jnp.asarray(self.skip_init, dt),
            )(theta)
            return skip + delta

    class Estimator(nn.Module):
        m: int
        hidden: Sequence[int] = (32, 32)

        @nn.compact
        def __call__(self, xx):
            h = xx.mean(0)
            for w in self.hidden:
                h = nn.gelu(nn.Dense(w)(h))
            return (
                nn.Dense(self.m, kernel_init=nn.initializers.zeros,
                         bias_init=nn.initializers.zeros)(h)
                + nn.Dense(self.m)(xx.mean(0))
            )

    class Screen(nn.Module):
        hidden: Sequence[int] = (64, 64)

        @nn.compact
        def __call__(self, theta):
            h = nn.gelu(nn.Dense(self.hidden[0], name="in_proj")(theta))
            for k, w in enumerate(self.hidden[1:]):
                h = nn.gelu(nn.Dense(w, name=f"h{k}")(h))
            return nn.Dense(2, name="out")(h)

    def screen_inputs(th_s, xb_s, steps: int, seed: int):
        mod = Screen()
        ps = mod.init(jr.PRNGKey(seed), th_s[0])
        tx = optax.adam(3e-3)
        st = tx.init(ps)
        n = th_s.shape[0]
        b = min(512, n)

        @jax.jit
        def one(ps, st, key):
            i = jr.randint(key, (b,), 0, n)

            def obj(ps):
                pr = jax.vmap(lambda t: mod.apply(ps, t))(th_s[i])
                k = ps["params"]["in_proj"]["kernel"]
                return (
                    jnp.mean((pr - xb_s[i]) ** 2)
                    + 1e-2 * jnp.sum(jnp.sqrt(jnp.sum(k ** 2, 1) + 1e-12))
                )

            l, g = jax.value_and_grad(obj)(ps)
            u, st = tx.update(g, st, ps)
            return optax.apply_updates(ps, u), st, l

        key = jr.PRNGKey(seed + 1)
        for _ in range(steps):
            key, sk = jr.split(key)
            ps, st, _ = one(ps, st, sk)
        return np.sqrt((np.asarray(ps["params"]["in_proj"]["kernel"]) ** 2).sum(1))

    def fit_joint(m, active, steps, seed):
        n_fit = th_fit.shape[0]
        bp = min(256, n_fit)
        act = jnp.asarray(np.asarray(active, dtype=np.int32))
        q0 = np.zeros((d, m))
        q0[np.asarray(active, dtype=int), np.arange(m)] = 1.0
        flat_mod = Flattener(m=m, skip_init=q0)
        est_mod = Estimator(m=m)
        k0, k1 = jr.split(jr.PRNGKey(seed), 2)
        p = {
            "est": est_mod.init(k1, x_fit[0]),
            "raw_log_scale": jnp.zeros(m),
            "flat": flat_mod.init(k0, th_fit[0]),
        }
        const = 0.5 * m * np.log(2 * np.pi) + (d - m) * LOGW
        eye_m = jnp.eye(m)
        l1_w = 2.0 / n_fit

        def core(p, theta):
            g = flat_mod.apply(p["flat"], theta)
            return jnp.exp(2.0 * p["raw_log_scale"]) * g

        def logdet_active(p, theta):
            j = jax.jacfwd(lambda a: core(p, theta.at[act].set(a)))(theta[act])
            return 0.5 * jnp.linalg.slogdet(j.T @ j + 1e-10 * eye_m)[1]

        def zvec(p, th_b, x_b):
            s = jnp.exp(2.0 * p["raw_log_scale"])
            return jax.vmap(
                lambda t, y: core(p, t) - s * est_mod.apply(p["est"], y)
            )(th_b, x_b)

        def nll_vec(p, th_b, x_b):
            s = jnp.exp(2.0 * p["raw_log_scale"])

            def one(theta, xx):
                r = core(p, theta) - s * est_mod.apply(p["est"], xx)
                return 0.5 * jnp.sum(r ** 2) - logdet_active(p, theta) + const

            return jax.vmap(one)(th_b, x_b)

        nll_mean = jax.jit(lambda p, t, y: jnp.mean(nll_vec(p, t, y)))
        warm = max(1, min(300, steps // 10))
        sched = optax.warmup_cosine_decay_schedule(
            1e-5, 1e-3, warm, max(steps, warm + 1), 1e-5
        )
        tx = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adamw(sched, weight_decay=1e-4),
        )
        st = tx.init(p)

        @jax.jit
        def step(p, st, key):
            kp, ka = jr.split(key)
            idx = jr.randint(kp, (bp,), 0, n_fit)
            th_aug = jr.uniform(ka, (min(512, 8 * bp), d), minval=-A_BOX, maxval=A_BOX)

            def obj(p):
                z = zvec(p, th_fit[idx], x_fit[idx])
                geom = -jnp.mean(jax.vmap(lambda t: logdet_active(p, t))(th_aug))
                k1 = p["flat"]["params"]["in_proj"]["kernel"]
                ks = p["flat"]["params"]["skip"]["kernel"]
                pen = jnp.sum(jnp.sqrt(jnp.sum(k1 ** 2, 1) + jnp.sum(ks ** 2, 1) + 1e-12))
                return 0.5 * jnp.mean(jnp.sum(z ** 2, 1)) + geom + const + l1_w * pen

            l, g = jax.value_and_grad(obj)(p)
            upd, st = tx.update(g, st, p)
            new = optax.apply_updates(p, upd)
            ok = jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(z)) for z in jax.tree_util.tree_leaves(g)
            ]))
            return jax.tree_util.tree_map(lambda o, n: jnp.where(ok, n, o), p, new), st, l

        key = jr.PRNGKey(seed + 7)
        ev = max(40, steps // 50)
        best, best_val = p, np.inf
        for i in range(steps):
            key, sk = jr.split(key)
            p, st, _ = step(p, st, sk)
            if i % ev == 0 or i == steps - 1:
                v = float(nll_mean(p, th_val, x_val))
                if np.isfinite(v) and v < best_val:
                    best_val, best = v, p
        p = best
        eta_fn = jax.jit(lambda t: jax.vmap(lambda u: core(p, u))(t))
        return eta_fn, best_val

    t0 = time.time()
    # Probe at full d. A tail-median null needs spare axes (the 20D protocol).
    # Here d is 2-4, so rank is read against the theoretical null lambda = 1.
    m_probe = d
    en = screen_inputs(th_j, xb, steps=cfg.screen_steps, seed=cfg.seed)
    order = np.argsort(-en)
    a_probe = tuple(sorted(int(v) for v in order[:m_probe]))
    eta_fn, _ = fit_joint(m_probe, a_probe, cfg.oneshot_steps, cfg.seed)
    _, lam_probe = spectrum(np.asarray(eta_fn(th_te_j)))
    nats = 0.5 * np.log(np.maximum(lam_probe, 1e-12))
    r_hat = int(np.sum(nats > cfg.nats_floor))
    r_hat = max(1, min(r_hat, m_probe))
    if r_hat != m_probe:
        a_hat = tuple(sorted(int(v) for v in order[:r_hat]))
        eta_fn, _ = fit_joint(r_hat, a_hat, cfg.oneshot_steps, cfg.seed + 1)
        active = a_hat
    else:
        active = a_probe
    eta = np.asarray(eta_fn(th_te_j))
    rec = recovery(true_coords(th_te), eta, r_hat)
    rec.update({
        "method": "oneshot",
        "r_hat": r_hat,
        "active": list(active),
        "screen_order": [int(v) for v in order],
        "probe_lambda": [float(v) for v in lam_probe],
        "seconds": time.time() - t0,
        "rank_ok": r_hat == TRUE_RANK,
        "set_ok": set(active) == {I0, I1} if r_hat == TRUE_RANK else False,
    })
    return rec


# ---------------------------------------------------------------------------
# Three-step (Fishnets + flatten). Square map in all d coordinates.
# ---------------------------------------------------------------------------
def run_threestep(th, x, th_te, x_te, cfg, outdir: Path) -> dict:
    import jax
    import jax.numpy as jnp

    from degeneracy_distillery.sr_utils import fit_theta_scaler
    from degeneracy_distillery.training_loop_fishnets import train_fishnets
    from degeneracy_distillery.training_loop_flatten import fit_flattening

    t0 = time.time()
    d = th.shape[1]
    x_flat = x.reshape(x.shape[0], -1)
    x_te_flat = x_te.reshape(x_te.shape[0], -1)
    scaler = fit_theta_scaler(th, feature_range=(-3.0, 3.0))
    th_s = scaler.transform(th).astype(np.float32)
    th_te_s = scaler.transform(th_te).astype(np.float32)

    fish_dir = outdir / f"fishnets_d{d}"
    train_fishnets(
        th_s,
        x_flat.astype(np.float32),
        th_te_s,
        x_te_flat.astype(np.float32),
        num_models=cfg.num_fishnets,
        hids_min=24,
        hids_max=64,
        n_layers=[2, 3],
        train_epochs=cfg.fish_epochs,
        train_min_epochs=min(80, cfg.fish_epochs),
        patience=20,
        train_batch_size=min(25, th.shape[0]),
        lr=5e-5,
        seed_model=cfg.seed + 201,
        seed_train=cfg.seed + 999,
        outdir=str(fish_dir),
        update_pbar_every=50,
    )
    fish = np.load(fish_dir / "fishnets_outputs.npz")
    thetas = jnp.array(fish["theta"])
    fs_np = np.asarray(fish["Fs"])
    wts = np.asarray(fish["ensemble_weights"])
    finite = np.isfinite(fs_np).all(axis=(1, 2, 3))
    if not finite.any():
        raise RuntimeError("all fishnet members produced non-finite Fishers")
    fs_np, wts = fs_np[finite], wts[finite]
    f_mean = np.average(fs_np.mean(1), axis=0, weights=wts)
    f_eigs = np.sort(np.linalg.eigvalsh(0.5 * (f_mean + f_mean.T)))[::-1]
    f_eigs = np.maximum(f_eigs, 0.0)
    rel = f_eigs / (f_eigs[0] + 1e-12)
    r_fish = int(np.sum(rel > cfg.fish_eig_floor))

    cwd = Path.cwd()
    os.chdir(outdir)
    try:
        w, ensemble_w, _, flatten_model = fit_flattening(
            jnp.array(fs_np),
            thetas,
            ensemble_weights=wts,
            hidden_size=64,
            n_layers=3,
            batch_size=min(50, th.shape[0]),
            epochs_phase1=cfg.flatten_epochs,
            epochs_phase2=cfg.flatten_epochs,
            finetune_epochs=max(50, cfg.flatten_epochs // 4),
            min_epochs=min(80, cfg.flatten_epochs),
            patience=30,
            lr_phase1=1e-6,
            lr_schedule_initial=7e-5,
            lr_decay=0.3,
            lr_finetune=4e-6,
            Fisher_to_flatten="average",
            norm_factor=None,
            norm_method="median_det",
            flattener_activation="softplus",
            noise=1e-4,
            seed=cfg.seed,
            output_prefix=f"flatten_d{d}",
            use_whitening=True,
            nn_inv=False,
            forward_backward_mlp=True,
            l1_alpha=0.0,
            do_plot=False,
            return_model=True,
            save_flatten_model_pickle=False,
            update_pbar_every=50,
        )
    finally:
        os.chdir(cwd)

    th_te_j = jnp.asarray(th_te_s)
    etas = []
    for w_i in ensemble_w:
        etas.append(np.asarray(jax.vmap(lambda t: flatten_model.apply(w_i, t))(th_te_j)))
    eta = np.average(np.stack(etas, 0), 0, weights=wts)
    rec = recovery(true_coords(th_te), eta, min(TRUE_RANK, d))
    rec.update({
        "method": "threestep",
        "r_hat": r_fish,
        "fisher_eigs": [float(v) for v in f_eigs],
        "fisher_rel": [float(v) for v in rel],
        "seconds": time.time() - t0,
        "rank_ok": r_fish == TRUE_RANK,
        "n_finite_fishnets": int(finite.sum()),
    })
    return rec


# ---------------------------------------------------------------------------
def summarize_row(d: int, rec: dict) -> str:
    r2 = rec["r2_true_from_topr"]
    return (
        f"d={d}  {rec['method']:9s}  r_hat={rec['r_hat']}  "
        f"rank_ok={rec.get('rank_ok')}  "
        f"R2(true|top-r)=[{r2[0]:.4f}, {r2[1]:.4f}]  "
        f"{rec['seconds']:.1f}s"
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--methods", nargs="+", default=["oneshot", "threestep"],
                   choices=["oneshot", "threestep"])
    p.add_argument("--nsims", type=int, default=None)
    p.add_argument("--ntest", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("new_idea/compare_rosen_ab_out"))
    p.add_argument("--quick", action="store_true", help="tiny CPU smoke")
    p.add_argument("--full", action="store_true", help="heavier budget, still not paper-scale")
    return p.parse_args()


def budgets(args):
    if args.quick:
        return dict(
            nsims=args.nsims or 400, ntest=args.ntest or 200,
            oneshot_steps=200, screen_steps=200,
            m_probe=3, nats_floor=0.10, fish_eig_floor=1e-2,
            num_fishnets=2, fish_epochs=80, flatten_epochs=80,
        )
    if args.full:
        return dict(
            nsims=args.nsims or 8000, ntest=args.ntest or 4000,
            oneshot_steps=4000, screen_steps=2000,
            m_probe=3, nats_floor=0.10, fish_eig_floor=1e-2,
            num_fishnets=5, fish_epochs=600, flatten_epochs=600,
        )
    return dict(
        nsims=args.nsims or 2000, ntest=args.ntest or 1000,
        oneshot_steps=1500, screen_steps=800,
        m_probe=3, nats_floor=0.10, fish_eig_floor=1e-2,
        num_fishnets=3, fish_epochs=250, flatten_epochs=250,
    )


def main():
    args = parse_args()
    b = budgets(args)
    cfg = argparse.Namespace(seed=args.seed, **b)
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"test A  dims={args.dims}  methods={args.methods}  "
        f"nsims={cfg.nsims}  ntest={cfg.ntest}"
    )
    print("truth: informative pair theta_0, theta_1; remaining coords unused")

    rows = []
    for d in args.dims:
        if d < 2:
            raise SystemExit("d must be >= 2")
        rng = np.random.default_rng(args.seed + 1000 * d)
        th, x = simulate(cfg.nsims, d, rng)
        th_te, x_te = simulate(cfg.ntest, d, np.random.default_rng(7777 + d))
        print(f"\n======== d={d}  ({d - 2} unused coords) ========")
        for method in args.methods:
            try:
                if method == "oneshot":
                    rec = run_oneshot(th, x, th_te, x_te, cfg)
                else:
                    rec = run_threestep(th, x, th_te, x_te, cfg, args.out)
            except Exception as exc:
                rec = {"method": method, "error": repr(exc), "r_hat": None,
                       "r2_true_from_topr": [float("nan"), float("nan")],
                       "rank_ok": False, "seconds": 0.0}
                print(f"  FAILED {method}: {exc}")
            rec["d"] = d
            rows.append(rec)
            if "error" not in rec:
                print("  " + summarize_row(d, rec))
                if method == "oneshot":
                    print(f"    screen {rec['screen_order']}  active {rec['active']}"
                          f"  set_ok={rec['set_ok']}")
                    print(f"    probe lambda {np.round(rec['probe_lambda'], 3)}")
                    print(f"    fit  lambda {np.round(rec['lambda'], 3)}")
                else:
                    print(f"    F eigs {np.round(rec['fisher_eigs'], 4)}")

    out_json = args.out / "compare_rosen_ab.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print("\n======== summary ========")
    print(f"{'d':>3}  {'method':9s}  {'r_hat':>5}  {'rank':>5}  "
          f"{'R2_0':>7}  {'R2_1':>7}  {'sec':>7}")
    for rec in rows:
        if "error" in rec:
            print(f"{rec['d']:3d}  {rec['method']:9s}  ERROR {rec['error']}")
            continue
        r2 = rec["r2_true_from_topr"]
        print(
            f"{rec['d']:3d}  {rec['method']:9s}  {rec['r_hat']:5}  "
            f"{str(rec['rank_ok']):>5}  {r2[0]:7.4f}  {r2[1]:7.4f}  "
            f"{rec['seconds']:7.1f}"
        )
    print(f"wrote {out_json}")
    ones = [r for r in rows if r.get("method") == "oneshot" and "error" not in r]
    three = [r for r in rows if r.get("method") == "threestep" and "error" not in r]
    if ones and three:
        print("\nread: three-step should tie or win on d=2. "
              "one-step should not collapse on d=3,4 unused coords.")


if __name__ == "__main__":
    main()
