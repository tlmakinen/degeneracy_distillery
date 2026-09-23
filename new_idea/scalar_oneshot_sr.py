#!/usr/bin/env python
"""One-step map + symbolic regression on the scalar Rosenbrock potential.

This is the SR validation for the case that breaks Fishnets-then-flatten.
For `scalar{3,4}` you observe only

    x ~ N(f(theta), sigma^2),   f(theta) = sum_i (theta_{i+1} - theta_i^2)^2 + (1 - theta_i)^2

so every coordinate enters and the likelihood Fisher has rank 1.  Three-step
reports an almost isotropic F and recovers nothing (R^2 = 0.21 / 0.035);
one-step reads r_hat = 1 and recovers f (R^2 ~ 0.998).  What was never checked
is whether SR can then write f down from eta.

Pipeline
--------
  1. probe fit at m = d, read the eta spectrum, pick r_hat
  2. K bootstrap members refit at m = r_hat  -> (eta, deta) ensemble
  3. sign/rotation alignment, SR grid, Operon SR
  4. score the MDL-selected expression against eta AND against the true f

`pow` is in `allowed_symbols` (dropping it costs every non-integer exponent),
with `square` alongside it so an integer-exponent form is also reachable.
Because `pow` invites the variable-in-exponent forms that
`forbid_x_in_pow_exponent` then deletes in place
(SR_PARETO_FILTER_INVESTIGATION.md), the Pareto front is analysed BOTH before
and after filtering, and both are scored against the known f.

    python new_idea/scalar_oneshot_sr.py --problems scalar3
    python new_idea/scalar_oneshot_sr.py --problems scalar3 scalar4 --full
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


def log(*a):
    print(*a, flush=True)


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # compare_rosen_ab
sys.path.insert(0, str(_HERE.parent))   # degeneracy_distillery (repo root, not installed)
from compare_rosen_ab import A_BOX, LOGW, PROBLEMS, poly_r2, spectrum  # noqa: E402


# ---------------------------------------------------------------------------
# One-step joint fit (faithful to compare_rosen_ab.run_oneshot, no screen step:
# the scalar problems are hard_cut=False so the active set is every coordinate)
# ---------------------------------------------------------------------------
def build_oneshot(th, x, problem, cfg):
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import optax

    jax.config.update("jax_enable_x64", True)

    d = th.shape[1]
    x_dim = x.shape[-1]
    mu_x = x.reshape(-1, x_dim).mean(0)
    sd_x = x.reshape(-1, x_dim).std(0) + 1e-12
    th_j = jnp.asarray(th)
    x_j = jnp.asarray((x - mu_x) / sd_x)
    xb = jnp.asarray((x.mean(1) - mu_x) / sd_x)   # replicate mean, for screening

    nval = max(40, int(0.15 * th.shape[0]))
    th_hold, x_hold = th_j[-nval:], x_j[-nval:]
    th_pool, x_pool = th_j[:-nval], x_j[:-nval]

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
        """Group-L1 probe: which theta coordinates the data actually depends on."""
        hidden: Sequence[int] = (64, 64)

        @nn.compact
        def __call__(self, theta):
            h = nn.gelu(nn.Dense(self.hidden[0], name="in_proj")(theta))
            for k, w in enumerate(self.hidden[1:]):
                h = nn.gelu(nn.Dense(w, name=f"h{k}")(h))
            return nn.Dense(x_dim, name="out")(h)

    def screen_inputs(steps: int, seed: int):
        mod = Screen()
        ps = mod.init(jr.PRNGKey(seed), th_j[0])
        tx = optax.adam(3e-3)
        st = tx.init(ps)
        n = th_j.shape[0]
        b = min(512, n)

        @jax.jit
        def one(ps, st, key):
            i = jr.randint(key, (b,), 0, n)

            def obj(ps):
                pr = jax.vmap(lambda t: mod.apply(ps, t))(th_j[i])
                k = ps["params"]["in_proj"]["kernel"]
                return (jnp.mean((pr - xb[i]) ** 2)
                        + 1e-2 * jnp.sum(jnp.sqrt(jnp.sum(k ** 2, 1) + 1e-12)))

            l, g = jax.value_and_grad(obj)(ps)
            u, st = tx.update(g, st, ps)
            return optax.apply_updates(ps, u), st, l

        key = jr.PRNGKey(seed + 1)
        for _ in range(steps):
            key, sk = jr.split(key)
            ps, st, _ = one(ps, st, sk)
        return np.sqrt((np.asarray(ps["params"]["in_proj"]["kernel"]) ** 2).sum(1))

    def fit_joint(m, steps, seed, boot=False, active=None):
        """Return (eta_fn, jac_fn, val_nll, scale).  boot=True resamples the fit set.

        `active` is the coordinate subset the log-det is taken over.  For the
        hard-cut problems that is the screened set (complement sits at the
        prior, contributing (d-m)*log w); for the rectangular problems it is
        every coordinate and there is no complement term.
        """
        key_boot = jr.PRNGKey(seed + 31337)
        if boot:
            idx_b = jr.randint(key_boot, (th_pool.shape[0],), 0, th_pool.shape[0])
            th_fit, x_fit = th_pool[idx_b], x_pool[idx_b]
        else:
            th_fit, x_fit = th_pool, x_pool
        n_fit = th_fit.shape[0]
        bp = min(256, n_fit)

        act_np = (np.arange(d) if active is None
                  else np.asarray(sorted(int(v) for v in active), dtype=np.int32))
        act = jnp.asarray(act_np)
        q0 = np.zeros((d, m))
        nset = min(m, len(act_np))
        q0[act_np[:nset], np.arange(nset)] = 1.0
        flat_mod = Flattener(m=m, skip_init=q0)
        est_mod = Estimator(m=m)
        k0, k1 = jr.split(jr.PRNGKey(seed), 2)
        p = {
            "est": est_mod.init(k1, x_fit[0]),
            "raw_log_scale": jnp.zeros(m),
            "flat": flat_mod.init(k0, th_fit[0]),
        }
        complement = (d - m) * LOGW if problem.hard_cut else 0.0
        const = 0.5 * m * np.log(2 * np.pi) + complement
        l1_w = 2.0 / n_fit

        def core(p, theta):
            g = flat_mod.apply(p["flat"], theta)
            return jnp.exp(2.0 * p["raw_log_scale"]) * g

        def logdet_active(p, theta):
            j = jax.jacfwd(lambda a: core(p, theta.at[act].set(a)))(theta[act])
            gram = j @ j.T if j.shape[0] <= j.shape[1] else j.T @ j
            return 0.5 * jnp.linalg.slogdet(gram + 1e-10 * jnp.eye(gram.shape[0]))[1]

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
                v = float(nll_mean(p, th_hold, x_hold))
                if np.isfinite(v) and v < best_val:
                    best_val, best = v, p
        p = best
        eta_fn = jax.jit(lambda t: jax.vmap(lambda u: core(p, u))(t))
        jac_fn = jax.jit(lambda t: jax.vmap(jax.jacfwd(lambda u: core(p, u)))(t))
        scale = np.asarray(np.exp(2.0 * np.asarray(p["raw_log_scale"])))
        return eta_fn, jac_fn, best_val, scale

    return fit_joint, screen_inputs


# ---------------------------------------------------------------------------
# Ensemble alignment.  m=1 has no rotation freedom, only a sign; m>1 goes
# through the same Procrustes path the three-step pipeline uses.
# ---------------------------------------------------------------------------
def align_members(ens: np.ndarray):
    """ens: (K, n, m) raw member outputs -> (K, n, m) centred + sign/rotation fixed."""
    K, n, m = ens.shape
    cen = ens - ens.mean(1, keepdims=True)
    if m == 1:
        ref = cen[0, :, 0]
        signs = np.array([
            1.0 if np.dot(cen[k, :, 0], ref) >= 0 else -1.0 for k in range(K)
        ])
        return cen * signs[:, None, None], signs.reshape(K, 1, 1) * np.ones((K, 1, 1))
    rots = []
    out = np.empty_like(cen)
    ref = cen[0]
    for k in range(K):
        u, _, vt = np.linalg.svd(cen[k].T @ ref)
        r = u @ vt
        rots.append(r)
        out[k] = cen[k] @ r
    return out, np.stack(rots)


def _predicate_worker(args):
    """Module-level so it is picklable by multiprocessing (a closure is not)."""
    n_params, forbid_self_trans, eq = args
    from degeneracy_distillery.sr_utils import sr_structure_predicate
    pred = sr_structure_predicate(
        n_params=n_params, forbid_self_transcendental=forbid_self_trans
    )
    return bool(pred(eq))


def timed_predicate(n_params: int, timeout_s: float, log_fn, forbid_self_trans=True):
    """`sr_structure_predicate` with a hard per-equation timeout.

    The predicate parses each equation with sympy, and on high-complexity rows
    that can take minutes to hours -- one row hung a 24h job (see notes at the
    top of this file).  SIGALRM does not help: sympy blocks inside C and the
    signal is only delivered once it returns to Python.  So each equation is
    evaluated in a worker process that can actually be killed.  A timeout
    counts as a reject, which is the conservative choice: an equation nobody
    can parse in `timeout_s` is not one to build a coordinate system on.
    """
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    state = {"timeouts": 0, "spent": 0.0}

    def predicate(eq: str) -> bool:
        t0 = time.time()
        pool = ctx.Pool(1)
        try:
            out = pool.apply_async(
                _predicate_worker, ((n_params, forbid_self_trans, str(eq)),)
            )
            return bool(out.get(timeout=timeout_s))
        except mp.TimeoutError:
            state["timeouts"] += 1
            log_fn(f"    predicate timed out after {timeout_s:.0f}s, rejecting: "
                   f"{str(eq)[:90]}...")
            return False
        except Exception as exc:                          # noqa: BLE001
            log_fn(f"    predicate raised {type(exc).__name__}, rejecting")
            return False
        finally:
            pool.terminate()
            pool.join()
            state["spent"] += time.time() - t0

    predicate.state = state                               # type: ignore[attr-defined]
    return predicate


def apply_rotations(arr: np.ndarray, rots: np.ndarray, jacobian: bool = False):
    """Apply the per-member alignment transform from `align_members`.

    arr       (K, n, m)      member outputs, already centred, or
              (K, n, m, d)   member Jacobians when jacobian=True
    rots      (K, 1, 1) signs for m=1, else (K, m, m) rotation matrices
    """
    K = arr.shape[0]
    m = arr.shape[2]
    if m == 1:
        signs = rots.reshape(K, 1, 1)
        return arr * (signs[..., None] if jacobian else signs)
    if jacobian:
        return np.stack([np.einsum("ij,nik->njk", rots[k], arr[k]) for k in range(K)])
    return np.stack([arr[k] @ rots[k] for k in range(K)])


def r2(a: np.ndarray, b: np.ndarray) -> float:
    """R^2 of a linear fit of `a` on `b` (b may be 1-D or 2-D)."""
    b = np.atleast_2d(b.T).T
    xf = np.column_stack([b, np.ones(len(a))])
    beta, *_ = np.linalg.lstsq(xf, a, rcond=None)
    return float(1.0 - ((a - xf @ beta) ** 2).mean() / (a.var() + 1e-12))


# ---------------------------------------------------------------------------
def run(problem, cfg, outdir: Path) -> dict:
    from degeneracy_distillery.sr_utils import (
        analyze_equations,
        filter_pareto_fronts,
        fit_symbolic_regression,
        fit_theta_scaler,
        get_y_sr,
    )

    t0 = time.time()
    d = problem.d
    rng = np.random.default_rng(cfg.seed + 1000 * (sum(map(ord, problem.name)) % 100))
    th, x = problem.simulate(cfg.nsims, rng)
    th_te, _ = problem.simulate(cfg.ntest, np.random.default_rng(7777 + d))
    f_true_te = problem.coord_fn(th_te)            # (ntest, 1) -- the target potential

    fit_joint, screen_inputs = build_oneshot(th, x, problem, cfg)

    # ---- 0. screen (hard-cut problems only).  The rectangular problems put
    # every coordinate in eta by construction, so there is nothing to screen.
    a_all = tuple(range(d))
    if problem.hard_cut:
        energy = screen_inputs(cfg.screen_steps, cfg.seed)
        order = [int(v) for v in np.argsort(-energy)]
        log(f"\n######## 0. screen ########")
        log(f"  input energy {np.round(energy, 4)}")
        log(f"  order (most used first) {order}")
    else:
        energy = np.full(d, np.nan)
        order = list(a_all)
        log("\n######## 0. screen: skipped (rectangular, all coordinates enter) ########")

    def active_for(m):
        if not problem.hard_cut:
            return a_all
        return tuple(sorted(order[:m]))

    # ---- 1. probe at m = d, read the spectrum, pick r_hat
    log(f"\n######## 1. probe fit at m = d = {d} ########")
    eta_fn, _, val_probe, _ = fit_joint(d, cfg.oneshot_steps, cfg.seed,
                                        active=active_for(d))
    _, lam_probe = spectrum(np.asarray(eta_fn(th_te)))
    nats = 0.5 * np.log(np.maximum(lam_probe, 1e-12))
    r_hat = max(1, min(int(np.sum(nats > cfg.nats_floor)), d))
    log(f"  probe lambda {np.round(lam_probe, 4)}")
    log(f"  nats         {np.round(nats, 4)}   floor {cfg.nats_floor}")
    log(f"  r_hat = {r_hat}   (true rank {problem.rank})   val NLL {val_probe:.4f}")

    # ---- 2. bootstrap ensemble refit at m = r_hat
    #
    # Every member is resampled, member 0 included.  Leaving member 0 on the
    # full fit set makes it systematically tighter than its peers, which biases
    # the ensemble spread -- and member 0 is also the alignment reference, so
    # the bias lands directly in y_std.
    #
    # eta is a map of theta alone, so the SR grid costs no simulator calls: draw
    # a fresh, denser theta grid and evaluate every member on it.  The test
    # points stay behind for analyze_equations.
    log(f"\n######## 2. {cfg.k_boot} bootstrap members at m = r_hat = {r_hat} ########")
    th_sr = np.random.default_rng(cfg.seed + 4242).uniform(
        -A_BOX, A_BOX, size=(cfg.sr_grid, d)
    )
    members, members_sr, jacs, scales = [], [], [], []
    for k in range(cfg.k_boot):
        e_fn, j_fn, v, s = fit_joint(
            r_hat, cfg.oneshot_steps, cfg.seed + 101 + k, boot=True,
            active=active_for(r_hat),
        )
        members.append(np.asarray(e_fn(th_te)))
        members_sr.append(np.asarray(e_fn(th_sr)))
        jacs.append(np.asarray(j_fn(th_te)))
        scales.append(s)
        log(f"  member {k}: val NLL {v:.4f}  scale {np.round(s, 4)}")
    ens = np.stack(members)                                  # (K, ntest, r_hat)
    ens_sr = np.stack(members_sr)                            # (K, sr_grid, r_hat)
    scales = np.stack(scales)
    log(f"  member scale spread (std/mean): "
        f"{np.round(scales.std(0) / (np.abs(scales.mean(0)) + 1e-12), 4)}")

    ens_al, rots = align_members(ens)
    eta = ens_al.mean(0)
    y_std_raw = ens_al.std(0, ddof=1)
    log(f"  aligned eta: median |y_std| {np.round(np.median(y_std_raw, 0), 4)}")

    # Same rotations carried to the SR grid (each member centred on its own grid
    # mean, exactly as on the test set).
    ens_sr_al = apply_rotations(ens_sr - ens_sr.mean(1, keepdims=True), rots)
    eta_sr = ens_sr_al.mean(0)
    y_std_sr_raw = ens_sr_al.std(0, ddof=1)

    # Leave-one-out calibration of the bootstrap spread (gw_oneshot convention).
    # z should look like t(K-2); a ratio above 1 means y_std is too small.
    K = cfg.k_boot
    mu = ens_al.mean(0)
    mu_loo = (K * mu[None] - ens_al) / (K - 1)
    sd_loo = np.sqrt(np.clip(
        (K * (ens_al.var(0, ddof=0)[None] + mu[None] ** 2) - ens_al ** 2) / (K - 1)
        - mu_loo ** 2, 1e-24, None))
    sd_loo *= np.sqrt((K - 1) / (K - 2))
    z_loo = (ens_al - mu_loo) / (sd_loo * np.sqrt(K / (K - 1)))
    zstd = z_loo.reshape(-1, r_hat).std(0)
    if K > 4:
        tref = np.sqrt((K - 2) / (K - 4))
        log(f"  LOO ensemble z std {np.round(zstd, 3)}  vs t_{K-2} reference "
            f"{tref:.4f}  ratio {np.round(zstd / tref, 3)}  (>1 = y_std too small)")
    else:
        tref = float("nan")
        log(f"  LOO ensemble z std {np.round(zstd, 3)}  (need K>4 for a t reference)")

    # Jacobian for analyze_equations: ensemble mean of the rotated member
    # Jacobians, not one member's, to match the eta that SR is fitting.
    jac_al = apply_rotations(np.stack(jacs), rots, jacobian=True)
    dy_sr = jac_al.mean(0)

    # recovery of the true potential, same cubic score as compare_rosen_ab
    r2_cubic = [poly_r2(f_true_te[:, j], eta) for j in range(f_true_te.shape[1])]
    r2_lin_eta_f = r2(eta[:, 0], f_true_te[:, 0])
    log(f"  cubic R^2(f_true | eta)      {np.round(r2_cubic, 4)}")
    log(f"  linear R^2(eta_0 | f_true)   {r2_lin_eta_f:.4f}   "
        "(close to 1 means eta is an affine image of f, as the volume term predicts)")

    # ---- 3. SR handoff
    log("\n######## 3. symbolic regression ########")
    scaler = fit_theta_scaler(th, feature_range=(-A_BOX, A_BOX))
    X = scaler.transform(th_te)
    X_sr = scaler.transform(th_sr)
    # One shared offset for both sets: an expression fitted on the grid has to
    # evaluate correctly on the test points, which it will not do if each set is
    # shifted by its own min.
    offset = eta_sr.min(0)
    y = eta - offset
    y_sr = eta_sr - offset
    y_std = np.maximum(y_std_raw, cfg.y_std_floor)
    y_std_sr = np.maximum(y_std_sr_raw, cfg.y_std_floor)
    frac_floored = (y_std_raw < cfg.y_std_floor).mean(0)
    log(f"  SR grid {X_sr.shape}, analyse on test {X.shape}")
    log(f"  fraction at y_std floor {cfg.y_std_floor}: {np.round(frac_floored, 3)}")

    sr_dir = outdir / f"sr_{problem.name}"
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

    # `pow` is in the symbol set, so the structure predicate matters here.
    # SR_PARETO_FILTER_INVESTIGATION.md measures it deleting 86-92% of the
    # front on kuramoto/qm7b -- including good expressions -- and
    # filter_pareto_fronts edits pareto.csv in place.  So analyse the front
    # BEFORE filtering as well, and report both.  This problem has a known
    # f(theta), so the two can be scored against it rather than argued about.
    Fs_id = np.repeat(np.eye(d)[None, ...], len(X), axis=0)
    analyse_kw = dict(
        parent_dir=str(sr_dir) + os.sep,
        n_params=d,
        equation_set="pareto",
        max_complexity_thresh=cfg.sr_max_length,
        length_penalty=2.0,
    )

    coord_sets: dict[str, list] = {}
    mdl_raw, frob_raw, _ = analyze_equations(
        X, y, y_std, dy_sr, Fs_id, equation_predicate=None, **analyse_kw
    )
    coord_sets["mdl_unfiltered"] = list(mdl_raw)
    coord_sets["frob_unfiltered"] = list(frob_raw)
    log(f"  MDL  (unfiltered front): {mdl_raw}")
    log(f"  Frob (unfiltered front): {frob_raw}")

    predicate = timed_predicate(d, cfg.predicate_timeout, log)
    summaries = filter_pareto_fronts(str(sr_dir), r_hat, predicate)
    log(f"  predicate: {predicate.state['timeouts']} timeouts, "
        f"{predicate.state['spent']:.1f}s total")
    kept = [int(s.get("kept", -1)) for s in summaries]
    original = [int(s.get("original", -1)) for s in summaries]
    removed = sum(int(s["removed"]) for s in summaries)
    log(f"  Pareto filter: removed {removed}, kept {kept} of {original}")

    if min(kept) > 0:
        try:
            mdl_coords, frob_coords, _ = analyze_equations(
                X, y, y_std, dy_sr, Fs_id, equation_predicate=predicate, **analyse_kw
            )
            coord_sets["mdl_filtered"] = list(mdl_coords)
            coord_sets["frob_filtered"] = list(frob_coords)
            log(f"  MDL  (filtered front):   {mdl_coords}")
            log(f"  Frob (filtered front):   {frob_coords}")
        except Exception as exc:                          # noqa: BLE001
            log(f"  filtered-front analysis failed: {exc!r}")
    else:
        log("  filter emptied a component's front; nothing left to analyse")

    # ---- 4. score the recovered expression against eta and against f
    log("\n######## 4. does SR write down f? ########")
    scores = {}
    for tag, coords in coord_sets.items():
        try:
            y_pred = np.asarray(get_y_sr(list(coords), X))
        except Exception as exc:                      # noqa: BLE001
            log(f"  {tag}: could not evaluate ({exc!r})")
            scores[tag] = {"error": repr(exc)}
            continue
        y_pred = y_pred.reshape(len(X), -1)
        # eta is defined only up to a rotation among the informative axes, so
        # matching SR component j to true coordinate j is the wrong test.  Score
        # each true coordinate against ALL r_hat expressions jointly -- the same
        # rotation-invariant form compare_rosen_ab.recovery() uses for eta.
        n_c = f_true_te.shape[1]
        cub = [poly_r2(f_true_te[:, k], y_pred) for k in range(n_c)]
        lin = [r2(f_true_te[:, k], y_pred) for k in range(n_c)]
        s = {
            "expr": list(coords),
            "r2_vs_eta": [r2(y[:, j], y_pred[:, j]) for j in range(r_hat)],
            "r2_true_from_sr_cubic": cub,
            "r2_true_from_sr_linear": lin,
            # the honest headline: the worst-recovered true coordinate
            "r2_f_from_expr_cubic": float(np.min(cub)),
            "r2_f_from_expr_linear": float(np.min(lin)),
            "r2_true_cubic_mean": float(np.mean(cub)),
        }
        scores[tag] = s
        log(f"  {tag}: R^2(eta | expr) {np.round(s['r2_vs_eta'], 4)}")
        log(f"      per true coord (cubic) {np.round(cub, 4)}")
        log(f"      WORST true coord: cubic {np.min(cub):.4f}  linear {np.min(lin):.4f}")

    np.savez(
        outdir / f"{problem.name}_oneshot_sr.npz",
        X=X, y=y, y_std=y_std, dy_sr=dy_sr, ens=ens_al,
        X_sr=X_sr, y_sr=y_sr, y_std_sr=y_std_sr, theta_sr=th_sr,
        loo_z_std=zstd, loo_t_ref=tref,
        f_true=f_true_te, theta_test=th_te,
        lam_probe=lam_probe, scales=scales,
    )

    return {
        "problem": problem.name,
        "d": d,
        "true_rank": problem.rank,
        "r_hat": r_hat,
        "rank_ok": r_hat == problem.rank,
        "probe_lambda": [float(v) for v in lam_probe],
        "r2_cubic_f_from_eta": [float(v) for v in r2_cubic],
        "r2_linear_eta_from_f": float(r2_lin_eta_f),
        "screen_energy": [float(v) for v in energy],
        "screen_order": order,
        "active": list(active_for(r_hat)),
        "set_ok": (set(active_for(r_hat)) == set(range(problem.d))
                   if (not problem.hard_cut or problem.rank == problem.d) else None),
        "predicate_timeouts": int(predicate.state["timeouts"]),
        "pareto_original": original,
        "pareto_kept": kept,
        "pareto_removed": removed,
        "loo_z_std": [float(v) for v in zstd],
        "loo_t_ref": float(tref),
        "sr": scores,
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--problems", nargs="+", default=["scalar3"], choices=list(PROBLEMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("new_idea/scalar_oneshot_sr_out"))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--sr-time-limit", type=int, default=None)
    # `pow` stays in: dropping it costs any non-integer exponent, and the CAMELS
    # run's one nonlinearity was (0.70*Omega_m + 0.31*IMFslope)^1.1, which
    # `square` cannot write.  `square` is kept alongside it so an integer-exponent
    # form is also reachable -- that route survives forbid_x_in_pow_exponent,
    # which is what culls 86-92% of the front on kuramoto/qm7b
    # (SR_PARETO_FILTER_INVESTIGATION.md).  Both fronts get analysed and scored.
    p.add_argument("--predicate-timeout", type=float, default=30.0,
                   help="seconds per equation in the Pareto structure filter "
                        "before it is rejected (0 = no limit)")
    p.add_argument("--allowed-symbols", type=str,
                   default="add,mul,div,pow,constant,variable,square,exp")
    return p.parse_args()


def budgets(args):
    if args.quick:
        return dict(nsims=400, ntest=200, oneshot_steps=200, screen_steps=200, k_boot=3,
                    sr_time_limit=args.sr_time_limit or 20, sr_grid=200)
    if args.full:
        return dict(nsims=8000, ntest=4000, oneshot_steps=4000, screen_steps=2000, k_boot=8,
                    sr_time_limit=args.sr_time_limit or 600, sr_grid=2000)
    return dict(nsims=2000, ntest=1000, oneshot_steps=1500, screen_steps=800, k_boot=6,
                sr_time_limit=args.sr_time_limit or 120, sr_grid=1000)


def main():
    args = parse_args()
    cfg = argparse.Namespace(
        seed=args.seed,
        nats_floor=0.10,
        y_std_floor=1e-3,
        sr_max_length=30,
        sr_max_depth=20,
        allowed_symbols=args.allowed_symbols,
        predicate_timeout=args.predicate_timeout,
        **budgets(args),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    log(f"problems={args.problems}  nsims={cfg.nsims}  ntest={cfg.ntest}  "
          f"K={cfg.k_boot}  sr_time_limit={cfg.sr_time_limit}s")
    log(f"allowed_symbols={cfg.allowed_symbols}")

    rows = []
    for name in args.problems:
        problem = PROBLEMS[name]
        log(f"\n================ {name}  d={problem.d}  true rank={problem.rank} "
              f"================")
        rows.append(run(problem, cfg, args.out))

    out_json = args.out / "scalar_oneshot_sr.json"
    out_json.write_text(json.dumps(rows, indent=2))
    log("\n================ summary ================")
    for r in rows:
        mdl = r["sr"].get("mdl_filtered") or r["sr"].get("mdl_unfiltered", {})
        eta_min = float(np.min(r['r2_cubic_f_from_eta']))
        # PASS needs BOTH: SR recovers every true coordinate, AND the rank is
        # right.  A too-large r_hat can still span the true coordinates, so the
        # R^2 alone will happily report success on a wrong-rank fit.
        sr_ok = mdl.get('r2_f_from_expr_cubic', 0.0) >= 0.99
        rank_ok = r['r_hat'] == r['true_rank']
        ok = "PASS" if (sr_ok and rank_ok) else ("FAIL-rank" if sr_ok else "FAIL-SR")
        log(f"{r['problem']:9s}  r_hat={r['r_hat']} (true {r['true_rank']}) "
              f"{'rank_ok' if r['r_hat']==r['true_rank'] else 'RANK_WRONG':10s}  "
              f"worst R2(true|eta)={eta_min:.4f}  "
              f"worst R2(true|SR)={mdl.get('r2_f_from_expr_cubic', float('nan')):.4f}  "
              f"{ok}  {r['seconds']:.0f}s")
        for e in mdl.get("expr", []):
            log(f"    {e}")
    log(f"wrote {out_json}")


if __name__ == "__main__":
    main()
