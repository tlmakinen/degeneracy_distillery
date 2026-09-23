#!/usr/bin/env python
"""Problem-agnostic one-step sweep driver.

Pipeline per trial:

    simulate -> group-L1 screen -> probe ladder -> K-member ensemble
    -> align -> SR -> MDL / flattening / frozen-NLL selection

Rosenbrock also exposes a three-step arm. That arm lives on the adapter.
The driver skips it when ``d > 16``.

    python scripts/oneshot_sweep.py --problem rosenbrock --dims 2 4 8 \\
        --arms oneshot threestep --num-trials 1 --quick
    python scripts/oneshot_sweep.py --problem rayleigh_benard \\
        --problem-arg n_params=3 --problem-arg mode=smoke --num-trials 1
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

METRIC_COLUMNS = [
    "problem", "arm", "d", "trial", "seed", "status", "failed_stage",
    "r_hat", "m_fit", "rank_correct", "jtj_rank_agreement_k", "screen_set_correct",
    "jtj_eigengap_rank", "jtj_eigengap_rank_screen", "median_y_std", "r2_true_min", "r2_sr_min",
    "nll_neural", "nll_symbolic", "nll_gap", "expression", "expression_mdl",
    "picks_agree", "complexity", "symbolic_recovered", "n_sims",
    "runtime_screen_s", "runtime_probe_s", "runtime_fit_s",
    "runtime_align_s", "runtime_sr_s", "runtime_frozen_s", "runtime_total_s",
    "peak_host_mem_mb",
]


def git_commit_hash() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def peak_host_mem_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(ru) / (1024.0 * 1024.0)
    return float(ru) / 1024.0


def coerce(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_problem_args(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        key, _, val = item.partition("=")
        if not key:
            continue
        out[key] = coerce(val)
    return out


def data_summary(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim > 2:
        axes = tuple(range(1, x.ndim - 1))
        return x.mean(axis=axes)
    return x


def screen_inputs(theta, xb, steps: int, seed: int, lam: float = 1e-2) -> np.ndarray:
    """Group-L1 column screen of a data summary on theta."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import flax.linen as nn
    import optax

    x_dim = int(np.asarray(xb).shape[-1])

    class Screen(nn.Module):
        hidden: tuple[int, ...] = (64, 64)

        @nn.compact
        def __call__(self, theta):
            h = nn.gelu(nn.Dense(self.hidden[0], name="in_proj")(theta))
            for k, w in enumerate(self.hidden[1:]):
                h = nn.gelu(nn.Dense(w, name=f"h{k}")(h))
            return nn.Dense(x_dim, name="out")(h)

    th = jnp.asarray(theta)
    xb_j = jnp.asarray(xb)
    mod = Screen()
    ps = mod.init(jr.PRNGKey(int(seed)), th[0])
    tx = optax.adam(3e-3)
    st = tx.init(ps)
    n = int(th.shape[0])
    b = min(512, n)

    @jax.jit
    def one(ps, st, key):
        i = jr.randint(key, (b,), 0, n)

        def obj(ps):
            pr = jax.vmap(lambda t: mod.apply(ps, t))(th[i])
            K = ps["params"]["in_proj"]["kernel"]
            return (
                jnp.mean((pr - xb_j[i]) ** 2)
                + lam * jnp.sum(jnp.sqrt(jnp.sum(K ** 2, 1) + 1e-12))
            )

        l, g = jax.value_and_grad(obj)(ps)
        u, st = tx.update(g, st, ps)
        return optax.apply_updates(ps, u), st, l

    key = jr.PRNGKey(int(seed) + 1)
    for _ in range(int(steps)):
        key, sk = jr.split(key)
        ps, st, l = one(ps, st, sk)
    K = np.asarray(ps["params"]["in_proj"]["kernel"])
    return np.sqrt((K ** 2).sum(1))


def make_fisher(J: np.ndarray, mode: str = "jtj", ridge: float = 1e-4) -> np.ndarray:
    J = np.asarray(J, dtype=np.float64)
    d = J.shape[-1]
    F = np.einsum("nmi,nmj->nij", J, J)
    if mode == "ridge":
        F = F + ridge * np.eye(d)
    elif mode == "identity":
        F = np.broadcast_to(np.eye(d), (J.shape[0], d, d)).copy()
    elif mode != "jtj":
        raise ValueError(f"unknown fisher mode {mode!r}")
    return F


def floor_y_std(y_std: np.ndarray, y: np.ndarray, floor_frac: float = 1e-3) -> np.ndarray:
    y_std = np.asarray(y_std, dtype=np.float64).copy()
    scale = np.median(np.abs(y), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return np.maximum(y_std, floor_frac * scale)


def compile_jax_coord(expr: str, n_params: int) -> Callable:
    import jax.numpy as jnp
    import sympy
    from degeneracy_distillery.sr_utils import _parse_sr_equation, safe_lambdify

    expr_s, _, _, pars = _parse_sr_equation(expr)
    xs = list(sympy.symbols(" ".join(f"X{i + 1}" for i in range(n_params)), real=True))
    bs = list(sympy.symbols([f"b{i}" for i in range(len(pars))], real=True))
    fn = safe_lambdify(bs + xs, expr_s, ["jax"])

    def coord_fn(t):
        t = jnp.asarray(t)
        val = fn(*pars, *t)
        return jnp.atleast_1d(jnp.asarray(val)).reshape(-1)

    return coord_fn


def compile_jax_stack(exprs: list[str], n_params: int) -> Callable:
    import jax.numpy as jnp

    fns = [compile_jax_coord(e, n_params) for e in exprs]

    def coord_fn(t):
        parts = [jnp.atleast_1d(jnp.asarray(fn(t))).reshape(-1) for fn in fns]
        return jnp.concatenate(parts)

    return coord_fn


def poly_r2(y: np.ndarray, feats: np.ndarray, deg: int = 3) -> float:
    y = np.asarray(y, dtype=np.float64).ravel()
    f = np.asarray(feats, dtype=np.float64)
    if f.ndim == 1:
        f = f[:, None]
    if y.size < 3 or f.shape[0] < 3:
        return float("nan")
    f = (f - f.mean(0)) / (f.std(0) + 1e-12)
    k = f.shape[1]
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


def r2_true_min(truth: np.ndarray | None, eta: np.ndarray) -> float:
    if truth is None:
        return float("nan")
    truth = np.asarray(truth)
    scores = [poly_r2(truth[:, k], eta) for k in range(truth.shape[1])]
    return float(np.min(scores)) if scores else float("nan")


def augment_aligned(fits, aligned, weights, X_sr, fisher_mode, ridge):
    from degeneracy_distillery.align_coords import apply_ensemble_alignment
    from degeneracy_distillery.preprocessing_utils import weighted_std
    import jax.numpy as jnp

    w = np.asarray(weights, dtype=np.float64)
    w = w / np.maximum(w.sum(), 1e-12)
    etas = np.stack([np.asarray(f.eta(X_sr), dtype=np.float64) for f in fits], 0)
    jacs = np.stack([np.asarray(f.jac(X_sr), dtype=np.float64) for f in fits], 0)
    ys, js = apply_ensemble_alignment(etas, jacs, aligned)
    y = np.average(ys, 0, weights=w)
    y_std = np.asarray(weighted_std(jnp.asarray(ys), weights=jnp.asarray(w), axis=0))
    dy = np.average(js, 0, weights=w)
    return y, y_std, dy, make_fisher(dy, fisher_mode, ridge)


def rescore_frozen(
    candidates: list[tuple[list[str], float, float]],
    th_fit, x_fit, th_val, x_val,
    lo, hi, steps, seed, neural_nll,
    use_log_features, estimator_factory,
) -> dict[str, Any]:
    from degeneracy_distillery.oneshot import train_oneshot

    rows = []
    best_nll = np.inf
    nll_pick = candidates[0][0]
    nll_pick_cx = candidates[0][2]
    d = int(np.asarray(th_fit).shape[1])
    for i, (exprs, dl, cx) in enumerate(candidates):
        print(f"[frozen] {i + 1}/{len(candidates)}  cx={cx:.1f}  {exprs}", flush=True)
        try:
            coord_fn = compile_jax_stack(list(exprs), d)
            m = int(np.asarray(coord_fn(np.asarray(th_fit)[0])).reshape(-1).size)
            fr = train_oneshot(
                th_fit, x_fit, th_val, x_val,
                m=m, m0=m, lo=lo, hi=hi,
                steps=int(steps), seed=int(seed) + 100 + i,
                coord_fn=coord_fn, verbose=False,
                use_log_features=use_log_features,
                estimator_factory=estimator_factory,
            )
            nll = float(np.mean(fr.nll_vec(th_val, x_val)))
        except Exception as exc:
            print(f"[frozen] failed: {exc}", flush=True)
            nll = float("nan")
        rows.append({"eq": exprs, "dl": dl, "complexity": cx, "nll": nll})
        if np.isfinite(nll) and (
            nll < best_nll - 1e-12
            or (abs(nll - best_nll) <= 1e-12 and cx < nll_pick_cx)
        ):
            best_nll = nll
            nll_pick = exprs
            nll_pick_cx = cx
    mdl_pick = candidates[0][0]
    return {
        "frozen_rows": rows,
        "nll_pick": nll_pick,
        "nll_pick_complexity": nll_pick_cx,
        "nll_symbolic": float(best_nll),
        "nll_neural": float(neural_nll),
        "nll_gap": float(best_nll - neural_nll) if np.isfinite(best_nll) else float("nan"),
        "picks_agree": list(nll_pick) == list(mdl_pick),
        "mdl_pick": mdl_pick,
    }


def standardise_x(x_tr, x_va):
    x_tr = np.asarray(x_tr)
    x_va = np.asarray(x_va)
    if x_tr.ndim > 2:
        mu = x_tr.reshape(-1, x_tr.shape[-1]).mean(0)
        sd = x_tr.reshape(-1, x_tr.shape[-1]).std(0) + 1e-12
        shape = (1,) * (x_tr.ndim - 1) + (x_tr.shape[-1],)
        mu, sd = mu.reshape(shape), sd.reshape(shape)
    else:
        mu = x_tr.mean(0)
        sd = x_tr.std(0) + 1e-12
    return (x_tr - mu) / sd, (x_va - mu) / sd, mu, sd


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--problem", required=True)
    p.add_argument("--problem-arg", action="append", default=[],
                   help="Adapter kwargs as key=value. Repeatable.")
    p.add_argument("--dims", nargs="+", type=int, default=None)
    p.add_argument("--arms", nargs="+", default=["oneshot"],
                   choices=["oneshot", "threestep"])
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--trial-start", type=int, default=0)
    p.add_argument("--nsims", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("oneshot_sweep"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--nval-frac", type=float, default=0.15)

    p.add_argument("--m-probe", type=int, default=None)
    p.add_argument("--probe-steps", type=int, default=None)
    p.add_argument("--rung-steps", type=int, default=None)
    p.add_argument("--ladder-k", type=float, default=2.0)
    p.add_argument("--rank-rule", choices=("info", "nll"), default="info",
                   help="info: count probe axes above --info-floor nats of held-out "
                        "information. nll: the NLL drop test, which depends on the "
                        "units of theta.")
    p.add_argument("--info-floor", type=float, default=1.0)
    p.add_argument("--screen-steps", type=int, default=None)
    p.add_argument("--oneshot-batch", type=int, default=512)
    p.add_argument("--oneshot-lr", type=float, default=1e-3)
    p.add_argument("--rank-min-gap", type=float, default=10.0)
    p.add_argument("--ensemble-k", type=int, default=4)
    p.add_argument("--ensemble-steps", type=int, default=None)
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--align-subsample", type=int, default=None)

    p.add_argument("--sr-time-limit", type=int, default=300)
    p.add_argument("--sr-n-aug", type=int, default=None)
    p.add_argument("--sr-n-aug-per-dim", type=int, default=1000)
    p.add_argument("--sr-max-length", type=int, default=None)
    p.add_argument("--sr-max-depth", type=int, default=None)
    p.add_argument("--sr-fisher", choices=("jtj", "ridge", "identity"), default="jtj")
    p.add_argument("--sr-fisher-ridge", type=float, default=1e-4)
    p.add_argument("--n-frozen-candidates", type=int, default=8)
    p.add_argument("--frozen-steps", type=int, default=None)
    p.add_argument("--skip-sr", action="store_true")

    p.add_argument("--num-fishnets", type=int, default=3)
    p.add_argument("--fish-epochs", type=int, default=None)
    p.add_argument("--flatten-epochs", type=int, default=None)
    p.add_argument("--keep-workdirs", action="store_true")
    return p.parse_args()


def apply_budgets(args: argparse.Namespace) -> argparse.Namespace:
    is_rb = str(args.problem) in {"rayleigh_benard", "rb"}
    if args.quick:
        args.nsims = args.nsims or (40 if is_rb else 400)
        args.n_test = args.n_test or (40 if is_rb else 200)
        args.probe_steps = args.probe_steps or 200
        args.rung_steps = args.rung_steps or 100
        args.screen_steps = args.screen_steps or 200
        args.ensemble_steps = args.ensemble_steps or 150
        args.frozen_steps = args.frozen_steps or 80
        args.sr_time_limit = min(args.sr_time_limit, 30)
        args.fish_epochs = args.fish_epochs or 80
        args.flatten_epochs = args.flatten_epochs or 80
        args.ensemble_k = min(args.ensemble_k, 2)
    else:
        # RB DNS cost is per draw. Default to the notebook smoke/full nsims
        # only when the caller did not set --nsims.
        if args.nsims is None:
            args.nsims = 100 if is_rb else 2000
        if args.n_test is None:
            args.n_test = 100 if is_rb else 1000
        args.probe_steps = args.probe_steps or 20000
        args.rung_steps = args.rung_steps or 5000
        args.screen_steps = args.screen_steps or 2000
        args.ensemble_steps = args.ensemble_steps or 5000
        args.frozen_steps = args.frozen_steps or 3000
        args.fish_epochs = args.fish_epochs or 250
        args.flatten_epochs = args.flatten_epochs or 250
    return args


def count_params(params) -> int:
    import jax
    return int(sum(np.asarray(z).size for z in jax.tree_util.tree_leaves(params)))


def run_oneshot_trial(problem, th, x, th_te, x_te, args, seed, workdir) -> dict:
    from degeneracy_distillery.oneshot import jtj_eigengap, whittle_ladder
    from degeneracy_distillery.oneshot_ensemble import fit_oneshot_ensemble

    runtimes: dict[str, float] = {}
    row: dict[str, Any] = {}
    n = int(th.shape[0])
    nval = max(20, int(args.nval_frac * n))
    x_tr, x_va, _, _ = standardise_x(x, x_te)
    th_fit, x_fit = th[:-nval], x_tr[:-nval]
    th_hold, x_hold = th[-nval:], x_tr[-nval:]
    lo = np.asarray(problem.theta_lo, dtype=float)
    hi = np.asarray(problem.theta_hi, dtype=float)
    if lo.size == 1:
        lo, hi = float(lo), float(hi)

    t0 = time.time()
    xb = data_summary(x_tr)
    scores = screen_inputs(th, xb, steps=int(args.screen_steps), seed=seed)
    runtimes["screen"] = time.time() - t0
    order = [int(v) for v in np.argsort(-scores)]
    row["screen_order"] = order
    row["screen_scores"] = [float(v) for v in scores]
    print(f"[screen] scores {np.round(scores, 4)}", flush=True)

    t0 = time.time()
    lad = whittle_ladder(
        th_fit, x_fit, th_hold, x_hold,
        lo=lo, hi=hi,
        m_probe=int(args.m_probe or problem.m_probe),
        probe_steps=int(args.probe_steps),
        rung_steps=int(args.rung_steps),
        seed=seed,
        ladder_k=args.ladder_k,
        rank_rule=args.rank_rule,
        info_floor=args.info_floor,
        batch=args.oneshot_batch,
        lr=args.oneshot_lr,
        min_gap=args.rank_min_gap,
        use_log_features=problem.use_log_features,
        estimator_factory=problem.estimator_factory,
    )
    runtimes["probe"] = time.time() - t0
    fit = lad["fit"]
    r_hat = int(lad["r_hat"])
    m_fit = int(fit.m)
    row["r_hat"] = r_hat
    row["m_fit"] = m_fit
    row["rank_rule"] = str(lad["rank_rule"])
    row["info_probe_nats"] = ";".join(f"{v:.3f}" for v in lad["info_probe"])
    row["info_final_nats"] = ";".join(f"{v:.3f}" for v in lad["info_final"])
    row["jtj_eigengap_rank"] = int(lad["jtj_gap"]["rank"])
    row["jtj_eigengap_rank_screen"] = int(lad["jtj_gap_screen"]["rank"])
    row["nll_neural"] = float(np.mean(fit.nll_vec(th_hold, x_hold)))
    row["n_outputs"] = m_fit
    row["n_params_count"] = count_params(fit.params)
    lam = np.asarray(lad["rungs"][0]["lam"], dtype=np.float64)
    if r_hat < lam.size:
        row["rank_margin"] = float(
            0.5 * np.log(max(lam[r_hat - 1], 1e-12))
            - 0.5 * np.log(max(lam[r_hat], 1e-12))
        )
    else:
        row["rank_margin"] = float("inf")
    row["sampling_band"] = float(np.sqrt(max(r_hat, 1) / max(nval, 1)))

    ens = fit_oneshot_ensemble(
        th_fit, x_fit, m_fit, int(args.ensemble_k),
        lo=lo, hi=hi, th_te=th_te, x_te=x_va,
        steps=int(args.ensemble_steps), seed=seed,
        use_log_features=problem.use_log_features,
        estimator_factory=problem.estimator_factory,
        bootstrap=not args.no_bootstrap,
        batch=args.oneshot_batch, lr=args.oneshot_lr,
        min_gap=args.rank_min_gap, r_hat=r_hat,
        align_subsample=args.align_subsample,
        init_fit=fit,
    )
    runtimes["fit"] = float(ens.get("runtime_fit_s", 0.0))
    runtimes["align"] = float(ens.get("runtime_align_s", 0.0))
    row["jtj_rank_agreement_k"] = int(ens["jtj_rank_agreement_k"])
    y = np.asarray(ens["y"], dtype=np.float64)
    y_std = floor_y_std(ens["y_std"], y)
    dy = np.asarray(ens["dy_sr"], dtype=np.float64)
    X = np.asarray(ens["X"], dtype=np.float64)
    Fs = np.asarray(ens["Fs"], dtype=np.float64)
    row["median_y_std"] = float(np.median(y_std))
    eta_te = np.asarray(fit.eta(th_te))
    J_te = np.asarray(fit.jac(th_te))
    truth = problem.truth_coords(th_te)
    row["r2_true_min"] = r2_true_min(truth, eta_te)

    if args.skip_sr:
        row["expression"] = None
        row["expression_mdl"] = None
        row["picks_agree"] = False
        row["r2_sr_min"] = float("nan")
        row["nll_symbolic"] = float("nan")
        row["nll_gap"] = float("nan")
        frozen_pick: list[str] | None = None
    else:
        hints = problem.sr_hints()
        n_aug = int(args.sr_n_aug if args.sr_n_aug else args.sr_n_aug_per_dim * problem.d)
        max_length = int(args.sr_max_length or hints.get("max_length") or max(25, 4 * problem.d + 8))
        max_depth = int(args.sr_max_depth or hints.get("max_depth") or max(10, problem.d + 4))
        rng_sr = np.random.default_rng(seed + 17)
        X_sr = problem.sample_prior(n_aug, rng_sr)
        t0 = time.time()
        y_sr, y_std_sr, dy_sr, Fs_sr = augment_aligned(
            ens["fits"], ens["aligned"], ens["weights"],
            X_sr, args.sr_fisher, args.sr_fisher_ridge,
        )
        y_std_sr = floor_y_std(y_std_sr, y_sr)
        X = np.concatenate([X, X_sr], 0)
        y = np.concatenate([y, y_sr], 0)
        y_std = np.concatenate([y_std, y_std_sr], 0)
        dy = np.concatenate([dy, dy_sr], 0)
        Fs = np.concatenate([Fs, Fs_sr], 0)
        from degeneracy_distillery.sr_utils import (
            fit_and_analyze_sr, sr_structure_predicate,
        )
        sr_dir = workdir / "sr"
        sr_dir.mkdir(parents=True, exist_ok=True)
        print(f"[sr] n={X.shape[0]}  m={y.shape[1]}  max_length={max_length}", flush=True)
        mdl_coords, frob_coords, analysis, _ = fit_and_analyze_sr(
            X, y, y_std, dy, Fs,
            n_params=problem.d,
            components_to_fit=list(range(int(y.shape[1]))),
            slice_fisher=False,
            parent_dir=str(sr_dir) + os.sep,
            test_size=0.5,
            random_state=32134 + seed,
            shuffle=True,
            time_limit=args.sr_time_limit,
            max_length=max_length,
            max_depth=max_depth,
            allowed_symbols=hints.get(
                "allowed_symbols",
                "add,mul,div,pow,constant,variable,sqrt",
            ),
            max_complexity_thresh=max(20, max_length),
            equation_set="pareto",
            length_penalty=2.0,
            equation_predicate=sr_structure_predicate(
                n_params=problem.d,
                forbid_self_transcendental=True,
                check_nested_exp=False,
                forbid_x_in_pow_exponent=True,
            ),
        )
        runtimes["sr"] = time.time() - t0
        n_comp = int(y.shape[1])
        eqs = [[str(e) for e in analysis["equations"][c]] for c in range(n_comp)]
        dls = [np.asarray(analysis["DL"][c], dtype=np.float64) for c in range(n_comp)]
        cxs = [np.asarray(analysis["complexity"][c], dtype=np.float64) for c in range(n_comp)]
        candidates: list[tuple[list[str], float, float]] = []
        mdl = [str(e) for e in mdl_coords]
        frob = [str(e) for e in frob_coords]
        candidates.append((mdl, float(sum(float(d[0]) if d.size else np.nan for d in dls)), float(sum(float(c[0]) if c.size else np.nan for c in cxs))))
        if frob != mdl:
            candidates.append((frob, float("nan"), float("nan")))
        max_len = min(len(e) for e in eqs) if eqs else 0
        for j in range(1, max_len):
            stack = [eqs[c][j] for c in range(n_comp)]
            dl = float(sum(float(dls[c][j]) for c in range(n_comp)))
            cx = float(sum(float(cxs[c][j]) for c in range(n_comp)))
            if any(s == stack for s, _, _ in candidates):
                continue
            candidates.append((stack, dl, cx))
            if len(candidates) >= int(args.n_frozen_candidates):
                break
        t0 = time.time()
        frozen = rescore_frozen(
            candidates, th_fit, x_fit, th_hold, x_hold,
            lo, hi, args.frozen_steps, seed, row["nll_neural"],
            problem.use_log_features, problem.estimator_factory,
        )
        runtimes["frozen"] = time.time() - t0
        frozen_pick = list(frozen["nll_pick"])
        row["expression"] = " | ".join(frozen_pick)
        row["expression_mdl"] = " | ".join(list(frozen["mdl_pick"]))
        row["picks_agree"] = bool(frozen["picks_agree"])
        row["complexity"] = float(frozen["nll_pick_complexity"])
        row["nll_symbolic"] = frozen["nll_symbolic"]
        row["nll_gap"] = frozen["nll_gap"]
        try:
            sr_fn = compile_jax_stack(frozen_pick, problem.d)
            eta_sr = np.asarray(
                [sr_fn(th_te[i]) for i in range(len(th_te))], dtype=np.float64,
            )
            if eta_sr.ndim == 1:
                eta_sr = eta_sr[:, None]
            row["r2_sr_min"] = r2_true_min(truth, eta_sr)
        except Exception as exc:
            print(f"[sr score] failed: {exc}", flush=True)
            row["r2_sr_min"] = float("nan")

    payload = {
        "r_hat": r_hat,
        "screen_order": order,
        "screen_scores": scores,
        "eta": eta_te,
        "J": J_te,
        "theta": th_te,
        "aux": getattr(problem, "last_aux", {}),
        "expression": row.get("expression"),
    }
    gates = problem.gates(payload)
    row.update(gates)
    if "rank_correct" not in row and problem.expected_rank is not None:
        row["rank_correct"] = bool(r_hat == int(problem.expected_rank))
    if "screen_set_correct" not in row:
        row["screen_set_correct"] = bool(gates.get("screen_set_correct", False))
    if truth is not None and np.isfinite(row.get("r2_sr_min", np.nan)):
        row["symbolic_recovered"] = bool(row["r2_sr_min"] >= 0.95)
    else:
        row["symbolic_recovered"] = bool(gates.get("exponents_recovered", False))
    row["_runtimes"] = runtimes
    row["_eta"] = eta_te
    row["_J"] = J_te
    return row


def run_threestep_trial(problem, th, x, th_te, x_te, args, seed, workdir) -> dict:
    if not hasattr(problem, "threestep_arm"):
        return {"status": "skipped_by_budget"}
    if int(problem.d) > 16:
        return {"status": "skipped_by_budget"}
    rec = problem.threestep_arm(
        th, x, th_te, x_te,
        seed=seed, outdir=workdir,
        num_fishnets=int(args.num_fishnets),
        fish_epochs=int(args.fish_epochs),
        flatten_epochs=int(args.flatten_epochs),
    )
    rec.setdefault("jtj_rank_agreement_k", float("nan"))
    rec.setdefault("median_y_std", float("nan"))
    rec.setdefault("r2_sr_min", float("nan"))
    rec.setdefault("nll_neural", float("nan"))
    rec.setdefault("nll_symbolic", float("nan"))
    rec.setdefault("nll_gap", float("nan"))
    rec.setdefault("expression", None)
    rec.setdefault("expression_mdl", None)
    rec.setdefault("picks_agree", False)
    rec.setdefault("symbolic_recovered", False)
    rec.setdefault("screen_set_correct", False)
    return rec


def main() -> None:
    args = apply_budgets(parse_args())
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    problem_kwargs = parse_problem_args(args.problem_arg)

    import jax
    from degeneracy_distillery.problems import get_problem

    if args.dims:
        dims = [int(v) for v in args.dims]
    else:
        probe = get_problem(args.problem, seed=0, **problem_kwargs)
        dims = [int(probe.d)]

    metrics_path = out_dir / "metrics.csv"
    rows: list[dict[str, Any]] = []
    done: set[tuple[str, str, int, int]] = set()
    if args.resume and metrics_path.exists():
        prev = pd.read_csv(metrics_path)
        rows = prev.to_dict("records")
        done = {
            (str(r.get("problem")), str(r.get("arm")), int(r["d"]), int(r["trial"]))
            for r in rows
            if str(r.get("status")) in {"ok", "skipped_by_budget"}
        }
        print(f"resuming: {len(done)} completed rows", flush=True)

    manifest = {
        "script": "oneshot_sweep.py",
        "problem": args.problem,
        "problem_args": problem_kwargs,
        "arms": list(args.arms),
        "dims": dims,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "git_commit": git_commit_hash(),
    }
    write_json(out_dir / "config_manifest.json", manifest)
    write_json(out_dir / "manifest.json", manifest)

    for arm in args.arms:
        for d in dims:
            for trial in range(args.trial_start, args.trial_start + args.num_trials):
                seed = int(args.seed + 7919 * trial + 31 * d)
                kw = dict(problem_kwargs)
                if "n_params" not in kw:
                    kw["d"] = d
                if args.nsims is not None and "nsims" not in kw:
                    kw.setdefault("nsims", args.nsims)
                problem = get_problem(args.problem, seed=seed, **kw)
                d_eff = int(problem.d)
                key = (problem.name, arm, d_eff, trial)
                if key in done:
                    continue
                tag = f"{problem.name}_{arm}_d{d_eff}_trial{trial}"
                print(f"\n=== {tag} (seed {seed}) ===", flush=True)
                row: dict[str, Any] = {
                    "problem": problem.name, "arm": arm, "d": d_eff,
                    "trial": int(trial), "seed": seed, "status": "ok",
                    "failed_stage": None, "n_sims": int(args.nsims),
                }
                workdir = out_dir / "runs" / tag
                workdir.mkdir(parents=True, exist_ok=True)
                t_all = time.time()
                stage = "sim"
                runtimes: dict[str, float] = {}
                try:
                    rng = np.random.default_rng(seed)
                    n_test = int(args.n_test)
                    th, x = problem.sample(int(args.nsims), rng)
                    aux_tr = dict(getattr(problem, "last_aux", {}) or {})
                    th_te, x_te = problem.sample(n_test, rng)
                    aux_te = dict(getattr(problem, "last_aux", {}) or {})
                    if aux_te:
                        problem.last_aux = aux_te

                    if arm == "threestep":
                        stage = "threestep"
                        t0 = time.time()
                        rec = run_threestep_trial(
                            problem, th, x, th_te, x_te, args, seed, workdir,
                        )
                        runtimes["fit"] = time.time() - t0
                        row.update(rec)
                    else:
                        stage = "oneshot"
                        rec = run_oneshot_trial(
                            problem, th, x, th_te, x_te, args, seed, workdir,
                        )
                        runtimes.update(rec.pop("_runtimes", {}))
                        rec.pop("_eta", None)
                        rec.pop("_J", None)
                        row.update(rec)
                    if row.get("status") is None:
                        row["status"] = "ok"
                except Exception as exc:
                    row["status"] = "failed"
                    row["failed_stage"] = stage
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"!!! {tag} failed at {stage}: {exc}", flush=True)
                    traceback.print_exc()
                    write_json(workdir / "run_record.json", {
                        "run_id": tag,
                        "problem": problem.name,
                        "master_seed": seed,
                        "status": "failed",
                        "failure_stage": stage,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "failure_traceback": traceback.format_exc(),
                    })

                for k, v in runtimes.items():
                    row[f"runtime_{k}_s"] = float(v)
                row["runtime_total_s"] = float(time.time() - t_all)
                row["peak_host_mem_mb"] = peak_host_mem_mb()
                if row.get("status") == "ok":
                    discovery_ok = bool(row.get("gate_ok", row.get("rank_correct", False)))
                    write_json(workdir / "run_record.json", {
                        "run_id": tag,
                        "problem": problem.name,
                        "master_seed": seed,
                        "status": "success",
                        "counts": {
                            "n_train_simulations": int(args.nsims),
                            "n_eval_simulations": int(args.n_test),
                        },
                        "discovery": {
                            "success": discovery_ok,
                            "r_hat": row.get("r_hat"),
                            "expression": row.get("expression"),
                            "physics_alignment": row.get(
                                "best_nusselt_abs_corr", row.get("r2_true_min"),
                            ),
                            "mdl_total": row.get("nll_symbolic"),
                            "complexity_total": row.get("complexity"),
                        },
                        "runtime_seconds": {
                            **{k: float(v) for k, v in runtimes.items()},
                            "total": row["runtime_total_s"],
                        },
                    })
                    # also at out_dir root when this is a one-seed RB task
                    write_json(out_dir / "run_record.json", {
                        "run_id": tag,
                        "problem": problem.name,
                        "master_seed": seed,
                        "status": "success",
                        "counts": {
                            "n_train_simulations": int(args.nsims),
                            "n_eval_simulations": int(args.n_test),
                        },
                        "discovery": {
                            "success": discovery_ok,
                            "r_hat": row.get("r_hat"),
                            "expression": row.get("expression"),
                            "physics_alignment": row.get(
                                "best_nusselt_abs_corr", row.get("r2_true_min"),
                            ),
                            "mdl_total": row.get("nll_symbolic"),
                            "complexity_total": row.get("complexity"),
                        },
                        "runtime_seconds": {"total": row["runtime_total_s"]},
                    })
                rows.append(row)
                pd.DataFrame(rows).to_csv(metrics_path, index=False)
                if not args.keep_workdirs and workdir.exists() and row.get("status") != "failed":
                    import shutil
                    shutil.rmtree(workdir, ignore_errors=True)

    print(f"\nwrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
