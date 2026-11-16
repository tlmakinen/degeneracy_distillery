#!/usr/bin/env python
"""Symbolic regression with Operon's native description length on the CAMELS
aligned ensembles (the Colab run of the top 13 parameters).

Needs the patched pyoperon of branch ``operon-native-mdl``:

    <worktree>/third_party/operon_mdl/venv/bin/python scripts/camels_sr_native_mdl.py \\
        --runs results/whittle_camels/colab_top13 --out-dir results/whittle_camels/sr_native

Targets. The ensemble files hold the aligned coordinates on the 205 test
simulations only, in the frame of one member, which is an arbitrary rotation.
The six informative directions span the same parameter subspace in every run,
so the targets are defined once for all runs: the varimax basis of that shared
subspace. In each run the six leading principal axes of the ensemble mean are
rotated so that their linear loadings match that basis. Every member is rotated
the same way, and ``y_std`` is the member spread of the rotated coordinate.

Search. One search per (split, coordinate, mode). The pick is the smallest
description length on the front, computed by Operon on its own tree.

  * Default: 155 of the 205 test points are searched, the other 50 held out.
  * ``--augmented empirical|uniform``: the search runs on fresh theta pushed
    through the fitted ensemble (``augmented_<draw>.npz`` in each run folder,
    written by ``oneshot_sweep.py --save-aug``). The augmented member
    coordinates get the rotation found on the test set. ``--n-train`` of them
    are searched, ``--n-holdout-aug`` held out, and every pick is also scored
    on the 205 real test simulations, which the search never sees.

Each pick is also evaluated on the other two splits (their test simulations,
and their augmented hold-out when there is one) against their own targets.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

MODES = {
    "dl": ["distillery_mdl", "length"],
    "r2+dl": ["r2", "distillery_mdl", "length"],
}
N_LEAD = 6


def varimax(L, it=500, tol=1e-10):
    p, q = L.shape
    R = np.eye(q)
    d = 0.0
    for _ in range(it):
        Lr = L @ R
        u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(0)) / p))
        R = u @ vt
        if s.sum() - d < tol:
            break
        d = s.sum()
    return L @ R


def load_run(path: Path):
    z = np.load(path / "ensemble.npz", allow_pickle=False)
    rec = json.loads((path / "record.json").read_text())
    return z["ys"], z["X"], rec["whittle"]["keep_names"]


def leading_axes(ys, X):
    """Six leading principal axes of the ensemble mean on the test set.

    Returns ``apply`` (maps member coordinates ``(K, n, m)`` onto those axes,
    standardised) and the linear loadings of the axes on standardised X.
    """
    y = ys.mean(0)
    mu = y.mean(0)
    _, _, Vt = np.linalg.svd(y - mu, full_matrices=False)
    P = Vt[:N_LEAD].T
    sd = ((y - mu) @ P).std(0)

    def apply(members):
        return ((np.asarray(members, dtype=np.float64) - mu) @ P) / sd

    Xs = (X - X.mean(0)) / X.std(0)
    B, *_ = np.linalg.lstsq(np.c_[Xs, np.ones(len(Xs))], apply(ys).mean(0), rcond=None)
    return apply, B[:-1]


def r2(t, p):
    return float(1.0 - np.mean((t - p) ** 2) / np.var(t))


def chi2(t, p, s):
    return float(np.mean(((t - p) / s) ** 2))


def wlinfit(X, c, s):
    """Weighted linear fit of every column of c on X; returns (d + 1, n_c) coefficients."""
    A = np.c_[X, np.ones(len(X))]
    out = np.zeros((A.shape[1], c.shape[1]))
    for j in range(c.shape[1]):
        out[:, j], *_ = np.linalg.lstsq(A / s[:, [j]], c[:, j] / s[:, j], rcond=None)
    return out


def lin(X, beta):
    return np.c_[X, np.ones(len(X))] @ beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory holding m13/ and rank/")
    ap.add_argument("--setting", default="m13")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--coords", type=int, nargs="+", default=list(range(N_LEAD)))
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--time-limit", type=int, default=60)
    ap.add_argument("--n-holdout", type=int, default=50, help="test points held out (no --augmented)")
    ap.add_argument("--augmented", choices=("empirical", "uniform"), default=None,
                    help="search on augmented_<draw>.npz instead of the test simulations")
    ap.add_argument("--n-train", type=int, default=4000, help="augmented points searched")
    ap.add_argument("--n-holdout-aug", type=int, default=5000, help="augmented points held out")
    ap.add_argument("--max-length", type=int, default=60)
    ap.add_argument("--max-depth", type=int, default=17)
    ap.add_argument("--symbols", default="add,mul,pow,constant,variable,exp,logabs,sqrt")
    ap.add_argument("--length-penalty", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--basis", default=None,
                    help="targets.npz of an earlier call: use its target basis L instead of deriving one "
                         "from these runs, so that expressions are comparable between runs")
    ap.add_argument("--residual", action="store_true",
                    help="search on the nonlinear remainder: the coordinate minus its weighted linear "
                         "fit on all 13 inputs (fitted per split on the searched points)")
    args = ap.parse_args()

    from pyoperon.sklearn import SymbolicRegressor

    root = Path(args.runs)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = lambda setting, t: root / setting / "runs" / f"iterative_camels_d35_trial{t}"

    # the shared subspace and its varimax basis, from all runs of both settings
    proj, per_run, names = [], {}, None
    for setting in ("m13", "rank"):
        for t in (0, 1, 2):
            ys, X, names = load_run(run_dir(setting, t))
            apply, B = leading_axes(ys, X)
            Q, _ = np.linalg.qr(B)
            proj.append(Q @ Q.T)
            per_run[(setting, t)] = (apply, B, X, ys)
    w, V = np.linalg.eigh(np.mean(proj, 0))
    print(f"shared subspace: eigenvalues of the mean projector {np.round(w[::-1][:8], 3)}", flush=True)
    L = varimax(V[:, ::-1][:, :N_LEAD])
    L = L[:, np.argsort(-np.abs(L).max(0))]
    L = L * np.sign(L[np.abs(L).argmax(0), np.arange(N_LEAD)])
    if args.basis:
        L_own = L
        L = np.load(args.basis, allow_pickle=False)["L"]
        Qo, _ = np.linalg.qr(L_own)
        print(f"basis from {args.basis}; share of each of its directions inside this run's shared subspace "
              f"{np.round(((Qo.T @ L) ** 2).sum(0), 3)}", flush=True)
    labels = []
    for j in range(N_LEAD):
        top = [i for i in np.argsort(-np.abs(L[:, j])) if abs(L[i, j]) >= 0.15]
        labels.append("  ".join(f"{names[i]} {L[i, j]:+.2f}" for i in top))
        print(f"target {j + 1}: {labels[-1]}", flush=True)

    def point_set(X, cs):
        """Mean and member spread of the rotated coordinates at the points X."""
        c = cs.mean(0)
        s = cs.std(0, ddof=1)
        return dict(X=np.ascontiguousarray(X, dtype=np.float64), c=c, s=np.maximum(s, 0.1 * np.median(s, 0)))

    def take(ps, idx):
        return {k: v[idx] for k, v in ps.items()}

    # rotated targets per split: the searched set, a held-out set, and the real test simulations
    data, save = {}, {}
    for t in (0, 1, 2):
        apply, B, X, ys = per_run[(args.setting, t)]
        R, *_ = np.linalg.lstsq(B, L, rcond=None)
        test = point_set(X, apply(ys) @ R)
        rng = np.random.default_rng(1000 + t)
        if args.augmented:
            z = np.load(run_dir(args.setting, t) / f"augmented_{args.augmented}.npz", allow_pickle=False)
            aug = point_set(z["X"], apply(z["ys"]) @ R)
            perm = rng.permutation(len(aug["c"]))
            n_tr = min(int(args.n_train), len(perm) // 2)
            fit, ho = take(aug, perm[:n_tr]), take(aug, perm[n_tr: n_tr + int(args.n_holdout_aug)])
            test_out = test                                  # never searched
            save.update({f"Xaug_{t}": aug["X"].astype(np.float32), f"caug_{t}": aug["c"].astype(np.float32),
                         f"saug_{t}": aug["s"].astype(np.float32)})
        else:
            perm = rng.permutation(len(test["c"]))
            fit, ho = take(test, perm[args.n_holdout:]), take(test, perm[: args.n_holdout])
            test_out = None
        beta = wlinfit(fit["X"], fit["c"], fit["s"])         # reference linear fit, on the searched points
        sets = dict(fit=fit, ho=ho, test=test_out)
        if args.residual:
            share = (fit["c"] - lin(fit["X"], beta)).var(0) / fit["c"].var(0)
            print(f"split {t}: nonlinear share of variance {np.round(share, 4)}", flush=True)
            for ps in (fit, ho, test_out):
                if ps is not None:
                    ps["c"] = ps["c"] - lin(ps["X"], beta)
        data[t] = dict(sets, beta=beta, cross_test=test if test_out is None else test_out)
        if args.residual and test_out is None:
            data[t]["cross_test"] = dict(test, c=test["c"] - lin(test["X"], beta))
        print(f"split {t}: searched {len(fit['c'])}, held out {len(ho['c'])}, test "
              f"{0 if test_out is None else len(test_out['c'])}; loading mismatch "
              f"{np.round(np.sqrt(((B @ R - L) ** 2).sum(0)), 3)}; median y_std {np.round(np.median(fit['s'], 0), 3)}; "
              f"sd {np.round(fit['c'].std(0), 2)}", flush=True)
        ct = point_set(X, apply(ys) @ R)                     # test set, full coordinate, for the plots
        save.update({f"X_{t}": ct["X"], f"c_{t}": ct["c"], f"c_std_{t}": ct["s"]})
    np.savez_compressed(out / "targets.npz", L=L, names=np.array(names), **save)

    rows = []
    for t in args.splits:
        d = data[t]
        fit, ho, test = d["fit"], d["ho"], d["test"]
        for j in args.coords:
            ytr, str_ = fit["c"][:, j], fit["s"][:, j]
            zero = bool(args.residual)                       # reference: zero for the remainder, else the linear fit
            ref = lambda ps: np.zeros(len(ps["X"])) if zero else lin(ps["X"], d["beta"])[:, j]
            for mode in args.modes:
                t0 = time.time()
                reg = SymbolicRegressor(
                    allowed_symbols=args.symbols, offspring_generator="basic",
                    optimizer_iterations=10, max_length=args.max_length, max_depth=args.max_depth,
                    initialization_method="btc", n_threads=args.threads, objectives=MODES[mode],
                    epsilon=1e-5, random_state=args.seed, reinserter="keep-best",
                    max_evaluations=int(1e8), generations=int(1e8), symbolic_mode=False,
                    max_time=int(args.time_limit),
                    add_model_scale_term=False, add_model_intercept_term=False,
                    model_selection_criterion="distillery_mdl",
                    dl_length_penalty=args.length_penalty, dl_hessian="finite_difference",
                    forbid_x_in_pow_exponent=True, forbid_self_transcendental=True,
                )
                reg.fit(fit["X"], ytr, y_std=str_)
                front = [f for f in reg.pareto_front_ if np.isfinite(f["distillery_mdl"])]
                best = min(front, key=lambda f: f["distillery_mdl"])
                tree = best["tree"]
                model = reg.get_model_string(tree, 6)
                used = sorted({int(v) for v in re.findall(r"X(\d+)", model)})
                pred = lambda ps: np.asarray(reg.evaluate_model(tree, ps["X"])).ravel()
                p_tr, p_ho = pred(fit), pred(ho)
                cross, cross_aug = {}, {}
                for u in (0, 1, 2):
                    if u != t:
                        cross[u] = r2(data[u]["cross_test"]["c"][:, j], pred(data[u]["cross_test"]))
                        if args.augmented:
                            cross_aug[u] = r2(data[u]["ho"]["c"][:, j], pred(data[u]["ho"]))
                row = dict(
                    split=t, coord=j + 1, target=labels[j], mode=mode, model=model,
                    augmented=args.augmented, n_search=int(len(ytr)),
                    length=int(tree.Length), dl=float(best["distillery_mdl"]),
                    dl_terms={k: float(v) for k, v in dict(best["distillery_mdl_terms"]).items()},
                    front_size=len(front), generations=int(reg.stats_.get("generations", -1)),
                    used=[names[i - 1] for i in used], n_used=len(used),
                    r2_train=r2(ytr, p_tr), r2_holdout=r2(ho["c"][:, j], p_ho),
                    chi2_holdout=chi2(ho["c"][:, j], p_ho, ho["s"][:, j]),
                    r2_cross=cross, r2_cross_aug=cross_aug,
                    residual=bool(args.residual),
                    null_chi2_train=chi2(ytr, 0.0, str_) if args.residual else None,
                    chi2_train=chi2(ytr, p_tr, str_),
                    lin13_r2_holdout=r2(ho["c"][:, j], ref(ho)),
                    lin13_chi2_holdout=chi2(ho["c"][:, j], ref(ho), ho["s"][:, j]),
                    seconds=round(time.time() - t0, 1),
                )
                if test is not None:
                    p_te = pred(test)
                    row.update(r2_test=r2(test["c"][:, j], p_te), chi2_test=chi2(test["c"][:, j], p_te, test["s"][:, j]),
                               lin13_r2_test=r2(test["c"][:, j], ref(test)),
                               lin13_chi2_test=chi2(test["c"][:, j], ref(test), test["s"][:, j]))
                rows.append(row)
                (out / "picks.json").write_text(json.dumps(rows, indent=1))
                extra = "" if test is None else f" | test sims R2 {row['r2_test']:.3f} (linear-13: {row['lin13_r2_test']:.3f})"
                print(f"[split {t} coord {j + 1} {mode}] len {row['length']} dl {row['dl']:.1f} "
                      f"gens {row['generations']} R2 train {row['r2_train']:.3f} holdout {row['r2_holdout']:.3f} "
                      f"(linear-13: {row['lin13_r2_holdout']:.3f}) chi2/pt {row['chi2_holdout']:.2f}{extra} "
                      f"cross {[round(v, 3) for v in cross.values()]}\n    uses {row['used']}\n    {model}",
                      flush=True)


if __name__ == "__main__":
    main()
