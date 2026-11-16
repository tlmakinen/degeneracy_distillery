#!/usr/bin/env python
"""Two-stage discovery in many ambient dimensions: whittle the inputs, then distil.

Procedure and results: notes/whittle_procedure.md.

Stage A (whittle). Fit a short one-step probe on the kept inputs plus decoy
columns, score every input by its held-out Fisher sensitivity
``sqrt(mean_theta sum_a (d eta_a / d theta_j)^2)`` per unit of prior width,
keep the top fraction, and repeat until ``--k-target`` inputs remain. All
stages reuse one set of simulations; a dropped input keeps varying in them and
is thereby marginalised.

* Decoys never reached the simulator, so they are inactive by construction.
  ``m`` of them occupy the first ``m`` input positions, because the map's skip
  connection is initialised to the identity on those columns and would
  otherwise inflate the score of whichever real inputs sat there. The remaining
  decoys set the protection threshold: an input scoring above
  ``--protect-factor`` times the highest decoy is never dropped.
* ``--decoy-kind uniform`` draws decoys uniformly on the prior box.
  ``permute`` copies real parameter columns with their rows shuffled, which
  keeps each column's own marginal and is the right choice for a fixed dataset
  whose marginals are not uniform. ``auto`` picks ``permute`` for problems that
  expose ``held_out()`` and ``uniform`` otherwise.
* Guard. A cut that removes an active input lowers the total held-out
  information of the next probe. If it falls by more than ``--guard-nats`` the
  cut is undone and repeated more gently.

Stage B (distil). The existing one-step pipeline (``run_oneshot_trial`` in
``oneshot_sweep.py``) runs unchanged on the surviving inputs. It is not told
which of them are active. ``--skip-discovery`` stops after Stage A and leaves
``whittle.json`` with the kept indices and names.

Modes
-----
``iterative``   Stage A with geometric cuts, then Stage B.            (the method)
``single-cut``  Stage A with one cut straight to ``--k-target``, then Stage B.
``direct``      No Stage A: the existing pipeline on all inputs, without
                symbolic regression.

Arguments this script does not know are passed to ``oneshot_sweep``'s parser,
so Stage B takes the usual options (``--ensemble-k``, ``--sr-time-limit``,
``--quick`` ...).

    # hidden banana, as in the d = 100 check
    python scripts/iterative_whittle.py --d 100 --num-trials 10 --skip-discovery --out-dir results/whittle_d100

    # a fixed dataset
    python scripts/iterative_whittle.py --problem camels --skip-discovery --k-target 10 \\
        --probe-repeats 3 --guard-nats 0.4 --out-dir results/whittle_camels
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(SCRIPTS_DIR), str(SCRIPTS_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# A problem restricted to a subset of its parameter columns.
# ---------------------------------------------------------------------------
class SubsetProblem:
    """Presents ``base`` with only the columns ``keep`` to the existing driver."""

    def __init__(self, base, keep):
        self.base = base
        self.keep = np.asarray(keep, dtype=int)
        self.d = int(self.keep.size)
        self.name = base.name
        self.param_names = tuple(base.param_names[i] for i in self.keep)
        self.theta_lo = np.asarray(base.theta_lo, dtype=float)[self.keep]
        self.theta_hi = np.asarray(base.theta_hi, dtype=float)[self.keep]
        self.use_log_features = base.use_log_features
        self.m_probe = int(min(base.m_probe, self.d))
        self.expected_rank = base.expected_rank
        self.last_aux: dict = {}
        # Hidden-banana bookkeeping: where the two active inputs sit in the subset.
        self.pos = None
        if hasattr(base, "I0") and hasattr(base, "I1"):
            where = {int(v): j for j, v in enumerate(self.keep)}
            self.pos = (where.get(int(base.I0)), where.get(int(base.I1)))
        if hasattr(base, "estimator_factories"):
            self.estimator_factories = base.estimator_factories

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.asarray(self.base.sample_prior(int(n), rng))[:, self.keep]

    def estimator_factory(self, m: int):
        return self.base.estimator_factory(m)

    def sr_hints(self) -> dict:
        return self.base.sr_hints()

    def truth_coords(self, theta: np.ndarray):
        if self.pos is None or None in self.pos:   # no known truth, or an active input was dropped
            return None
        th = np.asarray(theta)
        p0, p1 = self.pos
        return np.stack([th[:, p0], th[:, p1] - th[:, p0] ** 2], axis=1)

    def gates(self, payload: dict) -> dict[str, bool]:
        if self.pos is None:
            return dict(self.base.gates(payload))
        r_hat = int(payload.get("r_hat", -1))
        order = list(payload.get("screen_order", []))
        top2 = set(int(v) for v in order[:2]) if len(order) >= 2 else set()
        return {
            "rank_correct": bool(r_hat == 2),
            "screen_set_correct": bool(None not in self.pos and top2 == set(self.pos)),
        }


# ---------------------------------------------------------------------------
# Stage A.
# ---------------------------------------------------------------------------
def make_decoys(kind: str, n_cols: int, th_real: np.ndarray, lo_d: float, hi_d: float, rng) -> np.ndarray:
    """``n_cols`` columns that are independent of the data by construction."""
    n = th_real.shape[0]
    if kind == "uniform":
        return rng.uniform(lo_d, hi_d, size=(n, n_cols))
    cols = rng.choice(th_real.shape[1], size=n_cols, replace=th_real.shape[1] < n_cols)
    return np.stack([th_real[rng.permutation(n), c] for c in cols], axis=1)


def probe_scores(th_keep, th_all, x_std, lo_keep, hi_keep, lo_d, hi_d, estimator_factory, *,
                 m, n_decoy, steps, seed, rng, nval, use_log_features, decoy_kind, skip_fill="decoys",
                 score="all"):
    """One probe fit. Returns (score of kept inputs, decoy scores, info per axis).

    Scores are per unit of prior width, so inputs with different ranges compare.

    The map's skip connection starts as the identity on its first ``m`` input
    columns, which inflates their scores. ``skip_fill`` says what sits there:

    ``decoys``   ``m`` decoy columns. Every real input is scored without bias.
                 The map then starts with no information and has to find it;
                 that works on the hidden banana and fails on CAMELS.
    ``shuffle``  the real inputs in a random order. The ``m`` that land in the
                 skip positions get no score from this probe (NaN); average
                 over several probes with ``--probe-repeats``.
    ``none``     the real inputs in their given order. Scores of the first
                 ``m`` are inflated. For diagnosis only.
    """
    from degeneracy_distillery.oneshot import info_nats, spectrum, train_oneshot

    k = th_keep.shape[1]
    perm = rng.permutation(k) if skip_fill == "shuffle" else np.arange(k)
    m_skip = m if skip_fill == "decoys" else 0
    skip_cols = make_decoys(decoy_kind, m_skip, th_all, lo_d, hi_d, rng) if m_skip else np.empty((th_keep.shape[0], 0))
    tail_decoys = make_decoys(decoy_kind, n_decoy, th_all, lo_d, hi_d, rng)  # the null reference
    T = np.concatenate([skip_cols, th_keep[:, perm], tail_decoys], axis=1)
    lo = np.concatenate([np.full(m_skip, lo_d), lo_keep[perm], np.full(n_decoy, lo_d)])
    hi = np.concatenate([np.full(m_skip, hi_d), hi_keep[perm], np.full(n_decoy, hi_d)])
    uniform_box = bool(np.allclose(lo, lo[0]) and np.allclose(hi, hi[0]))
    fit = train_oneshot(
        T[:-nval], x_std[:-nval], T[-nval:], x_std[-nval:],
        m=m, m0=m,
        lo=float(lo[0]) if uniform_box else tuple(float(v) for v in lo),
        hi=float(hi[0]) if uniform_box else tuple(float(v) for v in hi),
        steps=int(steps), seed=int(seed),
        verbose=False, use_log_features=use_log_features, estimator_factory=estimator_factory,
    )
    info = np.asarray(info_nats(fit.eta(T[-nval:]), fit.eta_hat(x_std[-nval:])), dtype=float)
    J = fit.jac(T[-nval:])                                   # (nval, m, m_skip + k + n_decoy)
    if score == "informative":
        # Rotate the latent axes to the basis info_nats uses, and weight each axis by the
        # share of its variance the data predict, R^2_a = 1 - exp(-2 info_a). An axis the
        # data cannot predict then counts for nothing, whatever inputs feed it.
        _, _, V = spectrum(fit.eta(T[-nval:]))
        w = np.clip(1.0 - np.exp(-2.0 * info), 0.0, 1.0)
        J = np.einsum("ab,nbj->naj", np.asarray(V).T, J) * np.sqrt(w)[None, :, None]
    sens = np.sqrt((J ** 2).sum(1).mean(0)) * (hi - lo)
    s_real = np.empty(k)
    s_real[perm] = sens[m_skip: m_skip + k]
    if skip_fill == "shuffle" and k > m:
        s_real[perm[:m]] = np.nan                            # these sat in the skip positions
    return s_real, sens[m_skip + k:], info


def whittle(th, x_std, problem, args, seed: int, mode: str) -> dict:
    """Run Stage A. Returns the kept column indices, their names and a per-stage log."""
    rng = np.random.default_rng(seed + 991)
    lo_all = np.broadcast_to(np.asarray(problem.theta_lo, dtype=float), (th.shape[1],)).copy()
    hi_all = np.broadcast_to(np.asarray(problem.theta_hi, dtype=float), (th.shape[1],)).copy()
    lo_d, hi_d = float(np.median(lo_all)), float(np.median(hi_all))
    n, d = th.shape
    nval = max(20, int(args.nval_frac_a * n))
    m_probe = int(min(problem.m_probe, d))
    decoy_kind = args.decoy_kind
    if decoy_kind == "auto":
        decoy_kind = "permute" if hasattr(problem, "held_out") else "uniform"
    hidden = (int(problem.I0), int(problem.I1)) if hasattr(problem, "I0") and hasattr(problem, "I1") else None
    names = list(problem.param_names)

    keep = np.arange(d)
    if args.front:
        front = [names.index(v.strip()) for v in args.front.split(",") if v.strip()]
        keep = np.asarray(front + [j for j in range(d) if j not in front], dtype=int)
    frac = float(args.keep_frac)
    stages: list[dict] = []
    prev: dict | None = None            # state before the last cut, for the guard
    stage = 0
    while True:
        stage += 1
        t0 = time.time()
        reps_real, reps_dec, reps_info = [], [], []
        for r in range(int(args.probe_repeats)):
            s_r, s_d, info_r = probe_scores(
                th[:, keep], th, x_std, lo_all[keep], hi_all[keep], lo_d, hi_d, problem.estimator_factory,
                m=int(min(m_probe, keep.size)) if args.skip_fill != "decoys" else m_probe,
                n_decoy=int(args.n_decoy), steps=int(args.probe_steps_a),
                seed=seed + 13 * stage + 1009 * r, rng=rng, nval=nval,
                use_log_features=problem.use_log_features, decoy_kind=decoy_kind,
                skip_fill=args.skip_fill, score=args.score,
            )
            reps_real.append(s_r)
            reps_dec.append(s_d)
            reps_info.append(info_r)
        reps_real = np.asarray(reps_real, dtype=float)
        n_scored = np.isfinite(reps_real).sum(0)
        with np.errstate(invalid="ignore"):
            s_real = np.where(n_scored > 0, np.nansum(reps_real, axis=0) / np.maximum(n_scored, 1), np.inf)
        # an input that sat in the skip positions in every probe has no score: keep it
        s_dec = np.mean(reps_dec, axis=0)
        info = np.mean(reps_info, axis=0)
        info_tot = float(np.clip(info, 0.0, None).sum())
        thr = float(args.protect_factor) * float(s_dec.max())
        order = np.argsort(-s_real)
        rec = {
            "stage": stage, "n_inputs": int(keep.size),
            "info_nats": [round(float(v), 3) for v in info], "info_total": round(info_tot, 3),
            "decoy_max": round(float(s_dec.max()), 4), "decoy_median": round(float(np.median(s_dec)), 4),
            "protect_threshold": round(thr, 4),
            "n_protected": int((s_real > thr).sum()),
            "protected": [names[int(keep[i])] for i in order if s_real[i] > thr],
            "probes": int(args.probe_repeats),
            "top": [(int(keep[i]), names[int(keep[i])], _num(s_real[i])) for i in order[:8]],
            "scores": {names[int(keep[i])]: _num(s_real[i]) for i in order},
            "seconds": round(time.time() - t0, 1),
        }
        if args.skip_fill == "shuffle":
            rec["probes_scored"] = {names[int(keep[i])]: int(n_scored[i]) for i in order}
            if keep.size <= m_probe:
                rec["note"] = "every input sat in the skip positions; scores are not comparable with the decoys"
        if hidden is not None:
            rec["rank_of_hidden"] = {
                str(h): (int(np.where(keep[order] == h)[0][0]) + 1 if h in keep else None) for h in hidden
            }

        # Guard: did the cut that produced this input set lose information?
        if prev is not None and info_tot < prev["info_total"] - float(args.guard_nats) and prev["retries"] < 2:
            rec["guard"] = f"info fell {prev['info_total']:.2f} -> {info_tot:.2f}; cut undone"
            stages.append(rec)
            _say(rec)
            # redo the cut from the previous state, keeping twice as many
            k_gentle = min(prev["keep"].size - 1, max(int(args.k_target), 2 * keep.size))
            keep = _cut(prev["keep"], prev["s_real"], prev["thr"], k_gentle)
            prev = {**prev, "retries": prev["retries"] + 1}
            continue

        stages.append(rec)
        _say(rec)
        if keep.size <= int(args.k_target):
            break
        k_next = int(args.k_target) if mode == "single-cut" else max(int(args.k_target), int(np.ceil(keep.size * frac)))
        new_keep = _cut(keep, s_real, thr, k_next)
        if new_keep.size >= keep.size:      # everything is protected: cannot shrink further
            rec["stopped"] = "all remaining inputs are above the protection threshold"
            break
        prev = {"keep": keep, "s_real": s_real, "thr": thr, "info_total": info_tot, "retries": 0}
        keep = new_keep
    return {
        "keep": [int(v) for v in keep],
        "keep_names": [names[int(v)] for v in keep],
        "decoy_kind": decoy_kind,
        "stages": stages,
    }


def refit_subset(th, x_std, th_te, x_te_std, problem, keep, args, seed: int, workdir: Path) -> dict:
    """One-step fits on only the columns ``keep``; the other inputs are marginalised.

    ``m`` latent coordinates for both eta(theta) and the data summary; ``m = k``
    (the default) makes the map square on the kept inputs. Information is scored
    on the test split, which no fit sees.
    """
    from degeneracy_distillery.oneshot import info_nats, jtj_eigengap, train_oneshot

    keep = np.asarray(keep, dtype=int)
    k = int(keep.size)
    m = k if int(args.refit_m) < 0 else int(min(args.refit_m, k))
    lo = np.broadcast_to(np.asarray(problem.theta_lo, dtype=float), (th.shape[1],))[keep]
    hi = np.broadcast_to(np.asarray(problem.theta_hi, dtype=float), (th.shape[1],))[keep]
    uniform_box = bool(np.allclose(lo, lo[0]) and np.allclose(hi, hi[0]))
    n = th.shape[0]
    nval = max(20, int(args.nval_frac_a * n))
    T, Tte = th[:, keep], th_te[:, keep]
    members, dump = [], {}
    for e in range(int(args.refit_members)):
        t0 = time.time()
        fit = train_oneshot(
            T[:-nval], x_std[:-nval], T[-nval:], x_std[-nval:], m=m, m0=m,
            lo=float(lo[0]) if uniform_box else tuple(float(v) for v in lo),
            hi=float(hi[0]) if uniform_box else tuple(float(v) for v in hi),
            steps=int(args.refit_steps), seed=int(seed + 101 * e + 7),
            verbose=False, use_log_features=problem.use_log_features,
            estimator_factory=problem.estimator_factory,
        )
        eta_te, hat_te = fit.eta(Tte), fit.eta_hat(x_te_std)
        info_te = np.asarray(info_nats(eta_te, hat_te), dtype=float)
        info_va = np.asarray(info_nats(fit.eta(T[-nval:]), fit.eta_hat(x_std[-nval:])), dtype=float)
        J = fit.jac(Tte)
        gap = jtj_eigengap(J)
        rec = {
            "member": e, "m": m, "k": k,
            "info_test_nats": [round(float(v), 3) for v in info_te],
            "info_test_total": round(float(np.clip(info_te, 0, None).sum()), 3),
            "n_axes_above_half_nat": int((info_te > 0.5).sum()),
            "info_val_total": round(float(np.clip(info_va, 0, None).sum()), 3),
            "jtj_eigengap_rank": int(gap["rank"]),
            "seconds": round(time.time() - t0, 1),
        }
        members.append(rec)
        dump[f"eta_{e}"], dump[f"eta_hat_{e}"], dump[f"J_{e}"] = eta_te, hat_te, J
        print(f"[refit] {json.dumps(rec)}", flush=True)
    names = [problem.param_names[int(i)] for i in keep]
    out = {
        "keep": [int(v) for v in keep], "keep_names": names, "m": m, "steps": int(args.refit_steps),
        "members": members,
        "info_test_total_mean": round(float(np.mean([r["info_test_total"] for r in members])), 3),
        "n_axes_above_half_nat": [r["n_axes_above_half_nat"] for r in members],
    }
    tag = f"refit_k{k}_m{m}"
    np.savez_compressed(workdir / f"{tag}.npz", theta_test=Tte, keep=keep, **dump)
    (workdir / f"{tag}.json").write_text(json.dumps(out, indent=1))
    return out


def _num(v):
    return round(float(v), 4) if np.isfinite(v) else None


def _say(rec: dict) -> None:
    brief = {k: v for k, v in rec.items() if k != "scores"}
    print(f"[whittle] {json.dumps(brief)}", flush=True)


def _cut(keep: np.ndarray, s_real: np.ndarray, thr: float, k_next: int) -> np.ndarray:
    """Top ``k_next`` inputs by score, plus any input above the protection threshold."""
    order = np.argsort(-s_real)
    chosen = set(int(i) for i in order[:k_next]) | set(int(i) for i in np.where(s_real > thr)[0])
    return np.sort(keep[sorted(chosen)])


# ---------------------------------------------------------------------------
# Trials.
# ---------------------------------------------------------------------------
ROW_KEYS = [
    "problem", "mode", "d", "trial", "seed", "status", "hidden", "kept", "kept_names", "k_final", "n_stages",
    "both_active_kept", "info_total_first", "info_total_last", "r_hat", "rank_correct", "screen_set_correct",
    "r2_true_min", "r2_sr_min", "symbolic_recovered", "success", "expression",
    "refit_m", "refit_info_test", "refit_axes", "seconds_whittle", "seconds_discovery", "error",
]


def run_trial(args, sweep_args, problem_kwargs: dict, trial: int, out_dir: Path) -> dict:
    import oneshot_sweep as osw
    from degeneracy_distillery.problems.base import get_problem

    kw = dict(problem_kwargs)
    if args.d is not None:
        kw.setdefault("d", int(args.d))
    d_seed = int(args.d) if args.d is not None else 0
    seed = int(args.seed + 7919 * trial + 31 * d_seed)
    problem = get_problem(args.problem, seed=seed, **kw)
    d = int(problem.d)
    hidden = sorted([int(problem.I0), int(problem.I1)]) if hasattr(problem, "I0") and hasattr(problem, "I1") else None
    tag = f"{args.mode}_{problem.name}_d{d}_trial{trial}" if args.problem != "rosenbrock" else f"{args.mode}_d{d}_trial{trial}"
    workdir = out_dir / "runs" / tag
    workdir.mkdir(parents=True, exist_ok=True)
    row: dict = {"problem": problem.name, "mode": args.mode, "d": d, "trial": trial, "seed": seed, "status": "ok", "hidden": hidden}
    print(f"\n=== {tag} (seed {seed}" + (f", hidden {hidden}" if hidden else "") + ") ===", flush=True)

    rng = np.random.default_rng(seed)
    th, x = problem.sample(int(sweep_args.nsims), rng)
    if hasattr(problem, "held_out"):        # fixed dataset: the test split is carved out at load time
        th_te, x_te = problem.held_out()
        th_te, x_te = th_te[: int(sweep_args.n_test)], x_te[: int(sweep_args.n_test)]
    else:
        th_te, x_te = problem.sample(int(sweep_args.n_test), rng)
    th, x, th_te, x_te = (np.asarray(a) for a in (th, x, th_te, x_te))
    record: dict = {"tag": tag, "seed": seed, "hidden": hidden, "n_train": int(th.shape[0]), "n_test": int(th_te.shape[0])}

    if args.mode == "direct":
        keep = np.arange(d)
        sub = problem
        sweep_args.skip_sr = True
        row.update(kept="all", k_final=d, n_stages=0, both_active_kept=True if hidden else None)
    else:
        x_std, x_te_std, _, _ = osw.standardise_x(x, x_te)
        t0 = time.time()
        if args.only:
            want = [v.strip() for v in args.only.split(",") if v.strip()]
            idx = sorted(list(problem.param_names).index(v) for v in want)
            wh = {"keep": idx, "keep_names": [problem.param_names[i] for i in idx], "decoy_kind": None,
                  "stages": [{"stage": 0, "n_inputs": len(idx), "info_total": None, "note": "given by --only"}]}
        else:
            wh = whittle(th, x_std, problem, args, seed, args.mode)
        row["seconds_whittle"] = round(time.time() - t0, 1)
        keep = np.asarray(wh["keep"], dtype=int)
        record["whittle"] = wh
        row.update(
            kept=[int(v) for v in keep], kept_names=wh["keep_names"], k_final=int(keep.size), n_stages=len(wh["stages"]),
            both_active_kept=(bool(set(hidden) <= set(keep.tolist())) if hidden else None),
            info_total_first=wh["stages"][0]["info_total"], info_total_last=wh["stages"][-1]["info_total"],
        )
        sub = SubsetProblem(problem, keep)
        (workdir / "whittle.json").write_text(json.dumps(wh, indent=1))
        print(f"[whittle] kept {len(keep)} of {d}: {wh['keep_names']}", flush=True)
        if int(args.refit_m) != 0:
            rf = refit_subset(th, x_std, th_te, x_te_std, problem, keep, args, seed, workdir)
            record["refit"] = rf
            row.update(refit_m=rf["m"], refit_info_test=rf["info_test_total_mean"],
                       refit_axes=rf["n_axes_above_half_nat"])

    if not args.skip_discovery:
        t0 = time.time()
        rec = osw.run_oneshot_trial(sub, th[:, keep], x, th_te[:, keep], x_te, sweep_args, seed, workdir)
        row["seconds_discovery"] = round(time.time() - t0, 1)
        for k in ("r_hat", "rank_correct", "screen_set_correct", "r2_true_min", "r2_sr_min", "symbolic_recovered", "expression"):
            row[k] = rec.get(k)
        record["discovery"] = {k: v for k, v in rec.items() if not k.startswith("_") and not isinstance(v, np.ndarray)}
        if hidden is not None:
            r2n = row.get("r2_true_min")
            neural_ok = bool(r2n is not None and np.isfinite(r2n) and r2n >= 0.95)
            if args.mode == "direct":   # no symbolic stage: success means rank, screen and neural recovery
                row["success"] = bool(row.get("rank_correct") and row.get("screen_set_correct") and neural_ok)
            else:
                row["success"] = bool(
                    row.get("both_active_kept") and row.get("rank_correct")
                    and row.get("screen_set_correct") and row.get("symbolic_recovered")
                )
    (workdir / "record.json").write_text(json.dumps(record, indent=1, default=str))
    return row


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_KEYS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", default="rosenbrock", help="registry name in degeneracy_distillery.problems")
    p.add_argument("--problem-arg", action="append", default=[], help="k=v passed to the problem constructor")
    p.add_argument("--d", type=int, default=None, help="ambient dimension, for problems that take one (default 100 for rosenbrock)")
    p.add_argument("--mode", choices=("iterative", "single-cut", "direct"), default="iterative")
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--trial-start", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("runs/whittle"))
    p.add_argument("--k-target", type=int, default=5, help="stop whittling at this many inputs")
    p.add_argument("--keep-frac", type=float, default=0.25, help="fraction of inputs kept at each cut")
    p.add_argument("--n-decoy", type=int, default=20, help="decoy columns that set the protection threshold")
    p.add_argument("--decoy-kind", choices=("auto", "uniform", "permute"), default="auto")
    p.add_argument("--skip-fill", choices=("decoys", "shuffle", "none"), default="decoys",
                   help="what occupies the skip-connection positions; see probe_scores. Use shuffle with "
                        "--probe-repeats 5 or more when the map learns nothing with decoys there")
    p.add_argument("--score", choices=("all", "informative"), default="all",
                   help="all: sensitivity over every latent axis. informative: only along latent axes the data "
                        "predict, weighted by each axis's R^2")
    p.add_argument("--protect-factor", type=float, default=2.0,
                   help="never drop an input scoring above this many times the highest decoy")
    p.add_argument("--probe-steps-a", type=int, default=3000, help="training steps of each Stage A probe")
    p.add_argument("--probe-repeats", type=int, default=1,
                   help="average scores over this many probes per round (steadier with few simulations)")
    p.add_argument("--guard-nats", type=float, default=0.1)
    p.add_argument("--nval-frac-a", type=float, default=0.15)
    p.add_argument("--skip-discovery", action="store_true", help="Stage A only")
    p.add_argument("--front", default="", help="comma-separated input names to place first, i.e. in the skip positions")
    p.add_argument("--only", default="", help="comma-separated input names: skip the screen and use exactly these")
    p.add_argument("--refit-m", type=int, default=0,
                   help="step 2: refit the one-step map on the kept inputs with this many latents; -1 = as many as "
                        "inputs kept (square); 0 = off")
    p.add_argument("--refit-steps", type=int, default=20000)
    p.add_argument("--refit-members", type=int, default=4)
    args, rest = p.parse_known_args()
    if args.problem == "rosenbrock" and args.d is None:
        args.d = 100

    import oneshot_sweep as osw

    argv = sys.argv
    try:
        sys.argv = ["oneshot_sweep.py", "--problem", args.problem, *rest]
        sweep_args = osw.apply_budgets(osw.parse_args())
    finally:
        sys.argv = argv
    problem_kwargs = osw.parse_problem_args(args.problem_arg)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"config_{args.mode}.json").write_text(json.dumps(
        {"whittle": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "discovery": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(sweep_args).items()}}, indent=1))

    for trial in range(args.trial_start, args.trial_start + args.num_trials):
        try:
            row = run_trial(args, sweep_args, problem_kwargs, trial, out_dir)
        except Exception as exc:
            traceback.print_exc()
            row = {"problem": args.problem, "mode": args.mode, "d": args.d, "trial": trial, "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}"}
        append_row(out_dir / "metrics.csv", row)
        print(f"[trial] {json.dumps({k: row.get(k) for k in ROW_KEYS if k in row}, default=str)}", flush=True)


if __name__ == "__main__":
    main()
