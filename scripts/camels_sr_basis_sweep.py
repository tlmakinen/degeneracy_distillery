"""SR basis sweeps over a finished one-step run, with no neural refit.

The map is deterministic and already fitted, so every arm of a basis sweep can
reuse one run's augmented ``(X, y, y_std, dy, Fs)``. This loads the held-out
half of a completed all-35 run's ``split_data.npz`` and refits SR under a
chosen parameter basis.

Two modes:

``kladder``   top-k parameters by a global selector, for one k. Running
              k = 12..2 shows which terms survive as the basis is squeezed;
              k=2 is Omega_b and Omega_m.
``percomp``   each eta_i gets its own top-k parameters, selected from that
              component's own Jacobian row. ``--selector varj`` ranks by
              Var[d eta_i / d theta_j] over the prior, which is a
              *nonlinearity* score (a linear term has constant derivative and
              zero variance); ``meanj`` ranks by mean |d eta_i / d theta_j|,
              a plain sensitivity score. They disagree substantially.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def r2(a, b) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(1.0 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))


def load_run(run_dir: Path):
    npz = sorted(Path(run_dir).glob("runs/*/sr/split_data.npz"))
    if not npz:
        raise FileNotFoundError(f"no split_data.npz under {run_dir}")
    d = np.load(npz[0], allow_pickle=True)
    if int(d["n_params"]) != 35:
        raise ValueError(
            f"{npz[0]} was written with n_params={int(d['n_params'])}; "
            "the sweep needs a full 35-parameter run as its source"
        )
    return (
        np.asarray(d["X_test"], dtype=np.float64),
        np.asarray(d["y_test"], dtype=np.float64),
        np.asarray(d["y_std_test"], dtype=np.float64),
        np.asarray(d["dy_sr_test"], dtype=np.float64),
        np.asarray(d["Fs_test"], dtype=np.float64),
    )


def select_basis(args, dy, Fs, comp: int | None):
    """Column indices SR may use. ``comp`` is set only in percomp mode."""
    if args.mode == "kladder":
        sens = np.sqrt(np.clip(np.diag(Fs.mean(0)), 0.0, None))
        rank = np.argsort(-sens)
        return np.asarray(sorted(int(v) for v in rank[: args.k]), dtype=int)
    row = dy[:, comp, :]
    score = row.var(0) if args.selector == "varj" else np.abs(row).mean(0)
    rank = np.argsort(-score)
    return np.asarray(sorted(int(v) for v in rank[: args.k]), dtype=int)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from-run", type=Path, required=True)
    p.add_argument("--mode", choices=("kladder", "percomp"), required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--selector", choices=("fisher", "varj", "meanj"), default="fisher")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sr-time-limit", type=int, default=300)
    p.add_argument("--sr-max-length", type=int, default=30)
    p.add_argument("--sr-max-depth", type=int, default=6)
    p.add_argument("--sr-symbols", type=str,
                   default="add,mul,div,pow,constant,variable,square,exp")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from degeneracy_distillery.problems.camels import PARAM_NAMES
    from degeneracy_distillery.sr_utils import (
        fit_and_analyze_sr, get_y_sr, sr_structure_predicate,
    )

    X, y, y_std, dy, Fs = load_run(args.from_run)
    m = int(y.shape[1])
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] source {args.from_run}  X {X.shape}  m {m}", flush=True)
    print(f"[sweep] mode={args.mode} k={args.k} selector={args.selector}", flush=True)

    comps = [None] if args.mode == "kladder" else list(range(m))
    out: dict = {"mode": args.mode, "k": args.k, "selector": args.selector,
                 "source": str(args.from_run), "components": {}}
    t0 = time.time()

    for c in comps:
        keep = select_basis(args, dy, Fs, c)
        names = [PARAM_NAMES[i] for i in keep]
        fit_list = list(range(m)) if c is None else [c]
        tag = "all" if c is None else f"c{c}"
        print(f"\n[sweep] {tag}: basis {names}", flush=True)
        sub = args.out / f"sr_{tag}"
        sub.mkdir(parents=True, exist_ok=True)
        mdl, frob, analysis, _ = fit_and_analyze_sr(
            X[:, keep], y, y_std, dy[:, :, keep], Fs[:, keep][:, :, keep],
            n_params=int(keep.size),
            components_to_fit=fit_list,
            slice_fisher=False,
            parent_dir=str(sub) + os.sep,
            test_size=0.5, random_state=32134 + args.seed, shuffle=True,
            time_limit=args.sr_time_limit,
            max_length=args.sr_max_length, max_depth=args.sr_max_depth,
            allowed_symbols=args.sr_symbols,
            max_complexity_thresh=max(20, args.sr_max_length),
            equation_set="pareto", length_penalty=2.0,
            equation_predicate=sr_structure_predicate(
                n_params=int(keep.size), forbid_self_transcendental=True,
                check_nested_exp=False, forbid_x_in_pow_exponent=True,
            ),
            equation_timeout=60.0,
        )
        exprs = [str(e) for e in mdl]
        # Score on the same rows SR saw split from; relative comparison across
        # arms is the point, and every arm uses the identical split.
        pred = get_y_sr(exprs, X[:, keep])
        scores = {}
        for idx, j in enumerate(fit_list):
            col = idx if c is not None else j
            scores[f"c{j}"] = r2(y[:, j], pred[:, col])
        out["components"][tag] = {
            "basis_index": [int(v) for v in keep],
            "basis_names": names,
            "expressions": exprs,
            "r2": scores,
        }
        print(f"[sweep] {tag}: R2 { {k: round(v,4) for k,v in scores.items()} }", flush=True)

    out["runtime_s"] = time.time() - t0
    (args.out / "sweep.json").write_text(json.dumps(out, indent=2))
    print(f"\n[sweep] wrote {args.out / 'sweep.json'}  in {out['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
