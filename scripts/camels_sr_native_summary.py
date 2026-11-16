#!/usr/bin/env python
"""Summarise ``picks.json`` written by ``camels_sr_native_mdl.py``.

Run with the patched pyoperon (the expressions are evaluated with Operon's own
parser):

    <worktree>/third_party/operon_mdl/venv/bin/python scripts/camels_sr_native_summary.py \\
        results/whittle_camels/sr_native
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def operon_eval(expr, X):
    import pyoperon as op

    ds = op.Dataset(np.asfortranarray(np.column_stack((X, np.zeros(len(X))))))
    target = max(ds.Variables, key=lambda v: v.Index)
    names = {v.Name: v.Hash for v in ds.Variables if v.Hash != target.Hash}
    tree = op.InfixParser.Parse(str(expr), names)
    return np.asarray(op.Evaluate(op.DispatchTable(), tree, ds, op.Range(0, ds.Rows)), dtype=np.float64).ravel()


def main():
    out = Path(sys.argv[1])
    rows = json.loads((out / "picks.json").read_text())
    z = np.load(out / "targets.npz", allow_pickle=False)
    names = [str(n) for n in z["names"]]
    L = z["L"]
    rng = np.random.default_rng(0)
    Xp = rng.uniform(1.0, 2.0, size=(4000, len(names)))
    Xs = (Xp - 1.5) / np.sqrt(1.0 / 12.0)
    A = np.c_[Xs, np.ones(len(Xs))]

    for r in rows:
        f = operon_eval(r["model"], Xp)
        ok = np.isfinite(f)
        b, *_ = np.linalg.lstsq(A[ok], f[ok], rcond=None)
        r["finite_frac"] = float(ok.mean())
        r["linear_share"] = float(1 - np.var(f[ok] - A[ok] @ b) / np.var(f[ok]))
        r["loadings"] = b[:-1]
        j = r["coord"] - 1
        r["cos_target"] = float(b[:-1] @ L[:, j] / np.linalg.norm(b[:-1]))

    modes = sorted({r["mode"] for r in rows})
    for mode in modes:
        print(f"\n################ search: {mode}")
        for j in sorted({r["coord"] for r in rows}):
            sel = [r for r in rows if r["mode"] == mode and r["coord"] == j]
            print(f"\n== coordinate {j}: target  {sel[0]['target']}")
            for r in sel:
                cross = ", ".join(f"{v:.3f}" for v in r["r2_cross"].values())
                print(f"  split {r['split']}: nodes {r['length']:2d}  dl {r['dl']:6.1f}  R2 held-out {r['r2_holdout']:.3f} "
                      f"(13-input linear fit {r['lin13_r2_holdout']:.3f})  chi2/pt {r['chi2_holdout']:.2f} "
                      f"(linear {r['lin13_chi2_holdout']:.2f})  other splits {cross}  "
                      f"linear share {r['linear_share']:.3f}  cos to target {r['cos_target']:.3f}")
                if "r2_test" in r:
                    aug = ", ".join(f"{v:.3f}" for v in r.get("r2_cross_aug", {}).values())
                    print(f"           real test simulations: R2 {r['r2_test']:.3f} (13-input linear fit {r['lin13_r2_test']:.3f})  "
                          f"chi2/pt {r['chi2_test']:.2f} (linear {r['lin13_chi2_test']:.2f})  other splits' augmented sets {aug}")
                print(f"           {r['model']}")
            count = Counter(v for r in sel for v in r["used"])
            print("  inputs used (splits out of %d): " % len(sel) + ", ".join(f"{k} {v}" for k, v in count.most_common()))
            mean_load = np.mean([r["loadings"] / np.linalg.norm(r["loadings"]) for r in sel], 0)
            top = [i for i in np.argsort(-np.abs(mean_load)) if abs(mean_load[i]) >= 0.1]
            print("  mean normalised loadings of the picks: " + "  ".join(f"{names[i]} {mean_load[i]:+.2f}" for i in top))

    print("\n################ medians per search")
    for mode in modes:
        sel = [r for r in rows if r["mode"] == mode]
        med = lambda k: float(np.median([r[k] for r in sel]))
        cross = float(np.median([v for r in sel for v in r["r2_cross"].values()]))
        print(f"  {mode:6s} n={len(sel)}  nodes {med('length'):.0f}  dl {med('dl'):.1f}  inputs {med('n_used'):.0f}  "
              f"R2 held-out {med('r2_holdout'):.3f}  13-input linear {med('lin13_r2_holdout'):.3f}  chi2/pt {med('chi2_holdout'):.2f}  "
              f"other splits {cross:.3f}  linear share {med('linear_share'):.3f}  generations {med('generations'):.0f}")
        if all("r2_test" in r for r in sel):
            better = sum(r["r2_test"] > r["lin13_r2_test"] for r in sel)
            print(f"         real test simulations: R2 {med('r2_test'):.4f}  13-input linear {med('lin13_r2_test'):.4f}  "
                  f"chi2/pt {med('chi2_test'):.2f} (linear {med('lin13_chi2_test'):.2f})  picks better than the linear fit: {better} of {len(sel)}")


if __name__ == "__main__":
    main()
