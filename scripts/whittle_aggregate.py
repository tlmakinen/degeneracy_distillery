#!/usr/bin/env python
"""Combine first-round screening scores across trials of scripts/iterative_whittle.py.

    python scripts/whittle_aggregate.py results/whittle_camels/screen [more dirs ...] --top-k 14

For every input: the mean ratio of its score to the highest decoy of its own
trial, and how many trials protected it. Writes ``screen_summary.json`` in the
first directory, with the top-k names ready for ``--only``.
"""
from __future__ import annotations

import argparse, glob, json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("dirs", nargs="+", type=Path)
p.add_argument("--top-k", type=int, default=14)
a = p.parse_args()

files = sorted(f for d in a.dirs for f in glob.glob(str(d / "runs" / "*" / "whittle.json")))
ratio: dict[str, list[float]] = {}
prot: dict[str, int] = {}
info = []
for f in files:
    st = json.load(open(f))["stages"][0]
    info.append(st["info_total"])
    for name, sc in st["scores"].items():
        if sc is None:
            continue
        ratio.setdefault(name, []).append(sc / st["decoy_max"])
        prot[name] = prot.get(name, 0) + int(name in st["protected"])
rows = sorted(((np.mean(v), np.min(v), prot[k], len(v), k) for k, v in ratio.items()), reverse=True)
print(f"{len(files)} trials; held-out information per trial (nats): {[round(v, 2) for v in info]}")
print("rank  parameter                          score / highest decoy: mean (min)   protected in")
for i, (mean, lo, pr, n, name) in enumerate(rows, 1):
    mark = " <- top-k" if i <= a.top_k else ""
    print(f"{i:4d}  {name:34s} {mean:6.2f} ({lo:5.2f})                      {pr}/{n}{mark}")
top = [r[4] for r in rows[: a.top_k]]
out = {"n_trials": len(files), "top_k": a.top_k, "top": top, "only": ",".join(top),
       "table": [{"name": r[4], "ratio_mean": round(float(r[0]), 3), "ratio_min": round(float(r[1]), 3),
                  "protected": r[2], "trials": r[3]} for r in rows]}
(a.dirs[0] / "screen_summary.json").write_text(json.dumps(out, indent=1))
print("\n--only", out["only"])
