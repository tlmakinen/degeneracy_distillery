#!/usr/bin/env python
"""Redraw rosenbrock_npe_scaling.png from the packaged N=2000 metrics.

Heater-style: trial means with SEM error bars. Discovered is already on
the oracle-axis scale in ``metrics.csv`` (raw on the banana plus the
paired discovered-minus-raw gap). ``d=32`` has 8 scored trials.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 2:
        return float("nan")
    return float(x.std(ddof=1) / (len(x) ** 0.5))


def errorbar(ax, ok: pd.DataFrame, col: str, label: str, color: str, marker: str) -> None:
    rows = []
    for d, grp in ok.groupby("d", sort=True):
        y = pd.to_numeric(grp[col], errors="coerce").dropna()
        if y.empty:
            continue
        err = sem(y)
        rows.append((int(d), float(y.mean()), 0.0 if len(y) < 2 else err))
    if not rows:
        return
    ds, mu, err = zip(*rows)
    ax.errorbar(ds, mu, yerr=err, marker=marker, color=color, label=label, capsize=3)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-csv", type=Path, default=here / "metrics.csv")
    p.add_argument("--out-dir", type=Path, default=here)
    args = p.parse_args()
    df = pd.read_csv(args.in_csv)
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df
    if ok.empty:
        raise RuntimeError("no completed rows")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    errorbar(ax, ok, "raw_on_oracle_axes", "raw", "C0", "o")
    errorbar(ax, ok, "discovered_on_oracle_scale", "discovered", "C1", "s")
    errorbar(ax, ok, "oracle_on_oracle_axes", "oracle", "C2", "^")
    ax.set_xlabel("ambient dimension d")
    ax.set_ylabel("log-density on oracle axes (nats)")
    ax.set_title("Rosenbrock NPE at 2000 sims")
    ax.legend(frameon=False, fontsize=8)
    for d, grp in ok.groupby("d", sort=True):
        ax.text(int(d), ax.get_ylim()[1], f"n={len(grp)}",
                ha="center", va="bottom", fontsize=7, color="0.4")
    fig.tight_layout()
    fig.savefig(args.out_dir / "rosenbrock_npe_scaling.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "rosenbrock_npe_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_dir / 'rosenbrock_npe_scaling.png'} from {len(ok)} completed rows")


if __name__ == "__main__":
    main()
