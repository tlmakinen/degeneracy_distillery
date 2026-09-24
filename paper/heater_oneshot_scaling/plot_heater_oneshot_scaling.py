#!/usr/bin/env python
"""Redraw heater_oneshot_scaling.png from the packaged metrics.

Error bars are the standard error of trials with status ok. Trials that
failed before a posterior score are absent from metrics.csv.
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
        rows.append((int(d), float(grp[col].mean()), sem(grp[col])))
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
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    errorbar(ax, ok, "raw_on_discovered_marg", "raw on discovered axis", "C0", "o")
    errorbar(ax, ok, "discovered_on_discovered_marg", "discovered", "C1", "s")
    errorbar(ax, ok, "analytic_on_analytic_marg", "analytic (oracle axis)", "C2", "^")
    ax.set_xlabel("ambient dimension d")
    ax.set_ylabel("mean log-density on a scalar axis (nats)")
    ax.legend(frameon=False)
    ax.set_title("all completed trials (failed recoveries included)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "heater_oneshot_scaling.pdf")
    fig.savefig(args.out_dir / "heater_oneshot_scaling.png", dpi=150)
    plt.close(fig)
    print(f"wrote {args.out_dir / 'heater_oneshot_scaling.png'} from {len(ok)} completed rows")


if __name__ == "__main__":
    main()
