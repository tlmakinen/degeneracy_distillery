#!/usr/bin/env python
"""Aggregate one-step sweep metrics and draw the four-panel scaling figure.

Globs ``**/metrics.csv`` under ``--in-dir``. The figure is keyed off the
shared driver schema, so any problem swept over ``d`` works.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--problem", type=str, default=None)
    return p.parse_args()


def load_metrics(in_dir: Path) -> pd.DataFrame:
    paths = sorted(in_dir.glob("**/metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"no metrics.csv under {in_dir}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    keys = [c for c in ("problem", "arm", "d", "trial") if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last")
    return df


def sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 2:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def log_slope(d: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(d) & np.isfinite(y) & (d > 0) & (y > 0)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    x = np.log(d[mask])
    z = np.log(y[mask])
    coef, cov = np.polyfit(x, z, 1, cov=True) if mask.sum() > 2 else (
        np.polyfit(x, z, 1), np.array([[np.nan]])
    )
    se = float(np.sqrt(cov[0, 0])) if np.ndim(cov) == 2 else float("nan")
    return float(coef[0]), se


def summarise(ok: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    group_cols = [c for c in ("problem", "arm", "d") if c in ok.columns]
    for key, grp in ok.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        rec = {c: v for c, v in zip(group_cols, key)}
        rec["n"] = int(len(grp))
        for col in (
            "rank_margin", "sampling_band", "r2_true_min", "r2_sr_min",
            "nll_gap", "nll_neural", "runtime_fit_s", "runtime_total_s",
            "jtj_rank_agreement_k", "median_y_std",
        ):
            if col not in grp:
                continue
            rec[f"{col}_mean"] = float(pd.to_numeric(grp[col], errors="coerce").mean())
            rec[f"{col}_sem"] = sem(grp[col])
        if "rank_correct" in grp:
            rec["rank_correct_frac"] = float(grp["rank_correct"].astype(float).mean())
        if "screen_set_correct" in grp:
            rec["screen_set_correct_frac"] = float(
                grp["screen_set_correct"].astype(float).mean()
            )
        rows.append(rec)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def _xy(grp: pd.DataFrame, col: str):
    rows = []
    for d, g in grp.groupby("d", sort=True):
        rows.append((int(d), float(pd.to_numeric(g[col], errors="coerce").mean()), sem(g[col])))
    if not rows:
        return np.array([]), np.array([]), np.array([])
    d, mu, err = zip(*rows)
    return np.asarray(d), np.asarray(mu), np.asarray(err)


def draw_figure(ok: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(6.4, 11.2), sharex=True)
    arms = [a for a in ("oneshot", "threestep") if a in set(ok.get("arm", pd.Series(dtype=str)))]
    if not arms:
        arms = sorted(ok["arm"].unique()) if "arm" in ok.columns else ["oneshot"]
    colors = {"oneshot": "C0", "threestep": "C1"}
    markers = {"oneshot": "o", "threestep": "s"}

    ax = axes[0]
    oneshot = ok[ok["arm"] == "oneshot"] if "arm" in ok.columns else ok
    if {"rank_margin", "sampling_band"}.issubset(oneshot.columns) and not oneshot.empty:
        d, mu, err = _xy(oneshot, "rank_margin")
        _, band, _ = _xy(oneshot, "sampling_band")
        if d.size:
            ax.errorbar(d, mu, yerr=err, marker="o", color="C0", capsize=3, label="rank margin")
            ax.fill_between(d, 0.0, band, color="0.7", alpha=0.4, label="sampling band")
    ax.set_ylabel("rank margin (nats)")
    ax.legend(frameon=False)

    ax = axes[1]
    for arm in arms:
        sub = ok[ok["arm"] == arm] if "arm" in ok.columns else ok
        if "r2_true_min" in sub.columns:
            d, mu, err = _xy(sub, "r2_true_min")
            if d.size:
                ax.errorbar(
                    d, 1.0 - mu, yerr=err, marker=markers.get(arm, "o"),
                    color=colors.get(arm, None), capsize=3, label=f"{arm} neural",
                )
        if arm == "oneshot" and "r2_sr_min" in sub.columns:
            d, mu, err = _xy(sub, "r2_sr_min")
            if d.size:
                ax.errorbar(
                    d, 1.0 - mu, yerr=err, marker="^", color="C2",
                    capsize=3, label="oneshot SR",
                )
    ax.set_ylabel(r"$1 - R^2$")
    ax.set_yscale("log")
    ax.legend(frameon=False)

    ax = axes[2]
    if "nll_gap" in oneshot.columns and not oneshot.empty:
        d, mu, err = _xy(oneshot, "nll_gap")
        if d.size:
            ax.errorbar(d, mu, yerr=err, marker="o", color="C0", capsize=3)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_ylabel("NLL gap (nats)")

    ax = axes[3]
    for arm in arms:
        sub = ok[ok["arm"] == arm] if "arm" in ok.columns else ok
        col = "runtime_fit_s" if "runtime_fit_s" in sub.columns else "runtime_total_s"
        if col not in sub.columns:
            continue
        d, mu, err = _xy(sub, col)
        if not d.size:
            continue
        sl, se = log_slope(d, mu)
        label = f"{arm}  slope={sl:.2f}" if np.isfinite(sl) else arm
        ax.errorbar(
            d, mu, yerr=err, marker=markers.get(arm, "o"),
            color=colors.get(arm, None), capsize=3, label=label,
        )
    ax.set_ylabel("wall-clock fit (s)")
    ax.set_xlabel("ambient dimension d")
    ax.set_yscale("log")
    ax.legend(frameon=False)

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "oneshot_scaling.pdf")
    fig.savefig(out_dir / "oneshot_scaling.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(args.in_dir)
    if args.problem and "problem" in df.columns:
        df = df[df["problem"] == args.problem]
    ok = df[df["status"].isin(["ok", "success"])].copy() if "status" in df.columns else df
    if ok.empty:
        raise RuntimeError("no completed rows to plot")
    ok.to_csv(out_dir / "metrics_concat.csv", index=False)
    summary = summarise(ok, out_dir)
    draw_figure(ok, out_dir)
    print(summary.to_string(index=False))
    print(f"\nwrote tables and figure under {out_dir}")


if __name__ == "__main__":
    main()
