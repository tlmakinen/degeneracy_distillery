#!/usr/bin/env python
"""Aggregate per-task heater one-step sweep outputs into Tables 1-4.

Globs ``**/metrics.csv`` under ``--in-dir``, concatenates rows, pairs
gaps within trial before aggregating, fits a slope in nats per dimension
for each of the four common-axis columns, and draws the scaling figure
from the discovered axis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLS = [
    "raw_on_analytic_marg",
    "analytic_on_analytic_marg",
    "raw_on_discovered_marg",
    "discovered_on_discovered_marg",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--nsims", type=int, default=None)
    return p.parse_args()


def load_metrics(in_dir: Path) -> pd.DataFrame:
    paths = sorted(in_dir.glob("**/metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"no metrics.csv under {in_dir}")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if {"nsims", "d", "trial"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["nsims", "d", "trial"], keep="last")
    return df


def sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 2:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def slope_with_se(d: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(d) & np.isfinite(y)
    d, y = d[mask], y[mask]
    if d.size < 3:
        return float("nan"), float("nan")
    coef, cov = np.polyfit(d, y, 1, cov=True)
    return float(coef[0]), float(np.sqrt(cov[0, 0]))


def fmt(mean: float, err: float) -> str:
    if not np.isfinite(mean):
        return "nan"
    if not np.isfinite(err):
        return f"{mean:.3f}"
    return f"{mean:.3f} +- {err:.3f}"


def write_tables(ok: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    have = [c for c in COLS if c in ok.columns]

    # Table 1: common-axis log-probs
    t1_rows = []
    for d, grp in ok.groupby("d", sort=True):
        row = {"d": int(d), "n": int(len(grp))}
        for c in have:
            row[f"{c}_mean"] = float(grp[c].mean())
            row[f"{c}_sem"] = sem(grp[c])
        t1_rows.append(row)
    t1 = pd.DataFrame(t1_rows)
    t1.to_csv(out_dir / "table1_common_axis.csv", index=False)

    # Table 2: paired gaps within trial
    t2_rows = []
    if set(COLS).issubset(ok.columns):
        g = ok.assign(
            gap_analytic=ok["analytic_on_analytic_marg"] - ok["raw_on_analytic_marg"],
            gap_discovered=ok["discovered_on_discovered_marg"] - ok["raw_on_discovered_marg"],
        )
        for d, grp in g.groupby("d", sort=True):
            ga = grp["gap_analytic"]
            gd = grp["gap_discovered"]
            frac = gd / ga.replace(0, np.nan)
            t2_rows.append({
                "d": int(d),
                "n": int(len(grp)),
                "gap_analytic_mean": float(ga.mean()),
                "gap_analytic_sem": sem(ga),
                "gap_discovered_mean": float(gd.mean()),
                "gap_discovered_sem": sem(gd),
                "captured_fraction_mean": float(frac.mean()),
                "captured_fraction_sem": sem(frac),
            })
        t2 = pd.DataFrame(t2_rows)
        t2.to_csv(out_dir / "table2_gaps.csv", index=False)
    else:
        t2 = pd.DataFrame()

    # Slopes
    slope_rows = []
    for c in have:
        sub = ok.dropna(subset=[c])
        sl, se = slope_with_se(sub["d"].to_numpy(float), sub[c].to_numpy(float))
        slope_rows.append({
            "column": c,
            "slope_nats_per_dim": sl,
            "slope_se": se,
            "flat": bool(np.isfinite(se) and abs(sl) <= 2.0 * se),
        })
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(out_dir / "slopes.csv", index=False)

    # Table 3: native per-arm log-probs (not comparable across arms)
    native = [c for c in ("raw_log_prob", "analytic_log_prob", "discovered_log_prob")
              if c in ok.columns]
    t3_rows = []
    for d, grp in ok.groupby("d", sort=True):
        row = {"d": int(d), "n": int(len(grp))}
        for c in native:
            row[f"{c}_mean"] = float(grp[c].mean())
            row[f"{c}_sem"] = sem(grp[c])
        t3_rows.append(row)
    t3 = pd.DataFrame(t3_rows)
    t3.to_csv(out_dir / "table3_native_logprob.csv", index=False)

    # Table 4: recovery
    t4_rows = []
    for d, grp in ok.groupby("d", sort=True):
        n = int(len(grp))
        t4_rows.append({
            "d": int(d),
            "trials": n,
            "rank_correct": int(grp["rank_correct"].sum()) if "rank_correct" in grp else 0,
            "symbolic_recovered": (
                int(grp["symbolic_recovered"].sum()) if "symbolic_recovered" in grp else 0
            ),
            "median_abs_spearman": (
                float(grp["spearman_abs"].median()) if "spearman_abs" in grp else np.nan
            ),
            "median_complexity": (
                float(grp["complexity"].median()) if "complexity" in grp else np.nan
            ),
            "picks_agree": (
                int(grp["picks_agree"].sum()) if "picks_agree" in grp else 0
            ),
            "median_nll_gap": (
                float(grp["nll_gap"].median()) if "nll_gap" in grp else np.nan
            ),
        })
    t4 = pd.DataFrame(t4_rows)
    t4.to_csv(out_dir / "table4_recovery.csv", index=False)

    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "slopes": slopes}


def _errorbar(ax, ok: pd.DataFrame, col: str, label: str, color: str, marker: str) -> None:
    rows = []
    for d, grp in ok.groupby("d", sort=True):
        rows.append((int(d), float(grp[col].mean()), sem(grp[col])))
    ds, mu, err = zip(*rows)
    ax.errorbar(ds, mu, yerr=err, marker=marker, color=color, label=label, capsize=3)


def draw_figure(ok: pd.DataFrame, out_dir: Path, all_rows: pd.DataFrame | None = None) -> None:
    """Three panels: oracle NPE, recovered-only discovery, recovery rate.

    Averaging discovered-axis scores over failed recoveries hides the
    signal. The first figure file keeps the original all-trial overlay
    for the archive; the split figure is the one to read.
    """
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    need = {"raw_on_discovered_marg", "discovered_on_discovered_marg"}
    if not need.issubset(ok.columns):
        print("skipping figure: discovered-axis columns missing")
        return

    # Archive overlay (all ok trials). Same as before, so old links still work.
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    _errorbar(ax, ok, "raw_on_discovered_marg", "raw on discovered axis", "C0", "o")
    _errorbar(ax, ok, "discovered_on_discovered_marg", "discovered", "C1", "s")
    if "analytic_on_analytic_marg" in ok.columns:
        _errorbar(ax, ok, "analytic_on_analytic_marg", "analytic (oracle axis)", "C2", "^")
    ax.set_xlabel("ambient dimension d")
    ax.set_ylabel("mean log-density on a scalar axis (nats)")
    ax.legend(frameon=False)
    ax.set_title("all completed trials (failed recoveries included)")
    fig.tight_layout()
    fig.savefig(out_dir / "heater_oneshot_scaling.pdf")
    fig.savefig(out_dir / "heater_oneshot_scaling.png", dpi=150)
    plt.close(fig)

    rec = ok
    if "symbolic_recovered" in ok.columns:
        rec = ok[ok["symbolic_recovered"].astype(bool)]
    fail = ok
    if "symbolic_recovered" in ok.columns:
        fail = ok[~ok["symbolic_recovered"].astype(bool)]
    denom = all_rows if all_rows is not None and not all_rows.empty else ok

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))

    ax = axes[0]
    if {"raw_on_analytic_marg", "analytic_on_analytic_marg"}.issubset(ok.columns):
        _errorbar(ax, ok, "raw_on_analytic_marg", "raw on analytic axis", "C0", "o")
        _errorbar(ax, ok, "analytic_on_analytic_marg", "analytic (oracle)", "C2", "^")
    ax.set_xlabel("ambient dimension d")
    ax.set_ylabel("mean log-density (nats)")
    ax.set_title("A  oracle axis, all trials")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    if len(rec):
        _errorbar(ax, rec, "raw_on_discovered_marg", "raw, recovered trials", "C0", "o")
        _errorbar(ax, rec, "discovered_on_discovered_marg", "discovered, recovered", "C1", "s")
    if len(fail) and "discovered_on_discovered_marg" in fail.columns:
        rows = []
        for d, grp in fail.groupby("d", sort=True):
            rows.append((int(d), float(grp["discovered_on_discovered_marg"].mean()), sem(grp["discovered_on_discovered_marg"])))
        if rows:
            ds, mu, err = zip(*rows)
            ax.errorbar(
                ds, mu, yerr=err, marker="x", color="0.55",
                label="discovered, not recovered", capsize=3, linestyle=":",
            )
    ax.set_xlabel("ambient dimension d")
    ax.set_title("B  discovered axis, split by recovery")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    rows = []
    for d, grp in denom.groupby("d", sort=True):
        n = int(len(grp))
        n_rank = int(grp["rank_correct"].sum()) if "rank_correct" in grp else 0
        n_sr = int(grp["symbolic_recovered"].sum()) if "symbolic_recovered" in grp else 0
        rows.append((int(d), n_rank / n, n_sr / n, n))
    ds, rank_f, sr_f, ns = zip(*rows)
    ax.plot(ds, rank_f, marker="D", color="C3", label="rank $r=1$")
    ax.plot(ds, sr_f, marker="s", color="C1", label=r"symbolic $|\rho|\geq 0.99$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("ambient dimension d")
    ax.set_ylabel("fraction of all trials")
    ax.set_title("C  recovery (failures in denominator)")
    ax.legend(frameon=False, fontsize=8)
    for d, n in zip(ds, ns):
        ax.text(d, 1.0, f"n={n}", ha="center", va="bottom", fontsize=7, color="0.4")

    fig.suptitle("heater one-step discovery", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "heater_oneshot_scaling_split.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "heater_oneshot_scaling_split.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_tables(tables: dict) -> None:
    t1, t2, slopes, t4 = tables["t1"], tables["t2"], tables["slopes"], tables["t4"]
    print("\nTable 1  common-axis log-probs")
    if not t1.empty and set(COLS).issubset(
        {c.replace("_mean", "").replace("_sem", "") for c in t1.columns if c not in ("d", "n")}
        | set(COLS)
    ):
        print(f"{'d':>4}  {'raw (an)':>16}  {'analytic':>16}  "
              f"{'raw (di)':>16}  {'discovered':>16}")
        for _, r in t1.iterrows():
            print(
                f"{int(r['d']):4d}  "
                f"{fmt(r.get('raw_on_analytic_marg_mean', np.nan), r.get('raw_on_analytic_marg_sem', np.nan)):>16}  "
                f"{fmt(r.get('analytic_on_analytic_marg_mean', np.nan), r.get('analytic_on_analytic_marg_sem', np.nan)):>16}  "
                f"{fmt(r.get('raw_on_discovered_marg_mean', np.nan), r.get('raw_on_discovered_marg_sem', np.nan)):>16}  "
                f"{fmt(r.get('discovered_on_discovered_marg_mean', np.nan), r.get('discovered_on_discovered_marg_sem', np.nan)):>16}"
            )
    if not t2.empty:
        print("\nTable 2  paired gaps")
        print(f"{'d':>4}  {'analytic - raw':>18}  {'discovered - raw':>18}  {'captured':>10}")
        for _, r in t2.iterrows():
            print(
                f"{int(r['d']):4d}  "
                f"{fmt(r['gap_analytic_mean'], r['gap_analytic_sem']):>18}  "
                f"{fmt(r['gap_discovered_mean'], r['gap_discovered_sem']):>18}  "
                f"{r['captured_fraction_mean']:10.3f}"
            )
    print("\nSlopes (nats / dim)")
    for _, r in slopes.iterrows():
        print(f"  {r['column']:36s}  {fmt(r['slope_nats_per_dim'], r['slope_se']):>16}  "
              f"flat={r['flat']}")
    if not t4.empty:
        print("\nTable 4  recovery")
        print(f"{'d':>4}  {'rank':>8}  {'symbolic':>10}  {'|rho|':>8}  {'cx':>6}")
        for _, r in t4.iterrows():
            n = int(r["trials"])
            print(
                f"{int(r['d']):4d}  {int(r['rank_correct']):2d}/{n:<4d}  "
                f"{int(r['symbolic_recovered']):2d}/{n:<6d}  "
                f"{r['median_abs_spearman']:8.3f}  {r['median_complexity']:6.1f}"
            )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(args.in_dir)
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df
    if args.nsims is not None and "nsims" in ok.columns:
        ok = ok[ok["nsims"] == int(args.nsims)]
    if ok.empty:
        raise RuntimeError("no completed rows to plot")
    ok.to_csv(out_dir / "metrics_concat.csv", index=False)
    tables = write_tables(ok, out_dir)
    draw_figure(ok, out_dir, all_rows=df)
    print_tables(tables)
    print(f"\nwrote tables and figure under {out_dir}")


if __name__ == "__main__":
    main()
