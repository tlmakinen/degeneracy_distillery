#!/usr/bin/env python
"""Aggregate the Rosenbrock NPE scaling outputs into tables and a figure.

Globs ``**/npe_metrics.csv`` under ``--in-dir`` (default:
``/data103/makinen/degeneracy_experiments/rosenbrock_npe_scaling``),
pairs raw vs oracle and raw vs discovered gaps within each trial before
aggregating, fits a slope in nats per dimension, and draws three panels::

    A  raw vs oracle on the banana axes vs d
    B  raw vs discovered on the discovered axes vs d
    C  paired gaps (oracle - raw, discovered - raw) with SEM vs d

Failures and non-finite scores are counted per ``d`` in the recovery
table but omitted from the means. Trials that failed discovery
(``rank_correct`` or ``symbolic_recovered`` False) stay in the counts
and are annotated in the figure. Modelled on
``scripts/plot_heater_oneshot_scaling.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORACLE_COLS = ("raw_on_oracle_axes", "oracle_on_oracle_axes")
DISC_COLS = ("raw_on_discovered_axes", "discovered_on_discovered_axes")
ALL_SCORE_COLS = ORACLE_COLS + DISC_COLS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in-dir", type=Path,
        default=Path("/data103/makinen/degeneracy_experiments/rosenbrock_npe_scaling"),
    )
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Defaults to <in-dir>/plots.")
    return p.parse_args()


def load_metrics(in_dir: Path) -> pd.DataFrame:
    paths = sorted(in_dir.glob("**/npe_metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"no npe_metrics.csv under {in_dir}")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if {"d", "trial"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["d", "trial"], keep="last")
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
    have_score = [c for c in ALL_SCORE_COLS if c in ok.columns]

    # Table 1: shared-axis log densities per d.
    rows_shared = []
    for d, grp in ok.groupby("d", sort=True):
        row = {"d": int(d), "n": int(len(grp))}
        for c in have_score:
            row[f"{c}_mean"] = float(grp[c].mean())
            row[f"{c}_sem"] = sem(grp[c])
        rows_shared.append(row)
    t1 = pd.DataFrame(rows_shared)
    t1.to_csv(out_dir / "table1_shared_axis.csv", index=False)

    # Table 2: paired gaps within trial.
    gap_rows = []
    if set(ALL_SCORE_COLS).issubset(ok.columns):
        g = ok.assign(
            gap_oracle=ok["oracle_on_oracle_axes"] - ok["raw_on_oracle_axes"],
            gap_discovered=(
                ok["discovered_on_discovered_axes"] - ok["raw_on_discovered_axes"]
            ),
        )
        for d, grp in g.groupby("d", sort=True):
            go = grp["gap_oracle"]
            gd = grp["gap_discovered"]
            frac = gd / go.replace(0, np.nan)
            gap_rows.append({
                "d": int(d),
                "n": int(len(grp)),
                "gap_oracle_mean": float(go.mean()),
                "gap_oracle_sem": sem(go),
                "gap_discovered_mean": float(gd.mean()),
                "gap_discovered_sem": sem(gd),
                "captured_fraction_mean": float(frac.mean()),
                "captured_fraction_sem": sem(frac),
            })
    t2 = pd.DataFrame(gap_rows)
    if not t2.empty:
        t2.to_csv(out_dir / "table2_gaps.csv", index=False)

    # Slopes vs d.
    slope_rows = []
    for c in have_score:
        sub = ok.dropna(subset=[c])
        sl, se = slope_with_se(
            sub["d"].to_numpy(float), sub[c].to_numpy(float),
        )
        slope_rows.append({
            "column": c,
            "slope_nats_per_dim": sl,
            "slope_se": se,
            "flat": bool(np.isfinite(se) and abs(sl) <= 2.0 * se),
        })
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(out_dir / "slopes.csv", index=False)

    # Table 3: native per-arm log-probs.
    native = [c for c in ("raw_log_prob", "oracle_log_prob", "discovered_log_prob")
              if c in ok.columns]
    rows_native = []
    for d, grp in ok.groupby("d", sort=True):
        row = {"d": int(d), "n": int(len(grp))}
        for c in native:
            row[f"{c}_mean"] = float(grp[c].mean())
            row[f"{c}_sem"] = sem(grp[c])
        rows_native.append(row)
    t3 = pd.DataFrame(rows_native)
    t3.to_csv(out_dir / "table3_native_logprob.csv", index=False)

    # Table 4: diagnostic in-box fractions per d.
    inbox_cols = [c for c in ("raw_inbox_frac", "oracle_inbox_frac",
                              "discovered_inbox_frac") if c in ok.columns]
    rows_inbox = []
    for d, grp in ok.groupby("d", sort=True):
        row = {"d": int(d), "n": int(len(grp))}
        for c in inbox_cols:
            row[f"{c}_mean"] = float(grp[c].mean())
            row[f"{c}_min"] = float(grp[c].min())
        rows_inbox.append(row)
    t4 = pd.DataFrame(rows_inbox)
    t4.to_csv(out_dir / "table4_inbox.csv", index=False)

    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "slopes": slopes}


def _errorbar(ax, ok: pd.DataFrame, col: str, label: str,
              color: str, marker: str, linestyle: str = "-") -> None:
    rows = []
    for d, grp in ok.groupby("d", sort=True):
        rows.append((int(d), float(grp[col].mean()), sem(grp[col])))
    if not rows:
        return
    ds, mu, err = zip(*rows)
    ax.errorbar(
        ds, mu, yerr=err, marker=marker, color=color,
        label=label, capsize=3, linestyle=linestyle,
    )


def draw_figure(ok: pd.DataFrame, out_dir: Path,
                all_rows: pd.DataFrame | None = None) -> None:
    """Three panels: oracle axes, discovered axes, paired gaps."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    need = set(ALL_SCORE_COLS)
    if not need.issubset(ok.columns):
        print("skipping figure: shared-axis columns missing")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))

    ax = axes[0]
    _errorbar(ax, ok, "raw_on_oracle_axes", "raw on oracle axes", "C0", "o")
    _errorbar(ax, ok, "oracle_on_oracle_axes", "oracle", "C2", "^")
    ax.set_xlabel("ambient dimension d")
    ax.set_ylabel("mean log-density on shared axes (nats)")
    ax.set_title("A  oracle banana axes")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    rec = ok
    if "symbolic_recovered" in ok.columns:
        rec = ok[ok["symbolic_recovered"].astype(bool)]
        fail = ok[~ok["symbolic_recovered"].astype(bool)]
    else:
        fail = ok.iloc[:0]
    if len(rec):
        _errorbar(ax, rec, "raw_on_discovered_axes", "raw on discovered", "C0", "o")
        _errorbar(ax, rec, "discovered_on_discovered_axes",
                  "discovered", "C1", "s")
    if len(fail) and "discovered_on_discovered_axes" in fail.columns:
        _errorbar(
            ax, fail, "discovered_on_discovered_axes",
            "discovered, not recovered", "0.55", "x", linestyle=":",
        )
    ax.set_xlabel("ambient dimension d")
    ax.set_title("B  discovered axes (recovered trials)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    if set(ALL_SCORE_COLS).issubset(ok.columns):
        gap_oracle = ok["oracle_on_oracle_axes"] - ok["raw_on_oracle_axes"]
        gap_disc = ok["discovered_on_discovered_axes"] - ok["raw_on_discovered_axes"]
        g = ok.assign(gap_oracle=gap_oracle, gap_discovered=gap_disc)
        _errorbar(ax, g, "gap_oracle", "oracle - raw", "C2", "^")
        _errorbar(ax, g, "gap_discovered", "discovered - raw", "C1", "s")
    ax.axhline(0.0, color="0.5", linestyle=":", linewidth=0.8)
    ax.set_xlabel("ambient dimension d")
    ax.set_ylabel("paired log-density gap (nats)")
    ax.set_title("C  paired gaps within trial")
    ax.legend(frameon=False, fontsize=8)

    denom = all_rows if all_rows is not None and not all_rows.empty else ok
    for d, grp in denom.groupby("d", sort=True):
        n_ok = int((grp["status"] == "ok").sum()) if "status" in grp else int(len(grp))
        n = int(len(grp))
        for ax_ in axes:
            ax_.text(int(d), ax_.get_ylim()[1], f"n={n_ok}/{n}",
                     ha="center", va="bottom", fontsize=7, color="0.4")

    fig.suptitle("Rosenbrock NPE scaling at 2000 sims", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "rosenbrock_npe_scaling.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "rosenbrock_npe_scaling.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def print_tables(tables: dict) -> None:
    t1, t2, slopes, t4 = tables["t1"], tables["t2"], tables["slopes"], tables["t4"]
    print("\nTable 1  shared-axis log densities")
    if not t1.empty:
        print(f"{'d':>4}  {'raw (or)':>16}  {'oracle':>16}  "
              f"{'raw (di)':>16}  {'discovered':>16}")
        for _, r in t1.iterrows():
            print(
                f"{int(r['d']):4d}  "
                f"{fmt(r.get('raw_on_oracle_axes_mean', np.nan), r.get('raw_on_oracle_axes_sem', np.nan)):>16}  "
                f"{fmt(r.get('oracle_on_oracle_axes_mean', np.nan), r.get('oracle_on_oracle_axes_sem', np.nan)):>16}  "
                f"{fmt(r.get('raw_on_discovered_axes_mean', np.nan), r.get('raw_on_discovered_axes_sem', np.nan)):>16}  "
                f"{fmt(r.get('discovered_on_discovered_axes_mean', np.nan), r.get('discovered_on_discovered_axes_sem', np.nan)):>16}"
            )
    if not t2.empty:
        print("\nTable 2  paired gaps")
        print(f"{'d':>4}  {'oracle - raw':>18}  {'discovered - raw':>18}  "
              f"{'captured':>10}")
        for _, r in t2.iterrows():
            print(
                f"{int(r['d']):4d}  "
                f"{fmt(r['gap_oracle_mean'], r['gap_oracle_sem']):>18}  "
                f"{fmt(r['gap_discovered_mean'], r['gap_discovered_sem']):>18}  "
                f"{r['captured_fraction_mean']:10.3f}"
            )
    print("\nSlopes (nats / dim)")
    for _, r in slopes.iterrows():
        print(
            f"  {r['column']:36s}  "
            f"{fmt(r['slope_nats_per_dim'], r['slope_se']):>16}  "
            f"flat={r['flat']}"
        )
    if not t4.empty:
        print("\nTable 4  in-box fractions")
        print(f"{'d':>4}  {'raw':>16}  {'oracle':>16}  {'disc':>16}")
        for _, r in t4.iterrows():
            print(
                f"{int(r['d']):4d}  "
                f"{r.get('raw_inbox_frac_mean', np.nan):8.3f} "
                f"(min {r.get('raw_inbox_frac_min', np.nan):5.3f})  "
                f"{r.get('oracle_inbox_frac_mean', np.nan):8.3f} "
                f"(min {r.get('oracle_inbox_frac_min', np.nan):5.3f})  "
                f"{r.get('discovered_inbox_frac_mean', np.nan):8.3f} "
                f"(min {r.get('discovered_inbox_frac_min', np.nan):5.3f})"
            )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.in_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(args.in_dir)
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df
    if ok.empty:
        raise RuntimeError("no completed rows to plot")
    ok.to_csv(out_dir / "metrics_concat.csv", index=False)
    tables = write_tables(ok, out_dir)
    draw_figure(ok, out_dir, all_rows=df)
    print_tables(tables)
    print(f"\nwrote tables and figure under {out_dir}")


if __name__ == "__main__":
    main()
