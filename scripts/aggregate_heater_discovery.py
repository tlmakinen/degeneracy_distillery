#!/usr/bin/env python
"""Build the rebuttal tables from a heater discovery sweep.

Phase 3 runs as a SLURM array with one task per ambient dimension, because
``heater_discovery_dim_scaling_sweep.py`` rewrites ``metrics.csv`` after every
run and concurrent tasks sharing a directory would clobber each other. So the
sweep output is one ``metrics.csv`` per ``d`` rather than the single file the
notes' snippet assumes. This script globs them, concatenates, and produces the
four tables from ``notes/heater_discovery_scaling.md``.

    python scripts/aggregate_heater_discovery.py \
        /data103/makinen/degeneracy_experiments/heater_discovery/scaling_v1

Tables 1 and 2 are the headline and are built from the **common-axis** columns.
The per-arm ``*_log_prob`` columns are NOT comparable across arms -- the raw
arm's target is a density on R^d, so its log-prob carries d-dependent units and
would fall with d even for a perfect estimator -- and appear only as Table 3,
marked appendix-only.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

COMMON_AXIS = [
    "raw_on_analytic_marg", "analytic_on_analytic_marg",
    "raw_on_discovered_marg", "discovered_on_discovered_marg",
    "raw_on_fisher_linear_marg", "fisher_linear_on_fisher_linear_marg",
]
NATIVE = ["raw_log_prob", "analytic_log_prob", "discovered_log_prob",
          "fisher_linear_log_prob"]


def load(root: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(root, "**", "metrics.csv"), recursive=True))
    if not paths:
        sys.exit(f"no metrics.csv found under {root}")
    frames = []
    for p in paths:
        try:
            f = pd.read_csv(p)
        except Exception as exc:                      # a task killed mid-write
            print(f"  ! skipping unreadable {p}: {exc}", file=sys.stderr)
            continue
        f["_source"] = os.path.relpath(os.path.dirname(p), root)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)

    # A requeued task can re-emit a row it already wrote, so (nsims, d, trial)
    # collisions *within one output directory* are duplicates and the last wins.
    # Collisions ACROSS directories are not: Phase 3 gives each d its own
    # directory, so a cross-directory collision means the root spans two
    # different configurations and silently keeping one would drop half the
    # data. Warn loudly instead of quietly deduplicating.
    key = ["nsims", "d", "trial"]
    if set(key).issubset(df.columns):
        df = df.drop_duplicates(subset=key + ["_source"], keep="last")
        clash = df.duplicated(subset=key, keep=False)
        if clash.any():
            srcs = sorted(df.loc[clash, "_source"].unique())
            print(
                f"  ! WARNING: {int(clash.sum())} rows share (nsims,d,trial) across"
                f" different directories: {srcs}.\n"
                "    That means this root mixes configurations rather than holding"
                " one sweep.\n"
                "    Keeping ALL rows -- aggregate each configuration separately"
                " instead.",
                file=sys.stderr,
            )
    print(f"loaded {len(df)} rows ({n_raw} before dedup) from "
          f"{len(paths)} metrics.csv files")
    return df


def slopes(ok: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Straight-line fit in d, so 'stays flat' is measured, not eyeballed.

    A single slope is only an honest summary of a *monotone* curve. The raw arm
    here is not monotone -- it improves to d=4 (SNR grows with d on this prior,
    since E[log theta] > 0 means P grows) and then falls as the cost of an
    R^d target overtakes that gain. Fitting one line across the rise and the
    fall averages them to ~0 and reports "flat", which is exactly wrong. So
    flag non-monotonicity and also report the slope over the descending tail.
    """
    rows = []
    for c in cols:
        if c not in ok:
            continue
        sub = ok.dropna(subset=[c])
        if sub["d"].nunique() < 3:
            rows.append({"arm": c, "slope_nats_per_dim": np.nan, "sem": np.nan,
                         "verdict": "insufficient d"})
            continue
        fit, cov = np.polyfit(sub["d"], sub[c], 1, cov=True)
        s, se = float(fit[0]), float(np.sqrt(cov[0, 0]))

        # Per-d means, to check the curve actually has one sign of slope.
        m = sub.groupby("d")[c].mean().sort_index()
        diffs = np.diff(m.values)
        monotone = bool(np.all(diffs >= 0) or np.all(diffs <= 0))

        row = {"arm": c, "slope_nats_per_dim": round(s, 4), "sem": round(se, 4),
               "monotone": "yes" if monotone else "NO"}
        if monotone:
            row["verdict"] = "flat" if abs(s) < 2 * se else (
                "rising" if s > 0 else "degrading")
        else:
            # Refit from the peak onward: that tail is the part the scaling
            # claim is actually about.
            d_peak = int(m.idxmax())
            tail = sub[sub["d"] >= d_peak]
            if tail["d"].nunique() >= 3:
                f2, c2 = np.polyfit(tail["d"], tail[c], 1, cov=True)
                row["tail_slope_from_d"] = d_peak
                row["tail_slope"] = round(float(f2[0]), 4)
                row["tail_sem"] = round(float(np.sqrt(c2[0, 0])), 4)
            row["verdict"] = (
                f"NON-MONOTONE (peak at d={d_peak}) - single slope is "
                "misleading, quote per-d values"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="sweep root containing per-d subdirectories")
    ap.add_argument("--out", default=None, help="write CSVs here (default: root)")
    args = ap.parse_args()
    out = args.out or args.root
    os.makedirs(out, exist_ok=True)

    df = load(args.root)
    ok = df[df["status"] == "ok"] if "status" in df else df

    print(f"\ncompleted {len(ok)} / {len(df)} runs")
    if "failed_stage" in df:
        fs = df["failed_stage"].dropna()
        if len(fs):
            print("failures by stage:", dict(fs.value_counts()))

    have = [c for c in COMMON_AXIS if c in ok]
    if have:
        print("\n=== Table 1 (headline): common-axis log-prob vs d ===")
        print("mean +- sem over trials; all entries are densities on a single")
        print("SCALAR axis, so units match across arms and 'flat' is meaningful.")
        t1 = ok.groupby("d")[have].agg(["mean", "sem"]).round(3)
        print(t1.to_string())
        t1.to_csv(os.path.join(out, "table1_common_axis.csv"))

    if {"analytic_on_analytic_marg", "raw_on_analytic_marg"}.issubset(ok.columns):
        print("\n=== Table 2: within-axis gaps (units cancel exactly) ===")
        # Paired within trial before aggregating: the raw and coordinate arms in
        # a given trial share simulations and seed, so differencing per-d means
        # instead would discard the pairing and inflate the error bars.
        assign = dict(
            gap_analytic=ok["analytic_on_analytic_marg"] - ok["raw_on_analytic_marg"],
            gap_discovered=(ok["discovered_on_discovered_marg"]
                            - ok["raw_on_discovered_marg"]),
        )
        gap_cols = ["gap_analytic", "gap_discovered"]
        # The fisher-linear arm is deterministic and never fails, so its gap is
        # defined at every d -- including where the discovered arm has no valid
        # coordinate.
        if {"fisher_linear_on_fisher_linear_marg",
                "raw_on_fisher_linear_marg"}.issubset(ok.columns):
            assign["gap_fisher_linear"] = (
                ok["fisher_linear_on_fisher_linear_marg"]
                - ok["raw_on_fisher_linear_marg"])
            gap_cols.append("gap_fisher_linear")
        g = ok.assign(**assign)
        t2 = g.groupby("d")[gap_cols].agg(["mean", "sem"]).round(3)
        t2[("captured_fraction", "")] = (
            t2[("gap_discovered", "mean")] / t2[("gap_analytic", "mean")]
        ).round(3)
        print(t2.to_string())
        print("\ncaptured_fraction = share of the oracle's advantage that the")
        print("discovered coordinate actually captures. Read alongside Table 4:")
        print("it is only meaningful at d where recovery actually succeeded.")
        t2.to_csv(os.path.join(out, "table2_gaps.csv"))

    if have:
        print("\n=== Slopes in nats per dimension ===")
        sl = slopes(ok, have)
        print(sl.to_string(index=False))
        sl.to_csv(os.path.join(out, "slopes.csv"), index=False)

    nat = [c for c in NATIVE if c in ok]
    if nat:
        print("\n=== Table 3 (APPENDIX ONLY): native per-arm log-probs ===")
        print("NOT comparable across arms -- different target dimensionality.")
        print("Readable down a column, never across a row.")
        t3 = ok.groupby("d")[nat].agg(["mean", "sem"]).round(3)
        print(t3.to_string())
        t3.to_csv(os.path.join(out, "table3_native_appendix.csv"))

    print("\n=== Table 4: recovery rates (failures kept in the denominator) ===")
    rows = []
    for d, grp in df.groupby("d"):
        g_ok = grp[grp["status"] == "ok"] if "status" in grp else grp
        r = {"d": int(d), "trials": len(grp), "completed": len(g_ok)}
        if "rank_correct" in g_ok:
            r["rank_correct"] = f"{int(g_ok['rank_correct'].sum())}/{len(grp)}"
        if "symbolic_recovered" in g_ok:
            r["spearman_recovered"] = f"{int(g_ok['symbolic_recovered'].sum())}/{len(grp)}"
        if "levelset_recovered" in g_ok:
            r["levelset_recovered"] = f"{int(g_ok['levelset_recovered'].sum())}/{len(grp)}"
        for k, c in [("median_rho", "spearman_abs"),
                     ("median_levelset_U", "levelset_unexplained"),
                     ("median_U_linear_baseline", "levelset_unexplained_linear"),
                     ("median_alignment_score", "alignment_score"),
                     ("median_complexity", "complexity")]:
            if c in g_ok and g_ok[c].notna().any():
                r[k] = round(float(g_ok[c].median()), 4)
        if "failed_stage" in grp:
            r["failed_stages"] = ";".join(sorted(set(grp["failed_stage"].dropna())))
        rows.append(r)
    t4 = pd.DataFrame(rows)
    print(t4.to_string(index=False))
    t4.to_csv(os.path.join(out, "table4_recovery.csv"), index=False)
    print("\nA flat discovered-arm curve is only meaningful at d where recovery")
    print("succeeded. Mark cells above a failing d rather than dropping them.")

    rt = [c for c in df.columns if c.startswith("runtime_") and c.endswith("_s")]
    if rt:
        print("\n=== Per-stage runtime (seconds, median over runs) ===")
        print(df[rt].median().round(1).to_string())
    if {"nsims", "n_test"}.issubset(df.columns):
        calls = df["nsims"] + df["n_test"]
        print(f"\nsimulator calls per run = nsims + n_test = {sorted(set(calls))}")
    elif "nsims" in df:
        print(f"\nnsims per run = {sorted(set(df['nsims']))} "
              "(+ n_test; discovery runs on the fishnet held-out set)")
    if "n_augmented_coordinate_evaluations" in df:
        print("augmented SR coordinate evaluations (network evals, NOT "
              f"simulations): {sorted(set(df['n_augmented_coordinate_evaluations'].dropna()))}")

    print(f"\nwrote tables to {out}")


if __name__ == "__main__":
    main()
