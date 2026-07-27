#!/usr/bin/env python
"""Aggregate per-seed run_record.json files into a rebuttal-ready summary.

Generic across experiments -- keyed off each record's "problem" field, not the
directory name -- so the same script works for rosenbrock, sir, gw_taylorf2,
gw_imrphenomd, qm7b, etc. once each gets the same run_record.json schema.

Usage:
    python scripts/aggregate_seed_sweep.py <run_dir> [<run_dir> ...]

Each <run_dir> should contain seed_*/run_record.json files, e.g.
follow_up_results/rosenbrock/rebuttal. Writes aggregate_summary.json and
aggregate_summary.md into each <run_dir>.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_records(run_dir: Path) -> list[dict]:
    records = []
    for seed_dir in sorted(run_dir.glob("seed_*")):
        record_path = seed_dir / "run_record.json"
        if not record_path.is_file():
            print(f"  warning: no run_record.json in {seed_dir}, skipping")
            continue
        with open(record_path) as handle:
            records.append(json.load(handle))
    return records


def median_iqr(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    values = sorted(values)
    median = statistics.median(values)
    if len(values) >= 4:
        q1 = statistics.median(values[: len(values) // 2])
        q3 = statistics.median(values[(len(values) + 1) // 2 :])
    else:
        q1 = q3 = median
    return {"median": median, "q1": q1, "q3": q3}


def _values(records: list[dict], *path: str) -> list[float]:
    out = []
    for r in records:
        node = r
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if isinstance(node, (int, float)):
            out.append(float(node))
    return out


def summarize(run_dir: Path) -> dict:
    records = load_records(run_dir)
    if not records:
        raise SystemExit(f"no run_record.json files found under {run_dir}")

    problem = records[0].get("problem", run_dir.name)
    n_total = len(records)

    successes = [
        r for r in records
        if r.get("status") == "success" and r.get("discovery", {}).get("success")
    ]
    # Two distinct non-success categories, deliberately not lumped together:
    # a threshold miss ran the full pipeline cleanly but didn't clear the
    # discovery criteria (legitimate, expected per-seed variance -- the doc's
    # whole point in reporting X/10 rather than claiming 10/10); a crash means
    # some stage raised an exception before the pipeline could even finish.
    # Conflating these in one "Failures" table risks a rebuttal reader
    # mistaking ordinary recovery-rate variance for a software bug.
    threshold_misses = [
        r for r in records
        if r.get("status") == "success" and not r.get("discovery", {}).get("success")
    ]
    crashes = [r for r in records if r.get("status") == "failed"]

    counts = records[0].get("counts", {})

    representative = None
    if successes:
        by_alignment = sorted(
            successes, key=lambda r: r["discovery"]["physics_alignment"]
        )
        representative = by_alignment[len(by_alignment) // 2]

    threshold_miss_details = [
        {
            "run_id": r.get("run_id"),
            "master_seed": r.get("master_seed"),
            "physics_alignment": r.get("discovery", {}).get("physics_alignment"),
        }
        for r in threshold_misses
    ]
    crash_details = [
        {
            "run_id": r.get("run_id"),
            "master_seed": r.get("master_seed"),
            "failure_stage": r.get("failure_stage"),
            "failure_reason": r.get("failure_reason"),
        }
        for r in crashes
    ]

    return {
        "problem": problem,
        "run_dir": str(run_dir),
        "n_total": n_total,
        "n_success": len(successes),
        "n_threshold_miss": len(threshold_misses),
        "n_crashed": len(crashes),
        "recovery_count": f"{len(successes)}/{n_total}",
        "counts": counts,
        "physics_alignment": median_iqr(_values(successes, "discovery", "physics_alignment")),
        "held_out_frob_symbolic": median_iqr(_values(successes, "heldout_geometry", "frob_symbolic")),
        "complexity_total": median_iqr(_values(successes, "discovery", "complexity_total")),
        "mdl_total": median_iqr(_values(successes, "discovery", "mdl_total")),
        "representative_run_id": representative.get("run_id") if representative else None,
        "representative_expressions": (
            representative.get("discovery", {}).get("expressions_physical") if representative else None
        ),
        "threshold_misses": threshold_miss_details,
        "crashes": crash_details,
    }


def format_iqr(stat: dict) -> str:
    if stat["median"] != stat["median"]:  # NaN check
        return "n/a"
    return f"{stat['median']:.3f} [{stat['q1']:.3f}, {stat['q3']:.3f}]"


def write_markdown(summary: dict, out_path: Path) -> None:
    lines = [f"# {summary['problem']}: seed-sweep summary", ""]
    lines.append(
        f"Across {summary['n_total']} independent end-to-end trials "
        f"({summary['counts'].get('n_train_simulations', '?')} training simulations each), "
        f"the pipeline recovered the expected coordinate in {summary['recovery_count']} trials. "
        f"Symbolic-regression augmentation used "
        f"{summary['counts'].get('n_augmented_coordinate_evaluations', '?')} inexpensive evaluations "
        f"of the learned coordinate map and required no additional simulator calls."
    )
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Recovery count | {summary['recovery_count']} |")
    lines.append(f"| Training simulations | {summary['counts'].get('n_train_simulations', '?')} |")
    lines.append(f"| Held-out evaluation simulations | {summary['counts'].get('n_eval_simulations', '?')} |")
    lines.append(
        f"| Augmented coordinate evaluations | "
        f"{summary['counts'].get('n_augmented_coordinate_evaluations', '?')} |"
    )
    lines.append(f"| Physics alignment (median [IQR]) | {format_iqr(summary['physics_alignment'])} |")
    lines.append(
        f"| Held-out geometric loss, symbolic (median [IQR]) | "
        f"{format_iqr(summary['held_out_frob_symbolic'])} |"
    )
    lines.append(f"| Expression complexity (median [IQR]) | {format_iqr(summary['complexity_total'])} |")
    lines.append(f"| Expression MDL (median [IQR]) | {format_iqr(summary['mdl_total'])} |")
    lines.append("")

    if summary["representative_run_id"]:
        lines.append(f"Representative trial (median-success, `{summary['representative_run_id']}`):")
        for expr in summary["representative_expressions"] or []:
            lines.append(f"- `{expr}`")
        lines.append("")

    if summary["threshold_misses"]:
        lines.append(
            f"## Threshold misses ({len(summary['threshold_misses'])}) "
            "-- ran cleanly, did not clear the discovery criteria"
        )
        lines.append("")
        lines.append("| run_id | seed | physics_alignment |")
        lines.append("|---|---|---|")
        for m in summary["threshold_misses"]:
            pa = m.get("physics_alignment")
            pa_str = f"{pa:.3f}" if isinstance(pa, (int, float)) else "-"
            lines.append(f"| {m['run_id']} | {m['master_seed']} | {pa_str} |")
        lines.append("")

    if summary["crashes"]:
        lines.append(f"## Crashes ({len(summary['crashes'])}) -- did not finish the pipeline")
        lines.append("")
        lines.append("| run_id | seed | stage | reason |")
        lines.append("|---|---|---|---|")
        for c in summary["crashes"]:
            lines.append(
                f"| {c['run_id']} | {c['master_seed']} | "
                f"{c.get('failure_stage', '-')} | {c.get('failure_reason', '-')} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        print(f"Aggregating {run_dir} ...")
        summary = summarize(run_dir)
        json_path = run_dir / "aggregate_summary.json"
        md_path = run_dir / "aggregate_summary.md"
        with open(json_path, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        write_markdown(summary, md_path)
        print(f"  {summary['recovery_count']} recovered; wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
