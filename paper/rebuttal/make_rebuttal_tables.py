#!/usr/bin/env python
"""Build metrics.csv and the comparison tables from the copied rebuttal records.

Reads only `records/` inside this folder, so the pack is self-contained and
needs no access to /data103.

Every row is scored twice, at two correlation thresholds:

* **`HEADLINE_THRESHOLD` (0.6), uniform across all experiments** -> the
  `recovered` column and `comparison_table.md`.
* **The pre-registered per-experiment thresholds** -> the `recovered_prereg`
  column and `comparison_table_prereg.md`.

Both are kept because the uniform bar was chosen after the results were seen,
which `notes/neurips_discovery_reruns.md` warns against; the pre-registered
table is the anchor that shows what moved. Second conjuncts are held at their
pre-registered values in both scorings -- only the correlation bar varies,
matching `scripts/recompute_success_at_threshold.py`.

Two further conventions are fixed here because the sources disagree:

* **Quartiles are `numpy.percentile` (linear interpolation) over ALL seeds.**
  `scripts/aggregate_seed_sweep.py` instead uses a Tukey split-halves rule over
  successes only, which is why the `aggregate_summary.md` files copied into
  `records/` differ from the published table in the third decimal. The
  convention used here reproduces the published three-step row for all four
  experiments exactly; `verify()` asserts that against the pre-registered
  scoring on every build.
* **Recovery is always recomputed**, never read from a record's
  `discovery.success`. The three-step and one-step drivers set that field by
  different rules.

    python paper/rebuttal/make_rebuttal_tables.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RECORDS = HERE / "records"

# Uniform bar applied across every experiment. See the module docstring for why
# the pre-registered scoring is kept alongside it rather than replaced.
HEADLINE_THRESHOLD = 0.6

# Mirrors scripts/recompute_success_at_threshold.py::EXPERIMENTS. Kept as data
# rather than imported so this folder stands alone.
# experiment -> (second conjunct field, pre-registered corr bar, second bar)
CRITERIA = {
    "rosenbrock":    ("complementary_linear_alignment", 0.5, 0.5),
    "sir":           (None, 0.5, None),
    "gw_taylorf2":   (None, 0.75, None),
    "gw_imrphenomd": ("complementary_mass_diff_alignment", 0.75, 0.5),
}

LABELS = {
    "rosenbrock": "Rosenbrock",
    "sir": "SIR",
    "gw_taylorf2": "GW TaylorF2",
    "gw_imrphenomd": "GW IMRPhenomD",
}

# The one-step Rosenbrock adapter's gates() returns only rank and screen
# booleans (degeneracy_distillery/problems/rosenbrock.py:117), so its
# physics_alignment falls through oneshot_sweep.py's fallback chain to
# r2_true_min -- an R^2, not the Pearson correlation the three-step reports
# (scripts/rosenbrock_notebook_run.py:731). Not comparable, and its
# complementary conjunct does not exist at all, so recovery is left blank.
NOT_COMPARABLE = {("one_step", "rosenbrock")}

VARIANTS = {
    "sir_info_floor_0p5": ("SIR, info-floor 0.5", "sir", "--info-floor 0.5"),
    "gw_imrphenomd_forced_m2": (
        "GW IMRPhenomD, forced m=2", "gw_imrphenomd", "--rank-min-gap 1e9",
    ),
    # Byte-identical config to the baseline array; the only change is the code
    # version, so this is a reproducibility replicate rather than an ablation.
    "gw_taylorf2_rerun_mdl": (
        "GW TaylorF2, replicate", "gw_taylorf2", "identical config, rerun",
    ),
}

COLUMNS = [
    "arm", "experiment", "variant", "seed", "master_seed", "status",
    "physics_alignment", "alignment_statistic", "comparable",
    "second_conjunct_name", "second_conjunct_value", "second_threshold",
    "threshold", "recovered", "recovered_mdl",
    "threshold_prereg", "recovered_prereg", "recovered_prereg_mdl",
    "physics_alignment_mdl", "second_conjunct_value_mdl",
    "r_hat", "m_fit",
    "n_train_simulations", "n_eval_simulations",
    "n_augmented_coordinate_evaluations",
    "complexity_total", "mdl_total", "nll_neural", "nll_symbolic",
    "r2_true_min", "r2_sr_min",
    "expression", "expression_mdl", "source_path",
]


def _num(v):
    """None/NaN -> '' so the CSV never carries a fake zero."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return ""
    return int(f) if f.is_integer() else f


def read_rows(arm: str, root: Path, experiment: str, variant: str = "") -> list[dict]:
    """One row per seed directory under ``root``, scored at both thresholds."""
    key = VARIANTS[variant][1] if variant else experiment
    second_name, prereg_thr, thr2 = CRITERIA[key]
    comparable = (arm, key) not in NOT_COMPARABLE

    rows = []
    for sd in sorted(root.glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
        rec_path = sd / "run_record.json"
        if not rec_path.exists():
            continue
        rec = json.loads(rec_path.read_text())
        d = rec.get("discovery") or {}
        counts = rec.get("counts") or {}

        align = d.get("physics_alignment")
        second = d.get(second_name) if second_name else None
        # The MDL pick and the frozen-NLL pick can select different coordinate
        # stacks, so score both. Records written before db4b146 have no *_mdl
        # fields at all; those cells stay blank rather than echoing the NLL pick.
        align_mdl = d.get("physics_alignment_mdl")
        second_mdl = d.get(second_name + "_mdl") if second_name else None
        if align_mdl is None and d.get("expression_mdl") and d.get("picks_agree"):
            align_mdl, second_mdl = align, second

        def score(a, b, bar):
            if not comparable or a is None:
                return ""
            ok = abs(float(a)) >= bar
            if second_name is not None:
                ok = ok and abs(float(b or 0.0)) >= thr2
            return int(bool(ok))

        rows.append({
            "arm": arm,
            "experiment": key,
            "variant": variant,
            "seed": int(sd.name.split("_")[1]),
            "master_seed": _num(rec.get("master_seed")),
            "status": rec.get("status", ""),
            "physics_alignment": _num(align),
            "alignment_statistic": (
                "pearson_abs_corr" if comparable else "r2_true_min"),
            "comparable": int(bool(comparable)),
            "second_conjunct_name": second_name or "",
            "second_conjunct_value": _num(second),
            "second_threshold": _num(thr2),
            "threshold": HEADLINE_THRESHOLD,
            "recovered": score(align, second, HEADLINE_THRESHOLD),
            "recovered_mdl": score(align_mdl, second_mdl, HEADLINE_THRESHOLD),
            "threshold_prereg": prereg_thr,
            "recovered_prereg": score(align, second, prereg_thr),
            "recovered_prereg_mdl": score(align_mdl, second_mdl, prereg_thr),
            "physics_alignment_mdl": _num(align_mdl),
            "second_conjunct_value_mdl": _num(second_mdl),
            "r_hat": _num(d.get("r_hat")),
            "m_fit": _num(d.get("m_fit")),
            "n_train_simulations": _num(counts.get("n_train_simulations")),
            "n_eval_simulations": _num(counts.get("n_eval_simulations")),
            "n_augmented_coordinate_evaluations": _num(
                counts.get("n_augmented_coordinate_evaluations")),
            "complexity_total": _num(d.get("complexity_total")),
            "mdl_total": _num(d.get("mdl_total")),
            "nll_neural": _num(d.get("nll_neural")),
            "nll_symbolic": _num(d.get("nll_symbolic")),
            "r2_true_min": _num(d.get("r2_true_min")),
            "r2_sr_min": _num(d.get("r2_sr_min")),
            "expression": d.get("expression") or " | ".join(
                d.get("expressions_physical") or []),
            "expression_mdl": d.get("expression_mdl") or "",
            "source_path": str(sd.relative_to(HERE)),
        })
    return rows


def collect() -> list[dict]:
    rows: list[dict] = []
    for exp in CRITERIA:
        rows += read_rows("three_step", RECORDS / "three_step" / exp, exp)
        rows += read_rows("one_step", RECORDS / "one_step" / exp, exp)
    for variant in VARIANTS:
        rows += read_rows(
            "one_step", RECORDS / "one_step_variants" / variant, "", variant)
    return rows


def med_iqr(values) -> str:
    """Median (q1,q3) by numpy.percentile over every seed. See module docstring."""
    v = np.array([float(x) for x in values if x != ""], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "n/a"
    return (f"{np.median(v):.3f} ({np.percentile(v, 25):.3f},"
            f"{np.percentile(v, 75):.3f})")


def recovery(rows, field: str = "recovered") -> str:
    scored = [r for r in rows if r[field] != ""]
    if not scored:
        return "n/a"
    return f"{sum(int(r[field]) for r in scored)}/{len(rows)}"


def build_tables(rows, field: str, mdl_field: str, heading: str, preamble: str) -> str:
    base = [r for r in rows if not r["variant"]]
    out = [heading, "", preamble, "",
           "| Experiment | Train sims | Three-step recovered | Three-step alignment "
           "| One-step recovered | One-step alignment |",
           "|---|---|---|---|---|---|"]

    footnote = False
    for exp in CRITERIA:
        cells, nsims = [], ""
        for arm in ("three_step", "one_step"):
            sel = [r for r in base if r["experiment"] == exp and r["arm"] == arm]
            nsims = nsims or (sel[0]["n_train_simulations"] if sel else "")
            mark = "" if sel and sel[0]["comparable"] else " *"
            footnote = footnote or bool(mark)
            cells += [recovery(sel, field) + mark,
                      med_iqr([r["physics_alignment"] for r in sel]) + mark]
        out.append(f"| {LABELS[exp]} | {nsims} | " + " | ".join(cells) + " |")

    if footnote:
        out += ["", "`*` The one-step Rosenbrock arm reports `r2_true_min` (an R^2 of "
                    "the true coordinates on a cubic of the recovered ones), not the "
                    "Pearson correlation the three-step arm reports. The two alignment "
                    "cells are different statistics and must not be compared. Its "
                    "complementary conjunct `complementary_linear_alignment` is absent "
                    "from the one-step record, so recovery cannot be scored on the "
                    "published criterion either."]

    out += ["", "## Config variants", "",
            "Deliberate config changes, not reruns of the frozen baseline. Listed "
            "separately so the headline table stays like-for-like.", "",
            "| Variant | Change | Recovered (NLL pick) | Recovered (MDL pick) "
            "| Alignment |", "|---|---|---|---|---|"]
    for variant, (label, _exp, change) in VARIANTS.items():
        sel = [r for r in rows if r["variant"] == variant]
        out.append(f"| {label} | `{change}` | {recovery(sel, field)} | "
                   f"{recovery(sel, mdl_field)} | "
                   f"{med_iqr([r['physics_alignment'] for r in sel])} |")
    out += ["", "The MDL column reads `n/a` for the baseline one-step trees that "
                "predate `db4b146`, when `expression_mdl` was computed and then "
                "dropped before the record was written."]
    return "\n".join(out) + "\n"


PUBLISHED = {
    "rosenbrock":    ("10/10", "0.929 (0.814,0.960)"),
    "sir":           ("10/10", "0.674 (0.636,0.741)"),
    "gw_taylorf2":   ("10/10", "0.972 (0.966,0.979)"),
    "gw_imrphenomd": ("7/10",  "0.997 (0.988,0.999)"),
}


def verify(rows) -> None:
    """The pre-registered three-step column must reproduce the published table."""
    bad = []
    for exp, want in PUBLISHED.items():
        sel = [r for r in rows if r["experiment"] == exp
               and r["arm"] == "three_step" and not r["variant"]]
        got = (recovery(sel, "recovered_prereg"),
               med_iqr([r["physics_alignment"] for r in sel]))
        if got != want:
            bad.append(f"  {exp}: got {got}, published {want}")
    n3 = len([r for r in rows if r["arm"] == "three_step"])
    n1 = len([r for r in rows if r["arm"] == "one_step" and not r["variant"]])
    nv = len([r for r in rows if r["variant"]])
    if (n3, n1) != (40, 40) or not 20 <= nv <= 30:
        bad.append(f"  row counts: three_step={n3} one_step={n1} variants={nv}, "
                   "expected 40/40 and 20-30 variant rows")
    if bad:
        raise SystemExit("verification FAILED:\n" + "\n".join(bad))
    print(f"verified: pre-registered three-step column matches the published "
          f"table; rows {n3}/{n1}/{nv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    rows = collect()
    with (HERE / "metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    (HERE / "comparison_table.md").write_text(build_tables(
        rows, "recovered", "recovered_mdl",
        "# Rebuttal reruns: three-step vs one-step",
        f"Ten independent trials per experiment per arm, 500 training simulations "
        f"each. Recovery uses a **uniform correlation threshold of "
        f"{HEADLINE_THRESHOLD}** across all four experiments, with each "
        f"experiment's second conjunct held at its pre-registered value. "
        f"Alignment is median (q1,q3) by `numpy.percentile` over all ten seeds. "
        f"See `comparison_table_prereg.md` for the same data at the "
        f"pre-registered per-experiment thresholds."))

    (HERE / "comparison_table_prereg.md").write_text(build_tables(
        rows, "recovered_prereg", "recovered_prereg_mdl",
        "# Rebuttal reruns at the pre-registered thresholds",
        "The same records scored at each experiment's pre-registered "
        "correlation bar (Rosenbrock 0.5, SIR 0.5, GW TaylorF2 0.75, "
        "GW IMRPhenomD 0.75) rather than the uniform "
        f"{HEADLINE_THRESHOLD} used in `comparison_table.md`. This is the "
        "anchor: its three-step column reproduces the published rebuttal table "
        "exactly, and the build asserts that."))

    print(f"wrote metrics.csv ({len(rows)} rows), comparison_table.md "
          f"(uniform {HEADLINE_THRESHOLD}) and comparison_table_prereg.md")
    if not args.no_verify:
        verify(rows)


if __name__ == "__main__":
    main()
