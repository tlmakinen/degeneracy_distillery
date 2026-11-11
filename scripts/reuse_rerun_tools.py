#!/usr/bin/env python
"""Glue for the "discovery simulations reused in the NPE budget" rerun.

Two subcommands:

``exprs``      Turn a discovery driver's output directory into the
               ``--theta-to-eta`` strings the matching NPE sweep expects.
``summarize``  Collect one sweep's outputs into a per-budget table
               (``summary.csv`` / ``summary.json``).

See ``notes/reuse_rerun_results.md`` for the commands these were used with.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import sympy


# The GW sweeps train on masses MinMax-scaled from [5, 50] to [0.1, 1.0], which is
# exactly X = m / 50, and bind X1/X2 (and m1/m2) to those scaled columns.
GW_SCALED_PER_MASS = 50.0


PARAM_NAMES = {
    "sir": ("beta", "gamma", "I0_over_10"),
    "taylorf2": ("m1", "m2"),
    "imrphenomd": ("m1", "m2"),
}


def _parse(text: object, names: tuple[str, ...]) -> sympy.Expr:
    """Parse an SR expression string with the given names bound to plain symbols.

    Explicit locals matter: bare ``beta`` and ``gamma`` are special functions to
    sympy. The SR strings use ``^`` for powers.
    """
    local = {name: sympy.Symbol(name) for name in names}
    return sympy.sympify(str(text).replace("^", "**"), locals=local)


def sweep_expressions(problem: str, pruned_exprs: list, scaler_scale, scaler_min, sr_offset: float):
    """Exact discovered coordinates, rewritten in the sweep's input variables.

    The driver's final (pruned) expressions act on its own scaled inputs
    ``x_k = scale_k * theta_k + min_k (+ sr_offset)``. Substituting that scaler
    gives the coordinates as functions of physical theta with no rounding; the
    ``expressions_physical`` strings the driver prints are the same thing
    rounded to 3 decimals, which is lossy (see ``rounding_check`` in the output).
    The sweeps bind X1, X2, ... to physical theta (SIR) or to masses / 50 (GW).
    """
    n = len(pruned_exprs)
    driver_x = [sympy.Symbol(f"X{k + 1}") for k in range(n)]
    theta = [sympy.Symbol(f"_theta{k + 1}") for k in range(n)]
    sweep_x = [sympy.Symbol(f"X{k + 1}") for k in range(n)]
    per_sweep_unit = 1.0 if problem == "sir" else GW_SCALED_PER_MASS

    physical_fns, sweep_strings = [], []
    for text in pruned_exprs:
        expr = _parse(text, tuple(f"X{k + 1}" for k in range(n)))
        expr = expr.subs(
            {
                driver_x[k]: sympy.Float(float(scaler_scale[k]), 17) * theta[k]
                + sympy.Float(float(scaler_min[k]) + float(sr_offset), 17)
                for k in range(n)
            },
            simultaneous=True,
        )
        physical_fns.append(sympy.lambdify(theta, expr, modules="numpy"))
        in_sweep = expr.subs(
            {theta[k]: sympy.Float(per_sweep_unit, 17) * sweep_x[k] for k in range(n)},
            simultaneous=True,
        )
        sweep_strings.append(sympy.sstr(in_sweep, full_prec=True))
    # The SIR fishnets see (beta, gamma) only; I0/10 is carried through unchanged
    # as the third coordinate, as in the sweep's built-in default expressions.
    if problem == "sir":
        sweep_strings.append("X3")
    return sweep_strings, physical_fns


def prior_samples(problem: str, n: int = 200_000, seed: int = 0) -> np.ndarray:
    """Draws from the sweep's prior domain, in physical units (first len(exprs) columns)."""
    rng = np.random.default_rng(seed)
    if problem == "sir":
        theta = rng.uniform([0.1, 0.05], [1.0, 0.5], size=(n, 2))
        return theta[theta[:, 0] / theta[:, 1] >= 1.15]
    theta = rng.uniform(5.0, 50.0, size=(n, 2))
    return theta[theta[:, 0] >= theta[:, 1]]


def cmd_exprs(args: argparse.Namespace) -> None:
    import pickle

    discovery = Path(args.discovery_dir)
    record = json.loads((discovery / "run_record.json").read_text())
    pickles = sorted(discovery.glob("sr_results_*/sr_expressions.pkl"))
    if len(pickles) != 1:
        raise SystemExit(f"expected exactly one sr_results_*/sr_expressions.pkl under {discovery}")
    with open(pickles[0], "rb") as handle:
        sr = pickle.load(handle)

    names = PARAM_NAMES[args.problem]
    pruned = [str(e) for e in sr["pruned_exprs"]]
    sweep_strings, exact_fns = sweep_expressions(
        args.problem, pruned, sr["scaler_scale"], sr["scaler_min"], float(sr.get("sr_offset", 0.0))
    )

    # How far the driver's rounded "physical" printout is from the exact map.
    theta = prior_samples(args.problem)
    physical = record["discovery"]["expressions_physical"]
    rounding = []
    for k, (text, exact_fn) in enumerate(zip(physical, exact_fns)):
        rounded_fn = sympy.lambdify(
            [sympy.Symbol(name) for name in names], _parse(text, names), modules="numpy"
        )
        cols = [theta[:, j] for j in range(len(exact_fns))]
        pad = [np.ones(theta.shape[0])] * (len(names) - len(cols))
        with np.errstate(all="ignore"):
            exact = np.broadcast_to(np.asarray(exact_fn(*cols), dtype=float), (theta.shape[0],))
            rounded = np.broadcast_to(np.asarray(rounded_fn(*cols, *pad), dtype=float), (theta.shape[0],))
        both = np.isfinite(exact) & np.isfinite(rounded)
        rounding.append(
            {
                "component": k,
                "exact_nonfinite_fraction_on_prior": float((~np.isfinite(exact)).mean()),
                "rounded_nonfinite_fraction_on_prior": float((~np.isfinite(rounded)).mean()),
                "max_abs_difference": float(np.max(np.abs(exact[both] - rounded[both]))),
                "exact_range_on_prior": [float(np.nanmin(exact)), float(np.nanmax(exact))],
            }
        )

    payload = {
        "problem": args.problem,
        "discovery_dir": str(discovery),
        "expressions_pruned_driver_scaled": pruned,
        "expressions_canonical_mdl": record["discovery"].get("expressions_canonical"),
        "expressions_physical_rounded": physical,
        "driver_scaler": {
            "scale_": [float(v) for v in sr["scaler_scale"]],
            "min_": [float(v) for v in sr["scaler_min"]],
            "sr_offset": float(sr.get("sr_offset", 0.0)),
        },
        "sweep_input_convention": (
            "X1, X2, X3 = beta, gamma, I0/10 (physical)"
            if args.problem == "sir"
            else "X1, X2 = m1 / 50, m2 / 50 (the sweep's MinMax scaling of [5, 50] to [0.1, 1])"
        ),
        "sweep_theta_to_eta": sweep_strings,
        "rounding_check": rounding,
    }
    if args.out is not None:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def empirical_crps_batch(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS per point from posterior draws: E|X - y| - 0.5 E|X - X'|.

    samples: (n_samples, n_points); y: (n_points,). Non-finite draws are dropped
    per point. Same definition as ``sir_notebook_run.empirical_crps_batch``
    (kept separate so this tool does not import the JAX/SR discovery stack).
    """
    samples = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    crps = np.full(y.shape[0], np.nan)
    for j in range(y.shape[0]):
        col = samples[:, j]
        col = np.sort(col[np.isfinite(col)])
        n = col.size
        if n < 2:
            continue
        idx = np.arange(1, n + 1, dtype=np.float64)
        spread = 2.0 * np.sum((2.0 * idx - n - 1.0) * col) / (n * n)
        crps[j] = np.mean(np.abs(col - y[j])) - 0.5 * spread
    return crps


def pit_coverage_error(samples: np.ndarray, true_values: np.ndarray) -> tuple[np.ndarray, float]:
    """PIT of a scalar derived quantity and its mean |empirical CDF - uniform|.

    Same definition as ``sir_notebook_run.pit_coverage_error``.
    """
    samples = np.asarray(samples, dtype=np.float64)
    true_values = np.asarray(true_values, dtype=np.float64)
    pit = np.full(true_values.shape[0], np.nan)
    for j in range(true_values.shape[0]):
        col = samples[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            pit[j] = float((col < true_values[j]).sum()) / col.size
    return pit, calibration_error(pit)


def calibration_error(percentiles: np.ndarray) -> float:
    """Mean |sorted PIT - uniform quantile|: 0 for a calibrated marginal."""
    pit = np.sort(percentiles[np.isfinite(percentiles)])
    if pit.size == 0:
        return float("nan")
    uniform = (np.arange(1, pit.size + 1) - 0.5) / pit.size
    return float(np.mean(np.abs(pit - uniform)))


def central_coverage(percentiles: np.ndarray, level: float) -> float:
    """Fraction of truths inside the central ``level`` credible interval."""
    pit = percentiles[np.isfinite(percentiles)]
    if pit.size == 0:
        return float("nan")
    lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    return float(np.mean((pit >= lo) & (pit <= hi)))


def crossover_budget(
    budgets: list[int], theta_values: list[float], target: float, higher_is_better: bool
) -> dict[str, object]:
    """Smallest budget at which the theta arm reaches ``target`` (eta at the first budget).

    Linear interpolation in log N between the bracketing budgets.
    """
    sign = 1.0 if higher_is_better else -1.0
    gap = [sign * (v - target) for v in theta_values]
    if not all(np.isfinite(gap)):
        return {"status": "undefined", "estimate": None}
    if gap[0] >= 0:
        return {"status": "at_or_below_first_budget", "estimate": budgets[0]}
    for i in range(len(budgets) - 1):
        if gap[i] < 0 <= gap[i + 1]:
            frac = -gap[i] / (gap[i + 1] - gap[i])
            log_n = math.log(budgets[i]) + frac * (math.log(budgets[i + 1]) - math.log(budgets[i]))
            return {
                "status": "interpolated",
                "bracket": [budgets[i], budgets[i + 1]],
                "estimate": float(round(math.exp(log_n))),
            }
    return {"status": "not_reached", "estimate": None, "above": budgets[-1]}


def cmd_summarize(args: argparse.Namespace) -> None:
    sweep = Path(args.sweep_dir)
    manifest = json.loads((sweep / "manifest.json").read_text())["config"]
    metrics = pd.read_csv(sweep / "metrics.csv")
    budgets = sorted(int(n) for n in metrics["nsims"].unique())
    methods = [m for m in ("theta", "eta", "conventional_coordinates") if m in set(metrics["method"])]
    param_names = (
        ["beta", "gamma", "I0_over_10"] if args.problem == "sir" else ["m1_scaled", "m2_scaled"]
    )

    test_lp = pd.read_csv(sweep / "test_logprob.csv") if (sweep / "test_logprob.csv").exists() else None
    test_points = (
        np.load(sweep / "test_logprob_points.npz")
        if (sweep / "test_logprob_points.npz").exists()
        else None
    )
    coverage = np.load(sweep / "coverage_outputs.npz") if (sweep / "coverage_outputs.npz").exists() else None

    crps_fn = pit_fn = None
    if args.problem == "sir" and coverage is not None:
        crps_fn, pit_fn = empirical_crps_batch, pit_coverage_error

    rows = []
    for nsims in budgets:
        for method in methods:
            sel = metrics[(metrics["nsims"] == nsims) & (metrics["method"] == method)]
            row: dict[str, object] = {
                "nsims": nsims,
                "method": method,
                "n_ensemble_members": int(len(sel)),
                "val_logprob_mean": float(sel["best_validation_log_prob_theta_density"].mean()),
                "val_logprob_std": float(sel["best_validation_log_prob_theta_density"].std(ddof=1))
                if len(sel) > 1
                else float("nan"),
                "val_logprob_members": [float(v) for v in sel["best_validation_log_prob_theta_density"]],
                "best_epochs": [int(v) for v in sel["best_epoch"]],
            }
            if test_lp is not None:
                t = test_lp[(test_lp["nsims"] == nsims) & (test_lp["method"] == method)]
                if len(t):
                    row["test_logprob_mean"] = float(t["test_log_prob_theta_density_mean"].iloc[0])
                    row["test_logprob_sem"] = float(t["test_log_prob_theta_density_sem"].iloc[0])
                    row["test_logprob_median"] = float(t["test_log_prob_theta_density_median"].iloc[0])
                    row["test_n_finite"] = int(t["n_finite"].iloc[0])
            prefix = f"n{nsims}_{method}_coverage"
            if coverage is not None and f"{prefix}_percentiles" in coverage.files:
                pct = coverage[f"{prefix}_percentiles"]
                valid = coverage[f"{prefix}_valid_mask"]
                row["coverage_n_test"] = int(pct.shape[0])
                row["coverage_valid_sample_fraction"] = float(valid.mean())
                for k, name in enumerate(param_names):
                    row[f"calib_err_{name}"] = calibration_error(pct[:, k])
                    row[f"cov68_{name}"] = central_coverage(pct[:, k], 0.68)
                    row[f"cov95_{name}"] = central_coverage(pct[:, k], 0.95)
                if crps_fn is not None:
                    samples = coverage[f"{prefix}_posterior_samples"].astype(np.float64)
                    truth = coverage[f"{prefix}_theta_true"].astype(np.float64)
                    with np.errstate(all="ignore"):
                        r0_samples = samples[:, :, 0] / samples[:, :, 1]
                    r0_true = truth[:, 0] / truth[:, 1]
                    crps = crps_fn(r0_samples, r0_true)
                    _, pit_err = pit_fn(r0_samples, r0_true)
                    row["crps_r0_median"] = float(np.nanmedian(crps))
                    row["crps_r0_mean"] = float(np.nanmean(crps))
                    row["crps_r0_sem"] = float(
                        np.nanstd(crps, ddof=1) / np.sqrt(np.isfinite(crps).sum())
                    )
                    row["calib_err_r0"] = float(pit_err)
                    row["_crps_points"] = crps
            rows.append(row)

    # Paired eta - theta differences on the common test sets.
    paired = []
    for nsims in budgets:
        entry: dict[str, object] = {"nsims": nsims}
        if test_points is not None and f"n{nsims}_theta" in test_points.files:
            diff = test_points[f"n{nsims}_eta"] - test_points[f"n{nsims}_theta"]
            diff = diff[np.isfinite(diff)]
            entry["test_logprob_eta_minus_theta_mean"] = float(diff.mean())
            entry["test_logprob_eta_minus_theta_sem"] = float(diff.std(ddof=1) / np.sqrt(diff.size))
        by_method = {r["method"]: r for r in rows if r["nsims"] == nsims}
        if "_crps_points" in by_method.get("theta", {}):
            diff = by_method["eta"]["_crps_points"] - by_method["theta"]["_crps_points"]
            diff = diff[np.isfinite(diff)]
            entry["crps_r0_eta_minus_theta_mean"] = float(diff.mean())
            entry["crps_r0_eta_minus_theta_sem"] = float(diff.std(ddof=1) / np.sqrt(diff.size))
        v_theta, v_eta = by_method["theta"], by_method["eta"]
        entry["val_logprob_eta_minus_theta"] = float(v_eta["val_logprob_mean"] - v_theta["val_logprob_mean"])
        paired.append(entry)

    def series(method: str, key: str) -> list[float]:
        return [
            float(next(r for r in rows if r["nsims"] == n and r["method"] == method).get(key, float("nan")))
            for n in budgets
        ]

    crossover = {}
    for key, higher in (
        ("val_logprob_mean", True),
        ("test_logprob_mean", True),
        ("crps_r0_median", False),
        ("crps_r0_mean", False),
    ):
        theta_vals, eta_vals = series("theta", key), series("eta", key)
        if np.isfinite(eta_vals[0]):
            crossover[key] = crossover_budget(budgets, theta_vals, eta_vals[0], higher) | {
                "eta_at_first_budget": eta_vals[0],
                "theta_by_budget": dict(zip(map(str, budgets), theta_vals)),
            }

    for row in rows:
        row.pop("_crps_points", None)
    table = pd.DataFrame(rows)
    table.to_csv(sweep / "summary.csv", index=False)
    summary = {
        "problem": args.problem,
        "sweep_dir": str(sweep),
        "budgets": budgets,
        "theta_to_eta_expressions": manifest.get("theta_to_eta_expressions"),
        "eta_to_theta_mode": manifest.get("eta_to_theta_mode"),
        "training_pool": manifest.get("training_pool"),
        "paired_arm_seeds": manifest.get("paired_arm_seeds"),
        "rows": rows,
        "paired_eta_minus_theta": paired,
        "theta_budget_matching_eta_at_first_budget": crossover,
    }
    (sweep / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(table.drop(columns=["val_logprob_members", "best_epochs"]).to_string(index=False))
    print(json.dumps({"paired": paired, "crossover": crossover}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_exprs = sub.add_parser("exprs", help="discovery output dir -> sweep --theta-to-eta strings")
    p_exprs.add_argument("--problem", choices=["sir", "taylorf2", "imrphenomd"], required=True)
    p_exprs.add_argument("--discovery-dir", type=Path, required=True)
    p_exprs.add_argument("--out", type=Path, default=None)
    p_exprs.set_defaults(func=cmd_exprs)

    p_sum = sub.add_parser("summarize", help="sweep out-dir -> summary.csv / summary.json")
    p_sum.add_argument("--problem", choices=["sir", "taylorf2", "imrphenomd"], required=True)
    p_sum.add_argument("--sweep-dir", type=Path, required=True)
    p_sum.set_defaults(func=cmd_summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
