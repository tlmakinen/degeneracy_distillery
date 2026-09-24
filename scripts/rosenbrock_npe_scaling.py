#!/usr/bin/env python
"""NPE scaling for the Rosenbrock discovery grid, at a fixed sim budget.

For each ``(d, trial)`` the discovery driver already produced a symbolic
stack (``expression`` in ``metrics.csv``). This script trains three MAF NPE
arms at the discovery budget of 2000 simulations::

    raw          MAF over the full ``theta`` in ``[-3, 3]^d``.
    discovered   MAF over the trial's discovered symbolic stack,
                 standardised on training draws (``m_fit`` = 2 or 3 dims).
    oracle       MAF over the true banana coordinates
                 ``(theta_I0, theta_I1 - theta_I0^2)``, standardised.

The same estimator class in every arm keeps the comparison about the
coordinate, not the estimator. Simulations are regenerated from the same
seed as the discovery driver so the training data are identical.

Scoring extends the heater's ``marginal_log_probs_on_axis`` from a scalar
axis to ``k`` axes: draw posterior samples per validation observation,
project them onto the shared axes, fit a k-D Gaussian (mean, covariance),
score ``log N(true | mu, Sigma)``. Two comparisons per trial::

    raw vs oracle       on the true banana axes (2 dims)
    raw vs discovered   on the discovered axes (``m_fit`` dims, per trial)

MAF sampling bypasses ``LampeNPE.sample`` -- that method silently returns
prior samples when accept-reject against the prior box "takes too long",
and at high ``d`` the raw arm hits that branch. Every arm samples its
member flows directly, splitting draws by the ensemble weights, and each
arm reports its own ``*_inbox_frac`` diagnostic.

The driver reads discovery outputs from
``/data103/makinen/degeneracy_experiments/rosenbrock_oneshot_scaling_inforank/oneshot``
(configurable), writes one ``npe_metrics.csv`` per ``(d, trial)``, and
supports ``--resume``. Existing rows with ``status in {ok}`` are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

# Local imports: pull the NPE plumbing from the heater discovery sweep so
# the estimator, standardiser and prior conventions are identical to that
# experiment. compile_jax_stack from the oneshot driver handles the
# Rosenbrock expression grammar (constants inline, ``pow`` allowed).
import heater_discovery_dim_scaling_sweep as heater_base  # noqa: E402
import oneshot_sweep as oneshot  # noqa: E402


# =============================================================================
# Sampling and scoring helpers
# =============================================================================


def _torch():
    import torch
    return torch


def sample_from_ensemble(
    posterior: Any, x_obs: np.ndarray, n_samples: int, device: str,
) -> np.ndarray:
    """Direct flow sampling from a LampeEnsemble, split by ensemble weights.

    LampeNPE.sample rejects samples outside the prior box and, once the
    accepted-per-batch count drops to zero, returns *prior samples* with
    only a warning. On raw arms at high ``d`` this fires immediately.
    Sampling the flow directly is exactly the accept step without the
    rejection.
    """
    torch = _torch()
    posts = list(posterior.posteriors)
    w = np.asarray(
        posterior.weights.detach().cpu().numpy(), dtype=np.float64,
    )
    w = w / max(float(w.sum()), 1e-12)
    per = np.floor(n_samples * w).astype(int)
    remainder = int(n_samples - per.sum())
    if remainder > 0:
        order = np.argsort(-w)
        for k in range(remainder):
            per[order[k % len(per)]] += 1
    with torch.no_grad():
        x_t = torch.as_tensor(np.asarray(x_obs, dtype=np.float32), device=device)
        parts: list[np.ndarray] = []
        for npe, n in zip(posts, per):
            if int(n) <= 0:
                continue
            raw = npe.flow(x_t).sample((int(n),))
            th = npe.theta_transform(raw)
            parts.append(th.detach().cpu().numpy())
    if not parts:
        return np.empty((0,), dtype=np.float64)
    out = np.concatenate(parts, axis=0)
    if out.ndim == 1:
        out = out[:, None]
    return out.astype(np.float64)


def gaussian_matched_logp(
    vals: np.ndarray, truth_k: np.ndarray, ridge: float = 1e-8,
) -> float:
    """k-D Gaussian-matched log density on the shared axes.

    Fits mean and covariance to the (projected) posterior samples, adds a
    small trace-scaled ridge for numerical stability, and evaluates
    ``log N(truth_k | mu, Sigma)``. Returns ``nan`` if not enough finite
    samples are available or the covariance is singular.
    """
    v = np.asarray(vals, dtype=np.float64)
    if v.ndim == 1:
        v = v[:, None]
    truth_k = np.asarray(truth_k, dtype=np.float64).reshape(-1)
    k = int(v.shape[1])
    mask = np.isfinite(v).all(axis=1)
    v = v[mask]
    if v.shape[0] < max(4, k + 2):
        return float("nan")
    mu = v.mean(axis=0)
    if k == 1:
        var = float(v.var()) + ridge
        if not np.isfinite(var) or var <= 0.0:
            return float("nan")
        diff = float(truth_k[0]) - float(mu[0])
        return float(-0.5 * (np.log(2.0 * np.pi) + np.log(var)) - 0.5 * diff * diff / var)
    Sigma = np.cov(v.T)
    if Sigma.shape != (k, k):
        return float("nan")
    trace_scale = float(max(np.trace(Sigma) / max(k, 1), 1.0))
    Sigma = Sigma + ridge * trace_scale * np.eye(k)
    try:
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0 or not np.isfinite(logdet):
            return float("nan")
        diff = truth_k - mu
        sol = np.linalg.solve(Sigma, diff)
        return float(
            -0.5 * k * np.log(2.0 * np.pi)
            - 0.5 * logdet
            - 0.5 * float(diff @ sol)
        )
    except np.linalg.LinAlgError:
        return float("nan")


def inbox_fraction(samples: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Fraction of ``samples`` inside the closed box ``[low, high]``."""
    s = np.asarray(samples, dtype=np.float64)
    if s.size == 0:
        return float("nan")
    if s.ndim == 1:
        s = s[:, None]
    lo = np.asarray(low, dtype=np.float64).reshape(1, -1)
    hi = np.asarray(high, dtype=np.float64).reshape(1, -1)
    inside = np.all((s >= lo) & (s <= hi), axis=1)
    return float(inside.mean())


# =============================================================================
# Discovered coordinate
# =============================================================================


def stack_from_expression(expr: str) -> list[str]:
    """Split the discovery-driver's ``' | '`` joined expression back into a stack."""
    return [s.strip() for s in str(expr).split(" | ") if s.strip()]


def build_eta_projector(expr: str, d: int) -> Callable[[np.ndarray], np.ndarray]:
    """Compile the discovered stack into a vectorised ``theta -> (n, m_fit)``.

    ``compile_jax_stack`` from the one-step driver already understands the
    Rosenbrock expression grammar (constants inline, ``pow``, ``sqrt``);
    we call it once and vectorise the per-sample call ourselves so it can
    be applied to a matrix of posterior draws without recompilation.
    """
    exprs = stack_from_expression(expr)
    if not exprs:
        raise ValueError(f"empty expression stack: {expr!r}")
    fn = oneshot.compile_jax_stack(exprs, int(d))

    def project(theta: np.ndarray) -> np.ndarray:
        t = np.asarray(theta, dtype=np.float64)
        single = t.ndim == 1
        if single:
            t = t[None, :]
        rows = []
        for i in range(t.shape[0]):
            v = np.asarray(fn(t[i]), dtype=np.float64).reshape(-1)
            rows.append(v)
        out = np.stack(rows, axis=0)
        return out[0] if single else out

    return project


# =============================================================================
# Per-trial runner
# =============================================================================


@dataclass
class NPEArgs:
    """Just the fields ``heater_base.make_runner`` / ``train_npe`` reads."""

    batch_size: int
    learning_rate: float
    epochs: int
    patience: int
    hidden_features: int
    num_transforms: int
    repeats: int
    num_mdn_components: int = 4


def _npe_args(args: argparse.Namespace) -> NPEArgs:
    return NPEArgs(
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        epochs=int(args.epochs),
        patience=int(args.patience),
        hidden_features=int(args.hidden_features),
        num_transforms=int(args.num_transforms),
        repeats=int(args.repeats),
    )


def _train_maf_arm(
    target: np.ndarray, data: np.ndarray,
    low: np.ndarray, high: np.ndarray,
    args: NPEArgs, device: str, seed: int,
) -> tuple[float, np.ndarray, Any]:
    """Train one MAF NPE arm. Wraps ``heater_base.make_runner`` and forces the
    lampe backend's ``max_epochs`` cap, which the heater builder does not set
    (it sets the sbi-flavour key ``max_num_epochs``, so on lampe the default
    ``int(1e10)`` was in effect and only ``stop_after_epochs`` bounded it).
    """
    import torch
    from ili.dataloaders import NumpyLoader

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    runner = heater_base.make_runner(low, high, args, device, "maf")
    runner.train_args["max_epochs"] = int(args.epochs)
    loader = NumpyLoader(
        x=np.asarray(data, dtype=np.float32),
        theta=np.asarray(target, dtype=np.float32),
    )
    posterior, summaries = runner(loader=loader)
    val = np.asarray(summaries[0]["validation_log_probs"])
    return float(np.max(val)), val, posterior


def _standardise_x(x_tr: np.ndarray, x_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match ``oneshot_sweep.standardise_x`` for the flattened replicate data."""
    x_tr = np.asarray(x_tr, dtype=np.float64)
    x_va = np.asarray(x_va, dtype=np.float64)
    mu = x_tr.mean(0)
    sd = x_tr.std(0) + 1e-12
    return (x_tr - mu) / sd, (x_va - mu) / sd


def run_trial(args: argparse.Namespace, d: int, trial: int, device: str) -> dict[str, Any]:
    """Regenerate sims, train the three arms, score on shared axes."""
    from degeneracy_distillery.problems import get_problem

    seed = int(args.seed + 7919 * trial + 31 * d)
    print(f"\n=== rosenbrock_npe d={d} trial={trial} (seed={seed}) ===", flush=True)

    # 1. Load discovery row and rebuild the problem exactly like the driver.
    disc_path = Path(args.discovery_dir) / f"d{d}" / f"trial_{trial}" / "metrics.csv"
    if not disc_path.exists():
        raise FileNotFoundError(f"missing discovery metrics.csv: {disc_path}")
    disc_df = pd.read_csv(disc_path)
    disc = disc_df[disc_df["status"] == "ok"]
    if disc.empty:
        raise RuntimeError(f"no ok row in {disc_path}")
    row_disc = disc.iloc[0].to_dict()
    disc_seed = int(row_disc["seed"])
    if disc_seed != seed:
        # Discovery driver's formula must match; if it does not, trust the
        # recorded seed for regeneration and record the deviation.
        print(
            f"[warn] recorded discovery seed {disc_seed} != recomputed {seed}; "
            f"using recorded seed", flush=True,
        )
        seed = disc_seed
    problem = get_problem("rosenbrock", seed=seed, d=int(d))

    # 2. Regenerate the same training and test simulations.
    rng = np.random.default_rng(seed)
    th_tr, x_tr = problem.sample(int(args.nsims), rng)
    th_te, x_te = problem.sample(int(args.n_test), rng)
    x_tr_flat = np.asarray(x_tr, dtype=np.float32).reshape(x_tr.shape[0], -1)
    x_te_flat = np.asarray(x_te, dtype=np.float32).reshape(x_te.shape[0], -1)
    x_tr_std, x_te_std = _standardise_x(x_tr_flat, x_te_flat)
    x_tr_std = x_tr_std.astype(np.float32)
    x_te_std = x_te_std.astype(np.float32)

    # 3. Marginal evaluation subset (validation observations for scoring).
    n_marg = int(min(args.n_marginal_val, args.n_test))
    ev_idx = np.arange(n_marg)
    theta_ev = np.asarray(th_te[ev_idx], dtype=np.float64)
    data_ev = x_te_std[ev_idx]

    # 4. Oracle coordinates.
    truth_tr = np.asarray(problem.truth_coords(th_tr), dtype=np.float64)
    truth_ev = np.asarray(problem.truth_coords(theta_ev), dtype=np.float64)
    oracle_std = heater_base.AffineStandardiser.fit(truth_tr)
    truth_tr_s = np.asarray(oracle_std(truth_tr), dtype=np.float32)
    truth_ev_s = np.asarray(oracle_std(truth_ev), dtype=np.float64)

    # 5. Discovered coordinates.
    expression = row_disc.get("expression")
    if not isinstance(expression, str) or not expression:
        raise RuntimeError(f"discovery row has no expression: {row_disc}")
    project = build_eta_projector(expression, int(d))
    eta_tr = project(th_tr).astype(np.float64)
    eta_ev = project(theta_ev).astype(np.float64)
    finite_tr = np.isfinite(eta_tr).all(axis=1)
    if int(finite_tr.sum()) < 100:
        raise RuntimeError(
            f"discovered projector is non-finite on {int((~finite_tr).sum())}/"
            f"{finite_tr.size} training draws"
        )
    disc_std = heater_base.AffineStandardiser.fit(eta_tr[finite_tr])
    eta_tr_s = np.asarray(disc_std(eta_tr[finite_tr]), dtype=np.float32)
    eta_ev_s = np.asarray(disc_std(eta_ev), dtype=np.float64)
    m_fit = int(eta_tr_s.shape[1])

    # 6. Train the three arms. Every arm gets a distinct seed so the maps
    #    are independent samples from the training procedure.
    npe_args = _npe_args(args)
    A_BOX = 3.0
    STD_BOX = 5.0
    d_int = int(d)
    low_raw = np.full(d_int, -A_BOX, dtype=np.float32)
    high_raw = np.full(d_int, A_BOX, dtype=np.float32)
    low_oracle = np.full(2, -STD_BOX, dtype=np.float32)
    high_oracle = np.full(2, STD_BOX, dtype=np.float32)
    low_disc = np.full(m_fit, -STD_BOX, dtype=np.float32)
    high_disc = np.full(m_fit, STD_BOX, dtype=np.float32)

    runtimes: dict[str, float] = {}
    t0 = time.time()
    raw_lp, _, raw_post = _train_maf_arm(
        np.asarray(th_tr, dtype=np.float32), x_tr_std,
        low_raw, high_raw, npe_args, device, seed,
    )
    runtimes["train_raw"] = time.time() - t0

    t0 = time.time()
    oracle_lp, _, oracle_post = _train_maf_arm(
        truth_tr_s, x_tr_std, low_oracle, high_oracle, npe_args,
        device, seed + 1,
    )
    runtimes["train_oracle"] = time.time() - t0

    t0 = time.time()
    disc_lp, _, disc_post = _train_maf_arm(
        eta_tr_s, x_tr_std[finite_tr],
        low_disc, high_disc, npe_args, device, seed + 2,
    )
    runtimes["train_disc"] = time.time() - t0
    print(
        f"[npe] native val log-prob  raw={raw_lp:.3f}  oracle={oracle_lp:.3f}  "
        f"discovered={disc_lp:.3f}  (m_fit={m_fit})",
        flush=True,
    )

    # 7. Score on shared axes.
    t0 = time.time()
    n_samples = int(args.n_marginal_samples)

    raw_scores_oracle = np.empty(n_marg, dtype=np.float64)
    oracle_scores = np.empty(n_marg, dtype=np.float64)
    raw_scores_disc = np.empty(n_marg, dtype=np.float64)
    disc_scores = np.empty(n_marg, dtype=np.float64)
    raw_inbox = np.empty(n_marg, dtype=np.float64)
    oracle_inbox = np.empty(n_marg, dtype=np.float64)
    disc_inbox = np.empty(n_marg, dtype=np.float64)

    for i in range(n_marg):
        x_obs = data_ev[i]
        raw_samples = sample_from_ensemble(raw_post, x_obs, n_samples, device)
        oracle_samples = sample_from_ensemble(oracle_post, x_obs, n_samples, device)
        disc_samples = sample_from_ensemble(disc_post, x_obs, n_samples, device)

        raw_inbox[i] = inbox_fraction(raw_samples, low_raw, high_raw)
        oracle_inbox[i] = inbox_fraction(oracle_samples, low_oracle, high_oracle)
        disc_inbox[i] = inbox_fraction(disc_samples, low_disc, high_disc)

        # Oracle axes: banana coords, standardised.
        raw_on_or = oracle_std(np.asarray(problem.truth_coords(raw_samples))).astype(np.float64)
        oracle_on_or = np.asarray(oracle_samples, dtype=np.float64)
        raw_scores_oracle[i] = gaussian_matched_logp(raw_on_or, truth_ev_s[i])
        oracle_scores[i] = gaussian_matched_logp(oracle_on_or, truth_ev_s[i])

        # Discovered axes: standardised symbolic stack.
        raw_on_di = disc_std(project(raw_samples)).astype(np.float64)
        disc_on_di = np.asarray(disc_samples, dtype=np.float64)
        raw_scores_disc[i] = gaussian_matched_logp(raw_on_di, eta_ev_s[i])
        disc_scores[i] = gaussian_matched_logp(disc_on_di, eta_ev_s[i])
    runtimes["score"] = time.time() - t0

    row = {
        "problem": "rosenbrock",
        "arm": "npe",
        "d": int(d),
        "trial": int(trial),
        "seed": int(seed),
        "status": "ok",
        "failed_stage": None,
        "n_sims": int(args.nsims),
        "n_test": int(args.n_test),
        "n_eta": int(m_fit),
        "raw_log_prob": float(raw_lp),
        "oracle_log_prob": float(oracle_lp),
        "discovered_log_prob": float(disc_lp),
        "raw_on_oracle_axes": float(np.nanmean(raw_scores_oracle)),
        "oracle_on_oracle_axes": float(np.nanmean(oracle_scores)),
        "raw_on_discovered_axes": float(np.nanmean(raw_scores_disc)),
        "discovered_on_discovered_axes": float(np.nanmean(disc_scores)),
        "raw_inbox_frac": float(np.nanmean(raw_inbox)),
        "oracle_inbox_frac": float(np.nanmean(oracle_inbox)),
        "discovered_inbox_frac": float(np.nanmean(disc_inbox)),
        "rank_correct": bool(row_disc.get("rank_correct", False)),
        "symbolic_recovered": bool(row_disc.get("symbolic_recovered", False)),
        "expression": expression,
        "nde": "maf",
        "hidden_features": int(args.hidden_features),
        "num_transforms": int(args.num_transforms),
        "repeats": int(args.repeats),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "n_marginal_val": int(n_marg),
        "n_marginal_samples": int(n_samples),
        "device": str(device),
    }
    for k, v in runtimes.items():
        row[f"runtime_{k}_s"] = float(v)
    row["runtime_total_s"] = float(sum(runtimes.values()))
    print(
        f"[score] raw_on_oracle={row['raw_on_oracle_axes']:.3f}  "
        f"oracle={row['oracle_on_oracle_axes']:.3f}  "
        f"raw_on_disc={row['raw_on_discovered_axes']:.3f}  "
        f"disc={row['discovered_on_discovered_axes']:.3f}  "
        f"inbox raw={row['raw_inbox_frac']:.3f} "
        f"oracle={row['oracle_inbox_frac']:.3f} "
        f"disc={row['discovered_inbox_frac']:.3f}",
        flush=True,
    )
    return row


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dims", nargs="+", type=int, default=[2, 4, 8, 16])
    p.add_argument("--trials", nargs="+", type=int, default=None,
                   help="Explicit trial ids. Default: range(--num-trials).")
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--trial-start", type=int, default=0)
    p.add_argument("--discovery-dir", type=Path, required=True,
                   help="Root containing d{d}/trial_{t}/metrics.csv (the "
                        "'oneshot' folder under the discovery output dir).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Per-(d, trial) npe_metrics.csv goes to "
                        "<out-dir>/d{d}/trial_{t}/npe_metrics.csv.")
    p.add_argument("--nsims", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0,
                   help="Discovery driver was launched with args.seed=0, so "
                        "keep this at 0 to match the training simulations.")
    p.add_argument("--resume", action="store_true")

    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--hidden-features", type=int, default=50)
    p.add_argument("--num-transforms", type=int, default=5)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--n-marginal-val", type=int, default=200,
                   help="Number of validation observations scored.")
    p.add_argument("--n-marginal-samples", type=int, default=2000,
                   help="Posterior samples drawn per observation.")

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-threads", type=int, default=None,
                   help="torch.set_num_threads override "
                        "(default: SLURM_CPUS_PER_TASK or os.cpu_count()).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_threads is None:
        env = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get(
            "OMP_NUM_THREADS"
        )
        try:
            args.num_threads = int(env) if env else (os.cpu_count() or 1)
        except ValueError:
            args.num_threads = os.cpu_count() or 1
    import torch
    torch.set_num_threads(int(args.num_threads))
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  threads: {args.num_threads}", flush=True)

    trials = list(args.trials) if args.trials else list(
        range(args.trial_start, args.trial_start + args.num_trials)
    )

    manifest = {
        "script": "rosenbrock_npe_scaling.py",
        "arms": ["raw", "oracle", "discovered"],
        "nde": "maf",
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
    }

    for d in args.dims:
        for trial in trials:
            trial_dir = Path(args.out_dir) / f"d{int(d)}" / f"trial_{int(trial)}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = trial_dir / "npe_metrics.csv"
            rows: list[dict[str, Any]] = []
            if args.resume and metrics_path.exists():
                prev = pd.read_csv(metrics_path)
                rows = prev.to_dict("records")
                if any(
                    str(r.get("status")) == "ok"
                    and int(r.get("d")) == int(d)
                    and int(r.get("trial")) == int(trial)
                    for r in rows
                ):
                    print(
                        f"[resume] skipping d={d} trial={trial} "
                        f"(already ok in {metrics_path})", flush=True,
                    )
                    continue
            with open(trial_dir / "manifest.json", "w") as fh:
                json.dump({**manifest, "d": int(d), "trial": int(trial)},
                          fh, indent=2)
            try:
                row = run_trial(args, int(d), int(trial), device)
            except Exception as exc:
                row = {
                    "problem": "rosenbrock", "arm": "npe",
                    "d": int(d), "trial": int(trial),
                    "seed": int(args.seed + 7919 * trial + 31 * d),
                    "status": "failed",
                    "failed_stage": type(exc).__name__,
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_sims": int(args.nsims),
                }
                print(f"!!! d={d} trial={trial} failed: {exc}", flush=True)
                traceback.print_exc()
            rows.append(row)
            pd.DataFrame(rows).to_csv(metrics_path, index=False)
            print(f"[write] {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
