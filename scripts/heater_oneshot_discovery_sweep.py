#!/usr/bin/env python
"""Heater dimension sweep with the one-step loss, SR, and NPE.

Discovers the coordinate at every ambient ``d`` with a warm-started
descending ladder, fits ``K`` one-step maps at ``r_hat``, aligns them
with the existing Procrustes tooling, and hands the aligned
``(y, y_std)`` to pyoperon. Pareto candidates are rescored by frozen
one-step NLL, then three MDN NPE arms train (raw / analytic /
discovered). The scaling curve is built from the symbolic coordinate
the pipeline found.

Does not modify ``heater_dim_scaling_sweep.py``,
``heater_discovery_dim_scaling_sweep.py``, or ``heater_minimal_distillery.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import heater_discovery_dim_scaling_sweep as base  # noqa: E402


def compile_jax_coord(expr: str, n_params: int) -> Callable:
    """Compile an Operon/ESR string to a jax function ``theta -> (k,)``."""
    import jax.numpy as jnp
    import sympy
    from degeneracy_distillery.sr_utils import _parse_sr_equation, safe_lambdify

    expr_s, _, _, pars = _parse_sr_equation(expr)
    xs = list(sympy.symbols(" ".join(f"X{i + 1}" for i in range(n_params)), real=True))
    bs = list(sympy.symbols([f"b{i}" for i in range(len(pars))], real=True))
    fn = safe_lambdify(bs + xs, expr_s, ["jax"])

    def coord_fn(t):
        t = jnp.asarray(t)
        val = fn(*pars, *t)
        return jnp.atleast_1d(jnp.asarray(val)).reshape(-1)

    return coord_fn


def make_fisher(J: np.ndarray, mode: str, ridge: float = 1e-4) -> np.ndarray:
    """Implied Fisher ``J^T J``, optionally ridged or replaced by I."""
    J = np.asarray(J, dtype=np.float64)
    d = J.shape[-1]
    F = np.einsum("nmi,nmj->nij", J, J)
    if mode == "ridge":
        F = F + ridge * np.eye(d)
    elif mode == "identity":
        F = np.broadcast_to(np.eye(d), (J.shape[0], d, d)).copy()
    elif mode != "jtj":
        raise ValueError(f"unknown --sr-fisher {mode!r}")
    return F


def evaluate_ensemble(fits: list, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``eta`` and ``J`` for every member on a common ``theta``."""
    etas = np.stack([f.eta(theta) for f in fits], axis=0)
    jacs = np.stack([f.jac(theta) for f in fits], axis=0)
    return etas, jacs


def align_ensemble(
    fits: list,
    weights: np.ndarray,
    theta: np.ndarray,
    fisher_mode: str,
    ridge: float,
    seed: int,
    r_hat: int,
    d: int,
    align_mode: str = "procrustes",
) -> dict[str, Any]:
    """Align K one-step maps with ``process_ensemble_rotation_v2``."""
    from degeneracy_distillery.align_coords import process_ensemble_rotation_v2

    eta_ens, jac_ens = evaluate_ensemble(fits, theta)
    F_ens = np.stack(
        [make_fisher(jac_ens[k], fisher_mode, ridge=ridge) for k in range(len(fits))],
        axis=0,
    )
    best = int(np.argmax(np.asarray(weights)))
    datafile = {
        "theta": np.asarray(theta, dtype=np.float64),
        "eta_ensemble": eta_ens,
        "Jbar_ensemble": jac_ens,
        "F_ensemble": F_ens,
        "ensemble_weights": np.asarray(weights, dtype=np.float64),
        "norm_factor": np.array(1.0),
    }
    Favg = F_ens[best]
    n = int(theta.shape[0])
    canonicalize = "sign_only" if r_hat != d else "permute_and_sign"
    aligned = process_ensemble_rotation_v2(
        datafile,
        randidx=np.arange(n),
        Favg=Favg,
        best_model_idx=best,
        n_d=1.0,
        align_mode=align_mode,
        separate_nonlinearity=True,
        canonicalize=canonicalize,
        use_prior_normalization=True,
        restore_reference_mean=True,
        Fisher_to_flatten="best",
        verbose=True,
        offset_delta=0.1,
    )
    aligned["rotmats"] = np.asarray(aligned["rotmats"])
    aligned["eta_ensemble_raw"] = eta_ens
    aligned["jac_ensemble_raw"] = jac_ens
    return aligned


def alignment_replay_error(aligned: dict[str, Any]) -> float:
    """Max abs gap between ``aligned["ys"]`` and a replay of its own inputs."""
    from degeneracy_distillery.align_coords import apply_ensemble_alignment

    ys, _ = apply_ensemble_alignment(
        aligned["eta_ensemble_raw"], aligned["jac_ensemble_raw"], aligned,
    )
    mask = np.asarray(aligned["mask"], dtype=bool)
    ref = np.asarray(aligned["ys"], dtype=np.float64)
    return float(np.max(np.abs(ys[:, mask, :] - ref)))


def augment_aligned(
    fits: list,
    weights: np.ndarray,
    aligned: dict[str, Any],
    X_sr: np.ndarray,
    fisher_mode: str,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Push prior draws through the aligned ensemble, in the aligned frame.

    Rotation alone is not enough: alignment also centres each member,
    restores the reference mean and applies the global floor shift.
    """
    from degeneracy_distillery.align_coords import apply_ensemble_alignment
    from degeneracy_distillery.preprocessing_utils import weighted_std
    import jax.numpy as jnp

    w = np.asarray(weights, dtype=np.float64)
    w = w / np.maximum(w.sum(), 1e-12)
    etas = np.stack([np.asarray(f.eta(X_sr), dtype=np.float64) for f in fits], 0)
    jacs = np.stack([np.asarray(f.jac(X_sr), dtype=np.float64) for f in fits], 0)
    ys, js = apply_ensemble_alignment(etas, jacs, aligned)
    y = np.average(ys, axis=0, weights=w)
    y_std = np.asarray(weighted_std(jnp.asarray(ys), weights=jnp.asarray(w), axis=0))
    dy = np.average(js, axis=0, weights=w)
    Fs = make_fisher(dy, fisher_mode, ridge=ridge)
    return y, y_std, dy, Fs


def floor_y_std(y_std: np.ndarray, y: np.ndarray, floor_frac: float = 1e-3) -> np.ndarray:
    """Keep MDL from treating a dead axis as infinitely precise."""
    y_std = np.asarray(y_std, dtype=np.float64).copy()
    scale = np.median(np.abs(y), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    floor = floor_frac * scale
    return np.maximum(y_std, floor)


def rescore_frozen(
    candidates: list[tuple[str, float, float]],
    th_fit, x_fit, th_val, x_val,
    d: int, lo: float, hi: float,
    args: argparse.Namespace, seed: int,
    neural_nll: float,
) -> dict[str, Any]:
    from degeneracy_distillery.oneshot import train_oneshot

    rows = []
    best_nll = np.inf
    nll_pick = candidates[0][0]
    nll_pick_cx = candidates[0][2]
    for i, (eq, dl, cx) in enumerate(candidates):
        print(f"[frozen] {i + 1}/{len(candidates)}  cx={cx:.1f}  {eq}", flush=True)
        try:
            coord_fn = compile_jax_coord(eq, d)
            fr = train_oneshot(
                th_fit, x_fit, th_val, x_val,
                m=1, m0=1, lo=lo, hi=hi,
                steps=int(args.frozen_steps), seed=seed + 100 + i,
                coord_fn=coord_fn,
                verbose=False,
            )
            nll = float(np.mean(fr.nll_vec(th_val, x_val)))
        except Exception as exc:
            print(f"[frozen] failed: {exc}", flush=True)
            nll = float("nan")
        rows.append({"eq": eq, "dl": dl, "complexity": cx, "nll": nll})
        if np.isfinite(nll) and (
            nll < best_nll - 1e-12
            or (abs(nll - best_nll) <= 1e-12 and cx < nll_pick_cx)
        ):
            best_nll = nll
            nll_pick = eq
            nll_pick_cx = cx
    mdl_pick = candidates[0][0]
    return {
        "frozen_rows": rows,
        "nll_pick": nll_pick,
        "nll_pick_complexity": nll_pick_cx,
        "nll_symbolic": float(best_nll),
        "nll_neural": float(neural_nll),
        "nll_gap": float(best_nll - neural_nll) if np.isfinite(best_nll) else float("nan"),
        "picks_agree": nll_pick == mdl_pick,
        "mdl_pick": mdl_pick,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dims", nargs="+", type=int, default=[2, 3, 4, 6, 8, 10, 12])
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--trial-start", type=int, default=0,
                   help="First trial index. SLURM sets this so each task writes "
                        "the correct trial id when --num-trials is 1.")
    p.add_argument("--nsims", type=int, default=1000)
    p.add_argument("--nsims-list", nargs="+", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("heater_oneshot_scaling"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ladder-only", action="store_true",
                   help="Stop after the ladder. No SR, no NPE, no torch.")
    p.add_argument("--skip-npe", action="store_true")
    p.add_argument("--keep-workdirs", action="store_true")

    p.add_argument("--theta-min", type=float, default=1.0)
    p.add_argument("--theta-max", type=float, default=2.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--t-max", type=float, default=4.0)
    p.add_argument("--n-t", type=int, default=20)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--independent-npe-sims", action="store_true")
    p.add_argument("--nval-frac", type=float, default=0.15)

    p.add_argument("--m-probe", type=int, default=4)
    p.add_argument("--probe-steps", type=int, default=20000)
    p.add_argument("--rung-steps", type=int, default=5000)
    p.add_argument("--ladder-k", type=float, default=2.0)
    p.add_argument("--cold-rungs", action="store_true")
    p.add_argument("--no-refit-at-rhat", action="store_true",
                   help="Ablation: skip the fresh refit and keep the projected discovery map.")
    p.add_argument("--refit-steps", type=int, default=None,
                   help="Steps for the fresh refit at m_fit (default: --probe-steps).")
    p.add_argument("--oneshot-batch", type=int, default=512)
    p.add_argument("--oneshot-lr", type=float, default=1e-3)
    p.add_argument("--rank-rule", choices=("info", "nll"), default="info",
                   help="info: count probe axes above --info-floor nats. "
                        "nll: held-out NLL drop test (overcounts on a narrow prior box).")
    p.add_argument("--info-floor", type=float, default=1.0,
                   help="Nats of held-out information an axis needs to count toward r_hat.")
    p.add_argument("--rank-min-gap", type=float, default=10.0)
    p.add_argument("--expected-rank", type=int, default=1)
    p.add_argument("--n-ensemble", type=int, default=8,
                   help="Number of one-step maps at r_hat, including the ladder map.")
    p.add_argument("--ensemble-steps", type=int, default=5000)
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Train every ensemble member on the same pairs.")
    p.add_argument("--align-mode", choices=("procrustes", "kabsch", "none"),
                   default="procrustes")

    p.add_argument("--sr-time-limit", type=int, default=300)
    p.add_argument("--sr-n-aug", type=int, default=None)
    p.add_argument("--sr-n-aug-per-dim", type=int, default=1000)
    p.add_argument("--sr-max-length", type=int, default=None)
    p.add_argument("--sr-max-depth", type=int, default=None)
    p.add_argument("--sr-allowed-symbols", default="add,mul,div,constant,variable,sqrt",
                   help="No pow by default: Operon spends its budget on X_i ** (c X_j), "
                        "which the structure predicate then rejects.")
    p.add_argument("--sr-fisher", choices=("jtj", "ridge", "identity"), default="jtj")
    p.add_argument("--sr-fisher-ridge", type=float, default=1e-4)
    p.add_argument("--n-frozen-candidates", type=int, default=8)
    p.add_argument("--frozen-steps", type=int, default=3000)
    p.add_argument("--recovery-corr-thresh", type=float, default=0.99)

    p.add_argument("--nde-model", choices=("mdn", "maf"), default="mdn")
    p.add_argument("--raw-model", choices=("mdn", "maf"), default=None)
    p.add_argument("--num-mdn-components", type=int, default=4)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--hidden-features", type=int, default=50)
    p.add_argument("--num-transforms", type=int, default=5)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--n-marginal-val", type=int, default=200)
    p.add_argument("--n-marginal-samples", type=int, default=2000)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def _standardise_x(x_tr, x_va):
    mu = np.asarray(x_tr).mean(0)
    sd = np.asarray(x_tr).std(0) + 1e-12
    return (x_tr - mu) / sd, (x_va - mu) / sd, mu, sd


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto" and not (args.ladder_only or args.skip_npe):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "auto":
        device = "cpu"
    print(f"device: {device}", flush=True)

    import jax
    jax.config.update("jax_enable_x64", True)

    from degeneracy_distillery.oneshot import fit_ensemble, whittle_ladder

    cfg = base.ChainHeaterCfg(
        theta_min=args.theta_min, theta_max=args.theta_max, tau=args.tau,
        t_max=args.t_max, n_t=args.n_t, sigma=args.sigma,
    )
    raw_model = args.raw_model or args.nde_model
    eta_model = args.nde_model
    nsims_values = list(args.nsims_list) if args.nsims_list else [int(args.nsims)]

    metrics_path = out_dir / "metrics.csv"
    rows: list[dict[str, Any]] = []
    done: set[tuple[int, int, int]] = set()
    if args.resume and metrics_path.exists():
        prev = pd.read_csv(metrics_path)
        rows = prev.to_dict("records")
        done = {
            (int(r["nsims"]), int(r["d"]), int(r["trial"]))
            for r in rows
            if str(r.get("status")) == "ok"
        }
        print(f"resuming: {len(done)} completed runs found", flush=True)

    spectra: dict[str, np.ndarray] = {}
    expressions: dict[str, Any] = {}
    ladder_dump: dict[str, np.ndarray] = {}

    manifest = {
        "script": "heater_oneshot_discovery_sweep.py",
        "description": (
            "Chain-product heater scaling sweep. The coordinate is discovered "
            "at every d by a one-step warm-started ladder, then symbolised."
        ),
        "simulator": {
            "theta_min": cfg.theta_min, "theta_max": cfg.theta_max,
            "tau": cfg.tau, "t_max": cfg.t_max, "n_t": cfg.n_t, "sigma": cfg.sigma,
        },
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
        "nde": {"raw_model": raw_model, "eta_model": eta_model,
                "num_mdn_components": args.num_mdn_components},
        "arms": ["raw", "analytic", "discovered"],
        "success_criteria": {
            "rank_correct": f"ladder r_hat == {args.expected_rank}",
            "symbolic_recovered": (
                f"Spearman |rho| vs analytic axis >= {args.recovery_corr_thresh}"
            ),
        },
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    for nsims_idx, n_sim in enumerate(nsims_values):
        for d in args.dims:
            for trial in range(args.trial_start, args.trial_start + args.num_trials):
                if (n_sim, d, trial) in done:
                    continue
                seed = (args.seed + 7919 * trial + 31 * d + 1_000_003 * nsims_idx)
                tag = f"N{n_sim}_d{d}_trial{trial}"
                print(f"\n=== {tag} (seed {seed}) ===", flush=True)
                row: dict[str, Any] = {
                    "nsims": int(n_sim), "d": int(d), "trial": int(trial),
                    "seed": int(seed), "status": "ok", "failed_stage": None,
                    "error": None,
                }
                runtimes: dict[str, float] = {}
                workdir = out_dir / "runs" / tag
                workdir.mkdir(parents=True, exist_ok=True)
                stage = "sim"
                try:
                    rng = np.random.default_rng(seed)
                    theta_tr, data_tr = base.chain_dataset(n_sim, d, cfg, rng)
                    theta_va, data_va = base.chain_dataset(args.n_test, d, cfg, rng)
                    x_tr, x_va, _, _ = _standardise_x(data_tr, data_va)
                    nval = max(50, int(args.nval_frac * n_sim))
                    th_fit, x_fit = theta_tr[:-nval], x_tr[:-nval]
                    th_hold, x_hold = theta_tr[-nval:], x_tr[-nval:]

                    stage = "ladder"
                    t0 = time.time()
                    lad = whittle_ladder(
                        th_fit, x_fit, th_hold, x_hold,
                        lo=cfg.theta_min, hi=cfg.theta_max,
                        m_probe=args.m_probe,
                        probe_steps=args.probe_steps,
                        rung_steps=args.rung_steps,
                        seed=seed,
                        ladder_k=args.ladder_k,
                        cold_rungs=args.cold_rungs,
                        refit_at_rhat=not args.no_refit_at_rhat,
                        refit_steps=args.refit_steps,
                        rank_rule=args.rank_rule,
                        info_floor=args.info_floor,
                        batch=args.oneshot_batch,
                        lr=args.oneshot_lr,
                        min_gap=args.rank_min_gap,
                    )
                    runtimes["ladder"] = time.time() - t0
                    fit = lad["fit"]
                    r_hat = int(lad["r_hat"])
                    row["r_hat"] = r_hat
                    row["retained_rank"] = r_hat
                    row["m_fit"] = int(fit.m)
                    row["rank_correct"] = bool(r_hat == int(args.expected_rank))
                    row["refit_at_rhat"] = bool(lad.get("refit_at_rhat", True))
                    row["rank_rule"] = str(lad["rank_rule"])
                    row["info_probe_nats"] = ";".join(f"{v:.3f}" for v in lad["info_probe"])
                    row["info_final_nats"] = ";".join(f"{v:.3f}" for v in lad["info_final"])
                    row["jtj_eigengap_rank"] = int(lad["jtj_gap"]["rank"])
                    if lad.get("jtj_gap_screen") is not None:
                        row["jtj_eigengap_rank_screen"] = int(
                            lad["jtj_gap_screen"]["rank"]
                        )
                    row["nll_neural"] = float(np.mean(fit.nll_vec(th_hold, x_hold)))
                    for i, rec in enumerate(lad["rungs"]):
                        ladder_dump[f"{tag}_rung{i}_lam"] = np.asarray(rec["lam"])
                        ladder_dump[f"{tag}_rung{i}_nats"] = np.asarray(rec["nats"])
                        ladder_dump[f"{tag}_rung{i}_info_nats"] = np.asarray(rec["info_nats"])
                        if "dif_mean" in rec:
                            ladder_dump[f"{tag}_rung{i}_dif_mean"] = np.array([rec["dif_mean"]])
                            ladder_dump[f"{tag}_rung{i}_dif_se"] = np.array([rec["dif_se"]])
                    spectra[f"{tag}_jtj_median_relative"] = lad["jtj_gap"]["median_relative_spectrum"]
                    spectra[f"{tag}_jtj_gap_ratios"] = lad["jtj_gap"]["gap_ratios"]
                    if lad.get("jtj_gap_screen") is not None:
                        spectra[f"{tag}_jtj_screen_median_relative"] = (
                            lad["jtj_gap_screen"]["median_relative_spectrum"]
                        )
                        spectra[f"{tag}_jtj_screen_gap_ratios"] = (
                            lad["jtj_gap_screen"]["gap_ratios"]
                        )

                    if not args.ladder_only:
                        stage = "ensemble"
                        t0 = time.time()
                        ens = fit_ensemble(
                            th_fit, x_fit, th_hold, x_hold,
                            m=int(fit.m), lo=cfg.theta_min, hi=cfg.theta_max,
                            n_members=int(args.n_ensemble),
                            steps=int(args.ensemble_steps),
                            seed=seed,
                            init_fit=fit,
                            bootstrap=not args.no_bootstrap,
                            batch=args.oneshot_batch,
                            lr=args.oneshot_lr,
                        )
                        runtimes["ensemble"] = time.time() - t0
                        row["n_ensemble"] = int(len(ens["fits"]))

                        stage = "align"
                        t0 = time.time()
                        aligned = align_ensemble(
                            ens["fits"], ens["weights"],
                            np.asarray(theta_va, dtype=np.float64),
                            fisher_mode=args.sr_fisher,
                            ridge=args.sr_fisher_ridge,
                            seed=seed,
                            r_hat=r_hat,
                            d=d,
                            align_mode=args.align_mode,
                        )
                        runtimes["align"] = time.time() - t0
                        X = np.asarray(aligned["X"], dtype=np.float64)
                        y = np.asarray(aligned["y"], dtype=np.float64)
                        y_std = floor_y_std(aligned["y_std"], y)
                        dy = np.asarray(aligned["dy_sr"], dtype=np.float64)
                        Fs_al = np.asarray(aligned["Fs"], dtype=np.float64)
                        row["median_y_std"] = float(np.median(y_std[:, 0]))
                        print(
                            f"[align] n={X.shape[0]}  m={y.shape[1]}  "
                            f"median y_std={row['median_y_std']:.4g}",
                            flush=True,
                        )

                        stage = "sr"
                        n_aug = int(args.sr_n_aug if args.sr_n_aug else args.sr_n_aug_per_dim * d)
                        max_length = int(
                            args.sr_max_length if args.sr_max_length else max(25, 4 * d + 8)
                        )
                        max_depth = int(
                            args.sr_max_depth if args.sr_max_depth else max(10, d + 4)
                        )
                        from degeneracy_distillery.sr_utils import (
                            fit_and_analyze_sr, sr_structure_predicate,
                        )
                        rng_sr = np.random.default_rng(seed + 17)
                        X_sr = rng_sr.uniform(
                            cfg.theta_min, cfg.theta_max, size=(n_aug, d),
                        )
                        replay_err = alignment_replay_error(aligned)
                        row["align_replay_err"] = replay_err
                        print(f"[align] replay max |err| = {replay_err:.3g}", flush=True)
                        if not replay_err < 1e-6 * max(1.0, float(np.max(np.abs(aligned["ys"])))):
                            raise RuntimeError(
                                f"augmented rows would not share the aligned origin "
                                f"(replay err {replay_err:.3g})"
                            )
                        y_sr, y_std_sr, dy_sr, Fs_sr = augment_aligned(
                            ens["fits"], ens["weights"], aligned,
                            X_sr, args.sr_fisher, args.sr_fisher_ridge,
                        )
                        y_std_sr = floor_y_std(y_std_sr, y_sr)
                        X = np.concatenate([X, X_sr], 0)
                        y = np.concatenate([y, y_sr], 0)
                        y_std = np.concatenate([y_std, y_std_sr], 0)
                        dy = np.concatenate([dy, dy_sr], 0)
                        Fs_al = np.concatenate([Fs_al, Fs_sr], 0)
                        modes = [args.sr_fisher] + (
                            ["ridge"] if args.sr_fisher == "jtj" else []
                        )
                        mdl_coords = None
                        analysis = None
                        fisher_used = args.sr_fisher
                        last_err = None
                        t0 = time.time()
                        for mode in modes:
                            if mode == args.sr_fisher:
                                Fs = Fs_al
                            else:
                                Fs = make_fisher(dy, mode, ridge=args.sr_fisher_ridge)
                            sr_dir = workdir / f"sr_{mode}"
                            sr_dir.mkdir(parents=True, exist_ok=True)
                            print(
                                f"[sr] fisher={mode}  n={X.shape[0]}  "
                                f"max_length={max_length}",
                                flush=True,
                            )
                            try:
                                mdl_coords, frob_coords, analysis, _ = fit_and_analyze_sr(
                                    X, y, y_std, dy, Fs,
                                    n_params=d,
                                    components_to_fit=list(range(int(y.shape[1]))),
                                    slice_fisher=False,
                                    parent_dir=str(sr_dir) + os.sep,
                                    test_size=0.5,
                                    random_state=32134 + seed,
                                    shuffle=True,
                                    time_limit=args.sr_time_limit,
                                    max_length=max_length,
                                    max_depth=max_depth,
                                    allowed_symbols=args.sr_allowed_symbols,
                                    max_complexity_thresh=max(20, max_length),
                                    equation_set="pareto",
                                    length_penalty=2.0,
                                    equation_predicate=sr_structure_predicate(
                                        n_params=d,
                                        forbid_self_transcendental=True,
                                        check_nested_exp=False,
                                        forbid_x_in_pow_exponent=True,
                                    ),
                                )
                                fisher_used = mode
                                last_err = None
                                break
                            except Exception as exc:
                                last_err = exc
                                print(f"[sr] {mode} failed: {exc}", flush=True)
                        if last_err is not None or mdl_coords is None or analysis is None:
                            raise RuntimeError(f"SR failed: {last_err}")
                        runtimes["sr"] = time.time() - t0
                        row["sr_fisher_used"] = fisher_used
                        row["n_augmented_coordinate_evaluations"] = n_aug
                        row["sr_max_length"] = max_length

                        eqs = [str(e) for e in analysis["equations"][0]]
                        dls = np.asarray(analysis["DL"][0], dtype=np.float64)
                        cxs = np.asarray(analysis["complexity"][0], dtype=np.float64)
                        order = np.argsort(dls)
                        seen: set[str] = set()
                        candidates: list[tuple[str, float, float]] = []
                        for j in order:
                            eq = eqs[int(j)]
                            if eq in seen or not np.isfinite(dls[j]):
                                continue
                            seen.add(eq)
                            candidates.append((eq, float(dls[j]), float(cxs[j])))
                            if len(candidates) >= int(args.n_frozen_candidates):
                                break
                        if not candidates:
                            candidates = [(str(mdl_coords[0]), float("nan"), float("nan"))]

                        stage = "frozen"
                        t0 = time.time()
                        frozen = rescore_frozen(
                            candidates, th_fit, x_fit, th_hold, x_hold,
                            d, cfg.theta_min, cfg.theta_max,
                            args, seed, row["nll_neural"],
                        )
                        runtimes["frozen"] = time.time() - t0
                        row["nll_symbolic"] = frozen["nll_symbolic"]
                        row["nll_gap"] = frozen["nll_gap"]
                        row["picks_agree"] = bool(frozen["picks_agree"])
                        row["expression"] = frozen["nll_pick"]
                        row["expression_mdl"] = frozen["mdl_pick"]
                        row["complexity"] = float(frozen["nll_pick_complexity"])
                        expressions[tag] = {
                            "nll": frozen["nll_pick"],
                            "mdl": frozen["mdl_pick"],
                            "frob": [str(e) for e in frob_coords],
                            "candidates": frozen["frozen_rows"],
                            "sr_fisher_used": fisher_used,
                        }

                        stage = "recovery_scoring"
                        from scipy.stats import pearsonr, spearmanr
                        project = base.symbolic_projector([frozen["nll_pick"]], d)
                        eta_disc_va = np.asarray(project(theta_va), dtype=np.float64)
                        primary = eta_disc_va[:, 0]
                        eta_an_va = base.analytic_eta(theta_va, cfg, d).astype(np.float64)
                        product_va = np.prod(theta_va, axis=1).astype(np.float64)
                        good = np.isfinite(primary)
                        if good.sum() < 10:
                            raise RuntimeError(
                                "discovered expression is non-finite on held-out draws"
                            )
                        rho = spearmanr(primary[good], eta_an_va[good]).correlation
                        row["spearman_abs"] = float(abs(rho))
                        row["pearson_abs_logprod"] = float(
                            abs(pearsonr(primary[good], eta_an_va[good])[0])
                        )
                        row["pearson_abs_product"] = float(
                            abs(pearsonr(primary[good], product_va[good])[0])
                        )
                        row["symbolic_recovered"] = bool(
                            abs(rho) >= args.recovery_corr_thresh
                        )
                        print(
                            f"[recovery] |rho|={abs(rho):.4f}  "
                            f"rank_correct={row['rank_correct']}  "
                            f"symbolic_recovered={row['symbolic_recovered']}",
                            flush=True,
                        )
                    else:
                        row["symbolic_recovered"] = False

                    if not (args.ladder_only or args.skip_npe):
                        stage = "npe"
                        if args.independent_npe_sims:
                            npe_rng = np.random.default_rng(seed + 555_557)
                            theta_np, data_np = base.chain_dataset(n_sim, d, cfg, npe_rng)
                        else:
                            theta_np, data_np = theta_tr, data_tr
                        marg_rng = np.random.default_rng(seed + 999_983)
                        n_marg = int(min(args.n_marginal_val, args.n_test))
                        theta_ev, data_ev = base.chain_dataset(n_marg, d, cfg, marg_rng)

                        low_raw = np.full(d, cfg.theta_min, dtype=np.float32)
                        high_raw = np.full(d, cfg.theta_max, dtype=np.float32)
                        raw_lp, _, raw_post = base.train_npe(
                            theta_np, data_np, low_raw, high_raw,
                            args, device, seed, raw_model,
                        )
                        row["raw_log_prob"] = raw_lp

                        eta_an = base.analytic_eta(theta_np, cfg, d)[:, None]
                        low_an = np.array([-5.0], dtype=np.float32)
                        high_an = np.array([5.0], dtype=np.float32)
                        an_lp, _, an_post = base.train_npe(
                            eta_an, data_np, low_an, high_an,
                            args, device, seed + 1, eta_model,
                        )
                        row["analytic_log_prob"] = an_lp

                        eta_disc_tr = np.asarray(project(theta_np), dtype=np.float64)
                        std = base.AffineStandardiser.fit(eta_disc_tr)
                        eta_disc_std = std(eta_disc_tr)
                        finite_rows = np.isfinite(eta_disc_std).all(axis=1)
                        n_eta = int(eta_disc_std.shape[1])
                        low_di = np.full(n_eta, -5.0, dtype=np.float32)
                        high_di = np.full(n_eta, 5.0, dtype=np.float32)
                        di_lp, _, di_post = base.train_npe(
                            eta_disc_std[finite_rows], data_np[finite_rows],
                            low_di, high_di, args, device, seed + 2, eta_model,
                        )
                        row["discovered_log_prob"] = di_lp
                        row["n_eta"] = n_eta

                        eta_an_ev = base.analytic_eta(theta_ev, cfg, d).astype(np.float64)
                        mu_log, var_log = base.log_theta_moments_uniform(
                            cfg.theta_min, cfg.theta_max,
                        )
                        an_scale = float(np.sqrt(d * var_log))
                        an_offset = float(d * mu_log)

                        def to_analytic(theta_s: np.ndarray) -> np.ndarray:
                            t = np.clip(np.asarray(theta_s, dtype=np.float64), 1e-12, None)
                            return (np.log(t).sum(axis=1) - an_offset) / an_scale

                        row["raw_on_analytic_marg"] = float(np.nanmean(
                            base.marginal_log_probs_on_axis(
                                raw_post, data_ev, eta_an_ev, to_analytic,
                                args.n_marginal_samples, device,
                            )
                        ))
                        row["analytic_on_analytic_marg"] = float(np.nanmean(
                            base.marginal_log_probs_on_axis(
                                an_post, data_ev, eta_an_ev,
                                lambda s: np.asarray(s).reshape(-1),
                                args.n_marginal_samples, device,
                            )
                        ))
                        eta_di_ev = std(
                            np.asarray(project(theta_ev), dtype=np.float64)
                        )[:, 0].astype(np.float64)

                        def to_discovered(theta_s: np.ndarray) -> np.ndarray:
                            return std(
                                np.asarray(project(theta_s), dtype=np.float64)
                            )[:, 0]

                        row["raw_on_discovered_marg"] = float(np.nanmean(
                            base.marginal_log_probs_on_axis(
                                raw_post, data_ev, eta_di_ev, to_discovered,
                                args.n_marginal_samples, device,
                            )
                        ))
                        row["discovered_on_discovered_marg"] = float(np.nanmean(
                            base.marginal_log_probs_on_axis(
                                di_post, data_ev, eta_di_ev,
                                lambda s: np.asarray(s)[:, 0]
                                if np.asarray(s).ndim > 1
                                else np.asarray(s).reshape(-1),
                                args.n_marginal_samples, device,
                            )
                        ))
                        print(
                            f"[npe] raw={raw_lp:.4f}  analytic={an_lp:.4f}  "
                            f"discovered={di_lp:.4f}  (n_eta={n_eta} of {d})",
                            flush=True,
                        )

                except Exception as exc:
                    row["status"] = "failed"
                    row["failed_stage"] = stage
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"!!! {tag} failed at stage {stage}: {exc}", flush=True)
                    traceback.print_exc()

                for k, v in runtimes.items():
                    row[f"runtime_{k}_s"] = float(v)
                row["runtime_total_s"] = float(sum(runtimes.values()))
                rows.append(row)
                df = pd.DataFrame(rows)
                df.to_csv(metrics_path, index=False)
                base.write_recovery_table(df, out_dir)
                base.write_aggregate(df, out_dir)
                if spectra:
                    np.savez_compressed(out_dir / "rank_spectra.npz", **spectra)
                if ladder_dump:
                    np.savez_compressed(out_dir / "ladder_spectra.npz", **ladder_dump)
                if expressions:
                    with open(out_dir / "expressions.json", "w") as fh:
                        json.dump(expressions, fh, indent=2, default=str)
                if not args.keep_workdirs and workdir.exists():
                    import shutil
                    shutil.rmtree(workdir, ignore_errors=True)

    df = pd.DataFrame(rows)
    if not df.empty:
        np.savez_compressed(
            out_dir / "metrics.npz",
            **{c: df[c].to_numpy() for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c])},
        )
    manifest["completed_runs"] = int((df["status"] == "ok").sum()) if not df.empty else 0
    manifest["failed_runs"] = int((df["status"] == "failed").sum()) if not df.empty else 0
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
