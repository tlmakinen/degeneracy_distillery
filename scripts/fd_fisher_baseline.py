#!/usr/bin/env python
"""Finite-difference Fisher baseline for the SIR and GW discovery runs.

What it does
------------
For one finished three-step run directory, this script

1. regenerates the run's simulations from the recorded seed by calling the
   published driver's own ``simulator_data`` (so theta, the theta scaler and,
   for GW, the PCA basis are the run's own),
2. computes the Fisher matrix at every held-out theta by central finite
   differences of the *noiseless* simulator mean. All three problems have
   Gaussian noise of fixed covariance about a deterministic mean, so
   ``F = J^T J / sigma^2`` with ``J = d mu / d theta``,
3. loads the Fishnet ensemble from ``fishnets_outputs.npz`` and compares the
   validation-weighted ensemble mean with the finite-difference matrix, in the
   scaled coordinates the Fishnet was trained in.

SIR detail. The Fishnet is trained on ``(beta, gamma)`` with the initial count
``I0 ~ Poisson(I0_MEAN)`` left as an unobserved nuisance. The comparable
finite-difference quantity is the 3x3 Fisher in ``(beta, gamma, I0)`` with the
prior precision ``1 / I0_MEAN`` added on ``I0`` and then marginalised (Schur
complement). The conditional matrix (``I0`` known) is saved as well.

GW detail. The drivers fit the PCA basis with scikit-learn's default solver and
no ``random_state``, so the regenerated basis equals the run's only up to the
accuracy of the randomised SVD (and component signs). The script reports how
closely the regenerated data match the ``x`` saved with the Fishnet outputs.

Cost accounting. Each finite-difference Fisher needs ``2 * d`` noiseless
simulator calls per step size, at parameter values chosen by us. The script
reports the total. The Fishnet uses only the prior-drawn simulations.

Usage
-----
    python scripts/fd_fisher_baseline.py --self-test

    python scripts/fd_fisher_baseline.py --problem sir \
        --run-dir $SCRATCH/follow_up_results/sir/rebuttal/seed_0 \
        --out-dir $SCRATCH/fd_fisher_baseline/sir_seed0

    python scripts/fd_fisher_baseline.py --problem gw_taylorf2 --run-dir ...
    python scripts/fd_fisher_baseline.py --problem gw_imrphenomd --run-dir ...

``--mode``, ``--seed`` and (SIR) ``--noise-std`` default to the values in the
run's ``config_manifest.json``. The driver modules import JAX, so everything
except ``--self-test`` must run in the cluster environment.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
DRIVERS = {
    "sir": "sir_notebook_run",
    "gw_taylorf2": "gw_notebook_run",
    "gw_imrphenomd": "imrphenomd_notebook_run",
}


# ---------------------------------------------------------------------------
# Finite-difference Fisher (NumPy only).
# ---------------------------------------------------------------------------
def fd_jacobian(mean_fn: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Central-difference Jacobian d mean / d theta, shape (n_out, d)."""
    theta = np.asarray(theta, dtype=np.float64)
    cols = []
    for a, h in enumerate(steps):
        up, dn = theta.copy(), theta.copy()
        up[a] += h
        dn[a] -= h
        cols.append((np.asarray(mean_fn(up), float) - np.asarray(mean_fn(dn), float)) / (2.0 * h))
    return np.stack(cols, axis=1)


def fd_fisher(
    mean_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    steps0: np.ndarray,
    noise_var: float,
    rtol: float = 1e-2,
    max_halvings: int = 8,
) -> tuple[np.ndarray, int, bool, float]:
    """Fisher matrix J^T J / noise_var with step halving until it converges.

    Returns (F, n_simulator_calls, converged, relative_change_at_last_step).
    The step is halved until the relative Frobenius change of F is below
    ``rtol``. A simulator with a discontinuity inside the stencil (the TaylorF2
    ISCO cut-off) does not converge; the flag records that.
    """
    steps = np.asarray(steps0, dtype=np.float64)
    d = len(steps)
    J = fd_jacobian(mean_fn, theta, steps)
    F = J.T @ J / noise_var
    calls = 2 * d
    change = np.inf
    for _ in range(max_halvings):
        steps = steps / 2.0
        J_new = fd_jacobian(mean_fn, theta, steps)
        F_new = J_new.T @ J_new / noise_var
        calls += 2 * d
        change = float(np.linalg.norm(F_new - F) / max(np.linalg.norm(F_new), 1e-300))
        F = F_new
        if change < rtol:
            return F, calls, True, change
    return F, calls, False, change


def marginalise(F: np.ndarray, keep: list[int], prior_precision: np.ndarray | None = None) -> np.ndarray:
    """Schur complement of F onto ``keep``, after adding a diagonal prior precision."""
    F = np.array(F, dtype=np.float64)
    if prior_precision is not None:
        F = F + np.diag(prior_precision)
    drop = [i for i in range(F.shape[0]) if i not in keep]
    if not drop:
        return F[np.ix_(keep, keep)]
    A = F[np.ix_(keep, keep)]
    B = F[np.ix_(keep, drop)]
    C = F[np.ix_(drop, drop)]
    return A - B @ np.linalg.solve(C, B.T)


def to_scaled(F_phys: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Fisher in theta_scaled = scale * theta + offset coordinates."""
    inv = 1.0 / np.asarray(scale, dtype=np.float64)
    return F_phys * np.outer(inv, inv)


# ---------------------------------------------------------------------------
# Comparison metrics.
# ---------------------------------------------------------------------------
def _sym_eig(F: np.ndarray):
    w, v = np.linalg.eigh(0.5 * (F + F.T))
    return w, v


def compare(F_est: np.ndarray, F_ref: np.ndarray) -> dict[str, float]:
    """Metrics between an estimate and a reference SPD matrix (same coordinates).

    shape_cosine      Frobenius cosine between the two matrices. Scale free.
    shape_rel_err     ||M_est - M_ref||_F / ||M_ref||_F with M = F / tr F.
    stiff_cosine      |cos| between the leading eigenvectors.
    log10_trace_ratio log10(tr F_est / tr F_ref): the amplitude offset.
    log10_cond_*      log10 condition number of each matrix.
    affine_dist       sqrt(sum_i ln^2 lambda_i(F_ref^-1 F_est)); 0 iff equal.
    """
    out: dict[str, float] = {}
    fe, fr = np.asarray(F_est, float), np.asarray(F_ref, float)
    out["shape_cosine"] = float(np.sum(fe * fr) / (np.linalg.norm(fe) * np.linalg.norm(fr)))
    me, mr = fe / np.trace(fe), fr / np.trace(fr)
    out["shape_rel_err"] = float(np.linalg.norm(me - mr) / np.linalg.norm(mr))
    we, ve = _sym_eig(fe)
    wr, vr = _sym_eig(fr)
    out["stiff_cosine"] = float(abs(ve[:, -1] @ vr[:, -1]))
    out["log10_trace_ratio"] = float(np.log10(np.trace(fe) / np.trace(fr)))
    out["log10_cond_est"] = float(np.log10(we[-1] / max(we[0], 1e-300)))
    out["log10_cond_ref"] = float(np.log10(wr[-1] / max(wr[0], 1e-300)))
    try:
        lam = np.linalg.eigvals(np.linalg.solve(fr, fe)).real
        out["affine_dist"] = float(np.sqrt(np.sum(np.log(np.clip(lam, 1e-300, None)) ** 2)))
    except np.linalg.LinAlgError:
        out["affine_dist"] = float("nan")
    return out


def summarise(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Median and 16th/84th percentiles of every metric over theta points."""
    out = {}
    for key in rows[0]:
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[key] = {"median": float("nan"), "p16": float("nan"), "p84": float("nan")}
            continue
        p16, p50, p84 = np.percentile(vals, [16, 50, 84])
        out[key] = {"median": float(p50), "p16": float(p16), "p84": float(p84)}
    return out


# ---------------------------------------------------------------------------
# Problem adapters. Each wraps the published driver; nothing is reimplemented
# except the two lines that rebuild the frequency grid.
# ---------------------------------------------------------------------------
def load_driver(problem: str):
    name = DRIVERS[problem]
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))  # imrphenomd driver imports gw_notebook_run
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look the module up at class-body time
    spec.loader.exec_module(mod)
    return mod


@dataclasses.dataclass
class Problem:
    name: str
    theta_test_phys: np.ndarray      # (n, d_full) physical parameters of the held-out set
    data_test: np.ndarray            # (n, n_out) noisy data the Fishnet saw
    scale: np.ndarray                # MinMaxScaler.scale_ for the Fishnet's parameters
    offset: np.ndarray               # MinMaxScaler.min_
    mean_fn: Callable[[np.ndarray], np.ndarray]
    noise_var: float
    keep: list[int]                  # parameters the Fishnet was trained on
    prior_precision: np.ndarray | None
    step_widths: np.ndarray          # prior width per FD parameter (sets the step)
    notes: dict


def build_sir(mod, config, seed: int, noise_std: float | None) -> Problem:
    from scipy.integrate import odeint

    if noise_std is not None:
        mod.NOISE_STD = float(noise_std)
    data = mod.simulator_data(config, seed)
    scaler = mod.fit_theta_scaler(data["theta_train"], feature_range=(1.0, 2.0))
    theta = np.asarray(data["theta_test"], dtype=np.float64)   # (beta, gamma, I0/10)
    theta_phys = theta.copy()
    theta_phys[:, 2] *= 10.0                                    # I0 as a count

    def mean_fn(t: np.ndarray) -> np.ndarray:
        beta, gamma, i0 = t
        sol = odeint(mod.sir_odes, [mod.N_POP - i0, i0, 0.0], mod.T_OBS, args=(beta, gamma, mod.N_POP))
        return sol[:, 1] / mod.N_POP

    return Problem(
        name="sir",
        theta_test_phys=theta_phys,
        data_test=np.asarray(data["data_test"], dtype=np.float64),
        scale=np.asarray(scaler.scale_[:2], dtype=np.float64),
        offset=np.asarray(scaler.min_[:2], dtype=np.float64),
        mean_fn=mean_fn,
        noise_var=float(mod.NOISE_STD) ** 2,
        keep=[0, 1],
        prior_precision=np.array([0.0, 0.0, 1.0 / mod.I0_MEAN]),
        step_widths=np.array([mod.BETA_MAX - mod.BETA_MIN, mod.GAMMA_MAX - mod.GAMMA_MIN, float(mod.I0_MEAN)]),
        notes={"noise_std": float(mod.NOISE_STD), "n_pop": int(mod.N_POP), "i0_mean": float(mod.I0_MEAN)},
    )


def build_gw(mod, config, seed: int, kind: str) -> Problem:
    captured: dict = {}
    base_pca = mod.PCA

    class RecordingPCA(base_pca):  # capture the run's own fitted basis
        def fit(self, X, y=None):
            out = super().fit(X, y)
            captured["pca"] = self
            return out

    mod.PCA = RecordingPCA
    try:
        with tempfile.TemporaryDirectory(prefix=f"fd_{kind}_") as tmp:
            data = mod.simulator_data(config, seed, Path(tmp))
    finally:
        mod.PCA = base_pca
    pca = captured["pca"]
    scaler = mod.fit_theta_scaler(data["theta_train"], feature_range=(1.0, 2.0))

    gw = mod if kind == "gw_taylorf2" else mod.gw
    if kind == "gw_taylorf2":
        f_isco_lightest = 1.0 / (6**1.5 * np.pi * (mod.M1_MIN + mod.M2_MIN) * mod.M_SUN_SEC)
        freqs = np.arange(mod.F_LOW, min(f_isco_lightest, 1024.0), mod.DF)

        def waveform(m1, m2):
            return mod.taylor_f2_waveform(m1, m2, freqs)
    else:
        freqs = np.arange(mod.F_LOW, mod.F_HIGH, mod.DF)

        def waveform(m1, m2):
            return mod.imrphenomd_waveform(m1, m2, len(freqs))

    whiten = np.sqrt(4.0 * mod.DF) / np.sqrt(mod.a_ligo_psd(freqs))

    def mean_fn(t: np.ndarray) -> np.ndarray:
        h = waveform(float(t[0]), float(t[1])) * whiten
        return pca.transform(np.concatenate([h.real, h.imag])[None, :])[0]

    return Problem(
        name=kind,
        theta_test_phys=np.asarray(data["theta_test"], dtype=np.float64),
        data_test=np.asarray(data["data_test"], dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        offset=np.asarray(scaler.min_, dtype=np.float64),
        mean_fn=mean_fn,
        noise_var=1.0,  # unit noise, and the PCA projection has orthonormal rows
        keep=[0, 1],
        prior_precision=None,
        step_widths=np.array([gw.M1_MAX - gw.M1_MIN, gw.M2_MAX - gw.M2_MIN]),
        notes={
            "n_pca": int(pca.n_components_),
            "pca_variance_captured": float(pca.explained_variance_ratio_.sum()),
            "n_freq": int(len(freqs)),
        },
    )


# ---------------------------------------------------------------------------
# Main comparison.
# ---------------------------------------------------------------------------
def find_fishnets(run_dir: Path) -> Path:
    hits = sorted(run_dir.rglob("fishnets_outputs.npz"))
    if not hits:
        raise FileNotFoundError(f"no fishnets_outputs.npz under {run_dir}")
    return min(hits, key=lambda p: len(p.parts))  # the driver's own copy, not the SR copy


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = {}
    manifest_path = run_dir / "config_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    mode = args.mode or manifest.get("mode", "rebuttal")
    stage_seeds = manifest.get("stage_seeds", {})
    data_seed = args.data_seed if args.data_seed is not None else stage_seeds.get("data")
    if data_seed is None:
        raise SystemExit("no data seed: pass --data-seed or point --run-dir at a run with config_manifest.json")
    noise_std = args.noise_std if args.noise_std is not None else manifest.get("noise_std")

    mod = load_driver(args.problem)
    config = mod.CONFIGS[mode]
    if "config" in manifest and "nsims" in manifest["config"]:
        config = dataclasses.replace(config, nsims=int(manifest["config"]["nsims"]))
    print(f"[fd] problem={args.problem} mode={mode} nsims={config.nsims} data_seed={data_seed}")

    t0 = time.time()
    if args.problem == "sir":
        prob = build_sir(mod, config, int(data_seed), noise_std)
    else:
        prob = build_gw(mod, config, int(data_seed), args.problem)
    print(f"[fd] regenerated simulations in {time.time() - t0:.0f}s; notes={prob.notes}")

    fish_path = Path(args.fishnets) if args.fishnets else find_fishnets(run_dir)
    with np.load(fish_path) as f:
        theta_fish = np.asarray(f["theta"], dtype=np.float64)
        Fs = np.asarray(f["Fs"], dtype=np.float64)          # (members, n, d, d)
        weights = np.asarray(f["ensemble_weights"], dtype=np.float64)
        x_run = np.asarray(f["x"], dtype=np.float64) if "x" in f.files else None
    F_fish = np.einsum("m,mnab->nab", weights / weights.sum(), Fs)
    print(f"[fd] Fishnet ensemble: {Fs.shape[0]} members, {Fs.shape[1]} points, from {fish_path}")

    # Check 1: the regenerated held-out theta is the theta the Fishnet was evaluated at.
    theta_scaled = prob.theta_test_phys[:, prob.keep] * prob.scale + prob.offset
    n = min(len(theta_scaled), len(theta_fish))
    theta_err = float(np.max(np.abs(theta_scaled[:n] - theta_fish[:n])))
    print(f"[fd] max |theta_regenerated - theta_fishnet| = {theta_err:.2e}")
    if theta_err > 1e-3:
        raise SystemExit(
            "regenerated theta does not match the Fishnet outputs: wrong seed, mode or nsims. "
            "The comparison would pair matrices at different points."
        )

    # Check 2: data minus the noiseless mean has the stated noise level.
    n_chk = min(n, 64)
    resid = np.stack([prob.data_test[i] - prob.mean_fn(prob.theta_test_phys[i]) for i in range(n_chk)])
    resid_std = float(resid.std())
    stated = float(np.sqrt(prob.noise_var))
    print(f"[fd] residual std of data about the noiseless mean = {resid_std:.4g} (stated noise {stated:.4g})")
    if not (0.8 * stated < resid_std < 1.25 * stated):
        raise SystemExit(
            "the noiseless mean does not reproduce the run's data at the stated noise level; "
            "for SIR pass the --noise-std the run used."
        )

    # Check 3 (reported, not enforced): the regenerated data equal the data the
    # Fishnet saw. SIR is bit-reproducible, so a mismatch there means a different
    # noise level or seed. GW differs slightly through the PCA solver.
    data_rel_diff = None
    if x_run is not None and x_run.shape[1:] == prob.data_test.shape[1:]:
        a, b = x_run[:n], prob.data_test[:n]
        sign = np.sign(np.sum(a * b, axis=0, keepdims=True))
        sign[sign == 0] = 1.0
        data_rel_diff = float(np.linalg.norm(a - sign * b) / np.linalg.norm(a))
        print(f"[fd] relative difference between the run's data and the regenerated data = {data_rel_diff:.3g}")
        if data_rel_diff > 0.05:
            print("[fd] WARNING: the regenerated data do not match the run's data. Check --noise-std, "
                  "--data-seed and --mode before using these numbers.")

    idx = np.arange(n)
    if args.max_points and args.max_points < n:
        idx = np.random.default_rng(0).choice(n, args.max_points, replace=False)
        idx.sort()

    rows, F_fd_all, F_cond_all, conv, calls_total = [], [], [], [], 0
    steps0 = args.rel_step * prob.step_widths
    t0 = time.time()
    for k, i in enumerate(idx):
        F_full, calls, ok, change = fd_fisher(
            prob.mean_fn, prob.theta_test_phys[i], steps0, prob.noise_var, args.rtol, args.max_halvings
        )
        calls_total += calls
        conv.append(ok)
        F_marg = to_scaled(marginalise(F_full, prob.keep, prob.prior_precision), prob.scale)
        F_cond = to_scaled(F_full[np.ix_(prob.keep, prob.keep)], prob.scale)
        F_fd_all.append(F_marg)
        F_cond_all.append(F_cond)
        row = compare(F_fish[i], F_marg)
        row["fd_converged"] = float(ok)
        row["fd_rel_change"] = change
        rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"[fd] {k + 1}/{len(idx)} points, {time.time() - t0:.0f}s")

    conv = np.array(conv)
    good = [r for r, ok in zip(rows, conv) if ok]
    summary = {
        "problem": args.problem,
        "run_dir": str(run_dir),
        "fishnets": str(fish_path),
        "mode": mode,
        "data_seed": int(data_seed),
        "n_points": int(len(idx)),
        "n_fd_converged": int(conv.sum()),
        "fd_simulator_calls": int(calls_total),
        "fd_calls_per_point": float(calls_total / len(idx)),
        "fishnet_training_simulations": int(config.nsims),
        "rel_step": args.rel_step,
        "rtol": args.rtol,
        "theta_match_max_abs": theta_err,
        "data_rel_diff_vs_run": data_rel_diff,
        "residual_std": resid_std,
        "stated_noise_std": stated,
        "notes": prob.notes,
        "metrics_converged_points": summarise(good) if good else {},
        "metrics_all_points": summarise(rows),
    }

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "fd_fisher_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"fd_fisher_{args.problem}.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out_dir / f"fd_fisher_{args.problem}.npz",
        index=idx,
        theta_scaled=theta_scaled[idx],
        theta_phys=prob.theta_test_phys[idx],
        F_fd=np.array(F_fd_all),
        F_fd_conditional=np.array(F_cond_all),
        F_fishnet=F_fish[idx],
        fd_converged=conv,
        **{f"metric_{k}": np.array([r[k] for r in rows]) for k in rows[0]},
    )
    print_summary(summary)
    print(f"[fd] wrote {out_dir}")


def print_summary(summary: dict) -> None:
    m = summary["metrics_converged_points"] or summary["metrics_all_points"]

    def cell(key: str) -> str:
        s = m[key]
        return f"{s['median']:.3f} ({s['p16']:.3f}, {s['p84']:.3f})"

    print()
    print("| problem | points (FD converged) | shape cosine | stiff-direction cosine | "
          "log10 trace ratio | affine distance | FD calls | Fishnet sims |")
    print("|---|---|---|---|---|---|---|---|")
    print(
        f"| {summary['problem']} | {summary['n_points']} ({summary['n_fd_converged']}) | "
        f"{cell('shape_cosine')} | {cell('stiff_cosine')} | {cell('log10_trace_ratio')} | "
        f"{cell('affine_dist')} | {summary['fd_simulator_calls']} | "
        f"{summary['fishnet_training_simulations']} |"
    )
    print("Cells are median (16th, 84th percentile) over held-out theta, FD-converged points only.")


# ---------------------------------------------------------------------------
# Self-test: checks the numerics without the drivers (no JAX needed).
# ---------------------------------------------------------------------------
def self_test() -> None:
    from scipy.integrate import odeint
    from scipy.optimize import least_squares

    rng = np.random.default_rng(0)

    # 1. Linear-Gaussian model: FD Fisher equals A^T A / sigma^2 exactly.
    A = rng.normal(size=(12, 3))
    F, calls, ok, _ = fd_fisher(lambda t: A @ t, rng.normal(size=3), np.full(3, 1e-2), 0.3**2)
    assert ok and np.allclose(F, A.T @ A / 0.09, rtol=1e-8), "linear model"
    assert calls == 12, calls

    # 2. Marginalisation equals inverting the full matrix and taking the block.
    F3 = A.T @ A
    prior = np.array([0.0, 0.0, 0.7])
    block = np.linalg.inv(np.linalg.inv(F3 + np.diag(prior))[:2, :2])
    assert np.allclose(marginalise(F3, [0, 1], prior), block), "Schur complement"

    # 3. Rescaling: theta_s = s * theta, so F_s = F / (s s^T).
    s = np.array([2.0, 5.0])
    Fs = to_scaled(F3[:2, :2], s)
    v = rng.normal(size=2)
    assert np.isclose(v @ F3[:2, :2] @ v, (s * v) @ Fs @ (s * v)), "scaling"

    # 4. Metrics: identical matrices, and a pure rescale.
    m = compare(F3, F3)
    assert abs(m["shape_cosine"] - 1) < 1e-12 and m["affine_dist"] < 1e-6 and abs(m["stiff_cosine"] - 1) < 1e-12
    m = compare(10 * F3, F3)
    assert abs(m["log10_trace_ratio"] - 1) < 1e-12 and m["shape_rel_err"] < 1e-12

    # 5. SIR: the FD Fisher predicts the scatter of maximum-likelihood fits.
    n_pop, sigma, t_obs = 1000, 0.01, np.linspace(0, 50, 30)

    def odes(y, _t, beta, gamma):
        S, I, R = y
        return [-beta * S * I / n_pop, beta * S * I / n_pop - gamma * I, gamma * I]

    def mean(t):
        return odeint(odes, [n_pop - 8.0, 8.0, 0.0], t_obs, args=(t[0], t[1]))[:, 1] / n_pop

    truth = np.array([0.6, 0.2])
    F, _, ok, _ = fd_fisher(mean, truth, 1e-3 * np.array([0.9, 0.45]), sigma**2)
    assert ok, "SIR FD did not converge"
    fits = []
    for _ in range(400):
        x = mean(truth) + sigma * rng.normal(size=t_obs.size)
        fits.append(least_squares(lambda t: (mean(t) - x) / sigma, truth).x)
    cov = np.cov(np.array(fits).T)
    rel = np.linalg.norm(np.linalg.inv(cov) - F) / np.linalg.norm(F)
    print(f"[self-test] SIR: FD Fisher vs inverse covariance of 400 ML fits, relative error {rel:.3f}")
    assert rel < 0.2, rel
    print("[self-test] all checks passed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--self-test", action="store_true", help="check the numerics and exit (no drivers needed)")
    p.add_argument("--problem", choices=sorted(DRIVERS))
    p.add_argument("--run-dir", type=Path, help="finished three-step run directory (holds config_manifest.json)")
    p.add_argument("--fishnets", type=Path, default=None, help="fishnets_outputs.npz; default: found under --run-dir")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/fd_fisher_baseline")
    p.add_argument("--mode", default=None, help="driver config; default: from config_manifest.json")
    p.add_argument("--data-seed", type=int, default=None, help="default: stage_seeds.data from the manifest")
    p.add_argument("--noise-std", type=float, default=None, help="SIR only; default: manifest, then driver default")
    p.add_argument("--rel-step", type=float, default=1e-3, help="first FD step as a fraction of the prior width")
    p.add_argument("--rtol", type=float, default=1e-2, help="FD convergence tolerance on the Fisher matrix")
    p.add_argument("--max-halvings", type=int, default=10)
    p.add_argument("--max-points", type=int, default=None, help="subsample held-out points (default: all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.problem or not args.run_dir:
        raise SystemExit("--problem and --run-dir are required (or use --self-test)")
    run(args)


if __name__ == "__main__":
    main()
