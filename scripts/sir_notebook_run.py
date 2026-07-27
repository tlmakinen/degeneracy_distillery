#!/usr/bin/env python
"""Batch-safe version of tutorial_notebooks/sir_example.ipynb.

Recovers R0 = beta / gamma from noisy SIR infection-curve data via the
fishnets -> flatten -> align -> augment -> symbolic-regression pipeline. The
default ``smoke`` mode uses much smaller hyperparameters than the tuned
notebook so a Slurm GPU job can be monitored interactively. ``full`` mode
restores the notebook's tuned settings; ``rebuttal`` freezes the same
architecture/SR settings for the NeurIPS multi-seed rerun (n_train=500, which
already matches the notebook's own tuned nsims default).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from dataclasses import replace as dataclasses_replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import sympy
from scipy.integrate import odeint
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

import esr.generation.generator  # noqa: F401 - sanity-check ESR import.
from degeneracy_distillery.align_coords import load_and_process_data_v2
from degeneracy_distillery.postprocess_new import analyze_atom_sharing, regroup_like_terms
from degeneracy_distillery.postprocessing_utils import (
    check_flattening,
    diagnose_coordinate_rank_deficiency,
    flatten_with_numerical_jacobian,
    print_discovered_expressions,
    weighted_std,
)
from degeneracy_distillery.sr_utils import (
    analyze_equations,
    check_symbolic_invertibility,
    compute_DL,
    expressions_to_physical,
    filter_pareto_fronts,
    fit_symbolic_regression,
    fit_theta_scaler,
    sr_structure_predicate,
)
from degeneracy_distillery.training_loop_fishnets import train_fishnets
from degeneracy_distillery.training_loop_flatten import fit_flattening


# ---------------------------------------------------------------------------
# Frozen SIR simulator constants (matches tutorial_notebooks/sir_example.ipynb).
# ---------------------------------------------------------------------------
N_POP = 1000
I0_MEAN = 8
N_TIMEPOINTS = 30
T_MAX = 50.0
T_OBS = np.linspace(0, T_MAX, N_TIMEPOINTS)
NOISE_STD = 0.05
BETA_MIN, BETA_MAX = 0.1, 1.0
GAMMA_MIN, GAMMA_MAX = 0.05, 0.5
DELTA = 0.15  # supercriticality cut: keep beta/gamma >= 1 + delta
SR_OFFSET = 0.0
SR_LENGTH_PENALTY = 2.0
INVERTIBILITY_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class RunConfig:
    nsims: int
    num_fishnets: int
    fish_hids_min: int
    fish_hids_max: int
    fish_layers: tuple[int, int]
    fish_epochs: int
    fish_min_epochs: int
    fish_patience: int
    fish_train_batch_size: int
    flatten_n_layers: int
    flatten_finetune_epochs: int
    flatten_epochs_phase1: int
    flatten_epochs_phase2: int
    flatten_batch_size: int
    align_subsample: int
    sr_n_sr: int
    sr_time_limit: int
    sr_max_length: int
    sr_max_depth: int
    npe_epochs: int
    npe_hidden_features: int
    npe_num_transforms: int
    npe_repeats_maf: int
    npe_batch_size: int
    npe_learning_rate: float
    npe_posterior_samples: int
    npe_eval_points: int


CONFIGS = {
    "smoke": RunConfig(
        nsims=80,
        num_fishnets=3,
        fish_hids_min=20,
        fish_hids_max=48,
        fish_layers=(2, 3),
        fish_epochs=200,
        fish_min_epochs=20,
        fish_patience=10,
        fish_train_batch_size=16,
        flatten_n_layers=4,
        flatten_finetune_epochs=30,
        flatten_epochs_phase1=100,
        flatten_epochs_phase2=50,
        flatten_batch_size=16,
        align_subsample=300,
        sr_n_sr=600,
        sr_time_limit=20,
        sr_max_length=15,
        sr_max_depth=6,
        npe_epochs=20,
        npe_hidden_features=32,
        npe_num_transforms=3,
        npe_repeats_maf=2,
        npe_batch_size=16,
        npe_learning_rate=1e-3,
        npe_posterior_samples=200,
        npe_eval_points=10,
    ),
    "full": RunConfig(
        nsims=500,
        num_fishnets=10,
        fish_hids_min=50,
        fish_hids_max=300,
        fish_layers=(2, 5),
        fish_epochs=5000,
        fish_min_epochs=100,
        fish_patience=30,
        fish_train_batch_size=50,
        flatten_n_layers=8,
        flatten_finetune_epochs=200,
        flatten_epochs_phase1=1000,
        flatten_epochs_phase2=500,
        flatten_batch_size=50,
        align_subsample=4000,
        sr_n_sr=10000,
        sr_time_limit=60,
        sr_max_length=25,
        sr_max_depth=10,
        npe_epochs=1000,
        npe_hidden_features=50,
        npe_num_transforms=5,
        npe_repeats_maf=2,
        npe_batch_size=32,
        npe_learning_rate=1e-4,
        npe_posterior_samples=1000,
        npe_eval_points=200,
    ),
}

# NeurIPS rebuttal configuration: same frozen architecture/optimizer/SR settings
# as "full" (no per-seed retuning). n_train stays at the notebook's own tuned
# default of 500, which already matches the rebuttal protocol's n_train=500;
# n_sr (SR augmentation draws) also stays at the notebook's frozen value of
# 10,000 rather than being bumped to 2,000, per the rebuttal protocol's
# "unless a frozen existing configuration requires another value" carve-out.
CONFIGS["rebuttal"] = dataclasses_replace(CONFIGS["full"], nsims=500)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require_gpu_if_requested(require_gpu: bool) -> None:
    backend = jax.default_backend()
    devices = jax.devices()
    log(f"JAX backend: {backend}")
    log(f"JAX devices: {devices}")
    if require_gpu and backend != "gpu":
        raise SystemExit(
            "JAX did not initialize a GPU backend. This job should run on a GPU node."
        )


# Additive-stride master-seed derivation, matching the convention already used in
# scripts/rosen_nsims_logprob_sweep.py (run_seed = args.seed + nsims*10 + offset)
# and copied verbatim from scripts/rosenbrock_notebook_run.py. The stride is
# large enough that master seeds 0-9 never collide across stages.
STAGE_SEED_STRIDE = 10_000
STAGE_OFFSETS = {
    "data": 0,
    "fish_model": 1,
    "fish_train": 2,
    "flatten": 3,
    "align": 4,
    "sr_grid": 5,
    "sr_fit": 6,
    "validation": 7,
}


def derive_stage_seeds(master_seed: int) -> dict[str, int]:
    return {name: master_seed + offset * STAGE_SEED_STRIDE for name, offset in STAGE_OFFSETS.items()}


class _TimeoutError(Exception):
    pass


class time_limit:
    """SIGALRM-based hard timeout. sympy.solve can pathologically hang on messy
    float-coefficient rational expressions; this diagnostic is supplementary, so
    it must never be allowed to stall an entire (cluster) run."""

    def __init__(self, seconds: int):
        self.seconds = seconds

    def _raise(self, signum, frame):
        raise _TimeoutError(f"timed out after {self.seconds}s")

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._raise)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)


def git_commit_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage 1: data generation.
# ---------------------------------------------------------------------------
def sir_odes(y, _t, beta, gamma, N):
    S, I, R = y
    return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]


def _simulate_sir_curve(beta: float, gamma: float, noise_key) -> tuple[np.ndarray, float]:
    I0 = np.random.poisson(I0_MEAN)
    S0 = N_POP - I0
    sol = odeint(sir_odes, [S0, I0, 0.0], T_OBS, args=(beta, gamma, N_POP))
    I_t = sol[:, 1] / N_POP
    I_t = I_t + np.array(jr.normal(noise_key, shape=I_t.shape)) * NOISE_STD
    return I_t.astype(np.float32), I0 / 10.0


def _generate(rng_key, n: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    key1, key2 = jr.split(rng_key)
    pool = 4 * n
    beta_pool = np.random.uniform(BETA_MIN, BETA_MAX, pool)
    gamma_pool = np.random.uniform(GAMMA_MIN, GAMMA_MAX, pool)
    keep = (beta_pool / gamma_pool) >= (1.0 + DELTA)
    beta_pool = beta_pool[keep][:n]
    gamma_pool = gamma_pool[keep][:n]
    if len(beta_pool) < n:
        raise RuntimeError(
            f"supercriticality cut left only {len(beta_pool)}/{n} samples for '{label}'; "
            "increase the oversampling pool."
        )
    keys = jr.split(key2, n)
    data, i0s = [], []
    for i in tqdm(range(n), desc=label):
        d, i0 = _simulate_sir_curve(beta_pool[i], gamma_pool[i], keys[i])
        data.append(d)
        i0s.append(i0)
    theta = np.stack([beta_pool, gamma_pool, np.array(i0s)], axis=1).astype(np.float32)
    return theta, np.array(data, dtype=np.float32)


def simulator_data(config: RunConfig, seed: int) -> dict[str, np.ndarray]:
    # Dual-seeding matches the tuned notebook exactly: both the global NumPy RNG
    # (used by np.random.uniform/poisson inside generate()/simulate_sir) and the
    # JAX PRNGKey (used for per-sample noise via jr.split) are set from the same
    # derived stage seed.
    np.random.seed(seed)
    key = jr.PRNGKey(seed)
    key, sub = jr.split(key)
    theta_train, data_train = _generate(sub, config.nsims, "train")
    key, sub = jr.split(key)
    theta_test, data_test = _generate(sub, config.nsims, "test")
    log(f"theta_train {theta_train.shape}; data_train {data_train.shape}")
    return {
        "theta_train": theta_train,
        "data_train": data_train,
        "theta_test": theta_test,
        "data_test": data_test,
    }


def fit_scaler(data: dict[str, np.ndarray]):
    scaler = fit_theta_scaler(data["theta_train"], feature_range=(1.0, 2.0))
    theta_train_s = scaler.transform(data["theta_train"]).astype(np.float32)
    theta_test_s = scaler.transform(data["theta_test"]).astype(np.float32)
    log(f"scaled theta range: {theta_train_s.min(0)} to {theta_train_s.max(0)}")
    return scaler, theta_train_s, theta_test_s


def train_fishnet_ensemble(
    config: RunConfig,
    data: dict[str, np.ndarray],
    theta_train_s: np.ndarray,
    theta_test_s: np.ndarray,
    outdir: Path,
    seeds: dict[str, int],
) -> Path:
    embedding_net = nn.Sequential([nn.Dense(64), nn.gelu, nn.Dense(32), nn.gelu])
    fish_dir = outdir / "fishnets-sir"
    log(f"training fishnets into {fish_dir}")
    train_fishnets(
        theta_train_s[:, :2],
        data["data_train"],
        theta_test_s[:, :2],
        data["data_test"],
        num_models=config.num_fishnets,
        train_epochs=config.fish_epochs,
        train_min_epochs=config.fish_min_epochs,
        patience=config.fish_patience,
        n_layers=list(config.fish_layers),
        hids_min=config.fish_hids_min,
        hids_max=config.fish_hids_max,
        embedding_net=embedding_net,
        lr=5e-5,
        train_batch_size=config.fish_train_batch_size,
        seed_model=seeds["fish_model"],
        seed_train=seeds["fish_train"],
        outdir=str(fish_dir),
        update_pbar_every=25,
    )
    return fish_dir


# ---------------------------------------------------------------------------
# Stage 2: flattening.
# ---------------------------------------------------------------------------
def fit_flattener(config: RunConfig, fish_dir: Path, outdir: Path, seeds: dict[str, int]):
    with np.load(fish_dir / "fishnets_outputs.npz") as fish:
        thetas = jnp.array(fish["theta"])
        ensemble_weights = np.asarray(fish["ensemble_weights"])
        fs_np = np.asarray(fish["Fs"])

    finite_mask = np.isfinite(fs_np).all(axis=(1, 2, 3))
    if not finite_mask.any():
        raise RuntimeError("all fishnet ensemble members produced non-finite Fishers")
    if finite_mask.sum() < len(finite_mask):
        log(f"filtering {len(finite_mask) - int(finite_mask.sum())} non-finite fishnet members")
        fs_np = fs_np[finite_mask]
        ensemble_weights = ensemble_weights[finite_mask]

    log("fitting flattening model")
    cwd_before = Path.cwd()
    os.chdir(outdir)
    try:
        _w, ensemble_ws, _outputs_flatten, flatten_model = fit_flattening(
            jnp.array(fs_np),
            thetas,
            ensemble_weights=ensemble_weights,
            flattener_activation="softplus",
            loss_type="log_frob",
            forward_backward_mlp=True,
            forward_backward_invertibility_weight=1.0,
            n_layers=config.flatten_n_layers,
            offset=0.0,
            beta_det=0.1,
            noise=1e-2,
            batch_size=config.flatten_batch_size,
            finetune_epochs=config.flatten_finetune_epochs,
            epochs_phase1=config.flatten_epochs_phase1,
            epochs_phase2=config.flatten_epochs_phase2,
            lr_phase1=2e-6,
            lr_schedule_initial=7e-5,
            lr_decay=0.3,
            l1_alpha=0.0,
            do_plot=False,
            Fisher_to_flatten="average",
            output_prefix="sir_flatten",
            seed=seeds["flatten"],
            return_model=True,
            update_pbar_every=25,
        )
    finally:
        os.chdir(cwd_before)

    return ensemble_ws, flatten_model


# ---------------------------------------------------------------------------
# Stage 3: SR augmentation pool + coordinate alignment.
#
# The tuned notebook draws the augmented (theta, eta, delta-eta) SR pool BEFORE
# the alignment call (using the scaler + per-ensemble-member flatten model
# directly), then rotates that pool into the aligned frame using the rotation
# matrices produced by alignment. This ordering is preserved exactly.
# ---------------------------------------------------------------------------
def align_and_augment(
    config: RunConfig,
    outdir: Path,
    scaler,
    ensemble_ws,
    flatten_model,
    seeds: dict[str, int],
) -> dict:
    log("drawing SR augmentation pool")
    rng_sr = np.random.default_rng(seeds["sr_grid"])
    n_sr = config.sr_n_sr
    beta_sr = rng_sr.uniform(BETA_MIN, BETA_MAX, n_sr)
    gamma_sr = rng_sr.uniform(GAMMA_MIN, GAMMA_MAX, n_sr)
    i0_sr = rng_sr.poisson(I0_MEAN, n_sr) / 10.0
    theta_sr_phys = np.stack([beta_sr, gamma_sr, i0_sr], axis=1).astype(np.float32)

    # Supercriticality cut, mirroring the training set.
    keep = (beta_sr / gamma_sr) >= (1.0 + DELTA)
    theta_sr_phys = theta_sr_phys[keep]
    n_sr_post_cut = int(theta_sr_phys.shape[0])
    log(f"SR augmentation pool: {n_sr} drawn, {n_sr_post_cut} pass supercriticality cut")

    # Scale and select the (beta_s, gamma_s) channels the fishnet was trained on.
    x_sr = scaler.transform(theta_sr_phys)[:, :2].astype(np.float32)
    ys_sr = jnp.array(
        [jax.vmap(lambda x: flatten_model.apply(w_i, x))(x_sr) for w_i in ensemble_ws]
    )
    log(f"X_sr {x_sr.shape}; ys_sr {ys_sr.shape}")

    log("aligning coordinates")
    aligned = load_and_process_data_v2(
        datapath=str(outdir) + os.sep,
        filename="sir_flatten.npz",
        num_samps=config.align_subsample,
        seed=seeds["align"],
        process_ensemble=True,
        n_d=1.0,
        align_mode="procrustes",
        separate_nonlinearity=False,
        canonicalize="sign_only",
        use_prior_normalization=True,
        restore_reference_mean=False,
        Fisher_to_flatten="average",
        verbose=False,
    )

    x_full = np.asarray(aligned["X"])
    mask = x_full[:, 0] > 0.0  # standard mask for the SIR run, matches the notebook.
    x = x_full[mask]
    y = np.asarray(aligned["y"])[mask]
    y_std = np.asarray(aligned["y_std"])[mask]
    dy_sr = np.asarray(aligned["dy_sr"])[mask]
    Fs = np.asarray(aligned["Fs"])[mask]
    ys = np.array([np.asarray(yy)[mask] for yy in aligned["ys"]])
    n_params = x.shape[1]

    # Floor y so y >= 1 (PyOperon-friendly), matching the tuned notebook.
    y_offset = y.min(0) - 1.0
    y = y - y_offset
    ys = ys - y_offset

    # Rotate the SR pool into the same aligned frame.
    ys_sr_rot = np.array(
        [
            np.einsum("ij,bj->bi", aligned["rotmats"][i], ys_sr[i] - ys_sr[i].mean(0))
            for i in range(len(ys_sr))
        ]
    )
    y_std_sr = weighted_std(ys_sr_rot, aligned["ensemble_weights"])
    y_sr = np.average(ys_sr_rot, 0, aligned["ensemble_weights"])
    ys_sr_rot = ys_sr_rot - y_sr.min(0)
    y_sr = y_sr - y_sr.min(0)
    y_sr = y_sr + 1.0
    ys_sr_rot = ys_sr_rot + 1.0
    log(f"aligned X {x.shape}; y {y.shape}")

    return {
        "X": x,
        "y": y,
        "ys": ys,
        "y_std": y_std,
        "dy_sr": dy_sr,
        "Fs": Fs,
        "n_params": n_params,
        "X_sr": np.asarray(x_sr),
        "y_sr": y_sr,
        "y_std_sr": y_std_sr,
        "n_sr_drawn": n_sr,
        "n_sr_post_cut": n_sr_post_cut,
    }


# ---------------------------------------------------------------------------
# Stage 4: symbolic regression + MDL ranking.
# ---------------------------------------------------------------------------
def run_symbolic_regression(config: RunConfig, aligned: dict, outdir: Path, seeds: dict[str, int]):
    sr_dir = outdir / "sr_results_sir"
    sr_dir.mkdir(exist_ok=True)
    log(f"running symbolic regression into {sr_dir}")
    fit_symbolic_regression(
        aligned["X_sr"],
        aligned["y_sr"],
        aligned["y_std_sr"],
        parent_dir=str(sr_dir) + os.sep,
        random_state=seeds["sr_fit"],
        time_limit=config.sr_time_limit,
        max_length=config.sr_max_length,
        max_depth=config.sr_max_depth,
        allowed_symbols="add,mul,div,pow,constant,variable",
        verbose=True,
    )

    equation_predicate = sr_structure_predicate(
        n_params=aligned["n_params"],
        forbid_self_transcendental=True,
        check_nested_exp=False,
    )
    filter_summaries = filter_pareto_fronts(str(sr_dir), aligned["n_params"], equation_predicate)
    removed = sum(int(summary["removed"]) for summary in filter_summaries)
    log(f"removed {removed} self-transcendental/invalid equations from Pareto fronts")

    mdl_coords, frob_coords, analysis = analyze_equations(
        aligned["X"] + SR_OFFSET,
        aligned["y"],
        aligned["y_std"],
        aligned["dy_sr"],
        aligned["Fs"],
        parent_dir=str(sr_dir) + os.sep,
        n_params=aligned["n_params"],
        equation_set="pareto",
        max_complexity_thresh=20,
        length_penalty=SR_LENGTH_PENALTY,
        equation_predicate=equation_predicate,
        verbose=True,
    )
    return sr_dir, mdl_coords, frob_coords, analysis


def r0_correlations_and_gradients(physical_exprs, theta_test: np.ndarray) -> dict[str, object]:
    """Held-out Pearson/Spearman correlation with R0=beta/gamma, plus gradient
    cosine similarity between each discovered component's Jacobian and grad(R0)
    in physical (beta, gamma, I0_over_10) space, evaluated at held-out points.
    """
    beta_sym, gamma_sym, i0_sym = sympy.symbols("beta gamma I0_over_10")
    symbols_tuple = (beta_sym, gamma_sym, i0_sym)

    beta_t = theta_test[:, 0].astype(float)
    gamma_t = theta_test[:, 1].astype(float)
    i0_t = theta_test[:, 2].astype(float)
    r0 = beta_t / gamma_t

    grad_r0 = np.stack([1.0 / gamma_t, -beta_t / gamma_t**2, np.zeros_like(beta_t)], axis=1)
    grad_r0_norm = grad_r0 / np.linalg.norm(grad_r0, axis=1, keepdims=True)

    rows = []
    for i, expr in enumerate(physical_exprs):
        fn = sympy.lambdify(symbols_tuple, expr, modules="numpy")
        values = np.asarray(fn(beta_t, gamma_t, i0_t), dtype=float)
        values = np.broadcast_to(values, beta_t.shape).astype(float)

        if np.std(values) == 0 or not np.all(np.isfinite(values)):
            pearson_r, pearson_p = 0.0, 1.0
            spearman_r, spearman_p = 0.0, 1.0
        else:
            pearson_r, pearson_p = pearsonr(values, r0)
            spearman_r, spearman_p = spearmanr(values, r0)

        grads = []
        for sym in symbols_tuple:
            d_fn = sympy.lambdify(symbols_tuple, sympy.diff(expr, sym), modules="numpy")
            d_vals = np.asarray(d_fn(beta_t, gamma_t, i0_t), dtype=float)
            d_vals = np.broadcast_to(d_vals, beta_t.shape).astype(float)
            grads.append(d_vals)
        grad_expr = np.stack(grads, axis=1)
        norms = np.linalg.norm(grad_expr, axis=1)
        valid = np.isfinite(norms) & (norms > 1e-12)
        cos_sim = np.full(beta_t.shape[0], np.nan)
        cos_sim[valid] = np.sum(grad_expr[valid] * grad_r0_norm[valid], axis=1) / norms[valid]
        median_abs_grad_cosine = float(np.nanmedian(np.abs(cos_sim))) if valid.any() else 0.0

        rows.append(
            {
                "component": i,
                "expr": str(expr),
                "pearson_r0": float(pearson_r),
                "pearson_r0_pvalue": float(pearson_p),
                "spearman_r0": float(spearman_r),
                "spearman_r0_pvalue": float(spearman_p),
                "median_abs_grad_cosine_r0": median_abs_grad_cosine,
            }
        )

    best_idx = int(np.argmax([abs(row["pearson_r0"]) for row in rows]))
    best = rows[best_idx]
    return {
        "rows": rows,
        "best_component": best_idx,
        "best_pearson_abs": abs(best["pearson_r0"]),
        "best_spearman_abs": abs(best["spearman_r0"]),
        "best_grad_cosine": best["median_abs_grad_cosine_r0"],
    }


def validate_flatness(aligned: dict, scaler, mdl_coords, pruned_exprs) -> dict[str, float]:
    X_test = aligned["X"] + SR_OFFSET
    nn_flats = jax.vmap(flatten_with_numerical_jacobian)(aligned["dy_sr"], aligned["Fs"])
    mdl_flats, _ = check_flattening(mdl_coords, X=X_test, Fs=aligned["Fs"])
    pruned_flats, _ = check_flattening(pruned_exprs, X=X_test, Fs=aligned["Fs"])
    adhoc_coords = ["X1 / X2", "X1"]  # ad-hoc physics reference, matching the notebook.
    adhoc_flats, _ = check_flattening(adhoc_coords, X=X_test, Fs=aligned["Fs"])

    # Put the raw Fishers back into physical theta units (via the minmax scaler),
    # matching the notebook's "raw theta" reference row.
    delta_scale = (scaler.data_max_ - scaler.data_min_)[:2]
    Fs_vanilla = aligned["Fs"] / (delta_scale**2)

    def fro_score(Q):
        Q = np.asarray(Q)
        eye = np.eye(aligned["n_params"])
        return np.linalg.norm(Q - eye, axis=(-2, -1)) + np.linalg.norm(
            np.linalg.inv(Q) - eye, axis=(-2, -1)
        )

    return {
        "raw_theta": float(np.median(fro_score(Fs_vanilla))),
        "adhoc": float(np.median(fro_score(adhoc_flats))),
        "mdl": float(np.median(fro_score(mdl_flats))),
        "pruned": float(np.median(fro_score(pruned_flats))),
        "nn": float(np.median(fro_score(nn_flats))),
        "median_condition_raw": float(np.median(np.linalg.cond(np.asarray(Fs_vanilla)))),
        "median_condition_symbolic": float(np.median(np.linalg.cond(np.asarray(pruned_flats)))),
    }


# ---------------------------------------------------------------------------
# Stage 5 (secondary, time-boxed): downstream NPE + CRPS + coverage.
#
# No CRPS/coverage utility exists anywhere in this repo. This adapts the
# train_posterior / sample_one_observation machinery already built for the
# (unrelated) nsims-sweep pipeline in scripts/sir_nsims_logprob_sweep.py,
# rather than duplicating an NPE stack from scratch. properscoring is not
# installed in this venv, so CRPS is computed directly from the standard
# empirical formula CRPS(F, y) ~= E|X-y| - 0.5*E|X-X'|.
# ---------------------------------------------------------------------------
def empirical_crps_batch(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """samples: (n_samples, n_points); y: (n_points,) -> CRPS per point (n_points,).

    Uses the O(n log n) order-statistic identity for E|X-X'| (sort once per
    column) rather than the naive O(n_samples^2) pairwise-difference sum.
    Non-finite posterior draws (e.g. an amortized posterior placing mass near
    gamma=0, blowing up R0=beta/gamma) are dropped per-column before scoring,
    rather than allowed to contaminate the whole column via a stray inf/nan --
    the eval-point budget here is small (<=200), so a per-column loop is cheap.
    """
    samples = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_points = y.shape[0]
    crps = np.full(n_points, np.nan)
    for j in range(n_points):
        col = samples[:, j]
        col = col[np.isfinite(col)]
        n = col.size
        if n < 2:
            continue
        term1 = np.mean(np.abs(col - y[j]))
        sorted_col = np.sort(col)
        idx = np.arange(1, n + 1, dtype=np.float64)
        term2 = 2.0 * np.sum((2.0 * idx - n - 1.0) * sorted_col) / (n * n)
        crps[j] = term1 - 0.5 * term2
    return crps


def pit_coverage_error(samples: np.ndarray, true_values: np.ndarray) -> tuple[np.ndarray, float]:
    """Adapted from compute_coverage_diagnostics's per-point PIT/rank logic
    (scripts/sir_nsims_logprob_sweep.py), specialized to a scalar *derived*
    quantity (R0 or a discovered eta component) rather than a raw
    theta-parameter marginal -- compute_coverage_diagnostics assumes a 1:1
    correspondence between sample and true-value dimensions, which a derived
    ratio/expression does not have.

    Returns (pit_values, coverage_error): coverage_error is the mean absolute
    deviation of the empirical PIT CDF from Uniform(0, 1) -- ~0 for a
    well-calibrated posterior, larger under over/under-confidence.
    """
    samples = np.asarray(samples, dtype=np.float64)
    true_values = np.asarray(true_values, dtype=np.float64)
    n_points = true_values.shape[0]
    pit = np.full(n_points, np.nan)
    for j in range(n_points):
        col = samples[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        pit[j] = float((col < true_values[j]).sum()) / col.size
    valid_pit = np.sort(pit[np.isfinite(pit)])
    if valid_pit.size == 0:
        return pit, float("nan")
    empirical_cdf = (np.arange(1, valid_pit.size + 1) - 0.5) / valid_pit.size
    coverage_error = float(np.mean(np.abs(valid_pit - empirical_cdf)))
    return pit, coverage_error


def run_downstream_npe(
    config: RunConfig,
    data: dict[str, np.ndarray],
    physical_exprs,
    best_idx: int,
    seeds: dict[str, int],
) -> dict | None:
    """Returns None (leave inference.* fields null) if ltu-ili/torch aren't
    importable or if anything in this stage goes wrong -- this must never
    fail the overall run, per the rebuttal spec's explicit instruction to
    leave CRPS/coverage null rather than fabricate a number."""
    try:
        import torch

        script_dir = Path(__file__).resolve().parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from sir_nsims_logprob_sweep import (  # type: ignore
            sample_one_observation,
            train_posterior,
        )
    except Exception as exc:
        log(f"NPE/CRPS unavailable ({type(exc).__name__}: {exc}); leaving inference fields null")
        return None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"training downstream NPE posterior on device={device}")
        npe_args = SimpleNamespace(
            hidden_features=config.npe_hidden_features,
            num_transforms=config.npe_num_transforms,
            repeats_maf=config.npe_repeats_maf,
            batch_size=config.npe_batch_size,
            learning_rate=config.npe_learning_rate,
            epochs=config.npe_epochs,
        )
        prior_low = np.array([BETA_MIN, GAMMA_MIN, 0.0], dtype=np.float32)
        prior_high = np.array([BETA_MAX, GAMMA_MAX, 5.0], dtype=np.float32)

        posterior, _summaries = train_posterior(
            data["data_train"],
            data["theta_train"],
            prior_low,
            prior_high,
            npe_args,
            device,
            seed=seeds["validation"],
        )

        n_eval = min(config.npe_eval_points, data["theta_test"].shape[0])
        theta_true = data["theta_test"][:n_eval]
        data_obs = data["data_test"][:n_eval]
        n_samples = config.npe_posterior_samples

        posterior_samples = np.full(
            (n_samples, n_eval, theta_true.shape[1]), np.nan, dtype=np.float32
        )
        for i in range(n_eval):
            posterior_samples[:, i, :] = sample_one_observation(
                posterior, data_obs[i], n_samples, device
            )

        finite = np.isfinite(posterior_samples).all(axis=-1)
        log(f"NPE posterior samples: {int(finite.sum())}/{finite.size} finite")

        r0_true = theta_true[:, 0] / theta_true[:, 1]
        r0_samples = posterior_samples[:, :, 0] / posterior_samples[:, :, 1]
        crps_theta = float(np.nanmedian(empirical_crps_batch(r0_samples, r0_true)))
        _pit_theta, coverage_error_theta = pit_coverage_error(r0_samples, r0_true)

        eta_expr = physical_exprs[best_idx]
        beta_sym, gamma_sym, i0_sym = sympy.symbols("beta gamma I0_over_10")
        eta_fn = sympy.lambdify((beta_sym, gamma_sym, i0_sym), eta_expr, modules="numpy")

        eta_true = np.broadcast_to(
            np.asarray(eta_fn(theta_true[:, 0], theta_true[:, 1], theta_true[:, 2]), dtype=float),
            (n_eval,),
        )
        eta_samples = np.broadcast_to(
            np.asarray(
                eta_fn(
                    posterior_samples[:, :, 0], posterior_samples[:, :, 1], posterior_samples[:, :, 2]
                ),
                dtype=float,
            ),
            (n_samples, n_eval),
        )
        crps_eta = float(np.nanmedian(empirical_crps_batch(eta_samples, eta_true)))
        _pit_eta, coverage_error_eta = pit_coverage_error(eta_samples, eta_true)

        log(
            f"CRPS: R0={crps_theta:.4f} (coverage_error={coverage_error_theta:.4f}), "
            f"eta_{best_idx}={crps_eta:.4f} (coverage_error={coverage_error_eta:.4f})"
        )
        if coverage_error_theta > 0.15 or coverage_error_eta > 0.15:
            log(
                "WARNING: coverage_error exceeds 0.15 -- the posterior's PIT "
                "distribution looks meaningfully non-uniform; treat CRPS "
                "sharpness with caution (possible overconfidence/miscalibration)."
            )

        return {
            "n_eval_points": n_eval,
            "n_posterior_samples": n_samples,
            "crps_theta": crps_theta,
            "crps_eta": crps_eta,
            "coverage_error_theta": coverage_error_theta,
            "coverage_error_eta": coverage_error_eta,
        }
    except Exception as exc:
        log(
            f"NPE/CRPS stage failed non-fatally ({type(exc).__name__}: {exc}); "
            "leaving inference fields null"
        )
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIGS), default="smoke")
    parser.add_argument("--out-dir", type=Path, default=Path("results/sir_notebook_smoke"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--min-r0-corr",
        type=float,
        default=0.5,
        help="Fail discovery.success if no physical expression correlates this "
        "strongly (|Pearson r|) with R0 = beta / gamma.",
    )
    parser.add_argument(
        "--skip-npe",
        action="store_true",
        help="Skip the downstream NPE/CRPS/coverage stage entirely (inference "
        "fields stay null); the core discovery pipeline still runs and is scored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.mode]
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"running mode={args.mode}; outdir={outdir}")
    log(f"config={json.dumps(asdict(config), sort_keys=True)}")

    seeds = derive_stage_seeds(args.seed)
    log(f"stage seeds={json.dumps(seeds, sort_keys=True)}")

    run_id = f"sir_seed{args.seed}"
    counts = {
        "n_train_simulations": config.nsims,
        "n_eval_simulations": config.nsims,
        "n_pca_simulations": 0,
        "n_augmented_coordinate_evaluations": None,
        "n_downstream_npe_simulations": 0,
    }
    config_manifest = {
        "run_id": run_id,
        "problem": "sir",
        "master_seed": args.seed,
        "mode": args.mode,
        "config": asdict(config),
        "stage_seeds": seeds,
        "thresholds": {"min_r0_corr": args.min_r0_corr},
        "git_commit": git_commit_hash(),
    }
    with open(outdir / "config_manifest.json", "w") as handle:
        json.dump(config_manifest, handle, indent=2, sort_keys=True)

    require_gpu_if_requested(args.require_gpu)

    runtime_seconds: dict[str, float | None] = {}
    total_start = time.time()

    def write_failure(stage: str, exc: Exception) -> None:
        runtime_seconds["total"] = time.time() - total_start
        record = {
            "run_id": run_id,
            "problem": "sir",
            "master_seed": args.seed,
            "status": "failed",
            "failure_stage": stage,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(),
            "counts": counts,
            "runtime_seconds": runtime_seconds,
        }
        with open(outdir / "run_record.json", "w") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
        log(f"FAILED at stage={stage}: {exc}")

    try:
        stage_start = time.time()
        data = simulator_data(config, seeds["data"])
        scaler, theta_train_s, theta_test_s = fit_scaler(data)
        fish_dir = train_fishnet_ensemble(config, data, theta_train_s, theta_test_s, outdir, seeds)
        runtime_seconds["fishnets"] = time.time() - stage_start
    except Exception as exc:
        write_failure("fishnets", exc)
        raise

    try:
        stage_start = time.time()
        ensemble_ws, flatten_model = fit_flattener(config, fish_dir, outdir, seeds)
        runtime_seconds["flatten"] = time.time() - stage_start
    except Exception as exc:
        write_failure("flatten", exc)
        raise

    try:
        # Combines the SR-augmentation draw (fresh theta -> flatten-model push
        # through, i.e. the "augmented coordinate evaluations" stage) with
        # coordinate alignment, per the notebook's own ordering.
        stage_start = time.time()
        aligned = align_and_augment(config, outdir, scaler, ensemble_ws, flatten_model, seeds)
        runtime_seconds["alignment"] = time.time() - stage_start
        counts["n_augmented_coordinate_evaluations"] = aligned["n_sr_post_cut"]
    except Exception as exc:
        write_failure("alignment", exc)
        raise

    try:
        stage_start = time.time()
        sr_dir, mdl_coords, frob_coords, analysis = run_symbolic_regression(
            config, aligned, outdir, seeds
        )

        log("MDL coordinates")
        print_discovered_expressions([sympy.simplify(e).evalf(2) for e in mdl_coords])

        log("postprocessing expressions")
        analyze_atom_sharing(mdl_coords)
        pruned_exprs, rotation, prune_info = regroup_like_terms(
            mdl_coords,
            X=aligned["X"],
            Fs=aligned["Fs"],
            n_params=aligned["n_params"],
            method="atoms",
            do_snap=True,
            snap_rel_tol=0.1,
            snap_flat_tol=0.1,
            decimal=2,
            threshold=1.0,
        )
        print_discovered_expressions([sympy.simplify(e).evalf(2) for e in pruned_exprs])

        try:
            with time_limit(INVERTIBILITY_TIMEOUT_SECONDS):
                invertibility = check_symbolic_invertibility(pruned_exprs, verbose=False)
        except _TimeoutError:
            log(
                f"check_symbolic_invertibility did not finish within "
                f"{INVERTIBILITY_TIMEOUT_SECONDS}s; sympy.solve can hang on messy "
                "float-coefficient systems. Recording as unknown rather than blocking."
            )
            invertibility = {"is_symbolically_invertible": None, "timed_out": True}
        except Exception as exc:
            # sympy.solve can also fail outright (NotImplementedError etc.) on messy
            # float-coefficient systems, not just hang -- confirmed in local testing
            # (e.g. "could not solve 15.66*X2**0.65922 + ... "). This diagnostic is
            # supplementary and must never fail the whole run.
            log(
                f"check_symbolic_invertibility raised {type(exc).__name__}: {exc}; "
                "recording as unknown rather than failing the run."
            )
            invertibility = {"is_symbolically_invertible": None, "error": str(exc)}
        rank_info = diagnose_coordinate_rank_deficiency(
            pruned_exprs,
            X=aligned["X"],
            Fs=aligned["Fs"],
            n_params=aligned["n_params"],
        )

        physical_exprs = expressions_to_physical(
            pruned_exprs,
            scaler,
            sr_offset=SR_OFFSET,
            theta_names=("beta", "gamma", "I0_over_10"),
            decimal=3,
        )
        log("physical expressions (expect one to collapse toward beta / gamma)")
        for k, expr in enumerate(physical_exprs):
            print(f"  eta_{k} = {expr}", flush=True)

        correlations = r0_correlations_and_gradients(physical_exprs, data["theta_test"])
        log("R0 correlations/gradients")
        print(json.dumps(correlations, indent=2, sort_keys=True), flush=True)

        flatness = validate_flatness(aligned, scaler, mdl_coords, pruned_exprs)
        log("flatness scores")
        print(json.dumps(flatness, indent=2, sort_keys=True), flush=True)

        runtime_seconds["symbolic_regression"] = time.time() - stage_start
    except Exception as exc:
        write_failure("symbolic_regression", exc)
        raise

    # analysis["DL"] is per-component *normalized* (min-subtracted, so the winning
    # entry is always 0) -- not useful as a total. Recompute the raw DL/complexity
    # of the actual winning (mdl_coords) expressions directly via compute_DL.
    mdl_total = 0.0
    complexity_total = 0.0
    for i, eq in enumerate(mdl_coords):
        c_i, _, _, dl_i, _ = compute_DL(
            eq,
            i,
            aligned["X"],
            aligned["y"],
            aligned["y_std"],
            aligned["dy_sr"],
            aligned["Fs"],
            aligned["n_params"],
            length_penalty=SR_LENGTH_PENALTY,
        )
        mdl_total += float(dl_i)
        complexity_total += float(c_i)

    sr_dir.mkdir(exist_ok=True)
    with open(sr_dir / "sr_expressions.pkl", "wb") as handle:
        pickle.dump(
            {
                "X_test": aligned["X"],
                "Fs_test": aligned["Fs"],
                "mdl_coords": mdl_coords,
                "frob_coords": frob_coords,
                "pruned_exprs": pruned_exprs,
                "physical_exprs": [str(e) for e in physical_exprs],
                "correlations": correlations,
                "flatness": flatness,
                "analysis": analysis,
                "rotation": rotation,
                "prune_info": prune_info,
                "invertibility": invertibility,
                "rank_info": rank_info,
                "scaler_scale": scaler.scale_,
                "scaler_min": scaler.min_,
                "scaler_data_min": scaler.data_min_,
                "scaler_data_max": scaler.data_max_,
                "sr_offset": SR_OFFSET,
            },
            handle,
        )
    shutil.copytree(fish_dir, sr_dir / "fishnets-sir", dirs_exist_ok=True)
    shutil.copy2(outdir / "sir_flatten.npz", sr_dir / "sir_flatten.npz")
    flatten_model_pickle = outdir / "sir_flatten_flatten_model.pkl"
    if flatten_model_pickle.exists():
        shutil.copy2(flatten_model_pickle, sr_dir / "sir_flatten_flatten_model.pkl")
    shutil.make_archive(str(outdir / "sr_results_sir"), "zip", root_dir=sr_dir)
    log(f"saved artifacts under {sr_dir}")

    # --- Stage 5 (secondary, non-fatal): downstream NPE + CRPS + coverage. ---
    npe_result = None
    if not args.skip_npe:
        npe_start = time.time()
        npe_result = run_downstream_npe(
            config, data, physical_exprs, correlations["best_component"], seeds
        )
        runtime_seconds["npe"] = time.time() - npe_start
    else:
        log("--skip-npe set; leaving inference fields null")
        runtime_seconds["npe"] = None

    counts["n_downstream_npe_simulations"] = config.nsims if npe_result is not None else 0
    runtime_seconds["total"] = time.time() - total_start

    success = correlations["best_pearson_abs"] >= args.min_r0_corr

    run_record = {
        "run_id": run_id,
        "problem": "sir",
        "master_seed": args.seed,
        "status": "success",
        "counts": counts,
        "discovery": {
            "expressions_physical": [str(e) for e in physical_exprs],
            "expressions_canonical": [str(e) for e in mdl_coords],
            "success": success,
            "physics_alignment": correlations["best_pearson_abs"],
            "physics_alignment_spearman": correlations["best_spearman_abs"],
            "physics_alignment_grad_cosine": correlations["best_grad_cosine"],
            "best_component": correlations["best_component"],
            "mdl_total": mdl_total,
            "complexity_total": complexity_total,
            "symbolically_invertible": invertibility["is_symbolically_invertible"],
            "rank_deficient": bool(rank_info["rank_deficient"]),
        },
        "heldout_geometry": {
            "frob_raw": flatness["raw_theta"],
            "frob_neural": flatness["nn"],
            "frob_symbolic": flatness["pruned"],
            # Surfaced here (not just in prune_info inside sr_expressions.pkl) so a
            # rejected rotation is visible in aggregated results. rel_delta is
            # signed: negative means the rotation improved flatness.
            "rotation_accepted": bool(prune_info["rotation_accepted"]),
            "rotation_rel_delta": float(prune_info["rel_delta"]),
            "median_condition_raw": flatness["median_condition_raw"],
            "median_condition_symbolic": flatness["median_condition_symbolic"],
        },
        "inference": {
            "crps_theta": npe_result["crps_theta"] if npe_result else None,
            "crps_eta": npe_result["crps_eta"] if npe_result else None,
            "coverage_error_theta": npe_result["coverage_error_theta"] if npe_result else None,
            "coverage_error_eta": npe_result["coverage_error_eta"] if npe_result else None,
        },
        "runtime_seconds": runtime_seconds,
    }
    with open(outdir / "run_record.json", "w") as handle:
        json.dump(run_record, handle, indent=2, sort_keys=True)
    log(f"wrote run record to {outdir / 'run_record.json'}")

    if not success:
        raise SystemExit(
            "Recovery criteria not met: "
            f"best |pearson r0 corr|={correlations['best_pearson_abs']:.3f} "
            f"(min {args.min_r0_corr:.3f})"
        )

    log("SIR notebook batch run complete")


if __name__ == "__main__":
    main()
