#!/usr/bin/env python
"""Batch-safe version of tutorial_notebooks/gw_example.ipynb.

The ``full`` mode mirrors the notebook-sized GW TaylorF2/PCA workflow.  It also
saves diagnostic plots for the input data, learned flattened coordinates, and
physics/flatness validation so a Slurm run can be checked without opening the
notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy.optimize import brentq
from sklearn.decomposition import PCA
from tqdm import tqdm

import esr.generation.generator  # noqa: F401 - sanity-check ESR import.
from degeneracy_distillery.align_coords import load_and_process_data_v2
from degeneracy_distillery.postprocess_new import analyze_atom_sharing, regroup_like_terms
from degeneracy_distillery.postprocessing_utils import (
    check_flattening,
    flatten_with_numerical_jacobian,
    print_discovered_expressions,
    weighted_std,
)
from degeneracy_distillery.sr_utils import (
    analyze_equations,
    expressions_to_physical,
    filter_pareto_fronts,
    fit_symbolic_regression,
    fit_theta_scaler,
    sr_structure_predicate,
)
from degeneracy_distillery.training_loop_fishnets import train_fishnets
from degeneracy_distillery.training_loop_flatten import fit_flattening


M_SUN_SEC = 4.925491025543576e-6
MPC_SEC = 1.0292712503e14
M1_MIN, M1_MAX = 5.0, 50.0
M2_MIN, M2_MAX = 5.0, 50.0
D_L_MPC = 200.0
F_LOW, DF = 20.0, 0.5


@dataclass(frozen=True)
class RunConfig:
    nsims: int
    n_pca: int
    pca_basis_max: int
    num_fishnets: int
    fish_epochs: int
    fish_hids_min: int
    fish_hids_max: int
    fish_patience: int
    fish_layers: tuple[int, int]
    fish_batch_size: int
    flatten_hidden_size: int
    flatten_layers: int
    flatten_epochs_phase1: int
    flatten_epochs_phase2: int
    flatten_finetune_epochs: int
    flatten_min_epochs: int
    flatten_patience: int
    align_subsample: int
    sr_grid_size: int
    sr_time_limit: int
    sr_max_length: int
    sr_max_depth: int


CONFIGS = {
    "smoke": RunConfig(
        nsims=250,
        n_pca=20,
        pca_basis_max=500,
        num_fishnets=3,
        fish_epochs=1500,
        fish_hids_min=32,
        fish_hids_max=96,
        fish_patience=25,
        fish_layers=(2, 3),
        fish_batch_size=100,
        flatten_hidden_size=64,
        flatten_layers=2,
        flatten_epochs_phase1=800,
        flatten_epochs_phase2=900,
        flatten_finetune_epochs=150,
        flatten_min_epochs=400,
        flatten_patience=35,
        align_subsample=1000,
        sr_grid_size=1000,
        sr_time_limit=45,
        sr_max_length=25,
        sr_max_depth=15,
    ),
    "full": RunConfig(
        nsims=1000,
        n_pca=40,
        pca_basis_max=5000,
        num_fishnets=10,
        fish_epochs=5000,
        fish_hids_min=50,
        fish_hids_max=300,
        fish_patience=30,
        fish_layers=(3, 5),
        fish_batch_size=200,
        flatten_hidden_size=100,
        flatten_layers=3,
        flatten_epochs_phase1=2000,
        flatten_epochs_phase2=2500,
        flatten_finetune_epochs=500,
        flatten_min_epochs=1200,
        flatten_patience=50,
        align_subsample=4000,
        sr_grid_size=2000,
        sr_time_limit=120,
        sr_max_length=30,
        sr_max_depth=20,
    ),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require_gpu_if_requested(require_gpu: bool) -> None:
    backend = jax.default_backend()
    devices = jax.devices()
    log(f"JAX backend: {backend}")
    log(f"JAX devices: {devices}")
    if require_gpu and backend != "gpu":
        raise SystemExit("JAX did not initialize a GPU backend.")


def chirp_mass(m1, m2):
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def symmetric_mass_ratio(m1, m2):
    return (m1 * m2) / (m1 + m2) ** 2


def a_ligo_psd(f):
    f = np.asarray(f, dtype=float)
    x = f / 215.0
    psd = 1e-49 * (x**-4.14 + 2.0 + 2.0 * x**2)
    return np.where(f >= 10.0, psd, np.inf)


def taylor_f2_waveform(m1, m2, freqs, d_l_mpc=D_L_MPC):
    m_sec = (m1 + m2) * M_SUN_SEC
    eta = m1 * m2 / (m1 + m2) ** 2
    mc_sec = chirp_mass(m1, m2) * M_SUN_SEC
    d_l_sec = d_l_mpc * MPC_SEC

    f_isco = 1.0 / (6**1.5 * np.pi * m_sec)
    active = (freqs > 0) & (freqs < f_isco)
    v = np.zeros_like(freqs)
    v[active] = (np.pi * m_sec * freqs[active]) ** (1.0 / 3)

    amp = np.zeros_like(freqs)
    amp[active] = (
        np.sqrt(5.0 / 24)
        * np.pi ** (-2.0 / 3)
        * mc_sec ** (5.0 / 6)
        / d_l_sec
        * freqs[active] ** (-7.0 / 6)
    )

    c1 = 3715.0 / 756 + 55.0 / 9 * eta
    c15 = -16.0 * np.pi
    c2 = 15379365.0 / 508032 + 27145.0 / 504 * eta + 3085.0 / 72 * eta**2

    phase = np.zeros_like(freqs)
    vm = v[active]
    phase[active] = 3.0 / (128 * eta) * vm**-5 * (1 + c1 * vm**2 + c15 * vm**3 + c2 * vm**4)
    return amp * np.exp(1j * phase)


def simulator_data(config: RunConfig, seed: int, outdir: Path) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    key = jr.PRNGKey(seed)

    f_isco_lightest = 1.0 / (6**1.5 * np.pi * (M1_MIN + M2_MIN) * M_SUN_SEC)
    freqs = np.arange(F_LOW, min(f_isco_lightest, 1024.0), DF)
    whiten = np.sqrt(4.0 * DF) / np.sqrt(a_ligo_psd(freqs))
    n_freq = len(freqs)
    log(f"frequency grid: {F_LOW}-{freqs[-1]:.0f} Hz, {n_freq} bins")

    m1_all = rng.uniform(M1_MIN, M1_MAX, 2 * config.nsims)
    m2_all = rng.uniform(M2_MIN, M2_MAX, 2 * config.nsims)
    theta_all = np.stack([m1_all, m2_all], axis=1).astype(np.float32)

    log("building PCA basis")
    basis_idx = rng.choice(2 * config.nsims, min(2 * config.nsims, config.pca_basis_max), replace=False)
    bank = np.empty((len(basis_idx), 2 * n_freq))
    for j, i in enumerate(tqdm(basis_idx, desc="bank")):
        h = taylor_f2_waveform(m1_all[i], m2_all[i], freqs) * whiten
        bank[j] = np.concatenate([h.real, h.imag])
    pca = PCA(n_components=config.n_pca).fit(bank)
    cumvar = pca.explained_variance_ratio_.cumsum()
    log(f"PCA: {config.n_pca} components capture {cumvar[-1] * 100:.1f}% variance")

    log("generating noisy waveforms")
    _, sub = jr.split(key)
    keys = jr.split(sub, 2 * config.nsims)
    data_all = np.empty((2 * config.nsims, config.n_pca), dtype=np.float32)
    snr_all = np.empty(2 * config.nsims)
    for i in tqdm(range(2 * config.nsims), desc="waveforms"):
        h = taylor_f2_waveform(theta_all[i, 0], theta_all[i, 1], freqs) * whiten
        hvec = np.concatenate([h.real, h.imag])
        snr_all[i] = np.linalg.norm(hvec)
        noise = np.array(jr.normal(keys[i], shape=hvec.shape))
        data_all[i] = pca.transform((hvec + noise).reshape(1, -1)).flatten()

    theta_train, data_train = theta_all[: config.nsims], data_all[: config.nsims]
    theta_test, data_test = theta_all[config.nsims :], data_all[config.nsims :]
    mc_train = np.asarray(chirp_mass(theta_train[:, 0], theta_train[:, 1]))
    log(f"theta_train {theta_train.shape}; data_train {data_train.shape}")

    plot_input_summary(theta_train, data_train, mc_train, snr_all[: config.nsims], cumvar, outdir)
    return {
        "theta_train": theta_train,
        "data_train": data_train,
        "theta_test": theta_test,
        "data_test": data_test,
        "snr_train": snr_all[: config.nsims],
        "cumvar": cumvar,
    }


def plot_input_summary(theta_train, data_train, mc_train, snr_train, cumvar, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    sc = axes[0].scatter(theta_train[:, 0], theta_train[:, 1], c=mc_train, s=4, alpha=0.6, cmap="viridis")
    plt.colorbar(sc, ax=axes[0], label=r"$\mathcal{M}_c$")
    for mc_val in [10, 20, 30, 40]:
        m1_line = np.linspace(M1_MIN, M1_MAX, 200)
        m2_line = []
        for m1 in m1_line:
            try:
                m2 = brentq(lambda mm: chirp_mass(m1, mm) - mc_val, M2_MIN, min(m1, M2_MAX))
            except ValueError:
                m2 = np.nan
            m2_line.append(m2)
        m2_line = np.asarray(m2_line)
        valid = ~np.isnan(m2_line)
        axes[0].plot(m1_line[valid], m2_line[valid], "r--", alpha=0.5, lw=1)
    axes[0].set_xlabel(r"$m_1\;[M_\odot]$")
    axes[0].set_ylabel(r"$m_2\;[M_\odot]$")
    axes[0].set_title(r"Masses colored by $\mathcal{M}_c$")

    axes[1].hist(snr_train, bins=50, color="steelblue", alpha=0.75)
    axes[1].set_xlabel("whitened waveform norm")
    axes[1].set_title("SNR proxy distribution")

    for k in range(min(5, data_train.shape[1])):
        axes[2].hist(np.asarray(data_train[:, k]), bins=50, alpha=0.4, density=True, label=f"PC {k + 1}")
    axes[2].set_xlabel("PCA coefficient")
    axes[2].legend(fontsize=7)
    axes[2].set_title("PCA-compressed data")

    axes[3].plot(np.arange(1, len(cumvar) + 1), cumvar * 100, "o-", ms=3)
    axes[3].axhline(99, ls="--", color="grey", lw=0.8, label="99%")
    axes[3].set_xlabel("Number of PCA components")
    axes[3].set_ylabel("Cumulative variance (%)")
    axes[3].legend(fontsize=9)
    axes[3].set_title("PCA explained variance")

    fig.tight_layout()
    fig.savefig(outdir / "gw_input_summary.png", dpi=180)
    plt.close(fig)


def train_fishnet_ensemble(config: RunConfig, data: dict[str, object], outdir: Path):
    scaler = fit_theta_scaler(data["theta_train"], feature_range=(1.0, 2.0))
    theta_train_s = scaler.transform(data["theta_train"]).astype(np.float32)
    theta_test_s = scaler.transform(data["theta_test"]).astype(np.float32)
    log(f"scaled theta range: {theta_train_s.min(0)} to {theta_train_s.max(0)}")

    embedding_net = nn.Sequential(
        [
            nn.Dense(128),
            nn.gelu,
            nn.Dense(64),
            nn.gelu,
            nn.Dense(64),
            nn.gelu,
        ]
    )

    fish_dir = outdir / "fishnets-gw"
    log(f"training fishnets into {fish_dir}")
    train_fishnets(
        theta_train_s,
        data["data_train"],
        theta_test_s,
        data["data_test"],
        num_models=config.num_fishnets,
        train_epochs=config.fish_epochs,
        hids_min=config.fish_hids_min,
        hids_max=config.fish_hids_max,
        patience=config.fish_patience,
        n_layers=list(config.fish_layers),
        embedding_net=embedding_net,
        lr=5e-5,
        train_batch_size=config.fish_batch_size,
        outdir=str(fish_dir),
        update_pbar_every=25,
    )
    return fish_dir, scaler


def _matrix_log_psd(M, eps=1e-12):
    M_sym = 0.5 * (M + jnp.swapaxes(M, -1, -2))
    evals, evecs = jnp.linalg.eigh(M_sym)
    log_e = jnp.log(jnp.maximum(evals, eps))
    return (evecs * log_e[..., None, :]) @ jnp.swapaxes(evecs, -1, -2)


def _matrix_exp_sym(M):
    M_sym = 0.5 * (M + jnp.swapaxes(M, -1, -2))
    evals, evecs = jnp.linalg.eigh(M_sym)
    return (evecs * jnp.exp(evals)[..., None, :]) @ jnp.swapaxes(evecs, -1, -2)


def fit_flattener(config: RunConfig, fish_dir: Path, outdir: Path):
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

    f_network_ensemble = jnp.array(fs_np)
    log("computing log-Euclidean median Fisher average")
    log_f_ens = jax.vmap(jax.vmap(_matrix_log_psd))(f_network_ensemble)
    f_avg_le = jax.vmap(_matrix_exp_sym)(jnp.median(log_f_ens, axis=0))

    log("fitting flattening model")
    cwd_before = Path.cwd()
    os.chdir(outdir)
    try:
        w, ensemble_w, outputs_flatten, flatten_model = fit_flattening(
            f_network_ensemble,
            thetas,
            F_avg=f_avg_le,
            ensemble_weights=ensemble_weights,
            hidden_size=config.flatten_hidden_size,
            n_layers=config.flatten_layers,
            batch_size=250,
            epochs_phase1=config.flatten_epochs_phase1,
            epochs_phase2=config.flatten_epochs_phase2,
            finetune_epochs=config.flatten_finetune_epochs,
            min_epochs=config.flatten_min_epochs,
            patience=config.flatten_patience,
            lr_phase1=1e-5,
            lr_schedule_initial=1e-3,
            lr_decay=0.3,
            lr_finetune=2e-6,
            norm_factor=None,
            norm_method="median_max_eig",
            noise=1e-8,
            seed=0,
            flattener_activation="softplus",
            Fisher_to_flatten="average",
            output_prefix="gw_flatten",
            use_whitening=True,
            nn_inv=False,
            forward_backward_mlp=True,
            forward_backward_invertibility_weight=1.0,
            minmax_scale_inputs=True,
            grad_clip_norm=1.0,
            loss_type="log_frob",
            beta_det=0.1,
            augment_log_inputs=True,
            l1_alpha=0.0,
            do_plot=False,
            return_model=True,
            save_flatten_model_pickle=False,
            update_pbar_every=25,
        )
    finally:
        os.chdir(cwd_before)

    return w, ensemble_w, outputs_flatten, flatten_model


def align_and_sample_sr_grid(config: RunConfig, outdir: Path, ensemble_w, flatten_model):
    log("aligning coordinates")
    aligned = load_and_process_data_v2(
        datapath=str(outdir) + os.sep,
        filename="gw_flatten.npz",
        num_samps=config.align_subsample,
        seed=44,
        process_ensemble=True,
        n_d=1.0,
        align_mode="procrustes",
        separate_nonlinearity=True,
        canonicalize="permute_and_sign",
        use_prior_normalization=True,
        restore_reference_mean=True,
        Fisher_to_flatten="best",
        verbose=False,
    )

    x_all = aligned["X"]
    mask = x_all[:, 1] > 0.0
    if not mask.any():
        log("positive-X2 alignment mask was empty; using all aligned samples")
        mask = np.ones(len(x_all), dtype=bool)

    x = x_all[mask]
    y = aligned["y"][mask]
    y_std = aligned["y_std"][mask]
    dy_sr = aligned["dy_sr"][mask]
    fs = aligned["Fs"][mask]
    n_params = x.shape[1]
    log(f"aligned X {x.shape}; y {y.shape}")

    key = jr.PRNGKey(7)
    x_sr = jr.uniform(key, minval=x.min(0), maxval=x.max(0), shape=(config.sr_grid_size, n_params))
    ys_sr = jnp.array([jax.vmap(lambda xx: flatten_model.apply(w_i, xx))(x_sr) for w_i in ensemble_w])
    ys_sr_rot = np.array(
        [
            np.einsum("ij,bj->bi", aligned["rotmats"][i], ys_sr[i] - ys_sr[i].mean(0))
            for i in range(len(ys_sr))
        ]
    )
    y_std_sr = weighted_std(ys_sr_rot, aligned["ensemble_weights"])
    y_sr = np.average(ys_sr_rot, 0, aligned["ensemble_weights"])
    ys_sr_rot -= y_sr.min(0)
    y_sr -= y_sr.min(0)

    return {
        "data": aligned,
        "mask": mask,
        "X": x,
        "y": y,
        "y_std": y_std,
        "dy_sr": dy_sr,
        "Fs": fs,
        "n_params": n_params,
        "X_sr": np.asarray(x_sr),
        "y_sr": y_sr,
        "y_std_sr": y_std_sr,
    }


def run_symbolic_regression(config: RunConfig, aligned: dict, outdir: Path):
    sr_dir = outdir / "sr_results_gw"
    sr_dir.mkdir(exist_ok=True)
    log(f"running symbolic regression into {sr_dir}")
    fit_symbolic_regression(
        aligned["X_sr"],
        aligned["y_sr"],
        aligned["y_std_sr"],
        parent_dir=str(sr_dir) + os.sep,
        random_state=123,
        time_limit=config.sr_time_limit,
        max_length=config.sr_max_length,
        max_depth=config.sr_max_depth,
        allowed_symbols="add,mul,div,pow,constant,variable,exp",
        objectives=["r2", "length"],
    )

    equation_predicate = sr_structure_predicate(
        n_params=aligned["n_params"],
        forbid_self_transcendental=True,
    )
    filter_summaries = filter_pareto_fronts(
        str(sr_dir),
        aligned["n_params"],
        equation_predicate,
    )
    removed = sum(int(summary["removed"]) for summary in filter_summaries)
    log(f"removed {removed} self-transcendental/invalid equations from Pareto fronts")

    mdl_coords, frob_coords, analysis = analyze_equations(
        aligned["X"],
        aligned["y"],
        aligned["y_std"],
        aligned["dy_sr"],
        aligned["Fs"],
        parent_dir=str(sr_dir) + os.sep,
        n_params=aligned["n_params"],
        equation_set="pareto",
        max_complexity_thresh=20,
        length_penalty=2.0,
        equation_predicate=equation_predicate,
    )
    return sr_dir, mdl_coords, frob_coords, analysis


def validate_flatness(aligned: dict, mdl_coords, pruned_exprs) -> dict[str, float]:
    adhoc_coords = ["(X1 * X2) ^ (3./5.) / (X1 + X2) ^ (1./5.)", "X1 / X2"]
    adhoc_flats, _ = check_flattening(adhoc_coords, X=aligned["X"], Fs=aligned["Fs"])
    mdl_flats, _ = check_flattening(mdl_coords, X=aligned["X"], Fs=aligned["Fs"])
    pruned_flats, _ = check_flattening(pruned_exprs, X=aligned["X"], Fs=aligned["Fs"])
    nn_flats = jax.vmap(flatten_with_numerical_jacobian)(aligned["dy_sr"], aligned["Fs"])

    def fro_score(q):
        q = np.asarray(q)
        return np.linalg.norm(q - np.eye(aligned["n_params"]), axis=(-2, -1)) + np.linalg.norm(
            np.linalg.inv(q) - np.eye(aligned["n_params"]), axis=(-2, -1)
        )

    return {
        "raw_scaled": float(np.median(fro_score(aligned["Fs"]))),
        "adhoc_mc_q": float(np.median(fro_score(adhoc_flats))),
        "mdl": float(np.median(fro_score(mdl_flats))),
        "pruned": float(np.median(fro_score(pruned_flats))),
        "nn": float(np.median(fro_score(nn_flats))),
    }


def physics_correlations(physical_exprs) -> dict[str, object]:
    m1, m2 = sympy.symbols("m1 m2")
    rng = np.random.default_rng(123)
    samples = rng.uniform([M1_MIN, M2_MIN], [M1_MAX, M2_MAX], size=(5000, 2))
    targets = {
        "log_chirp_mass": np.log(chirp_mass(samples[:, 0], samples[:, 1])),
        "log_symmetric_mass_ratio": np.log(symmetric_mass_ratio(samples[:, 0], samples[:, 1])),
        "log_total_mass": np.log(samples[:, 0] + samples[:, 1]),
        "q_m2_over_m1": samples[:, 1] / samples[:, 0],
        "product_m1_m2": samples[:, 0] * samples[:, 1],
    }

    rows = []
    for i, expr in enumerate(physical_exprs):
        fn = sympy.lambdify((m1, m2), expr, modules="numpy")
        values = np.asarray(fn(samples[:, 0], samples[:, 1]), dtype=float)
        values = np.broadcast_to(values, (samples.shape[0],))
        row = {"component": i, "expr": str(expr)}
        for name, target in targets.items():
            if np.std(values) == 0 or not np.isfinite(values).all():
                corr = 0.0
            else:
                corr = float(np.corrcoef(values, target)[0, 1])
            row[name] = corr
        rows.append(row)

    best = max(max(abs(row[name]) for name in targets) for row in rows)
    return {"rows": rows, "best_abs_corr": best}


def plot_coordinate_maps(aligned: dict, outdir: Path) -> None:
    x = aligned["X"]
    y = aligned["y"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sc0 = axes[0].scatter(x[:, 0], x[:, 1], c=y[:, 0], s=6, alpha=0.75, cmap="viridis")
    plt.colorbar(sc0, ax=axes[0], label=r"$\eta_1$")
    axes[0].set_xlabel("scaled/aligned X1")
    axes[0].set_ylabel("scaled/aligned X2")
    axes[0].set_title("Learned coordinate 1")

    sc1 = axes[1].scatter(x[:, 0], x[:, 1], c=y[:, 1], s=6, alpha=0.75, cmap="magma")
    plt.colorbar(sc1, ax=axes[1], label=r"$\eta_2$")
    axes[1].set_xlabel("scaled/aligned X1")
    axes[1].set_ylabel("scaled/aligned X2")
    axes[1].set_title("Learned coordinate 2")

    axes[2].scatter(y[:, 0], y[:, 1], c=x[:, 0] + x[:, 1], s=6, alpha=0.75, cmap="cividis")
    axes[2].set_xlabel(r"$\eta_1$")
    axes[2].set_ylabel(r"$\eta_2$")
    axes[2].set_title("Flattened coordinate plane")
    fig.tight_layout()
    fig.savefig(outdir / "gw_flattened_coordinates.png", dpi=180)
    plt.close(fig)


def plot_physics_validation(aligned: dict, physical_exprs, flatness: dict[str, float], outdir: Path) -> None:
    m1, m2 = sympy.symbols("m1 m2")
    x = aligned["X"]
    targets = {
        r"$\ln\mathcal{M}_c$": np.log(chirp_mass(x[:, 0], x[:, 1])),
        r"$\ln\eta_{\rm sym}$": np.log(symmetric_mass_ratio(x[:, 0], x[:, 1])),
        r"$\ln M_{\rm tot}$": np.log(x[:, 0] + x[:, 1]),
        r"$q$": x[:, 1] / x[:, 0],
    }
    expr_values = []
    for expr in physical_exprs:
        fn = sympy.lambdify((m1, m2), expr, modules="numpy")
        expr_values.append(np.asarray(fn(x[:, 0], x[:, 1]), dtype=float))

    fig, axes = plt.subplots(len(expr_values), len(targets), figsize=(3.5 * len(targets), 3.0 * len(expr_values)))
    axes = np.asarray(axes).reshape(len(expr_values), len(targets))
    for i, values in enumerate(expr_values):
        for j, (label, target) in enumerate(targets.items()):
            ax = axes[i, j]
            ax.scatter(target, values, s=5, alpha=0.45)
            corr = np.corrcoef(values, target)[0, 1] if np.std(values) > 0 else 0.0
            ax.set_title(f"eta {i} vs {label}\nr={corr:.2f}")
            ax.set_xlabel(label)
            ax.set_ylabel(f"eta {i}")
    fig.tight_layout()
    fig.savefig(outdir / "gw_physics_correlations.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(flatness)
    vals = [flatness[name] for name in names]
    ax.bar(names, vals, color="steelblue")
    ax.set_yscale("log")
    ax.set_ylabel(r"median $\|Q-I\|_F + \|Q^{-1}-I\|_F$")
    ax.set_title("Flatness validation")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(outdir / "gw_flatness_scores.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIGS), default="full")
    parser.add_argument("--out-dir", type=Path, default=Path("results/gw_notebook_full"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--min-physics-corr",
        type=float,
        default=0.75,
        help="Fail if no discovered physical expression correlates with a standard GW mass coordinate this strongly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.mode]
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"running mode={args.mode}; outdir={outdir}")
    log(f"config={json.dumps(asdict(config), sort_keys=True)}")

    require_gpu_if_requested(args.require_gpu)

    data = simulator_data(config, args.seed, outdir)
    fish_dir, scaler = train_fishnet_ensemble(config, data, outdir)
    _, ensemble_w, _, flatten_model = fit_flattener(config, fish_dir, outdir)
    aligned = align_and_sample_sr_grid(config, outdir, ensemble_w, flatten_model)
    plot_coordinate_maps(aligned, outdir)

    sr_dir, mdl_coords, frob_coords, analysis = run_symbolic_regression(config, aligned, outdir)
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
        do_inner_snap=True,
        decimal=1,
        threshold=0.1,
    )
    print_discovered_expressions(pruned_exprs, name_map={"X1": "m1", "X2": "m2"})

    physical_exprs = expressions_to_physical(
        pruned_exprs,
        scaler,
        sr_offset=0.0,
        theta_names=("m1", "m2"),
        decimal=3,
    )
    log("physical expressions")
    for k, expr in enumerate(physical_exprs):
        print(f"  eta_{k} = {expr}", flush=True)

    correlations = physics_correlations(physical_exprs)
    log("physical expression correlations")
    print(json.dumps(correlations, indent=2, sort_keys=True), flush=True)

    flatness = validate_flatness(aligned, mdl_coords, pruned_exprs)
    log("flatness scores")
    print(json.dumps(flatness, indent=2, sort_keys=True), flush=True)
    plot_physics_validation(aligned, physical_exprs, flatness, outdir)

    with open(sr_dir / "sr_expressions.pkl", "wb") as handle:
        pickle.dump(
            {
                "mdl_coords": mdl_coords,
                "frob_coords": frob_coords,
                "pruned_exprs": pruned_exprs,
                "physical_exprs": [str(e) for e in physical_exprs],
                "correlations": correlations,
                "flatness": flatness,
                "analysis": analysis,
                "rotation": rotation,
                "prune_info": prune_info,
                "scaler_scale": scaler.scale_,
                "scaler_min": scaler.min_,
                "scaler_data_min": scaler.data_min_,
                "scaler_data_max": scaler.data_max_,
            },
            handle,
        )
    shutil.copytree(fish_dir, sr_dir / "fishnets-gw", dirs_exist_ok=True)
    shutil.copy2(outdir / "gw_flatten.npz", sr_dir / "gw_flatten.npz")
    shutil.make_archive(str(outdir / "sr_results_gw"), "zip", root_dir=sr_dir)
    log(f"saved artifacts under {sr_dir}")

    if correlations["best_abs_corr"] < args.min_physics_corr:
        raise SystemExit(
            "No physical expression correlated strongly enough with standard GW mass coordinates: "
            f"{correlations['best_abs_corr']:.3f} < {args.min_physics_corr:.3f}"
        )

    log("GW notebook batch run complete")


if __name__ == "__main__":
    main()
