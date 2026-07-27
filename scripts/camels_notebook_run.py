#!/usr/bin/env python3
"""CAMELS-SB35 degeneracy distillery — full pipeline.

Converts scratch_notebooks/example_camels.ipynb to a batch-safe script.
CAMELS-SB35: 35-parameter cosmological/astrophysical inference from 14 galaxy
scaling-relation summary statistics (344 features total).

Pipeline
--------
  1. Load HDF5: theta (1024×35) + observables (1024×344).
  2. Train JAX fishnet ensemble on the full 35-parameter space.
  3. Analyse Fisher matrix structure and auto-select top-K parameters:
       (a) informativeness: diag(mean_F)
       (b) degeneracy score: Σⱼ Var[Fᵢⱼ] per parameter
       (c) eigenspectrum of mean_F
       (d) hierarchical clustering on Fisher correlation matrix
  4. Extract K×K block submatrix of the Fishers (or optionally retrain on K-subset).
  5. Fit flattening model on K-dimensional subspace.
  6. Sample SR grid in aligned coordinate space; run symbolic regression.

Modes
-----
  smoke : reduced hyperparameters for quick validation
  full  : notebook-scale hyperparameters for a full GPU run

Flags
-----
  --param-indices '0,1,2,3,4,5'  override Fisher auto-selection
  --retrain-subset                retrain fishnets on the K-param subset
  --skip-fishnets --from-dir DIR  load saved fishnets; skip training
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

import h5py
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import esr.generation.generator  # noqa: F401 — sanity-check ESR import
from degeneracy_distillery.align_coords import load_and_process_data_v2
from degeneracy_distillery.postprocess_new import analyze_atom_sharing, regroup_like_terms
from degeneracy_distillery.postprocessing_utils import (
    check_flattening,
    flatten_with_numerical_jacobian,
    print_discovered_expressions,
    weighted_std,
)
from degeneracy_distillery.preprocessing_utils import get_eigenvalues
from degeneracy_distillery.sr_utils import (
    analyze_equations,
    filter_pareto_fronts,
    fit_symbolic_regression,
    sr_structure_predicate,
)
from degeneracy_distillery.training_loop_fishnets import train_fishnets
from degeneracy_distillery.training_loop_flatten import fit_flattening


# ---------------------------------------------------------------------------
# Parameter metadata  (order from data_scratch/SB35_param_minmax.csv)
# ---------------------------------------------------------------------------
PARAM_NAMES = [
    "Omega_m",                    # 0
    "sigma_8",                    # 1
    "A_SN1",                      # 2  WindEnergyIn1e51erg
    "A_AGN1",                     # 3  RadioFeedbackFactor   ← AGN, NOT SN2
    "A_SN2",                      # 4  VariableWindVelFactor
    "A_AGN2",                     # 5  RadioFeedbackReiorientationFactor
    "Omega_b",                    # 6
    "H_0",                        # 7
    "n_s",                        # 8
    "MaxSfrTimescale",            # 9
    "FactorForSofterEQS",         # 10
    "IMFslope",                   # 11
    "SNII_MinMass",               # 12
    "ThermalWindFraction",        # 13
    "VariableWindSpecMomentum",   # 14
    "WindFreeTravelDensFac",      # 15
    "MinWindVel",                 # 16
    "WindEnergyReductionFactor",  # 17
    "WindEnergyReductionMetallicity", # 18
    "WindEnergyReductionExponent",    # 19
    "WindDumpFactor",             # 20
    "SeedBlackHoleMass",          # 21
    "BH_AccretionFactor",         # 22
    "BH_EddingtonFactor",         # 23
    "BH_FeedbackFactor",          # 24
    "BH_RadiativeEfficiency",     # 25
    "QuasarThreshold",            # 26
    "QuasarThresholdPower",       # 27
    "UVB_H0_beta",                # 28
    "UVB_H0_Deltaz",              # 29
    "UVB_Hep_beta",               # 30
    "UVB_Hep_Deltaz",             # 31
    "SNIa_Rate_Norm",             # 32
    "SNIa_Rate_DTD_power",        # 33
    "SofteningLength",            # 34
]
N_PARAMS_FULL = 35

CAMELS_PHYSICS_PRIOR = [0, 1, 2, 3, 4, 5]  # standard 6-param block (for sanity checks)

OBSERVABLES = [
    "MBH_Mh_s61", "MBH_Mh_s90",
    "Mg_Mh_s61",  "Mg_Mh_s90",
    "Ms_Mh_s61",  "Ms_Mh_s90",
    "Rs_Ms_s61",  "Rs_Ms_s90",
    "SFRH",       "SFRH_100Myr",
    "SFR_Ms_s61", "SFR_Ms_s90",
    "Zs_Ms_s61",  "Zs_Ms_s90",
]
N_SIMS = 1024


# ---------------------------------------------------------------------------
# Run configurations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    # Fishnets
    num_fishnets: int
    fish_hids_min: int
    fish_hids_max: int
    fish_layers: tuple[int, int]
    fish_epochs: int
    fish_min_epochs: int
    fish_patience: int
    fish_batch_size: int
    # Fisher analysis
    n_clusters: int
    top_k_params: int
    param_k: int        # number of parameters for flattening + SR
    # Flattening
    flatten_layers: int
    flatten_epochs_phase1: int
    flatten_epochs_phase2: int
    flatten_finetune_epochs: int
    flatten_min_epochs: int
    flatten_patience: int
    align_subsample: int
    # SR
    sr_grid_size: int
    sr_time_limit: int
    sr_max_length: int
    sr_max_depth: int


CONFIGS = {
    "smoke": RunConfig(
        num_fishnets=3,
        fish_hids_min=50,
        fish_hids_max=150,
        fish_layers=(2, 3),
        fish_epochs=500,
        fish_min_epochs=50,
        fish_patience=15,
        fish_batch_size=64,
        n_clusters=4,
        top_k_params=8,
        param_k=4,
        flatten_layers=5,
        flatten_epochs_phase1=2000,
        flatten_epochs_phase2=1000,
        flatten_finetune_epochs=200,
        flatten_min_epochs=400,
        flatten_patience=30,
        align_subsample=500,
        sr_grid_size=1000,
        sr_time_limit=60,
        sr_max_length=20,
        sr_max_depth=8,
    ),
    "full": RunConfig(
        num_fishnets=20,
        fish_hids_min=100,
        fish_hids_max=300,
        fish_layers=(2, 5),
        fish_epochs=5000,
        fish_min_epochs=200,
        fish_patience=20,
        fish_batch_size=200,
        n_clusters=5,
        top_k_params=10,
        param_k=6,
        flatten_layers=8,
        flatten_epochs_phase1=10000,
        flatten_epochs_phase2=5000,
        flatten_finetune_epochs=1000,
        flatten_min_epochs=1200,
        flatten_patience=50,
        align_subsample=3000,
        sr_grid_size=5000,
        sr_time_limit=600,
        sr_max_length=30,
        sr_max_depth=10,
    ),
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require_gpu_if_requested(require_gpu: bool) -> None:
    backend = jax.default_backend()
    log(f"JAX backend: {backend},  devices: {jax.devices()}")
    if require_gpu and backend != "gpu":
        raise SystemExit("JAX did not initialise a GPU backend.")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_camels_data(data_path: Path, config: RunConfig, seed: int = 42):
    log(f"loading CAMELS HDF5 from {data_path}")
    with h5py.File(data_path, "r") as f:
        P = np.asarray(f["Parameters"][...], dtype=np.float32)
        obs_blocks = []
        for name in OBSERVABLES:
            arr = np.asarray(f[name][...], dtype=np.float32)
            # file convention: (n_chan, N_sims) → (N_sims, n_chan)
            block = arr[:, :N_SIMS].T if arr.shape[0] != N_SIMS else arr[:N_SIMS]
            obs_blocks.append(block)

    theta_raw = (P[:, :N_SIMS].T if P.shape[0] == N_PARAMS_FULL else P[:N_SIMS])
    data_raw = np.concatenate(obs_blocks, axis=1)
    data_raw = np.nan_to_num(data_raw, nan=0.0, posinf=0.0, neginf=0.0)
    log(f"theta {theta_raw.shape}  data {data_raw.shape}")

    scaler = MinMaxScaler()
    theta_scaled = scaler.fit_transform(theta_raw).astype(np.float32)

    theta_train, theta_test, data_train, data_test = train_test_split(
        theta_scaled, data_raw, test_size=0.2, random_state=seed
    )
    log(f"train: {theta_train.shape}  test: {theta_test.shape}")

    meta = {
        "theta_min": scaler.data_min_.astype(np.float32),
        "theta_max": scaler.data_max_.astype(np.float32),
        "param_names": np.array(PARAM_NAMES),
    }
    return theta_train, data_train, theta_test, data_test, scaler, meta


# ---------------------------------------------------------------------------
# Fishnet training (full 35-param, JAX tabular)
# ---------------------------------------------------------------------------
def train_fishnet_ensemble(
    config: RunConfig,
    theta_train: np.ndarray,
    data_train: np.ndarray,
    theta_test: np.ndarray,
    data_test: np.ndarray,
    outdir: Path,
) -> Path:
    fish_dir = outdir / "fishnets-camels"
    fish_dir.mkdir(parents=True, exist_ok=True)
    log(f"training {config.num_fishnets} JAX fishnets (full {N_PARAMS_FULL}D) → {fish_dir}")
    train_fishnets(
        jnp.array(theta_train),
        jnp.array(data_train),
        jnp.array(theta_test),
        jnp.array(data_test),
        num_models=config.num_fishnets,
        train_epochs=config.fish_epochs,
        train_min_epochs=config.fish_min_epochs,
        patience=config.fish_patience,
        hids_min=config.fish_hids_min,
        hids_max=config.fish_hids_max,
        n_layers=list(config.fish_layers),
        lr=5e-5,
        train_batch_size=config.fish_batch_size,
        outdir=str(fish_dir),
        update_pbar_every=25,
    )
    return fish_dir


# ---------------------------------------------------------------------------
# Fisher structure analysis + parameter selection
# ---------------------------------------------------------------------------
def analyze_fisher_structure(config: RunConfig, fish_dir: Path, outdir: Path) -> dict:
    log("analysing Fisher matrix structure")
    with np.load(fish_dir / "fishnets_outputs.npz") as npz:
        F_ensemble = np.asarray(npz["Fs"])  # (n_models, n_test, 35, 35)

    finite_mask = np.isfinite(F_ensemble).all(axis=(1, 2, 3))
    F_ensemble = F_ensemble[finite_mask]
    log(f"F_ensemble shape: {F_ensemble.shape}")

    mean_F = F_ensemble.mean(axis=(0, 1))   # (35, 35)
    var_F  = F_ensemble.var(axis=(0, 1))    # (35, 35)

    # (a) Informativeness: diagonal of mean_F
    info_scores = np.diag(mean_F)
    info_rank   = np.argsort(info_scores)[::-1]

    # (b) Degeneracy score: summed off-diagonal variance per parameter
    off_diag_var = var_F.copy()
    np.fill_diagonal(off_diag_var, 0.0)
    deg_scores = off_diag_var.sum(axis=1)
    deg_rank   = np.argsort(deg_scores)[::-1]

    # (c) Eigendecomposition of mean_F
    eigvals, eigvecs = np.linalg.eigh(mean_F)  # ascending order
    eigvals_desc = eigvals[::-1]
    eigvecs_desc = eigvecs[:, ::-1]

    # (d) Hierarchical clustering on Fisher correlation matrix
    d_diag = np.maximum(np.abs(np.diag(mean_F)), 1e-10) ** 0.5
    corr_F = mean_F / np.outer(d_diag, d_diag)
    corr_F = np.clip(corr_F, -1.0, 1.0)
    dist_full = np.clip(1.0 - np.abs(corr_F), 0.0, None)
    np.fill_diagonal(dist_full, 0.0)   # squareform requires exact zeros on diagonal
    dist_mat = squareform(dist_full)
    Z = linkage(dist_mat, method="ward")
    cluster_labels = fcluster(Z, t=config.n_clusters, criterion="maxclust")

    top_k = config.top_k_params
    top_info_idx = info_rank[:top_k]
    top_deg_idx  = deg_rank[:top_k]
    union_idx    = np.array(sorted(set(top_info_idx.tolist()) | set(top_deg_idx.tolist())))

    log(f"top-{top_k} informative : {[PARAM_NAMES[i] for i in top_info_idx]}")
    log(f"top-{top_k} degenerate  : {[PARAM_NAMES[i] for i in top_deg_idx]}")
    log(f"union ({len(union_idx)} params): {[PARAM_NAMES[i] for i in union_idx]}")

    _plot_heatmaps(mean_F, var_F, outdir)
    _plot_eigenspectrum(eigvals_desc, outdir)
    _plot_parameter_scores(info_scores, deg_scores, top_k, outdir)
    _plot_clustering(var_F, Z, cluster_labels, outdir)

    results = {
        "mean_F":         mean_F,
        "var_F":          var_F,
        "info_scores":    info_scores,
        "deg_scores":     deg_scores,
        "eigvals":        eigvals_desc,
        "eigvecs":        eigvecs_desc,
        "cluster_labels": cluster_labels,
        "top_info_idx":   top_info_idx,
        "top_deg_idx":    top_deg_idx,
        "union_idx":      union_idx,
    }
    np.savez(outdir / "fisher_analysis.npz", **results)
    log(f"saved fisher_analysis.npz + 4 plots to {outdir}")
    return results


def select_parameters(
    fish_dir: Path,
    config: RunConfig,
    fisher_results: dict | None = None,
    param_indices_override=None,
) -> np.ndarray:
    """Return sorted array of K parameter indices for flattening + SR."""
    if param_indices_override is not None:
        selected = np.sort(np.asarray(param_indices_override, dtype=int))
        log(f"parameter selection overridden: {[PARAM_NAMES[i] for i in selected]}")
        return selected

    if fisher_results is None:
        with np.load(fish_dir / "fishnets_outputs.npz") as npz:
            F_ensemble = np.asarray(npz["Fs"])
        finite_mask = np.isfinite(F_ensemble).all(axis=(1, 2, 3))
        F_ensemble = F_ensemble[finite_mask]
        mean_F = F_ensemble.mean((0, 1))
        var_F  = F_ensemble.var((0, 1))
        info_scores = np.diag(mean_F)
        off_diag_var = var_F.copy()
        np.fill_diagonal(off_diag_var, 0.0)
        deg_scores = off_diag_var.sum(axis=1)
    else:
        info_scores = fisher_results["info_scores"]
        deg_scores  = fisher_results["deg_scores"]

    diag_norm = info_scores / (info_scores.max() + 1e-30)
    deg_norm  = deg_scores  / (deg_scores.max()  + 1e-30)
    score     = diag_norm + deg_norm
    selected  = np.sort(np.argsort(score)[-config.param_k:])

    # Diagnostic table
    order = np.argsort(score)[::-1]
    header = (
        f"{'Rank':>4}  {'Idx':>3}  {'Name':<35}  "
        f"{'info':>10}  {'deg':>10}  {'score':>8}"
    )
    log(f"Parameter selection diagnostic (top 20 of {N_PARAMS_FULL}):")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for rank, i in enumerate(order[:20], 1):
        marker = "  <-- selected" if i in selected else ""
        print(
            f"{rank:4d}  {i:3d}  {PARAM_NAMES[i]:<35}  "
            f"{info_scores[i]:10.4f}  {deg_scores[i]:10.4f}  {score[i]:8.4f}{marker}",
            flush=True,
        )
    print(f"\nSelected {config.param_k}: indices={selected.tolist()}", flush=True)
    print(f"Names: {[PARAM_NAMES[i] for i in selected]}", flush=True)

    for idx, name in [(0, "Omega_m"), (1, "sigma_8")]:
        if idx not in selected:
            log(f"WARNING: {name} (index {idx}) not selected — check Fisher diagnostics")

    return selected


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------
def fit_flattener(
    config: RunConfig,
    fish_dir: Path,
    param_indices: np.ndarray,
    outdir: Path,
    retrain_subset: bool = False,
    train_data=None,
):
    """Fit the flattening model on the K-dimensional parameter subspace.

    retrain_subset=False (default): extract K×K block from existing 35D Fishers.
    retrain_subset=True: re-train fishnets on just the K selected params.
    train_data: (theta_train, data_train, theta_test, data_test) — required for retrain.
    """
    if retrain_subset:
        if train_data is None:
            raise ValueError("train_data is required for --retrain-subset")
        theta_tr, data_tr, theta_te, data_te = train_data
        sub_fish_dir = outdir / "fishnets-camels-sub"
        sub_fish_dir.mkdir(exist_ok=True)
        log(f"retraining fishnets on {len(param_indices)}-param subset → {sub_fish_dir}")
        train_fishnets(
            jnp.array(theta_tr[:, param_indices]),
            jnp.array(data_tr),
            jnp.array(theta_te[:, param_indices]),
            jnp.array(data_te),
            num_models=config.num_fishnets,
            train_epochs=config.fish_epochs,
            train_min_epochs=config.fish_min_epochs,
            patience=config.fish_patience,
            hids_min=config.fish_hids_min,
            hids_max=config.fish_hids_max,
            n_layers=list(config.fish_layers),
            lr=5e-5,
            train_batch_size=config.fish_batch_size,
            outdir=str(sub_fish_dir),
            update_pbar_every=25,
        )
        with np.load(sub_fish_dir / "fishnets_outputs.npz") as npz:
            thetas = jnp.array(npz["theta"])
            ensemble_weights = np.asarray(npz["ensemble_weights"])
            fs_np = np.asarray(npz["Fs"])
    else:
        log(f"extracting {len(param_indices)}-param block submatrix from {N_PARAMS_FULL}D Fishers")
        with np.load(fish_dir / "fishnets_outputs.npz") as npz:
            thetas_full = jnp.array(npz["theta"])
            ensemble_weights = np.asarray(npz["ensemble_weights"])
            fs_np = np.asarray(npz["Fs"])
        thetas = thetas_full[:, param_indices]
        fs_np  = fs_np[:, :, param_indices, :][:, :, :, param_indices]

    finite_mask = np.isfinite(fs_np).all(axis=(1, 2, 3))
    if not finite_mask.any():
        raise RuntimeError("all fishnet ensemble members produced non-finite Fishers")
    if finite_mask.sum() < len(finite_mask):
        n_bad = len(finite_mask) - int(finite_mask.sum())
        log(f"filtering {n_bad} non-finite ensemble members")
        fs_np = fs_np[finite_mask]
        ensemble_weights = ensemble_weights[finite_mask]

    log(f"fitting flattening model on {len(param_indices)}D subspace")
    cwd_before = Path.cwd()
    os.chdir(outdir)
    try:
        w, ensemble_w, outputs_flatten, flatten_model = fit_flattening(
            jnp.array(fs_np),
            thetas,
            ensemble_weights=ensemble_weights,
            hidden_size=256,
            n_layers=config.flatten_layers,
            batch_size=50,
            epochs_phase1=config.flatten_epochs_phase1,
            epochs_phase2=config.flatten_epochs_phase2,
            finetune_epochs=config.flatten_finetune_epochs,
            min_epochs=config.flatten_min_epochs,
            patience=config.flatten_patience,
            lr_phase1=1e-4,
            lr_schedule_initial=1e-4,
            lr_decay=0.3,
            lr_finetune=4e-6,
            Fisher_to_flatten="average",
            norm_factor=None,
            norm_method="median_max_eig",
            flattener_activation="softplus",
            loss_type="log_frob",
            use_whitening=True,
            nn_inv=False,
            forward_backward_mlp=True,
            minmax_scale_inputs=True,
            augment_log_inputs=True,
            grad_clip_norm=1.0,
            noise=1e-3,
            l1_alpha=0.0,
            seed=0,
            output_prefix="camels_flattening",
            do_plot=False,
            return_model=True,
            save_flatten_model_pickle=True,
            update_pbar_every=25,
        )
    finally:
        os.chdir(cwd_before)

    return w, ensemble_w, outputs_flatten, flatten_model


# ---------------------------------------------------------------------------
# Coordinate alignment + SR grid
# ---------------------------------------------------------------------------
def align_and_sample_sr_grid(
    config: RunConfig, outdir: Path, ensemble_w, flatten_model
) -> dict:
    log("aligning coordinates")
    aligned = load_and_process_data_v2(
        datapath=str(outdir) + os.sep,
        filename="camels_flattening.npz",
        num_samps=config.align_subsample,
        seed=44,
        process_ensemble=True,
        n_d=1.0,
        align_mode="kabsch",
        separate_nonlinearity=False,
        canonicalize="sign_only",
        use_prior_normalization=True,
        restore_reference_mean=False,
        Fisher_to_flatten="average",
        verbose=False,
    )

    x = aligned["X"]
    mask = x[:, 0] > 0.0
    x    = x[mask]
    y    = aligned["y"][mask]
    ys   = np.array([yy[mask] for yy in aligned["ys"]])
    y_std  = aligned["y_std"][mask]
    dy_sr  = aligned["dy_sr"][mask]
    Fs     = aligned["Fs"][mask]
    n_params = x.shape[1]

    ymin_ = y.min(0) - 1.0
    y  -= ymin_
    ys -= ymin_
    log(f"aligned X {x.shape}; y {y.shape} (min={y.min(0).round(2)})")

    # Sample uniformly in aligned coordinate space (not physical theta)
    key = jr.PRNGKey(456)
    x_sr = jr.uniform(
        key,
        minval=jnp.array(x.min(0)),
        maxval=jnp.array(x.max(0)),
        shape=(config.sr_grid_size, n_params),
    )
    ys_sr = jnp.array(
        [jax.vmap(lambda xx: flatten_model.apply(w_i, xx))(x_sr) for w_i in ensemble_w]
    )
    ys_sr_rot = np.array(
        [
            np.einsum("ij,bj->bi", aligned["rotmats"][i], ys_sr[i] - ys_sr[i].mean(0))
            for i in range(len(ys_sr))
        ]
    )
    y_std_sr = weighted_std(ys_sr_rot, aligned["ensemble_weights"])
    y_sr = np.average(ys_sr_rot, 0, aligned["ensemble_weights"])
    ys_sr_rot -= y_sr.min(0)
    y_sr      -= y_sr.min(0)
    y_sr      += 1
    ys_sr_rot += 1

    return {
        "data":     aligned,
        "X":        x,
        "y":        y,
        "ys":       ys,
        "y_std":    y_std,
        "dy_sr":    dy_sr,
        "Fs":       Fs,
        "n_params": n_params,
        "X_sr":     np.asarray(x_sr),
        "y_sr":     y_sr,
        "y_std_sr": y_std_sr,
    }


# ---------------------------------------------------------------------------
# Symbolic regression
# ---------------------------------------------------------------------------
def run_symbolic_regression(
    config: RunConfig, aligned: dict, outdir: Path
) -> tuple[Path, list, list, dict]:
    sr_dir = outdir / "sr_results_camels"
    sr_dir.mkdir(exist_ok=True)
    log(f"running symbolic regression into {sr_dir}")

    fit_symbolic_regression(
        aligned["X_sr"],
        aligned["y_sr"],
        aligned["y_std_sr"],
        parent_dir=str(sr_dir) + os.sep,
        random_state=32134,
        time_limit=config.sr_time_limit,
        max_length=config.sr_max_length,
        max_depth=config.sr_max_depth,
        allowed_symbols="add,mul,div,pow,constant,variable,sqrt,logabs",
        verbose=True,
    )

    equation_predicate = sr_structure_predicate(
        n_params=aligned["n_params"],
        forbid_self_transcendental=False,
        check_nested_exp=False,
    )
    filter_summaries = filter_pareto_fronts(
        str(sr_dir), aligned["n_params"], equation_predicate
    )
    removed = sum(int(s["removed"]) for s in filter_summaries)
    log(f"removed {removed} invalid equations from Pareto fronts")

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
        verbose=True,
    )
    return sr_dir, mdl_coords, frob_coords, analysis


# ---------------------------------------------------------------------------
# Flatness validation
# ---------------------------------------------------------------------------
def validate_flatness(aligned: dict, mdl_coords: list, pruned_exprs: list) -> dict:
    nn_flats     = jax.vmap(flatten_with_numerical_jacobian)(aligned["dy_sr"], aligned["Fs"])
    mdl_flats, _ = check_flattening(mdl_coords, X=aligned["X"], Fs=aligned["Fs"])
    pru_flats, _ = check_flattening(pruned_exprs, X=aligned["X"], Fs=aligned["Fs"])

    def fro_score(q):
        return np.linalg.norm(np.asarray(q) - np.eye(aligned["n_params"]), axis=(-2, -1))

    scores = {
        "raw_theta": float(np.median(fro_score(aligned["Fs"]))),
        "mdl":       float(np.median(fro_score(mdl_flats))),
        "pruned":    float(np.median(fro_score(pru_flats))),
        "nn":        float(np.median(fro_score(nn_flats))),
    }
    evalues_nn = np.asarray(jax.vmap(get_eigenvalues)(nn_flats)).ravel()
    scores["nn_median_abs_log_eigenvalue"] = float(
        np.median(np.abs(np.log(np.clip(evalues_nn, 1e-12, None))))
    )
    return scores


# ---------------------------------------------------------------------------
# Plotting helpers  (Fisher analysis)
# ---------------------------------------------------------------------------
def _plot_heatmaps(mean_F: np.ndarray, var_F: np.ndarray, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    n = mean_F.shape[0]
    ticks = list(range(0, n, 5))
    tick_labels = [PARAM_NAMES[i] for i in ticks]

    vmax = np.abs(mean_F).max()
    im0 = axes[0].imshow(mean_F, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_title("Mean Fisher  ⟨F⟩", fontsize=13)
    for ax_ in [axes[0]]:
        ax_.set_xticks(ticks); ax_.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax_.set_yticks(ticks); ax_.set_yticklabels(tick_labels, fontsize=7)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    vv = np.log1p(var_F)
    im1 = axes[1].imshow(vv, cmap="viridis", aspect="auto")
    axes[1].set_title("log(1 + Var[F])", fontsize=13)
    axes[1].set_xticks(ticks); axes[1].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_yticks(ticks); axes[1].set_yticklabels(tick_labels, fontsize=7)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    fig.savefig(outdir / "fisher_heatmaps.png", dpi=150)
    plt.close(fig)
    log("saved fisher_heatmaps.png")


def _plot_eigenspectrum(eigvals_desc: np.ndarray, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    idx = np.arange(len(eigvals_desc))
    abs_ev = np.abs(eigvals_desc)

    axes[0].semilogy(idx, abs_ev, "o-", ms=4)
    axes[0].set_xlabel("eigenvalue index"); axes[0].set_ylabel("|λ| (log)")
    axes[0].set_title("Fisher eigenspectrum")

    cum = np.cumsum(abs_ev) / abs_ev.sum() * 100
    axes[1].plot(idx, cum)
    for level, ls in [(90, "--"), (99, ":")]:
        axes[1].axhline(level, ls=ls, c="gray", label=f"{level}%")
    axes[1].set_xlabel("eigenvalue index"); axes[1].set_ylabel("cumulative variance (%)")
    axes[1].set_title("Cumulative spectrum"); axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "fisher_eigenspectrum.png", dpi=150)
    plt.close(fig)
    log("saved fisher_eigenspectrum.png")


def _plot_parameter_scores(
    info_scores: np.ndarray, deg_scores: np.ndarray, top_k: int, outdir: Path
) -> None:
    n = len(info_scores)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(n)
    for ax, scores, title, color in [
        (axes[0], info_scores, "Informativeness  diag(⟨F⟩)",    "steelblue"),
        (axes[1], deg_scores,  "Degeneracy  Σⱼ Var[Fᵢⱼ]",       "darkorange"),
    ]:
        rank = np.argsort(scores)[::-1]
        bar_colors = [color if i in rank[:top_k] else "lightgrey" for i in range(n)]
        ax.bar(x, scores, color=bar_colors)
        ax.set_xticks(x); ax.set_xticklabels(PARAM_NAMES, rotation=90, fontsize=6)
        ax.set_title(f"{title}  (top-{top_k} highlighted)", fontsize=10)
        ax.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(outdir / "fisher_parameter_scores.png", dpi=150)
    plt.close(fig)
    log("saved fisher_parameter_scores.png")


def _plot_clustering(
    var_F: np.ndarray, Z: np.ndarray, cluster_labels: np.ndarray, outdir: Path
) -> None:
    n = var_F.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    n_clusters = len(np.unique(cluster_labels))
    cut_height = Z[-(n_clusters), 2] if n_clusters <= len(Z) else Z[-1, 2]
    dendrogram(
        Z, ax=axes[0], labels=PARAM_NAMES,
        leaf_rotation=90, leaf_font_size=6, color_threshold=cut_height,
    )
    axes[0].set_title("Fisher clustering (Ward linkage)")

    order = np.argsort(cluster_labels)
    vv = np.log1p(var_F[np.ix_(order, order)])
    ticks = list(range(0, n, 5))
    ordered_names = [PARAM_NAMES[i] for i in order]
    im = axes[1].imshow(vv, cmap="viridis", aspect="auto")
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([ordered_names[i] for i in ticks], rotation=45, ha="right", fontsize=7)
    axes[1].set_yticks(ticks)
    axes[1].set_yticklabels([ordered_names[i] for i in ticks], fontsize=7)
    axes[1].set_title("log(1 + Var[F]) reordered by cluster")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    fig.savefig(outdir / "fisher_clustering.png", dpi=150)
    plt.close(fig)
    log("saved fisher_clustering.png")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIGS), default="smoke")
    parser.add_argument("--out-dir", type=Path, default=Path("results/camels_notebook_smoke"))
    parser.add_argument(
        "--data-path", type=Path, default=Path("data_scratch/data_L50_TNG_v3.hdf5"),
        help="Path to the CAMELS HDF5 file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--param-indices", type=str, default=None,
        help="Comma-separated parameter indices overriding Fisher selection, e.g. '0,1,2,3,4,5'",
    )
    parser.add_argument(
        "--retrain-subset", action="store_true", default=False,
        help=(
            "Retrain fishnets on the K-param subset before flattening. "
            "Default: use K×K block submatrix of the full 35D Fishers. "
            "Incompatible with --skip-fishnets."
        ),
    )
    parser.add_argument(
        "--from-dir", type=Path, default=None,
        help="Path to a prior run directory. Required when --skip-fishnets is set.",
    )
    parser.add_argument(
        "--skip-fishnets", action="store_true", default=False,
        help="Skip fishnet training; load fishnets_outputs.npz from --from-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.mode]
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"mode={args.mode}  outdir={outdir}")
    log(f"config={json.dumps(asdict(config), sort_keys=True)}")

    require_gpu_if_requested(args.require_gpu)

    if args.retrain_subset and args.skip_fishnets:
        raise ValueError("--retrain-subset and --skip-fishnets are mutually exclusive")

    param_indices_override = None
    if args.param_indices is not None:
        param_indices_override = [int(x.strip()) for x in args.param_indices.split(",")]

    if args.skip_fishnets:
        if args.from_dir is None:
            raise ValueError("--from-dir is required when --skip-fishnets is set")
        from_dir = args.from_dir.resolve()
        log(f"skipping fishnet training; loading artifacts from {from_dir}")
        fish_dir = from_dir / "fishnets-camels"
        if not (fish_dir / "fishnets_outputs.npz").exists():
            raise FileNotFoundError(f"fishnets_outputs.npz not found in {fish_dir}")
        meta_data = np.load(from_dir / "camels_meta.npz", allow_pickle=True)
        meta = {k: meta_data[k] for k in meta_data.files}
        sel_path = from_dir / "param_selection.npz"
        if param_indices_override is None and sel_path.exists():
            sel_data = np.load(sel_path)
            param_indices_override = sel_data["param_indices"].tolist()
            log(f"recovered param_indices from prior run: {param_indices_override}")
        np.savez(outdir / "camels_meta.npz", **meta)
        train_data = None
        fisher_results = None
    else:
        theta_train, data_train, theta_test, data_test, scaler, meta = load_camels_data(
            args.data_path, config, seed=args.seed
        )
        np.savez(outdir / "camels_meta.npz", **meta)
        fish_dir = train_fishnet_ensemble(
            config, theta_train, data_train, theta_test, data_test, outdir
        )
        train_data = (theta_train, data_train, theta_test, data_test)
        fisher_results = analyze_fisher_structure(config, fish_dir, outdir)

    param_indices = select_parameters(
        fish_dir, config, fisher_results, param_indices_override
    )
    np.savez(
        outdir / "param_selection.npz",
        param_indices=param_indices,
        param_names_selected=np.array([PARAM_NAMES[i] for i in param_indices]),
    )
    log(f"selected parameters: {[PARAM_NAMES[i] for i in param_indices]}")

    _, ensemble_w, _, flatten_model = fit_flattener(
        config, fish_dir, param_indices, outdir,
        retrain_subset=args.retrain_subset,
        train_data=train_data,
    )
    aligned = align_and_sample_sr_grid(config, outdir, ensemble_w, flatten_model)
    sr_dir, mdl_coords, frob_coords, analysis = run_symbolic_regression(
        config, aligned, outdir
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
        decimal=1,
        threshold=0.05,
    )
    print_discovered_expressions([sympy.simplify(e).evalf(2) for e in pruned_exprs])

    flatness = validate_flatness(aligned, mdl_coords, pruned_exprs)
    log("flatness scores")
    print(json.dumps(flatness, indent=2, sort_keys=True), flush=True)

    sr_dir.mkdir(exist_ok=True)
    with open(sr_dir / "sr_expressions.pkl", "wb") as handle:
        pickle.dump(
            {
                "mdl_coords":           mdl_coords,
                "frob_coords":          frob_coords,
                "pruned_exprs":         pruned_exprs,
                "flatness":             flatness,
                "analysis":             analysis,
                "rotation":             rotation,
                "prune_info":           prune_info,
                "param_indices":        param_indices.tolist(),
                "param_names_selected": [PARAM_NAMES[i] for i in param_indices],
                "theta_min":            meta["theta_min"],
                "theta_max":            meta["theta_max"],
            },
            handle,
        )
    shutil.copytree(fish_dir, sr_dir / "fishnets-camels", dirs_exist_ok=True)
    shutil.copy2(outdir / "camels_flattening.npz", sr_dir / "camels_flattening.npz")
    shutil.make_archive(str(outdir / "sr_results_camels"), "zip", root_dir=sr_dir)
    log(f"saved artifacts under {sr_dir}")


if __name__ == "__main__":
    main()
