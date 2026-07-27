#!/usr/bin/env python
"""Batch-safe version of tutorial_notebooks/imrphenomd_example.ipynb."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import signal
import subprocess
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from dataclasses import replace as dataclasses_replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import sympy
from pycbc.waveform import get_fd_waveform
from sklearn.decomposition import PCA
from tqdm import tqdm

import esr.generation.generator  # noqa: F401 - sanity-check ESR import.
import gw_notebook_run as gw
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


warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

M_SUN_SEC = 4.925491025543576e-6
M1_MIN, M1_MAX = 5.0, 50.0
M2_MIN, M2_MAX = 5.0, 50.0
D_L_MPC = 200.0
F_LOW, F_HIGH, DF = 20.0, 2048.0, 0.5
APPROX = "IMRPhenomD"


@dataclass(frozen=True)
class RunConfig:
    nsims: int
    n_pca: int
    pca_basis_max: int
    num_fishnets: int
    top_flatten_members: int
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
        num_fishnets=4,
        top_flatten_members=4,
        fish_epochs=1500,
        fish_hids_min=32,
        fish_hids_max=96,
        fish_patience=25,
        fish_layers=(2, 3),
        fish_batch_size=100,
        flatten_hidden_size=64,
        flatten_layers=3,
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
        num_fishnets=20,
        # 10 -> 20 on 2026-07-26. num_fishnets was already 20, but only the top
        # `top_flatten_members` by ensemble weight were passed to the flattener, so
        # the effective ensemble at the flattening/augmentation stage was 10 -- and
        # that is precisely the stage whose spread becomes y_std_sr, the per-point
        # uncertainty symbolic regression consumes. Using all 20 keeps that error
        # estimate consistent with the other experiments. Note this DOES change the
        # frozen "full" config that the already-reported 7/10 IMRPhenomD rebuttal
        # result was produced under, so that result must be re-run (or explicitly
        # labelled as the pre-change configuration) rather than mixed with new runs.
        top_flatten_members=20,
        fish_epochs=5000,
        fish_hids_min=50,
        fish_hids_max=300,
        fish_patience=30,
        fish_layers=(3, 5),
        fish_batch_size=200,
        flatten_hidden_size=100,
        flatten_layers=5,
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

# NeurIPS rebuttal configuration: same frozen architecture/optimizer/SR settings as
# "full" (no per-seed retuning), only n_train and n_aug bumped per the rebuttal
# protocol (500 training simulations, 2000 augmented coordinate evaluations).
CONFIGS["rebuttal"] = dataclasses_replace(CONFIGS["full"], nsims=500, sr_grid_size=2000)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require_gpu_if_requested(require_gpu: bool) -> None:
    backend = jax.default_backend()
    devices = jax.devices()
    log(f"JAX backend: {backend}")
    log(f"JAX devices: {devices}")
    if require_gpu and backend != "gpu":
        raise SystemExit("JAX did not initialize a GPU backend.")


# Additive-stride master-seed derivation, matching the convention already used in
# scripts/rosenbrock_notebook_run.py and scripts/gw_notebook_run.py
# (run_seed = args.seed + offset * stride). Stride is large enough that master seeds
# 0-9 never collide across stages.
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
INVERTIBILITY_TIMEOUT_SECONDS = 30


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


def chirp_mass(m1, m2):
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def symmetric_mass_ratio(m1, m2):
    return (m1 * m2) / (m1 + m2) ** 2


def a_ligo_psd(f):
    f = np.asarray(f, dtype=float)
    x = f / 215.0
    psd = 1e-49 * (x**-4.14 + 2.0 + 2.0 * x**2)
    return np.where(f >= 10.0, psd, np.inf)


def imrphenomd_waveform(m1, m2, n_freq):
    m_large, m_small = (float(m1), float(m2)) if m1 >= m2 else (float(m2), float(m1))
    hp, _ = get_fd_waveform(
        approximant=APPROX,
        mass1=m_large,
        mass2=m_small,
        spin1z=0.0,
        spin2z=0.0,
        delta_f=DF,
        f_lower=F_LOW,
        f_final=F_HIGH,
        distance=D_L_MPC,
    )
    h_full = np.asarray(hp.data, dtype=complex)
    h = np.zeros(n_freq, dtype=complex)
    i_start = int(round(F_LOW / DF))
    end = min(i_start + n_freq, h_full.size)
    h[: end - i_start] = h_full[i_start:end]
    return h


def simulator_data(config: RunConfig, seed: int, outdir: Path) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    key = jr.PRNGKey(seed)

    freqs = np.arange(F_LOW, F_HIGH, DF)
    whiten = np.sqrt(4.0 * DF) / np.sqrt(a_ligo_psd(freqs))
    n_freq = len(freqs)
    log(f"frequency grid: {F_LOW}-{freqs[-1]:.0f} Hz, {n_freq} bins")

    m1_all = rng.uniform(M1_MIN, M1_MAX, 2 * config.nsims)
    m2_all = rng.uniform(M2_MIN, M2_MAX, 2 * config.nsims)
    theta_all = np.stack([m1_all, m2_all], axis=1).astype(np.float32)

    # Compute each noiseless waveform exactly once and reuse it for both the PCA
    # basis fit and the noisy-data construction below (previously imrphenomd_waveform
    # was called a second time per sample for the noisy loop, silently doubling the
    # (expensive, pycbc-generated) waveform cost for no reason).
    log("computing noiseless IMRPhenomD waveforms (shared by PCA basis fit + noisy data)")
    hvecs_all = np.empty((2 * config.nsims, 2 * n_freq))
    for i in tqdm(range(2 * config.nsims), desc="IMRPhenomD waveforms"):
        h = imrphenomd_waveform(theta_all[i, 0], theta_all[i, 1], n_freq) * whiten
        hvecs_all[i] = np.concatenate([h.real, h.imag])

    log("building IMRPhenomD PCA basis")
    basis_idx = rng.choice(2 * config.nsims, min(2 * config.nsims, config.pca_basis_max), replace=False)
    pca = PCA(n_components=config.n_pca).fit(hvecs_all[basis_idx])
    cumvar = pca.explained_variance_ratio_.cumsum()
    log(f"PCA: {config.n_pca} components capture {cumvar[-1] * 100:.1f}% variance")

    log("generating noisy IMRPhenomD waveforms")
    _, sub = jr.split(key)
    keys = jr.split(sub, 2 * config.nsims)
    data_all = np.empty((2 * config.nsims, config.n_pca), dtype=np.float32)
    snr_all = np.empty(2 * config.nsims)
    for i in tqdm(range(2 * config.nsims), desc="noisy data"):
        hvec = hvecs_all[i]
        snr_all[i] = np.linalg.norm(hvec)
        noise = np.array(jr.normal(keys[i], shape=hvec.shape))
        data_all[i] = pca.transform((hvec + noise).reshape(1, -1)).flatten()

    theta_train, data_train = theta_all[: config.nsims], data_all[: config.nsims]
    theta_test, data_test = theta_all[config.nsims :], data_all[config.nsims :]
    mc_train = np.asarray(chirp_mass(theta_train[:, 0], theta_train[:, 1]))
    log(f"theta_train {theta_train.shape}; data_train {data_train.shape}")
    gw.plot_input_summary(theta_train, data_train, mc_train, snr_all[: config.nsims], cumvar, outdir)
    (outdir / "gw_input_summary.png").rename(outdir / "imrphenomd_input_summary.png")
    return {
        "theta_train": theta_train,
        "data_train": data_train,
        "theta_test": theta_test,
        "data_test": data_test,
        # PCA basis is fit on noiseless waveforms already required for train+test data
        # generation above -- shared preprocessing, not an additional simulator cost.
        "n_pca_basis_waveforms": int(len(basis_idx)),
    }


def train_fishnet_ensemble(
    config: RunConfig, data: dict[str, object], outdir: Path, seeds: dict[str, int]
):
    scaler = fit_theta_scaler(data["theta_train"], feature_range=(1.0, 2.0))
    theta_train_s = scaler.transform(data["theta_train"]).astype(np.float32)
    theta_test_s = scaler.transform(data["theta_test"]).astype(np.float32)
    log(f"scaled theta range: {theta_train_s.min(0)} to {theta_train_s.max(0)}")

    embedding_net = nn.Sequential(
        [nn.Dense(128), nn.gelu, nn.Dense(64), nn.gelu, nn.Dense(64), nn.gelu]
    )
    fish_dir = outdir / "fishnets-imr"
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
        seed_model=seeds["fish_model"],
        seed_train=seeds["fish_train"],
        outdir=str(fish_dir),
        update_pbar_every=25,
    )
    return fish_dir, scaler


def fit_flattener(config: RunConfig, fish_dir: Path, outdir: Path, seeds: dict[str, int]):
    with np.load(fish_dir / "fishnets_outputs.npz") as fish:
        thetas = jnp.array(fish["theta"])
        ensemble_weights = np.asarray(fish["ensemble_weights"])
        fs_np = np.asarray(fish["Fs"])

    finite_mask = np.isfinite(fs_np).all(axis=(1, 2, 3))
    if not finite_mask.any():
        raise RuntimeError("all fishnet ensemble members produced non-finite Fishers")
    fs_np = fs_np[finite_mask]
    ensemble_weights = ensemble_weights[finite_mask]

    topn = min(config.top_flatten_members, len(ensemble_weights))
    best = np.argsort(ensemble_weights)[-topn:]
    log(f"using top {topn} fishnet ensemble members for flattening")
    ensemble_weights = ensemble_weights[best]
    f_network_ensemble = jnp.array(fs_np[best])

    log("computing log-Euclidean median Fisher average")
    log_f_ens = jax.vmap(jax.vmap(gw._matrix_log_psd))(f_network_ensemble)
    f_avg_le = jax.vmap(gw._matrix_exp_sym)(jnp.median(log_f_ens, axis=0))

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
            seed=seeds["flatten"],
            flattener_activation="softplus",
            Fisher_to_flatten="average",
            output_prefix="imr_flatten",
            use_whitening=True,
            nn_inv=False,
            forward_backward_mlp=False,
            forward_backward_invertibility_weight=1.0,
            minmax_scale_inputs=True,
            grad_clip_norm=1.0,
            loss_type="log_frob",
            beta_det=0.1,
            augment_log_inputs=True,
            l1_alpha=0.0,
            do_plot=False,
            return_model=True,
            save_flatten_model_pickle=True,
            update_pbar_every=25,
        )
    finally:
        os.chdir(cwd_before)
    return w, ensemble_w, outputs_flatten, flatten_model


def align_and_sample_sr_grid(
    config: RunConfig, outdir: Path, ensemble_w, flatten_model, seeds: dict[str, int]
):
    log("aligning coordinates")
    aligned = load_and_process_data_v2(
        datapath=str(outdir) + os.sep,
        filename="imr_flatten.npz",
        num_samps=config.align_subsample,
        seed=seeds["align"],
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

    key = jr.PRNGKey(seeds["sr_grid"])
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


# Shared with the mdl_total recomputation in main() so the raw (non-normalized)
# description length reported in run_record.json is computed under the same
# length_penalty analyze_equations used to select the winning expressions.
SR_LENGTH_PENALTY = 2.0


def run_symbolic_regression(config: RunConfig, aligned: dict, outdir: Path, seeds: dict[str, int]):
    sr_dir = outdir / "sr_results_imr"
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
        allowed_symbols="add,mul,div,pow,constant,variable,exp",
        objectives=["r2", "length"],
    )
    equation_predicate = sr_structure_predicate(
        n_params=aligned["n_params"],
        check_nested_exp=True,
        max_exp_nesting=1,
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
        length_penalty=SR_LENGTH_PENALTY,
        equation_predicate=equation_predicate,
    )
    return sr_dir, mdl_coords, frob_coords, analysis


def mass_targets_correlations(physical_exprs, seed: int) -> dict[str, object]:
    """Doc's actual IMRPhenomD criterion: BOTH a total-mass-like (M=m1+m2) AND a
    complementary asymmetric (delta_m=m1-m2) component recovered -- not just a
    single best correlation against any of the GW TaylorF2 targets."""
    m1, m2 = sympy.symbols("m1 m2")
    rng = np.random.default_rng(seed)
    samples = rng.uniform([M1_MIN, M2_MIN], [M1_MAX, M2_MAX], size=(5000, 2))
    targets = {
        "total_mass": samples[:, 0] + samples[:, 1],
        "mass_difference": samples[:, 0] - samples[:, 1],
        "log_chirp_mass": np.log(chirp_mass(samples[:, 0], samples[:, 1])),
        "q_m2_over_m1": samples[:, 1] / samples[:, 0],
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

    best_total_mass = max(abs(row["total_mass"]) for row in rows)
    best_mass_difference = max(abs(row["mass_difference"]) for row in rows)
    return {
        "rows": rows,
        "total_mass_abs_corr": best_total_mass,
        "mass_difference_abs_corr": best_mass_difference,
    }


def validate_flatness(aligned: dict, mdl_coords, pruned_exprs) -> dict[str, float]:
    adhoc_coords = ["(X1 * X2) ^ (3./5.) / (X1 + X2) ^ (1./5.)", "X2 / X1"]
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
        "median_condition_raw": float(np.median(np.linalg.cond(np.asarray(aligned["Fs"])))),
        "median_condition_symbolic": float(np.median(np.linalg.cond(np.asarray(pruned_flats)))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIGS), default="full")
    parser.add_argument("--out-dir", type=Path, default=Path("results/imrphenomd_notebook_full"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--min-total-mass-corr",
        type=float,
        default=0.75,
        help="Fail if no physical expression correlates this strongly with M=m1+m2.",
    )
    parser.add_argument(
        "--min-mass-diff-corr",
        type=float,
        default=0.5,
        help="Fail if no physical expression correlates this strongly with the complementary "
        "asymmetric direction delta_m=m1-m2 (recovery requires both directions).",
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

    run_id = f"gw_imrphenomd_seed{args.seed}"
    config_manifest = {
        "run_id": run_id,
        "problem": "gw_imrphenomd",
        "master_seed": args.seed,
        "mode": args.mode,
        "config": asdict(config),
        "stage_seeds": seeds,
        "thresholds": {
            "min_total_mass_corr": args.min_total_mass_corr,
            "min_mass_diff_corr": args.min_mass_diff_corr,
        },
        "git_commit": git_commit_hash(),
    }
    with open(outdir / "config_manifest.json", "w") as handle:
        json.dump(config_manifest, handle, indent=2, sort_keys=True)

    require_gpu_if_requested(args.require_gpu)

    runtime_seconds: dict[str, float | None] = {}
    total_start = time.time()
    n_pca_basis_waveforms = 0

    def write_failure(stage: str, exc: Exception) -> None:
        runtime_seconds["total"] = time.time() - total_start
        record = {
            "run_id": run_id,
            "problem": "gw_imrphenomd",
            "master_seed": args.seed,
            "status": "failed",
            "failure_stage": stage,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(),
            "counts": {
                "n_train_simulations": config.nsims,
                "n_eval_simulations": config.nsims,
                "n_pca_simulations": n_pca_basis_waveforms,
                "pca_basis_shared_with_training": True,
                "n_augmented_coordinate_evaluations": config.sr_grid_size,
                "n_downstream_npe_simulations": 0,
            },
            "runtime_seconds": runtime_seconds,
        }
        with open(outdir / "run_record.json", "w") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
        log(f"FAILED at stage={stage}: {exc}\n{traceback.format_exc()}")

    try:
        stage_start = time.time()
        data = simulator_data(config, seeds["data"], outdir)
        n_pca_basis_waveforms = data["n_pca_basis_waveforms"]
        fish_dir, scaler = train_fishnet_ensemble(config, data, outdir, seeds)
        runtime_seconds["fishnets"] = time.time() - stage_start
    except Exception as exc:
        write_failure("fishnets", exc)
        raise

    try:
        stage_start = time.time()
        _, ensemble_w, _, flatten_model = fit_flattener(config, fish_dir, outdir, seeds)
        runtime_seconds["flatten"] = time.time() - stage_start
    except Exception as exc:
        write_failure("flatten", exc)
        raise

    try:
        # align_and_sample_sr_grid both aligns coordinates and draws+evaluates the
        # augmented SR grid (fresh theta samples pushed through every ensemble
        # flattening member) -- flatten -> augment -> SR ordering, per the doc.
        stage_start = time.time()
        aligned = align_and_sample_sr_grid(config, outdir, ensemble_w, flatten_model, seeds)
        gw.plot_coordinate_maps(aligned, outdir)
        (outdir / "gw_flattened_coordinates.png").rename(outdir / "imrphenomd_flattened_coordinates.png")
        runtime_seconds["alignment"] = time.time() - stage_start
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
            do_inner_snap=True,
            inner_snap_rel_tol=0.1,
            inner_snap_flat_tol=0.1,
            inner_snap_decimal=3,
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

        correlations = mass_targets_correlations(physical_exprs, seeds["validation"])
        log("physical expression correlations")
        print(json.dumps(correlations, indent=2, sort_keys=True), flush=True)

        flatness = validate_flatness(aligned, mdl_coords, pruned_exprs)
        log("flatness scores")
        print(json.dumps(flatness, indent=2, sort_keys=True), flush=True)
        gw.plot_physics_validation(aligned, physical_exprs, flatness, outdir)
        (outdir / "gw_physics_correlations.png").rename(outdir / "imrphenomd_physics_correlations.png")
        (outdir / "gw_flatness_scores.png").rename(outdir / "imrphenomd_flatness_scores.png")

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
            # sympy.solve can also raise outright (e.g. NotImplementedError) on some
            # discovered expressions, not just hang -- this is a supplementary
            # diagnostic and must never fail the whole run over it.
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
        runtime_seconds["symbolic_regression"] = time.time() - stage_start
    except Exception as exc:
        write_failure("symbolic_regression", exc)
        raise

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
                "invertibility": invertibility,
                "rank_info": rank_info,
                "scaler_scale": scaler.scale_,
                "scaler_min": scaler.min_,
                "scaler_data_min": scaler.data_min_,
                "scaler_data_max": scaler.data_max_,
            },
            handle,
        )
    shutil.copytree(fish_dir, sr_dir / "fishnets-imr", dirs_exist_ok=True)
    shutil.copy2(outdir / "imr_flatten.npz", sr_dir / "imr_flatten.npz")
    shutil.make_archive(str(outdir / "sr_results_imr"), "zip", root_dir=sr_dir)
    log(f"saved artifacts under {sr_dir}")

    runtime_seconds["npe"] = None
    runtime_seconds["total"] = time.time() - total_start

    geometric_loss_beats_mc_q = flatness["pruned"] < flatness["adhoc_mc_q"]
    success = (
        correlations["total_mass_abs_corr"] >= args.min_total_mass_corr
        and correlations["mass_difference_abs_corr"] >= args.min_mass_diff_corr
        and geometric_loss_beats_mc_q
    )

    run_record = {
        "run_id": run_id,
        "problem": "gw_imrphenomd",
        "master_seed": args.seed,
        "status": "success",
        "counts": {
            "n_train_simulations": config.nsims,
            "n_eval_simulations": config.nsims,
            "n_pca_simulations": n_pca_basis_waveforms,
            "pca_basis_shared_with_training": True,
            "n_augmented_coordinate_evaluations": config.sr_grid_size,
            "n_downstream_npe_simulations": 0,
        },
        "discovery": {
            "expressions_physical": [str(e) for e in physical_exprs],
            "expressions_canonical": [str(e) for e in mdl_coords],
            "success": success,
            "physics_alignment": correlations["total_mass_abs_corr"],
            "complementary_mass_diff_alignment": correlations["mass_difference_abs_corr"],
            "geometric_loss_beats_mc_q": geometric_loss_beats_mc_q,
            "mdl_total": mdl_total,
            "complexity_total": complexity_total,
            "symbolically_invertible": invertibility["is_symbolically_invertible"],
            "rank_deficient": bool(rank_info["rank_deficient"]),
        },
        "heldout_geometry": {
            "frob_raw": flatness["raw_scaled"],
            "frob_neural": flatness["nn"],
            "frob_symbolic": flatness["pruned"],
            "frob_adhoc_mc_q": flatness["adhoc_mc_q"],
            "median_condition_raw": flatness["median_condition_raw"],
            "median_condition_symbolic": flatness["median_condition_symbolic"],
        },
        "inference": {
            "crps_theta": None,
            "crps_eta": None,
            "coverage_error_theta": None,
            "coverage_error_eta": None,
        },
        "runtime_seconds": runtime_seconds,
    }
    with open(outdir / "run_record.json", "w") as handle:
        json.dump(run_record, handle, indent=2, sort_keys=True)
    log(f"wrote run record to {outdir / 'run_record.json'}")

    if not success:
        raise SystemExit(
            "Recovery criteria not met: "
            f"total_mass_corr={correlations['total_mass_abs_corr']:.3f} (min {args.min_total_mass_corr:.3f}), "
            f"mass_diff_corr={correlations['mass_difference_abs_corr']:.3f} (min {args.min_mass_diff_corr:.3f}), "
            f"geometric_loss_beats_mc_q={geometric_loss_beats_mc_q}"
        )
    log("IMRPhenomD notebook batch run complete")


if __name__ == "__main__":
    main()
