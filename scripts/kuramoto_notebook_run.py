#!/usr/bin/env python
"""Degeneracy distillery on the noisy Kuramoto synchronisation model.

``N`` phase oscillators with quenched random natural frequencies obey

    dphi_i = ( omega_i + (K/N) * sum_j sin(phi_j - phi_i) ) dt + sqrt(2 D) dW_i,
    omega_i ~ Normal(0, sigma_omega^2)

with parameters ``theta = (K, sigma_omega, D)``.  The natural frequencies are
latent and are redrawn for every simulation, so the marginal likelihood of an
observed trajectory requires integrating over ``N`` latent frequencies and the
Brownian paths: it is intractable, and only forward simulation is available.

The observable is what an oscillator experiment records: the Kuramoto order
parameter magnitude ``r(t) = |mean_j exp(i phi_j)|`` on a fixed real-time grid
after the transient, together with two permutation-invariant summaries of the
population -- quantiles of the per-oscillator drift rate relative to the
collective phase, and quantiles of the time-averaged alignment
``mean_t cos(phi_i - psi)``.  The order parameter trace alone leaves the phase
noise ``D`` unidentifiable; the population quantiles are what expose the width
of the locked cluster, which is set by ``D``.

Rescaling time by ``sigma_omega`` shows that the dynamics are controlled by the
natural coordinates

    K / sigma_omega  (distance from the synchronisation transition),
    D / sigma_omega  (relative noise),
    sigma_omega      (the overall clock rate),

rather than by ``(K, sigma_omega, D)`` individually.  A successful run recovers
that ratio basis, which is tested here both by correlation and by fitting
log-log exponents (a coordinate tracking ``K/sigma_omega`` must have exponent
vector proportional to ``(1, -1, 0)`` in ``(log K, log sigma_omega, log D)``).

Usage
-----
    python scripts/kuramoto_notebook_run.py --mode smoke --master-seed 0
    python scripts/kuramoto_notebook_run.py --mode full --master-seed 3 \
        --out-dir results/rebuttal_discovery/kuramoto/seed_3
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from dataclasses import asdict, dataclass, replace
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

# K/sigma spans [0.53, 6.4], straddling the mean-field transition
# K_c/sigma = 2*sqrt(2/pi) = 1.60 without spending most of the prior in the
# fully synchronised regime where r saturates.
K_MIN, K_MAX = 0.8, 3.2
SIGMA_MIN, SIGMA_MAX = 0.5, 1.5
D_MIN, D_MAX = 0.05, 0.80

THETA_NAMES = ("K", "sigma", "D")

# Population quantiles reported alongside the order-parameter trace.
POPULATION_QUANTILES = jnp.linspace(0.05, 0.95, 9)

# Ratio basis expressed as exponent vectors in (log K, log sigma, log D).
RATIO_BASIS = {
    "log_K_over_sigma": np.array([1.0, -1.0, 0.0]),
    "log_D_over_sigma": np.array([0.0, -1.0, 1.0]),
    "log_sigma": np.array([0.0, 1.0, 0.0]),
}


@dataclass(frozen=True)
class RunConfig:
    nsims: int
    n_oscillators: int
    dt: float
    transient_time: float
    n_observations: int
    observation_spacing: float
    sim_chunk: int
    num_fishnets: int
    fish_epochs: int
    fish_hids_min: int
    fish_hids_max: int
    fish_patience: int
    fish_layers: tuple[int, int]
    fish_batch_size: int
    flatten_batch_size: int
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
        nsims=200,
        n_oscillators=64,
        dt=0.01,
        transient_time=10.0,
        n_observations=32,
        observation_spacing=0.25,
        sim_chunk=200,
        num_fishnets=4,
        fish_epochs=1500,
        fish_hids_min=32,
        fish_hids_max=96,
        fish_patience=25,
        fish_layers=(2, 3),
        fish_batch_size=50,
        flatten_batch_size=100,
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
        sr_max_depth=12,
    ),
    "full": RunConfig(
        nsims=500,
        n_oscillators=128,
        dt=0.005,
        transient_time=20.0,
        n_observations=64,
        observation_spacing=0.25,
        sim_chunk=500,
        num_fishnets=20,
        fish_epochs=4000,
        fish_hids_min=50,
        fish_hids_max=300,
        fish_patience=30,
        fish_layers=(3, 5),
        fish_batch_size=100,
        flatten_batch_size=250,
        flatten_hidden_size=128,
        flatten_layers=5,
        flatten_epochs_phase1=2000,
        flatten_epochs_phase2=2500,
        flatten_finetune_epochs=500,
        flatten_min_epochs=1200,
        flatten_patience=50,
        align_subsample=4000,
        sr_grid_size=2000,
        sr_time_limit=300,
        sr_max_length=30,
        sr_max_depth=16,
    ),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def derive_seeds(master_seed: int) -> dict[str, int]:
    state = np.random.SeedSequence(master_seed).generate_state(8)
    names = (
        "simulator",
        "fishnet_model",
        "fishnet_train",
        "flatten",
        "align",
        "sr_grid",
        "sr_search",
        "spare",
    )
    return {name: int(value % (2**31 - 1)) for name, value in zip(names, state)}


def require_gpu_if_requested(require_gpu: bool) -> None:
    backend = jax.default_backend()
    log(f"JAX backend: {backend}")
    log(f"JAX devices: {jax.devices()}")
    if require_gpu and backend != "gpu":
        raise SystemExit("JAX did not initialize a GPU backend.")


# --------------------------------------------------------------------------
# Simulator: batched Euler-Maruyama for the noisy Kuramoto model
# --------------------------------------------------------------------------


def _order_parameter(phases: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.exp(1j * phases), axis=-1)


def _advance(phases, key, omegas, coupling, noise_scale, dt, n_steps):
    def body(carry, _):
        phases, key = carry
        z = _order_parameter(phases)[:, None]
        # (K/N) sum_j sin(phi_j - phi_i) == K * Im( z * exp(-i phi_i) )
        drift = omegas + coupling * jnp.imag(z * jnp.exp(-1j * phases))
        key, sub = jr.split(key)
        increment = noise_scale * jnp.sqrt(dt) * jr.normal(sub, phases.shape)
        return (phases + drift * dt + increment, key), None

    (phases, key), _ = jax.lax.scan(body, (phases, key), None, length=n_steps)
    return phases, key


def _simulate_chunk(key, theta_chunk, config: RunConfig) -> jnp.ndarray:
    batch = theta_chunk.shape[0]
    n_osc = config.n_oscillators
    coupling = theta_chunk[:, 0][:, None]
    sigma = theta_chunk[:, 1][:, None]
    noise_scale = jnp.sqrt(2.0 * theta_chunk[:, 2])[:, None]

    key, sub_omega, sub_phase = jr.split(key, 3)
    omegas = sigma * jr.normal(sub_omega, (batch, n_osc))
    phases = jr.uniform(sub_phase, (batch, n_osc), minval=0.0, maxval=2.0 * jnp.pi)

    transient_steps = int(round(config.transient_time / config.dt))
    gap_steps = max(1, int(round(config.observation_spacing / config.dt)))

    phases, key = _advance(
        phases, key, omegas, coupling, noise_scale, config.dt, transient_steps
    )

    # Phases are never wrapped, so differencing gives unwrapped drift directly.
    start = phases
    alignment = jnp.zeros((batch, n_osc))
    records = []
    for _ in range(config.n_observations):
        phases, key = _advance(
            phases, key, omegas, coupling, noise_scale, config.dt, gap_steps
        )
        z = _order_parameter(phases)
        records.append(jnp.abs(z))
        alignment = alignment + jnp.cos(phases - jnp.angle(z)[:, None])

    window = config.n_observations * gap_steps * config.dt
    drift = (phases - start) / window
    drift = drift - drift.mean(axis=-1, keepdims=True)
    alignment = alignment / config.n_observations

    return jnp.concatenate(
        [
            jnp.stack(records, axis=1),
            jnp.quantile(drift, POPULATION_QUANTILES, axis=-1).T,
            jnp.quantile(alignment, POPULATION_QUANTILES, axis=-1).T,
        ],
        axis=-1,
    )


def simulator_data(config: RunConfig, seed: int, outdir: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = 2 * config.nsims

    theta_all = np.stack(
        [
            rng.uniform(K_MIN, K_MAX, total),
            rng.uniform(SIGMA_MIN, SIGMA_MAX, total),
            rng.uniform(D_MIN, D_MAX, total),
        ],
        axis=1,
    ).astype(np.float32)

    log(f"integrating {total} Kuramoto systems of {config.n_oscillators} oscillators")
    key = jr.PRNGKey(seed)
    chunks = []
    for start in range(0, total, config.sim_chunk):
        stop = min(start + config.sim_chunk, total)
        key, sub = jr.split(key)
        traces = np.asarray(_simulate_chunk(sub, jnp.asarray(theta_all[start:stop]), config))
        if not np.isfinite(traces).all():
            raise RuntimeError("non-finite order parameter; reduce dt")
        chunks.append(traces)
        log(f"  simulated {stop}/{total}")
    data_all = np.concatenate(chunks, axis=0).astype(np.float32)

    theta_train, data_train = theta_all[: config.nsims], data_all[: config.nsims]
    theta_test, data_test = theta_all[config.nsims :], data_all[config.nsims :]
    log(f"theta_train {theta_train.shape}; data_train {data_train.shape}")

    plot_input_summary(theta_train, data_train, config, outdir)
    return {
        "theta_train": theta_train,
        "data_train": data_train,
        "theta_test": theta_test,
        "data_test": data_test,
    }


def plot_input_summary(theta, data, config: RunConfig, outdir: Path) -> None:
    ratio = theta[:, 0] / theta[:, 1]
    times = config.observation_spacing * np.arange(1, config.n_observations + 1)
    traces = data[:, : config.n_observations]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sc = axes[0].scatter(theta[:, 0], theta[:, 1], c=ratio, s=8, cmap="viridis")
    plt.colorbar(sc, ax=axes[0], label="K / sigma")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("sigma_omega")
    axes[0].set_title("Prior samples coloured by K / sigma")

    order = np.argsort(ratio)
    span = np.ptp(ratio)
    for idx in order[:: max(1, len(order) // 30)]:
        axes[1].plot(
            times,
            traces[idx],
            color=plt.cm.viridis((ratio[idx] - ratio.min()) / span),
            alpha=0.7,
            lw=1,
        )
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("r(t)")
    axes[1].set_title("Observable: order parameter traces")

    axes[2].scatter(ratio, traces.mean(axis=1), s=8, alpha=0.6)
    axes[2].axvline(2.0 * np.sqrt(2.0 / np.pi), ls="--", color="grey", label="mean-field K_c")
    axes[2].set_xlabel("K / sigma")
    axes[2].set_ylabel("time-averaged r")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Synchronisation transition")

    fig.tight_layout()
    fig.savefig(outdir / "kuramoto_input_summary.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------
# Distillery stages
# --------------------------------------------------------------------------


def train_fishnet_ensemble(config: RunConfig, data, seeds, outdir: Path):
    # [1, 2] is the distillery convention, shared with the GW scripts. It keeps
    # every scaled parameter strictly positive and of order unity, so symbolic
    # regression can form ratios and logs without a zero in the denominator or
    # argument. The scaled theta passes through the flattener and alignment
    # unchanged, so expressions_to_physical below uses sr_offset=0.
    scaler = fit_theta_scaler(data["theta_train"], feature_range=(1.0, 2.0))
    theta_train_s = scaler.transform(data["theta_train"]).astype(np.float32)
    theta_test_s = scaler.transform(data["theta_test"]).astype(np.float32)
    log(f"scaled theta range: {theta_train_s.min(0)} to {theta_train_s.max(0)}")

    embedding_net = nn.Sequential(
        [nn.Dense(128), nn.gelu, nn.Dense(64), nn.gelu, nn.Dense(64), nn.gelu]
    )

    fish_dir = outdir / "fishnets-kuramoto"
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
        seed_model=seeds["fishnet_model"],
        seed_train=seeds["fishnet_train"],
        outdir=str(fish_dir),
        update_pbar_every=25,
    )
    return fish_dir, scaler


def _matrix_log_psd(matrix, eps=1e-12):
    sym = 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))
    evals, evecs = jnp.linalg.eigh(sym)
    return (evecs * jnp.log(jnp.maximum(evals, eps))[..., None, :]) @ jnp.swapaxes(
        evecs, -1, -2
    )


def _matrix_exp_sym(matrix):
    sym = 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))
    evals, evecs = jnp.linalg.eigh(sym)
    return (evecs * jnp.exp(evals)[..., None, :]) @ jnp.swapaxes(evecs, -1, -2)


def fit_flattener(config: RunConfig, fish_dir: Path, seeds, outdir: Path):
    with np.load(fish_dir / "fishnets_outputs.npz") as fish:
        thetas = jnp.array(fish["theta"])
        ensemble_weights = np.asarray(fish["ensemble_weights"])
        fs_np = np.asarray(fish["Fs"])

    finite = np.isfinite(fs_np).all(axis=(1, 2, 3))
    if not finite.any():
        raise RuntimeError("all fishnet ensemble members produced non-finite Fishers")
    if finite.sum() < len(finite):
        log(f"filtering {len(finite) - int(finite.sum())} non-finite fishnet members")
        fs_np = fs_np[finite]
        ensemble_weights = ensemble_weights[finite]

    ensemble = jnp.array(fs_np)
    log_ens = jax.vmap(jax.vmap(_matrix_log_psd))(ensemble)
    f_avg = jax.vmap(_matrix_exp_sym)(jnp.median(log_ens, axis=0))

    log("fitting flattening model")
    cwd_before = Path.cwd()
    os.chdir(outdir)
    try:
        w, ensemble_w, outputs, flatten_model = fit_flattening(
            ensemble,
            thetas,
            F_avg=f_avg,
            ensemble_weights=ensemble_weights,
            hidden_size=config.flatten_hidden_size,
            n_layers=config.flatten_layers,
            batch_size=config.flatten_batch_size,
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
            output_prefix="kuramoto_flatten",
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
    return w, ensemble_w, outputs, flatten_model


def align_and_augment(config: RunConfig, seeds, outdir: Path, ensemble_w, flatten_model):
    log("aligning coordinates")
    aligned = load_and_process_data_v2(
        datapath=str(outdir) + os.sep,
        filename="kuramoto_flatten.npz",
        num_samps=config.align_subsample,
        seed=seeds["align"],
        process_ensemble=True,
        n_d=1.0,
        align_mode="procrustes",
        separate_nonlinearity=True,
        canonicalize="permute_and_sign",
        use_prior_normalization=True,
        restore_reference_mean=True,
        Fisher_to_flatten="average",
        verbose=False,
    )

    x = aligned["X"]
    y = aligned["y"]
    y = y - y.min(0)
    n_params = x.shape[1]
    log(f"aligned X {x.shape}; y {y.shape}")

    log(f"augmenting SR training set with {config.sr_grid_size} fresh prior draws")
    key = jr.PRNGKey(seeds["sr_grid"])
    x_sr = jr.uniform(
        key, minval=x.min(0), maxval=x.max(0), shape=(config.sr_grid_size, n_params)
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
    y_sr = y_sr - y_sr.min(0)

    return {
        "data": aligned,
        "X": x,
        "y": y,
        "y_std": aligned["y_std"],
        "dy_sr": aligned["dy_sr"],
        "Fs": aligned["Fs"],
        "n_params": n_params,
        "X_sr": np.asarray(x_sr),
        "y_sr": y_sr,
        "y_std_sr": y_std_sr,
    }


def run_symbolic_regression(config: RunConfig, aligned: dict, seeds, outdir: Path):
    sr_dir = outdir / "sr_results_kuramoto"
    sr_dir.mkdir(exist_ok=True)
    log(f"running symbolic regression into {sr_dir}")
    fit_symbolic_regression(
        aligned["X_sr"],
        aligned["y_sr"],
        aligned["y_std_sr"],
        parent_dir=str(sr_dir) + os.sep,
        random_state=seeds["sr_search"],
        time_limit=config.sr_time_limit,
        max_length=config.sr_max_length,
        max_depth=config.sr_max_depth,
        allowed_symbols="add,mul,div,pow,constant,variable,square,sqrt,logabs",
        objectives=["r2", "length"],
    )

    predicate = sr_structure_predicate(
        n_params=aligned["n_params"], forbid_self_transcendental=True
    )
    summaries = filter_pareto_fronts(str(sr_dir), aligned["n_params"], predicate)
    removed = sum(int(s["removed"]) for s in summaries)
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
        max_complexity_thresh=18,
        length_penalty=2.0,
        equation_predicate=predicate,
    )
    return sr_dir, mdl_coords, frob_coords, analysis


def _evaluate(physical_exprs, samples):
    k_sym, sigma_sym, d_sym = sympy.symbols("K sigma D")
    k, sigma, d = samples[:, 0], samples[:, 1], samples[:, 2]
    values = []
    for expr in physical_exprs:
        fn = sympy.lambdify((k_sym, sigma_sym, d_sym), expr, modules="numpy")
        out = np.asarray(fn(k, sigma, d), dtype=float)
        values.append(np.broadcast_to(out, (samples.shape[0],)))
    return values


def physics_correlations(physical_exprs) -> dict[str, object]:
    rng = np.random.default_rng(123)
    samples = rng.uniform(
        [K_MIN, SIGMA_MIN, D_MIN], [K_MAX, SIGMA_MAX, D_MAX], size=(5000, 3)
    )
    k, sigma, d = samples[:, 0], samples[:, 1], samples[:, 2]

    targets = {
        "log_K_over_sigma": np.log(k / sigma),
        "log_D_over_sigma": np.log(d / sigma),
        "log_sigma": np.log(sigma),
        "K_over_sigma": k / sigma,
        "D_over_sigma": d / sigma,
    }

    rows = []
    for i, values in enumerate(_evaluate(physical_exprs, samples)):
        finite = np.isfinite(values)
        row = {"component": i, "expr": str(physical_exprs[i])}
        for name, target in targets.items():
            if finite.sum() < 100 or np.std(values[finite]) == 0:
                row[name] = 0.0
            else:
                row[name] = float(np.corrcoef(values[finite], target[finite])[0, 1])
        rows.append(row)

    best_coupling = max(
        max(abs(row["log_K_over_sigma"]), abs(row["K_over_sigma"])) for row in rows
    )
    best_noise = max(
        max(abs(row["log_D_over_sigma"]), abs(row["D_over_sigma"])) for row in rows
    )
    return {
        "rows": rows,
        "best_coupling_ratio_abs_corr": best_coupling,
        "best_noise_ratio_abs_corr": best_noise,
        "best_ratio_abs_corr": min(best_coupling, best_noise),
    }


def log_exponent_fit(physical_exprs) -> dict[str, object]:
    """Fit each coordinate as a power law in (K, sigma, D) and match the ratio basis."""
    rng = np.random.default_rng(321)
    samples = rng.uniform(
        [K_MIN, SIGMA_MIN, D_MIN], [K_MAX, SIGMA_MAX, D_MAX], size=(4000, 3)
    )
    design = np.concatenate(
        [np.ones((samples.shape[0], 1)), np.log(samples)], axis=1
    )

    fits = []
    for i, values in enumerate(_evaluate(physical_exprs, samples)):
        finite = np.isfinite(values)
        if finite.sum() < 100:
            continue
        coef, *_ = np.linalg.lstsq(design[finite], values[finite], rcond=None)
        exponents = coef[1:]
        norm = np.linalg.norm(exponents)
        if norm < 1e-9:
            continue
        unit = exponents / norm
        similarities = {
            name: float(abs(unit @ (basis / np.linalg.norm(basis))))
            for name, basis in RATIO_BASIS.items()
        }
        best = max(similarities, key=similarities.get)
        fits.append(
            {
                "component": i,
                "exponents_log_K_sigma_D": [float(v) for v in exponents],
                "cosine_similarity": similarities,
                "closest_ratio": best,
                "closest_cosine": similarities[best],
                "relative_residual": float(
                    np.std(values[finite] - design[finite] @ coef)
                    / (np.std(values[finite]) + 1e-12)
                ),
            }
        )

    matched = {name: 0.0 for name in RATIO_BASIS}
    for fit in fits:
        for name, value in fit["cosine_similarity"].items():
            matched[name] = max(matched[name], value)
    return {
        "fits": fits,
        "best_cosine_per_ratio": matched,
        "worst_of_best_cosine": min(matched.values()) if matched else 0.0,
    }


def validate_flatness(aligned: dict, mdl_coords, pruned_exprs) -> dict[str, float]:
    # Textbook nondimensionalisation baseline.
    adhoc = ["X1 / X2", "X3 / X2", "X2"]
    adhoc_flats, _ = check_flattening(adhoc, X=aligned["X"], Fs=aligned["Fs"])
    mdl_flats, _ = check_flattening(mdl_coords, X=aligned["X"], Fs=aligned["Fs"])
    pruned_flats, _ = check_flattening(pruned_exprs, X=aligned["X"], Fs=aligned["Fs"])
    nn_flats = jax.vmap(flatten_with_numerical_jacobian)(aligned["dy_sr"], aligned["Fs"])

    identity = np.eye(aligned["n_params"])

    def fro_score(q):
        return np.linalg.norm(np.asarray(q) - identity, axis=(-2, -1))

    return {
        "raw_theta": float(np.median(fro_score(aligned["Fs"]))),
        "adhoc_ratio_basis": float(np.median(fro_score(adhoc_flats))),
        "mdl": float(np.median(fro_score(mdl_flats))),
        "pruned": float(np.median(fro_score(pruned_flats))),
        "nn": float(np.median(fro_score(nn_flats))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIGS), default="smoke")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/kuramoto_notebook")
    )
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sr-time-limit",
        type=int,
        default=None,
        help="Override the symbolic-regression budget per component, in seconds.",
    )
    parser.add_argument(
        "--min-ratio-corr",
        type=float,
        default=0.7,
        help="Fail unless both K/sigma and D/sigma are recovered at least this strongly.",
    )
    parser.add_argument(
        "--min-ratio-cosine",
        type=float,
        default=0.8,
        help="Fail unless every ratio-basis direction is matched to this cosine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.mode]
    if args.sr_time_limit is not None:
        config = replace(config, sr_time_limit=args.sr_time_limit)
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = derive_seeds(args.master_seed)

    log(f"running mode={args.mode}; outdir={outdir}")
    log(f"master_seed={args.master_seed}; derived seeds={json.dumps(seeds)}")
    log(f"config={json.dumps(asdict(config), sort_keys=True)}")
    require_gpu_if_requested(args.require_gpu)

    timings: dict[str, float] = {}
    t_start = time.time()

    t0 = time.time()
    data = simulator_data(config, seeds["simulator"], outdir)
    timings["simulation"] = time.time() - t0

    t0 = time.time()
    fish_dir, scaler = train_fishnet_ensemble(config, data, seeds, outdir)
    timings["fishnets"] = time.time() - t0

    t0 = time.time()
    _, ensemble_w, _, flatten_model = fit_flattener(config, fish_dir, seeds, outdir)
    timings["flatten"] = time.time() - t0

    t0 = time.time()
    aligned = align_and_augment(config, seeds, outdir, ensemble_w, flatten_model)
    timings["alignment_and_augmentation"] = time.time() - t0

    t0 = time.time()
    sr_dir, mdl_coords, frob_coords, analysis = run_symbolic_regression(
        config, aligned, seeds, outdir
    )
    timings["symbolic_regression"] = time.time() - t0

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
        snap_rel_tol=0.2,
        snap_flat_tol=0.2,
        decimal=2,
        threshold=0.5,
    )
    print_discovered_expressions(
        pruned_exprs, name_map={"X1": "K", "X2": "sigma", "X3": "D"}
    )

    physical_exprs = expressions_to_physical(
        pruned_exprs, scaler, sr_offset=0.0, theta_names=THETA_NAMES, decimal=3
    )
    log("physical expressions")
    for k, expr in enumerate(physical_exprs):
        print(f"  eta_{k} = {expr}", flush=True)

    correlations = physics_correlations(physical_exprs)
    exponents = log_exponent_fit(physical_exprs)
    log("physical expression correlations")
    print(json.dumps(correlations, indent=2, sort_keys=True), flush=True)
    log("power-law exponent fits")
    print(json.dumps(exponents, indent=2, sort_keys=True), flush=True)

    flatness = validate_flatness(aligned, mdl_coords, pruned_exprs)
    log("flatness scores")
    print(json.dumps(flatness, indent=2, sort_keys=True), flush=True)

    timings["total"] = time.time() - t_start
    corr_ok = correlations["best_ratio_abs_corr"] >= args.min_ratio_corr
    cosine_ok = exponents["worst_of_best_cosine"] >= args.min_ratio_cosine
    success = bool(corr_ok and cosine_ok)

    summary = {
        "run_id": f"kuramoto_seed{args.master_seed}",
        "problem": "kuramoto",
        "master_seed": args.master_seed,
        "mode": args.mode,
        "status": "success" if success else "criterion_not_met",
        "seeds": seeds,
        "counts": {
            "n_train_simulations": config.nsims,
            "n_eval_simulations": config.nsims,
            "n_augmented_coordinate_evaluations": config.sr_grid_size,
            "n_oscillators": config.n_oscillators,
            "n_observations": config.n_observations,
            "observable_dimension": config.n_observations + 2 * len(POPULATION_QUANTILES),
        },
        "discovery": {
            "expressions_physical": [str(e) for e in physical_exprs],
            "success": success,
            "best_ratio_abs_corr": correlations["best_ratio_abs_corr"],
            "best_coupling_ratio_abs_corr": correlations["best_coupling_ratio_abs_corr"],
            "best_noise_ratio_abs_corr": correlations["best_noise_ratio_abs_corr"],
            "worst_of_best_cosine": exponents["worst_of_best_cosine"],
        },
        "heldout_geometry": flatness,
        "runtime_seconds": timings,
    }
    with open(outdir / "run_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    with open(sr_dir / "sr_expressions.pkl", "wb") as handle:
        pickle.dump(
            {
                "mdl_coords": mdl_coords,
                "frob_coords": frob_coords,
                "pruned_exprs": pruned_exprs,
                "physical_exprs": [str(e) for e in physical_exprs],
                "correlations": correlations,
                "exponents": exponents,
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
    shutil.copytree(fish_dir, sr_dir / "fishnets-kuramoto", dirs_exist_ok=True)
    shutil.copy2(outdir / "kuramoto_flatten.npz", sr_dir / "kuramoto_flatten.npz")
    shutil.make_archive(str(outdir / "sr_results_kuramoto"), "zip", root_dir=sr_dir)
    log(f"saved artifacts under {sr_dir}")

    if not success:
        raise SystemExit(
            "Did not recover the ratio basis: "
            f"corr={correlations['best_ratio_abs_corr']:.3f}, "
            f"cosine={exponents['worst_of_best_cosine']:.3f}"
        )
    log("Kuramoto distillery run complete")


if __name__ == "__main__":
    main()
