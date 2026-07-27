#!/usr/bin/env python
"""Degeneracy distillery on forced 2D turbulence (Kolmogorov flow).

We solve the incompressible 2D Navier-Stokes equations in vorticity form on a
periodic box with sinusoidal (Kolmogorov) forcing,

    d_t w + u . grad w = nu * lap w - f0 * k_f * cos(k_f * y),

with ``theta = (f0, nu)`` drawn from a prior and the forcing wavenumber ``k_f``
held fixed (it is an integer on a periodic box, so it cannot be varied smoothly).

The observable is the *normalised*, time-averaged, radially-averaged enstrophy
spectrum in the statistically steady state.  Dividing out the amplitude removes
the velocity scale, so by dimensional analysis the observable distribution
depends on the two parameters only through the single Reynolds number

    Re = f0 / (nu^2 * k_f^3)      i.e.      log Re = log f0 - 2 log nu + const.

Two details make that invariance hold numerically rather than only on paper.
Each run starts from the laminar Kolmogorov solution
``omega = -U k_f cos(k_f y)``, ``U = f0 / (nu k_f^2)``, perturbed by noise of
*relative* amplitude, because spinning up to the laminar state from rest takes
``O(Re)`` turnover times and would otherwise leave high-Re runs far from steady
state.  The step size is then set from ``U`` so every run advances the same
number of eddy turnovers and satisfies the same CFL condition.  A single
snapshot is dominated by realisation noise, so the spectrum is averaged over
many decorrelated snapshots, exactly as a laboratory measurement would be.

A successful run recovers one identifiable coordinate aligned with ``log Re``
and one nuisance direction, from spectra alone and with no dimensional analysis
supplied to the method.

Usage
-----
    python scripts/kolmogorov_notebook_run.py --mode smoke --master-seed 0
    python scripts/kolmogorov_notebook_run.py --mode full --master-seed 3 \
        --out-dir results/rebuttal_discovery/kolmogorov/seed_3
"""

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

# Prior spans Re ~ 14 to ~390 at k_f = 4: unstable everywhere (the Kolmogorov
# flow loses stability near Re ~ 1) but still resolved on a 64^2 grid.
F0_MIN, F0_MAX = 0.8, 2.5
NU_MIN, NU_MAX = 0.010, 0.030
K_FORCING = 4

# Initial perturbation relative to the laminar vorticity amplitude, so that the
# initial condition is identical in nondimensional terms across the prior.
IC_PERTURBATION = 0.05

THETA_NAMES = ("f0", "nu")


@dataclass(frozen=True)
class RunConfig:
    nsims: int
    grid: int
    spin_steps: int
    n_spectra: int
    spectrum_gap_steps: int
    cfl: float
    n_spectral_bins: int
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
        nsims=100,
        grid=32,
        spin_steps=1200,
        n_spectra=12,
        spectrum_gap_steps=100,
        cfl=0.15,
        n_spectral_bins=8,
        sim_chunk=100,
        num_fishnets=4,
        fish_epochs=1500,
        fish_hids_min=32,
        fish_hids_max=96,
        fish_patience=25,
        fish_layers=(2, 3),
        fish_batch_size=50,
        flatten_batch_size=50,
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
        sr_max_length=20,
        sr_max_depth=10,
    ),
    "full": RunConfig(
        nsims=500,
        grid=64,
        spin_steps=3500,
        n_spectra=28,
        spectrum_gap_steps=200,
        cfl=0.15,
        n_spectral_bins=16,
        sim_chunk=250,
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
        sr_max_length=25,
        sr_max_depth=12,
    ),
}

# NeurIPS rebuttal configuration. "full" already trains on 500 simulations (the
# campaign target used by the sibling scripts) and augments with 2000 fresh
# coordinate-map evaluations, so "rebuttal" is a same-values alias -- this is
# already the most expensive of the three intractable-likelihood experiments
# (~30-45 min pure solver time on CPU per the handoff doc), and there is no
# slack to double it the way rosenbrock's rebuttal config bumps nsims 250->500.
CONFIGS["rebuttal"] = replace(CONFIGS["full"])

# Shared with the mdl_total recomputation in main() so the raw (non-normalized)
# description length reported in run_record.json is computed under the same
# length_penalty analyze_equations used to select the winning expressions.
SR_LENGTH_PENALTY = 2.0
INVERTIBILITY_TIMEOUT_SECONDS = 30


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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


def reynolds_number(f0: np.ndarray, nu: np.ndarray) -> np.ndarray:
    return f0 / (nu**2 * K_FORCING**3)


# --------------------------------------------------------------------------
# Simulator: batched pseudospectral Navier-Stokes
# --------------------------------------------------------------------------


def _spectral_operators(grid: int):
    wavenumbers = jnp.fft.fftfreq(grid, d=1.0 / grid)
    kx = wavenumbers[:, None]
    ky = wavenumbers[None, :]
    ksq = kx**2 + ky**2
    inv_ksq = jnp.where(ksq == 0, 0.0, 1.0 / jnp.where(ksq == 0, 1.0, ksq))
    dealias = (jnp.abs(kx) < grid / 3) & (jnp.abs(ky) < grid / 3)
    return kx, ky, ksq, inv_ksq, dealias.astype(jnp.complex64)


def _forcing_hat(grid: int) -> jnp.ndarray:
    """Curl of ``f0 * sin(k_f y) x_hat`` at unit amplitude, in Fourier space."""
    y = 2.0 * jnp.pi * jnp.arange(grid) / grid
    forcing = -K_FORCING * jnp.cos(K_FORCING * y)[None, :] * jnp.ones((grid, 1))
    return jnp.fft.fft2(forcing)


def _nonlinear(w_hat, kx, ky, inv_ksq, dealias):
    psi_hat = w_hat * inv_ksq
    u = jnp.real(jnp.fft.ifft2(1j * ky * psi_hat))
    v = jnp.real(jnp.fft.ifft2(-1j * kx * psi_hat))
    w_x = jnp.real(jnp.fft.ifft2(1j * kx * w_hat))
    w_y = jnp.real(jnp.fft.ifft2(1j * ky * w_hat))
    return jnp.fft.fft2(-(u * w_x + v * w_y)) * dealias


def _integrate(w_hat, f0, nu, dt, n_steps, ops, base_forcing):
    """Integrating-factor RK4: viscosity is handled exactly, advection by RK4."""
    kx, ky, ksq, inv_ksq, dealias = ops
    decay = nu[:, None, None] * ksq[None, :, :]
    forcing_hat = f0[:, None, None] * base_forcing[None, :, :]
    step = dt[:, None, None]

    def rhs(tau, v_hat):
        w = v_hat * jnp.exp(-decay * tau)
        return jnp.exp(decay * tau) * (
            _nonlinear(w, kx, ky, inv_ksq, dealias) + forcing_hat
        )

    def body(w_hat, _):
        k1 = rhs(0.0, w_hat)
        k2 = rhs(step / 2, w_hat + step / 2 * k1)
        k3 = rhs(step / 2, w_hat + step / 2 * k2)
        k4 = rhs(step, w_hat + step * k3)
        v_new = w_hat + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return v_new * jnp.exp(-decay * step), None

    w_hat, _ = jax.lax.scan(body, w_hat, None, length=n_steps)
    return w_hat


def _shell_masks(grid: int, n_bins: int) -> jnp.ndarray:
    wavenumbers = np.fft.fftfreq(grid, d=1.0 / grid)
    kmag = np.sqrt(wavenumbers[:, None] ** 2 + wavenumbers[None, :] ** 2)
    shell = np.rint(kmag).astype(int)
    return jnp.asarray(
        np.stack([(shell == k).astype(np.float32) for k in range(1, n_bins + 1)])
    )


def _shell_power(w_hat, masks) -> jnp.ndarray:
    return jnp.einsum("bxy,sxy->bs", jnp.abs(w_hat) ** 2, masks)


def _simulate_chunk(key, theta_chunk, config: RunConfig, ops, base_forcing, masks):
    f0 = theta_chunk[:, 0]
    nu = theta_chunk[:, 1]

    # Laminar velocity scale; setting dt proportional to 1/U makes every
    # simulation advance the same number of eddy turnovers, and enforces CFL.
    velocity = f0 / (nu * K_FORCING**2)
    dt = config.cfl * (2.0 * jnp.pi / config.grid) / velocity

    amplitude = (velocity * K_FORCING)[:, None, None]
    y = 2.0 * jnp.pi * jnp.arange(config.grid) / config.grid
    laminar = -amplitude * jnp.cos(K_FORCING * y)[None, None, :]
    noise = jr.normal(key, (theta_chunk.shape[0], config.grid, config.grid))
    w_hat = jnp.fft.fft2(laminar + IC_PERTURBATION * amplitude * noise)

    w_hat = _integrate(w_hat, f0, nu, dt, config.spin_steps, ops, base_forcing)

    total = jnp.zeros((theta_chunk.shape[0], config.n_spectral_bins))
    for _ in range(config.n_spectra):
        w_hat = _integrate(
            w_hat, f0, nu, dt, config.spectrum_gap_steps, ops, base_forcing
        )
        total = total + _shell_power(w_hat, masks)

    total = total / jnp.sum(total, axis=-1, keepdims=True)
    return jnp.log(total + 1e-12), w_hat


def simulator_data(config: RunConfig, seed: int, outdir: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = 2 * config.nsims

    theta_all = np.stack(
        [
            rng.uniform(F0_MIN, F0_MAX, total),
            rng.uniform(NU_MIN, NU_MAX, total),
        ],
        axis=1,
    ).astype(np.float32)

    ops = _spectral_operators(config.grid)
    base_forcing = _forcing_hat(config.grid)
    masks = _shell_masks(config.grid, config.n_spectral_bins)

    per_step = config.cfl * K_FORCING / config.grid
    spin_turnovers = config.spin_steps * per_step
    avg_turnovers = config.n_spectra * config.spectrum_gap_steps * per_step
    log(
        f"integrating {total} flows on {config.grid}^2: {spin_turnovers:.1f} turnovers "
        f"spin-up, then {config.n_spectra} spectra over {avg_turnovers:.1f} turnovers"
    )

    key = jr.PRNGKey(seed)
    chunks = []
    example_field = None
    for start in range(0, total, config.sim_chunk):
        stop = min(start + config.sim_chunk, total)
        key, sub = jr.split(key)
        spectra, w_hat = _simulate_chunk(
            sub, jnp.asarray(theta_all[start:stop]), config, ops, base_forcing, masks
        )
        spectra = np.asarray(spectra)
        if not np.isfinite(spectra).all():
            raise RuntimeError(
                "non-finite spectra: the solver went unstable, lower cfl in the config"
            )
        # A pile-up in the last shell means the dissipation range is unresolved.
        if np.median(spectra[:, -1]) > -5.0:
            log(
                f"WARNING: median log occupancy of the cutoff shell is "
                f"{np.median(spectra[:, -1]):.2f}; the grid may be too coarse for this Re range"
            )
        if example_field is None:
            example_field = np.asarray(jnp.real(jnp.fft.ifft2(w_hat[0])))
        chunks.append(spectra)
        log(f"  simulated {stop}/{total}")
    data_all = np.concatenate(chunks, axis=0).astype(np.float32)

    theta_train, data_train = theta_all[: config.nsims], data_all[: config.nsims]
    theta_test, data_test = theta_all[config.nsims :], data_all[config.nsims :]
    log(f"theta_train {theta_train.shape}; data_train {data_train.shape}")

    plot_input_summary(theta_train, data_train, example_field, outdir)
    return {
        "theta_train": theta_train,
        "data_train": data_train,
        "theta_test": theta_test,
        "data_test": data_test,
    }


def plot_input_summary(theta, data, example_field, outdir: Path) -> None:
    re = reynolds_number(theta[:, 0], theta[:, 1])
    order = np.argsort(re)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sc = axes[0].scatter(theta[:, 0], theta[:, 1], c=np.log10(re), s=8, cmap="magma")
    plt.colorbar(sc, ax=axes[0], label="log10 Re")
    axes[0].set_xlabel("f0")
    axes[0].set_ylabel("nu")
    axes[0].set_title("Prior samples coloured by Reynolds number")

    for idx in order[:: max(1, len(order) // 25)]:
        axes[1].plot(
            np.arange(1, data.shape[1] + 1),
            data[idx],
            color=plt.cm.magma((np.log10(re[idx]) - np.log10(re).min()) / np.ptp(np.log10(re))),
            alpha=0.7,
            lw=1,
        )
    axes[1].set_xlabel("shell wavenumber")
    axes[1].set_ylabel("log normalised enstrophy")
    axes[1].set_title("Observable: time-averaged normalised spectra")

    if example_field is not None:
        im = axes[2].imshow(example_field.T, origin="lower", cmap="RdBu_r")
        plt.colorbar(im, ax=axes[2], label="vorticity")
    axes[2].set_title("Example steady-state vorticity")

    fig.tight_layout()
    fig.savefig(outdir / "kolmogorov_input_summary.png", dpi=180)
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

    fish_dir = outdir / "fishnets-kolmogorov"
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
            output_prefix="kolmogorov_flatten",
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
            save_flatten_model_pickle=True,
            update_pbar_every=25,
        )
    finally:
        os.chdir(cwd_before)
    return w, ensemble_w, outputs, flatten_model


def align_and_augment(config: RunConfig, seeds, outdir: Path, ensemble_w, flatten_model):
    log("aligning coordinates")
    aligned = load_and_process_data_v2(
        datapath=str(outdir) + os.sep,
        filename="kolmogorov_flatten.npz",
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
    sr_dir = outdir / "sr_results_kolmogorov"
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
        max_complexity_thresh=16,
        length_penalty=SR_LENGTH_PENALTY,
        equation_predicate=predicate,
    )
    return sr_dir, mdl_coords, frob_coords, analysis


def physics_correlations(physical_exprs) -> dict[str, object]:
    f0_sym, nu_sym = sympy.symbols("f0 nu")
    rng = np.random.default_rng(123)
    samples = rng.uniform([F0_MIN, NU_MIN], [F0_MAX, NU_MAX], size=(5000, 2))
    f0, nu = samples[:, 0], samples[:, 1]

    targets = {
        "log_Re": np.log(reynolds_number(f0, nu)),
        "Re": reynolds_number(f0, nu),
        # Velocity scale: the natural second (non-Reynolds) coordinate.
        "log_velocity_scale": np.log(f0 / (nu * K_FORCING**2)),
        "log_f0": np.log(f0),
        "log_nu": np.log(nu),
    }

    rows = []
    for i, expr in enumerate(physical_exprs):
        fn = sympy.lambdify((f0_sym, nu_sym), expr, modules="numpy")
        values = np.asarray(fn(f0, nu), dtype=float)
        values = np.broadcast_to(values, (samples.shape[0],))
        finite = np.isfinite(values)
        row = {"component": i, "expr": str(expr)}
        for name, target in targets.items():
            if finite.sum() < 100 or np.std(values[finite]) == 0:
                row[name] = 0.0
            else:
                row[name] = float(np.corrcoef(values[finite], target[finite])[0, 1])
        rows.append(row)

    best_re = max(max(abs(row["log_Re"]), abs(row["Re"])) for row in rows)
    return {
        "rows": rows,
        "best_reynolds_abs_corr": best_re,
        "reynolds_exponent_reference": "log Re = log f0 - 2 log nu - 3 log k_f",
    }


def reynolds_exponent_fit(physical_exprs) -> dict[str, object]:
    """Regress each discovered coordinate on (log f0, log nu).

    The Reynolds coordinate should give a slope ratio of -2 on log nu relative to
    log f0, which is a sharper test than a bare correlation.
    """
    f0_sym, nu_sym = sympy.symbols("f0 nu")
    rng = np.random.default_rng(321)
    samples = rng.uniform([F0_MIN, NU_MIN], [F0_MAX, NU_MAX], size=(4000, 2))
    f0, nu = samples[:, 0], samples[:, 1]
    design = np.stack([np.ones_like(f0), np.log(f0), np.log(nu)], axis=1)

    fits = []
    for i, expr in enumerate(physical_exprs):
        fn = sympy.lambdify((f0_sym, nu_sym), expr, modules="numpy")
        values = np.asarray(fn(f0, nu), dtype=float)
        values = np.broadcast_to(values, (samples.shape[0],))
        finite = np.isfinite(values)
        if finite.sum() < 100:
            continue
        coef, *_ = np.linalg.lstsq(design[finite], values[finite], rcond=None)
        exponent = float(coef[2] / coef[1]) if abs(coef[1]) > 1e-9 else float("nan")
        residual = float(
            np.std(values[finite] - design[finite] @ coef) / (np.std(values[finite]) + 1e-12)
        )
        fits.append(
            {
                "component": i,
                "slope_log_f0": float(coef[1]),
                "slope_log_nu": float(coef[2]),
                "nu_over_f0_exponent": exponent,
                "relative_residual": residual,
            }
        )

    valid = [f for f in fits if np.isfinite(f["nu_over_f0_exponent"])]
    best = min(
        valid, key=lambda f: abs(f["nu_over_f0_exponent"] + 2.0), default=None
    )
    return {
        "fits": fits,
        "best_exponent": best["nu_over_f0_exponent"] if best else None,
        "best_exponent_error": abs(best["nu_over_f0_exponent"] + 2.0) if best else None,
    }


def validate_flatness(aligned: dict, mdl_coords, pruned_exprs) -> dict[str, float]:
    # Dimensional analysis baseline: (log Re, log U) up to constants.
    adhoc = ["log(X1) - 2*log(X2)", "log(X1) - log(X2)"]
    adhoc_flats, _ = check_flattening(adhoc, X=aligned["X"], Fs=aligned["Fs"])
    mdl_flats, _ = check_flattening(mdl_coords, X=aligned["X"], Fs=aligned["Fs"])
    pruned_flats, _ = check_flattening(pruned_exprs, X=aligned["X"], Fs=aligned["Fs"])
    nn_flats = jax.vmap(flatten_with_numerical_jacobian)(aligned["dy_sr"], aligned["Fs"])

    identity = np.eye(aligned["n_params"])

    def fro_score(q):
        return np.linalg.norm(np.asarray(q) - identity, axis=(-2, -1))

    return {
        "raw_theta": float(np.median(fro_score(aligned["Fs"]))),
        "adhoc_dimensional_analysis": float(np.median(fro_score(adhoc_flats))),
        "mdl": float(np.median(fro_score(mdl_flats))),
        "pruned": float(np.median(fro_score(pruned_flats))),
        "nn": float(np.median(fro_score(nn_flats))),
        "median_condition_raw": float(np.median(np.linalg.cond(np.asarray(aligned["Fs"])))),
        "median_condition_symbolic": float(np.median(np.linalg.cond(np.asarray(pruned_flats)))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIGS), default="smoke")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/kolmogorov_notebook")
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
        "--grid",
        type=int,
        default=None,
        help="Override the spectral grid; raise it if the cutoff-shell warning fires.",
    )
    parser.add_argument(
        "--min-reynolds-corr",
        type=float,
        default=0.9,
        help="Fail unless some coordinate tracks log Re at least this strongly.",
    )
    parser.add_argument(
        "--max-exponent-error",
        type=float,
        default=0.4,
        help="Fail unless the fitted log-nu / log-f0 exponent is within this of -2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.mode]
    if args.sr_time_limit is not None:
        config = replace(config, sr_time_limit=args.sr_time_limit)
    if args.grid is not None:
        config = replace(config, grid=args.grid)
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = derive_seeds(args.master_seed)

    log(f"running mode={args.mode}; outdir={outdir}")
    log(f"master_seed={args.master_seed}; derived seeds={json.dumps(seeds)}")
    log(f"config={json.dumps(asdict(config), sort_keys=True)}")

    run_id = f"kolmogorov_seed{args.master_seed}"
    counts = {
        "n_train_simulations": config.nsims,
        "n_eval_simulations": config.nsims,
        "n_augmented_coordinate_evaluations": config.sr_grid_size,
        "n_pca_simulations": 0,
        "n_downstream_npe_simulations": 0,
        "grid": config.grid,
        "n_timesteps_per_simulation": config.spin_steps
        + config.n_spectra * config.spectrum_gap_steps,
        "n_spectra_averaged": config.n_spectra,
        "forcing_wavenumber": K_FORCING,
    }
    config_manifest = {
        "run_id": run_id,
        "problem": "kolmogorov",
        "master_seed": args.master_seed,
        "mode": args.mode,
        "config": asdict(config),
        "stage_seeds": seeds,
        "thresholds": {
            "min_reynolds_corr": args.min_reynolds_corr,
            "max_exponent_error": args.max_exponent_error,
        },
        "git_commit": git_commit_hash(),
    }
    with open(outdir / "config_manifest.json", "w") as handle:
        json.dump(config_manifest, handle, indent=2, sort_keys=True)

    require_gpu_if_requested(args.require_gpu)

    runtime_seconds: dict[str, float | None] = {}
    t_start = time.time()

    def write_failure(stage: str, exc: Exception) -> None:
        runtime_seconds["total"] = time.time() - t_start
        record = {
            "run_id": run_id,
            "problem": "kolmogorov",
            "master_seed": args.master_seed,
            "status": "failed",
            "failure_stage": stage,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(),
            "counts": counts,
            "runtime_seconds": runtime_seconds,
        }
        with open(outdir / "run_record.json", "w") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
        log(f"FAILED at stage={stage}: {exc}\n{traceback.format_exc()}")

    try:
        t0 = time.time()
        data = simulator_data(config, seeds["simulator"], outdir)
        runtime_seconds["simulation"] = time.time() - t0
    except Exception as exc:
        write_failure("simulation", exc)
        raise

    try:
        t0 = time.time()
        fish_dir, scaler = train_fishnet_ensemble(config, data, seeds, outdir)
        runtime_seconds["fishnets"] = time.time() - t0
    except Exception as exc:
        write_failure("fishnets", exc)
        raise

    try:
        t0 = time.time()
        _, ensemble_w, _, flatten_model = fit_flattener(config, fish_dir, seeds, outdir)
        runtime_seconds["flatten"] = time.time() - t0
    except Exception as exc:
        write_failure("flatten", exc)
        raise

    try:
        t0 = time.time()
        aligned = align_and_augment(config, seeds, outdir, ensemble_w, flatten_model)
        runtime_seconds["alignment_and_augmentation"] = time.time() - t0
    except Exception as exc:
        write_failure("alignment_and_augmentation", exc)
        raise

    try:
        t0 = time.time()
        sr_dir, mdl_coords, frob_coords, analysis = run_symbolic_regression(
            config, aligned, seeds, outdir
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
            snap_rel_tol=0.2,
            snap_flat_tol=0.2,
            decimal=2,
            threshold=0.5,
        )
        print_discovered_expressions(pruned_exprs, name_map={"X1": "f0", "X2": "nu"})

        physical_exprs = expressions_to_physical(
            pruned_exprs, scaler, sr_offset=0.0, theta_names=THETA_NAMES, decimal=3
        )
        log("physical expressions")
        for k, expr in enumerate(physical_exprs):
            print(f"  eta_{k} = {expr}", flush=True)

        correlations = physics_correlations(physical_exprs)
        exponents = reynolds_exponent_fit(physical_exprs)
        log("physical expression correlations")
        print(json.dumps(correlations, indent=2, sort_keys=True), flush=True)
        log("Reynolds exponent fits")
        print(json.dumps(exponents, indent=2, sort_keys=True), flush=True)

        flatness = validate_flatness(aligned, mdl_coords, pruned_exprs)
        log("flatness scores")
        print(json.dumps(flatness, indent=2, sort_keys=True), flush=True)

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
            # sympy.solve can also raise outright (e.g. NotImplementedError,
            # KeyError('ComplexInfinity')) on some discovered expressions, not just
            # hang -- this is a supplementary diagnostic and must never fail the
            # whole run over it.
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
        runtime_seconds["symbolic_regression"] = time.time() - t0
    except Exception as exc:
        write_failure("symbolic_regression", exc)
        raise

    # analysis["DL"] is per-component *normalized* (min-subtracted, so the winning
    # entry is always 0) -- not useful as a total. Recompute the raw DL/complexity
    # of the actual winning (mdl_coords) expressions directly via compute_DL, under
    # the same length_penalty used by analyze_equations above (2.0 here, not
    # rosenbrock's 3.0 -- this script's own SR call sets length_penalty=2.0).
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
                "invertibility": invertibility,
                "rank_info": rank_info,
                "scaler_scale": scaler.scale_,
                "scaler_min": scaler.min_,
                "scaler_data_min": scaler.data_min_,
                "scaler_data_max": scaler.data_max_,
            },
            handle,
        )
    shutil.copytree(fish_dir, sr_dir / "fishnets-kolmogorov", dirs_exist_ok=True)
    shutil.copy2(outdir / "kolmogorov_flatten.npz", sr_dir / "kolmogorov_flatten.npz")
    shutil.make_archive(str(outdir / "sr_results_kolmogorov"), "zip", root_dir=sr_dir)
    log(f"saved artifacts under {sr_dir}")

    runtime_seconds["total"] = time.time() - t_start
    corr_ok = correlations["best_reynolds_abs_corr"] >= args.min_reynolds_corr
    exp_error = exponents["best_exponent_error"]
    exp_ok = exp_error is not None and exp_error <= args.max_exponent_error
    success = bool(corr_ok and exp_ok)

    run_record = {
        "run_id": run_id,
        "problem": "kolmogorov",
        "master_seed": args.master_seed,
        "status": "success",
        "counts": counts,
        "discovery": {
            "expressions_physical": [str(e) for e in physical_exprs],
            "success": success,
            "physics_alignment": correlations["best_reynolds_abs_corr"],
            "best_exponent": exponents["best_exponent"],
            "best_exponent_error": exp_error,
            "mdl_total": mdl_total,
            "complexity_total": complexity_total,
            "symbolically_invertible": invertibility["is_symbolically_invertible"],
            "rank_deficient": bool(rank_info["rank_deficient"]),
        },
        "heldout_geometry": {
            "frob_raw": flatness["raw_theta"],
            "frob_neural": flatness["nn"],
            "frob_symbolic": flatness["pruned"],
            "frob_adhoc": flatness["adhoc_dimensional_analysis"],
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
            f"corr={correlations['best_reynolds_abs_corr']:.3f} (min {args.min_reynolds_corr:.3f}), "
            f"exponent_error={exp_error} (max {args.max_exponent_error:.3f})"
        )
    log("Kolmogorov distillery run complete")


if __name__ == "__main__":
    main()
