#!/usr/bin/env python
"""Degeneracy distillery on 2D Rayleigh-Benard convection (stress-free DNS).

This is the fourth intractable-likelihood problem, companion to the Ising,
Kolmogorov and Kuramoto scripts (see notes/neurips_intractable_examples.md).
Like Kolmogorov it is a forced/driven fluid PDE with no closed-form likelihood
over the observable; unlike Kolmogorov it is three-parameter and carries a weak
(nearly degenerate) nuisance direction, so it also answers the "does the method
work beyond two or three tightly-identifiable parameters" objection.

Physics
-------
We solve the 2D Boussinesq equations in vorticity-streamfunction + temperature
form on a box ``[0, Gamma] x [0, 1]`` (thermal-diffusion units, layer height 1),

    d_t omega + u.grad omega = Pr lap omega + Ra*Pr d_x theta
    d_t theta + u.grad theta = lap theta + w
    lap psi = -omega,   u = d_z psi,   w = -d_x psi

with stress-free, isothermal boundaries at z = 0, 1.  Under those boundary
conditions omega, theta and psi all vanish on the plates and are represented as
sine series in z (DST) and Fourier in x, which keeps the whole solver FFT/matmul
based and GPU-batchable exactly like the Kolmogorov script.  The configuration
is validated against the exact linear onset ``Ra_c = 27 pi^4 / 4 = 657.5``.

``theta = (log10 Ra, log10 Pr, log10 Gamma)``.  The aspect ratio Gamma is the
nondimensional box width, so varying it varies the horizontal wavenumber grid;
the observable is therefore binned by *physical* wavenumber so it stays a fixed
length across the prior.

Observable
----------
Per simulation: the time-averaged, radially-binned, separately-normalised
*thermal-variance* and *enstrophy* shell spectra in the statistically steady
state.  Normalising each spectrum to unit sum divides out the amplitude, so no
dimensional heat-flux number is handed to the network -- the Nusselt and
Reynolds scalings must be *recovered* from spectrum shape alone.  Nu and a
Reynolds proxy are computed alongside purely as the known-answer targets for the
recovery gate; they are never part of the network input.

Known answer / gate
-------------------
Nu(Ra, Pr) and Re(Ra, Pr) are emergent power laws (measured from the DNS itself,
not assumed).  A successful run recovers one coordinate tracking log Nu whose
(log Ra, log Pr) slope ratio matches the DNS-measured Nusselt scaling, one
coordinate tracking the Reynolds proxy, and isolates log Gamma as the weak
direction.  The Nusselt-exponent match is the sharp check, analogous to the
Reynolds ``-2`` exponent in the Kolmogorov script.

Usage
-----
    python scripts/rayleigh_benard_notebook_run.py --mode smoke --master-seed 0
    python scripts/rayleigh_benard_notebook_run.py --mode full --master-seed 3 \
        --out-dir results/rebuttal_discovery/rayleigh_benard/seed_3
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

# Prior in log10.  Ra is kept well above the stress-free onset (Ra_c = 657.5) and
# below the point where the plume boundary layers stop being resolved on the full
# grid; Pr straddles unity; Gamma spans a factor of two in aspect ratio, the weak
# direction.  These ranges are re-checkable with the resolution warning below.
LOG_RA_MIN, LOG_RA_MAX = 3.3, 4.3          # Ra in [~2000, ~20000]
LOG_PR_MIN, LOG_PR_MAX = -0.5, 0.5         # Pr in [~0.32, ~3.2]
LOG_G_MIN, LOG_G_MAX = 0.0, 0.3            # Gamma in [1.0, ~2.0]

RA_CRIT_STRESS_FREE = 27.0 * np.pi**4 / 4.0  # 657.5, used only for reporting.

THETA_NAMES = ("logRa", "logPr", "logGamma")


@dataclass(frozen=True)
class RunConfig:
    nsims: int
    nx: int
    nz: int
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
        nx=48,
        nz=32,
        spin_steps=6000,          # ~18 free-fall times: Nu ~80-90% developed
        n_spectra=12,
        spectrum_gap_steps=150,
        cfl=0.10,
        n_spectral_bins=8,
        sim_chunk=50,
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
        nx=96,
        nz=64,
        spin_steps=26000,         # ~32 free-fall times: fully developed convection
        n_spectra=28,
        spectrum_gap_steps=400,   # ~0.5 free-fall time between snapshots
        cfl=0.08,
        n_spectral_bins=16,
        sim_chunk=125,
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
# Simulator: batched pseudo-spectral stress-free Rayleigh-Benard
#
# Coefficients live in (kx, n) space: Fourier in x, sine (DST) in z, n = 1..nz.
# omega, theta, psi are pure sine series in z (they vanish on the plates).  The
# vertical basis matrices depend only on nz and are shared across the prior; only
# kx (hence the Laplacian and the shell masks) depends on Gamma, so those are
# rebuilt per simulation inside the vmap.
# --------------------------------------------------------------------------


def _vertical_basis(nz: int):
    m = np.arange(nz)
    n = np.arange(1, nz + 1)
    phi_sin = np.sin(np.pi * np.outer(m + 1, n) / (nz + 1))  # (nz, nz)
    phi_cos = np.cos(np.pi * np.outer(m + 1, n) / (nz + 1))
    kz = n * np.pi  # H = 1
    return (
        jnp.asarray(phi_sin, jnp.float32),
        jnp.asarray(phi_cos, jnp.float32),
        jnp.asarray(kz, jnp.float32),
        2.0 / (nz + 1),
    )


def _shell_edges(nz: int, n_bins: int) -> np.ndarray:
    # Physical |k| bins.  Lowest mode is ~pi (kz of n=1); the resolved top is the
    # vertical 2/3-dealias cut, the Gamma-independent ceiling.
    k_hi = (2.0 / 3.0) * np.pi * nz
    return np.linspace(np.pi * 0.75, k_hi, n_bins + 1)


def _make_single_sim(config: RunConfig, phi_sin, phi_cos, kz, inv_norm, k_edges):
    nx, nz = config.nx, config.nz
    dz = 1.0 / (nz + 1)
    dealias_z = jnp.asarray((np.arange(1, nz + 1) <= (2 * nz) // 3).astype(np.float32))
    n_bins = config.n_spectral_bins
    k_edges = jnp.asarray(k_edges, jnp.float32)

    def to_phys_sin(A, phasor):
        z = jnp.einsum("kn,mn->km", A, phi_sin)
        return jnp.real(jnp.fft.ifft(z, axis=0))

    def to_phys_cos(A):
        z = jnp.einsum("kn,mn->km", A, phi_cos)
        return jnp.real(jnp.fft.ifft(z, axis=0))

    def to_coeff_sin(f):
        fk = jnp.fft.fft(f, axis=0)
        return inv_norm * jnp.einsum("km,mn->kn", fk, phi_sin)

    def single(key, theta_row):
        log_ra, log_pr, log_g = theta_row
        Ra = 10.0**log_ra
        Pr = 10.0**log_pr
        Gamma = 10.0**log_g

        Lx = Gamma
        kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx) * (nx / Lx)  # (nx,)
        kx_c = kx[:, None]
        lap = -(kx[:, None] ** 2 + kz[None, :] ** 2)  # (nx, nz)
        inv_lap = jnp.where(lap == 0, 0.0, 1.0 / jnp.where(lap == 0, 1.0, lap))
        kmag = jnp.sqrt(kx[:, None] ** 2 + kz[None, :] ** 2)
        masks = jnp.stack(
            [
                ((kmag >= k_edges[s]) & (kmag < k_edges[s + 1])).astype(jnp.float32)
                for s in range(n_bins)
            ]
        )
        dealias = (
            (jnp.abs(kx) < (2.0 / 3.0) * (jnp.pi * nx / Lx)).astype(jnp.float32)[:, None]
            * dealias_z[None, :]
        )

        # free-fall velocity scale sets the step; floor keeps near-onset runs sane
        U = jnp.sqrt(Ra * Pr) + 1.0
        dx = Lx / nx
        dt = config.cfl * jnp.minimum(dx, dz) / U

        decay_o = Pr * (-lap)
        decay_t = -lap
        Eo1 = jnp.exp(-decay_o * dt)
        Et1 = jnp.exp(-decay_t * dt)
        Eoh = jnp.exp(-decay_o * dt / 2)
        Eth = jnp.exp(-decay_t * dt / 2)

        def rhs(o, t):
            psi = -o * inv_lap
            u = to_phys_cos(psi * kz[None, :])         # d_z psi
            w = to_phys_sin(-1j * kx_c * psi, None)    # -d_x psi
            ox = to_phys_sin(1j * kx_c * o, None)
            oz = to_phys_cos(o * kz[None, :])
            tx = to_phys_sin(1j * kx_c * t, None)
            tz = to_phys_cos(t * kz[None, :])
            adv_o = to_coeff_sin(u * ox + w * oz) * dealias
            adv_t = to_coeff_sin(u * tx + w * tz) * dealias
            d_o = -adv_o + Ra * Pr * (1j * kx_c) * t
            d_t = -adv_t + (-1j * kx_c * psi)          # + w source
            return d_o, d_t

        def step(carry, _):
            o, t = carry
            k1o, k1t = rhs(o, t)
            o2 = Eoh * (o + dt / 2 * k1o); t2 = Eth * (t + dt / 2 * k1t)
            k2o, k2t = rhs(o2, t2)
            o3 = Eoh * o + dt / 2 * k2o; t3 = Eth * t + dt / 2 * k2t
            k3o, k3t = rhs(o3, t3)
            o4 = Eo1 * o + dt * Eoh * k3o; t4 = Et1 * t + dt * Eth * k3t
            k4o, k4t = rhs(o4, t4)
            o_n = Eo1 * o + dt / 6 * (Eo1 * k1o + 2 * Eoh * k2o + 2 * Eoh * k3o + k4o)
            t_n = Et1 * t + dt / 6 * (Et1 * k1t + 2 * Eth * k2t + 2 * Eth * k3t + k4t)
            return (o_n, t_n), None

        def advance(o, t, n):
            (o, t), _ = jax.lax.scan(step, (o, t), None, length=n)
            return o, t

        # Initial condition: an O(0.3) random thermal perturbation.  A small kick
        # would spend O(20) free-fall times in the linear-growth phase before the
        # rolls saturate; a finite-amplitude one saturates promptly, so the fixed
        # spin-up budget is spent on developed convection rather than on growth.
        noise = jr.normal(key, (nx, nz), dtype=jnp.float32)
        theta0 = jnp.fft.fft(noise, axis=0).astype(jnp.complex64) * 0.3
        omega0 = jnp.zeros((nx, nz), dtype=jnp.complex64)
        omega0 = omega0 * dealias
        theta0 = theta0 * dealias

        o, t = advance(omega0, theta0, config.spin_steps)

        therm = jnp.zeros(n_bins)
        enst = jnp.zeros(n_bins)
        nu_acc = 0.0
        re_acc = 0.0
        for _ in range(config.n_spectra):
            o, t = advance(o, t, config.spectrum_gap_steps)
            therm = therm + jnp.einsum("xn,sxn->s", jnp.abs(t) ** 2, masks)
            enst = enst + jnp.einsum("xn,sxn->s", jnp.abs(o) ** 2, masks)
            # Nu and Reynolds proxy from physical-space fields
            psi = -o * inv_lap
            w = to_phys_sin(-1j * kx_c * psi, None)
            u = to_phys_cos(psi * kz[None, :])
            th = to_phys_sin(t, None)
            nu_acc = nu_acc + (1.0 + jnp.mean(w * th))
            u_rms = jnp.sqrt(jnp.mean(u**2 + w**2))
            re_acc = re_acc + u_rms / Pr
        therm = therm / (jnp.sum(therm) + 1e-30)
        enst = enst / (jnp.sum(enst) + 1e-30)
        obs = jnp.concatenate([jnp.log(therm + 1e-12), jnp.log(enst + 1e-12)])
        nu = nu_acc / config.n_spectra
        re = re_acc / config.n_spectra
        return obs, nu, re

    return single


def simulator_data(config: RunConfig, seed: int, outdir: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = 2 * config.nsims

    theta_all = np.stack(
        [
            rng.uniform(LOG_RA_MIN, LOG_RA_MAX, total),
            rng.uniform(LOG_PR_MIN, LOG_PR_MAX, total),
            rng.uniform(LOG_G_MIN, LOG_G_MAX, total),
        ],
        axis=1,
    ).astype(np.float32)

    phi_sin, phi_cos, kz, inv_norm = _vertical_basis(config.nz)
    k_edges = _shell_edges(config.nz, config.n_spectral_bins)
    single = _make_single_sim(config, phi_sin, phi_cos, kz, inv_norm, k_edges)
    batched = jax.jit(jax.vmap(single))

    log(
        f"integrating {total} RB flows on {config.nx}x{config.nz} "
        f"(Ra_c stress-free = {RA_CRIT_STRESS_FREE:.1f}); "
        f"{config.spin_steps} spin-up steps, then {config.n_spectra} spectra"
    )

    key = jr.PRNGKey(seed)
    obs_chunks, nu_chunks, re_chunks = [], [], []
    for start in range(0, total, config.sim_chunk):
        stop = min(start + config.sim_chunk, total)
        key, sub = jr.split(key)
        keys = jr.split(sub, stop - start)
        obs, nu, re = batched(keys, jnp.asarray(theta_all[start:stop]))
        obs = np.asarray(obs)
        nu = np.asarray(nu)
        re = np.asarray(re)
        if not np.isfinite(obs).all():
            raise RuntimeError(
                "non-finite RB spectra: the solver went unstable; lower cfl or "
                "narrow the top of the Ra prior"
            )
        # A pile-up in the top shell means the plume boundary layers are unresolved.
        top = np.median(obs[:, config.n_spectral_bins - 1])
        if top > -5.0:
            log(
                f"WARNING: median log occupancy of the top thermal shell is {top:.2f}; "
                f"the grid may be too coarse for this Ra range"
            )
        obs_chunks.append(obs)
        nu_chunks.append(nu)
        re_chunks.append(re)
        log(f"  simulated {stop}/{total}  (median Nu={np.median(nu):.2f})")

    data_all = np.concatenate(obs_chunks, axis=0).astype(np.float32)
    nu_all = np.concatenate(nu_chunks, axis=0).astype(np.float32)
    re_all = np.concatenate(re_chunks, axis=0).astype(np.float32)

    n = config.nsims
    out = {
        "theta_train": theta_all[:n],
        "data_train": data_all[:n],
        "theta_test": theta_all[n:],
        "data_test": data_all[n:],
        "nu_test": nu_all[n:],
        "re_test": re_all[n:],
        "nu_train": nu_all[:n],
        "re_train": re_all[:n],
    }
    log(f"theta_train {out['theta_train'].shape}; data_train {out['data_train'].shape}")
    plot_input_summary(out, config, outdir)
    return out


def measured_scaling(theta: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Regress log(values) on (log10 Ra, log10 Pr, log10 Gamma) as measured by the DNS."""
    finite = np.isfinite(values) & (values > 0)
    design = np.stack(
        [np.ones(finite.sum()), theta[finite, 0], theta[finite, 1], theta[finite, 2]],
        axis=1,
    )
    coef, *_ = np.linalg.lstsq(design, np.log10(values[finite]), rcond=None)
    return {
        "intercept": float(coef[0]),
        "slope_log_ra": float(coef[1]),
        "slope_log_pr": float(coef[2]),
        "slope_log_gamma": float(coef[3]),
    }


def plot_input_summary(data: dict, config: RunConfig, outdir: Path) -> None:
    theta = data["theta_train"]
    obs = data["data_train"]
    nu = data["nu_train"]
    nb = config.n_spectral_bins

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sc = axes[0].scatter(theta[:, 0], theta[:, 1], c=np.log10(np.maximum(nu, 1e-3)), s=10, cmap="magma")
    plt.colorbar(sc, ax=axes[0], label="log10 Nu")
    axes[0].set_xlabel("log10 Ra")
    axes[0].set_ylabel("log10 Pr")
    axes[0].set_title("Prior samples coloured by Nusselt number")

    order = np.argsort(theta[:, 0])
    for idx in order[:: max(1, len(order) // 25)]:
        c = plt.cm.magma(
            (theta[idx, 0] - LOG_RA_MIN) / max(LOG_RA_MAX - LOG_RA_MIN, 1e-9)
        )
        axes[1].plot(np.arange(nb), obs[idx, :nb], color=c, alpha=0.7, lw=1)
    axes[1].set_xlabel("physical-k shell")
    axes[1].set_ylabel("log normalised thermal variance")
    axes[1].set_title("Observable: thermal-variance spectra")

    axes[2].scatter(theta[:, 0], np.log10(np.maximum(nu, 1e-3)), s=8, alpha=0.5)
    axes[2].set_xlabel("log10 Ra")
    axes[2].set_ylabel("log10 Nu")
    axes[2].set_title("Nusselt-Rayleigh scaling (DNS)")

    fig.tight_layout()
    fig.savefig(outdir / "rayleigh_benard_input_summary.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------
# Distillery stages (structurally identical to the Kolmogorov script)
# --------------------------------------------------------------------------


def train_fishnet_ensemble(config: RunConfig, data, seeds, outdir: Path):
    # [1, 2] keeps every scaled parameter strictly positive and O(1), so SR can
    # form ratios and logs cleanly; the scaled theta passes through the flattener
    # and alignment unchanged, so expressions_to_physical uses sr_offset=0.
    scaler = fit_theta_scaler(data["theta_train"], feature_range=(1.0, 2.0))
    theta_train_s = scaler.transform(data["theta_train"]).astype(np.float32)
    theta_test_s = scaler.transform(data["theta_test"]).astype(np.float32)
    log(f"scaled theta range: {theta_train_s.min(0)} to {theta_train_s.max(0)}")

    embedding_net = nn.Sequential(
        [nn.Dense(128), nn.gelu, nn.Dense(64), nn.gelu, nn.Dense(64), nn.gelu]
    )

    fish_dir = outdir / "fishnets-rayleigh_benard"
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
            output_prefix="rayleigh_benard_flatten",
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
        filename="rayleigh_benard_flatten.npz",
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
    sr_dir = outdir / "sr_results_rayleigh_benard"
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
        length_penalty=2.0,
        equation_predicate=predicate,
    )
    return sr_dir, mdl_coords, frob_coords, analysis


def physics_correlations(physical_exprs, data: dict) -> dict[str, object]:
    """Correlate each discovered coordinate with the DNS-measured Nu, Re and Gamma.

    Unlike Kolmogorov (where Re is analytic), Nu and Re are emergent here, so we
    evaluate the coordinates at the *test* parameter points and correlate against
    the simulated Nu_test / Re_test.
    """
    logRa, logPr, logG = sympy.symbols("logRa logPr logGamma")
    theta = data["theta_test"]
    nu = np.asarray(data["nu_test"], dtype=float)
    re = np.asarray(data["re_test"], dtype=float)
    ok = np.isfinite(nu) & (nu > 0) & np.isfinite(re) & (re > 0)
    theta, nu, re = theta[ok], nu[ok], re[ok]

    targets = {
        "log_Nu": np.log(nu),
        "log_Re": np.log(re),
        "log_Ra": np.log(10.0) * theta[:, 0],
        "log_Pr": np.log(10.0) * theta[:, 1],
        "log_Gamma": np.log(10.0) * theta[:, 2],
    }

    rows = []
    for i, expr in enumerate(physical_exprs):
        fn = sympy.lambdify((logRa, logPr, logG), expr, modules="numpy")
        vals = np.asarray(
            fn(theta[:, 0], theta[:, 1], theta[:, 2]), dtype=float
        )
        vals = np.broadcast_to(vals, (theta.shape[0],))
        finite = np.isfinite(vals)
        row = {"component": i, "expr": str(expr)}
        for name, target in targets.items():
            if finite.sum() < 50 or np.std(vals[finite]) == 0:
                row[name] = 0.0
            else:
                row[name] = float(np.corrcoef(vals[finite], target[finite])[0, 1])
        rows.append(row)

    return {
        "rows": rows,
        "best_nusselt_abs_corr": max(abs(r["log_Nu"]) for r in rows),
        "best_reynolds_abs_corr": max(abs(r["log_Re"]) for r in rows),
        "best_gamma_abs_corr": max(abs(r["log_Gamma"]) for r in rows),
    }


def nusselt_exponent_fit(physical_exprs, data: dict) -> dict[str, object]:
    """Sharp check: the Nu-tracking coordinate must point the same way in
    parameter space as the DNS Nusselt scaling.  A discovered coordinate is only
    defined up to scale, so we compare the *direction* of its (log Ra, log Pr,
    log Gamma) gradient with the DNS-measured Nu gradient via cosine similarity --
    scale-invariant, and robust when the Pr-dependence of Nu is weak (which makes
    a bare slope ratio ill-conditioned).  This is the RB analogue of the Reynolds
    ``-2`` exponent test in the Kolmogorov script."""
    logRa, logPr, logG = sympy.symbols("logRa logPr logGamma")
    theta = data["theta_test"]
    nu = np.asarray(data["nu_test"], dtype=float)
    ok = np.isfinite(nu) & (nu > 0)
    theta, nu = theta[ok], nu[ok]

    nu_scaling = measured_scaling(theta, nu)
    # Cosine is taken over the *identifiable* (log Ra, log Pr) subspace only.  The
    # observable is nearly blind to aspect ratio (Gamma is the flat direction), so
    # the true Nu(Ra, Pr, Gamma) gradient's Gamma component -- non-zero here from a
    # finite-size effect on the roll count -- cannot be recovered from spectra and
    # would wrongly sink a full 3D comparison.  We check the thermodynamic scaling
    # and report the Gamma dependence separately.
    ref_grad = np.array([nu_scaling["slope_log_ra"], nu_scaling["slope_log_pr"]])
    ref_grad = ref_grad / (np.linalg.norm(ref_grad) + 1e-12)

    design = np.stack(
        [np.ones(theta.shape[0]), theta[:, 0], theta[:, 1], theta[:, 2]], axis=1
    )
    fits = []
    for i, expr in enumerate(physical_exprs):
        fn = sympy.lambdify((logRa, logPr, logG), expr, modules="numpy")
        vals = np.asarray(fn(theta[:, 0], theta[:, 1], theta[:, 2]), dtype=float)
        vals = np.broadcast_to(vals, (theta.shape[0],))
        finite = np.isfinite(vals)
        if finite.sum() < 50 or np.std(vals[finite]) == 0:
            continue
        coef, *_ = np.linalg.lstsq(design[finite], vals[finite], rcond=None)
        grad2 = coef[1:3]
        gnorm = np.linalg.norm(grad2)
        cosine = float(np.dot(grad2, ref_grad) / (gnorm + 1e-12)) if gnorm > 1e-12 else 0.0
        fits.append(
            {
                "component": i,
                "slope_log_ra": float(coef[1]),
                "slope_log_pr": float(coef[2]),
                "slope_log_gamma": float(coef[3]),
                "nusselt_rapr_cosine": abs(cosine),
            }
        )

    best = max(fits, key=lambda f: f["nusselt_rapr_cosine"], default=None)
    return {
        "dns_nusselt_scaling": nu_scaling,
        "dns_nusselt_rapr_gradient_unit": ref_grad.tolist(),
        "fits": fits,
        "best_nusselt_rapr_cosine": best["nusselt_rapr_cosine"] if best else None,
    }


def validate_flatness(aligned: dict, mdl_coords, pruned_exprs) -> dict[str, float]:
    # Ad-hoc baseline: the bare scaled parameters, i.e. no nondimensional grouping.
    adhoc = ["X1", "X2", "X3"]
    adhoc_flats, _ = check_flattening(adhoc, X=aligned["X"], Fs=aligned["Fs"])
    mdl_flats, _ = check_flattening(mdl_coords, X=aligned["X"], Fs=aligned["Fs"])
    pruned_flats, _ = check_flattening(pruned_exprs, X=aligned["X"], Fs=aligned["Fs"])
    nn_flats = jax.vmap(flatten_with_numerical_jacobian)(aligned["dy_sr"], aligned["Fs"])

    identity = np.eye(aligned["n_params"])

    def fro_score(q):
        return np.linalg.norm(np.asarray(q) - identity, axis=(-2, -1))

    return {
        "raw_theta": float(np.median(fro_score(aligned["Fs"]))),
        "adhoc_raw_parameters": float(np.median(fro_score(adhoc_flats))),
        "mdl": float(np.median(fro_score(mdl_flats))),
        "pruned": float(np.median(fro_score(pruned_flats))),
        "nn": float(np.median(fro_score(nn_flats))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIGS), default="smoke")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/rayleigh_benard_notebook")
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
        "--nx", type=int, default=None, help="Override horizontal grid points."
    )
    parser.add_argument(
        "--nz", type=int, default=None, help="Override vertical grid points."
    )
    parser.add_argument(
        "--cfl", type=float, default=None, help="Override the CFL factor (lower if the solver goes unstable)."
    )
    parser.add_argument(
        "--min-nusselt-corr",
        type=float,
        default=0.9,
        help="Fail unless some coordinate tracks log Nu at least this strongly.",
    )
    parser.add_argument(
        "--min-nusselt-cosine",
        type=float,
        default=0.9,
        help="Fail unless the Nu-coordinate's parameter gradient aligns with the "
        "DNS-measured Nusselt scaling gradient at least this well (cosine).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.mode]
    if args.sr_time_limit is not None:
        config = replace(config, sr_time_limit=args.sr_time_limit)
    if args.nx is not None:
        config = replace(config, nx=args.nx)
    if args.nz is not None:
        config = replace(config, nz=args.nz)
    if args.cfl is not None:
        config = replace(config, cfl=args.cfl)
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
        pruned_exprs, name_map={"X1": "logRa", "X2": "logPr", "X3": "logGamma"}
    )

    physical_exprs = expressions_to_physical(
        pruned_exprs, scaler, sr_offset=0.0, theta_names=THETA_NAMES, decimal=3
    )
    log("physical expressions")
    for k, expr in enumerate(physical_exprs):
        print(f"  eta_{k} = {expr}", flush=True)

    correlations = physics_correlations(physical_exprs, data)
    exponents = nusselt_exponent_fit(physical_exprs, data)
    log("physical expression correlations")
    print(json.dumps(correlations, indent=2, sort_keys=True), flush=True)
    log("Nusselt exponent fits")
    print(json.dumps(exponents, indent=2, sort_keys=True), flush=True)

    flatness = validate_flatness(aligned, mdl_coords, pruned_exprs)
    log("flatness scores")
    print(json.dumps(flatness, indent=2, sort_keys=True), flush=True)

    timings["total"] = time.time() - t_start
    corr_ok = correlations["best_nusselt_abs_corr"] >= args.min_nusselt_corr
    cosine = exponents["best_nusselt_rapr_cosine"]
    cosine_ok = cosine is not None and cosine >= args.min_nusselt_cosine
    success = bool(corr_ok and cosine_ok)

    summary = {
        "run_id": f"rayleigh_benard_seed{args.master_seed}",
        "problem": "rayleigh_benard_convection",
        "master_seed": args.master_seed,
        "mode": args.mode,
        "status": "success" if success else "criterion_not_met",
        "seeds": seeds,
        "counts": {
            "n_train_simulations": config.nsims,
            "n_eval_simulations": config.nsims,
            "n_augmented_coordinate_evaluations": config.sr_grid_size,
            "grid": [config.nx, config.nz],
            "n_timesteps_per_simulation": config.spin_steps
            + config.n_spectra * config.spectrum_gap_steps,
            "n_spectra_averaged": config.n_spectra,
            "ra_crit_stress_free": RA_CRIT_STRESS_FREE,
        },
        "discovery": {
            "expressions_physical": [str(e) for e in physical_exprs],
            "success": success,
            "best_nusselt_abs_corr": correlations["best_nusselt_abs_corr"],
            "best_reynolds_abs_corr": correlations["best_reynolds_abs_corr"],
            "best_gamma_abs_corr": correlations["best_gamma_abs_corr"],
            "dns_nusselt_scaling": exponents["dns_nusselt_scaling"],
            "best_nusselt_rapr_cosine": cosine,
        },
        "heldout_geometry": {
            **flatness,
            # Surfaced here (not just in prune_info inside sr_expressions.pkl) so a
            # rejected rotation is visible in aggregated results. rel_delta is
            # signed: negative means the rotation improved flatness.
            "rotation_accepted": bool(prune_info["rotation_accepted"]),
            "rotation_rel_delta": float(prune_info["rel_delta"]),
        },
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
    shutil.copytree(fish_dir, sr_dir / "fishnets-rayleigh_benard", dirs_exist_ok=True)
    shutil.copy2(
        outdir / "rayleigh_benard_flatten.npz", sr_dir / "rayleigh_benard_flatten.npz"
    )
    shutil.make_archive(str(outdir / "sr_results_rayleigh_benard"), "zip", root_dir=sr_dir)
    log(f"saved artifacts under {sr_dir}")

    if not success:
        raise SystemExit(
            "Did not recover the Nusselt coordinate: "
            f"corr={correlations['best_nusselt_abs_corr']:.3f}, gradient_cosine={cosine}"
        )
    log("Rayleigh-Benard distillery run complete")


if __name__ == "__main__":
    main()
