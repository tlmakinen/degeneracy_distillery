#!/usr/bin/env python
"""Degeneracy distillery on mass-action enzyme kinetics.

The full reaction scheme

    E + S  <->(k_f, k_r)  ES  ->(k_cat)  E + P

is integrated as mass action with ``theta = (k_f, k_r, k_cat)`` and a known
enzyme concentration ``E_0``.  The observable is what an enzyme assay records:
fractional product conversion ``P(t)/S_0`` on a fixed log-spaced time grid, for
three initial substrate concentrations that bracket the Michaelis constant,
corrupted by additive assay noise.

Under the quasi-steady-state approximation (valid here because ``E_0`` is two
orders of magnitude below ``S_0``) the progress curves depend on the rate
constants only through

    V_max = k_cat * E_0        (with E_0 known, this measures k_cat)
    K_M   = (k_r + k_cat)/k_f

so the three-dimensional parameter space carries a one-dimensional degeneracy:
moving ``k_f`` and ``k_r`` together at fixed ``(k_r + k_cat)/k_f`` leaves every
progress curve unchanged.  This is the same rank-2-in-3D structure as the Ising
experiment.

WHY THIS EXPERIMENT EXISTS.  Every other degeneracy in the paper -- J/T, the
Reynolds number, K/sigma, the chirp mass, S8 -- is a product of powers, and can
therefore be recovered by fitting log-log exponents.  ``K_M`` cannot: it is a
sum inside a quotient.  The best power law in (k_f, k_r, k_cat) misfits it by
about 10 percent in the median and by up to 70 percent in the tail.  Recovering
``K_M`` therefore demonstrates that the symbolic-regression stage finds
structure that dimensional analysis and exponent fitting provably cannot, which
is the sharpest available answer to the objection that the method is
dimensional analysis with a neural network attached.

The run reports, as its headline metric, how well the discovered coordinate
predicts log K_M compared with the best possible power law.

Usage
-----
    python scripts/enzyme_notebook_run.py --mode smoke --master-seed 0
    python scripts/enzyme_notebook_run.py --mode full --master-seed 3 \
        --out-dir results/rebuttal_discovery/enzyme/seed_3
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

# Log-uniform priors on the rate constants.  k_r and k_cat span comparable
# decades so that neither dominates (k_r + k_cat): that is what makes the sum
# badly approximated by a power law, which is the point of the experiment.
KF_MIN, KF_MAX = 0.5, 3.0
KR_MIN, KR_MAX = 0.3, 6.0
KCAT_MIN, KCAT_MAX = 0.5, 4.0

# Known enzyme concentration.  Two orders of magnitude below the smallest S_0,
# which is what makes the quasi-steady-state reduction exact to well below the
# assay noise.
E0_FIXED = 0.02

# Substrate concentrations bracket the K_M prior, which is how a real assay
# separates V_max from K_M rather than measuring only their ratio.
S0_VALUES = (1.0, 3.0, 9.0)
N_TIMES = 16
T_FIRST, T_LAST = 2.0, 600.0

# Assay noise on fractional conversion.  Probed: this places the two stiff
# directions at SNR 72 and 17 and the null direction at SNR 0.09, so the
# degeneracy is genuinely below the noise floor rather than merely small.
SIGMA_OBS = 0.02

THETA_NAMES = ("k_f", "k_r", "k_cat")


@dataclass(frozen=True)
class RunConfig:
    nsims: int
    dt: float
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
    # Condition number the Fisher is floored to before flattening. The other
    # experiments in the paper hand the flattener a Fisher conditioned between
    # 5 and 50 and it works; this problem's is order 1e5 untreated.
    fisher_ridge_target_cond: float


CONFIGS = {
    "smoke": RunConfig(
        nsims=150,
        dt=0.02,
        sim_chunk=150,
        num_fishnets=4,
        fish_epochs=1500,
        fish_hids_min=32,
        fish_hids_max=96,
        fish_patience=25,
        fish_layers=(2, 3),
        fish_batch_size=50,
        flatten_batch_size=75,
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
        fisher_ridge_target_cond=40.0,
    ),
    "full": RunConfig(
        nsims=500,
        dt=0.02,
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
        sr_max_length=30,
        sr_max_depth=16,
        fisher_ridge_target_cond=40.0,
    ),
}

OBS_TIMES = np.geomspace(T_FIRST, T_LAST, N_TIMES)


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
# Target coordinates
# --------------------------------------------------------------------------


def michaelis_constant(kf, kr, kcat):
    return (kr + kcat) / kf


def vmax(kcat):
    return kcat * E0_FIXED


# --------------------------------------------------------------------------
# Simulator: batched RK4 on the mass-action system
# --------------------------------------------------------------------------


def _rhs(state, kf, kr, kcat):
    """Mass action with the conservation law E = E_0 - ES applied."""
    s, es = state[..., 0], state[..., 1]
    binding = kf * (E0_FIXED - es) * s
    return jnp.stack([-binding + kr * es, binding - (kr + kcat) * es], axis=-1)


def _integrate(theta, s0, dt):
    """Fractional conversion at each observation time, for one S_0."""
    kf, kr, kcat = theta[:, 0], theta[:, 1], theta[:, 2]
    state = jnp.stack([s0, jnp.zeros_like(s0)], axis=-1)

    def step(state, _):
        k1 = _rhs(state, kf, kr, kcat)
        k2 = _rhs(state + 0.5 * dt * k1, kf, kr, kcat)
        k3 = _rhs(state + 0.5 * dt * k2, kf, kr, kcat)
        k4 = _rhs(state + dt * k3, kf, kr, kcat)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None

    outputs, t_now = [], 0.0
    for t_target in OBS_TIMES:
        n_steps = int(round((t_target - t_now) / dt))
        state, _ = jax.lax.scan(step, state, None, length=n_steps)
        t_now += n_steps * dt
        outputs.append((s0 - state[..., 0] - state[..., 1]) / s0)
    return jnp.stack(outputs, axis=-1)


def _simulate_chunk(key, theta_chunk, dt):
    curves = jnp.concatenate(
        [
            _integrate(theta_chunk, jnp.full(theta_chunk.shape[0], s0), dt)
            for s0 in S0_VALUES
        ],
        axis=-1,
    )
    return curves + SIGMA_OBS * jr.normal(key, curves.shape)


def sample_prior(rng: np.random.Generator, n: int) -> np.ndarray:
    def logu(lo, hi):
        return np.exp(rng.uniform(np.log(lo), np.log(hi), n))

    return np.stack(
        [logu(KF_MIN, KF_MAX), logu(KR_MIN, KR_MAX), logu(KCAT_MIN, KCAT_MAX)],
        axis=1,
    ).astype(np.float32)


def simulator_data(config: RunConfig, seed: int, outdir: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = 2 * config.nsims
    theta_all = sample_prior(rng, total)

    log(f"integrating {total} progress-curve sets at S_0 = {list(S0_VALUES)}")
    key = jr.PRNGKey(seed)
    chunks = []
    for start in range(0, total, config.sim_chunk):
        stop = min(start + config.sim_chunk, total)
        key, sub = jr.split(key)
        block = np.asarray(
            _simulate_chunk(sub, jnp.asarray(theta_all[start:stop]), config.dt)
        )
        if not np.isfinite(block).all():
            raise RuntimeError("non-finite progress curves; reduce dt")
        chunks.append(block)
        log(f"  simulated {stop}/{total}")
    data_all = np.concatenate(chunks, axis=0).astype(np.float32)

    # Curves that never leave the noise floor carry no information; warn rather
    # than fail, since a handful is tolerable but a large fraction is not.
    finals = data_all[:, N_TIMES - 1 :: N_TIMES]
    dead = float((finals < 5.0 * SIGMA_OBS).mean())
    if dead > 0.05:
        log(f"WARNING: {dead:.1%} of progress curves stay within 5 sigma of zero")

    theta_train, data_train = theta_all[: config.nsims], data_all[: config.nsims]
    theta_test, data_test = theta_all[config.nsims :], data_all[config.nsims :]
    log(f"theta_train {theta_train.shape}; data_train {data_train.shape}")

    plot_input_summary(theta_train, data_train, outdir)
    return {
        "theta_train": theta_train,
        "data_train": data_train,
        "theta_test": theta_test,
        "data_test": data_test,
    }


def plot_input_summary(theta, data, outdir: Path) -> None:
    km = michaelis_constant(theta[:, 0], theta[:, 1], theta[:, 2])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sc = axes[0].scatter(theta[:, 0], theta[:, 1], c=np.log10(km), s=8, cmap="viridis")
    plt.colorbar(sc, ax=axes[0], label="log10 K_M")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("k_f")
    axes[0].set_ylabel("k_r")
    axes[0].set_title("Prior samples coloured by K_M")

    order = np.argsort(km)
    span = np.ptp(np.log10(km))
    for idx in order[:: max(1, len(order) // 30)]:
        colour = plt.cm.viridis((np.log10(km[idx]) - np.log10(km).min()) / span)
        axes[1].plot(OBS_TIMES, data[idx, :N_TIMES], color=colour, alpha=0.7, lw=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("fractional conversion")
    axes[1].set_title(f"Observable: progress curves at S_0 = {S0_VALUES[0]}")

    # The half-conversion time is the classic readout, and it should collapse
    # onto (V_max, K_M) rather than onto any individual rate.
    half = np.array(
        [np.interp(0.5, data[i, :N_TIMES], OBS_TIMES) for i in range(len(data))]
    )
    axes[2].scatter(km, half, c=np.log10(theta[:, 2]), s=8, cmap="plasma")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("K_M")
    axes[2].set_ylabel("time to 50% conversion")
    axes[2].set_title("Half-time vs K_M (colour: log k_cat)")

    fig.tight_layout()
    fig.savefig(outdir / "enzyme_input_summary.png", dpi=180)
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

    fish_dir = outdir / "fishnets-enzyme"
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


def fisher_spectrum_report(fish_dir: Path) -> dict[str, object]:
    """Screening diagnostic: how many directions does the data actually constrain?

    Reports the eigenspectrum of the prior-averaged Fisher, which locates the
    degenerate directions, alongside the spread of the Fisher across the prior,
    which is what decides whether a linear reparameterisation would suffice.
    The two answer different questions and the run records both.
    """
    with np.load(fish_dir / "fishnets_outputs.npz") as fish:
        fs = np.asarray(fish["Fs"])

    finite = np.isfinite(fs).all(axis=(1, 2, 3))
    fs = fs[finite]
    per_sample = fs.mean(axis=0)  # average over ensemble members

    mean_fisher = per_sample.mean(axis=0)
    evals = np.linalg.eigvalsh(mean_fisher)[::-1]
    ratios = evals / max(evals[0], 1e-30)

    # Spread of the metric across the prior, relative to its mean scale.
    frob_mean = np.linalg.norm(mean_fisher)
    spread = float(np.linalg.norm(per_sample.std(axis=0)) / max(frob_mean, 1e-30))

    gaps = evals[:-1] / np.maximum(evals[1:], 1e-30)
    n_stiff = int(np.argmax(gaps) + 1) if len(gaps) else len(evals)

    return {
        "mean_fisher_eigenvalues": [float(v) for v in evals],
        "eigenvalue_ratios_to_largest": [float(v) for v in ratios],
        "largest_spectral_gap_after_index": n_stiff,
        "gap_size": float(gaps.max()) if len(gaps) else float("nan"),
        "relative_fisher_spread_across_prior": spread,
        "expected_n_stiff": 2,
        "screening_agrees_with_theory": bool(n_stiff == 2),
    }


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


def apply_fisher_ridge(ensemble, f_avg, target_cond: float):
    """Floor the Fisher spectrum before flattening. This is the screening step.

    The flattening objective contains a ``||Q^-1 - I||`` term, so a direction
    the data cannot see contributes an unbounded amount to the loss and the
    network would need an unbounded Jacobian stretch to satisfy it, which the
    invertibility penalty forbids. On this problem the true Fisher condition
    number is order 1e5 and the objective is dominated entirely by the null
    direction; the flattener then makes no progress on the two directions that
    actually carry information.

    Adding a constant ``lambda * I`` is exactly a Gaussian prior on the scaled
    parameters, so the flattened coordinates whiten the *posterior* rather than
    the likelihood. Directions well above ``lambda`` are untouched; the
    degenerate one is pinned at the prior scale instead of running off to zero.
    ``lambda`` is global rather than per-sample so that it acts as a genuine
    prior and does not distort how the metric varies across parameter space.

    Set ``target_cond`` to 0 or below to disable.
    """
    evals = np.linalg.eigvalsh(np.asarray(f_avg))
    top = float(np.median(evals[:, -1]))
    before = float(np.median(evals[:, -1] / np.maximum(evals[:, 0], 1e-30)))

    if target_cond is None or target_cond <= 0:
        return ensemble, f_avg, {"applied": False, "condition_before": before}

    lam = top / float(target_cond)
    eye = jnp.eye(f_avg.shape[-1])
    f_avg_r = f_avg + lam * eye
    ensemble_r = ensemble + lam * eye

    ev_after = np.linalg.eigvalsh(np.asarray(f_avg_r))
    after = float(np.median(ev_after[:, -1] / np.maximum(ev_after[:, 0], 1e-30)))
    return (
        ensemble_r,
        f_avg_r,
        {
            "applied": True,
            "target_condition": float(target_cond),
            "median_top_eigenvalue": top,
            "lambda": float(lam),
            "condition_before": before,
            "condition_after": after,
        },
    )


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

    ensemble, f_avg, ridge_info = apply_fisher_ridge(
        ensemble, f_avg, config.fisher_ridge_target_cond
    )
    log("Fisher ridge (screening step)")
    print(json.dumps(ridge_info, indent=2, sort_keys=True), flush=True)

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
            output_prefix="enzyme_flatten",
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
    return w, ensemble_w, outputs, flatten_model, ridge_info


def align_and_augment(config: RunConfig, seeds, outdir: Path, ensemble_w, flatten_model):
    log("aligning coordinates")
    aligned = load_and_process_data_v2(
        datapath=str(outdir) + os.sep,
        filename="enzyme_flatten.npz",
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
    sr_dir = outdir / "sr_results_enzyme"
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
        allowed_symbols="add,sub,mul,div,pow,constant,variable,square,sqrt,logabs",
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


# --------------------------------------------------------------------------
# Physics checks
# --------------------------------------------------------------------------


def _evaluate(physical_exprs, samples):
    kf_s, kr_s, kcat_s = sympy.symbols("k_f k_r k_cat")
    kf, kr, kcat = samples[:, 0], samples[:, 1], samples[:, 2]
    values = []
    for expr in physical_exprs:
        fn = sympy.lambdify((kf_s, kr_s, kcat_s), expr, modules="numpy")
        out = np.asarray(fn(kf, kr, kcat), dtype=float)
        values.append(np.broadcast_to(out, (samples.shape[0],)))
    return values


def physics_correlations(physical_exprs) -> dict[str, object]:
    rng = np.random.default_rng(123)
    samples = sample_prior(rng, 5000).astype(np.float64)
    kf, kr, kcat = samples[:, 0], samples[:, 1], samples[:, 2]
    km = michaelis_constant(kf, kr, kcat)

    targets = {
        "log_K_M": np.log(km),
        "K_M": km,
        "log_k_cat": np.log(kcat),
        "log_V_max": np.log(vmax(kcat)),
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

    best_km = max(max(abs(r["log_K_M"]), abs(r["K_M"])) for r in rows)
    best_kcat = max(abs(r["log_k_cat"]) for r in rows)
    return {
        "rows": rows,
        "best_michaelis_abs_corr": best_km,
        "best_kcat_abs_corr": best_kcat,
        "worst_of_best": min(best_km, best_kcat),
    }


def power_law_comparison(physical_exprs) -> dict[str, object]:
    """The headline metric.

    K_M = (k_r + k_cat)/k_f is a sum inside a quotient, so no power law in the
    rate constants can express it. Fit the best power law, then ask how much
    better the discovered coordinate does. A large gap is the evidence that
    symbolic regression found structure exponent fitting cannot.
    """
    rng = np.random.default_rng(321)
    samples = sample_prior(rng, 8000).astype(np.float64)
    kf, kr, kcat = samples[:, 0], samples[:, 1], samples[:, 2]
    log_km = np.log(michaelis_constant(kf, kr, kcat))

    # Best power law in the three rate constants.
    design = np.stack([np.ones_like(kf), np.log(kf), np.log(kr), np.log(kcat)], axis=1)
    coef, *_ = np.linalg.lstsq(design, log_km, rcond=None)
    pl_resid = log_km - design @ coef
    pl_r2 = float(1.0 - np.var(pl_resid) / np.var(log_km))
    pl_frac = np.abs(np.exp(pl_resid) - 1.0)

    # The same question for the sum alone, which isolates the non-power-law
    # structure from the exactly-power-law 1/k_f prefactor.
    log_sum = np.log(kr + kcat)
    design_sum = np.stack([np.ones_like(kr), np.log(kr), np.log(kcat)], axis=1)
    coef_sum, *_ = np.linalg.lstsq(design_sum, log_sum, rcond=None)
    sum_resid = log_sum - design_sum @ coef_sum
    sum_r2 = float(1.0 - np.var(sum_resid) / np.var(log_sum))

    # How well does each discovered coordinate predict log K_M? The coordinate
    # is determined only up to a smooth monotone reparameterisation, so allow a
    # low-order polynomial response before scoring.
    best = {"component": None, "r2": -np.inf, "expr": None}
    per_component = []
    for i, values in enumerate(_evaluate(physical_exprs, samples)):
        finite = np.isfinite(values)
        if finite.sum() < 500:
            continue
        v = values[finite]
        v = (v - v.mean()) / (v.std() + 1e-12)
        basis = np.stack([np.ones_like(v), v, v**2, v**3], axis=1)
        c, *_ = np.linalg.lstsq(basis, log_km[finite], rcond=None)
        r2 = float(
            1.0 - np.var(log_km[finite] - basis @ c) / np.var(log_km[finite])
        )
        per_component.append({"component": i, "r2_vs_log_K_M": r2})
        if r2 > best["r2"]:
            best = {"component": i, "r2": r2, "expr": str(physical_exprs[i])}

    return {
        "power_law_exponents_kf_kr_kcat": [float(v) for v in coef[1:]],
        "power_law_r2_for_log_K_M": pl_r2,
        "power_law_median_frac_error": float(np.median(pl_frac)),
        "power_law_p90_frac_error": float(np.percentile(pl_frac, 90)),
        "power_law_worst_frac_error": float(pl_frac.max()),
        "power_law_r2_for_log_sum_only": sum_r2,
        "per_component_r2_vs_log_K_M": per_component,
        "best_component": best["component"],
        "best_expr": best["expr"],
        "best_r2_vs_log_K_M": float(best["r2"]) if np.isfinite(best["r2"]) else 0.0,
        "improvement_over_power_law": (
            float(best["r2"] - pl_r2) if np.isfinite(best["r2"]) else 0.0
        ),
    }


def kcat_exponent_fit(physical_exprs) -> dict[str, object]:
    """The k_cat coordinate IS a power law, with exponent vector (0, 0, 1).

    Reporting it alongside the K_M result shows the method handles both regimes:
    it recovers the power-law coordinate exactly and the non-power-law one too.
    """
    rng = np.random.default_rng(654)
    samples = sample_prior(rng, 4000).astype(np.float64)
    design = np.concatenate(
        [np.ones((samples.shape[0], 1)), np.log(samples)], axis=1
    )
    target = np.array([0.0, 0.0, 1.0])

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
        cosine = float(abs(exponents / norm @ target))
        fits.append(
            {
                "component": i,
                "exponents_log_kf_kr_kcat": [float(v) for v in exponents],
                "cosine_to_kcat_direction": cosine,
                "relative_residual": float(
                    np.std(values[finite] - design[finite] @ coef)
                    / (np.std(values[finite]) + 1e-12)
                ),
            }
        )

    return {
        "fits": fits,
        "best_cosine_to_kcat": max((f["cosine_to_kcat_direction"] for f in fits), default=0.0),
    }


def validate_flatness(aligned: dict, mdl_coords, pruned_exprs) -> dict[str, float]:
    # Textbook Michaelis-Menten reduction: k_cat, K_M, and k_f for the
    # remaining unconstrained direction.
    adhoc = ["X3", "(X2 + X3) / X1", "X1"]
    adhoc_flats, _ = check_flattening(adhoc, X=aligned["X"], Fs=aligned["Fs"])
    mdl_flats, _ = check_flattening(mdl_coords, X=aligned["X"], Fs=aligned["Fs"])
    pruned_flats, _ = check_flattening(pruned_exprs, X=aligned["X"], Fs=aligned["Fs"])
    nn_flats = jax.vmap(flatten_with_numerical_jacobian)(aligned["dy_sr"], aligned["Fs"])

    identity = np.eye(aligned["n_params"])

    def fro_score(q):
        return np.linalg.norm(np.asarray(q) - identity, axis=(-2, -1))

    return {
        "raw_theta": float(np.median(fro_score(aligned["Fs"]))),
        "adhoc_michaelis_menten": float(np.median(fro_score(adhoc_flats))),
        "mdl": float(np.median(fro_score(mdl_flats))),
        "pruned": float(np.median(fro_score(pruned_flats))),
        "nn": float(np.median(fro_score(nn_flats))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIGS), default="smoke")
    parser.add_argument("--out-dir", type=Path, default=Path("results/enzyme_notebook"))
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument(
        "--require-gpu", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--sr-time-limit",
        type=int,
        default=None,
        help="Override the symbolic-regression budget per component, in seconds.",
    )
    parser.add_argument(
        "--nsims",
        type=int,
        default=None,
        help=(
            "Override the training simulation count. Below roughly 500 the "
            "fishnets do not resolve this degeneracy: at 150 the recovered "
            "Fisher condition number is 28 against a true 6.9e5, and the null "
            "direction is off by 14 degrees. At 500 it is 364 and 1.2 degrees."
        ),
    )
    parser.add_argument(
        "--num-fishnets",
        type=int,
        default=None,
        help="Override the fishnet ensemble size.",
    )
    parser.add_argument(
        "--fisher-ridge-target-cond",
        type=float,
        default=None,
        help=(
            "Condition number to floor the Fisher spectrum to before "
            "flattening. Pass 0 to disable and flatten the raw likelihood "
            "Fisher, which on this problem is conditioned at order 1e5."
        ),
    )
    parser.add_argument(
        "--min-michaelis-corr",
        type=float,
        default=0.9,
        help="Fail unless some coordinate tracks K_M at least this strongly.",
    )
    parser.add_argument(
        "--min-power-law-gain",
        type=float,
        default=0.02,
        help=(
            "Fail unless the discovered coordinate beats the best power law for "
            "log K_M by at least this much R^2. This is the headline claim."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.mode]
    if args.sr_time_limit is not None:
        config = replace(config, sr_time_limit=args.sr_time_limit)
    if args.nsims is not None:
        config = replace(
            config,
            nsims=args.nsims,
            sim_chunk=min(config.sim_chunk, args.nsims),
            # flatten_batch_size larger than the sample count silently drops
            # every sample inside fit_flattening's robust norm estimate.
            flatten_batch_size=min(config.flatten_batch_size, max(1, args.nsims // 2)),
        )
    if args.num_fishnets is not None:
        config = replace(config, num_fishnets=args.num_fishnets)
    if args.fisher_ridge_target_cond is not None:
        config = replace(
            config, fisher_ridge_target_cond=args.fisher_ridge_target_cond
        )
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

    screening = fisher_spectrum_report(fish_dir)
    log("Fisher screening diagnostic")
    print(json.dumps(screening, indent=2, sort_keys=True), flush=True)

    t0 = time.time()
    _, ensemble_w, _, flatten_model, ridge_info = fit_flattener(
        config, fish_dir, seeds, outdir
    )
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
        pruned_exprs, name_map={"X1": "k_f", "X2": "k_r", "X3": "k_cat"}
    )

    physical_exprs = expressions_to_physical(
        pruned_exprs, scaler, sr_offset=0.0, theta_names=THETA_NAMES, decimal=3
    )
    log("physical expressions")
    for k, expr in enumerate(physical_exprs):
        print(f"  eta_{k} = {expr}", flush=True)

    correlations = physics_correlations(physical_exprs)
    power_law = power_law_comparison(physical_exprs)
    kcat_fit = kcat_exponent_fit(physical_exprs)
    log("physical expression correlations")
    print(json.dumps(correlations, indent=2, sort_keys=True), flush=True)
    log("power-law comparison (headline metric)")
    print(json.dumps(power_law, indent=2, sort_keys=True), flush=True)
    log("k_cat exponent fit")
    print(json.dumps(kcat_fit, indent=2, sort_keys=True), flush=True)

    flatness = validate_flatness(aligned, mdl_coords, pruned_exprs)
    log("flatness scores")
    print(json.dumps(flatness, indent=2, sort_keys=True), flush=True)

    timings["total"] = time.time() - t_start
    corr_ok = correlations["best_michaelis_abs_corr"] >= args.min_michaelis_corr
    gain_ok = power_law["improvement_over_power_law"] >= args.min_power_law_gain
    success = bool(corr_ok and gain_ok)

    summary = {
        "run_id": f"enzyme_seed{args.master_seed}",
        "problem": "enzyme_kinetics",
        "master_seed": args.master_seed,
        "mode": args.mode,
        "status": "success" if success else "criterion_not_met",
        "seeds": seeds,
        "counts": {
            "n_train_simulations": config.nsims,
            "n_eval_simulations": config.nsims,
            "n_augmented_coordinate_evaluations": config.sr_grid_size,
            "n_substrate_concentrations": len(S0_VALUES),
            "n_times_per_curve": N_TIMES,
            "observable_dimension": len(S0_VALUES) * N_TIMES,
        },
        "screening": screening,
        "fisher_ridge": ridge_info,
        "discovery": {
            "expressions_physical": [str(e) for e in physical_exprs],
            "success": success,
            "best_michaelis_abs_corr": correlations["best_michaelis_abs_corr"],
            "best_kcat_abs_corr": correlations["best_kcat_abs_corr"],
            "power_law_r2_for_log_K_M": power_law["power_law_r2_for_log_K_M"],
            "discovered_r2_for_log_K_M": power_law["best_r2_vs_log_K_M"],
            "improvement_over_power_law": power_law["improvement_over_power_law"],
            "best_cosine_to_kcat": kcat_fit["best_cosine_to_kcat"],
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
                "power_law": power_law,
                "kcat_fit": kcat_fit,
                "flatness": flatness,
                "screening": screening,
                "fisher_ridge": ridge_info,
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
    shutil.copytree(fish_dir, sr_dir / "fishnets-enzyme", dirs_exist_ok=True)
    shutil.copy2(outdir / "enzyme_flatten.npz", sr_dir / "enzyme_flatten.npz")
    shutil.make_archive(str(outdir / "sr_results_enzyme"), "zip", root_dir=sr_dir)
    log(f"saved artifacts under {sr_dir}")

    if not success:
        raise SystemExit(
            "Did not recover the Michaelis-Menten reduction: "
            f"corr={correlations['best_michaelis_abs_corr']:.3f}, "
            f"power-law gain={power_law['improvement_over_power_law']:+.4f}"
        )
    log("enzyme kinetics distillery run complete")


if __name__ == "__main__":
    main()
