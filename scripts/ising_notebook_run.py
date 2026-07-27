#!/usr/bin/env python
"""Degeneracy distillery on the 2D Ising model (intractable likelihood).

The Boltzmann weight of a spin configuration is

    p(s | J, T, h) = exp( (J/T) * sum_<ij> s_i s_j + (h/T) * sum_i s_i ) / Z(J, T, h)

so the likelihood is normalised by an intractable partition function Z, and the
configuration distribution depends on the three parameters ``theta = (J, T, h)``
*only* through the two combinations

    K = J / T     and     B = h / T.

The model therefore has an exact one-dimensional degeneracy along the ray
``(J, T, h) -> (lam*J, lam*T, lam*h)``.  A successful run recovers two
identifiable coordinates aligned with ``J/T`` and ``h/T`` plus one nuisance
direction along the scaling ray, from raw spin configurations alone.

The field is kept strictly positive so the magnetisation stays unimodal; this
avoids the ``+m/-m`` symmetry breaking that would make the posterior genuinely
multimodal (a documented limitation of the method).

Usage
-----
    python scripts/ising_notebook_run.py --mode smoke --master-seed 0
    python scripts/ising_notebook_run.py --mode full --master-seed 3 \
        --out-dir results/rebuttal_discovery/ising/seed_3
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

# JAX on GPU defaults to reduced-precision (TF32-like) matmuls; on CPU it uses
# full float32. That precision gap is a known source of GPU-only NaN losses in
# nets that feed into a Fisher-information log-determinant (small negative
# eigenvalues from precision loss blow up under log). Force full float32
# matmul precision unconditionally so GPU runs match the (working) CPU
# numerics, rather than depending on a Slurm launcher remembering to set
# JAX_DEFAULT_MATMUL_PRECISION.
jax.config.update("jax_default_matmul_precision", "float32")

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

# Priors are strictly positive so symbolic regression can use log/pow safely,
# and h > 0 keeps the magnetisation unimodal.  The ranges put J/T in
# [0.125, 0.556], straddling the h=0 critical coupling 0.4407 without spending
# most of the prior in the saturated ferromagnetic phase, where the
# magnetisation pins at 1 and carries almost no information about h/T.
J_MIN, J_MAX = 0.4, 1.0
T_MIN, T_MAX = 1.8, 3.2
H_MIN, H_MAX = 0.05, 0.50

THETA_NAMES = ("J", "T", "h")


@dataclass(frozen=True)
class RunConfig:
    nsims: int
    lattice: int
    n_snapshots: int
    equil_sweeps: int
    snapshot_gap: int
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
        lattice=12,
        n_snapshots=4,
        equil_sweeps=400,
        snapshot_gap=25,
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
        lattice=16,
        n_snapshots=8,
        equil_sweeps=1500,
        snapshot_gap=50,
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

# NeurIPS rebuttal configuration. Unlike Rosenbrock/GW/etc., Ising's own "full"
# mode already sits at the rebuttal-campaign target of 500 training simulations
# and 2000 augmented coordinate evaluations (see CONFIGS["full"] above:
# nsims=500, sr_grid_size=2000) -- those numbers were sized directly against the
# lattice/Metropolis cost tradeoff documented in neurips_intractable_examples.md,
# not left at a smaller notebook-era default the way Rosenbrock's was. So
# "rebuttal" here is "full" by another name; the explicit replace (a no-op on
# the two fields the sibling scripts vary) is kept anyway so the CLI/launcher
# convention (`--mode rebuttal`) matches the other five scripts and so a future
# change to CONFIGS["full"] doesn't silently drift the rebuttal campaign's
# simulation budget out from under it.
CONFIGS["rebuttal"] = replace(CONFIGS["full"], nsims=500, sr_grid_size=2000)

# Shared with the mdl_total recomputation in main() so the raw (non-normalized)
# description length reported in run_record.json is computed under the same
# length_penalty the analyze_equations call below uses to select the winning
# expressions.
SR_LENGTH_PENALTY = 2.0
INVERTIBILITY_TIMEOUT_SECONDS = 30


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


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def derive_seeds(master_seed: int) -> dict[str, int]:
    """Deterministically expand one master seed into per-stage seeds."""
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
# Simulator: batched checkerboard Metropolis
# --------------------------------------------------------------------------


def _checkerboard_masks(lattice: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    parity = np.indices((lattice, lattice)).sum(axis=0) % 2
    return jnp.asarray(parity == 0), jnp.asarray(parity == 1)


def _neighbour_sum(spins: jnp.ndarray) -> jnp.ndarray:
    return (
        jnp.roll(spins, 1, axis=-2)
        + jnp.roll(spins, -1, axis=-2)
        + jnp.roll(spins, 1, axis=-1)
        + jnp.roll(spins, -1, axis=-1)
    )


def _sweep(spins, key, coupling, field, mask_a, mask_b):
    """One full checkerboard Metropolis sweep, batched over simulations.

    ``coupling = J/T`` and ``field = h/T`` are already in reduced units, which
    makes the exact degeneracy of the simulator explicit.
    """
    for mask in (mask_a, mask_b):
        neighbours = _neighbour_sum(spins)
        # Energy change in units of kT.
        delta = 2.0 * spins * (coupling * neighbours + field)
        key, sub = jr.split(key)
        draws = jr.uniform(sub, spins.shape)
        # exp(-max(delta,0)) == 1 whenever delta <= 0, so downhill moves always accept.
        accept = draws < jnp.exp(-jnp.maximum(delta, 0.0))
        spins = jnp.where(jnp.logical_and(mask, accept), -spins, spins)
    return spins, key


def _run_sweeps(spins, key, coupling, field, mask_a, mask_b, n_sweeps):
    def body(carry, _):
        spins, key = carry
        spins, key = _sweep(spins, key, coupling, field, mask_a, mask_b)
        return (spins, key), None

    (spins, key), _ = jax.lax.scan(body, (spins, key), None, length=n_sweeps)
    return spins, key


def _simulate_chunk(key, theta_chunk, config: RunConfig) -> jnp.ndarray:
    """Return stacked decorrelated spin snapshots for a chunk of parameters."""
    lattice = config.lattice
    mask_a, mask_b = _checkerboard_masks(lattice)

    j_vals = theta_chunk[:, 0][:, None, None]
    t_vals = theta_chunk[:, 1][:, None, None]
    h_vals = theta_chunk[:, 2][:, None, None]
    coupling = j_vals / t_vals
    field = h_vals / t_vals

    key, sub = jr.split(key)
    spins = jnp.where(
        jr.uniform(sub, (theta_chunk.shape[0], lattice, lattice)) < 0.5, -1.0, 1.0
    )

    spins, key = _run_sweeps(
        spins, key, coupling, field, mask_a, mask_b, config.equil_sweeps
    )

    snapshots = []
    for _ in range(config.n_snapshots):
        spins, key = _run_sweeps(
            spins, key, coupling, field, mask_a, mask_b, config.snapshot_gap
        )
        snapshots.append(spins)

    stacked = jnp.stack(snapshots, axis=1)
    return stacked.reshape(theta_chunk.shape[0], -1)


def simulator_data(config: RunConfig, seed: int, outdir: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = 2 * config.nsims

    theta_all = np.stack(
        [
            rng.uniform(J_MIN, J_MAX, total),
            rng.uniform(T_MIN, T_MAX, total),
            rng.uniform(H_MIN, H_MAX, total),
        ],
        axis=1,
    ).astype(np.float32)

    log(f"running Metropolis for {total} configurations on {config.lattice}^2 lattice")
    key = jr.PRNGKey(seed)
    chunks = []
    for start in range(0, total, config.sim_chunk):
        stop = min(start + config.sim_chunk, total)
        key, sub = jr.split(key)
        chunks.append(
            np.asarray(_simulate_chunk(sub, jnp.asarray(theta_all[start:stop]), config))
        )
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
    coupling = theta[:, 0] / theta[:, 1]
    magnetisation = data.mean(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sc = axes[0].scatter(theta[:, 0], theta[:, 1], c=coupling, s=6, cmap="viridis")
    plt.colorbar(sc, ax=axes[0], label="J/T")
    axes[0].axhline(0.0)
    axes[0].set_xlabel("J")
    axes[0].set_ylabel("T")
    axes[0].set_title("Prior samples coloured by J/T")

    axes[1].scatter(coupling, magnetisation, s=6, alpha=0.6)
    axes[1].axvline(0.4407, ls="--", color="grey", label="2D Ising K_c (h=0)")
    axes[1].set_xlabel("J/T")
    axes[1].set_ylabel("mean magnetisation")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Order parameter vs reduced coupling")

    axes[2].imshow(
        data[0, : config.lattice**2].reshape(config.lattice, config.lattice),
        cmap="binary",
    )
    axes[2].set_title("Example configuration")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.tight_layout()
    fig.savefig(outdir / "ising_input_summary.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------
# Distillery stages
# --------------------------------------------------------------------------


def _safe_std(a: jnp.ndarray, axis) -> jnp.ndarray:
    """`sqrt(var + eps)` -- a drop-in for `.std()` with a finite gradient at zero.

    THIS IS LOAD-BEARING, do not "simplify" it back to `a.std(axis=axis)`.
    `std` is `sqrt(var)`, and `d/dz sqrt(z)` is infinite at `z = 0`, so a channel
    with exactly zero variance produces a finite forward pass but NaN gradients,
    which poisons every parameter in the ensemble on the first backward pass.

    Ising hits this constantly rather than rarely. The conv stack uses CIRCULAR
    padding, so a spatially *constant* input maps to a spatially *constant*
    feature map at every position -- exact zero variance in all channels at once.
    A fully magnetised lattice (all spins +1) is exactly such an input, and the
    prior is deliberately set up to produce them: J/T spans [0.125, 0.556],
    straddling the critical coupling 0.4407, and above it the magnetisation pins
    at 1 (the module docstring for the prior says as much). Reproduced directly:
    a saturated configuration gives `grad_finite=False` with `.std()` and
    `grad_finite=True` with this function.

    This was the cause of the GPU fishnet-training NaN (loss=nan by epoch ~24 on
    every ensemble member, `Ensemble weights: [0. 0. 0. 0.]`). It is not really a
    "GPU bug" -- CPU is vulnerable to the same thing and merely got luckier on
    which configurations it drew; float32 also underflows near-zero variance to
    exactly zero, so even an almost-constant channel triggers it.
    """
    return jnp.sqrt(jnp.var(a, axis=axis) + 1e-6)


class SnapshotEncoder(nn.Module):
    """Permutation-invariant encoder for a set of lattice configurations.

    A dense encoder on the flattened snapshots has ~5e5 parameters and overfits
    badly at these simulation budgets.  Convolutions with circular padding
    respect the periodic boundary conditions, and pooling over space and then
    over snapshots makes the embedding invariant to lattice translations and to
    snapshot ordering, which cuts the parameter count by more than an order of
    magnitude.  The std-pooled channels carry the fluctuation information that
    mean pooling alone would discard.
    """

    lattice: int
    n_snapshots: int
    channels: tuple[int, ...] = (16, 32, 32)
    out_features: int = 64

    @nn.compact
    def __call__(self, x):
        h = x.reshape(self.n_snapshots, self.lattice, self.lattice, 1)
        for width in self.channels:
            h = nn.gelu(nn.Conv(width, (3, 3), padding="CIRCULAR")(h))
        per_snapshot_mean = h.mean(axis=(1, 2))
        per_snapshot_std = _safe_std(h, (1, 2))
        pooled = jnp.concatenate(
            [
                per_snapshot_mean.mean(axis=0),
                _safe_std(per_snapshot_mean, 0),
                per_snapshot_std.mean(axis=0),
            ]
        )
        return nn.gelu(nn.Dense(self.out_features)(pooled))


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

    embedding_net = SnapshotEncoder(
        lattice=config.lattice, n_snapshots=config.n_snapshots
    )

    fish_dir = outdir / "fishnets-ising"
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
            output_prefix="ising_flatten",
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
    """Align the coordinate ensemble, then build the augmented SR training set.

    The augmentation step draws fresh parameters from the prior and evaluates the
    already-trained coordinate ensemble on them.  These are cheap network
    evaluations and consume no additional simulator calls.
    """
    log("aligning coordinates")
    aligned = load_and_process_data_v2(
        datapath=str(outdir) + os.sep,
        filename="ising_flatten.npz",
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
    y_min = y.min(0)
    y = y - y_min
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
    sr_dir = outdir / "sr_results_ising"
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
        allowed_symbols="add,mul,div,pow,constant,variable,square,logabs",
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
        length_penalty=SR_LENGTH_PENALTY,
        equation_predicate=predicate,
    )
    return sr_dir, mdl_coords, frob_coords, analysis


def physics_correlations(physical_exprs) -> dict[str, object]:
    """Correlate discovered coordinates with the known reduced variables."""
    j_sym, t_sym, h_sym = sympy.symbols("J T h")
    rng = np.random.default_rng(123)
    samples = rng.uniform(
        [J_MIN, T_MIN, H_MIN], [J_MAX, T_MAX, H_MAX], size=(5000, 3)
    )
    j, t, h = samples[:, 0], samples[:, 1], samples[:, 2]

    targets = {
        "log_J_over_T": np.log(j / t),
        "log_h_over_T": np.log(h / t),
        "J_over_T": j / t,
        "h_over_T": h / t,
        # The scaling ray is the exact nuisance direction.
        "log_T_scale": np.log(t),
    }
    identifiable = ("log_J_over_T", "log_h_over_T", "J_over_T", "h_over_T")

    rows = []
    for i, expr in enumerate(physical_exprs):
        fn = sympy.lambdify((j_sym, t_sym, h_sym), expr, modules="numpy")
        values = np.asarray(fn(j, t, h), dtype=float)
        values = np.broadcast_to(values, (samples.shape[0],))
        row = {"component": i, "expr": str(expr)}
        finite = np.isfinite(values)
        for name, target in targets.items():
            if np.std(values[finite]) == 0 or finite.sum() < 100:
                row[name] = 0.0
            else:
                row[name] = float(np.corrcoef(values[finite], target[finite])[0, 1])
        rows.append(row)

    best_coupling = max(
        max(abs(row["log_J_over_T"]), abs(row["J_over_T"])) for row in rows
    )
    best_field = max(max(abs(row["log_h_over_T"]), abs(row["h_over_T"])) for row in rows)
    return {
        "rows": rows,
        "best_coupling_abs_corr": best_coupling,
        "best_field_abs_corr": best_field,
        "best_identifiable_abs_corr": min(best_coupling, best_field),
        "identifiable_targets": list(identifiable),
    }


def validate_flatness(aligned: dict, mdl_coords, pruned_exprs) -> dict[str, float]:
    # The textbook reduced coordinates are the natural ad-hoc baseline.
    adhoc = ["X1 / X2", "X3 / X2", "X2"]
    adhoc_flats, _ = check_flattening(adhoc, X=aligned["X"], Fs=aligned["Fs"])
    mdl_flats, _ = check_flattening(mdl_coords, X=aligned["X"], Fs=aligned["Fs"])
    pruned_flats, _ = check_flattening(pruned_exprs, X=aligned["X"], Fs=aligned["Fs"])
    nn_flats = jax.vmap(flatten_with_numerical_jacobian)(aligned["dy_sr"], aligned["Fs"])

    identity = np.eye(aligned["n_params"])

    def fro_score(q):
        return np.linalg.norm(np.asarray(q) - identity, axis=(-2, -1))

    scores = {
        "raw_theta": float(np.median(fro_score(aligned["Fs"]))),
        "adhoc_reduced": float(np.median(fro_score(adhoc_flats))),
        "mdl": float(np.median(fro_score(mdl_flats))),
        "pruned": float(np.median(fro_score(pruned_flats))),
        "nn": float(np.median(fro_score(nn_flats))),
    }
    scores["median_condition_raw"] = float(np.median(np.linalg.cond(np.asarray(aligned["Fs"]))))
    scores["median_condition_symbolic"] = float(np.median(np.linalg.cond(np.asarray(pruned_flats))))
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIGS), default="smoke")
    parser.add_argument("--out-dir", type=Path, default=Path("results/ising_notebook"))
    parser.add_argument(
        "--master-seed",
        type=int,
        default=0,
        help="Single seed controlling simulator, networks, alignment and SR.",
    )
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sr-time-limit",
        type=int,
        default=None,
        help="Override the symbolic-regression budget per component, in seconds.",
    )
    parser.add_argument(
        "--n-snapshots",
        type=int,
        default=None,
        help="Override snapshots per simulation; raise this if h/T is not recovered.",
    )
    parser.add_argument(
        "--min-identifiable-corr",
        type=float,
        default=0.7,
        help="Fail unless both J/T and h/T are recovered at least this strongly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.mode]
    if args.sr_time_limit is not None:
        config = replace(config, sr_time_limit=args.sr_time_limit)
    if args.n_snapshots is not None:
        config = replace(config, n_snapshots=args.n_snapshots)
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = derive_seeds(args.master_seed)

    log(f"running mode={args.mode}; outdir={outdir}")
    log(f"master_seed={args.master_seed}; derived seeds={json.dumps(seeds)}")
    log(f"config={json.dumps(asdict(config), sort_keys=True)}")

    run_id = f"ising_seed{args.master_seed}"
    counts = {
        "n_train_simulations": config.nsims,
        "n_eval_simulations": config.nsims,
        "n_augmented_coordinate_evaluations": config.sr_grid_size,
        "n_pca_simulations": 0,
        "n_downstream_npe_simulations": 0,
        "lattice_sites": config.lattice**2,
        "snapshots_per_simulation": config.n_snapshots,
    }
    config_manifest = {
        "run_id": run_id,
        "problem": "ising",
        "master_seed": args.master_seed,
        "mode": args.mode,
        "config": asdict(config),
        "stage_seeds": seeds,
        "thresholds": {
            "min_identifiable_corr": args.min_identifiable_corr,
        },
        "git_commit": git_commit_hash(),
    }
    with open(outdir / "config_manifest.json", "w") as handle:
        json.dump(config_manifest, handle, indent=2, sort_keys=True)

    require_gpu_if_requested(args.require_gpu)

    timings: dict[str, float] = {}
    t_start = time.time()

    def write_failure(stage: str, exc: Exception) -> None:
        timings["total"] = time.time() - t_start
        record = {
            "run_id": run_id,
            "problem": "ising",
            "master_seed": args.master_seed,
            "status": "failed",
            "failure_stage": stage,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(),
            "counts": counts,
            "runtime_seconds": timings,
        }
        with open(outdir / "run_record.json", "w") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
        log(f"FAILED at stage={stage}: {exc}\n{traceback.format_exc()}")

    try:
        t0 = time.time()
        data = simulator_data(config, seeds["simulator"], outdir)
        timings["simulation"] = time.time() - t0

        t0 = time.time()
        fish_dir, scaler = train_fishnet_ensemble(config, data, seeds, outdir)
        timings["fishnets"] = time.time() - t0
    except Exception as exc:
        write_failure("fishnets", exc)
        raise

    try:
        t0 = time.time()
        _, ensemble_w, _, flatten_model = fit_flattener(config, fish_dir, seeds, outdir)
        timings["flatten"] = time.time() - t0
    except Exception as exc:
        write_failure("flatten", exc)
        raise

    try:
        t0 = time.time()
        aligned = align_and_augment(config, seeds, outdir, ensemble_w, flatten_model)
        timings["alignment_and_augmentation"] = time.time() - t0
    except Exception as exc:
        write_failure("alignment", exc)
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
        print_discovered_expressions(
            pruned_exprs, name_map={"X1": "J", "X2": "T", "X3": "h"}
        )

        physical_exprs = expressions_to_physical(
            pruned_exprs, scaler, sr_offset=0.0, theta_names=THETA_NAMES, decimal=3
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
        timings["symbolic_regression"] = time.time() - t0
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
    shutil.copytree(fish_dir, sr_dir / "fishnets-ising", dirs_exist_ok=True)
    shutil.copy2(outdir / "ising_flatten.npz", sr_dir / "ising_flatten.npz")
    shutil.make_archive(str(outdir / "sr_results_ising"), "zip", root_dir=sr_dir)
    log(f"saved artifacts under {sr_dir}")

    timings["total"] = time.time() - t_start
    success = correlations["best_identifiable_abs_corr"] >= args.min_identifiable_corr

    run_record = {
        "run_id": run_id,
        "problem": "ising",
        "master_seed": args.master_seed,
        "status": "success",
        "seeds": seeds,
        "counts": counts,
        "discovery": {
            "expressions_physical": [str(e) for e in physical_exprs],
            "expressions_canonical": [str(e) for e in mdl_coords],
            "success": bool(success),
            "physics_alignment": correlations["best_identifiable_abs_corr"],
            "best_coupling_abs_corr": correlations["best_coupling_abs_corr"],
            "best_field_abs_corr": correlations["best_field_abs_corr"],
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
            "frob_adhoc": flatness["adhoc_reduced"],
            "median_condition_raw": flatness["median_condition_raw"],
            "median_condition_symbolic": flatness["median_condition_symbolic"],
        },
        "inference": {
            "crps_theta": None,
            "crps_eta": None,
            "coverage_error_theta": None,
            "coverage_error_eta": None,
        },
        "runtime_seconds": timings,
    }
    with open(outdir / "run_record.json", "w") as handle:
        json.dump(run_record, handle, indent=2, sort_keys=True)
    log(f"wrote run record to {outdir / 'run_record.json'}")

    if not success:
        raise SystemExit(
            "Did not recover both reduced coordinates: "
            f"{correlations['best_identifiable_abs_corr']:.3f} < {args.min_identifiable_corr:.3f}"
        )
    log("Ising distillery run complete")


if __name__ == "__main__":
    main()
