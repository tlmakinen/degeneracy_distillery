#!/usr/bin/env python
"""Batch-safe conversion of scratch_notebooks/experiment_QM7b.ipynb.

QM7b molecular dataset: 5-parameter inference from graph-structured molecular data.
Fishnets are trained with a GATv2 graph network (PyTorch); flattening and SR run in JAX.

smoke mode: reduced hyperparameters for quick validation.
full  mode: notebook-scale hyperparameters for a full GPU run.
rebuttal mode: same frozen architecture/optimizer/SR settings as "full", with
n_train subsampled to 500 paired molecule observations and the SR-augmentation
grid bumped to 2000, per the NeurIPS rebuttal protocol.

QM7b is a *fixed empirical* paired graph/property dataset (7,211 molecules,
14 precomputed quantum-chemical regression targets each) -- there is no
generative simulator and no evaluable likelihood here. Counts reported below
are molecule-pair counts (train/held-out split sizes), not simulator calls.
See the "chemical_hypothesis" block written into config_manifest.json for the
predeclared theta/x definitions, the (empirically verified, not fabricated)
hypothesized chemical relation, and the predeclared success criterion.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import signal
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from dataclasses import replace as dataclasses_replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import sympy
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric import nn as gnn
from torch_geometric.loader.dataloader import Collater
from torch_geometric.nn import aggr

import esr.generation.generator  # noqa: F401 - sanity-check ESR import
from degeneracy_distillery.align_coords import load_and_process_data_v2
from degeneracy_distillery.postprocess_new import analyze_atom_sharing, regroup_like_terms
from degeneracy_distillery.postprocessing_utils import (
    check_flattening,
    diagnose_coordinate_rank_deficiency,
    flatten_with_numerical_jacobian,
    print_discovered_expressions,
    weighted_std,
)
from degeneracy_distillery.preprocessing_utils import get_eigenvalues
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
from degeneracy_distillery.torch import train_fishnets_dataset
from degeneracy_distillery.training_loop_flatten import fit_flattening


DIM_THETA = 5  # use first 5 of 14 QM7b labels as parameters

# Column order of the 14 QM7b regression targets, per the MoleculeNet/DeepChem
# qm7b.mat curation (Montavon et al. 2013, "Machine learning of molecular
# electronic properties in chemical compound space"): (0) PBE0 atomization
# energy, (1) ZINDO excitation energy of maximal optical absorption,
# (2) ZINDO highest absorption intensity, (3) ZINDO HOMO, (4) ZINDO LUMO,
# (5) ZINDO 1st excitation energy, (6) ZINDO ionization potential,
# (7) ZINDO electron affinity, (8) PBE0 HOMO, (9) PBE0 LUMO, (10) GW HOMO,
# (11) GW LUMO, (12) PBE0 polarizability, (13) SCS polarizability.
# DIM_THETA=5 therefore uses columns 0-4 as theta:
THETA_NAMES = ("atom_e", "exc_e", "abs_int", "homo", "lumo")
THETA_DESCRIPTIONS = {
    "atom_e": "PBE0 atomization energy (column 0 of the QM7b regression targets)",
    "exc_e": "ZINDO excitation energy of maximal optical absorption (column 1)",
    "abs_int": "ZINDO highest absorption intensity (column 2)",
    "homo": "ZINDO HOMO eigenvalue (column 3)",
    "lumo": "ZINDO LUMO eigenvalue (column 4)",
}


# ---------------------------------------------------------------------------
# Run configurations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    num_fishnets: int
    fish_hids_min: int
    fish_hids_max: int
    fish_layers: tuple[int, int]
    fish_epochs: int
    fish_min_epochs: int
    fish_patience: int
    fish_batch_size: int
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
    # None => keep the existing ~90% train / 10% held-out split unchanged.
    # Set (e.g. rebuttal=500) to additionally subsample the train split down
    # to a fixed number of paired molecule observations.
    n_train_molecules: int | None = None
    # Coordinate alignment. QM7b has historically used kabsch/sign_only with
    # separate_nonlinearity=False, unlike every other experiment script, which
    # uses the load_and_process_data_v2 default of procrustes. Exposed here so
    # the two can be compared without editing the call site.
    align_mode: str = "kabsch"
    separate_nonlinearity: bool = False


CONFIGS = {
    "smoke": RunConfig(
        num_fishnets=3,
        fish_hids_min=10,
        fish_hids_max=64,
        fish_layers=(1, 3),
        fish_epochs=500,
        fish_min_epochs=50,
        fish_patience=15,
        fish_batch_size=256,
        flatten_layers=4,
        flatten_epochs_phase1=1000,
        flatten_epochs_phase2=500,
        flatten_finetune_epochs=200,
        flatten_min_epochs=200,
        flatten_patience=30,
        align_subsample=500,
        sr_grid_size=500,
        sr_time_limit=60,
        sr_max_length=20,
        sr_max_depth=8,
    ),
    "full": RunConfig(
        # 10 -> 20 on 2026-07-26. QM7b was the only intractable/broken experiment
        # below the 20-network floor used by rosenbrock/imrphenomd/ising/kolmogorov/
        # kuramoto. A larger ensemble tightens the standard error on the per-point
        # ensemble spread y_std_sr, which is what symbolic regression consumes as its
        # per-point uncertainty, so an under-sized ensemble makes that weighting
        # noisy. Measured ensemble spread at 20 members (on kuramoto) was already
        # well-calibrated -- signal/spread ~23-31 -- so this is a calibration
        # improvement, NOT expected on its own to fix QM7b's 0/10; see the
        # Pareto-front filtering issue documented alongside.
        num_fishnets=20,
        fish_hids_min=10,
        fish_hids_max=100,
        fish_layers=(1, 4),
        fish_epochs=4000,
        fish_min_epochs=100,
        fish_patience=20,
        fish_batch_size=256,
        flatten_layers=7,
        flatten_epochs_phase1=10000,
        flatten_epochs_phase2=5000,
        flatten_finetune_epochs=1000,
        flatten_min_epochs=1200,
        flatten_patience=50,
        align_subsample=4000,
        sr_grid_size=5000,
        sr_time_limit=600,
        sr_max_length=30,
        sr_max_depth=10,
    ),
}

# NeurIPS rebuttal configuration: same frozen architecture/optimizer/SR settings as
# "full" (no per-seed retuning). QM7b has no "nsims" knob (it's a fixed dataset,
# not a simulator) -- the rebuttal-protocol equivalent of "500 training
# simulations" is 500 training molecule-pairs, via n_train_molecules. The SR
# augmentation grid is bumped to 2000 per the rebuttal protocol.
CONFIGS["rebuttal"] = dataclasses_replace(
    CONFIGS["full"], sr_grid_size=2000, n_train_molecules=500
)

# Exploratory variant of "rebuttal": every seed under the official "rebuttal" config
# failed the geometric_improvement criterion, several by a large margin (e.g. one seed
# had frob_symbolic ~1000x worse than frob_raw) -- not obviously an SR-search-budget
# problem, but worth an honest empirical test rather than assuming. 3x the per-component
# SR time limit (600s -> 1800s), everything else identical/frozen to "rebuttal". Kept as
# a separate named config rather than mutating "rebuttal" itself, since "rebuttal" has
# already-reported results.
CONFIGS["rebuttal_longsr"] = dataclasses_replace(CONFIGS["rebuttal"], sr_time_limit=1800)

# Procrustes-alignment arm. Everything else is frozen to "rebuttal"; only the
# alignment changes, so this is a clean A/B against a "rebuttal"-config control
# run from the same fishnet artifacts. Alignment sits downstream of fishnet and
# flattener training, so this arm can reuse trained fishnets via
# --skip-fishnets and does not need a GPU.
CONFIGS["rebuttal_procrustes"] = dataclasses_replace(
    CONFIGS["rebuttal"], align_mode="procrustes", separate_nonlinearity=True
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require_gpu_if_requested(require_gpu: bool) -> None:
    backend = jax.default_backend()
    devices = jax.devices()
    log(f"JAX backend: {backend}")
    log(f"JAX devices: {devices}")
    log(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if require_gpu and backend != "gpu":
        raise SystemExit(
            "JAX did not initialize a GPU backend. This job should run on a GPU node."
        )
    if require_gpu and not torch.cuda.is_available():
        raise SystemExit(
            "PyTorch CUDA is not available. This job should run on a GPU node."
        )


# Additive-stride master-seed derivation, matching the convention already used in
# scripts/rosenbrock_notebook_run.py, scripts/gw_notebook_run.py and
# scripts/imrphenomd_notebook_run.py (run_seed = args.seed + offset * stride).
# The stride is large enough that master seeds 0-9 never collide across stages.
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


# ---------------------------------------------------------------------------
# Graph Attention Network (GATv2) embedding
# ---------------------------------------------------------------------------
class GATNetwork(nn.Module):
    def __init__(
        self,
        in_channels,
        gcn_channels,
        gcn_heads,
        dense_channels,
        out_channels,
        drop_p=0.1,
        edge_dim=None,
    ):
        super().__init__()
        # Avoid SoftmaxAggregation(learn=True) — it calls torch_scatter's scatter_softmax
        # which lacks sm_70 kernels in recent wheels. Use 5 standard aggregations instead.
        self.graph_aggr = aggr.MultiAggregation(
            aggrs=["sum", "mean", "std", "min", "max"],
            mode="cat",
        )
        self.dropout = nn.Dropout(p=drop_p)
        self.gcn_channels = gcn_channels
        self.gcn_heads = gcn_heads
        self.edge_dim = edge_dim

        self.conv1 = gnn.GATv2Conv(
            in_channels, gcn_channels[0], heads=gcn_heads[0], edge_dim=edge_dim
        )
        self.convs = nn.ModuleList(
            [
                gnn.GATv2Conv(
                    gcn_channels[i] * gcn_heads[i],
                    gcn_channels[i + 1],
                    heads=gcn_heads[i + 1],
                    edge_dim=edge_dim,
                )
                for i in range(len(gcn_channels) - 2)
            ]
        )
        self.conv2 = gnn.GATv2Conv(
            gcn_channels[-2] * gcn_heads[-2],
            gcn_channels[-1],
            heads=gcn_heads[-1],
            concat=False,
            edge_dim=edge_dim,
        )
        in_dense = gcn_channels[-1] * len(self.graph_aggr.aggrs)
        self.fc1 = nn.Linear(in_dense, dense_channels[0])
        self.fcs = nn.ModuleList(
            [nn.Linear(dense_channels[i], dense_channels[i + 1]) for i in range(len(dense_channels) - 1)]
        )
        self.fc2 = nn.Linear(dense_channels[-1], out_channels)

    def forward(self, x):
        node_features = torch.ones(x.num_nodes, 1).to(x.y.device)
        edge_index, edge_attr = x.edge_index, x.edge_attr
        ptr = x.ptr if hasattr(x, "ptr") else None

        h = F.relu(self.conv1(node_features, edge_index, edge_attr))
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_attr))
        h = self.conv2(h, edge_index, edge_attr)
        h = self.graph_aggr(h, ptr=ptr)

        h = F.relu(self.fc1(h))
        for fc in self.fcs:
            h = self.dropout(h)
            h = F.relu(fc(h))
        return self.fc2(h)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_qm7b_data(
    data_dir: Path, config: RunConfig, seed: int
) -> tuple[DataLoader, DataLoader, dict]:
    log(f"loading QM7b dataset from {data_dir}")
    dataset = pyg.datasets.QM7b(root=str(data_dir))

    rng = np.random.default_rng(seed)
    mask = rng.random(len(dataset)) < 0.9
    train_idx_full = np.nonzero(mask)[0]
    test_idx = np.nonzero(~mask)[0]

    if config.n_train_molecules is not None and config.n_train_molecules < len(train_idx_full):
        train_idx = np.sort(
            rng.choice(train_idx_full, size=config.n_train_molecules, replace=False)
        )
        log(
            f"subsampling train split from {len(train_idx_full)} to "
            f"{config.n_train_molecules} molecule pairs (rebuttal protocol)"
        )
    else:
        train_idx = train_idx_full

    train_data = dataset[train_idx]
    test_data = dataset[test_idx]
    log(f"train molecules: {len(train_data)}, test molecules: {len(test_data)}")

    # Fit the theta scaler on the (possibly subsampled) train split only, in
    # physical units, using the shared fit_theta_scaler helper so that
    # expressions_to_physical can invert SR's scaled-theta expressions back to
    # physical theta later on -- previously this script scaled theta ad hoc via
    # (y - min) / (max - min) computed over the *combined* train+test dataset
    # and never kept a scaler object around at all.
    theta_train_physical = dataset.y[train_idx][:, :DIM_THETA].numpy().astype(np.float64)
    theta_test_physical = dataset.y[test_idx][:, :DIM_THETA].numpy().astype(np.float64)
    # feature_range=(1,2), matching every other experiment script (gw_notebook_run.py,
    # imrphenomd_notebook_run.py, sir_notebook_run.py, and the original SIR tutorial
    # notebook) -- NOT (0,1), which was a real deviation introduced here. (0,1) lets
    # scaled theta sit arbitrarily close to 0; (1,2) keeps it bounded safely away from
    # 0. Confirmed this matters concretely: substituting the observed sqrt(0.001*atom_e
    # + 1.228) discovered-expression pattern back through the (0,1)-scaler's implied
    # constant, the sqrt argument goes negative (-0.98) at the most negative atom_e in
    # the dataset -- exactly the "invalid value encountered in sqrt" warning seen in
    # every QM7b run. With (1,2) the same expression's constant shifts by +1 and stays
    # positive at that same extreme. Likely a material contributor to (perhaps the
    # whole explanation for) QM7b's 0/10 geometric_improvement failures and the
    # ComplexInfinity crash -- not necessarily a genuine finding about the dataset.
    scaler = fit_theta_scaler(theta_train_physical, feature_range=(1.0, 2.0))

    collater = Collater(dataset)

    def collate_fn(batch):
        batch = collater(batch)
        theta_raw = batch.y[:, :DIM_THETA].numpy().astype(np.float64)
        scaled = scaler.transform(theta_raw)
        return batch, torch.as_tensor(scaled, dtype=torch.float32)

    train_loader = DataLoader(
        train_data,
        batch_size=config.fish_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.fish_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    meta = {
        "scaler_scale": scaler.scale_,
        "scaler_min": scaler.min_,
        "scaler_data_min": scaler.data_min_,
        "scaler_data_max": scaler.data_max_,
        "n_train_molecules": np.asarray(len(train_data)),
        "n_eval_molecules": np.asarray(len(test_data)),
        "theta_holdout_physical": theta_test_physical,
    }
    return train_loader, test_loader, meta


# ---------------------------------------------------------------------------
# Fishnet ensemble (PyTorch + GNN)
# ---------------------------------------------------------------------------
def train_fishnet_ensemble(
    config: RunConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    outdir: Path,
    seeds: dict[str, int],
) -> Path:
    # GATNetwork's weight init otherwise draws from PyTorch's global (unseeded)
    # RNG state -- seed it globally first, matching the convention already used
    # in scripts/sir_nsims_logprob_sweep.py's train_posterior.
    np.random.seed(seeds["fish_model"])
    random.seed(seeds["fish_model"])
    torch.manual_seed(seeds["fish_model"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds["fish_model"])

    emb = GATNetwork(
        in_channels=1,
        gcn_channels=[4, 8],
        gcn_heads=[4, 4],
        dense_channels=[32, 16],
        out_channels=6,
        edge_dim=1,
    )
    fish_dir = outdir / "fishnets-qm7b"
    log(f"training GNN fishnets into {fish_dir}")
    train_fishnets_dataset(
        train_loader,
        test_loader,
        n_params=DIM_THETA,
        embedding_net=emb,
        hids_min=config.fish_hids_min,
        hids_max=config.fish_hids_max,
        n_layers=list(config.fish_layers),
        num_models=config.num_fishnets,
        train_epochs=config.fish_epochs,
        train_min_epochs=config.fish_min_epochs,
        patience=config.fish_patience,
        train_batch_size=config.fish_batch_size,
        lr=2e-5,
        prefetch_batches=0,
        seed_model=seeds["fish_model"],
        seed_train=seeds["fish_train"],
        device="cuda",
        amp_dtype=torch.bfloat16,
        outdir=str(fish_dir),
        update_pbar_every=25,
    )
    return fish_dir


# ---------------------------------------------------------------------------
# Flattening (JAX)
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
            seed=seeds["flatten"],
            output_prefix="qm7b_flattening",
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
    config: RunConfig, outdir: Path, ensemble_w, flatten_model, seeds: dict[str, int]
) -> dict:
    log("aligning coordinates")
    aligned = load_and_process_data_v2(
        datapath=str(outdir) + os.sep,
        filename="qm7b_flattening.npz",
        num_samps=config.align_subsample,
        seed=seeds["align"],
        process_ensemble=True,
        n_d=1.0,
        align_mode=config.align_mode,
        separate_nonlinearity=config.separate_nonlinearity,
        canonicalize="sign_only",
        use_prior_normalization=True,
        restore_reference_mean=False,
        Fisher_to_flatten="average",
        verbose=False,
    )

    x = aligned["X"]
    mask = x[:, 0] > 0.0
    x = x[mask]
    y = aligned["y"][mask]
    ys = np.array([yy[mask] for yy in aligned["ys"]])
    y_std = aligned["y_std"][mask]
    dy_sr = aligned["dy_sr"][mask]
    Fs = aligned["Fs"][mask]
    n_params = x.shape[1]

    # Shift so y min = 1 (matching notebook convention)
    ymin_ = y.min(0) - 1.0
    y -= ymin_
    ys -= ymin_
    log(f"aligned X {x.shape}; y {y.shape} (min={y.min(0).round(2)})")

    # Data augmentation: sample uniformly in aligned X coordinate space
    key = jr.PRNGKey(seeds["sr_grid"])
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
    y_sr -= y_sr.min(0)
    y_sr += 1
    ys_sr_rot += 1

    return {
        "data": aligned,
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
    }


# Shared with the mdl_total recomputation in main() so the raw (non-normalized)
# description length reported in run_record.json is computed under the same
# length_penalty analyze_equations used to select the winning expressions.
SR_LENGTH_PENALTY = 2.0


# ---------------------------------------------------------------------------
# Symbolic regression
# ---------------------------------------------------------------------------
def run_symbolic_regression(
    config: RunConfig, aligned: dict, outdir: Path, seeds: dict[str, int]
) -> tuple[Path, list, list, dict]:
    sr_dir = outdir / "sr_results_qm7b"
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
        allowed_symbols="add,mul,div,pow,constant,variable,sqrt,logabs",
        verbose=True,
    )

    # forbid_self_transcendental flipped False -> True on 2026-07-26. QM7b was the
    # only experiment in the suite still allowing self-transcendental forms (every
    # other script -- rosenbrock, gw, imrphenomd, sir, ising, kolmogorov, kuramoto --
    # already sets True), and it is also the worst-performing (0/10 under its frozen
    # criterion, with symbolic held-out flatness up to ~1000x worse than the raw
    # parameters). Nested/self-transcendental forms are exactly what lets SR
    # curve-fit a good *value* match with wild derivatives, which is fatal here
    # because the flatness metric depends on the Jacobian, not the value. Note this
    # only *filters* the Pareto front -- it cannot add good expressions that SR never
    # found -- so it is a partial mitigation, not a fix for the degenerate-front
    # problem documented in follow_up_results/*/rebuttal/*threshold_analysis*.md.
    equation_predicate = sr_structure_predicate(
        n_params=aligned["n_params"],
        forbid_self_transcendental=True,
        check_nested_exp=False,
    )
    filter_summaries = filter_pareto_fronts(
        str(sr_dir),
        aligned["n_params"],
        equation_predicate,
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
        length_penalty=SR_LENGTH_PENALTY,
        equation_predicate=equation_predicate,
        verbose=True,
    )
    return sr_dir, mdl_coords, frob_coords, analysis


# ---------------------------------------------------------------------------
# Validation / predeclared discovery criterion
# ---------------------------------------------------------------------------
def chemical_correlations(physical_exprs, theta_holdout: np.ndarray, seed: int) -> dict[str, object]:
    """Predeclared QM7b discovery criterion, evaluated on held-out molecules.

    Hypothesis (see also the "chemical_hypothesis" block written to
    config_manifest.json): the ZINDO HOMO-LUMO gap (theta_lumo - theta_homo,
    i.e. columns 4-3 of the QM7b regression targets) is an approximate
    surrogate for the ZINDO excitation energy of maximal optical absorption
    (theta_exc_e, column 1), reflecting the standard single-particle
    (Koopmans-like) approximation used for semi-empirical excitation
    energies. This was empirically checked against the real downloaded QM7b
    dataset before being adopted here (not fabricated): corr(lumo-homo,
    exc_e) = 0.62 across all 7,211 molecules -- a moderate, not exact/tight,
    correlation, reported honestly as such.
    """
    atom_e, exc_e, abs_int, homo, lumo = sympy.symbols(" ".join(THETA_NAMES))
    rng = np.random.default_rng(seed)
    n = theta_holdout.shape[0]
    idx = rng.choice(n, size=min(n, 5000), replace=False)
    samples = theta_holdout[idx]
    targets = {
        "homo_lumo_gap": samples[:, 4] - samples[:, 3],
        "excitation_energy": samples[:, 1],
    }

    rows = []
    for i, expr in enumerate(physical_exprs):
        fn = sympy.lambdify((atom_e, exc_e, abs_int, homo, lumo), expr, modules="numpy")
        values = np.asarray(fn(*[samples[:, j] for j in range(5)]), dtype=float)
        values = np.broadcast_to(values, (samples.shape[0],))
        row = {"component": i, "expr": str(expr)}
        for name, target in targets.items():
            if np.std(values) == 0 or not np.isfinite(values).all():
                corr = 0.0
            else:
                corr = float(np.corrcoef(values, target)[0, 1])
            row[name] = corr
        rows.append(row)

    best_gap = max(abs(row["homo_lumo_gap"]) for row in rows)
    best_exc = max(abs(row["excitation_energy"]) for row in rows)
    return {
        "rows": rows,
        "best_gap_abs_corr": best_gap,
        "best_excitation_abs_corr": best_exc,
    }


def validate_flatness(aligned: dict, mdl_coords: list, pruned_exprs: list) -> dict:
    nn_flats = jax.vmap(flatten_with_numerical_jacobian)(aligned["dy_sr"], aligned["Fs"])
    mdl_flats, _ = check_flattening(mdl_coords, X=aligned["X"], Fs=aligned["Fs"])
    pruned_flats, _ = check_flattening(pruned_exprs, X=aligned["X"], Fs=aligned["Fs"])

    def fro_score(q):
        return np.linalg.norm(np.asarray(q) - np.eye(aligned["n_params"]), axis=(-2, -1))

    scores = {
        "raw_theta": float(np.median(fro_score(aligned["Fs"]))),
        "mdl": float(np.median(fro_score(mdl_flats))),
        "pruned": float(np.median(fro_score(pruned_flats))),
        "nn": float(np.median(fro_score(nn_flats))),
    }
    evalues_nn = np.asarray(jax.vmap(get_eigenvalues)(nn_flats)).ravel()
    scores["nn_median_abs_log_eigenvalue"] = float(
        np.median(np.abs(np.log(np.clip(evalues_nn, 1e-12, None))))
    )
    scores["median_condition_raw"] = float(np.median(np.linalg.cond(np.asarray(aligned["Fs"]))))
    scores["median_condition_symbolic"] = float(np.median(np.linalg.cond(np.asarray(pruned_flats))))
    return scores


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIGS), default="smoke")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/qm7b_notebook_smoke")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/qm7b"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--from-dir", type=Path, default=None,
        help="Path to a prior run directory. Required when --skip-fishnets is set.",
    )
    parser.add_argument(
        "--skip-flatten", action="store_true", default=False,
        help="Reuse the flattener from --from-dir instead of refitting it. "
             "Required for CPU runs: refitting diverges to non-finite "
             "eta_ensemble on CPU for this problem.",
    )
    parser.add_argument(
        "--skip-fishnets", action="store_true", default=False,
        help="Skip fishnet training and re-run flattening+SR from saved fishnets_outputs.npz in --from-dir.",
    )
    parser.add_argument(
        "--min-gap-corr",
        type=float,
        default=0.5,
        help="Fail if no physical expression correlates this strongly (held-out) with the "
        "hypothesized ZINDO HOMO-LUMO gap (lumo - homo).",
    )
    parser.add_argument(
        "--geometric-improvement-margin",
        type=float,
        default=0.8,
        help="Fail unless held-out symbolic flattening beats the raw-theta baseline by this "
        "fractional margin (pruned_frob < raw_theta_frob * margin).",
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

    run_id = f"qm7b_seed{args.seed}"

    chemical_hypothesis = {
        "theta_definition": {
            name: THETA_DESCRIPTIONS[name] for name in THETA_NAMES
        },
        "x_definition": (
            "Molecular graph built from the QM7b Coulomb matrix (edge_index/edge_attr "
            "give bonded/nonbonded pairwise Coulomb interactions); embedded via a "
            "GATv2-based graph attention network (GATNetwork) before entering the "
            "fishnet ensemble."
        ),
        "hypothesized_chemical_relation": (
            "The ZINDO HOMO-LUMO gap (theta_lumo - theta_homo) approximates the ZINDO "
            "excitation energy of maximal optical absorption (theta_exc_e), per the "
            "standard single-particle (Koopmans-like) approximation for semi-empirical "
            "excitation energies."
        ),
        "hypothesis_provenance": (
            "Empirically verified against the real downloaded QM7b dataset before being "
            "adopted as the predeclared criterion (not fabricated): "
            "corr(lumo - homo, exc_e) = 0.62 across all 7,211 QM7b molecules. This is a "
            "moderate, not an exact/tight, correlation and is reported honestly as such -- "
            "it is not claimed to be a precise physical law."
        ),
        "predeclared_success_criterion": (
            "success = (max |corr| between any discovered physical expression and the "
            "held-out HOMO-LUMO gap >= min_gap_corr) AND (held-out symbolic flattening "
            "Frobenius loss beats the raw-theta baseline by >= geometric_improvement_margin, "
            "i.e. frob_symbolic < frob_raw * geometric_improvement_margin)."
        ),
        "graph_model_and_split": (
            "GATv2Conv-based GATNetwork embedding feeding a fishnet ensemble; 90%/10% "
            "random train/held-out split over all 7,211 QM7b molecules, seeded by "
            "seeds['data']; rebuttal mode additionally subsamples the train split down "
            "to config.n_train_molecules paired observations (held-out split size is "
            "left unchanged)."
        ),
        "likelihood_status": (
            "QM7b is a fixed empirical paired graph/property dataset with no generative "
            "simulator and no evaluable likelihood. It is NOT claimed to be an "
            "intractable-likelihood simulator; counts below are molecule-pair counts, "
            "not simulator calls."
        ),
    }

    config_manifest = {
        "run_id": run_id,
        "problem": "qm7b",
        "master_seed": args.seed,
        "mode": args.mode,
        "config": asdict(config),
        "stage_seeds": seeds,
        "thresholds": {
            "min_gap_corr": args.min_gap_corr,
            "geometric_improvement_margin": args.geometric_improvement_margin,
        },
        "chemical_hypothesis": chemical_hypothesis,
        "git_commit": git_commit_hash(),
    }
    with open(outdir / "config_manifest.json", "w") as handle:
        json.dump(config_manifest, handle, indent=2, sort_keys=True)

    require_gpu_if_requested(args.require_gpu)

    runtime_seconds: dict[str, float | None] = {}
    total_start = time.time()
    counts = {
        "n_train_simulations": 0,
        "n_eval_simulations": 0,
        "n_pca_simulations": 0,
        "n_augmented_coordinate_evaluations": config.sr_grid_size,
        "n_downstream_npe_simulations": 0,
        "counts_are_molecule_pairs": True,
    }

    def write_failure(stage: str, exc: Exception) -> None:
        runtime_seconds["total"] = time.time() - total_start
        record = {
            "run_id": run_id,
            "problem": "qm7b",
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
        log(f"FAILED at stage={stage}: {exc}\n{traceback.format_exc()}")

    try:
        stage_start = time.time()
        if args.skip_fishnets:
            if args.from_dir is None:
                raise ValueError("--from-dir is required when --skip-fishnets is set")
            from_dir = args.from_dir.resolve()
            log(f"skipping fishnet training; loading artifacts from {from_dir}")
            meta_data = np.load(from_dir / "qm7b_meta.npz")
            meta = {k: meta_data[k] for k in meta_data.files}
            fish_dir = from_dir / "fishnets-qm7b"
            if not (fish_dir / "fishnets_outputs.npz").exists():
                raise FileNotFoundError(f"fishnets_outputs.npz not found in {fish_dir}")
            np.savez(outdir / "qm7b_meta.npz", **meta)
            scaler = SimpleNamespace(scale_=meta["scaler_scale"], min_=meta["scaler_min"])
            counts["n_train_simulations"] = int(meta["n_train_molecules"])
            counts["n_eval_simulations"] = int(meta["n_eval_molecules"])
            theta_holdout_physical = meta["theta_holdout_physical"]
        else:
            train_loader, test_loader, meta = load_qm7b_data(args.data_dir, config, seeds["data"])
            np.savez(outdir / "qm7b_meta.npz", **meta)
            scaler = SimpleNamespace(scale_=meta["scaler_scale"], min_=meta["scaler_min"])
            counts["n_train_simulations"] = int(meta["n_train_molecules"])
            counts["n_eval_simulations"] = int(meta["n_eval_molecules"])
            theta_holdout_physical = meta["theta_holdout_physical"]
            fish_dir = train_fishnet_ensemble(config, train_loader, test_loader, outdir, seeds)
        runtime_seconds["fishnets"] = time.time() - stage_start
    except Exception as exc:
        write_failure("fishnets", exc)
        raise

    try:
        stage_start = time.time()
        if args.skip_flatten:
            # Reuse a previously fitted flattener instead of refitting it. Needed
            # for the alignment A/B: refitting on CPU reproducibly diverges for
            # QM7b (every eta_ensemble entry comes back non-finite, which then
            # fails in load_and_process_data), while the saved GPU-fitted
            # flattener is finite everywhere. Reusing it also makes the
            # comparison exact -- identical fishnets AND identical flattener, so
            # align_mode is the only thing that differs between arms.
            if args.from_dir is None:
                raise ValueError("--from-dir is required when --skip-flatten is set")
            src = args.from_dir.resolve()
            log(f"skipping flattener fit; loading artifacts from {src}")
            for fname in ("qm7b_flattening.npz", "qm7b_flattening_flatten_model.pkl"):
                if not (src / fname).exists():
                    raise FileNotFoundError(f"{fname} not found in {src}")
                shutil.copy2(src / fname, outdir / fname)
            with open(outdir / "qm7b_flattening_flatten_model.pkl", "rb") as handle:
                saved_flat = pickle.load(handle)
            ensemble_w = saved_flat["ensemble_ws"]
            flatten_model = saved_flat["flatten_model"]
            eta_ens = np.load(outdir / "qm7b_flattening.npz")["eta_ensemble"]
            finite_frac = float(np.isfinite(eta_ens).mean())
            log(f"reused flattener: eta_ensemble finite fraction {finite_frac:.3f}")
            if finite_frac < 1.0:
                raise ValueError(
                    f"reused flattener has non-finite eta_ensemble "
                    f"(finite fraction {finite_frac:.3f}); refusing to continue"
                )
        else:
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
            decimal=1,
            threshold=0.05,
        )
        print_discovered_expressions([sympy.simplify(e).evalf(2) for e in pruned_exprs])

        physical_exprs = expressions_to_physical(
            pruned_exprs,
            scaler,
            sr_offset=0.0,
            theta_names=THETA_NAMES,
            decimal=3,
        )
        log("physical expressions")
        for k, expr in enumerate(physical_exprs):
            print(f"  eta_{k} = {expr}", flush=True)

        correlations = chemical_correlations(physical_exprs, theta_holdout_physical, seeds["validation"])
        log("physical expression correlations (held-out molecules)")
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
            },
            handle,
        )
    shutil.copytree(fish_dir, sr_dir / "fishnets-qm7b", dirs_exist_ok=True)
    shutil.copy2(outdir / "qm7b_flattening.npz", sr_dir / "qm7b_flattening.npz")
    shutil.make_archive(str(outdir / "sr_results_qm7b"), "zip", root_dir=sr_dir)
    log(f"saved artifacts under {sr_dir}")

    runtime_seconds["npe"] = None
    runtime_seconds["total"] = time.time() - total_start

    geometric_improvement = flatness["pruned"] < flatness["raw_theta"] * args.geometric_improvement_margin
    success = (
        correlations["best_gap_abs_corr"] >= args.min_gap_corr
        and geometric_improvement
    )

    run_record = {
        "run_id": run_id,
        "problem": "qm7b",
        "master_seed": args.seed,
        "status": "success",
        "counts": counts,
        "discovery": {
            "expressions_physical": [str(e) for e in physical_exprs],
            "expressions_canonical": [str(e) for e in mdl_coords],
            "success": success,
            "physics_alignment": correlations["best_gap_abs_corr"],
            "excitation_energy_alignment": correlations["best_excitation_abs_corr"],
            "geometric_improvement": geometric_improvement,
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
            f"gap_corr={correlations['best_gap_abs_corr']:.3f} (min {args.min_gap_corr:.3f}), "
            f"geometric_improvement={geometric_improvement} "
            f"(frob_symbolic={flatness['pruned']:.4f}, "
            f"frob_raw*margin={flatness['raw_theta'] * args.geometric_improvement_margin:.4f})"
        )

    log("QM7b notebook batch run complete")


if __name__ == "__main__":
    main()
