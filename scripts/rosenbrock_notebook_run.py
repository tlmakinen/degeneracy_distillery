#!/usr/bin/env python
"""Batch-safe version of tutorial_notebooks/rosenbrock_example.ipynb.

The default ``smoke`` mode keeps the same pipeline as the notebook but uses the
smaller hyperparameters from the slow Rosenbrock test so a Slurm GPU job can be
monitored interactively. ``full`` mode restores the notebook's heavier settings.
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
from dataclasses import asdict, dataclass
from dataclasses import replace as dataclasses_replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import jax.random as jr
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
from degeneracy_distillery.training_loop_fishnets import train_fishnets
from degeneracy_distillery.training_loop_flatten import fit_flattening


@dataclass(frozen=True)
class RunConfig:
    nsims: int
    n_d: int
    num_fishnets: int
    fish_hids_min: int
    fish_hids_max: int
    fish_layers: tuple[int, int]
    fish_epochs: int
    fish_min_epochs: int
    fish_patience: int
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
        n_d=25,
        num_fishnets=3,
        fish_hids_min=24,
        fish_hids_max=64,
        fish_layers=(2, 3),
        fish_epochs=1000,
        fish_min_epochs=100,
        fish_patience=20,
        flatten_hidden_size=64,
        flatten_layers=3,
        flatten_epochs_phase1=1000,
        flatten_epochs_phase2=1000,
        flatten_finetune_epochs=250,
        flatten_min_epochs=250,
        flatten_patience=40,
        align_subsample=800,
        sr_grid_size=1000,
        sr_time_limit=30,
        sr_max_length=20,
        sr_max_depth=8,
    ),
    "full": RunConfig(
        nsims=250,
        n_d=50,
        num_fishnets=20,
        fish_hids_min=10,
        fish_hids_max=300,
        fish_layers=(2, 5),
        fish_epochs=1000,
        fish_min_epochs=100,
        fish_patience=20,
        flatten_hidden_size=256,
        flatten_layers=7,
        flatten_epochs_phase1=1000,
        flatten_epochs_phase2=2000,
        flatten_finetune_epochs=250,
        flatten_min_epochs=250,
        flatten_patience=40,
        align_subsample=4000,
        sr_grid_size=2000,
        sr_time_limit=300,
        sr_max_length=25,
        sr_max_depth=10,
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
        raise SystemExit(
            "JAX did not initialize a GPU backend. This job should run on a GPU node."
        )


# Additive-stride master-seed derivation, matching the convention already used in
# scripts/rosen_nsims_logprob_sweep.py (run_seed = args.seed + nsims*10 + offset).
# The stride is large enough that master seeds 0-9 never collide across stages.
STAGE_SEED_STRIDE = 10_000
INVERTIBILITY_TIMEOUT_SECONDS = 30
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


def simulator_data(config: RunConfig, seed: int) -> dict[str, np.ndarray]:
    key = jr.PRNGKey(seed)
    sigma = jnp.diag(jnp.array([1.0, 2.0]) ** 2)

    def simulator(rng, theta):
        x_mean = jnp.array([theta[0], theta[1] - theta[0] ** 2])
        return jr.multivariate_normal(rng, mean=x_mean, cov=sigma, shape=(config.n_d,)).reshape(-1)

    k1, k2 = jr.split(key)
    theta_train = np.asarray(jr.uniform(k1, (config.nsims, 2), minval=-3.0, maxval=3.0))
    data_train = np.asarray(jax.vmap(simulator)(jr.split(k1, config.nsims), theta_train))
    theta_test = np.asarray(jr.uniform(k2, (config.nsims, 2), minval=-3.0, maxval=3.0))
    data_test = np.asarray(jax.vmap(simulator)(jr.split(k2, config.nsims), theta_test))
    log(f"theta_train {theta_train.shape}; data_train {data_train.shape}")
    return {
        "theta_train": theta_train,
        "data_train": data_train,
        "theta_test": theta_test,
        "data_test": data_test,
    }


def train_fishnet_ensemble(
    config: RunConfig, data: dict[str, np.ndarray], outdir: Path, seeds: dict[str, int]
) -> Path:
    scaler = fit_theta_scaler(data["theta_train"], feature_range=(-3.0, 3.0))
    theta_train_s = scaler.transform(data["theta_train"]).astype(np.float32)
    theta_test_s = scaler.transform(data["theta_test"]).astype(np.float32)
    log(f"scaled theta range: {theta_train_s.min(0)} to {theta_train_s.max(0)}")

    fish_dir = outdir / "fishnets-rosen"
    log(f"training fishnets into {fish_dir}")
    train_fishnets(
        theta_train_s,
        data["data_train"],
        theta_test_s,
        data["data_test"],
        num_models=config.num_fishnets,
        hids_min=config.fish_hids_min,
        hids_max=config.fish_hids_max,
        n_layers=list(config.fish_layers),
        train_epochs=config.fish_epochs,
        train_min_epochs=config.fish_min_epochs,
        patience=config.fish_patience,
        train_batch_size=25,
        lr=5e-5,
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
            hidden_size=config.flatten_hidden_size,
            n_layers=config.flatten_layers,
            batch_size=50,
            epochs_phase1=config.flatten_epochs_phase1,
            epochs_phase2=config.flatten_epochs_phase2,
            finetune_epochs=config.flatten_finetune_epochs,
            min_epochs=config.flatten_min_epochs,
            patience=config.flatten_patience,
            lr_phase1=1e-6,
            lr_schedule_initial=7e-5,
            lr_decay=0.3,
            lr_finetune=4e-6,
            Fisher_to_flatten="average",
            norm_factor=None,
            norm_method="median_det",
            flattener_activation="softplus",
            noise=1e-4,
            seed=seeds["flatten"],
            output_prefix="rosen_flatten",
            use_whitening=True,
            nn_inv=False,
            forward_backward_mlp=True,
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
        filename="rosen_flatten.npz",
        num_samps=config.align_subsample,
        seed=seeds["align"],
        process_ensemble=True,
        n_d=1.0,
        align_mode="procrustes",
        separate_nonlinearity=True,
        canonicalize="sign_only",
        use_prior_normalization=True,
        restore_reference_mean=False,
        Fisher_to_flatten="average",
        verbose=False,
    )
    x = aligned["X"]
    y = aligned["y"]
    y_min = y.min(0)
    y = y - y_min
    ys = aligned["ys"] - y_min
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
        "X": x,
        "y": y,
        "ys": ys,
        "y_std": aligned["y_std"],
        "dy_sr": aligned["dy_sr"],
        "Fs": aligned["Fs"],
        "n_params": n_params,
        "X_sr": np.asarray(x_sr),
        "y_sr": y_sr,
        "y_std_sr": y_std_sr,
    }


# Shared with the mdl_total recomputation in main() so the raw (non-normalized)
# description length reported in run_record.json is computed under the same
# length_penalty analyze_equations used to select the winning expressions.
SR_LENGTH_PENALTY = 3.0


def run_symbolic_regression(config: RunConfig, aligned: dict, outdir: Path, seeds: dict[str, int]):
    sr_dir = outdir / "sr_results_rosen"
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
        allowed_symbols="add,mul,div,pow,constant,variable,square",
        verbose=True,
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
        max_complexity_thresh=15,
        length_penalty=SR_LENGTH_PENALTY,
        equation_predicate=equation_predicate,
        verbose=True,
    )
    return sr_dir, mdl_coords, frob_coords, analysis


def expression_correlations(physical_exprs, seed: int) -> dict[str, object]:
    theta1, theta2 = sympy.symbols("theta1 theta2")
    rng = np.random.default_rng(seed)
    samples = rng.uniform(-3.0, 3.0, size=(4000, 2))
    targets = {
        "theta2_plus_theta1_sq": samples[:, 1] + samples[:, 0] ** 2,
        "theta2_minus_theta1_sq": samples[:, 1] - samples[:, 0] ** 2,
        "theta1": samples[:, 0],
        "theta2": samples[:, 1],
    }

    rows = []
    for i, expr in enumerate(physical_exprs):
        fn = sympy.lambdify((theta1, theta2), expr, modules="numpy")
        values = np.asarray(fn(samples[:, 0], samples[:, 1]), dtype=float)
        values = np.broadcast_to(values, (samples.shape[0],))
        row = {"component": i, "expr": str(expr)}
        for name, target in targets.items():
            if np.std(values) == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(values, target)[0, 1])
            row[name] = corr
        rows.append(row)

    best_rosen = max(
        max(abs(row["theta2_plus_theta1_sq"]), abs(row["theta2_minus_theta1_sq"]))
        for row in rows
    )
    best_theta1 = max(abs(row["theta1"]) for row in rows)
    return {"rows": rows, "best_rosen_abs_corr": best_rosen, "best_theta1_abs_corr": best_theta1}


def validate_flatness(aligned: dict, mdl_coords, pruned_exprs) -> dict[str, float]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIGS), default="smoke")
    parser.add_argument("--out-dir", type=Path, default=Path("results/rosenbrock_notebook_smoke"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--min-rosen-corr",
        type=float,
        default=0.5,
        help="Fail if no physical expression correlates this strongly with theta2 +/- theta1^2.",
    )
    parser.add_argument(
        "--min-theta1-corr",
        type=float,
        default=0.5,
        help="Fail if no physical expression correlates this strongly with the complementary "
        "linear direction theta1 (recovery requires both directions, per the discovery criteria).",
    )
    return parser.parse_args()


def git_commit_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.mode]
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"running mode={args.mode}; outdir={outdir}")
    log(f"config={json.dumps(asdict(config), sort_keys=True)}")

    seeds = derive_stage_seeds(args.seed)
    log(f"stage seeds={json.dumps(seeds, sort_keys=True)}")

    run_id = f"rosenbrock_seed{args.seed}"
    counts = {
        "n_train_simulations": config.nsims,
        "n_eval_simulations": config.nsims,
        "n_pca_simulations": 0,
        "n_augmented_coordinate_evaluations": config.sr_grid_size,
        "n_downstream_npe_simulations": 0,
    }
    config_manifest = {
        "run_id": run_id,
        "problem": "rosenbrock",
        "master_seed": args.seed,
        "mode": args.mode,
        "config": asdict(config),
        "stage_seeds": seeds,
        "thresholds": {
            "min_rosen_corr": args.min_rosen_corr,
            "min_theta1_corr": args.min_theta1_corr,
        },
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
            "problem": "rosenbrock",
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
        data = simulator_data(config, seeds["data"])
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
        # This stage both aligns coordinates and draws+evaluates the augmented SR
        # grid (fresh theta samples pushed through every ensemble flattening
        # member), per the doc's "flatten -> augment -> SR" ordering. The result's
        # X_sr/y_sr/y_std_sr (not the small aligned set) are what SR fits below.
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
            snap_rel_tol=0.5,
            snap_flat_tol=0.5,
            decimal=2,
            threshold=2.0,
        )
        print_discovered_expressions([sympy.simplify(e).evalf(2) for e in pruned_exprs])

        physical_exprs = expressions_to_physical(
            pruned_exprs,
            scaler,
            sr_offset=0.0,
            theta_names=("theta1", "theta2"),
            decimal=3,
        )
        log("physical expressions")
        for k, expr in enumerate(physical_exprs):
            print(f"  eta_{k} = {expr}", flush=True)

        correlations = expression_correlations(physical_exprs, seeds["validation"])
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
                "scaler_data_min": scaler.data_min_,
                "scaler_data_max": scaler.data_max_,
            },
            handle,
        )
    shutil.copytree(fish_dir, sr_dir / "fishnets-rosen", dirs_exist_ok=True)
    shutil.copy2(outdir / "rosen_flatten.npz", sr_dir / "rosen_flatten.npz")
    shutil.make_archive(str(outdir / "sr_results_rosen"), "zip", root_dir=sr_dir)
    log(f"saved artifacts under {sr_dir}")

    runtime_seconds["npe"] = None
    runtime_seconds["total"] = time.time() - total_start

    success = (
        correlations["best_rosen_abs_corr"] >= args.min_rosen_corr
        and correlations["best_theta1_abs_corr"] >= args.min_theta1_corr
    )

    run_record = {
        "run_id": run_id,
        "problem": "rosenbrock",
        "master_seed": args.seed,
        "status": "success",
        "counts": counts,
        "discovery": {
            "expressions_physical": [str(e) for e in physical_exprs],
            "expressions_canonical": [str(e) for e in mdl_coords],
            "success": success,
            "physics_alignment": correlations["best_rosen_abs_corr"],
            "complementary_linear_alignment": correlations["best_theta1_abs_corr"],
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
            f"rosen_corr={correlations['best_rosen_abs_corr']:.3f} (min {args.min_rosen_corr:.3f}), "
            f"theta1_corr={correlations['best_theta1_abs_corr']:.3f} (min {args.min_theta1_corr:.3f})"
        )

    log("Rosenbrock notebook batch run complete")


if __name__ == "__main__":
    main()
