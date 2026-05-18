"""End-to-end smoke test of the Distillery pipeline on the Rosenbrock toy.

Mirrors ``tutorial_notebooks/tiny_quick_rosenbrock_example.ipynb`` but with
*tiny* configs so the full pipeline finishes in ~5 minutes:

  * 3-member Fisher-network ensemble, ~50 epochs each
  * short flattening run (phase 1 + phase 2 + finetune, all small)
  * coordinate alignment via ``load_and_process_data_v2``
  * ~30 s of pyoperon symbolic regression
  * ``analyze_equations`` + ``regroup_like_terms`` postprocessing
  * a sanity check that the flattened-frame eigenvalues cluster near 1

Each stage is its own ``test_*`` function; the heavy artefacts are produced
once by session-scoped fixtures and threaded through, so the suite walks
the same pipeline the notebook does without retraining anything.

Run just this file with::

    pytest tests/test_rosenbrock_pipeline.py -m slow -v

Or include it in the full suite::

    pytest -m slow
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional-dependency gating: skip the whole module if heavy backends absent.
# ---------------------------------------------------------------------------
_jax = pytest.importorskip("jax", reason="JAX required for the Distillery pipeline")
pytest.importorskip("flax", reason="flax required for the Distillery pipeline")
pytest.importorskip(
    "pyoperon", reason="pyoperon required for the symbolic-regression stage"
)
pytest.importorskip(
    "esr", reason="ESR required for analyze_equations MDL scoring"
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import sympy  # noqa: E402

from degeneracy_distillery.training_loop_fishnets import train_fishnets  # noqa: E402
from degeneracy_distillery.training_loop_flatten import fit_flattening  # noqa: E402
from degeneracy_distillery.align_coords import load_and_process_data_v2  # noqa: E402
from degeneracy_distillery.sr_utils import (  # noqa: E402
    analyze_equations,
    expressions_to_physical,
    fit_symbolic_regression,
    fit_theta_scaler,
    sr_structure_predicate,
)
from degeneracy_distillery.postprocess_new import (  # noqa: E402
    analyze_atom_sharing,
    regroup_like_terms,
)
from degeneracy_distillery.postprocessing_utils import (  # noqa: E402
    check_flattening,
    flatten_with_numerical_jacobian,
    weighted_std,
)
from degeneracy_distillery.preprocessing_utils import get_eigenvalues  # noqa: E402

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Tiny tunables — keep aligned with the docstring's "~5 min" budget.
# ---------------------------------------------------------------------------
NUM_FISHNETS = 3
NSIMS = 200
N_D = 25  # samples per simulation; flatten_dim = 2 * N_D = 50
FISHNET_EPOCHS = 100  # ~60 was just enough to let one member diverge with this seed
FISHNET_BATCH = 25
FLATTEN_EPOCHS_PHASE1 = 40
FLATTEN_EPOCHS_PHASE2 = 80
FLATTEN_FINETUNE_EPOCHS = 20
FLATTEN_MIN_EPOCHS = 20
FLATTEN_PATIENCE = 25
SR_TIME_LIMIT = 30  # seconds, per component
ALIGN_SUBSAMPLE = 800


# ---------------------------------------------------------------------------
# Fixtures: each builds on the previous one and is session-scoped so the
# whole module runs the pipeline exactly once.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    """Isolated scratch directory for fishnets/flatten/SR artefacts."""
    return tmp_path_factory.mktemp("rosen_pipeline")


@pytest.fixture(scope="module")
def simulator_data() -> Dict[str, np.ndarray]:
    """Rosenbrock-like Gaussian simulator: mu_x = (theta1, theta2 - theta1^2)."""
    key = jr.PRNGKey(0)
    Sigma = jnp.diag(jnp.array([1.0, 2.0]) ** 2)

    def simulator(rng, theta):
        x_mean = jnp.array([theta[0], theta[1] - theta[0] ** 2])
        return jr.multivariate_normal(
            rng, mean=x_mean, cov=Sigma, shape=(N_D,)
        ).reshape(-1)

    k1, k2 = jr.split(key)
    theta_train = np.asarray(jr.uniform(k1, (NSIMS, 2), minval=-3.0, maxval=3.0))
    data_train = np.asarray(jax.vmap(simulator)(jr.split(k1, NSIMS), theta_train))
    theta_test = np.asarray(jr.uniform(k2, (NSIMS, 2), minval=-3.0, maxval=3.0))
    data_test = np.asarray(jax.vmap(simulator)(jr.split(k2, NSIMS), theta_test))

    return dict(
        theta_train=theta_train,
        data_train=data_train,
        theta_test=theta_test,
        data_test=data_test,
    )


@pytest.fixture(scope="module")
def scaled_data(simulator_data):
    """Apply the same theta scaler the notebook uses."""
    scaler = fit_theta_scaler(simulator_data["theta_train"], feature_range=(-3.0, 3.0))
    theta_train_s = scaler.transform(simulator_data["theta_train"]).astype(np.float32)
    theta_test_s = scaler.transform(simulator_data["theta_test"]).astype(np.float32)
    return dict(scaler=scaler, theta_train_s=theta_train_s, theta_test_s=theta_test_s)


@pytest.fixture(scope="module")
def fishnets_run(simulator_data, scaled_data, workdir) -> Path:
    """Train a tiny Fisher-network ensemble; return the outdir."""
    outdir = workdir / "fishnets-rosen"
    train_fishnets(
        scaled_data["theta_train_s"],
        simulator_data["data_train"],
        scaled_data["theta_test_s"],
        simulator_data["data_test"],
        num_models=NUM_FISHNETS,
        hids_min=24,
        hids_max=64,
        n_layers=[2, 3],
        train_epochs=FISHNET_EPOCHS,
        train_min_epochs=20,
        patience=15,
        train_batch_size=FISHNET_BATCH,
        lr=5e-5,
        seed_model=201,
        seed_train=999,
        outdir=str(outdir),
        update_pbar_every=25,
    )
    return outdir


@pytest.fixture(scope="module")
def flatten_run(fishnets_run, workdir):
    """Run a short flattening fit on top of the ensemble Fishers."""
    fish = np.load(fishnets_run / "fishnets_outputs.npz")
    thetas = jnp.array(fish["theta"])
    ensemble_weights_np = np.asarray(fish["ensemble_weights"])
    Fs_np = np.asarray(fish["Fs"])

    # Defensive: with a tiny ensemble (``NUM_FISHNETS=3``) a single diverged
    # member can produce all-NaN Fishers that poison ``allFs.mean(0)`` and
    # blow up later consumers. Filter them out before flattening; assert that
    # we still have at least one usable member so the test fails loudly if
    # *every* fishnet diverged (real regression vs. expected flakiness).
    finite_mask = np.isfinite(Fs_np).all(axis=(1, 2, 3))
    n_finite = int(finite_mask.sum())
    assert n_finite >= 1, "all fishnet ensemble members produced non-finite Fishers"
    if n_finite < Fs_np.shape[0]:
        Fs_np = Fs_np[finite_mask]
        ensemble_weights_np = ensemble_weights_np[finite_mask]

    F_network_ensemble = jnp.array(Fs_np)
    ensemble_weights = ensemble_weights_np

    # `fit_flattening` writes ``{output_prefix}.npz`` to CWD, so use a chdir
    # context to keep the artefact inside ``workdir``.
    cwd_before = Path.cwd()
    os.chdir(workdir)
    try:
        w, ensemble_w, outputs_flatten, flatten_model = fit_flattening(
            F_network_ensemble,
            thetas,
            ensemble_weights=ensemble_weights,
            hidden_size=64,
            n_layers=3,
            batch_size=50,
            epochs_phase1=FLATTEN_EPOCHS_PHASE1,
            epochs_phase2=FLATTEN_EPOCHS_PHASE2,
            finetune_epochs=FLATTEN_FINETUNE_EPOCHS,
            min_epochs=FLATTEN_MIN_EPOCHS,
            patience=FLATTEN_PATIENCE,
            lr_phase1=1e-6,
            lr_schedule_initial=7e-5,
            lr_decay=0.3,
            lr_finetune=4e-6,
            Fisher_to_flatten="average",
            norm_factor=None,
            norm_method="median_det",
            flattener_activation="softplus",
            noise=1e-4,
            seed=0,
            output_prefix="rosen_flatten",
            use_whitening=True,
            nn_inv=False,
            forward_backward_mlp=True,
            l1_alpha=0.0,
            do_plot=False,
            return_model=True,
            save_flatten_model_pickle=False,
            update_pbar_every=25,
        )
    finally:
        os.chdir(cwd_before)

    return dict(
        npz_path=workdir / "rosen_flatten.npz",
        w=w,
        ensemble_w=ensemble_w,
        outputs=outputs_flatten,
        model=flatten_model,
    )


@pytest.fixture(scope="module")
def aligned_data(flatten_run, workdir) -> Dict[str, Any]:
    """Run the Procrustes alignment / nonlinearity-rotation step."""
    data = load_and_process_data_v2(
        datapath=str(workdir) + os.sep,
        filename="rosen_flatten.npz",
        num_samps=ALIGN_SUBSAMPLE,
        seed=44,
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

    X = data["X"]
    y = data["y"]
    y_min = y.min(0)
    y = y - y_min
    ys = data["ys"] - y_min

    return dict(
        data=data,
        X=X,
        y=y,
        ys=ys,
        y_std=data["y_std"],
        dy_sr=data["dy_sr"],
        Fs=data["Fs"],
        n_params=X.shape[1],
    )


@pytest.fixture(scope="module")
def sr_grid(aligned_data, flatten_run):
    """Sample a fresh SR grid in the aligned X frame and push through the flow."""
    X = aligned_data["X"]
    n_params = aligned_data["n_params"]
    ensemble_w = flatten_run["ensemble_w"]
    flatten_model = flatten_run["model"]
    data = aligned_data["data"]

    key = jr.PRNGKey(7)
    X_sr = jr.uniform(
        key, minval=X.min(0), maxval=X.max(0), shape=(1000, n_params)
    )

    ys_sr = jnp.array([
        jax.vmap(lambda x: flatten_model.apply(w_i, x))(X_sr) for w_i in ensemble_w
    ])

    ys_sr_rot = np.array([
        np.einsum(
            "ij,bj->bi", data["rotmats"][i], ys_sr[i] - ys_sr[i].mean(0)
        )
        for i in range(len(ys_sr))
    ])

    y_std_sr = weighted_std(ys_sr_rot, data["ensemble_weights"])
    y_sr = np.average(ys_sr_rot, 0, data["ensemble_weights"])
    ys_sr_rot -= y_sr.min(0)
    y_sr -= y_sr.min(0)

    return dict(X_sr=np.asarray(X_sr), y_sr=y_sr, y_std_sr=y_std_sr)


@pytest.fixture(scope="module")
def sr_results(sr_grid, workdir) -> Path:
    """Run ~30 s of pyoperon SR; return the parent results dir."""
    parent_dir = workdir / "sr_results_rosen"
    parent_dir.mkdir(exist_ok=True)
    fit_symbolic_regression(
        sr_grid["X_sr"],
        sr_grid["y_sr"],
        sr_grid["y_std_sr"],
        parent_dir=str(parent_dir) + os.sep,
        random_state=32134,
        time_limit=SR_TIME_LIMIT,
        max_length=20,
        max_depth=8,
        allowed_symbols="add,mul,div,pow,constant,variable,square",
        verbose=False,
    )
    return parent_dir


@pytest.fixture(scope="module")
def analyzed_equations(sr_results, aligned_data):
    """MDL/Frobenius ranking of the SR pareto fronts."""
    mdl_coords, frob_coords, analysis = analyze_equations(
        aligned_data["X"],
        aligned_data["y"],
        aligned_data["y_std"],
        aligned_data["dy_sr"],
        aligned_data["Fs"],
        parent_dir=str(sr_results) + os.sep,
        n_params=aligned_data["n_params"],
        equation_set="pareto",
        max_complexity_thresh=15,
        length_penalty=3.0,
        equation_predicate=sr_structure_predicate(
            n_params=aligned_data["n_params"],
            forbid_self_transcendental=True,
        ),
        verbose=False,
    )
    return dict(mdl_coords=mdl_coords, frob_coords=frob_coords, analysis=analysis)


# ---------------------------------------------------------------------------
# Per-stage tests. Each one is small and just asserts the contract that the
# next stage relies on.
# ---------------------------------------------------------------------------
def test_simulator_shapes(simulator_data):
    flat_dim = 2 * N_D
    assert simulator_data["theta_train"].shape == (NSIMS, 2)
    assert simulator_data["data_train"].shape == (NSIMS, flat_dim)
    assert simulator_data["theta_test"].shape == (NSIMS, 2)
    assert simulator_data["data_test"].shape == (NSIMS, flat_dim)
    assert np.all(np.isfinite(simulator_data["data_train"]))


def test_theta_scaler_roundtrip(simulator_data, scaled_data):
    scaler = scaled_data["scaler"]
    theta_train = simulator_data["theta_train"]
    theta_train_s = scaled_data["theta_train_s"]

    assert theta_train_s.shape == theta_train.shape
    assert theta_train_s.min() >= -3.0 - 1e-4
    assert theta_train_s.max() <= 3.0 + 1e-4
    inverted = scaler.inverse_transform(theta_train_s)
    np.testing.assert_allclose(inverted, theta_train, rtol=1e-4, atol=1e-4)


def test_fishnets_outputs_on_disk(fishnets_run):
    npz_path = fishnets_run / "fishnets_outputs.npz"
    assert npz_path.is_file(), "train_fishnets did not write fishnets_outputs.npz"

    with np.load(npz_path) as fish:
        for key in ("theta", "Fs", "ensemble_weights"):
            assert key in fish.files, f"missing key {key!r} in fishnets outputs"
        theta = fish["theta"]
        Fs = fish["Fs"]
        ensemble_weights = fish["ensemble_weights"]

    assert theta.shape == (NSIMS, 2)
    assert Fs.shape == (NUM_FISHNETS, NSIMS, 2, 2)
    assert ensemble_weights.shape == (NUM_FISHNETS,)
    assert np.all(np.isfinite(Fs)), "fishnet Fishers contain non-finite entries"


def test_fishers_are_positive_definite(fishnets_run):
    """Per-sample Fishers should be SPD for a well-defined Gaussian model."""
    with np.load(fishnets_run / "fishnets_outputs.npz") as fish:
        Fs = fish["Fs"]
    eigs = np.linalg.eigvalsh(Fs)
    assert np.all(eigs > -1e-3), "Fisher eigenvalues went meaningfully negative"


def test_flatten_run_produces_npz_and_model(flatten_run):
    assert flatten_run["npz_path"].is_file(), "flatten npz was not written"
    assert flatten_run["model"] is not None, "fit_flattening did not return a model"
    assert flatten_run["ensemble_w"] is not None
    with np.load(flatten_run["npz_path"]) as out:
        for key in ("theta", "eta", "Jacobians", "F_ensemble", "ensemble_weights"):
            assert key in out.files, f"flatten output missing key {key!r}"


def test_aligned_shapes(aligned_data):
    n = aligned_data["X"].shape[0]
    p = aligned_data["n_params"]
    assert p == 2
    assert aligned_data["y"].shape == (n, p)
    assert aligned_data["y_std"].shape == (n, p)
    assert aligned_data["dy_sr"].shape[0] == n
    assert aligned_data["Fs"].shape == (n, p, p)
    assert "rotmats" in aligned_data["data"]


def test_sr_grid_is_well_formed(sr_grid):
    X_sr = sr_grid["X_sr"]
    y_sr = sr_grid["y_sr"]
    y_std_sr = sr_grid["y_std_sr"]
    assert X_sr.shape == (1000, 2)
    assert y_sr.shape == (1000, 2)
    assert y_std_sr.shape == (1000, 2)
    assert np.all(np.isfinite(y_sr))
    assert np.all(y_std_sr >= 0)


def test_sr_results_written(sr_results, aligned_data):
    """One component_*/pareto.csv per fitted y dimension."""
    for j in range(aligned_data["n_params"]):
        pareto = sr_results / f"component_{j + 1}" / "pareto.csv"
        assert pareto.is_file(), f"missing pareto.csv for component {j + 1}"
        assert pareto.stat().st_size > 0


def test_analyze_equations_returns_coords(analyzed_equations, aligned_data):
    mdl = analyzed_equations["mdl_coords"]
    frob = analyzed_equations["frob_coords"]
    assert len(mdl) == aligned_data["n_params"]
    assert len(frob) == aligned_data["n_params"]
    for eq in mdl + frob:
        assert isinstance(eq, str) and len(eq) > 0
        sympy.sympify(eq)


def test_postprocess_regroup(analyzed_equations, aligned_data):
    mdl_coords = analyzed_equations["mdl_coords"]
    report = analyze_atom_sharing(mdl_coords)
    assert isinstance(report, dict)

    pruned_exprs, R, info = regroup_like_terms(
        mdl_coords,
        X=aligned_data["X"],
        Fs=aligned_data["Fs"],
        n_params=aligned_data["n_params"],
        method="atoms",
        do_snap=True,
        snap_rel_tol=0.5,
        snap_flat_tol=0.5,
        decimal=2,
        threshold=2.0,
    )
    assert len(pruned_exprs) == aligned_data["n_params"]
    assert R.shape == (aligned_data["n_params"], aligned_data["n_params"])


def test_back_to_physical_theta(analyzed_equations, aligned_data, scaled_data):
    mdl_coords = analyzed_equations["mdl_coords"]
    pruned_exprs, _, _ = regroup_like_terms(
        mdl_coords,
        X=aligned_data["X"],
        Fs=aligned_data["Fs"],
        n_params=aligned_data["n_params"],
        method="atoms",
        do_snap=True,
        snap_rel_tol=0.5,
        snap_flat_tol=0.5,
        decimal=2,
        threshold=2.0,
    )
    physical_exprs = expressions_to_physical(
        pruned_exprs,
        scaled_data["scaler"],
        sr_offset=0.0,
        theta_names=("theta1", "theta2"),
        decimal=3,
    )
    assert len(physical_exprs) == aligned_data["n_params"]
    free_names = {str(s) for e in physical_exprs for s in e.free_symbols}
    assert free_names.issubset({"theta1", "theta2"})


def test_validation_flatness_eigenvalues(analyzed_equations, aligned_data):
    """Flatness check: NN flattening should beat the raw-theta baseline."""
    nn_flats = jax.vmap(flatten_with_numerical_jacobian)(
        aligned_data["dy_sr"], aligned_data["Fs"]
    )

    def frob_to_identity(Q):
        n = Q.shape[-1]
        return np.linalg.norm(np.asarray(Q) - np.eye(n), axis=(-2, -1))

    raw_score = float(np.median(frob_to_identity(aligned_data["Fs"])))
    nn_score = float(np.median(frob_to_identity(nn_flats)))
    assert nn_score < raw_score, (
        f"NN flattening ({nn_score:.3f}) should beat raw theta ({raw_score:.3f})"
    )

    evalues_nn = np.asarray(jax.vmap(get_eigenvalues)(nn_flats)).ravel()
    log_evalues = np.log(np.clip(evalues_nn, 1e-12, None))
    assert np.median(np.abs(log_evalues)) < np.log(10.0), (
        "Flattened eigenvalues are not within an order of magnitude of 1"
    )

    mdl_flats, _ = check_flattening(
        analyzed_equations["mdl_coords"],
        X=aligned_data["X"],
        Fs=aligned_data["Fs"],
    )
    assert mdl_flats.shape == aligned_data["Fs"].shape
