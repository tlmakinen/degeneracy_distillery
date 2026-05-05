"""Minimal end-to-end Degeneracy Distillery on the toy heater simulator.

This script runs the full Distillery pipeline on the rank-1 product-degeneracy
heater simulator from :mod:`degeneracy_distillery.product_degeneracy`:

    y(t) = (theta_1 * theta_2) * (1 - exp(-t / tau)) + noise(t)

so the only identifiable combination of (theta_1, theta_2) is the scalar
product ``theta_1 * theta_2``.  We train a small Fishnet ensemble
(``num_models=10``), fit a flattening flow with the SIR-notebook defaults,
preprocess, run PyOperon symbolic regression, and finally print the
discovered expressions (and a pruned version).

Run from the repository root, e.g.::

    python scripts/heater_minimal_distillery.py --seed 0 --nsims 1000

The script saves intermediate artefacts in ``--out-dir`` (default
``heater_distillery_run``) and prints the discovered eta coordinates.

Expected sanity-check outcome
-----------------------------
With ``theta = (V, I)`` drawn uniformly from a positive box and the data
depending only on the product ``P = V*I``, the Fisher information matrix is
*exactly rank-1* (up to noise injected by ``noise=1e-2``).  After flattening
+ varimax we therefore expect:

* one component to be a simple monotone function of the product, e.g.
  ``X1 * X2``, ``sqrt(X1*X2)``, or ``log(X1*X2)``;
* the other component to be unconstrained / dominated by the regularization,
  often a small expression that pruning can reduce to (near) zero or a
  trivial single-variable form.

After ``regroup_like_terms`` snapping the cleaned expression for the
identifiable axis should look essentially like ``c * X1 * X2`` for some
constant ``c``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0,
                        help="Master seed; controls JAX simulator and downstream RNGs.")
    parser.add_argument("--nsims", type=int, default=1000,
                        help="Number of training (and test) simulations.")
    parser.add_argument("--num-fishnets", type=int, default=10,
                        help="Fishnet ensemble size.")
    parser.add_argument("--out-dir", type=Path, default=Path("heater_distillery_run"),
                        help="Directory for intermediate artefacts.")
    parser.add_argument("--sr-time-limit", type=int, default=60 * 3,
                        help="PyOperon time budget in seconds.")
    parser.add_argument("--fishnet-epochs", type=int, default=5000)
    parser.add_argument("--flatten-epochs-phase1", type=int, default=1000)
    parser.add_argument("--flatten-epochs-phase2", type=int, default=500)
    parser.add_argument("--flatten-finetune-epochs", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. simulate (JAX, seed-controlled) ----------------------------------
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from degeneracy_distillery.product_degeneracy import (
        ToyConfig, make_dataset, simulator_heater,
    )

    cfg = ToyConfig()
    master_key = jr.PRNGKey(args.seed)
    train_key, test_key = jr.split(master_key)

    print(f"Simulating heater data: nsims={args.nsims}, n_t={cfg.n_t}, "
          f"theta in [{cfg.theta_min}, {cfg.theta_max}]^2")
    theta_train, data_train = make_dataset(args.nsims, cfg, train_key,
                                           simulator=simulator_heater)
    theta_test, data_test = make_dataset(args.nsims, cfg, test_key,
                                         simulator=simulator_heater)
    print(f"  theta_train: {theta_train.shape}    data_train: {data_train.shape}")
    print(f"  theta_test:  {theta_test.shape}     data_test:  {data_test.shape}")

    np.savez(out_dir / "heater_data.npz",
             theta_train=np.asarray(theta_train),
             data_train=np.asarray(data_train),
             theta_test=np.asarray(theta_test),
             data_test=np.asarray(data_test))

    # --- 2. fishnet ensemble -------------------------------------------------
    import flax.linen as nn
    from degeneracy_distillery.training_loop_fishnets import train_fishnets

    embedding_net = nn.Sequential([
        nn.Dense(64), nn.gelu,
        nn.Dense(32), nn.gelu,
    ])

    fishnets_dir = out_dir / "fishnets"
    print(f"\nTraining {args.num_fishnets} Fishnets -> {fishnets_dir}")
    _ = train_fishnets(
        theta_train, data_train,
        theta_test, data_test,
        num_models=args.num_fishnets,
        train_epochs=args.fishnet_epochs,
        patience=30,
        n_layers=[2, 5],
        hids_min=50,
        hids_max=300,
        embedding_net=embedding_net,
        lr=5e-5,
        train_batch_size=200,
        seed_model=201 + args.seed,
        seed_train=999 + args.seed,
        outdir=str(fishnets_dir),
    )

    # --- 3. flattening flow --------------------------------------------------
    from degeneracy_distillery.training_loop_flatten import fit_flattening

    fishnets_npz = np.load(fishnets_dir / "fishnets_outputs.npz")
    thetas = jnp.array(fishnets_npz["theta"])
    F_network_ensemble = jnp.array(fishnets_npz["Fs"])
    ensemble_weights = fishnets_npz["ensemble_weights"]
    print(f"\nLoaded fishnets ensemble: thetas={thetas.shape}, "
          f"Fs={F_network_ensemble.shape}, weights={ensemble_weights.shape}")

    flattening_prefix = str(out_dir / "heater_flattened")
    print(f"\nFitting flattening flow -> {flattening_prefix}.npz")
    fit_flattening(
        F_network_ensemble=F_network_ensemble,
        θs=thetas,
        ensemble_weights=ensemble_weights,
        flattener_activation="softplus",
        loss_type="log_frob",
        forward_backward_mlp=True,
        forward_backward_invertibility_weight=1.0,
        n_layers=5,
        offset=0.0,
        beta_det=0.1,
        noise=1e-2,
        batch_size=250,
        finetune_epochs=args.flatten_finetune_epochs,
        epochs_phase1=args.flatten_epochs_phase1,
        epochs_phase2=args.flatten_epochs_phase2,
        lr_phase1=2e-6,
        lr_schedule_initial=7e-5,
        lr_decay=0.3,
        l1_alpha=0.0,
        do_plot=False,
        seed=args.seed,
        output_prefix=flattening_prefix,
        Fisher_to_flatten="best",
    )

    # --- 4. preprocessing ----------------------------------------------------
    from degeneracy_distillery.align_coords import load_and_process_data_v2

    flattened_npz = Path(flattening_prefix + ".npz")
    print(f"\nLoading & preprocessing flattened coords from {flattened_npz}")
    data = load_and_process_data_v2(
        datapath=str(flattened_npz.parent) + os.sep,
        filename=flattened_npz.name,
        num_samps=min(4000, args.nsims),
        seed=44 + args.seed,
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
    X = data["X"]
    mask = X[:, 0] > 0.0
    X = X[mask]
    y = data["y"][mask]
    y_std = data["y_std"][mask]
    dy_sr = data["dy_sr"][mask]
    Fs = data["Fs"][mask]
    n_params = X.shape[1]
    print(f"  X: {X.shape}  y: {y.shape}  Fs: {Fs.shape}  n_params={n_params}")

    # --- 5. symbolic regression ---------------------------------------------
    from degeneracy_distillery.sr_utils import fit_and_analyze_sr

    sr_dir = out_dir / "sr_results_heater"
    sr_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRunning PyOperon SR for {args.sr_time_limit}s -> {sr_dir}")
    mdl_coords, frob_coords, _analysis, split_data = fit_and_analyze_sr(
        X, y, y_std, dy_sr, Fs,
        n_params=n_params,
        parent_dir=str(sr_dir) + os.sep,
        test_size=0.5,
        random_state=32134 + args.seed,
        shuffle=True,
        time_limit=args.sr_time_limit,
        max_length=25,
        max_depth=10,
        allowed_symbols="add,mul,div,pow,constant,variable,sqrt",
        max_complexity_thresh=20,
        equation_set="pareto",
    )

    print("\n" + "=" * 60)
    print("SYMBOLIC REGRESSION RAW RESULTS")
    print("=" * 60)
    print(f"\nBest MDL coordinates : {mdl_coords}")
    print(f"Best Frobenius coords: {frob_coords}")
    print("\nGround truth: one coord should approximate X1*X2 (the dissipated power)")

    # --- 6. print discovered expressions ------------------------------------
    import sympy

    from degeneracy_distillery.postprocessing_utils import print_discovered_expressions

    print("\n--- Simplified MDL coordinates (1-significant-figure constants) ---")
    print_discovered_expressions(
        [sympy.simplify(p).evalf(1) for p in mdl_coords]
    )

    # --- 7. pruning via atom regrouping --------------------------------------
    from degeneracy_distillery.postprocess_new import (
        analyze_atom_sharing, regroup_like_terms,
    )

    X_test = split_data["X_test"]
    Fs_test = split_data["Fs_test"]

    name_map = {"V": "X1", "I": "X2"}
    print("\n--- Atom-sharing report ---")
    _ = analyze_atom_sharing(mdl_coords)

    print("\n--- Pruning + snapping (regroup_like_terms, method='atoms') ---")
    pruned_exprs, _R, _info = regroup_like_terms(
        mdl_coords, X=X_test, Fs=Fs_test, n_params=n_params,
        method="atoms",
        do_snap=True, snap_rel_tol=0.1, snap_flat_tol=0.1,
        decimal=1,
        threshold=0.001,
    )
    print_discovered_expressions(pruned_exprs, name_map)

    print("\n--- Sympy-simplified pruned coordinates (1 sig-fig) ---")
    print_discovered_expressions(
        [sympy.simplify(p).evalf(1) for p in pruned_exprs], name_map,
    )


if __name__ == "__main__":
    main()
