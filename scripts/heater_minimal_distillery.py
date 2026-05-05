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

Resuming
--------
Each stage writes its outputs to ``--out-dir``, so you can resume from any
later stage with ``--start-from {sim,fishnets,flatten,preprocess,sr,prune}``::

    # rerun only the flattening + downstream steps, reusing the existing
    # fishnet ensemble:
    python scripts/heater_minimal_distillery.py --start-from flatten \
        --loss-type squared_frob_det --beta-det 0.1 --noise 1e-3

Tuning the flattener for rank-1 problems
----------------------------------------
This problem has a *structurally rank-1 Fisher* (only the product
``theta_1 * theta_2`` enters the data), so the ideal Jacobian satisfies
``|det J| -> 0`` everywhere -- not ``|det J| ~ 1``.  Two flags matter:

* ``--no-invertibility-mlp``  **strongly recommended for rank-deficient
  problems.**  The default ``forward_backward_mlp=True`` adds a
  ``mean((theta - theta_rec)^2)`` penalty that pulls ``J`` toward identity,
  which is the *opposite* of what a rank-1 problem wants.  Disabling it
  lets the flow collapse volume in the unidentifiable direction.

* ``--loss-type``: ``log_frob`` works once the inverse-MLP is off.  Its
  asymptotic value of ``~0.7`` is the structural rank-1 floor (set by
  ``||Q^{-1} - I||_F`` in the reweighted Frobenius), not a sign of bad
  training.  ``--loss-type squared_frob`` is a cleaner alternative for
  rank-1 cases since it omits the inverse term entirely.

Diagnostic rule of thumb: ``det F(eta) ~ 1`` is *not* the right target for
a rank-1 problem -- it usually means the flow is stuck near identity and
the eta coordinates collapse to ``theta``.  Trust the downstream
correlation table instead (one eta strongly correlated with the product
axis, one with the nuisance axis, ~zero cross-talk).

Expected sanity-check outcome
-----------------------------
With ``theta = (V, I)`` drawn uniformly from a positive box and the data
depending only on the product ``P = V*I``, the Fisher information matrix is
*exactly rank-1* (up to the Cholesky regularization).  After flattening +
varimax/Procrustes alignment we therefore expect:

* one component to be a simple monotone function of the product, e.g.
  ``X1 * X2``, ``sqrt(X1*X2)``, or ``log(X1*X2)``;
* the other component to be unconstrained / dominated by the regularization,
  often a small expression that pruning can reduce to (near) zero or a
  trivial single-variable form.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


STAGES = ("sim", "fishnets", "flatten", "preprocess", "sr", "prune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # --- pipeline control ---
    parser.add_argument("--seed", type=int, default=0,
                        help="Master seed; controls JAX simulator and downstream RNGs.")
    parser.add_argument("--out-dir", type=Path, default=Path("heater_distillery_run"),
                        help="Directory for intermediate artefacts.")
    parser.add_argument("--start-from", choices=STAGES, default="sim",
                        help="Skip earlier stages and resume from here. Each stage's "
                             "inputs are loaded from --out-dir.")

    # --- simulation ---
    parser.add_argument("--nsims", type=int, default=1000,
                        help="Number of training (and test) simulations.")
    parser.add_argument("--theta-min", type=float, default=1.0,
                        help="Lower bound of the (V, I) prior box.")
    parser.add_argument("--theta-max", type=float, default=2.0,
                        help="Upper bound of the (V, I) prior box.")
    parser.add_argument("--n-t", type=int, default=20,
                        help="Number of time samples per simulation (data dimensionality).")
    parser.add_argument("--t-max", type=float, default=4.0,
                        help="Observation horizon (in units of tau).")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Thermal time constant of the heater plant.")
    parser.add_argument("--sigma-heater", type=float, default=0.2,
                        help="Per-timepoint Gaussian observation noise stdev.")

    # --- fishnets ---
    parser.add_argument("--num-fishnets", type=int, default=10,
                        help="Fishnet ensemble size.")
    parser.add_argument("--fishnet-epochs", type=int, default=5000)

    # --- flattener ---
    parser.add_argument(
        "--loss-type",
        choices=("log_frob", "frob", "squared_frob", "squared_frob_det"),
        default="log_frob",
        help="Flattening loss form. For rank-1 Fishers (heater) prefer "
             "'squared_frob_det' to avoid the runaway ||Q^-1 - I||_F term.",
    )
    parser.add_argument("--beta-det", type=float, default=0.1,
                        help="Weight of the (log det Q)^2 barrier "
                             "(only used by loss_type='squared_frob_det').")
    parser.add_argument("--noise", type=float, default=1e-2,
                        help="Cholesky noise added to F per sample. Lower for "
                             "more accurate alignment, higher for stability.")
    parser.add_argument("--flatten-epochs-phase1", type=int, default=1000)
    parser.add_argument("--flatten-epochs-phase2", type=int, default=500)
    parser.add_argument("--flatten-finetune-epochs", type=int, default=200)
    parser.add_argument(
        "--no-invertibility-mlp", action="store_true",
        help="Disable the residual-MLP forward/backward invertibility constraint. "
             "Useful for rank-1 problems where requiring eta ~= theta pins the "
             "flow near identity and prevents real compression.",
    )
    parser.add_argument(
        "--invertibility-weight", type=float, default=1.0,
        help="Weight on the forward-backward MLP reconstruction penalty "
             "(theta - theta_rec)^2. Lower this (e.g. 0.1, 0.01) to let the "
             "flattener compress more aggressively. Ignored when "
             "--no-invertibility-mlp is set.",
    )
    parser.add_argument(
        "--align-mode",
        choices=("procrustes", "kabsch", "none"),
        default="procrustes",
        help="Ensemble-rotation mode used by load_and_process_data_v2.",
    )

    # --- symbolic regression ---
    parser.add_argument("--sr-time-limit", type=int, default=60 * 3,
                        help="PyOperon time budget in seconds.")

    return parser.parse_args()


def _stage_index(name: str) -> int:
    return STAGES.index(name)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start_idx = _stage_index(args.start_from)

    fishnets_dir = out_dir / "fishnets"
    flattening_prefix = str(out_dir / "heater_flattened")
    flattened_npz = Path(flattening_prefix + ".npz")
    sr_dir = out_dir / "sr_results_heater"
    split_data_path = sr_dir / "split_data.npz"
    data_npz = out_dir / "heater_data.npz"

    print(f"Resuming from stage: {args.start_from} (index {start_idx})")

    # --- 1. simulate (JAX, seed-controlled) ----------------------------------
    if start_idx <= _stage_index("sim"):
        import jax.random as jr

        from degeneracy_distillery.product_degeneracy import (
            ToyConfig, make_dataset, simulator_heater,
        )

        cfg = ToyConfig(
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            tau=args.tau,
            t_max=args.t_max,
            n_t=args.n_t,
            sigma_heater=args.sigma_heater,
        )
        master_key = jr.PRNGKey(args.seed)
        train_key, test_key = jr.split(master_key)

        print(f"\n[sim] nsims={args.nsims}, n_t={cfg.n_t}, "
              f"theta in [{cfg.theta_min}, {cfg.theta_max}]^2  "
              f"(prior factor {cfg.theta_max/cfg.theta_min:.1f}x per param, "
              f"{(cfg.theta_max/cfg.theta_min)**2:.1f}x for the product)")
        theta_train, data_train = make_dataset(args.nsims, cfg, train_key,
                                               simulator=simulator_heater)
        theta_test, data_test = make_dataset(args.nsims, cfg, test_key,
                                             simulator=simulator_heater)
        print(f"  theta_train: {theta_train.shape}    data_train: {data_train.shape}")
        print(f"  theta_test:  {theta_test.shape}     data_test:  {data_test.shape}")

        np.savez(data_npz,
                 theta_train=np.asarray(theta_train),
                 data_train=np.asarray(data_train),
                 theta_test=np.asarray(theta_test),
                 data_test=np.asarray(data_test))
    else:
        print(f"\n[sim] SKIPPED (expecting {data_npz})")

    # --- 2. fishnet ensemble -------------------------------------------------
    if start_idx <= _stage_index("fishnets"):
        import flax.linen as nn
        import jax.numpy as jnp
        from degeneracy_distillery.training_loop_fishnets import train_fishnets

        npz = np.load(data_npz)
        theta_train = jnp.asarray(npz["theta_train"])
        data_train = jnp.asarray(npz["data_train"])
        theta_test = jnp.asarray(npz["theta_test"])
        data_test = jnp.asarray(npz["data_test"])

        embedding_net = nn.Sequential([
            nn.Dense(64), nn.gelu,
            nn.Dense(32), nn.gelu,
        ])

        print(f"\n[fishnets] training {args.num_fishnets} Fishnets -> {fishnets_dir}")
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
    else:
        print(f"\n[fishnets] SKIPPED (expecting {fishnets_dir / 'fishnets_outputs.npz'})")

    # --- 3. flattening flow --------------------------------------------------
    if start_idx <= _stage_index("flatten"):
        import jax.numpy as jnp
        from degeneracy_distillery.training_loop_flatten import fit_flattening

        fishnets_npz = np.load(fishnets_dir / "fishnets_outputs.npz")
        thetas = jnp.array(fishnets_npz["theta"])
        F_network_ensemble = jnp.array(fishnets_npz["Fs"])
        ensemble_weights = fishnets_npz["ensemble_weights"]
        print(f"\n[flatten] loaded fishnets ensemble: thetas={thetas.shape}, "
              f"Fs={F_network_ensemble.shape}, weights={ensemble_weights.shape}")
        fwd_bwd_mlp = not args.no_invertibility_mlp
        print(f"[flatten] loss_type={args.loss_type}  beta_det={args.beta_det}  "
              f"noise={args.noise}")
        print(f"[flatten] forward_backward_mlp={fwd_bwd_mlp}  "
              f"invertibility_weight={args.invertibility_weight}")
        print(f"[flatten] writing -> {flattening_prefix}.npz")
        fit_flattening(
            F_network_ensemble=F_network_ensemble,
            θs=thetas,
            ensemble_weights=ensemble_weights,
            flattener_activation="softplus",
            loss_type=args.loss_type,
            forward_backward_mlp=fwd_bwd_mlp,
            forward_backward_invertibility_weight=args.invertibility_weight,
            n_layers=5,
            offset=0.0,
            beta_det=args.beta_det,
            noise=args.noise,
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
    else:
        print(f"\n[flatten] SKIPPED (expecting {flattened_npz})")

    # --- 4. preprocessing ----------------------------------------------------
    X = y = y_std = dy_sr = Fs = None
    n_params = 0
    if start_idx <= _stage_index("preprocess"):
        from degeneracy_distillery.align_coords import load_and_process_data_v2

        print(f"\n[preprocess] loading flattened coords from {flattened_npz}")
        print(f"[preprocess] align_mode={args.align_mode}")
        data = load_and_process_data_v2(
            datapath=str(flattened_npz.parent) + os.sep,
            filename=flattened_npz.name,
            num_samps=min(4000, args.nsims),
            seed=44 + args.seed,
            process_ensemble=True,
            n_d=1.0,
            align_mode=args.align_mode,
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
    else:
        print("\n[preprocess] SKIPPED")

    # --- 5. symbolic regression ---------------------------------------------
    mdl_coords = frob_coords = None
    split_data = None
    if start_idx <= _stage_index("sr"):
        from degeneracy_distillery.sr_utils import (
            fit_and_analyze_sr, sr_structure_predicate,
        )

        sr_dir.mkdir(parents=True, exist_ok=True)
        # equation_predicate is applied strictly post-hoc inside
        # analyze_equations -- it does NOT affect PyOperon's search.
        sr_predicate = sr_structure_predicate(
            n_params=n_params,
            forbid_self_transcendental=True,
            check_nested_exp=False,
            forbid_x_in_pow_exponent=True,
        )
        print(f"\n[sr] running PyOperon for {args.sr_time_limit}s -> {sr_dir}")
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
            length_penalty=2.0,
            equation_predicate=sr_predicate,
        )

        print("\n" + "=" * 60)
        print("SYMBOLIC REGRESSION RAW RESULTS")
        print("=" * 60)
        print(f"\nBest MDL coordinates : {mdl_coords}")
        print(f"Best Frobenius coords: {frob_coords}")
        print("\nGround truth: one coord should approximate X1*X2 (the dissipated power)")
    else:
        # --start-from prune: re-derive mdl_coords from saved equations + split_data
        from degeneracy_distillery.sr_utils import (
            analyze_equations, sr_structure_predicate,
        )

        print(f"\n[sr] SKIPPED -- re-loading split_data + Pareto front from {sr_dir}")
        sd = np.load(split_data_path, allow_pickle=True)
        split_data = {k: sd[k] for k in sd.files}
        X_test = split_data["X_test"]
        y_test = split_data["y_test"]
        y_std_test = split_data["y_std_test"]
        dy_sr_test = split_data["dy_sr_test"]
        Fs_test = split_data["Fs_test"]
        n_params = X_test.shape[1]

        mdl_coords, frob_coords, _ = analyze_equations(
            X_test, y_test, y_std_test, dy_sr_test, Fs_test,
            parent_dir=str(sr_dir) + os.sep,
            n_params=n_params,
            equation_set="pareto",
            max_complexity_thresh=20,
            length_penalty=2.0,
            equation_predicate=sr_structure_predicate(
                n_params=n_params,
                forbid_self_transcendental=True,
                forbid_x_in_pow_exponent=True,
            ),
        )

    # --- 6. print discovered expressions ------------------------------------
    import sympy
    from degeneracy_distillery.postprocessing_utils import print_discovered_expressions

    print("\n--- Simplified MDL coordinates (1-significant-figure constants) ---")
    print_discovered_expressions(
        [sympy.simplify(p).evalf(1) for p in mdl_coords]
    )

    # --- 7. pruning via atom regrouping --------------------------------------
    if start_idx <= _stage_index("prune"):
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

        # --- 8. correlation with candidate physics axes ---------------------
        _print_physics_correlations(
            X_test, mdl_coords, pruned_exprs,
        )


def _print_physics_correlations(
    X_test: np.ndarray,
    mdl_coords,
    pruned_exprs,
) -> None:
    """Print Pearson |r| of each discovered eta against candidate physics axes.

    The truly identifiable axis is the dissipated power ``P = X1 * X2``.  The
    unidentifiable axis is ``log(X1 / X2)``.  Local approximations to the
    product (linear sum, log-sum, etc.) are included to make it obvious when
    SR settles for a local rather than global match.
    """
    from scipy.stats import pearsonr

    from degeneracy_distillery.postprocessing_utils import get_y_sr

    X = np.asarray(X_test)
    x1, x2 = X[:, 0], X[:, 1]
    candidates = {
        "X1*X2  (true identifiable axis)":           x1 * x2,
        "sqrt(X1*X2)":                                np.sqrt(np.clip(x1 * x2, 1e-12, None)),
        "log(X1*X2)":                                 np.log(np.clip(x1 * x2, 1e-12, None)),
        "X1+X2  (local approx of sqrt(X1*X2))":       x1 + x2,
        "log(X1/X2)  (true nuisance axis)":           np.log(np.clip(x1, 1e-12, None))
                                                      - np.log(np.clip(x2, 1e-12, None)),
        "X1-X2":                                      x1 - x2,
    }

    def _print_block(title: str, exprs) -> None:
        print(f"\n{title}")
        try:
            eta = np.asarray(get_y_sr(exprs, X))
        except Exception as err:  # pragma: no cover - defensive
            print(f"  (get_y_sr failed: {err})")
            return
        if eta.ndim == 1:
            eta = eta[:, None]
        n_eta = eta.shape[1]

        max_label = max(len(label) for label in candidates)
        header = " " * (max_label + 2) + "".join(
            f"  |r| η_{i+1:<3}" for i in range(n_eta)
        )
        print(header)
        print(" " * (max_label + 2) + "  -------" * n_eta)
        for label, vec in candidates.items():
            row = label.ljust(max_label) + "  "
            for i in range(n_eta):
                eta_i = eta[:, i]
                if not np.all(np.isfinite(eta_i)):
                    row += "    n/a "
                    continue
                r, _ = pearsonr(eta_i, vec)
                row += f"  {abs(r):>6.3f}"
            print(row)

    _print_block(
        "--- |Pearson r| of MDL coords vs candidate axes "
        "(closer to 1.0 = better match) ---",
        list(mdl_coords),
    )
    _print_block(
        "--- |Pearson r| of pruned coords vs candidate axes ---",
        list(pruned_exprs),
    )
    print(
        "\nExpected: one of the η's correlates very strongly (|r| > 0.99) with "
        "X1*X2, sqrt(X1*X2), or log(X1*X2) -- those are equivalent up to a\n"
        "monotone reparameterization. The orthogonal η should be poorly\n"
        "correlated with everything (it's the unidentifiable direction).\n"
    )


if __name__ == "__main__":
    main()
