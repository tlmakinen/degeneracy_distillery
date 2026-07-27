#!/usr/bin/env python
"""Re-run the regrouping step of a completed run under the fixed
signed flatness acceptance test (commit 4514c64).

Training / flattening / SR are untouched; only regroup_like_terms and
everything downstream of it is recomputed.
"""
import argparse, json, os, pickle, shutil, sys, datetime

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # login node has no usable GPU
import numpy as np

from degeneracy_distillery.align_coords import load_and_process_data_v2
from degeneracy_distillery.postprocess_new import regroup_like_terms
from degeneracy_distillery.postprocessing_utils import check_flattening
from degeneracy_distillery.sr_utils import expressions_to_physical

# Per-problem arguments, transcribed verbatim from each scripts/*_notebook_run.py.
# These genuinely differ between problems (align_mode, canonicalize,
# Fisher_to_flatten, snap tolerances), so a single generic recipe is wrong.
PROBLEMS = {
    "gw": dict(
        srdir="sr_results_gw", npz="gw_flatten.npz",
        mask=dict(col=1, fallback_all=True),
        align=dict(process_ensemble=True, n_d=1.0, align_mode="procrustes",
                   separate_nonlinearity=True, canonicalize="permute_and_sign",
                   use_prior_normalization=True, restore_reference_mean=True,
                   Fisher_to_flatten="best"),
        regroup=dict(method="atoms", do_snap=True, snap_rel_tol=0.1,
                     snap_flat_tol=0.1, do_inner_snap=True, decimal=1, threshold=0.1),
        theta_names=("m1", "m2"),
    ),
    "imr": dict(
        srdir="sr_results_imr", npz="imr_flatten.npz",
        mask=dict(col=1, fallback_all=True),
        align=dict(process_ensemble=True, n_d=1.0, align_mode="procrustes",
                   separate_nonlinearity=True, canonicalize="permute_and_sign",
                   use_prior_normalization=True, restore_reference_mean=True,
                   Fisher_to_flatten="best"),
        regroup=dict(method="atoms", do_snap=True, snap_rel_tol=0.1,
                     snap_flat_tol=0.1, do_inner_snap=True,
                     inner_snap_rel_tol=0.1, inner_snap_flat_tol=0.1,
                     inner_snap_decimal=3, decimal=1, threshold=0.1),
        theta_names=("m1", "m2"),
    ),
    "ising": dict(
        srdir="sr_results_ising", npz="ising_flatten.npz",
        mask=None,
        align=dict(process_ensemble=True, n_d=1.0, align_mode="procrustes",
                   separate_nonlinearity=True, canonicalize="permute_and_sign",
                   use_prior_normalization=True, restore_reference_mean=True,
                   Fisher_to_flatten="average"),
        regroup=dict(method="atoms", do_snap=True, snap_rel_tol=0.2,
                     snap_flat_tol=0.2, decimal=2, threshold=0.5),
        theta_names=("J", "T", "h"),
    ),
    "kolmogorov": dict(
        srdir="sr_results_kolmogorov", npz="kolmogorov_flatten.npz",
        mask=None,
        align=dict(process_ensemble=True, n_d=1.0, align_mode="procrustes",
                   separate_nonlinearity=True, canonicalize="permute_and_sign",
                   use_prior_normalization=True, restore_reference_mean=True,
                   Fisher_to_flatten="average"),
        regroup=dict(method="atoms", do_snap=True, snap_rel_tol=0.2,
                     snap_flat_tol=0.2, decimal=2, threshold=0.5),
        theta_names=("f0", "nu"),
    ),
    "kuramoto": dict(
        srdir="sr_results_kuramoto", npz="kuramoto_flatten.npz",
        mask=None,
        align=dict(process_ensemble=True, n_d=1.0, align_mode="procrustes",
                   separate_nonlinearity=True, canonicalize="permute_and_sign",
                   use_prior_normalization=True, restore_reference_mean=True,
                   Fisher_to_flatten="average"),
        regroup=dict(method="atoms", do_snap=True, snap_rel_tol=0.2,
                     snap_flat_tol=0.2, decimal=2, threshold=0.5),
        theta_names=("K", "sigma", "D"),
    ),
    "qm7b": dict(
        srdir="sr_results_qm7b", npz="qm7b_flattening.npz",
        mask=dict(col=0, fallback_all=False),
        align=dict(process_ensemble=True, n_d=1.0, align_mode="kabsch",
                   separate_nonlinearity=False, canonicalize="sign_only",
                   use_prior_normalization=True, restore_reference_mean=False,
                   Fisher_to_flatten="average"),
        regroup=dict(method="atoms", do_snap=True, snap_rel_tol=0.1,
                     snap_flat_tol=0.1, decimal=1, threshold=0.05),
        theta_names=("atom_e", "exc_e", "abs_int", "homo", "lumo"),
    ),
    "rayleigh_benard": dict(
        srdir="sr_results_rayleigh_benard", npz="rayleigh_benard_flatten.npz",
        mask=None,
        align=dict(process_ensemble=True, n_d=1.0, align_mode="procrustes",
                   separate_nonlinearity=True, canonicalize="permute_and_sign",
                   use_prior_normalization=True, restore_reference_mean=True,
                   Fisher_to_flatten="average"),
        regroup=dict(method="atoms", do_snap=True, snap_rel_tol=0.2,
                     snap_flat_tol=0.2, decimal=2, threshold=0.5),
        theta_names=("logRa", "logPr", "logGamma"),
    ),
    "rosen": dict(
        srdir="sr_results_rosen", npz="rosen_flatten.npz",
        mask=None,
        align=dict(process_ensemble=True, n_d=1.0, align_mode="procrustes",
                   separate_nonlinearity=True, canonicalize="sign_only",
                   use_prior_normalization=True, restore_reference_mean=False,
                   Fisher_to_flatten="average"),
        regroup=dict(method="atoms", do_snap=True, snap_rel_tol=0.5,
                     snap_flat_tol=0.5, decimal=2, threshold=2.0),
        theta_names=("theta1", "theta2"),
    ),
    "sir": dict(
        srdir="sr_results_sir", npz="sir_flatten.npz",
        mask=dict(col=0, fallback_all=False),
        align=dict(process_ensemble=True, n_d=1.0, align_mode="procrustes",
                   separate_nonlinearity=False, canonicalize="sign_only",
                   use_prior_normalization=True, restore_reference_mean=False,
                   Fisher_to_flatten="average"),
        regroup=dict(method="atoms", do_snap=True, snap_rel_tol=0.1,
                     snap_flat_tol=0.1, decimal=2, threshold=1.0),
        theta_names=("beta", "gamma", "I0_over_10"),
    ),
}

FIX_COMMIT = "4514c64"

# ref_flat must reproduce the stored value, but the original runs executed on
# GPU in float32 while this recomputation runs on CPU, so exact equality is not
# achievable. Observed noise across 84 runs peaks at 2.1e-04 relative and
# scales with |ref_flat|. A genuinely wrong reconstruction (wrong align seed,
# wrong align_subsample, missing mask) shifts ref_flat by O(1), i.e. four
# orders of magnitude above this gate.
REF_FLAT_TOL = 1e-3

# Fallback for runs with no config_manifest.json, transcribed from the CONFIGS
# table of the corresponding scripts/*_notebook_run.py. Only consulted when the
# run directory itself does not record align_subsample.
ALIGN_SUBSAMPLE_BY_MODE = {
    "rayleigh_benard": {"smoke": 1000, "full": 4000},
}


class _Scaler:
    """Minimal stand-in: expressions_to_physical only reads scale_ / min_."""
    def __init__(self, scale, min_):
        self.scale_ = np.asarray(scale, dtype=float)
        self.min_ = np.asarray(min_, dtype=float)


def _median_frob(flats, n_params):
    identity = np.eye(n_params)
    return float(np.median(np.linalg.norm(np.asarray(flats) - identity, axis=(-2, -1))))


def run_one(rundir, problem, dry_run=False):
    spec = PROBLEMS[problem]
    rundir = os.path.abspath(rundir)
    pkl_path = os.path.join(rundir, spec["srdir"], "sr_expressions.pkl")

    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)

    # --- provenance inputs: align seed and align_subsample -------------------
    manifest_p = os.path.join(rundir, "config_manifest.json")
    summary_p = os.path.join(rundir, "run_summary.json")
    record_p = os.path.join(rundir, "run_record.json")

    align_seed = num_samps = None
    if os.path.exists(manifest_p):
        man = json.load(open(manifest_p))
        align_seed = man.get("stage_seeds", {}).get("align")
        num_samps = man.get("config", {}).get("align_subsample")
    if (align_seed is None or num_samps is None) and os.path.exists(summary_p):
        doc = json.load(open(summary_p))
        if align_seed is None:
            align_seed = doc.get("seeds", {}).get("align")
        if num_samps is None:
            # Runs written by the rebuttal_discovery launchers have no
            # config_manifest.json, so align_subsample is not stored anywhere in
            # the run directory. Fall back to the CONFIGS table of the script
            # that produced the run, keyed by the mode it recorded.
            num_samps = ALIGN_SUBSAMPLE_BY_MODE.get(problem, {}).get(doc.get("mode"))
    if align_seed is None or num_samps is None:
        raise SystemExit(f"{rundir}: could not resolve align seed / align_subsample "
                         f"(seed={align_seed}, num_samps={num_samps})")

    aligned = load_and_process_data_v2(
        datapath=rundir + os.sep, filename=spec["npz"],
        num_samps=num_samps, seed=align_seed, verbose=False, **spec["align"]
    )

    # Reproduce the run script's post-alignment masking. This is NOT optional:
    # gw/imr/qm7b/sir all subset X and Fs before regrouping, so skipping it
    # would silently score against a different dataset.
    X = np.asarray(aligned["X"])
    Fs = np.asarray(aligned["Fs"])
    mspec = spec["mask"]
    if mspec is not None:
        m = X[:, mspec["col"]] > 0.0
        if not m.any():
            if not mspec["fallback_all"]:
                raise SystemExit(f"{rundir}: alignment mask empty and no fallback")
            m = np.ones(len(X), dtype=bool)
        X, Fs = X[m], Fs[m]
    n_params = X.shape[1]

    old_pi = saved["prune_info"]
    old_pruned_exprs = saved["pruned_exprs"]
    old_physical = saved.get("physical_exprs")
    old_flatness = dict(saved.get("flatness", {}))

    pruned_exprs, rotation, prune_info = regroup_like_terms(
        saved["mdl_coords"], X=X, Fs=Fs,
        n_params=n_params, **spec["regroup"]
    )

    # Self-check: ref_flat is the score of the *unrotated* mdl_coords and is
    # unaffected by the bug, so it must reproduce the stored value up to
    # float noise. A mismatch means X/Fs were rebuilt wrongly.
    ref_old, ref_new = float(old_pi["ref_flat"]), float(prune_info["ref_flat"])
    ref_err = abs(ref_new - ref_old) / max(abs(ref_old), 1e-30)
    if ref_err > REF_FLAT_TOL:
        raise SystemExit(
            f"{rundir}: ref_flat mismatch ({ref_old:.8f} -> {ref_new:.8f}, "
            f"rel err {ref_err:.2e}); refusing to write. Coordinate "
            f"reconstruction does not match the original run."
        )

    pruned_flats, _ = check_flattening(pruned_exprs, X=X, Fs=Fs)
    new_pruned_score = _median_frob(pruned_flats, n_params)
    new_cond = float(np.median(np.linalg.cond(np.asarray(pruned_flats))))

    scaler = _Scaler(saved["scaler_scale"], saved["scaler_min"])
    physical_exprs = expressions_to_physical(
        pruned_exprs, scaler, sr_offset=float(saved.get("sr_offset", 0.0)),
        theta_names=spec["theta_names"], decimal=3,
    )

    result = dict(
        rundir=rundir, problem=problem, align_seed=align_seed, num_samps=num_samps,
        ref_flat_check=ref_err,
        old=dict(ref_flat=old_pi["ref_flat"], post_flat=old_pi["post_flat"],
                 final_flat=old_pi.get("final_flat"), rel_delta=old_pi.get("rel_delta"),
                 rotation_accepted=old_pi["rotation_accepted"],
                 pruned_median_frob=old_flatness.get("pruned"),
                 exprs=[str(e) for e in old_pruned_exprs],
                 physical=[str(e) for e in (old_physical or [])]),
        new=dict(ref_flat=prune_info["ref_flat"], post_flat=prune_info["post_flat"],
                 final_flat=prune_info.get("final_flat"),
                 rel_delta=prune_info.get("rel_delta"),
                 rotation_accepted=prune_info["rotation_accepted"],
                 pruned_median_frob=new_pruned_score,
                 exprs=[str(e) for e in pruned_exprs],
                 physical=[str(e) for e in physical_exprs]),
    )

    if dry_run:
        return result

    # --- write back ---------------------------------------------------------
    for p in (pkl_path, record_p, summary_p):
        if os.path.exists(p) and not os.path.exists(p + ".prefix_backup"):
            shutil.copy2(p, p + ".prefix_backup")

    saved["pruned_exprs"] = pruned_exprs
    saved["physical_exprs"] = physical_exprs
    saved["rotation"] = rotation
    saved["prune_info"] = prune_info
    flat = dict(saved.get("flatness", {}))
    flat["pruned"] = new_pruned_score
    if "median_condition_symbolic" in flat:
        flat["median_condition_symbolic"] = new_cond
    saved["flatness"] = flat
    saved["postprocess_fix"] = dict(
        commit=FIX_COMMIT, rerun_utc=datetime.datetime.utcnow().isoformat() + "Z",
        side="post-fix",
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(saved, f)

    for p in (record_p, summary_p):
        if not os.path.exists(p):
            continue
        doc = json.load(open(p))
        hg = doc.get("heldout_geometry")
        if isinstance(hg, dict):
            for key in ("pruned", "frob_symbolic"):
                if key in hg:
                    hg[key] = new_pruned_score
            if "median_condition_symbolic" in hg:
                hg["median_condition_symbolic"] = new_cond
            # Task 5: surface the acceptance decision in aggregated results.
            hg["rotation_accepted"] = bool(prune_info["rotation_accepted"])
            hg["rotation_rel_delta"] = float(prune_info["rel_delta"])
        doc["postprocess_fix"] = saved["postprocess_fix"]
        with open(p, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir")
    ap.add_argument("--problem", choices=sorted(PROBLEMS))
    ap.add_argument("--manifest", help="JSON list of {rundir, problem, sweep}")
    ap.add_argument("--sweep", help="process only this sweep from --manifest")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json-out")
    a = ap.parse_args()

    if a.manifest:
        import traceback
        items = json.load(open(a.manifest))
        if a.sweep:
            items = [i for i in items if i["sweep"] == a.sweep]
        if not items:
            raise SystemExit(f"no runs in manifest for sweep {a.sweep!r}")
        out = []
        for i, it in enumerate(items, 1):
            print(f"[{i}/{len(items)}] {it['sweep']} {it['rundir']}", flush=True)
            try:
                r = run_one(it["rundir"], it["problem"], dry_run=a.dry_run)
                r["status"] = "ok"
            except SystemExit as e:
                r = dict(rundir=it["rundir"], problem=it["problem"],
                         status="refused", error=str(e))
                print("   REFUSED:", e, flush=True)
            except Exception as e:
                r = dict(rundir=it["rundir"], problem=it["problem"], status="error",
                         error=f"{type(e).__name__}: {e}",
                         tb=traceback.format_exc()[-1500:])
                print("   ERROR:", type(e).__name__, e, flush=True)
            r["sweep"] = it["sweep"]
            out.append(r)
            if a.json_out:
                with open(a.json_out, "w") as f:
                    json.dump(out, f, indent=1, default=str)
        ok = sum(1 for r in out if r.get("status") == "ok")
        print(f"\ndone: {ok}/{len(out)} ok", flush=True)
        raise SystemExit(0 if ok == len(out) else 1)

    if not (a.rundir and a.problem):
        raise SystemExit("need --rundir and --problem, or --manifest")
    res = run_one(a.rundir, a.problem, dry_run=a.dry_run)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(res, f, indent=1, default=str)
    o, n = res["old"], res["new"]
    print(f"{res['problem']} {res['rundir']}")
    print(f"  align_seed={res['align_seed']} num_samps={res['num_samps']}")
    print(f"  OLD ref={o['ref_flat']:.4f} post={o['post_flat']:.4f} "
          f"final={o['final_flat']} accepted={o['rotation_accepted']} "
          f"pruned_median={o['pruned_median_frob']}")
    print(f"  NEW ref={n['ref_flat']:.4f} post={n['post_flat']:.4f} "
          f"final={n['final_flat']} accepted={n['rotation_accepted']} "
          f"pruned_median={n['pruned_median_frob']:.4f}")


if __name__ == "__main__":
    main()
