# Correction: flatness acceptance test rejected improvements

Applies to every experiment that calls `regroup_like_terms`, which is all of
them: Rosenbrock, SIR, TaylorF2, IMRPhenomD, QM7b, Ising, Kolmogorov and
Kuramoto. Read this before aggregating any results produced before 2026-07-27,
and before comparing old runs against new ones.

## What was wrong

`degeneracy_distillery/postprocess_new.py` scores a candidate coordinate system
with `_flat_score`, which returns `mean ||flat - I||_F` over the dataset.
**Lower is better.** A score of 0 is a perfectly isotropic Fisher.

Three acceptance tests compared a candidate against the incumbent using the
*absolute* change:

```python
rel_delta = abs(score - ref) / max(ref, 1e-30)
accept = rel_delta < flat_tol
```

Because of the `abs`, a candidate that improved flatness by more than
`flat_tol` was rejected on exactly the same grounds as one that degraded it by
more than `flat_tol`. The default `flat_tol` is 0.1, so any rotation or snap
that improved the flatness score by more than 10 percent was discarded, and the
code fell back to the unrotated expressions while printing
`rotation rejected (flatness degraded too much)`.

The larger the improvement, the more certain the rejection. This preferentially
threw away the best rotations.

The three sites were the rotation acceptance in `regroup_like_terms` (around
line 1093), the shared-coefficient snap in `snap_shared_coefficients` (line
665), and the inner-factor snap (line 888).

## The fix

The comparison is now signed at all three sites:

```python
rel_delta = (score - ref) / max(ref, 1e-30)
accept = rel_delta < flat_tol
```

An improvement of any size is now accepted unconditionally. `flat_tol` retains
its intended meaning, which is a bound on how much flatness you are willing to
give up in exchange for the interpretability gained by regrouping terms. The
verbose log now prints a signed delta, so `rel Δ = -55.51%` means the rotation
improved flatness by 55 percent.

## Measured impact on the smoke runs

Checked against the `prune_info` dictionary pickled in every
`sr_results_*/sr_expressions.pkl`. Two of the four smoke runs were affected:

| Run | Flatness before | After rotation | Old outcome | Correct outcome |
|---|---|---|---|---|
| Kolmogorov | 2.299 | 1.023 | rejected | accept, 2.2x better |
| Ising | 368.2 | 107145.7 | rejected | reject (genuine degradation) |
| Kuramoto | 0.964 | 1.330 | rejected | reject (genuine degradation) |

A fourth case, an in-progress enzyme-kinetics experiment not yet committed, was
where this first surfaced: it wanted a rotation from 10.363 to 1.335, a 7.8x
improvement, and was refused.

Re-running `regroup_like_terms` on the saved Kolmogorov artifacts with the fix
confirms the rotation is now accepted and the log reports `rel Δ = -55.51%`. A
rank repair then fires and replaces a dependent coordinate, landing at a final
flatness of 1.044 against the original 2.299.

Note that the fix does not indiscriminately accept more rotations. Ising and
Kuramoto are still correctly rejected, because those rotations genuinely made
the geometry worse. The change only stops the code from discarding wins.

## What this means for the campaign

**Any completed run needs its postprocessing redone, not its training.** The bug
lives entirely downstream of the fishnets, the flattener, and the symbolic
regression search. Fisher estimates, flattening networks, and Pareto fronts are
all unaffected. So are `run_summary.json` fields under `heldout_geometry` that
report `raw_theta`, `nn`, and `mdl`, since those do not depend on the regrouping
step.

What does change is the `pruned` flatness score and the final regrouped
expressions, which are the ones you would paste into a rebuttal table.

## Recovering an existing run without retraining

Everything needed is already on disk in the run directory. Load `mdl_coords`
from the pickle, rebuild `X` and `Fs` from the saved `*_flatten.npz` using the
same alignment seed the script used, and call `regroup_like_terms` again. This
takes seconds on a CPU:

```python
import os, pickle
from degeneracy_distillery.align_coords import load_and_process_data_v2
from degeneracy_distillery.postprocess_new import regroup_like_terms

import json

OUT, PROBLEM = "results/.../kolmogorov/seed_0/", "kolmogorov"

with open(os.path.join(OUT, f"sr_results_{PROBLEM}", "sr_expressions.pkl"), "rb") as f:
    saved = pickle.load(f)
with open(os.path.join(OUT, "run_summary.json")) as f:
    align_seed = json.load(f)["seeds"]["align"]

aligned = load_and_process_data_v2(
    datapath=OUT, filename=f"{PROBLEM}_flatten.npz",
    num_samps=1000,          # align_subsample from the RunConfig used
    seed=align_seed,
    process_ensemble=True, n_d=1.0, align_mode="procrustes",
    separate_nonlinearity=True, canonicalize="permute_and_sign",
    use_prior_normalization=True, restore_reference_mean=True,
    Fisher_to_flatten="average", verbose=False,
)

pruned, rotation, info = regroup_like_terms(
    saved["mdl_coords"], X=aligned["X"], Fs=aligned["Fs"],
    n_params=aligned["X"].shape[1], method="atoms", do_snap=True,
    snap_rel_tol=0.2, snap_flat_tol=0.2, decimal=2, threshold=0.5,
)
print(info["ref_flat"], "->", info["post_flat"], info["rotation_accepted"])
```

Use the `align` seed recorded in that run's `run_summary.json`, not a fresh
one, or the coordinates will not match the ones the expressions were fit
against. Then re-run `expressions_to_physical` with the scaler parameters
stored in the same pickle to regenerate the physical expressions.

**Do not mix results across the fix.** If some seeds in a sweep were
postprocessed before and some after, the `pruned` column is not comparable
across them and any variance you compute over seeds is contaminated. Either
redo the postprocessing for the whole sweep or record which side of the fix
each run came from.

## Why it went unnoticed

The failure is silent and looks like a safety feature working. The printed
message says the rotation was rejected because flatness degraded, which reads
as the tolerance doing its job. Nothing in `run_summary.json` surfaces
`rotation_accepted`, so the only way to see it was to open the pickle or read
the verbose console output closely.

Worth considering for the future: promote `prune_info["rotation_accepted"]` and
the signed `rel_delta` into `run_summary.json` for every experiment, so a
rejected rotation is visible in the aggregated results rather than buried in a
pickle.
