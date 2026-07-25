# NeurIPS Discovery Reruns: Cluster Handoff

## Goal

Demonstrate that the Degeneracy Distillery reliably recovers scientifically meaningful nonlinear coordinates from small training budgets, rather than succeeding only for selected runs.

The primary deliverable is a compact, rebuttal-ready summary across independent trials:

> Across 10 independent end-to-end trials using 500 training simulations each, the augmented pipeline recovered the expected nonlinear scientific coordinate in \(X/Y\) runs. Symbolic-regression augmentation used 2,000 inexpensive evaluations of the learned coordinate map and required no additional simulator calls.

## Experiment priority

Core:

1. Rosenbrock
2. SIR
3. GW TaylorF2
4. GW IMRPhenomD
5. QM7b

Optional, after the core results are secure:

6. Reynolds-number discovery
7. Weak-lensing \(S_8\)

Reynolds is a useful, broadly recognizable positive control if its existing external cluster implementation is genuinely ready. Treat it as recovery of a known governing dimensionless variable, not as a novel physical discovery.

Weak lensing is supporting evidence rather than a critical rerun. It is scientifically relevant, but overlaps with the known-coordinate recovery story already supplied by SIR and TaylorF2. Include it only if the discovery pipeline can be rerun cleanly without delaying the core aggregation.

## Existing starting points

- Rosenbrock: `scripts/rosenbrock_notebook_run.py`
- SIR tuned reference: `tutorial_notebooks/sir_example.ipynb`
  - There is currently no `sir_notebook_run.py`; create a batch-safe wrapper from the tuned notebook.
- GW TaylorF2: `scripts/gw_notebook_run.py`
- GW IMRPhenomD: `scripts/imrphenomd_notebook_run.py`
- QM7b: use the existing cluster implementation.
  - The local repository contains only dataset-training infrastructure in `degeneracy_distillery/training_loop_fishnets_dataset.py`, not a tuned QM7b rerun configuration.
- Reynolds: use the external cluster implementation; no Reynolds experiment exists in this repository.
- Weak lensing:
  - Discovery notebooks/artifacts: `plots/wl_w0wa_Omega_c_sigma8_flattening.ipynb`
  - Inference sweep: `scripts/wl_2d_nsims_logprob_sweep.py`
  - The inference sweep injects existing learned coordinates; it is not itself an end-to-end rediscovery run.

Relevant reusable utilities:

- SR and MDL: `degeneracy_distillery/sr_utils.py`
- Symbolic diagnostics and held-out flattening: `degeneracy_distillery/postprocessing_utils.py`
- Symbolic regrouping/pruning: `degeneracy_distillery/postprocess_new.py`
- Ensemble alignment/canonicalization: `degeneracy_distillery/align_coords.py`
- Fisher diagnostics: `degeneracy_distillery/diagnostics.py`
- Rosenbrock integration reference: `tests/test_rosenbrock_pipeline.py`

## Experimental protocol

### Independent trials

Run 10 independent end-to-end trials per core experiment using master seeds `0` through `9`.

Each master seed must deterministically control:

- Training simulation or dataset split
- Evaluation simulation or held-out split
- Fishnet initialization and training
- Flattener initialization and training
- Alignment subsampling
- SR augmentation draws
- Symbolic-regression search
- Downstream NPE training, where run

Do not hold the simulated dataset fixed while changing only the network seed.

The current scripts contain several hard-coded stage seeds. Expose them through a master-seed mechanism before launching the array.

### Frozen configuration

1. Run `seed=0` as a plumbing pilot.
2. Confirm outputs and metric collection.
3. Freeze all architecture, optimizer, SR operator, pruning, complexity, and stopping settings.
4. Launch seeds `0` through `9` without per-seed retuning.
5. Record all failures rather than silently rerunning with favorable settings.

Use the configurations already tuned in the notebooks/scripts. Add a rebuttal configuration with:

- `n_train = 500` parameter-data pairs for discovery
- A separate held-out evaluation set
- `n_aug = 2,000` fresh prior draws for SR augmentation, unless a frozen existing configuration requires another value
- Existing ensemble sizes and training settings

## Augmented symbolic-regression stage

After learning the neural coordinate ensemble \(\eta=f(\theta)\):

1. Draw 2,000 fresh \(\theta\) values from the prior.
2. Evaluate every aligned coordinate-ensemble member at those points.
3. Construct \((\theta,\bar{\eta},\delta\eta)\).
4. Run SR on this dense augmented dataset.

These are cheap neural-network evaluations and must not be counted as simulator calls.

Save the original and augmented sample counts separately.

## Simulation accounting

Record these quantities separately for every run:

- Training simulator calls used for discovery
- Held-out simulator calls used only for evaluation
- PCA-basis waveform generations
- Augmented \((\theta,\eta,\delta\eta)\) evaluations
- Downstream NPE training simulations
- Fishnet, flattening, alignment, SR, NPE, and total wall time

Use the phrase **500 training simulations**, not **500 total simulator calls**, when a separate simulated test set is used.

For both GW experiments, PCA waveform generation counts as simulator use. Cache noiseless waveforms so the same waveform is not regenerated for PCA fitting and noisy-data construction. If a fixed PCA basis is reused, label its construction as shared preprocessing and report its cost separately.

For QM7b, report numbers of training and held-out paired observations rather than simulator calls. Describe it as a paired graph/property problem with no evaluable likelihood unless its external implementation genuinely defines an intractable generative likelihood.

## Per-experiment discovery criteria

Predefine these criteria before inspecting the multi-seed results. Do not compare raw expression strings. Simplify/canonicalize expressions and compare their numerical functions or Jacobians while allowing component permutation, sign, scaling, and algebraic equivalence.

### Rosenbrock

Success requires:

- A quadratic degeneracy-resolving direction numerically equivalent to the analytic Rosenbrock coordinate, up to convention
- A complementary independent approximately linear direction

Report:

- Maximum held-out absolute correlation with the analytic target
- Jacobian alignment
- Held-out geometric loss
- Expression complexity/MDL
- Recovery count out of 10

The current `expression_correlations` and `validate_flatness` functions in `scripts/rosenbrock_notebook_run.py` provide a starting point.

### SIR

Success requires at least one discovered component strongly aligned with

\[
R_0=\frac{\beta}{\gamma}.
\]

Report:

- Held-out Pearson and Spearman correlation with \(R_0\)
- Gradient cosine similarity with \(\nabla R_0\)
- Held-out geometric improvement
- Expression complexity/MDL
- Recovery count out of 10
- Fixed-budget \(R_0\) CRPS
- Calibration/coverage alongside CRPS

CRPS is not currently implemented as a reusable repository utility; port the calculation from the updated SIR analysis or implement it in the batch aggregation code.

### GW TaylorF2

Evaluate the discovered coordinate set against the conventional \((\mathcal M_c,q)\) basis, allowing coordinate mixing.

Report:

- Canonical correlation or Jacobian-subspace alignment with \((\mathcal M_c,q)\)
- Recovery frequency of chirp-mass-like structure
- Held-out geometric loss for raw masses, learned neural coordinates, symbolic coordinates, and \((\mathcal M_c,q)\)
- Expression complexity/MDL

Reuse the physics-correlation and flatness machinery in `scripts/gw_notebook_run.py`.

### GW IMRPhenomD

Predefine success as:

- A strong total-mass-like component
- A complementary asymmetric mass component
- Better held-out geometric loss than \((\mathcal M_c,q)\)

Report correlations or Jacobian alignment with:

\[
M=m_1+m_2,\qquad \Delta m=m_1-m_2,\qquad \mathcal M_c,\qquad q.
\]

Score the coordinate set, not one exact symbolic string. Report how often the total-mass/asymmetry structure appears and how often it outperforms conventional coordinates geometrically.

### QM7b

Before launching, document in the run manifest:

- Exact definitions of \(\theta\) and graph-valued \(x\)
- The expected or hypothesized chemical relation
- A predeclared criterion for successful discovery
- The graph model and dataset split
- Whether the likelihood is truly intractable or simply unavailable

Report:

- Cross-split/seed stability
- Held-out coordinate or Jacobian agreement
- Symbolic complexity
- Predictive or geometric improvement
- Recovery count using the predeclared chemical criterion

Do not call QM7b an intractable-likelihood simulator unless that claim is technically justified by the cluster implementation.

### Reynolds number

If the external implementation is ready, test recovery of a coordinate equivalent to

\[
\mathrm{Re}=\frac{\rho vL}{\mu}.
\]

Use numerical function/Jacobian equivalence rather than string matching. Report it as a controlled recovery of a familiar dimensionless governing variable.

### Weak-lensing \(S_8\)

If run, test recovery of an \(S_8\)-like family

\[
S_8=\sigma_8\left(\frac{\Omega_m}{0.3}\right)^\alpha.
\]

Report:

- Recovered exponent \(\alpha\) and its dispersion across trials
- Alignment with \(S_8\)
- Held-out geometric improvement
- Recovery count

If using one fixed cosmological simulation suite, describe the trials as independent subsampling or split trials, not fresh end-to-end simulator trials.

## Common per-run output

Write one machine-readable JSON record per experiment and seed. Include:

```json
{
  "run_id": "sir_seed3",
  "problem": "sir",
  "master_seed": 3,
  "status": "success",
  "counts": {
    "n_train_simulations": 500,
    "n_eval_simulations": 500,
    "n_pca_simulations": 0,
    "n_augmented_coordinate_evaluations": 2000,
    "n_downstream_npe_simulations": 500
  },
  "discovery": {
    "expressions_physical": [],
    "expressions_canonical": [],
    "success": true,
    "physics_alignment": 0.97,
    "mdl_total": 0.0,
    "complexity_total": 0,
    "symbolically_invertible": true,
    "rank_deficient": false
  },
  "heldout_geometry": {
    "frob_raw": 0.0,
    "frob_neural": 0.0,
    "frob_symbolic": 0.0,
    "median_condition_raw": 0.0,
    "median_condition_symbolic": 0.0
  },
  "inference": {
    "crps_theta": null,
    "crps_eta": null,
    "coverage_error_theta": null,
    "coverage_error_eta": null
  },
  "runtime_seconds": {
    "fishnets": 0.0,
    "flatten": 0.0,
    "alignment": 0.0,
    "symbolic_regression": 0.0,
    "npe": 0.0,
    "total": 0.0
  }
}
```

Save:

- Per-run JSON
- Combined CSV/JSON
- Final symbolic expressions
- SR Pareto fronts and MDL analysis
- Flattening artifacts
- Configuration manifest
- Logs and failure reason

## Common metrics

Primary discovery metrics:

1. Structural recovery rate, reported as a numerator and denominator
2. Median physics alignment with interquartile range
3. Median held-out geometric improvement
4. Median expression complexity or MDL
5. Exact simulation and augmentation counts

Secondary downstream metrics:

- CRPS
- Marginal coverage deviation or TARP
- Validation log probability in target coordinates
- Figure of merit, where scientifically meaningful

Sharper CRPS must be accompanied by calibration to rule out overconfidence.

Useful existing functions include:

- `analyze_equations` / `compute_DL`
- `check_flattening`
- `diagnose_coordinate_rank_deficiency`
- `check_symbolic_invertibility`
- `expressions_to_physical`
- `regroup_like_terms`
- `load_and_process_data_v2`
- `diagnose_low_information`

TARP currently exists only in plotting notebooks through the external `tarp` package. Marginal coverage diagnostics already exist in the NPE sweep scripts.

## Cluster execution

Use Slurm arrays over experiments and master seeds.

Suggested staging:

1. GPU job: simulation/data preparation, Fishnet ensemble, flattener ensemble, alignment artifacts
2. Dependent CPU job: augmented coordinate evaluation, SR, post-processing, held-out metrics
3. Optional GPU job: paired downstream NPE/CRPS/calibration evaluation
4. CPU aggregation job after all arrays finish

Existing cluster conventions:

- Generic NPE runner: `scripts/slurm_gpu_sweep.sh`
- Common variables: `REPO_DIR`, `VENV_DIR`, `OUT_BASE`, `MODE`, `TARGET`
- Existing targets cover Rosenbrock, SIR, GW TaylorF2, GW IMRPhenomD, WL, and heater

Use unique output directories such as:

```text
results/rebuttal_discovery/<experiment>/seed_<seed>/
```

Do not overwrite existing paper artifacts.

## Aggregation and rebuttal deliverables

Aggregate all 10 trials without cherry-picking:

- Median and interquartile range
- Recovery count \(X/10\)
- Every failed run and failure stage
- A representative expression selected by a predefined rule, such as the median-success trial

Produce a compact summary with one row per experiment and these columns:

- Training simulations or paired observations
- Evaluation simulations or observations
- Augmented coordinate evaluations
- Successful recoveries
- Physics alignment
- Held-out geometric improvement
- Expression complexity
- CRPS/calibration where applicable

Also produce:

- One short methods paragraph describing independent trials and augmentation
- One concise result sentence per experiment
- A global simulation-accounting statement
- A rebuttal-ready plain-text/Markdown table

Do not rely on new figures: NeurIPS responses permit new textual results but no file uploads.

## Final checks

- Confirm all reported metrics use held-out data.
- Confirm the simulator-count language is exact.
- Confirm all stage seeds vary with the master seed.
- Confirm no per-seed retuning occurred.
- Confirm failures remain in the denominator.
- Confirm equivalent expressions are scored numerically/Jacobian-wise rather than by exact strings.
- Confirm QM7b likelihood language is technically correct.
- Confirm Reynolds comes from the actual external fluid-dynamics implementation, not the unrelated heater product example.
- Confirm WL is labeled as split robustness if fresh simulations were not generated.
