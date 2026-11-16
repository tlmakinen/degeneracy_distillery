# Whittling inputs with the one-step loss: procedure and patch

Written 14 Nov 2026 for future use, in particular on CAMELS. The design
rationale and the full d = 100 results are in
`notes/iterative_whittle_design.md`; this file is the how-to.

## What it does

Given parameter–simulation pairs with many input parameters, it finds the
small set of inputs the data respond to, using the one-step loss (Eq. 6) and no
new simulations. Discovery then runs on that set. An input that is dropped
keeps varying in the simulations, so it is marginalised, as I₀ is in SIR.

## Procedure

One round:

1. **Probe.** Train a short one-step map with `m_probe` latent coordinates on
   the kept inputs plus decoy columns.
2. **Score.** For each input, the held-out Fisher sensitivity per unit of prior
   width: `s_j = (hi_j − lo_j) · sqrt( mean_θ Σ_a w_a (∂η_a/∂θ_j)² )`.
   With `--score all`, `w_a = 1` and this is `sqrt(diag(JᵀJ))` of the learned
   map. With `--score informative`, the latent axes are rotated to their
   principal directions and `w_a = 1 − exp(−2·info_a)` is the share of axis a's
   variance that the data predict, so an axis the data cannot see counts for
   nothing.
3. **Threshold.** The decoys never reached the simulator. An input scoring
   above `protect-factor` (2) times the highest decoy is "protected": the data
   respond to it, and it is never dropped.
4. **Cut.** Keep the top `keep-frac` of the inputs, plus every protected one.
5. **Guard.** If the next probe's total held-out information (sum of
   `info_nats` over axes) fell by more than `guard-nats`, undo the cut and
   repeat it more gently.

Rounds repeat until `k-target` inputs remain, or until every remaining input is
protected.

## The patch

No existing file was changed. New files:

| file | what |
|---|---|
| `scripts/iterative_whittle.py` | the procedure, for any registered problem |
| `scripts/rebuttal_checks/run_whittle_local.sh` | laptop launcher (sets the library path pyoperon needs on macOS) |
| `scripts/rebuttal_checks/summarise_whittle.py` | table of the banana trials |
| `scripts/rebuttal_checks/whittle_prototype.py` | the first one-seed prototype |

Reused from the package and the sweep driver: `train_oneshot`,
`OneShotFit.jac`, `info_nats`, `standardise_x`, `parse_problem_args`,
`get_problem`, and `run_oneshot_trial` for the optional discovery stage.
`SubsetProblem` in the script presents a problem with a subset of its parameter
columns to that driver.

## Running it

Laptop:

```bash
scripts/rebuttal_checks/run_whittle_local.sh --problem camels --skip-discovery \
  --skip-fill none --score informative --probe-repeats 3 --k-target 14 \
  --keep-frac 0.5 --guard-nats 0.5 --out-dir results/whittle_camels
```

Cluster:

```bash
mkdir -p logs
sbatch --job-name=whittle scripts/job.sbatch python scripts/iterative_whittle.py \
  --problem camels --skip-discovery --skip-fill none --score informative \
  --probe-repeats 3 --k-target 14 --keep-frac 0.5 --guard-nats 0.5 \
  --out-dir /data103/makinen/degeneracy_experiments/whittle_camels
```

Options that matter:

| option | default | meaning |
|---|---|---|
| `--problem`, `--problem-arg k=v` | `rosenbrock` | any problem in the registry |
| `--skip-discovery` | off | stop after whittling |
| `--k-target` | 5 | stop at this many inputs |
| `--keep-frac` | 0.25 | fraction kept per cut |
| `--score` | `all` | `informative` counts sensitivity only along latent axes the data predict; see below |
| `--skip-fill` | `decoys` | what sits in the skip-connection positions; see below |
| `--probe-repeats` | 1 | probes averaged per round |
| `--probe-steps-a` | 3000 | training steps per probe |
| `--n-decoy` | 20 | decoys that set the threshold |
| `--decoy-kind` | `auto` | `uniform` on the box, or `permute` (shuffled copies of real columns); `auto` uses `permute` for fixed datasets |
| `--protect-factor` | 2.0 | protection threshold, in units of the highest decoy |
| `--guard-nats` | 0.1 | tolerance of the guard |

Unknown options go to the sweep driver's parser, so the discovery stage takes
`--ensemble-k`, `--sr-time-limit` and so on.

Output, per trial, in `<out-dir>/runs/<tag>/whittle.json`:

- `keep`, `keep_names`: the surviving inputs;
- `stages`: one record per round with `info_nats`, `info_total`, the score of
  every input (`scores`), the decoy maximum, the threshold, the `protected`
  inputs, and any guard message.

## Result on the hidden banana, d = 100

Two active inputs hidden among 100, 2000 simulations (1700 training, 300 held
out), defaults above, ten trials with different hidden pairs.

- 10 of 10 trials kept both active inputs.
- They ranked first and second among all 100 inputs at the first round, and the
  threshold selected exactly those two at every round.
- Held-out information rose from a median of 0.62 nats with 100 inputs to 3.51
  with 5.

Table: `notes/iterative_whittle_design.md`. Outputs: `results/whittle_d100/`.
Scores in those files are in raw units; the script now reports them per unit
of prior width, which multiplies them by 6 on this problem and changes no
decision.

## Three things learned

**1. The skip connection biases the score.** The map's skip connection is
initialised to the identity on its first `m_probe` input columns
(`degeneracy_distillery/oneshot.py:95-99`). Whatever inputs sit there start
with unit sensitivity, and with `--score all` they keep an inflated score.
There are two remedies:

| remedy | options | works on |
|---|---|---|
| put decoys in the skip positions, so the bias lands on known-inactive columns | `--skip-fill decoys --score all` (the defaults) | the banana. On CAMELS the map then learns nothing. |
| leave the real inputs where they are and score only along latent axes the data predict | `--skip-fill none --score informative` | CAMELS. An inactive input in a skip position ends up on an axis the data cannot predict and scores at decoy level. |

The second remedy removes most of the inflation, not all of it. Checked on a
banana trial at d = 100 with one active input (2) in a skip position and no
decoys there: inputs 0 and 3 fell to decoy level (0.44, 0.49 against a decoy
maximum of 0.47), but input 1, also inactive and in a skip position, scored
1.27 and crossed the threshold of 0.94. The other active input (73) ranked
third and survived the cut without being protected. So with this remedy, treat
a skip-position input that is only a few times the decoy maximum as
unconfirmed. On the banana the first remedy is cleaner (10 of 10 trials, no
false positive).

A third option, `--skip-fill shuffle`, puts a random set of real inputs in the
skip positions for each probe and leaves those unscored. It works less well
than the second remedy (see the CAMELS table) and is kept only for comparison.

The `fisher` screen behind `--sr-top-k` in `scripts/oneshot_sweep.py` ranks by
`sqrt(diag(JᵀJ))` over all axes and so carries the bias.

**2. "Above every decoy" is too weak a threshold.** With 98 inactive inputs and
10 decoys, about nine inactive inputs beat the highest decoy by chance. Twice
the highest of 20 decoys separated cleanly.

**3. The guard is noisy.** With 3000-step probes the information estimate
moves by a few tenths of a nat between fits. At the default 0.1 nats the guard
fired 8 times in the ten banana trials, every time a false alarm. Use 0.4 to
0.5, or more probe repeats.

## Using it on CAMELS

What is different from the banana:

- **A fixed dataset.** 819 training and 205 held-out simulations, 35 parameters
  scaled to [1, 2], `m_probe = 12`. The script takes the test split from
  `held_out()` and uses permuted decoys, because 19 of the 35 marginals are not
  uniform on the scaled box.
- **No exactly inactive inputs.** Parameters are more or less active.
  Marginalising a weakly active one adds noise to the rest, so expect the
  protected set to be larger and the cuts to cost a little information.
- **Fewer simulations.** Scores are noisier; average several probes.

Diagnostics, one round on all 35 inputs (seed 0, laptop, 1500-step probes):

| setting | held-out information (nats) | what the scores show |
|---|---|---|
| `--skip-fill decoys --score all` | 0.00 (0.07 with 10000 steps) | nothing learned; every input at decoy level |
| `--skip-fill none --score all` | 6.6; six axes above 0.5 | the map learns, but 12 of the 14 protected inputs are exactly the first 12 columns: the ranking is the skip bias |
| `--skip-fill shuffle --score all`, 6 probes | 2.8 | the map learns less when the skip does not sit on the informative inputs; only 3 inputs protected |
| `--skip-fill none --score informative`, 2 probes | 6.2; six axes above 0.5 | full information and a ranking free of the bias: see below |

So on CAMELS the map needs real inputs in the skip positions to learn, and the
informative-axes score is what removes their inflation.

First result with `--skip-fill none --score informative` (one round, 2 probes;
threshold 1.40 = twice the highest decoy, 0.70):

| parameter | score | | parameter | score |
|---|---|---|---|---|
| Omega_m | 11.5 | | WindDumpFactor | 2.7 |
| Omega_b | 11.2 | | MinWindVel | 2.3 |
| H_0 | 10.8 | | WindFreeTravelDensFac | 2.2 |
| IMFslope | 10.4 | | MaxSfrTimescale | 2.0 |
| n_s | 5.8 | | SNII_MinMass | 1.9 |
| sigma_8 | 5.7 | | SofteningLength | 1.6 |
| A_SN2 | 5.6 | | *A_AGN2* | *1.3, below threshold* |
| A_SN1 | 5.5 | | *A_AGN1* | *0.74, at decoy level* |

- 14 parameters are above the threshold; the other 21 score between 0.49 and
  1.32.
- Three of the first 12 columns are now below the threshold (A_AGN1, A_AGN2,
  FactorForSofterEQS). Scored over all axes they were at 7 or more. That is
  the bias being removed.
- This is one seed and one round. It is a check that the script behaves, not a
  CAMELS result.
- MaxSfrTimescale (2.0) sits in a skip position and is 2.8 times the decoy
  maximum, the same ratio as the false positive in the banana check. Confirm it
  with a second run in which the parameter columns are in a different order
  before counting it as active. The eight parameters at 5.5 and above are
  well clear of that.

Cautions for CAMELS:

- **Keep `m_probe` at or above the informative rank.** The informative-axes
  score needs the uninformative axes to exist, so that inactive inputs have
  somewhere to go. With about 7 informative directions, `m_probe = 12` is fine.
- **Recheck the earlier Fisher ranking.** `notes/camels_report.md` ranks
  parameters by Fisher sensitivity from a map whose skip sat on the first 12
  columns (Ω_m … IMFslope). Those ranks are confounded with column order. In
  particular the AGN amplitudes, 7th and 8th there, fall to decoy level here.
- **One seed is not a result.** Run several seeds and keep the inputs that are
  protected in most of them.
- **Discovery afterwards.** Drop `--skip-discovery` to run the sweep driver's
  one-step pipeline on the survivors, or pass `keep_names` to
  `scripts/camels_square.py`.

## CAMELS: first runs of the two-step procedure (16 Nov 2026)

Raw 344 standardised features as the data vector (`n_pca=0`), laptop CPU.
Launcher: `scripts/camels_whittle_runs.sh screen`, then `refit 13 11 14`.
Outputs: `results/whittle_camels/`.

**Step 1, screen.** One round on all 35 parameters, `--skip-fill none --score
informative`, three probes of 1500 steps per run. Five seeds in the default
column order and three in a check order whose skip positions hold the eight
strongest parameters and four of the weakest. Each seed is a different
train/test split. Combined with `scripts/whittle_aggregate.py`.

| rank | parameter | score / highest decoy | above threshold (2) in |
|---|---|---|---|
| 1 | Omega_m | 24.6 | 8/8 |
| 2 | IMFslope | 23.4 | 8/8 |
| 3 | Omega_b | 21.5 | 8/8 |
| 4 | H_0 | 18.8 | 8/8 |
| 5 | sigma_8 | 12.0 | 8/8 |
| 6 | n_s | 11.5 | 8/8 |
| 7 | A_SN1 | 10.5 | 8/8 |
| 8 | A_SN2 | 9.5 | 8/8 |
| 9 | WindDumpFactor | 6.9 | 8/8 |
| 10 | WindFreeTravelDensFac | 5.2 | 8/8 |
| 11 | SNII_MinMass | 4.6 | 8/8 |
| 12 | MinWindVel | 2.8 | 8/8 |
| 13 | WindEnergyReductionFactor | 2.3 | 8/8 |
| 14 | QuasarThreshold | 2.2 | 6/8 |
| 15 | VariableWindSpecMomentum | 2.2 | 7/8 |
| 16 | MaxSfrTimescale | 2.2 | 5/8 |
| 17 | ThermalWindFraction | 2.1 | 6/8 |

The other 18 score 0.75 to 1.95; A_AGN1 and A_AGN2 are at about 1.0. Held-out
information is 7.6 to 8.6 nats per run, with six axes above 0.5 nats and a
seventh near 0.4.

The check order matters for inputs in the skip positions: MaxSfrTimescale
scores 2.37 there and 1.83 outside, A_AGN2 1.10 and 0.80, A_AGN1 1.03 and 0.82.
Inputs outside those positions score the same in both orders. MaxSfrTimescale
is therefore not counted as active.

**Step 2, refit on the top k only.** One-step loss with η(θ) and the data
summary both k-dimensional; the other parameters are left in the simulations
and marginalised. Four members (independent restarts), 6000 steps, seed-0
split. Information is on the 205 test simulations.

| fit | parameters | latents | held-out information (nats), mean ± sd over members | axes above 0.5 nat | 8th axis (nats) |
|---|---|---|---|---|---|
| top 11 | ranks 1–11 | 11 | 8.63 ± 0.19 | 6, 7, 6, 7 | 0.10 |
| top 13 | ranks 1–13 | 13 | 8.93 ± 0.15 | 7, 7, 7, 7 | 0.31 |
| top 14 | ranks 1–14 | 14 | 8.95 ± 0.15 | 7, 7, 6, 7 | 0.33 |
| reference | all 35 | 12 | 8.46 ± 0.19 | 6, 7, 7, 7 | 0.04 |

- **Nothing is lost by marginalising the rest, and a little is gained.** The
  13-parameter fit holds 8.93 nats against 8.46 with all 35. Fewer inputs to
  fit from 819 simulations outweighs whatever the dropped parameters carried.
- **k = 13 is the choice the data support.** Ranks 12 and 13 add about 0.3 nats
  over the top 11, as an eighth direction. Rank 14 adds nothing.
- **Seven directions clear 0.5 nats in every member at k = 13**, as in
  `notes/camels_report.md`, with an eighth at about 0.3 that the all-parameter
  fit does not show.

Limits: the refits use one split; the members are restarts, not an aligned
ensemble (see the Colab section below for that); no symbolic regression.

## Next step: aligned ensemble on Colab

`notebooks/camels_top13_ensemble_colab.ipynb` on branch `camels-whittle` runs
the neural compression step as an aligned ensemble on the 13 selected
parameters (raw features, 10 members with varied data-network architectures,
bootstrapped, three splits), once with 13 latents and once at the informative
rank, without symbolic regression. It needs the 3.4 MB data file uploaded and,
if the repository is private, a `GH_TOKEN` Colab secret. The run was checked
from a fresh clone of the branch in an environment with only JAX, Flax, Optax,
h5py and pyoperon. `notebooks/` is git-ignored, so the notebook was added with
`git add -f`.

## CAMELS: aligned ensemble on the top 13, Colab (16 Nov 2026)

Run from the notebook above on a Colab GPU. Outputs copied to
`results/whittle_camels/colab_top13/` (`m13/`, `rank/`, `summary.txt`);
analysis: `scripts/camels_ensemble_summary.py`. Settings: probe 20000 steps,
members 5000 steps, 10 members, bootstrap, varied data networks, 819 training
and 205 test simulations per split, three splits.

The first attempt failed on Colab because its preinstalled Flax was older than
its JAX (`jax.core.get_opaque_trace_state was removed`). The notebook now
upgrades Flax and checks the pair before any run.

| | split 0 | split 1 | split 2 |
|---|---|---|---|
| rank by the 0.5-nat rule | 6 | 6 | 6 |
| held-out information, 13 latents (nats) | 8.63 | 7.81 | 8.28 |
| held-out information, 7 latents (nats) | 7.43 | 6.67 | 6.83 |
| weak axes 7 and 8, 13 latents (nats) | 0.32, 0.34 | 0.44, 0.28 | 0.36, 0.23 |
| member spread / range, six leading axes, 13 latents | 0.13–0.22 | 0.12–0.20 | 0.16–0.24 |
| member spread / range, six leading axes, 7 latents | 0.21–0.30 | 0.23–0.33 | 0.22–0.40 |
| calibration, worst leave-one-out ratio, 13 / 7 latents | 1.10 / 1.14 | 1.34 / 1.19 | 1.28 / 1.07 |

"Range" is the standard deviation of the coordinate over the test simulations.
"Leading axes" are the principal axes of the ensemble-mean coordinates; their
variances track the information per axis, so the first six are the informative
ones. The error on the ensemble mean is the spread divided by sqrt(10).

- **Six informative directions, the same ones in every run.** The parameter
  subspaces spanned by the six leading axes agree across the three splits and
  the two settings to within 12 degrees (principal angles). A seventh direction
  is not shared (23 to 79 degrees). On one split, the six leading axes of the
  13-latent and 7-latent fits have canonical correlations of 0.99 or more for
  five, and 0.86 to 0.90 for the sixth.
- **What the six directions are**, in one sparse (varimax) basis of the shared
  subspace, loadings on the standardised scaled parameters:
  - Omega_b alone (+1.00); H_0 alone (+0.98); Omega_m alone (+0.97);
  - IMFslope +0.90, WindDumpFactor +0.26, sigma_8 −0.22, n_s −0.20;
  - A_SN2 +0.80, WindFreeTravelDensFac −0.49, WindDumpFactor +0.26, MinWindVel +0.22;
  - A_SN1 +0.70, n_s −0.43, sigma_8 −0.41, SNII_MinMass −0.33.

  Each of the six runs captures each of these directions at 0.96 or more
  (squared cosine). sigma_8 and n_s are not constrained on their own.
- **The coordinates are close to linear in the 13 scaled parameters.**
  Cross-validated R² of a linear fit is 0.96 to 0.99 per axis; a quadratic fit
  reaches 0.99 or more. The non-linear part is 1 to 4% of the variance, about
  the size of the squared error on the ensemble mean (0.2 to 1.7%). Symbolic
  regression will mostly return linear combinations. The map has a linear skip
  path, so this is what the map learned from 819 simulations, not proof that
  the sufficient coordinates are linear.
- **The 7-latent fit holds 1.2 to 1.5 nats less than the 13-latent fit and its
  members disagree more.** The two weak axes account for 0.6 to 0.7 nats of the
  difference.
- **The member spread does not say which directions the data constrain.** In
  the 13-latent fit the uninformative directions have the *smallest* spread
  (8 to 12% of their range). Every member is warm-started from the same eta(theta)
  network (`fit_oneshot_ensemble`, `init_fit`), and nothing moves the directions
  the data do not reach. The spread measures stability under resampling and a
  change of data network from a common start. Use the information rule for the
  rank, and the spread only as the weight for symbolic regression.
- **The spread is slightly too small in a few coordinates** (leave-one-out
  ratio up to 1.34), and it varies by a factor of about four across simulations.

Not saved by the run, and needed to do better: the data summaries of each
member on the test set and the information axes. Without them the informative
directions of the 13-latent fit are identified only through the variance of
the coordinates.

## CAMELS: symbolic regression on the aligned coordinates (16 Nov 2026)

Written up separately: `notes/camels_top13_sr_native.md` (alignment to a common
basis, linear part and nonlinear remainder, native-MDL Operon search, figures).

## CAMELS: deeper data network, second Colab run (prepared 16 Nov 2026)

The aligned coordinates of the first run are 98 to 99% linear in the 13 scaled
parameters. Both networks start linear: the data network eta_hat(x) was
256-256-128 (GELU) with a zero-initialised output layer plus a linear shortcut
from the 344 features, and eta(theta) has the same structure. To test whether
that start holds the map near-linear, branch `camels-whittle` changes the
CAMELS data network to 344 -> 512 -> 512 -> 256 -> 128 -> m with no linear
shortcut (`EST_FEATURES`, `EST_SKIP` in `problems/camels.py`; override with
`--problem-arg est_features=256x256x128 --problem-arg est_skip=true`). The
other ensemble members draw 3 to 5 layers of width 128 to 512, also without
the shortcut. Other problems keep the shortcut (`Estimator.skip` defaults to
true). The eta(theta) network is unchanged.

`ensemble.npz` now also holds, per member and before alignment, `eta_raw` and
the data summaries `eta_hat_raw` on the test simulations, and `rotmats`.

Each run also writes, unless `--save-aug 0`:

- `models.pkl`: the fitted ensemble as plain arrays plus an architecture
  description. `load_ensemble_models(path)` returns `(fits, aligned, weights)`
  and `aligned_at(fits, aligned, weights, theta)` gives the aligned member
  coordinates at any theta (`degeneracy_distillery/oneshot_ensemble.py`).
- `augmented_empirical.npz`, `augmented_uniform.npz`: `--save-aug` fresh theta
  each (default 20000) with `ys` (every member), `y`, `y_std`, in the frame of
  `ensemble.npz`. Symbolic regression should be fitted to these, not to the 205
  test simulations. *Empirical* follows the training marginals (four of the 13
  parameters are log-sampled); *uniform* covers the box evenly and so puts
  weight where the training simulations are sparse.

The notebook writes to `results/camels_top13_deep` and downloads
`camels_top13_deep.zip`. Compare against the first run (same splits): held-out
information 8.63 / 7.81 / 8.28 nats at 13 latents, remainder after a linear
fit 0.7 to 2.1% of the variance per coordinate.

## CAMELS: parameter map without the linear path, third Colab run (prepared 16 Nov 2026)

The deeper data network changed nothing material (rank 6 in all splits;
8.44 / 7.23 / 8.13 nats at 13 latents against 8.63 / 7.81 / 8.28; coordinates
still 98 to 99% linear; same six-dimensional subspace to within 5 degrees;
`results/whittle_camels/colab_top13_deep/compare_with_first_run.txt`). So the
data-side shortcut was not what kept the coordinates near-linear.

The parameter map eta(theta) is a linear skip plus an MLP branch. With
`--delta-init zeros` it starts exactly linear. `--delta-init glorot` hardly
changes that: at initialisation the branch is about 0.15 of the skip in size
and the map is 0.4 to 0.5% nonlinear over the box. New option
`--delta-init mlp`: no linear skip, eta is the MLP alone (13 -> 128 -> 128 ->
128 -> m, GELU, glorot output layer), 18 to 20% nonlinear at initialisation.

Notebook: `notebooks/camels_top13_ensemble_nonlinear_theta_colab.ipynb`
(`THETA_MAP = "mlp"`, deeper data network as in the second run). Output
`results/camels_top13_theta_mlp`, zip `camels_top13_theta_mlp.zip`. Compare with
`scripts/camels_deep_compare.py`.

## Not done

- Discovery on the survivors of the d = 100 trials (stopped at the author's
  request; discovery from a few inputs is the setting of Fig. 2A).
- The controls of the design note (`direct`, `single-cut`), the weakly active
  banana variant, d = 500.
- A fix to the skip-connection initialisation itself, which would remove the
  bias at its source.
