# Rerun: discovery and NPE budget sweeps with the discovery simulations inside the budget

Run on 11 Nov 2026 on branch `reuse-rerun` (cut from `square-completion`, a55e38b).
One run per problem: seed 0, the `rebuttal` config (500 discovery training
simulations), CPU only. SIR, TaylorF2 and IMRPhenomD all completed.

## Bottom line

**In this rerun η does not beat θ once the discovery simulations are counted.**

1. **Discovery uses 1000 simulations, not 500.** Each driver draws 500 training
   simulations and 500 held-out ones. The held-out set is not a passive test
   set: the fishnets early-stop on it, the ensemble weights come from it, and
   the Fisher matrices the flattening network and the symbolic regression are
   fit to are evaluated at those 500 points (`train_fishnets` saves
   `theta=theta_test`, `Fs=F(data_test)`). So an η arm trained on N = 500 has
   cost 1000 simulator calls. In the main runs below the pool is therefore
   discovery-train (rows 0–499), discovery-held-out (rows 500–999), then fresh
   simulations, and **N = 1000 is the first budget at which both arms have made
   the same number of simulator calls.**
2. **At N ≥ 1000 there is no consistent η advantage in any problem.** At
   N = 1000 the held-out test log-prob difference η − θ is −0.28 ± 0.06 (SIR),
   −0.13 ± 0.10 (TaylorF2) and −0.02 ± 0.04 (IMRPhenomD). With a second,
   independent pool the same three cells are +0.38, −0.21 and +0.36: the sign
   is not stable.
3. **At N = 500 η is ahead on both metrics only for TaylorF2** (+0.49 nats on
   validation, +0.49 ± 0.05 on held-out test). θ reaches that score at about 650–730
   simulations, and θ at N = 1000, which is what the η arm had actually spent,
   is ahead of η at N = 500 in all three problems (table in the next section).
4. **SIR discovery did not find R0 at this seed.** It returned an almost exactly
   affine map of (β, γ). The SIR sweep therefore compares two parameterisations
   that differ by a linear change of variables, and it shows no gain.
5. **IMRPhenomD shows no consistent gain.** The symbolic coordinates it found
   are less flat on held-out data than raw θ (the driver's held-out flatness
   score, `heldout_geometry.frob_*`, is 8.07 against 6.49; lower is flatter),
   although the driver's own success test passes.

Two limits on how far this goes:

- **One seed.** The same arm at the same budget moves by up to 1.4 nats
  (validation) and 0.7 nats (test) between two independent pools (see
  "How noisy is one seed"), and SIR's affine coordinates, where no real difference
  is expected, show η − θ anywhere from −0.28 to +0.50. Only one difference
  in this rerun is clearly larger than that: TaylorF2's test log-prob at
  N = 5000 (+0.58 ± 0.22, not seen on validation).
- **These are not the published coordinates.** Symbolic regression did not
  reproduce the expressions hardcoded in the sweep scripts. As a control I ran
  the same pools with those built-in expressions: η is then ahead at
  N = 500–1000 for SIR and at N = 500 for TaylorF2, and not for IMRPhenomD. So
  the finding is "seed-0 discovery on CPU did not produce coordinates that
  help", not "the published coordinates do not help".

## What this changes in the reply to the reviewers

- The sentence "at any budget N ≥ 500 both arms have made N simulator calls" is
  not true for these drivers. With the held-out simulations also placed in the
  pool, as here, it is true for N ≥ 1000. At N = 500 the η arm has made 1000.
  The tutorial notebooks have the same structure (`2 * nsims` simulations), so
  this is not specific to the `rebuttal` config. Whether the cluster runs
  behind Figs. 2B and 4A counted the held-out half I could not check from this
  machine.
- For the two GW problems the PCA basis is fit to the noiseless waveforms at all
  1000 discovery parameter points. That adds no simulator calls, but it means
  the compression seen by both arms at N = 500 already rests on 1000 points.

Equal-cost comparison, main runs (validation / held-out test log-prob, higher is better):

| problem | θ at N=1000 (val / test) | η at N=500 (val / test) | η at N=1000 (val / test) |
|---|---|---|---|
| SIR | 4.98 / 5.98 | 4.24 / 4.43 | 4.99 / 5.69 |
| TaylorF2 | -4.83 / -4.71 | -5.63 / -5.11 | -5.10 / -4.83 |
| IMRPhenomD | -6.00 / -5.41 | -6.39 / -7.09 | -6.00 / -5.43 |

## What was run

```bash
# per problem (sir | taylorf2 | imrphenomd)
scripts/run_reuse_rerun.sh <problem> discovery        # discovery driver, saves discovery_sims.npz
scripts/run_reuse_rerun.sh <problem> sweep            # main run: pool = train + held-out + fresh
POOL=train_only scripts/run_reuse_rerun.sh <problem> sweep   # pool = train + fresh
```

`discovery` runs `sir_notebook_run.py --noise-std 0.01` / `gw_notebook_run.py` /
`imrphenomd_notebook_run.py` with `--mode rebuttal --seed 0 --no-require-gpu
--save-training-data`. `sweep` rewrites the discovered expressions for the sweep
(`reuse_rerun_tools.py exprs`), runs the matching `*_nsims_logprob_sweep.py` with
`--nsims 500 1000 2500 5000 --seed 0 --training-data <archive> --theta-to-eta
<discovered> --eta-inverse numeric --paired-arm-seeds --run-coverage
--coverage-all-nsims --eval-test-logprob`, and tabulates the result
(`reuse_rerun_tools.py summarize`). All new behaviour is behind new options; the
scripts' defaults are unchanged and still run.

How to read the tables:

- **val log-prob**: the sweep's `best_validation_log_prob_theta_density`, mean ±
  standard deviation over the two ensemble members the sweep trains per arm
  (`--repeats-maf 2`). The validation set is 10% of the N simulations, so 50
  simulations at N = 500, and it is the set early stopping selects on.
- **test log-prob** (added): mean log q(θ | x) over 1000 fresh simulations that
  no arm trained or selected on, in θ density, ± standard error over those 1000
  points. For η the Jacobian is applied per point.
- **Δ test (η − θ)**: per-point paired difference on that test set, ± standard
  error. The error bars cover test-set sampling only, not training randomness.
- **68% cov.**: fraction of 1000 test truths inside the central 68% posterior
  interval, per parameter (0.68 is calibrated). **calib. err.**: mean distance
  between the sorted posterior ranks and a uniform distribution (0 is calibrated).
  GW coverage uses the sweep's existing filter (m1 ≥ 10), 987 test cases.
- **CRPS R0** (SIR): from the same 1000 × 1000 coverage draws, with the driver's
  CRPS definition. Lower is better.
- **prior fallbacks**: number of coverage sampling calls (out of about 2000 per row)
  in which ltu-ili's rejection sampler gave up and returned prior samples
  instead of posterior samples. They inflate the CRPS mean and are more frequent
  for θ, so read the CRPS median, not the mean.

## SIR

Discovery returned (driver printout, physical units):

```
eta_0 = 0.31*beta - 0.022*gamma + 0.221
eta_1 = 0.034*beta + 1.672*gamma + 0.692
```

In the driver's scaled inputs these are `0.27*X1 - 0.01*X2` and
`0.03*X1 + 0.75*X2`: linear. The third sweep coordinate is I0/10, passed through
unchanged as in the sweep's built-in expressions. The driver reports success
(best |Pearson r| with R0 = 0.59, threshold 0.5), held-out flatness 18.3 (raw θ),
9.4 (symbolic), 0.64 (neural flattening).

| N | arm | val log-prob (2 members) | test log-prob | Δ test (η−θ) | CRPS R0 median | CRPS R0 mean | Δ CRPS (η−θ) | 68% cov. beta/gamma/I0 | calib. err. | prior fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| 500 | θ | 3.74 ± 0.37 | 4.31 ± 0.14 |  | 0.0910 | 0.1539 ± 0.0089 |  | 0.71/0.78/0.68 | 0.040/0.040/0.031 | 13 |
|  | η | 4.24 ± 0.24 | 4.43 ± 0.26 | +0.12 ± 0.17 | 0.0802 | 0.1380 ± 0.0075 | -0.0159 ± 0.0037 | 0.68/0.75/0.68 | 0.051/0.037/0.041 | 9 |
| 1000 | θ | 4.98 ± 0.63 | 5.98 ± 0.17 |  | 0.0577 | 0.1212 ± 0.0201 |  | 0.67/0.66/0.66 | 0.029/0.014/0.014 | 13 |
|  | η | 4.99 ± 0.83 | 5.69 ± 0.18 | -0.28 ± 0.06 | 0.0642 | 0.1127 ± 0.0137 | -0.0086 ± 0.0226 | 0.68/0.69/0.65 | 0.038/0.014/0.032 | 4 |
| 2500 | θ | 6.69 ± 0.02 | 7.69 ± 0.07 |  | 0.0316 | 0.0803 ± 0.0153 |  | 0.68/0.67/0.67 | 0.012/0.026/0.010 | 12 |
|  | η | 6.57 ± 0.17 | 7.51 ± 0.07 | -0.18 ± 0.03 | 0.0364 | 0.0637 ± 0.0034 | -0.0166 ± 0.0148 | 0.68/0.67/0.67 | 0.013/0.023/0.014 | 4 |
| 5000 | θ | 7.72 ± 0.04 | 8.06 ± 0.07 |  | 0.0272 | 0.0566 ± 0.0061 |  | 0.69/0.67/0.69 | 0.013/0.036/0.010 | 15 |
|  | η | 7.71 ± 0.02 | 8.14 ± 0.07 | +0.08 ± 0.02 | 0.0264 | 0.0469 ± 0.0024 | -0.0097 ± 0.0049 | 0.68/0.66/0.68 | 0.014/0.014/0.011 | 10 |

Budget at which θ matches η at N=500: validation: ≈660 (between 500 and 1000); test: ≈526 (between 500 and 1000); CRPS R0 median: ≈625 (between 500 and 1000); CRPS R0 mean: ≈701 (between 500 and 1000). With the train-only pool: validation: ≈599 (between 500 and 1000); test: ≤500 (θ is already at least as good); CRPS R0 median: ≈618 (between 500 and 1000); CRPS R0 mean: ≈647 (between 500 and 1000).

Verdict: no gain. The signs of η − θ are mixed across budgets and across the two
pool layouts. The +0.50 on validation at N = 500 is not confirmed on the test
set (+0.12 ± 0.17, and −0.26 ± 0.55 on the other test set). Since the two arms
differ only by an affine relabelling, this row is a direct measurement of how
large a validation "gain" at N = 500 can be with nothing behind it.

Simulator constants: the sweep's `SirConfig` and the driver agree on population
(1000), time grid (30 points on [0, 50]), I0 ~ Poisson(8), priors and the
R0 ≥ 1.15 cut. Both default to noise 0.05, so 0.01 was passed to both. The
archive now carries these constants and the sweep refuses a pool that does not
match. Wall-clock: discovery 13.2 min, sweep 9.3 min.

## TaylorF2

Discovery returned (driver printout, physical units):

```
eta_0 = 0.003*m1 - 0.045*(0.022*m2 + 0.887)**2 + 0.141
eta_1 = -0.003*m2 + 1.0*(2.098 - 0.085*exp(1.703*exp(0.006*m1 + 0.007*m2)))**1.427 - 0.119
```

**This printout is not the coordinate the pipeline found.** It is rounded to 3
decimals, which here drops a term, differs from the true `eta_1` by up to 0.35
(on a range of 1.8), and is undefined on 0.17% of the prior (negative base under
a fractional power near m1, m2 ≈ 50). The sweep used the unrounded expression in
the driver's scaled inputs,

```
eta_0 = 0.003126 + 1.006946*((0.082412*X2)*(-0.539560*X2) + 0.154548*X1)
eta_1 = -0.000005 + 1.000003*(((-0.103181 + 0.020084*X2)*exp(exp(0.264449*X1 + 0.335501*X2)) + 2.097520)^1.426537 - 0.133984*X2)
```

composed with the driver's scaler (`notes/reuse_rerun_data/taylorf2/sweep_exprs.json`).
It is finite on the whole prior and has no closed-form inverse, hence the
numerical inverse. Held-out flatness: 21.4 (raw θ), 3.0 (symbolic), 2.7 (neural).
Evaluated exactly, `eta_0` tracks m1 − m2 (r = 1.00 over the prior square) and
`eta_1` tracks total mass (r = −0.99) and log chirp mass (r = −0.93). The run
record's `chirp_mass_alignment = 0.013` should be disregarded: the driver
computes its physics correlations from the rounded printout, where `eta_1` is
non-finite on part of the sample and its correlation is set to 0.

| N | arm | val log-prob (2 members) | test log-prob | Δ test (η−θ) | 68% cov. m1/m2 | calib. err. | prior fallbacks |
|---|---|---|---|---|---|---|---|
| 500 | θ | -6.12 ± 0.09 | -5.60 ± 0.05 |  | 0.64/0.74 | 0.057/0.040 | 4 |
|  | η | -5.63 ± 0.35 | -5.11 ± 0.07 | +0.49 ± 0.05 | 0.62/0.66 | 0.037/0.065 | 3 |
| 1000 | θ | -4.83 ± 0.45 | -4.71 ± 0.13 |  | 0.67/0.69 | 0.047/0.018 | 5 |
|  | η | -5.10 ± 0.12 | -4.83 ± 0.07 | -0.13 ± 0.10 | 0.66/0.71 | 0.049/0.036 | 2 |
| 2500 | θ | -3.18 ± 0.15 | -3.19 ± 0.11 |  | 0.66/0.66 | 0.044/0.026 | 6 |
|  | η | -3.47 ± 0.03 | -3.23 ± 0.15 | -0.05 ± 0.10 | 0.69/0.67 | 0.014/0.029 | 2 |
| 5000 | θ | -2.35 ± 0.06 | -2.39 ± 0.27 |  | 0.67/0.66 | 0.032/0.019 | 20 |
|  | η | -2.40 ± 0.06 | -1.81 ± 0.10 | +0.58 ± 0.22 | 0.70/0.69 | 0.024/0.015 | 2 |

Budget at which θ matches η at N=500: validation: ≈651 (between 500 and 1000); test: ≈731 (between 500 and 1000). With the train-only pool: validation: ≈687 (between 500 and 1000); test: ≈690 (between 500 and 1000).

Verdict: η is ahead at N = 500 by about half a nat on both metrics, behind or
level at 1000 and 2500, and ahead on test (not on validation) at 5000. That
pattern repeats in the train-only pool. θ needs roughly 650–730 simulations to
match η at 500, so the saving is real but smaller than the 500 extra simulations
discovery spent. Wall-clock: discovery 8.2 min, sweep 5.4 min.

## IMRPhenomD

Discovery returned (driver printout, physical units):

```
eta_0 = 0.02*m1 + 0.099
eta_1 = (0.006*m1 + 0.011*m2 - 0.768*exp(0.006*m1) + 0.671)/(0.01*m1 - 1.28*exp(0.006*m1) + 0.38)
```

The sweep used the unrounded form, `0.9*X1 - 0.7` and
`-0.5*X2/(-0.427596*X1 + exp(0.278071*X1)) + 0.6` in the driver's scaled inputs
(rounding error here is at most 0.045). Held-out flatness: 6.5 (raw θ), 8.1
(symbolic), 2.2 (neural). The symbolic coordinates are worse than raw θ on this
score; the driver's success test only compares against chirp mass and mass ratio.

| N | arm | val log-prob (2 members) | test log-prob | Δ test (η−θ) | 68% cov. m1/m2 | calib. err. | prior fallbacks |
|---|---|---|---|---|---|---|---|
| 500 | θ | -6.65 ± 0.01 | -6.83 ± 0.18 |  | 0.59/0.56 | 0.043/0.037 | 12 |
|  | η | -6.39 ± 0.15 | -7.09 ± 0.32 | -0.26 ± 0.16 | 0.54/0.52 | 0.049/0.054 | 6 |
| 1000 | θ | -6.00 ± 0.31 | -5.41 ± 0.08 |  | 0.60/0.63 | 0.031/0.049 | 6 |
|  | η | -6.00 ± 0.08 | -5.43 ± 0.07 | -0.02 ± 0.04 | 0.61/0.61 | 0.035/0.051 | 9 |
| 2500 | θ | -4.56 ± 0.26 | -4.37 ± 0.15 |  | 0.69/0.67 | 0.043/0.032 | 5 |
|  | η | -4.57 ± 0.04 | -4.17 ± 0.11 | +0.20 ± 0.09 | 0.67/0.66 | 0.050/0.051 | 5 |
| 5000 | θ | -3.61 ± 0.30 | -2.95 ± 0.09 |  | 0.72/0.68 | 0.039/0.046 | 7 |
|  | η | -3.61 ± 0.07 | -2.91 ± 0.07 | +0.04 ± 0.07 | 0.74/0.69 | 0.048/0.063 | 6 |

Budget at which θ matches η at N=500: validation: ≈659 (between 500 and 1000); test: ≤500 (θ is already at least as good). With the train-only pool: validation: ≈595 (between 500 and 1000); test: ≤500 (θ is already at least as good).

Verdict: no consistent gain. At N = 500 validation favours η (+0.26) and the test
set favours θ (−0.26 ± 0.16); from N = 1000 the arms are within noise.
Wall-clock: discovery 12.6 min, sweep 3.3 min.

## The pool exactly as specified (train-only)

The request was a pool of the 500 discovery training simulations followed by
fresh draws. I made the pool with the held-out simulations in rows 500–999 the
main run because that is the only layout in which the budget axis is the total
number of simulator calls. The train-only layout was run as well. The N = 500
models are the same in both (same simulations, same seeds, identical validation
numbers); the test sets and everything from N = 1000 are independent draws.

η − θ, main runs (train + held-out + fresh):

| N | SIR Δ val | SIR Δ test | TaylorF2 Δ val | TaylorF2 Δ test | IMRPhenomD Δ val | IMRPhenomD Δ test |
|---|---|---|---|---|---|---|
| 500 | +0.50 | +0.12 ± 0.17 | +0.49 | +0.49 ± 0.05 | +0.26 | -0.26 ± 0.16 |
| 1000 | +0.01 | -0.28 ± 0.06 | -0.27 | -0.13 ± 0.10 | -0.01 | -0.02 ± 0.04 |
| 2500 | -0.13 | -0.18 ± 0.03 | -0.28 | -0.05 ± 0.10 | -0.01 | +0.20 ± 0.09 |
| 5000 | -0.00 | +0.08 ± 0.02 | -0.05 | +0.58 ± 0.22 | +0.01 | +0.04 ± 0.07 |

η − θ, train-only pool (train + fresh; here the η arm is 500 simulator calls ahead of its budget at every N):

| N | SIR Δ val | SIR Δ test | TaylorF2 Δ val | TaylorF2 Δ test | IMRPhenomD Δ val | IMRPhenomD Δ test |
|---|---|---|---|---|---|---|
| 500 | +0.50 | -0.26 ± 0.55 | +0.49 | +0.36 ± 0.06 | +0.26 | -0.22 ± 0.17 |
| 1000 | +0.08 | +0.38 ± 0.10 | -0.20 | -0.21 ± 0.09 | +0.19 | +0.36 ± 0.12 |
| 2500 | +0.03 | -0.00 ± 0.02 | -0.02 | -0.24 ± 0.07 | +0.50 | +0.09 ± 0.05 |
| 5000 | +0.01 | -0.04 ± 0.02 | +0.34 | +0.31 ± 0.07 | +0.18 | +0.08 ± 0.07 |

Only TaylorF2 shows a pattern on the test metric that repeats across the two
layouts (+ at 500, − at 1000 and 2500, + at 5000). IMRPhenomD's test differences
at 2500 and 5000 are positive in both but small (+0.04 to +0.20).

## How noisy is one seed

The θ arm at the same budget, trained on two independent pools:

| problem | N | θ val, main | θ val, train-only | θ test, main | θ test, train-only |
|---|---|---|---|---|---|
| SIR | 1000 | 4.98 | 5.65 | 5.98 | 5.52 |
| SIR | 2500 | 6.69 | 6.47 | 7.69 | 7.03 |
| SIR | 5000 | 7.72 | 7.41 | 8.06 | 7.68 |
| TaylorF2 | 1000 | -4.83 | -5.05 | -4.71 | -4.78 |
| TaylorF2 | 2500 | -3.18 | -4.58 | -3.19 | -3.64 |
| TaylorF2 | 5000 | -2.35 | -2.44 | -2.39 | -1.95 |
| IMRPhenomD | 1000 | -6.00 | -5.62 | -5.41 | -5.69 |
| IMRPhenomD | 2500 | -4.56 | -5.56 | -4.37 | -4.73 |
| IMRPhenomD | 5000 | -3.61 | -3.33 | -2.95 | -2.97 |

("main" has the discovery held-out simulations in rows 500–999; "train-only"
has fresh ones. Different test sets too.) Two realisations of the same arm
differ by 0.1–1.4 nats.

The paired differences η − θ are less noisy than that, because both arms share
simulations, split and initialisation. The SIR rows calibrate them directly:
there the two arms differ by an affine change of variables, so no real
difference is expected, and the observed differences run from −0.28 to +0.50.
TaylorF2 at N = 500 (+0.49 on both metrics) sits at the top edge of that range.
It is the best-supported gain in this rerun, and one seed cannot confirm it.

## Control: the sweeps' built-in (published) expressions on the same pools

Not a reuse run, since those expressions come from an earlier discovery. It
separates "this setup cannot show a gain" from "this seed's discovery found
weaker coordinates". Same pools, seeds and options as the main runs; only
`--theta-to-eta` is left at its default. η − θ:

| N | SIR Δ val | SIR Δ test | TaylorF2 Δ val | TaylorF2 Δ test | IMRPhenomD Δ val | IMRPhenomD Δ test |
|---|---|---|---|---|---|---|
| 500 | +0.72 | +0.29 ± 0.18 | +0.37 | +0.38 ± 0.05 | +0.37 | -1.27 ± 0.56 |
| 1000 | +0.53 | +0.39 ± 0.16 | -0.36 | -0.01 ± 0.10 | -0.34 | -0.12 ± 0.06 |
| 2500 | +0.04 | -0.07 ± 0.04 | -0.44 | -0.33 ± 0.18 | -0.14 | +0.15 ± 0.10 |
| 5000 | -0.07 | +0.06 ± 0.03 | -0.25 | +0.42 ± 0.23 | +0.31 | +0.16 ± 0.07 |

SIR: η ahead at 500 and 1000 (CRPS R0 median 0.066 against 0.091 at 500, 0.047
against 0.058 at 1000), level from 2500. TaylorF2: ahead at 500 (+0.37 validation,
+0.38 ± 0.05 test), then level or behind on validation. IMRPhenomD (the script's built-in
expressions are the TaylorF2 ones): validation +0.37 at 500 but test −1.27 ±
0.56. Full tables: `notes/reuse_rerun_data/<problem>/sweep_published_exprs/`.

## Deviations from the published configuration

Discovery:

- CPU (Apple M4 Pro, 14 cores) with `--no-require-gpu`, not a V100. Two or three
  jobs ran side by side, which affects wall-clock only.
- A fresh environment built today from `pyproject.toml` and the package list in
  `setup_slurm_venv.sh`: Python 3.12.13 (arm64), jax 0.12.0.1, flax 0.12.12,
  torch 2.15.0, ltu-ili at 0df745b, lampe 0.9.0, pyoperon 0.6.2, ESR at e2337cb,
  pycbc 2.11.0. The cluster venv's versions are not recorded in the repository.
  Full list: `notes/reuse_rerun_data/env_freeze.txt`.
- SIR: `--noise-std 0.01` (script default 0.05), as requested.
- New `--save-training-data` option on all three drivers. The GW drivers'
  `simulator_data` now also returns the PCA mean and components, the frequency
  grid and the whitening vector. No existing output changes.

Sweeps:

- `--seed 0` (default 42) and budgets 500, 1000, 2500, 5000 (default 100, 500,
  1000, 5000, 10000).
- Training pool read from the discovery archive (`--training-data`, new for the
  two GW sweeps) and extended with fresh simulations.
- η expressions from this run's discovery, unrounded, with a numerical inverse
  and the Jacobian taken from the forward map (`--eta-inverse numeric`), instead
  of the built-in expressions and their closed-form inverses.
- `--paired-arm-seeds`: both arms get the same torch seed. By default the two
  arms' seeds differ by one, and because ltu-ili draws the 90/10 split from that
  seed the arms then train and validate on different simulations (8 of 50
  validation simulations in common at N = 500). With the flag the split, the
  initial weights and the minibatch order are identical.
- SIR: `--noise-std 0.01`.
- GW: the PCA basis is the discovery run's (40 components fit to the 1000
  noiseless discovery waveforms; 91.2% of the variance for TaylorF2, 99.4% for
  IMRPhenomD), not the sweep's own fit to a bank of up to 5000 waveforms.
- GW: the discovery drivers draw masses on the full square [5, 50]², the sweeps
  on the triangle m1 ≥ m2. Both waveform models depend on the masses
  symmetrically, so a discovery simulation with m1 < m2 keeps its data and has
  its label swapped (477 of 1000). A folded uniform square is the sweep's prior.
- Coverage at every budget (`--coverage-all-nsims`, new for SIR) and the held-out
  test log-prob (`--eval-test-logprob`) are additional outputs.
- `--device cpu`.

Not changed: MAF with 5 transforms and 50 hidden features, 2 ensemble members
per arm per budget, batch size 32, learning rate 1e-4, 1000 test simulations,
1000 × 1000 coverage draws, validation smoothing window 10. Nothing had to be
reduced for runtime.

Two properties of the installed ltu-ili that apply to every sweep here and may
or may not match the cluster:

- `--epochs` has no effect. The sweeps pass `max_num_epochs`, the lampe runner
  reads `max_epochs`, so training ends by early stopping (patience 30). Best
  epochs were between about 20 and 260.
- The rejection sampler returns prior samples for an observation when fewer than
  about a third of its candidate draws fall inside the prior box (the "prior
  fallbacks" column).

## Checks

- Pool rows 0–499 equal the discovery `theta_train` / `data_train` bit for bit
  in all three problems (for GW, after the label swap); rows 500–999 equal the
  discovery held-out set.
- GW: discovery data minus the sweep simulator's noiseless projection at the
  same masses has mean 0.01 and standard deviation 1.00 per component, i.e. unit
  noise, for both waveform models. The archive's frequency grid, whitening
  vector, mass range and number of components are compared with the sweep's on
  load and a mismatch is an error.
- Numerical inverse against the built-in closed-form inverses: same samples to
  1e-7 and the same Jacobian correction to 6 digits (GW) and 4 digits (SIR,
  whose built-in inverse has rounded constants). Round trip on 3000 pool and
  test points: none lost, in every run.
- With `--paired-arm-seeds` the two arms' training and validation tensors are
  identical.
- The N = 500 validation numbers are identical in the main and train-only runs,
  as they should be.
- The three sweeps still run with their default options.

Not done: more than one seed; a repeat of discovery; a comparison with the
paper's numbers, which are not in the repository.

## Wall-clock

| stage | SIR | TaylorF2 | IMRPhenomD |
|---|---|---|---|
| discovery | 793 s | 491 s | 756 s |
| sweep, main pool | 560 s | 324 s | 196 s |
| sweep, train-only pool | 431 s | 192 s | 182 s |
| sweep, built-in expressions | 446 s | 201 s | 183 s |

Discovery splits as fishnets / flattening / symbolic regression: 222 / 411 / 123 s
(SIR), 105 / 123 / 254 s (TaylorF2), 220 / 279 / 246 s (IMRPhenomD).

## Files

- `scripts/run_reuse_rerun.sh`: the run recipe above.
- `scripts/reuse_pool_utils.py`: discovery-archive loader, frozen PCA, numerical
  inverse, forward Jacobian, test log-prob.
- `scripts/reuse_rerun_tools.py`: expression rewriting and the summary tables.
- `notes/reuse_rerun_data/<problem>/`: `discovery/run_record.json`,
  `sweep_exprs.json` (printed, pruned and exact expressions, and the rounding
  check), and per run `metrics.csv`, `metrics_aggregate.csv`, `test_logprob.csv`,
  `summary.csv`, `summary.json`, `manifest.json`.
- Large outputs (archives, posterior samples, coverage draws) are under
  `results/reuse_rerun/`, which is git-ignored.

On macOS the pyoperon 0.6.2 wheel needs `libxxhash` on the dynamic library path
and mpi4py needs an MPI library (`pip install mpich`); set
`REUSE_DYLD_FALLBACK_LIBRARY_PATH` for the run script. Neither applies on Linux.
