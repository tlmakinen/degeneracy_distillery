# Handoff: square-completed loss on CAMELS

Written 19 Oct 2026. Covers work from 30 Sep to 5 Oct. Nothing is running.

Read `notes/camels_report.md` for results. Read `square_completion_loss.md`
for the method. This file says what exists, where it is, and what to do next.

---

## 1. State

**Everything is uncommitted.** Last commit is `38b2b19`, 25 Sep. Commit before
anything else.

Branch: `new_idea`. No jobs queued or running.

## 2. Files

### New, and specific to this work

| path | lines | what |
|---|---|---|
| `degeneracy_distillery/square_completion.py` | 162 | the method: `LogitBox`, `AffineCoupling`, `CouplingFlow`, `SquareEta`, `square_loss` |
| `scripts/camels_square.py` | 304 | driver: screen, train, align, calibrate, rank diagnostics |
| `scripts/slurm/camels_square.sh` | | launcher |
| `degeneracy_distillery/problems/camels.py` | 242 | CAMELS adapter, registered in `problems/base.py` |
| `scripts/camels_sr_basis_sweep.py` | | SR basis sweeps, CPU only, no neural refit |
| `scripts/slurm/camels_sr_basis_sweep.sh` | | |
| `scripts/camels_oneshot_sweep.sh` (slurm) | | rectangular-loss CAMELS runs |
| `scripts/plot_camels_fisher.py` | | Fisher mean/variance heatmaps |
| `scripts/plot_camels_latent_scatter.py` | | latents against their strongest drivers |
| `notes/camels_report.md` | 653 | all results |
| `notes/camels_oneshot_rebuild.md` | | the rectangular-loss rebuild |
| `square_completion_loss.md` | 155 | the handoff note this work implements |

### Modified shared code — do not lose these

| path | change |
|---|---|
| `degeneracy_distillery/oneshot.py` | `Estimator.act` field; `EST_ACTS`; `estimator_bank`; `estimator_factory_for`; `Flattener.delta_init`; **`smooth_leaky` removed from the default bank** |
| `degeneracy_distillery/oneshot_ensemble.py` | `ensemble_calibration`; `estimator_factories`; `delta_init` |
| `scripts/oneshot_sweep.py` | architecture bank wiring; `held_out()` split; `--sr-top-k`; `--sr-screen`; `--refit-pad`; `--delta-init`; Fisher diagnostics; `ensemble.npz` dump; two bug fixes |
| `degeneracy_distillery/problems/base.py` | registers `camels` |

## 3. Dependencies

`square_completion.py` imports only `jax`, `flax`, `numpy`, `math`. It moves
anywhere as-is.

`camels_square.py` does not. It needs:

```
align_coords.process_ensemble_rotation_v2   -> preprocessing_utils
oneshot.estimator_bank, oneshot.spectrum
oneshot_ensemble.ensemble_calibration
problems.base.get_problem, problems.camels.PARAM_NAMES
sr_utils (for the SR sweeps)
```

About 6000 lines. A port should depend on `degeneracy_distillery`, not copy it.

## 4. Results

`/data103/makinen/degeneracy_experiments/`

| directory | size | what |
|---|---|---|
| `camels_square/` | 8 MB | 34 runs: `v2K*` (K-screen), `permK*` (permutation control), `rankK*`, `infoK*`, smokes |
| `camels_oneshot_sweep/` | 761 MB | rectangular-loss runs: `aug20k_*`, `init*`, `full_m7_*` |
| `camels_sr_basis_sweep/` | 116 MB | k-ladder and per-component SR |
| `camels_pilot_fisher.npz` | | 35x35 prior-averaged Fisher used by the screen |

Symlinked at `data_scratch/camels_oneshot_sweep` and
`data_scratch/camels_sr_basis_sweep`. `camels_square` is not symlinked yet.

Each `camels_square` run holds `summary.json` and `ensemble.npz` (aligned
members, `y_std`, per-direction info, `lam_a`).

## 5. How to run

```bash
# square-completed loss, d=12 participating, K active
K_ACTIVE=7 TAG=myrun sbatch scripts/slurm/camels_square.sh

# knobs: D_KEEP K_ACTIVE N_PCA FLOW_LAYERS FLOW_WIDTH BOX_MARGIN
#        PRIOR_REG ENSEMBLE_K STEPS SEED EVAL_EVERY
# MODE=smoke gives 3 members, 300 steps

# rectangular loss, for comparison
MODE=full PRIOR_DRAW=empirical INFO_FLOOR=0.25 REFIT_PAD=0 \
  ENSEMBLE_K=10 SR_N_AUG=20000 sbatch scripts/slurm/camels_oneshot_sweep.sh

# SR sweeps on a finished rectangular run. CPU, no neural refit.
MODE=kladder sbatch --array=0-10 scripts/slurm/camels_sr_basis_sweep.sh
```

GPU is required for training: the JAX CPU backend fails for this pipeline at
d >= 3. The SR sweeps are CPU and run 13-wide with no queue.

## 6. What is established

- The flow is a verified bijection. Log-det matches autodiff to 1e-14, det
  sign is +1 everywhere, and `det = exp(sum s)` with `|s| <= 2` guarantees it.
  This fixes pitfall 1 of the method note; the reference `EtaD` is skip + MLP
  and can fold.
- Held-out NLL is a proper score. This is the first such number in the
  project. Every `nll_gap` in sections 1 to 11 of the report bounds the
  pushforward of eta, not a posterior.
- Calibration works: LOO ratios 1.00 to 1.08 at K=7, better than the
  rectangular loss managed. The data-side ensemble carries over unchanged --
  one flow architecture, a different `t(x)` per member, so `y_std` stays
  defined on the augmented draws.
- **A rank-7 map captures well under half the recoverable information.**
  `K=12` extracts 3.70 nats against `K=7`'s 1.78, held out, with the
  initialisation advantage removed. Section 12.7 and 12.8.

## 7. What is not established

- **`r_hat = 7` is weaker than sections 2 and 4 claim.** All four spectral
  routes derive from a Fisher computed on a map trained with `m = 7`, which
  cannot detect more than 7 directions by construction.
- **The K-screen does not select K.** The models nest, so held-out NLL is
  non-increasing in K. Its fine structure also fails a permutation control.
  Read it as "more statistics keep helping, with diminishing returns".
- **Early stopping fires at step 100 to 200 of 1500**, so results sit close to
  initialisation. Both K-screens are partly measuring initialisation quality.
- The flow reaches ~3.4 nats against the Fisher's 6.31. Cause unknown: the
  Fisher may be optimistic, 819 simulations may be the cap, or `t(x)` may be.
- Everything is one seed.

## 8. Traps

1. **The logit box must strictly contain the data.** `fit_theta_scaler`
   min-max scales to exactly `[1, 2]`, putting the extreme simulation on the
   edge where the logit diverges. Use `--box-margin 0.05`.
2. **The flow overfits 819 simulations.** `-log|det J|` is unbounded below:
   train reached `-28` while held-out reached `+999`. Needs `--prior-reg` and
   early stopping. More regularisation does not raise the ceiling.
3. **`smooth_leaky` grows like `x^2`** as `x -> -inf`; four layers compounds
   to about `x^16`. Removed from the default bank. It was invisible under the
   bounded rectangular loss.
4. **Per-direction information is gauge-dependent.** `eta_a` is defined up to
   `R` in `O(K)`. Use the prior-relative spectrum of `J_a^T J_a`, which is
   invariant -- but note it tracks `K+1`, because averaging rotating rank-K
   subspaces spans more than K.
5. **Zero-initialised couplings hand `eta_a` the first K coordinates**, which
   are the top-K parameters by Fisher score. That flatters low K. Use
   permutations; it costs ~0.8 nats of apparent performance and is honest.
6. **`sed` on `${VAR:-default}` silently fails** -- `$` anchors end-of-line.
   One K-screen ran at 6000 steps when 1500 was intended.

## 9. Next

1. **Commit.** Three weeks of work is untracked.
2. **Gauge-fix `eta_a` and run SR on it.** Section 6 of the method note:
   rotate `R` in `O(K)` to minimise `E ||R J_a||_1`, then read. No SR has been
   run on the square-completed coordinates at all.
3. **Settle the 3.4 vs 6.31 nats gap** on the `unified-sbi` lensing toy, where
   an exact oracle exists. Cheaper than on CAMELS.
4. **Several seeds per arm.** The init effect repeated across two feature
   sets; the NLL effect did not.
5. If the K-screen is wanted, it needs a complexity penalty or a
   non-nesting comparison. As formulated it cannot select K.
