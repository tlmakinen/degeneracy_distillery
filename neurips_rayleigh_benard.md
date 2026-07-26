# Rayleigh-Bénard Convection: Intractable-Likelihood Example #4

Companion to [neurips_intractable_examples.md](neurips_intractable_examples.md),
which covers the first three no-tractable-likelihood problems (2D Ising, forced
2D turbulence / Kolmogorov, noisy Kuramoto). This document covers a fourth,
built to the same standard: a forward-simulable problem with no closed-form
likelihood over the observable, a *known* right answer, and the same distillery
pipeline and gate conventions.

Script: [scripts/rayleigh_benard_notebook_run.py](scripts/rayleigh_benard_notebook_run.py)

This supersedes the shelved `experiment_rayleigh_benard.ipynb` notebook, which
used a Grossmann-Lohse *surrogate* closure. A surrogate is an algebraic
root-find, i.e. effectively a tractable likelihood, so it could not carry the
intractable claim. This version is a real direct numerical simulation.

## Why add it when Kolmogorov already covers fluids

Two reasons, and only the second is new relative to Kolmogorov.

1. It is a second, physically distinct intractable-likelihood fluid problem
   (thermal convection rather than forced shear turbulence), so the fluids claim
   does not rest on a single simulator.

2. **It is three-parameter and carries a weak, nearly degenerate direction.**
   Kolmogorov is two-parameter with one identifiable coordinate. Ising and
   Kuramoto are two- and three-parameter. Rayleigh-Bénard is the cleanest place
   to answer the objection that the method only works when every parameter is
   tightly identifiable: here the observable is *nearly blind* to one of the
   three parameters, and the method isolates that as the flat direction while
   still recovering the governing coordinate.

## The problem

`theta = (log10 Ra, log10 Pr, log10 Gamma)` — Rayleigh number, Prandtl number,
aspect ratio. We solve the 2D Boussinesq equations in vorticity-streamfunction
+ temperature form on a box `[0, Gamma] x [0, 1]` (thermal-diffusion units,
layer height 1),

```
d_t omega + u.grad omega = Pr lap omega + Ra*Pr d_x theta
d_t theta  + u.grad theta = lap theta + w
lap psi = -omega,   u = d_z psi,   w = -d_x psi
```

with **stress-free, isothermal** boundaries at `z = 0, 1`. Under those boundary
conditions `omega`, `theta` and `psi` all vanish on the plates and are
represented as a sine series in `z` (DST) and Fourier in `x`. That keeps the
whole solver FFT/matmul based and GPU-batchable exactly like the Kolmogorov
script — no Chebyshev, no MPI, no Dedalus. Stress-free is a standard,
recognised RB configuration (it is the one most of the theoretical literature
uses), it reaches a statistically steady state, and it has an exact linear onset
we use to validate the solver.

Aspect ratio is the nondimensional box width, so varying `Gamma` varies the
horizontal wavenumber grid. The observable is therefore binned by **physical**
wavenumber, not grid index, so it stays a fixed length across the prior.

### Observable

Per simulation: the time-averaged, radially-binned, **separately normalised**
thermal-variance and enstrophy shell spectra in the statistically steady state.
Normalising each spectrum to unit sum divides out the amplitude, so no
dimensional heat-flux number is handed to the network — the Nusselt and Reynolds
scalings must be *recovered* from spectrum shape alone. `Nu = 1 + <w theta>` and
a Reynolds proxy `Re = u_rms / Pr` are computed alongside purely as the
known-answer targets for the recovery gate; they are **never** part of the
network input. This is the same discipline as Kolmogorov, where the amplitude is
divided out so the observable depends on the parameters only through `Re`.

## The honest degeneracy story

This is the one subtlety worth stating plainly in the paper, because it is more
interesting than a clean textbook nondimensionalisation.

The observable is controlled primarily by `Ra` (convective vigour), secondarily
by `Pr`, and is **nearly blind to `Gamma`**. So `Gamma` is the flat direction
*of the observable*.

Separately, the *true* Nusselt number here scales as roughly
`Nu ~ Ra^0.51 Pr^0.04 Gamma^1.10` (DNS-measured, see below): at the small
aspect ratios in this prior (1 to 2), `Nu` genuinely depends on `Gamma` through
a finite-size effect on the roll count. So the full `Nu(Ra, Pr, Gamma)` gradient
has a real `Gamma` component that the observable cannot see.

The consequence for the gate: the recovery check for the Nusselt coordinate is
taken in the **identifiable `(Ra, Pr)` subspace only**. We verify that the
discovered Nusselt coordinate has the correct *thermodynamic* scaling, and we
report `Gamma`'s flatness separately as the discovered degeneracy rather than
penalising the coordinate for not recovering a dependence the data does not
contain. Pushing `Gamma` to larger aspect ratios would recover the pristine
"`Nu` depends only on `Ra`, `Pr`" story at higher grid cost; it is a knob, not a
requirement.

## Simulator validation already done

These come from direct probes of the solver, not the full pipeline, and are
worth rechecking on the cluster if you change any prior.

**Linear onset (the decisive correctness check).** 2D stress-free RB has an
*exact* onset at `Ra_c = 27 pi^4 / 4 = 657.51`. The solver reproduces the linear
dispersion relation to three decimals across `Ra` 500 to 1000 and crosses zero
growth-rate exactly at `Ra_c` (measured after the fast-decaying eigenmode
transient is allowed to die). This validates the buoyancy coupling, the signs,
the diffusion handling and the sine/cosine transforms together.

**Nonlinear convection.** With a finite-amplitude initial perturbation the
convection saturates and `Nu` plateaus by about 25 free-fall times:
`Nu = 2.66, 4.19, 5.42` at `Ra = 3000, 8000, 15000` (`Pr = 1`, `Gamma = 1.5`),
a clean monotonic Nusselt-Rayleigh scaling. Two solver details matter and
should not be tidied away:

- The initial thermal perturbation is finite amplitude (`O(0.3)`), not
  infinitesimal. From a tiny kick the flow spends `O(20)` free-fall times in the
  linear-growth phase before the rolls saturate, and the spin-up budget would be
  wasted on growth rather than on developed convection. This is the direct
  analogue of the Kolmogorov script starting from the laminar solution plus a
  relative perturbation instead of from rest.
- Spin-up is `~30` free-fall times (`spin_steps`). Too short and `Nu` sits at 1
  (no heat transport) and the whole known-answer gate collapses, because a
  Nusselt number pinned at 1 carries no `Ra` dependence.

**Identifiability (cross-validated ridge `R^2` from the raw observable, a
pessimistic lower bound on what the Fishnet ensemble extracts):**

- `log Ra` 0.93, `log Nu` 0.95, `log Re` 0.71, `log Pr` 0.49, **`log Gamma`
  0.01**. The near-zero `Gamma` number is the point: the observable is blind to
  aspect ratio, which is exactly the flat direction the method should find.

**DNS Nusselt scaling** (regression of `log10 Nu` on the three log parameters):
`slope_log_ra = 0.51`, `slope_log_pr = 0.04`, `slope_log_gamma = 1.10`.

## Smoke-run results

Run end to end at `--mode smoke` on CPU (100 training simulations, `40x28`
grid, 25 s of SR per component, gates relaxed so the whole pipeline runs). Every
stage completed and all artifacts were written. Treat the numbers as evidence
the plumbing works and the recovery is plausible at a tiny budget, not as
results — full mode uses `96x64`, 500 simulations and 300 s of SR.

- **The gate passes even at smoke budget.** Best Nusselt correlation 0.92 (the
  default threshold is 0.90); Nusselt `(Ra, Pr)` gradient cosine 0.997 (the
  discovered coordinate has the correct thermodynamic scaling direction).
- The three discovered coordinates were: `eta_0` a function of `log Ra` and
  `log Gamma` tracking `Nu`/`Ra` (corr 0.92 / 0.92); `eta_1 = 0.228 log Pr`,
  a **clean isolated Prandtl coordinate** (corr 1.00 with `log Pr`); `eta_2`
  another `Ra`-aligned coordinate (corr 0.96 with `log Ra`). No coordinate
  tracks `Gamma` strongly (best 0.38) — the weak direction is left flat.
- **Held-out geometry** `||Q - I||_F`: neural map 0.94 against 1.43 for the raw
  parameters, so the flattening network clearly improves the geometry. But the
  SR expressions came out at 2.42, *worse* than raw. **This is the same
  SR-is-the-bottleneck pattern the other three problems show in smoke** — the
  flattening network finds a good coordinate map and SR fails to express it at a
  25 s budget. Note the discovery still passes: the SR expressions correlate
  well with `Nu` and have the right `(Ra, Pr)` scaling even though their raw
  flatness is poor. Escalate `--sr-time-limit` before anything else on the full
  runs.
- Runtime split (CPU): 175 s simulation, 27 s fishnets, 25 s flatten, 79 s SR,
  307 s total. On a GPU the simulation and fishnet stages collapse.

## Gate and how a successful run reads

`run_summary.json` records `discovery.success`, which is `True` when both:

- some coordinate tracks `log Nu` at correlation `>= --min-nusselt-corr`
  (default 0.90), and
- that coordinate's `(log Ra, log Pr)` gradient aligns with the DNS-measured
  Nusselt scaling at cosine `>= --min-nusselt-cosine` (default 0.90).

Reported alongside, not gated: the DNS Nusselt scaling exponents, the Reynolds
correlation, `Gamma`'s (low) correlation as the discovered flat direction, and
the held-out flatness for raw / ad-hoc / MDL / pruned / neural coordinates.

The sentence to aim for in the paper:

> On 2D Rayleigh-Bénard convection with no tractable likelihood, from normalised
> turbulence spectra alone and with no heat-flux number supplied, the method
> recovers a coordinate aligned with the Nusselt number whose Rayleigh-Prandtl
> scaling matches the DNS, and isolates aspect ratio as the flat direction.

## How to run

Smoke first, on one node, to confirm the environment. On a machine without a
CUDA GPU (e.g. Apple Metal, which JAX does not report as a `gpu` backend) pass
`--no-require-gpu`, and relax the gates so a smoke run exercises the whole
pipeline instead of exiting at the gate:

```bash
python scripts/rayleigh_benard_notebook_run.py --mode smoke --master-seed 0 \
    --out-dir /tmp/smoke/rayleigh_benard --no-require-gpu \
    --min-nusselt-corr 0 --min-nusselt-cosine 0
```

Then the seed campaign, one GPU job per seed:

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/rayleigh_benard_notebook_run.py --mode full --master-seed $seed \
      --out-dir results/rebuttal_discovery/rayleigh_benard/seed_$seed
done
```

Overrides that avoid a code edit: `--sr-time-limit` (the first escalation),
`--nx` / `--nz` (raise if the cutoff-shell warning fires), and `--cfl` (lower if
the solver goes non-finite).

## Parameter scaling

Consistent with the other three scripts: `theta` is scaled to `[1, 2]` before
the fishnet stage via `fit_theta_scaler(..., feature_range=(1.0, 2.0))`, nothing
between the fishnet stage and SR shifts the range, and
`expressions_to_physical(..., sr_offset=0.0)` is therefore correct. Every target
here is a log or a ratio (`Nu(Ra, Pr)`, `Re(Ra, Pr)`), so keeping the scaled
inputs strictly positive and `O(1)` lets SR form quotients and logs cleanly.

## Risks and knobs

- **The full runs have not been executed** — only smoke, on CPU. The simulator
  is validated against exact linear theory and the nonlinear Nusselt scaling,
  and smoke exercises every stage, but the full `96x64` budgets and the recovery
  thresholds at full budget are untested. Run one full seed and inspect the
  input-summary figure and `run_summary.json` before launching all ten.
- **`Nu` pinned near 1.** Spin-up is too short or the IC amplitude too small.
  Both are set in the config; do not lower them to save time without checking
  the median `Nu` printed during simulation stays well above 1.
- **Solver goes non-finite.** Plume boundary layers are unresolved at the top of
  the `Ra` prior. Raise `--nz` / `--nx` or lower `--cfl`; the script warns
  automatically when the top thermal shell fills up.
- **SR is the most likely failure point at full budget**, as in the other three
  problems. Compare `heldout_geometry.nn` against `heldout_geometry.pruned`: if
  `nn` is small and `pruned` is not, the geometry is fine and the SR budget is
  the problem — raise `--sr-time-limit`.
- **The `Gamma` story.** If a reviewer wants the pristine "aspect ratio is a
  pure nuisance for `Nu`" claim, widen the `Gamma` prior to larger aspect ratios
  (and raise the grid). At `Gamma` in `[1, 2]` the finite-size `Nu(Gamma)`
  dependence is real and is handled by restricting the gate to the `(Ra, Pr)`
  subspace.
