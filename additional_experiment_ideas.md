# Additional Experiment Ideas: Intractable-Likelihood Discovery

Third document in the set:

- [notes/neurips_discovery_reruns.md](neurips_discovery_reruns.md) — rerunning the
  existing experiments (Rosenbrock, SIR, TaylorF2, IMRPhenomD, QM7b) across seeds.
- [notes/neurips_intractable_examples.md](neurips_intractable_examples.md) — the
  three new intractable-likelihood problems that are built and validated.
- **This document** — the Lennard-Jones proposal written up in full, the other
  candidates we considered and rejected, and the transferable lessons from
  building the first three.

All of this exists to answer one reviewer objection: that the method needs a
tractable likelihood. It does not, and the strongest reply is a set of problems
whose likelihoods are unavailable for *different* reasons, so the objection
cannot be dismissed as one special case.

## Status

Built, simulator-validated, two of three run end to end:

- **2D Ising** — intractable partition function `Z(J, T, h)`. Recovers `J/T` and
  `h/T` from raw spin configurations. One exact degeneracy (the scaling ray).
- **Kolmogorov flow** — chaotic PDE trajectory, no density in closed form.
  Recovers `Re = f0/(nu^2 k_f^3)` from turbulence spectra. One exact degeneracy.
- **Noisy Kuramoto** — latent natural frequencies, so the likelihood would need
  marginalising over `N` latents and the Brownian paths. Recovers the ratio
  basis `K/sigma`, `D/sigma`, `sigma`. No exact degeneracy; curved geometry.

Proposed, not built:

- **Lennard-Jones fluid** — intractable configurational partition function.
  Two exact degeneracies in four parameters. Detailed below.

## Lennard-Jones fluid

The original note on this was "excellent but more setup," and that remains the
honest summary. It is the best *concept* of the four and the worst fit to the
rebuttal schedule.

### The system

Pair potential, truncated and shifted at `r_c = 2.5 sigma`:

```
U(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ]
```

Parameters `theta = (eps, sigma, T, rho)`: well depth, collision diameter,
temperature, number density.

### Why the likelihood is unavailable

The configurational partition function

```
Z = (1/N!) integral over V of d^{3N}r exp(-beta U({r}))
```

is a `3N`-dimensional integral over a strongly correlated integrand. This is the
textbook intractable normaliser, and it is a strictly harder case than Ising:
continuous rather than discrete, off-lattice, and with no exact solution in any
limit. A reviewer who wants to argue the Ising normaliser is "just a sum" has
nothing to say here.

### What the method should discover

The law of corresponding states. Define reduced temperature and density

```
T* = k_B T / eps        rho* = rho sigma^3
```

Every equilibrium structural property, expressed in reduced units, depends on
the four parameters *only* through these two combinations. Four parameters, two
identifiable coordinates, **two exact degeneracy directions**.

That is the reason to want this experiment. Everything built so far has at most
one degeneracy; nothing in the paper currently demonstrates the method resolving
a two-dimensional null space. It also raises the parameter count to four, which
answers the unspoken worry that the approach only works in two or three
dimensions.

Both targets are sharp:

- `log T - log eps` — a ratio, structurally like Ising's `J/T`.
- `log rho + 3 log sigma` — a power law with **exponent 3**, testable the same
  way the Reynolds exponent of -2 is tested in the Kolmogorov script. Fit each
  discovered coordinate against `(log eps, log sigma, log T, log rho)` and check
  the slope on `log sigma` comes out at three times the slope on `log rho`. That
  is a falsifiable number, not a correlation.

### The observable is the crux

Get this wrong and the experiment is meaningless. If the observable is expressed
in reduced units, `sigma` has been divided out by hand and the answer has been
smuggled in.

**Recommended**: the static structure factor `S(k)` sampled at wavevectors in
units of the mean interparticle spacing `rho^(1/3)`. Density is a control
variable the experimenter sets, so nondimensionalising by it is legitimate,
whereas nondimensionalising by `sigma` is not. Both axes are then dimensionless
and the observable depends only on `(T*, rho*)`, giving the clean two-degeneracy
structure.

**Worth running as an ablation**: the radial distribution function `g(r)` on an
absolute grid in lab units. Now the first-peak position at roughly `1.12 sigma`
makes `sigma` separately identifiable, so there are three identifiable
coordinates and only one degeneracy.

The pair is more interesting than either alone. Same physics, same parameters,
two different measurements, and the degeneracy structure changes. That is a
concrete demonstration that the Fisher geometry the method recovers is a
property of the *experiment*, not just the model — which is a genuinely useful
point for the paper and one that no current experiment makes.

### Implementation notes

**Simulate in reduced units.** Map `(eps, sigma, T, rho)` to `(T*, rho*)`, run
the reduced-unit simulation, emit the dimensionless observable. The inference
network never sees `eps` or `sigma`, so the degeneracy holds by construction to
machine precision. This is not cheating: the simulator is entitled to know its
own parameters, exactly as the Ising script computes `J/T` and `h/T` internally
because that is literally what the Boltzmann weight depends on. A useful side
effect is that the simulation only ever explores a two-dimensional space.

**The cutoff must be at `2.5 sigma`**, fixed in reduced units. A cutoff fixed in
lab units breaks corresponding states and degrades the degeneracy from exact to
approximate. This is the same class of error as the Kolmogorov spin-up bug: a
detail that looks cosmetic and silently invalidates the experiment.

**Integration.** Velocity Verlet with a Langevin thermostat, `dt = 0.005 tau`
where `tau = sigma sqrt(m/eps)`. Integrating in reduced units means every run
advances the same reduced time automatically, which is the trick that had to be
added by hand for Kolmogorov. Mass is irrelevant because the observable is
static structure, so fix `m = 1`.

**Time-average the observable.** A single configuration's `S(k)` is dominated by
realisation noise. Accumulate over several hundred decorrelated configurations.
This was worth roughly a factor of ten in signal-to-noise for Kolmogorov and
will matter at least as much here.

**Keep the prior supercritical.** The LJ critical point sits near `T* = 1.32`,
`rho* = 0.31`. Straying into the two-phase region causes phase separation, very
slow equilibration, and a bimodal observable — the same multimodality trap the
Ising script avoids by keeping the field strictly positive. Suggested box:
`T*` in `[1.5, 3.0]`, `rho*` in `[0.4, 0.85]`. Then either draw `(T*, rho*)` in
that box and invert to lab parameters, or draw the four lab parameters and
reject draws that fall outside it.

### Cost, honestly

This is the expensive one. With `N = 256` to `512` particles in 3D, naive
`O(N^2)` forces are actually GPU-friendly when batched over simulations, but the
arithmetic is unforgiving: 1000 simulations at 10,000 steps with `N = 256` is
roughly `6e11` pair evaluations, which is hours on a single A100 rather than
minutes.

Three ways to make it fit:

- Use `jax-md`, which is JAX-native and has neighbour lists. Cleanest, but adds
  a dependency that the other scripts do not have.
- Drop to `N = 128` and compensate with more time samples. Finite-size effects
  will distort `S(k)` at small `k`; truncate the low-`k` bins rather than
  pretending they are clean.
- Halve the simulation budget to 250 train and 250 evaluation.

Recommendation: fourth priority. Build it if the first three land and there is
time; otherwise describe it in the paper as the planned four-parameter,
two-degeneracy extension. Do not let it delay the seed campaign on the three
that already work.

## Candidates considered and set aside

From the original brainstorm, with the reason each lost:

- **Vicsek active matter** — discover the combination of noise, interaction
  radius and density that controls flocking. Visually the most striking option.
  Rejected because the "correct" symbolic coordinate is not uniquely established
  in the literature, so there is no clean ground truth to validate against, and
  a reviewer could reasonably dispute whatever the method returns.
- **Stochastic Lotka-Volterra via Gillespie** — a classic likelihood-free
  benchmark; would discover rate ratios and characteristic timescales. Easy to
  implement and unimpeachably intractable, but too familiar to read as a
  discovery result. Keep in reserve as a fast substitute if one of the three
  proves unstable.
- **Gray-Scott reaction-diffusion** — pattern morphology from random initial
  conditions, discovering the diffusion and reaction combinations that control
  it. Excellent figures, but the ground truth is weak in the same way as Vicsek.
- **M/G/1 queue** — a canonical SBI benchmark, very fast. Rejected purely on
  narrative grounds: no scientific-discovery appeal, which is the entire point
  of the reframing.

One general rule came out of this and is worth keeping: **avoid any further
deterministic simulator with additive Gaussian noise.** The skeptical reviewers
will correctly point out that such a likelihood is evaluable, and the objection
we are trying to kill comes straight back.

## Lessons from building the first three

These cost real debugging time and transfer directly to Lennard-Jones or
anything else added later.

1. **Probe identifiability before building the pipeline.** A cross-validated
   ridge regression from the raw observable onto each target coordinate takes
   minutes and is a pessimistic lower bound on what the network can extract.
   Kuramoto's noise parameter scored `R^2 = -0.01` from the order parameter
   trace alone — completely unidentifiable. Without that check it would have
   failed deep in the pipeline where the cause is invisible.

2. **Time-average anything chaotic.** A single snapshot of a turbulent flow or a
   fluid configuration is realisation noise with a small systematic signal on
   top. Averaging took the Kolmogorov same-Reynolds scatter from comparable to
   the signal down to roughly a tenth of it.

3. **Nondimensionalise the integration schedule, not just the analysis.**
   Kolmogorov runs were reaching wildly different physical states because
   spin-up takes `O(Re)` turnover times. Sizing the step from the velocity scale
   so every run advances equal reduced time fixed it. Lennard-Jones gets this
   free by integrating in reduced units.

4. **Keep the prior out of degenerate regions.** Saturated or multimodal regimes
   look identical to a broken method in the summary statistics. The Ising field
   is strictly positive to avoid symmetry breaking, and the coupling range stops
   short of where magnetisation pins at one and stops carrying information.

5. **Put the symmetry in the embedding.** A dense encoder on 2048 raw spins has
   about 524k parameters and overfits within 200 epochs at these budgets. A
   convolutional encoder pooling over space and then over snapshots has 20k,
   is exactly permutation and translation invariant, and trains an order of
   magnitude longer before stopping.

6. **Use an exponent test, not just a correlation.** Correlations against the
   target coordinates are inflated because the targets are themselves correlated
   over a box prior — `log Re` and `log U` correlate at 0.84 simply by
   construction. The log-log slope ratio (-2 for Reynolds, 3 for `rho sigma^3`)
   is the falsifiable quantity and the one to quote.

7. **Watch which stage actually failed.** In the Kolmogorov smoke run the
   flattening network reached a held-out flatness of 0.19 against 0.99 for raw
   parameters, while symbolic regression came in at 1.59, worse than doing
   nothing. The geometry was fine and SR was the bottleneck. Comparing
   `heldout_geometry.nn` against `heldout_geometry.pruned` in the summary JSON
   separates these two failure modes immediately, and they call for completely
   different fixes.
