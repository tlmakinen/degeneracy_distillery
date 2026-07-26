# Results Tables: Intractable-Likelihood Discovery

Two tables here are paste-ready now, and one is a template to fill from the
cluster runs. Read the warning before pasting anything.

> **Do not paste the smoke numbers into the rebuttal.** They come from a single
> seed at roughly a third of the full simulation budget, two of the three runs
> were competing for CPU with duplicate processes, and the learned expressions
> are not yet in a form that visibly matches the textbook answer. See
> "Do you need to re-run?" at the bottom. The answer is yes.

## Table 1: Experiment design (paste-ready, does not depend on run results)

| Problem | Parameters | Textbook coordinates | Exact degeneracies | Sharp recovery test |
|---|---|---|---|---|
| 2D Ising | `(J, T, h)` | `J/T`, `h/T` | 1 (scaling ray `(λJ, λT, λh)`) | correlation with `log(J/T)`, `log(h/T)` |
| Kolmogorov flow | `(f0, ν)` | `Re = f0/(ν² k_f³)` | 1 | log-log slope ratio `= −2` |
| Noisy Kuramoto | `(K, σ_ω, D)` | `K/σ_ω`, `D/σ_ω`, `σ_ω` | 0 (curved, no null space) | cosine to ratio basis |
| Lennard-Jones (proposed) | `(ε, σ, T, ρ)` | `k_BT/ε`, `ρσ³` | 2 | log-log slope ratio `= 3` |

Why the likelihood is unavailable in each case, which is the point of the set:
Ising is normalised by an intractable partition function `Z(J,T,h)`; Kolmogorov
observables are functionals of a chaotic PDE trajectory with no density in closed
form; Kuramoto has latent natural frequencies that would need marginalising along
with the Brownian paths; Lennard-Jones has a `3N`-dimensional configurational
integral. Four different reasons, so the objection cannot be dismissed as one
special case.

## Table 2: Smoke-run status (internal only — plumbing check, not results)

Single seed, reduced budgets, CPU. Full mode uses 500 simulations, 20 fishnets
and 300 s of symbolic regression per component.

| Problem | Sims | Best correlation | Sharp test | Flatness: raw / textbook / neural / symbolic |
|---|---|---|---|---|
| Ising | 200 | `J/T` 0.95, `h/T` 0.60 | n/a | 1.35 / 1.69 / 0.92 / 1.31 |
| Kolmogorov | 100 | `log Re` 0.92 | exponent −0.93 (want −2) | 0.99 / 0.93 / 0.19 / 1.59 |
| Kuramoto | 200 | `K/σ` 0.82, `D/σ` 0.69 | cosines 0.82 / 0.53 / 0.98 | 1.15 / 1.16 / 0.95 / 0.82 |

Flatness is the median Frobenius distance of the transformed Fisher from the
identity, so lower is better. "Textbook" is the hand-written dimensionless
reduction, "neural" is the learned coordinate map, "symbolic" is the distilled
expression.

Two patterns are already visible and both are likely to survive the full runs:

**Symbolic regression is the bottleneck, not the geometry.** On Kolmogorov the
neural map reached 0.19 against 0.99 for raw parameters, and symbolic regression
then came in at 1.59 — worse than doing nothing. Ising shows the same gap (0.92
neural, 1.31 symbolic). Only Kuramoto had SR improve on the neural map.

**The textbook reduction does not flatten the geometry.** On Ising it is worse
than the raw parameters (1.69 vs 1.35) and on Kuramoto it is a wash (1.16 vs
1.15). If this holds up it is a better rebuttal line than anything about
efficiency, because it directly answers "isn't this just nondimensionalisation?"

## The learned expressions, as they currently stand

This is the honest reason not to paste yet. Ising, where the textbook answer is
`J/T` and `h/T`:

```
η₁ = 0.265·T + (log|2.237·h + 0.884| + 5.222)·(0.062·J − 0.025·T + 0.02) − 0.487
η₂ = (0.448·J + 0.054·T − 1.826·(log|2.237·h + 0.884| − 0.437)² + 0.053)/(0.716·T − 0.289)
η₃ = 0.999·log|(1.042·J + 0.501·h − 2.561·(0.716·T − 0.289)^0.262 + 0.372)²| + 0.001
```

Kolmogorov, where the answer is `log f0 − 2·log ν`:

```
η₁ = (0.497·f0 − 21.176·ν + 0.183)/(0.61·f0 + 0.477)
η₂ = 43.104·ν − 0.332
```

The correlations are respectable but the symbolic forms are not recognisable as
the textbook coordinates — `η₂` for Kolmogorov is essentially just `ν`. A
reviewer shown these next to a column headed "`Re = f0/(ν²k_f³)`" would conclude
the method does not work. The correlation numbers cannot carry a claim the
expressions visibly contradict.

## Table 3: Template for the real results

Fill from `run_summary.json` across the seed campaign.

| Problem | Train sims | Recovered | Median correlation | Sharp test (median) | Flatness: raw / textbook / learned |
|---|---|---|---|---|---|
| 2D Ising | 500 | ?/10 | | n/a | |
| Kolmogorov flow | 500 | ?/10 | | exponent, target −2 | |
| Noisy Kuramoto | 500 | ?/10 | | cosine, target 1 | |

Target sentence:

> Across 10 independent trials on three problems with no tractable likelihood,
> each using 500 training simulations, the method recovered the known governing
> coordinates in X/10, Y/10 and Z/10 runs. Symbolic regression used 2,000
> additional evaluations of the learned coordinate map and no further simulator
> calls.

## Do you need to re-run?

Yes, and the smoke numbers should not be used even as a provisional stand-in.
Five reasons, in order of how much they matter:

1. **The expressions are not presentable.** This is the blocker. A results table
   whose whole purpose is to show learned coordinates matching textbook ones
   cannot ship with expressions that do not match. Everything else is fixable by
   waiting; this one decides whether there is a table at all.

2. **One seed proves nothing about reliability.** The claim being made is a
   recovery rate — "the nonlinear expressions pop every time". That is a
   statement about the distribution over seeds, and a single run cannot support
   it regardless of how good it looks.

3. **The budgets are a third of full.** Ising ran 200 simulations on a 12² lattice
   with 4 snapshots; full mode is 500 on 16² with 8. Kolmogorov ran 100
   simulations on a 32² grid averaging 12 spectra; full is 500 on 64² averaging
   28. Symbolic regression had 45 s per component against 300 s. The identifiability
   probe predicts Ising's `h/T` alone should move from about 0.55 to 0.73 on the
   snapshot count change.

4. **Two runs were CPU-contended.** Three duplicate Ising processes were
   competing for cores, so even the training those runs did get was degraded.

5. **CPU is the wrong hardware.** The Ising run spent 5997 of 6171 seconds in the
   fishnet ensemble purely because convolutions are slow without a GPU.

What the smoke runs did establish: every stage of all three pipelines executes
end to end, writes its artifacts, and computes its diagnostics correctly. That
was their job and they did it.

### What to change before the real runs

Beyond simply using `--mode full` and ten seeds:

- **The symbolic regression budget is already raised.** Because SR was the
  bottleneck in two of three problems, the full-mode default is now 300 s per
  component rather than 120 s. Escalate further with `--sr-time-limit` if the
  expressions still come back unreadable.
- **Run one seed per problem and inspect the expressions before launching all
  thirty.** Specifically compare `heldout_geometry.nn` against
  `heldout_geometry.pruned`. If the neural map is flat and the symbolic one is
  not, the geometry is fine and only the SR stage needs more budget — a very
  different fix from anything touching the simulator or the network.
- **Consider raising Ising's `--n-snapshots` to 16.** It is the cheapest
  available improvement to the weakest coordinate and costs only Metropolis
  sweeps.
