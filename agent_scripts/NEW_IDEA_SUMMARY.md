# One-step degeneracy map (new idea)

This branch holds a one-step alternative to Fishnets then flatten.
The notes live in `new_idea/`. The chat logs that produced them live in this folder.

## Why

The current pipeline first learns a Fisher, then learns a flattener.
That split keeps costly simulations off the symbolic-regression step.
It also forces a square map and an inverse of a near-singular Fisher.
The heater problem hits that limit: rank collapse and a false floor on `Q = I`.

The new idea trains one pair of maps. `eta(theta)` is a data-independent coordinate map. `eta_hat(x)` estimates those coordinates from data.

The joint score is

```
L = E[ 0.5 ||eta(theta) - eta_hat(x)||^2 - log |det J| ]
```

This is the negative log likelihood of

```
q(theta | x) = N(eta(theta); eta_hat(x), I) |det J|
```

when `eta` is a diffeomorphism (`m = d`).
The stationary point is the same as `info_loss`: locally `J^T J ~ F`.
Uninformative directions sit at the prior, without extra Fisher rows.

## Two models

**Hard cut (`main.tex`, `rosenbrock20_oneshot.py`).**
Split `theta = (theta_A, theta_B)` with `|A| = m`.
`eta` maps `theta_A`. The complement stays at the prior `pi_B`.
Under a box prior, `A` must be a coordinate subset.

A group-L1 screen ranks coordinates. The set `A` is the top `m`.
Absolute NLL is then a normalised box posterior.

**Rectangular geometry (`main_mod.tex`, `*_noscreen.py`, GW scripts).**
`eta` maps all of `theta` to `R^m`.
The volume term is `0.5 log det(J J^T)` with rectangular `J`.
There is no `pi_B` term and no hard `A`/`B` cut.
Soft attribution uses Jacobian column norms after the fit.

Absolute geometry NLL is **not** a normalised `-log q(theta | x)` when `m < d`.
It is also **not** a variational bound on `-log p(theta | x)`.
Compare candidates only under the same objective.

## Rank without a sweep

Fit one probe at a generous `m`.
Read the information spectrum `lambda_i = Var_pi(eta_i)`.
Null axes sit near `lambda = 1`.
Set `r_hat` to the count of axes above a nats floor, after you normalise by the measured null.

Check that `r_hat` is stable across a wide floor.
Refit at `r_hat`.
Bootstrap at the probe `m` to confirm the rank. Do not use that spread as `y_std`.

Recommended hybrid when a coordinate complement exists:

1. Rectangular probe, then read the spectrum.
2. Soft-screen column norms to get a candidate set `A`.
3. Hard-cut retrain on that `A`.
4. Check `z`, complement PIT, and leakage.

Do **not** hard-cut when every parameter enters the likelihood and only `m` combinations carry information (heater, often GW). Stay rectangular.

## What can go wrong

- Volume can stretch weak axes. Do not pick `m` by `argmin L`.
- Spare axes are under-penalised without `pi_B`.
- Anything in `ker J` is invisible (the fibre problem).
- A mean-only estimator misses shape parameters. Diagnostics can still pass.
- "All parameters enter" does not remove the fibre. Full rank with `m = d` does.

Guardrails: `z ~ N(0, I)`, spectrum vs floor, column norms, bootstrap vote on `r_hat`, paired candidate scores.

## Files in `new_idea/`

| File | Role |
|---|---|
| `main.tex` | Hard-cut model, box prior, screen, sweep tables |
| `main_mod.tex` | Rectangular geometry procedure that the later scripts follow |
| `heater_oneshot.py` | Heater: one-step map, no square inverse |
| `rosenbrock20_sweep.py` | 20D Rosenbrock, box prior, `m` sweep, MDL |
| `rosenbrock20_overspec.py` | Same problem, over-specified `m` spectrum study |
| `rosenbrock20_oneshot.py` | One probe, hard screen, refit at `r_hat` |
| `rosenbrock20_oneshot_noscreen.py` | Rectangular oneshot, soft Jacobian screen |
| `gw_oneshot.py` | GW oneshot |
| `gw_detection_extrinsics.py` | GW with detection extrinsics |

## Transcripts in this folder

| File | What it covers |
|---|---|
| `cursor_errors_new_idea.md` | First joint-loss design, Rosenbrock 20D, uniform prior, screen, one-shot rank |
| `cursor_errors_new_idea_heater.md` | Same thread with more heater and training-bug detail |
| `cursor_procedure_description_update.md` | `main_mod.tex` rewrite to match the no-screen scripts |
| `cursor_formalism.md` | Same procedure thread, plus: geometry NLL is not a bound, fibre failure, hybrid protocol |

## Results that motivated the protocol

On 20D Rosenbrock with a box prior, a single probe at `m = 6` gave spectrum `[537, 211, 1.08, 1.01, 1.00, 0.99]`.
That read `r_hat = 2`. The refit recovered the true pair and test NLL `30.944` vs floor `30.931`.
All eight bootstrap members agreed on the rank.

A Stiefel retraction on `V` produced NaNs at start (repeated eigenvalues).
Leave `V` unconstrained if you use a Gaussian subspace model.
A box prior cannot use a free `V`. Use a coordinate subset.

The heater "structural rank-1 floor" is likely an optimisation artefact.
The Fisher diagonals use `softplus + 1e-4`. That is not a property of the likelihood.
