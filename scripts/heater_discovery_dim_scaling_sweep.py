"""Discovery-at-every-dimension scaling sweep for the chain-product heater.

The simulator is the same generalisation of the rank-1 heater used by
``scripts/heater_dim_scaling_sweep.py``::

    y(t) = (theta_1 * theta_2 * ... * theta_d) * (1 - exp(-t / tau)) + noise(t)

so only the scalar product ``P = prod_i theta_i`` enters the data and the
intrinsic dimension of the data manifold is exactly 1 for every ``d``.

Difference from ``heater_dim_scaling_sweep.py``
----------------------------------------------
That script supplies the 1-D distilled coordinate *analytically* at every
``d`` (see its own note: "we use an *analytic* distilled coordinate rather
than re-running the Distillery for every d"), and demonstrates discovery
separately at ``d = 2`` in ``scripts/heater_minimal_distillery.py``. The
scaling claim is therefore conditional on already knowing the answer.

This script instead runs the **full discovery pipeline independently at
every d** -- Fishnet ensemble, flattening flow, Procrustes alignment,
Fisher-based rank selection, and symbolic regression -- and trains the
downstream NPE on the coordinate the pipeline actually found.

Neither of the two existing heater scripts is imported or modified, so the
submitted numbers remain reproducible.

Three NPE arms per run
----------------------
``raw``
    Target is ``theta`` in R^d.
``analytic``
    Target is the 1-D standardised log-product. This is the oracle ceiling,
    retained so the discovered arm can be quoted as a fraction of the
    achievable gain.
``discovered``
    Target is the symbolic coordinate found at this ``d``, standardised
    using prior draws only.

All three arms use the **same** density estimator (an MDN by default). The
existing sweep gives the raw arm a MAF and only the distilled arm an MDN,
which confounds coordinate with architecture as soon as the discovered
target is also low-dimensional: an autoregressive flow on a 1-D target
degenerates to a stack of conditional scalar bijectors with nothing to
autoregress over, and mis-localises sharp posteriors. Holding the estimator
fixed means any surviving gap is attributable to the coordinate.

Evaluation axes
---------------
Log-probabilities on different coordinates are not comparable, and the
discovered coordinate differs across seeds and across ``d``. Both arms of
each comparison are therefore evaluated on one common scalar axis, and only
where the push-forward is exact:

* analytic axis -- ``raw`` (theta samples pushed through the analytic
  projector) against ``analytic`` (identity);
* discovered axis -- ``raw`` (theta samples pushed through the discovered
  symbolic expression) against ``discovered`` (identity).

The discovered-axis comparison uses no oracle knowledge. The analytic
coordinate is used only for *scoring* and for reporting the rank
correlation between the two axes; it is never used to train the discovered
arm.

Rank selection
--------------
Two independent steps that must agree:

1. ``r`` = number of eigenvalues of the prior-normalised mean theta-space
   Fisher above ``--rank-floor-rel`` times the largest (correct answer here
   is 1 at every ``d``);
2. each eta axis is scored by the fraction of its Jacobian energy lying in
   the top-``r`` Fisher eigen-subspace, and the ``r`` highest-scoring axes
   are kept.

Step 2 deliberately avoids relying on any axis-ordering convention:
``nonlinearity_rotation`` orders by descending nonlinearity energy while
``fisher_order_canonicalize(mode="permute_and_sign")`` puts the largest
Fisher eigenvalue *last*, so assuming a fixed position would risk fitting
SR to pure regulariser noise. Axes are *not* selected by nonlinearity
energy, because the ``d - 1`` junk axes have Jacobians driven by the
regulariser and can carry more apparent nonlinearity than the true axis
once ``d`` is large.

SR is fitted only to the surviving axes via ``components_to_fit`` with
``slice_fisher=False`` -- slicing would symmetrically drop theta inputs,
but the surviving coordinate depends on all ``d`` of them.

Success criteria (fixed before running)
---------------------------------------
``rank_correct``
    the rule returns ``r == --expected-rank`` (1 for this simulator).
``symbolic_recovered``
    Spearman ``|rho|`` between the discovered coordinate and the analytic
    log-product axis, on held-out draws, at or above
    ``--recovery-corr-thresh``.

Spearman is primary because any monotone function of the product carries the
same information. Pearson against both ``P`` and the standardised
log-product is also recorded: high Spearman with low Pearson means SR found
a monotone but differently-curved representative.

Outputs
-------
* ``metrics.csv`` / ``metrics.npz`` -- one row per (nsims, d, trial),
  rewritten after every run so a job killed by a wall-clock limit loses
  nothing.
* ``recovery_table.csv`` -- the rebuttal table: per ``d``, rank and symbolic
  recovery counts with denominators, median ``|rho|``, median complexity.
* ``metrics_aggregate.csv`` -- per (nsims, d) mean/std/sem/n of the NPE
  columns on both evaluation axes.
* ``rank_spectra.npz`` -- full Fisher eigenvalue and nonlinearity spectra per
  run, so ``--rank-floor-rel`` can be revisited without rerunning.
* ``expressions.json`` -- discovered expressions and complexities per run.
* ``manifest.json`` -- configuration, written at start and refreshed at end.

Usage
-----
Cheap rank-rule ablation over the Fisher regulariser (no SR, no NPE, and no
torch/ltu-ili import)::

    for nz in 1e-4 1e-3 1e-2 1e-1; do
      python scripts/heater_discovery_dim_scaling_sweep.py \\
        --dims 2 3 4 6 8 10 12 --num-trials 3 --nsims 1000 \\
        --rank-only --flatten-noise "$nz" \\
        --out-dir heater_rank_ablation_noise"$nz"
    done

Full end-to-end sweep::

    python scripts/heater_discovery_dim_scaling_sweep.py \\
        --dims 2 3 4 6 8 10 12 --num-trials 10 --nsims 1000 \\
        --flatten-noise 1e-3 --sr-time-limit 300 \\
        --out-dir heater_discovery_scaling_v1

Per-run failures are recorded in ``metrics.csv`` with the failing stage and
the sweep continues. ``--resume`` skips (nsims, d, trial) combinations
already present.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Simulator
# =============================================================================

@dataclass
class ChainHeaterCfg:
    """Configuration for the chain-product heater simulator."""

    theta_min: float = 1.0
    theta_max: float = 2.0
    tau: float = 1.0
    t_max: float = 4.0
    n_t: int = 20
    sigma: float = 0.2

    @property
    def thermal_kernel(self) -> np.ndarray:
        t = np.linspace(0.0, self.t_max * self.tau, self.n_t, dtype=np.float32)
        return (1.0 - np.exp(-t / self.tau)).astype(np.float32)


def chain_dataset(
    n: int, d: int, cfg: ChainHeaterCfg, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``(theta, y)`` for ``y = (prod_i theta_i) * kernel + noise``."""
    theta = rng.uniform(cfg.theta_min, cfg.theta_max, size=(n, d)).astype(np.float32)
    power = np.prod(theta, axis=1).astype(np.float32)
    mean = power[:, None] * cfg.thermal_kernel
    noise = rng.normal(scale=cfg.sigma, size=mean.shape).astype(np.float32)
    return theta, (mean + noise).astype(np.float32)


def log_theta_moments_uniform(a: float, b: float) -> tuple[float, float]:
    """Analytic mean and variance of ``log theta`` for ``theta ~ U[a, b]``.

    Kept identical to the implementation in ``heater_dim_scaling_sweep.py`` so
    the analytic (oracle) arm matches the existing experiment exactly.
    """
    if a <= 0.0:
        raise ValueError("log moments require a positive prior lower bound")
    if b <= a:
        raise ValueError("theta_max must exceed theta_min")
    width = b - a
    mu = (b * np.log(b) - b - a * np.log(a) + a) / width
    e_sq = (
        b * np.log(b) ** 2 - 2.0 * b * np.log(b) + 2.0 * b
        - a * np.log(a) ** 2 + 2.0 * a * np.log(a) - 2.0 * a
    ) / width
    return float(mu), float(e_sq - mu ** 2)


def analytic_eta(theta: np.ndarray, cfg: ChainHeaterCfg, d: int) -> np.ndarray:
    """Standardised log-product: the oracle 1-D coordinate, ``(n,)``."""
    mu_log, var_log = log_theta_moments_uniform(cfg.theta_min, cfg.theta_max)
    scale = float(np.sqrt(d * var_log))
    return ((np.log(theta).sum(axis=1) - d * mu_log) / scale).astype(np.float32)


# =============================================================================
# Local workaround: the flattening module's exponential visualisation grid
# =============================================================================

class capped_visualisation_grid:
    """Bound the ``num_pts ** n_params`` grid at the end of ``fit_flattening``.

    ``training_loop_flatten.fit_flattening`` finishes with a coordinate
    visualisation block (around lines 1448-1473) that builds a full
    tensor-product grid over *all* ``n_params`` axes and evaluates the model on
    it::

        if n_params > 3: num_pts = 5
        grds = jnp.meshgrid(xs, ys, *extra)
        X = jnp.stack([g.flatten() for g in grds], axis=-1)
        etas = jax.vmap(mymodel)(X)[:, :2]
        if do_plot:                       # only the PLOTTING is guarded
            ...

    The ``do_plot`` guard covers only the plotting, so the grid and the vmap run
    unconditionally and ``etas`` is discarded when ``do_plot=False`` -- which is
    what this sweep passes. The cost is ``5 ** d`` rows:

        d=8  -> 390,625        (works, pure waste)
        d=10 -> 9,765,625      -> JaxRuntimeError INTERNAL: Autotuning failed
        d=12 -> 244,140,625    -> JaxRuntimeError RESOURCE_EXHAUSTED

    so every d >= 10 run died in the flattening stage before this workaround,
    for reasons unrelated to the science. The block sits *after* ``np.savez``,
    so results are unaffected -- it is wasted compute that happens to be fatal
    at high d.

    ``jnp.meshgrid`` is used nowhere else in ``training_loop_flatten``, so
    scoping the patch to the ``fit_flattening`` call is safe. The shared module
    is deliberately left unmodified so other experiments are unaffected.
    """

    def __init__(self, max_side: int = 32):
        self.max_side = int(max_side)
        self._saved: Optional[Callable] = None

    def __enter__(self) -> "capped_visualisation_grid":
        import jax.numpy as jnp

        real = jnp.meshgrid
        self._saved = real
        side = self.max_side

        def _capped(*arrays, **kwargs):
            # Keep a usable 2-D slice for the first two axes; the remaining
            # axes are constant dummy values in this block, so collapsing them
            # to a single point changes nothing that is used.
            trimmed = []
            for i, a in enumerate(arrays):
                a = jnp.asarray(a)
                trimmed.append(a[:side] if i < 2 else a[:1])
            return real(*trimmed, **kwargs)

        jnp.meshgrid = _capped
        return self

    def __exit__(self, *exc) -> None:
        import jax.numpy as jnp

        if self._saved is not None:
            jnp.meshgrid = self._saved
            self._saved = None


# =============================================================================
# Product level-set recovery test
# =============================================================================

def product_preserving_partner(
    theta: np.ndarray, cfg: ChainHeaterCfg, rng: np.random.Generator,
) -> np.ndarray:
    """Return ``theta'`` with **exactly** the same product, inside the prior box.

    Picks two coordinates and applies ``(t_i, t_j) -> (c * t_i, t_j / c)``,
    which leaves ``prod_i theta_i`` invariant by construction. ``c`` is drawn
    log-uniformly over the widest interval keeping both coordinates inside
    ``[theta_min, theta_max]``, so no rejection sampling is needed and the
    construction works unchanged at every ``d``.
    """
    lo, hi = float(cfg.theta_min), float(cfg.theta_max)
    t = np.array(theta, dtype=np.float64, copy=True)
    n, d = t.shape
    if d < 2:
        return t
    r = np.arange(n)
    i = rng.integers(0, d, n)
    j = (i + 1 + rng.integers(0, d - 1, n)) % d          # j != i
    ti, tj = t[r, i], t[r, j]
    c_lo = np.maximum(lo / ti, tj / hi)
    c_hi = np.minimum(hi / ti, tj / lo)
    c_lo = np.maximum(c_lo, 1e-12)
    c_hi = np.maximum(c_hi, c_lo)
    c = c_lo * (c_hi / c_lo) ** rng.uniform(size=n)
    t[r, i] = np.clip(c * ti, lo, hi)
    t[r, j] = np.clip(tj / c, lo, hi)
    return t


def level_set_unexplained(
    fn: Callable[[np.ndarray], np.ndarray],
    cfg: ChainHeaterCfg,
    d: int,
    rng: np.random.Generator,
    n_pairs: int,
) -> float:
    """Fraction of a coordinate's variance that the product cannot explain.

    The data depend on ``theta`` only through ``P = prod_i theta_i``, so a
    correct distilled coordinate is *some* monotone function ``h(P)`` -- the
    whole equivalence class is correct, not just the analytic representative.
    Every member of that class is constant on the level sets of ``P``, so::

        U = E_P[ Var(eta | P) ] / Var(eta)      in [0, 1]

    is exactly zero iff the coordinate is a function of the product alone.
    Estimated from matched pairs using ``E[(eta - eta')^2 | P] = 2 Var(eta|P)``.

    This tests membership in the equivalence class directly, whereas the
    Spearman score tests proximity to one hand-picked representative
    (``h = log``). That distinction matters on this prior: ``log`` is nearly
    affine on a narrow positive box, so a purely additive coordinate -- which
    has no multiplicative structure at all -- scores |rho| ~ 0.996 and is
    indistinguishable from a true recovery under any Spearman threshold. Its
    ``U`` is small but nonzero, and crucially it is *measurable*, which is why
    the linear surrogate is recorded alongside as the reference to beat.

    ``U`` is invariant to shift and scale, so the arbitrary units SR returns
    need no standardisation.

    Returns ``nan`` if the coordinate is degenerate (zero variance) or
    non-finite on too many draws.
    """
    theta = rng.uniform(
        cfg.theta_min, cfg.theta_max, size=(int(n_pairs), d)
    ).astype(np.float64)
    partner = product_preserving_partner(theta, cfg, rng)

    v1 = np.asarray(fn(theta), dtype=np.float64).reshape(theta.shape[0], -1)[:, 0]
    v2 = np.asarray(fn(partner), dtype=np.float64).reshape(partner.shape[0], -1)[:, 0]
    good = np.isfinite(v1) & np.isfinite(v2)
    if good.sum() < 10:
        return float("nan")
    v1, v2 = v1[good], v2[good]
    total_var = float(np.var(np.concatenate([v1, v2])))
    if not np.isfinite(total_var) or total_var <= 1e-30:
        return float("nan")
    return float(np.mean((v1 - v2) ** 2) / (2.0 * total_var))


# =============================================================================
# Standardisation of a discovered coordinate
# =============================================================================

@dataclass
class AffineStandardiser:
    """Prior-estimated affine map to zero mean and unit variance per axis.

    The analytic coordinate in the existing sweep is deliberately
    standardised, because a raw ``prod_i theta_i`` has range
    ``[theta_min^d, theta_max^d]`` and a sharp posterior becomes a near-delta
    in standardised units. A discovered coordinate arrives on an arbitrary
    scale, so it needs the same treatment or it would lose to the oracle for
    purely numerical reasons. The constants come from prior draws only -- no
    knowledge of the true coordinate is used.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "AffineStandardiser":
        v = np.atleast_2d(np.asarray(values, dtype=np.float64))
        if v.shape[0] == 1 and values.ndim == 1:
            v = v.T
        mean = v.mean(axis=0)
        scale = v.std(axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return cls(mean=mean, scale=scale)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        if v.ndim == 1:
            v = v[:, None]
        return ((v - self.mean) / self.scale).astype(np.float32)


def symbolic_projector(
    expressions: list[str], n_params: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Compile discovered expressions in ``X1..Xn`` into a vectorised callable.

    Lambdified once so it can be applied per-observation to posterior sample
    blocks without re-parsing. Falls back to ``get_y_sr`` if lambdify fails
    on an exotic expression.
    """
    import sympy

    symbols = sympy.symbols([f"X{i + 1}" for i in range(n_params)])
    try:
        funcs = [
            sympy.lambdify(symbols, sympy.sympify(expr), modules="numpy")
            for expr in expressions
        ]
    except Exception:
        from degeneracy_distillery.postprocessing_utils import get_y_sr

        def _fallback(theta: np.ndarray) -> np.ndarray:
            out = np.asarray(get_y_sr(list(expressions), np.asarray(theta)))
            return out[:, None] if out.ndim == 1 else out

        return _fallback

    def _project(theta: np.ndarray) -> np.ndarray:
        t = np.asarray(theta, dtype=np.float64)
        if t.ndim == 1:
            t = t[None, :]
        cols = [t[:, i] for i in range(n_params)]
        out = []
        for fn in funcs:
            val = np.asarray(fn(*cols), dtype=np.float64)
            out.append(np.broadcast_to(val, (t.shape[0],)) if val.ndim == 0 else val)
        return np.stack(out, axis=1)

    return _project


# =============================================================================
# Rank selection
# =============================================================================

def select_informative_axes(
    Fs: np.ndarray,
    dy: np.ndarray,
    prior_scales: np.ndarray,
    floor_rel: float,
    max_rank: Optional[int] = None,
    method: str = "eigengap",
    min_gap: float = 10.0,
) -> dict[str, Any]:
    """Pick the eta axes that track identifiable directions.

    Step 1 reads the rank off the **per-sample** prior-normalised Fisher
    spectra: each ``F(theta_b)`` is normalised by its own leading eigenvalue,
    and the median relative spectrum across samples is compared to
    ``floor_rel``.

    Using the per-sample spectra rather than the spectrum of the *mean* Fisher
    is essential and not a detail. This simulator has a Fisher that is exactly
    rank 1 at every theta, but the informative direction ``g ~ P/theta``
    rotates across the prior, so the mean of those rank-1 outer products is
    full rank. On the analytic Fisher the per-sample median relative spectrum
    is ``[1, 1e-8, 4e-10, ...]`` while the mean gives ``[1, 0.012, 0.009,
    0.007]`` -- the latter sits right on a 1e-2 threshold and would make the
    retained rank an artefact of the cutoff. Whitening and normalisation
    upstream are invertible congruences ``F -> W^T F W``, which preserve
    per-sample rank exactly, so this diagnostic survives them; the mean-based
    one does not.

    Step 2 scores each eta axis by the fraction of its Jacobian energy lying
    in the per-sample informative eigen-subspace, averaged over samples, and
    keeps the top ``r``. The subspace is taken per sample for the same reason.
    This is also independent of axis ordering, which matters because the two
    orderings in play (nonlinearity energy, and Fisher-eigenvalue
    canonicalisation) put the informative axis in different places.

    ``Fs`` and ``dy`` must be the matched pair returned by
    ``load_and_process_data_v2`` so their bases agree.

    Returns the retained rank, kept axis indices, per-axis scores, and the
    spectra needed to revisit ``floor_rel`` after the fact.
    """
    from degeneracy_distillery.align_coords import (
        linearity_residual, mean_fisher_eigen, nonlinearity_spectrum,
    )

    F = np.asarray(Fs, dtype=np.float64)
    if F.ndim == 2:
        F = F[None]
    s = np.asarray(prior_scales, dtype=np.float64).reshape(-1)
    F = F * s[None, :, None] * s[None, None, :]
    F = 0.5 * (F + np.swapaxes(F, -1, -2))

    eigvals_b, eigvecs_b = np.linalg.eigh(F)
    eigvals_b = eigvals_b[:, ::-1]
    eigvecs_b = eigvecs_b[:, :, ::-1]

    lead = eigvals_b[:, :1]
    usable = np.isfinite(eigvals_b).all(axis=1) & (lead[:, 0] > 0.0)
    if not usable.any():
        raise RuntimeError("no sample has a positive finite Fisher eigenvalue")
    relative_b = np.abs(eigvals_b[usable]) / np.maximum(lead[usable], 1e-300)
    median_relative = np.median(relative_b, axis=0)

    n_dim = int(median_relative.size)
    gap_ratios = (
        median_relative[:-1] / np.maximum(median_relative[1:], 1e-300)
        if n_dim > 1 else np.zeros(0)
    )
    if method == "eigengap":
        # A fixed relative floor is not usable here: the fishnet-estimated
        # Fisher has a noise plateau around 1e-2 relative (against 1e-8 for the
        # analytic Fisher), so a 1e-2 cutoff lands inside the plateau and the
        # retained rank becomes an artefact of the threshold rather than of the
        # structure. The spectra instead show a clean multiplicative gap
        # followed by a tight plateau, so cut at the largest gap. If no gap
        # clears ``min_gap`` the spectrum has no plateau and the model is
        # treated as full rank.
        if gap_ratios.size == 0:
            rank = n_dim
        else:
            idx = int(np.argmax(gap_ratios))
            rank = idx + 1 if float(gap_ratios[idx]) >= min_gap else n_dim
    elif method == "floor":
        rank = int(np.sum(median_relative > floor_rel))
    else:
        raise ValueError(f"unknown rank method: {method!r}")
    rank = max(1, min(rank, n_dim))
    if max_rank is not None:
        rank = min(rank, int(max_rank))

    J = np.asarray(dy, dtype=np.float64)
    if J.shape[0] != F.shape[0]:
        raise ValueError(
            f"Jacobian batch {J.shape[0]} does not match Fisher batch {F.shape[0]}"
        )
    V = eigvecs_b[:, :, :rank]
    proj = np.einsum("bij,bjk->bik", J, V)
    num = (proj ** 2).sum(axis=-1)
    den = (J ** 2).sum(axis=-1)
    ratio = num / np.maximum(den, 1e-300)
    scores = np.nanmean(np.where(den > 0.0, ratio, np.nan), axis=0)
    scores = np.nan_to_num(scores, nan=0.0)

    keep = np.sort(np.argsort(-scores)[:rank]).astype(int)

    # Recorded as a diagnostic, NOT as a pass/fail gate. For this simulator the
    # discarded axes are driven by the Fisher regulariser and carry most of the
    # Jacobian nonlinearity energy, so leakage close to 1 is the expected
    # outcome and is itself evidence that nonlinearity energy would be the
    # wrong quantity to select axes by.
    nl_spectrum = np.asarray(nonlinearity_spectrum(J), dtype=np.float64)
    leakage = float(linearity_residual(J, tuple(int(i) for i in keep)))

    # Recorded for auditing only. See the docstring for why the mean spectrum
    # is not used to set the rank.
    mean_eigvals, _ = mean_fisher_eigen(
        Fs, prior_scales=prior_scales, ascending=False,
    )
    mean_eigvals = np.asarray(mean_eigvals, dtype=np.float64)
    mean_largest = float(np.max(np.abs(mean_eigvals))) or 1.0

    return {
        "rank": rank,
        "keep_axes": keep,
        "axis_scores": scores,
        "median_relative_spectrum": median_relative,
        "gap_ratios": gap_ratios,
        "per_sample_relative_p90": np.percentile(relative_b, 90, axis=0),
        "mean_fisher_eigvals_relative": np.abs(mean_eigvals) / mean_largest,
        "nonlinearity_spectrum": nl_spectrum,
        "discarded_jacobian_leakage": leakage,
    }


# =============================================================================
# Discovery pipeline (one run at one d)
# =============================================================================

def run_discovery(
    theta_train: np.ndarray,
    data_train: np.ndarray,
    theta_val: np.ndarray,
    data_val: np.ndarray,
    workdir: Path,
    args: argparse.Namespace,
    seed: int,
    runtimes: dict[str, float],
) -> dict[str, Any]:
    """Fishnets -> flattening -> alignment -> rank selection -> SR."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import flax.linen as nn

    from degeneracy_distillery.align_coords import load_and_process_data_v2
    from degeneracy_distillery.training_loop_fishnets import train_fishnets
    from degeneracy_distillery.training_loop_flatten import fit_flattening

    d = int(theta_train.shape[1])
    workdir.mkdir(parents=True, exist_ok=True)
    fishnets_dir = workdir / "fishnets"
    flatten_prefix = str(workdir / "flattened")

    # --- Fishnet ensemble ---------------------------------------------------
    stage_start = time.time()
    embedding_net = nn.Sequential([nn.Dense(64), nn.gelu, nn.Dense(32), nn.gelu])
    train_fishnets(
        jnp.asarray(theta_train), jnp.asarray(data_train),
        jnp.asarray(theta_val), jnp.asarray(data_val),
        num_models=args.num_fishnets,
        train_epochs=args.fishnet_epochs,
        patience=30,
        n_layers=[2, 5],
        hids_min=50,
        hids_max=300,
        embedding_net=embedding_net,
        lr=5e-5,
        train_batch_size=200,
        seed_model=201 + seed,
        seed_train=999 + seed,
        outdir=str(fishnets_dir),
    )
    runtimes["fishnets"] = time.time() - stage_start

    # --- Flattening + alignment + rank, with restart selection --------------
    # Rank-deficient regime: the forward/backward invertibility penalty pulls
    # J toward the identity, which is the opposite of what d-1 unidentifiable
    # directions need, so it is off by default here.
    #
    # The flattening optimum is strongly seed-sensitive. At identical settings
    # some seeds isolate the informative direction and some do not, and the
    # difference is already visible in the kept axis's Fisher-alignment score
    # *before* any comparison to ground truth: measured 0.998 for a run that
    # recovered the product exactly (level-set U ~ 1e-31) against 0.65-0.79 for
    # runs that returned a single-variable or affine coordinate. Restarting the
    # flow and keeping the highest-scoring attempt is therefore an oracle-free
    # selection rule -- it reads the Fisher and the Jacobian only, never the
    # analytic coordinate -- and it is reported as part of the method rather
    # than applied silently.
    #
    # The fishnet ensemble is trained once and reused across restarts, so K
    # restarts cost roughly (fishnets + K * flatten) rather than K full runs.
    with np.load(fishnets_dir / "fishnets_outputs.npz") as fish:
        thetas = jnp.asarray(fish["theta"])
        fs_np = np.asarray(fish["Fs"])
        ensemble_weights = np.asarray(fish["ensemble_weights"])

    finite = np.isfinite(fs_np).all(axis=tuple(range(1, fs_np.ndim)))
    if not finite.any():
        raise RuntimeError("all fishnet ensemble members produced non-finite Fishers")
    if finite.sum() < finite.size:
        fs_np = fs_np[finite]
        ensemble_weights = ensemble_weights[finite]

    # fit_flattening truncates the sample axis to a multiple of batch_size, and
    # when the sample count is *below* batch_size that drops every sample and
    # fails downstream with an opaque empty-median error. The samples reaching
    # this stage are the fishnet held-out set (i.e. --n-test), not --nsims, so
    # clamp rather than assume.
    n_flat = int(thetas.shape[0])
    flat_batch = min(int(args.flatten_batch_size), n_flat)
    if flat_batch < 1:
        raise RuntimeError(
            f"flattening received {n_flat} samples; increase --n-test"
        )
    if flat_batch != int(args.flatten_batch_size):
        print(
            f"[flatten] clamping batch_size {args.flatten_batch_size} -> "
            f"{flat_batch} for {n_flat} available samples",
            flush=True,
        )

    prior_scales = np.full(d, args.theta_max - args.theta_min, dtype=np.float64)

    # --- Fisher-linear coordinate (no flattening, no SR, deterministic) ------
    # The learned Fisher's informative *direction* turns out to be essentially
    # constant across theta -- measured spread (mean |cos| of each per-sample top
    # eigenvector against the mean) is 0.9998-1.0000, against a true
    # grad-log-P spread of 0.9855 at d=4. That is consistent with the
    # architecture: the fishnet Fisher head is applied to the *data* only, and
    # y = P*kernel + noise carries information about P but not about the
    # individual theta_i, so the direction cannot track theta.
    #
    # Given a near-constant direction, projecting theta onto its mean is a
    # deterministic 1-D reduction that needs none of the fragile machinery. It
    # is oracle-free (Fisher only) and, unlike lambda_max, its level-set score
    # is stable in d (~0.006 at d=2 to ~0.012 at d=6, versus lambda_max
    # degrading to 0.155 by d=12, because lambda_max ~ P^2 sum(1/theta_i^2)
    # carries a second degree of freedom that varies at fixed P).
    #
    # Caveat worth reporting alongside any result from this arm: it works here
    # *because* the direction field is near-constant, which is measurable from
    # the Fisher alone. `fisher_direction_spread` is recorded so the
    # applicability of the linear reduction can be checked rather than assumed --
    # a spread well below 1 means a linear projection is not enough and the
    # flattening flow is genuinely needed.
    F_mean_persample = 0.5 * (fs_np + np.swapaxes(fs_np, -1, -2))
    _w_ens = np.asarray(ensemble_weights, dtype=np.float64)
    _w_ens = _w_ens / max(_w_ens.sum(), 1e-300)
    F_ps = np.einsum("m,mbij->bij", _w_ens, F_mean_persample)
    _ev, _V = np.linalg.eigh(F_ps)
    _g = _V[:, :, -1]
    _g = _g * np.sign(_g @ _g[0])[:, None]          # resolve per-sample sign
    _u = _g / np.maximum(np.linalg.norm(_g, axis=1, keepdims=True), 1e-300)
    fisher_v = _u.mean(axis=0)
    fisher_v = fisher_v / max(float(np.linalg.norm(fisher_v)), 1e-300)
    _m = fisher_v / max(float(np.linalg.norm(fisher_v)), 1e-300)
    fisher_direction_spread = float(np.abs(_u @ _m).mean())
    print(f"[fisher-linear] direction spread {fisher_direction_spread:.4f} "
          f"(1.0 = perfectly constant field); v = "
          f"{np.array2string(fisher_v, precision=3)}", flush=True)

    def _flatten_align_rank(attempt_seed: int, prefix: str) -> dict[str, Any]:
        """One flattening restart, through to its Fisher-alignment score."""
        t0 = time.time()
        # See capped_visualisation_grid: without this, every d >= 10 run dies in
        # the flattening stage on a 5**d visualisation grid that is discarded.
        with capped_visualisation_grid():
            _w, ens_ws, _outputs, model = fit_flattening(
                F_network_ensemble=jnp.asarray(fs_np),
                θs=thetas,
                ensemble_weights=ensemble_weights,
                flattener_activation="softplus",
                loss_type=args.loss_type,
                forward_backward_mlp=not args.no_invertibility_mlp,
                forward_backward_invertibility_weight=args.invertibility_weight,
                hidden_size=args.flatten_hidden_size,
                n_layers=args.flatten_n_layers,
                offset=0.0,
                beta_det=args.beta_det,
                noise=args.flatten_noise,
                batch_size=flat_batch,
                finetune_epochs=args.flatten_finetune_epochs,
                epochs_phase1=args.flatten_epochs_phase1,
                epochs_phase2=args.flatten_epochs_phase2,
                lr_phase1=2e-6,
                lr_schedule_initial=7e-5,
                lr_decay=0.3,
                l1_alpha=0.0,
                do_plot=False,
                seed=attempt_seed,
                output_prefix=prefix,
                Fisher_to_flatten="best",
                return_model=True,
                # The module is used in-process for SR augmentation, so the per-run
                # pickle is pure I/O overhead across a few hundred runs and would
                # drag in a cloudpickle dependency the sweep does not need.
                save_flatten_model_pickle=False,
            )
        t_flat = time.time() - t0

        t0 = time.time()
        npz = Path(prefix + ".npz")
        al = load_and_process_data_v2(
            datapath=str(npz.parent) + os.sep,
            filename=npz.name,
            num_samps=min(4000, int(theta_train.shape[0])),
            seed=44 + attempt_seed,
            process_ensemble=True,
            n_d=1.0,
            align_mode=args.align_mode,
            separate_nonlinearity=True,
            canonicalize="permute_and_sign",
            use_prior_normalization=True,
            restore_reference_mean=True,
            Fisher_to_flatten="best",
            verbose=False,
        )
        Xa = np.asarray(al["X"])
        m = np.isfinite(Xa).all(axis=1) & (Xa > 0.0).all(axis=1)
        Xa = Xa[m]
        if Xa.shape[0] == 0:
            raise RuntimeError("alignment left no finite positive samples")
        ya = np.asarray(al["y"])[m]
        ystd = np.asarray(al["y_std"])[m]
        dysr = np.asarray(al["dy_sr"])[m]
        Fsa = np.asarray(al["Fs"])[m]
        t_align = time.time() - t0

        t0 = time.time()
        ri = select_informative_axes(
            Fsa, dysr, prior_scales, args.rank_floor_rel, max_rank=args.max_rank,
            method=args.rank_method, min_gap=args.rank_min_gap,
        )
        t_rank = time.time() - t0

        kept = [int(i) for i in ri["keep_axes"]]
        score = float(np.mean([ri["axis_scores"][i] for i in kept])) if kept else 0.0
        return {
            "score": score, "rank_info": ri, "keep_axes": kept, "aligned": al,
            "X": Xa, "y": ya, "y_std": ystd, "dy_sr": dysr, "Fs": Fsa,
            "flatten_model": model, "ensemble_ws": ens_ws,
            "t_flat": t_flat, "t_align": t_align, "t_rank": t_rank,
        }

    n_restarts = max(1, int(args.flatten_restarts))
    attempts: list[dict[str, Any]] = []
    restart_errors: list[str] = []
    t_flat = t_align = t_rank = 0.0
    for k in range(n_restarts):
        # Distinct seeds per restart; k=0 keeps the original single-run seed so
        # --flatten-restarts 1 reproduces earlier behaviour exactly.
        att_seed = seed if k == 0 else seed + 100_003 * k
        prefix = flatten_prefix if k == 0 else f"{flatten_prefix}_r{k}"
        try:
            att = _flatten_align_rank(att_seed, prefix)
        except Exception as exc:
            # A restart that blows up numerically (e.g. the non-finite
            # eta_ensemble seen with noise=1e-2 + squared_frob_det) is recorded
            # and skipped rather than failing the run, provided some restart
            # survives.
            restart_errors.append(f"restart{k}: {type(exc).__name__}: {exc}")
            print(f"[flatten] restart {k} failed: {type(exc).__name__}: {exc}",
                  flush=True)
            continue
        t_flat += att["t_flat"]; t_align += att["t_align"]; t_rank += att["t_rank"]
        attempts.append(att)
        print(f"[flatten] restart {k}: rank {att['rank_info']['rank']}, "
              f"keep {att['keep_axes']}, alignment score {att['score']:.4f}",
              flush=True)

    if not attempts:
        raise RuntimeError(
            "every flattening restart failed: " + " | ".join(restart_errors)
        )

    scores_all = [float(a["score"]) for a in attempts]
    best_idx = int(np.argmax(scores_all))
    best = attempts[best_idx]
    runtimes["flatten"] = t_flat
    runtimes["alignment"] = t_align
    runtimes["rank"] = t_rank

    rank_info = best["rank_info"]
    keep_axes = best["keep_axes"]
    aligned = best["aligned"]
    X, y, y_std = best["X"], best["y"], best["y_std"]
    dy_sr, Fs = best["dy_sr"], best["Fs"]
    flatten_model, ensemble_ws = best["flatten_model"], best["ensemble_ws"]

    spec = rank_info["median_relative_spectrum"]
    if n_restarts > 1:
        print(f"[flatten] selected restart {best_idx} of {len(attempts)} "
              f"(scores {np.array2string(np.asarray(scores_all), precision=4)})",
              flush=True)
    print(
        f"[rank] retained rank {rank_info['rank']} of {d}; keep axes {keep_axes}; "
        f"median relative spectrum "
        f"{np.array2string(spec[:min(4, spec.size)], precision=4)}; "
        f"axis scores "
        f"{np.array2string(rank_info['axis_scores'], precision=3)}",
        flush=True,
    )

    result: dict[str, Any] = {
        "rank_info": rank_info,
        "n_params": d,
        "fisher_v": fisher_v,
        "fisher_direction_spread": fisher_direction_spread,
        "n_flatten_restarts": n_restarts,
        "n_flatten_restarts_ok": len(attempts),
        "selected_restart": best_idx,
        "alignment_score": float(best["score"]),
        "restart_scores": scores_all,
        "restart_errors": restart_errors,
    }
    if args.rank_only:
        return result

    # --- Symbolic regression on the surviving axes only ---------------------
    # slice_fisher=False is essential: slicing would drop theta inputs
    # symmetrically with eta outputs, but the surviving coordinate depends on
    # all d thetas. All d inputs, r outputs.
    stage_start = time.time()
    from degeneracy_distillery.sr_utils import (
        fit_and_analyze_sr, sr_structure_predicate,
    )

    sr_dir = workdir / "sr_results"
    sr_dir.mkdir(parents=True, exist_ok=True)
    n_aug = int(args.sr_n_aug if args.sr_n_aug else args.sr_n_aug_per_dim * d)
    # A d-way product needs roughly 2d-1 tokens; the usual default of 25 makes
    # a 12-way product literally unrepresentable.
    max_length = int(args.sr_max_length if args.sr_max_length else max(25, 4 * d + 8))
    max_depth = int(args.sr_max_depth if args.sr_max_depth else max(10, d + 4))
    print(
        f"[sr] fitting components {keep_axes} with n_aug={n_aug}, "
        f"max_length={max_length}, max_depth={max_depth}",
        flush=True,
    )

    mdl_coords, frob_coords, analysis, split_data = fit_and_analyze_sr(
        X, y, y_std, dy_sr, Fs,
        n_params=d,
        components_to_fit=keep_axes,
        slice_fisher=False,
        parent_dir=str(sr_dir) + os.sep,
        test_size=0.5,
        random_state=32134 + seed,
        shuffle=True,
        time_limit=args.sr_time_limit,
        max_length=max_length,
        max_depth=max_depth,
        allowed_symbols="add,mul,div,pow,constant,variable,sqrt",
        max_complexity_thresh=max(20, 3 * d),
        equation_set="pareto",
        length_penalty=2.0,
        equation_predicate=sr_structure_predicate(
            n_params=d,
            forbid_self_transcendental=True,
            check_nested_exp=False,
            forbid_x_in_pow_exponent=True,
        ),
        flatten_model=flatten_model,
        ensemble_w=ensemble_ws,
        rotmats=aligned["rotmats"],
        ensemble_weights=aligned["ensemble_weights"],
        n_sr_samples=n_aug,
        key=jr.PRNGKey(7717 + seed),
    )
    runtimes["sr"] = time.time() - stage_start

    complexities: list[float] = []
    try:
        for comp_idx in range(len(mdl_coords)):
            ibest = analysis["ibest_mdl"][comp_idx]
            complexities.append(float(analysis["complexity"][comp_idx][ibest]))
    except Exception:
        complexities = []

    result.update({
        "mdl_coords": [str(e) for e in mdl_coords],
        "frob_coords": [str(e) for e in frob_coords],
        "complexities": complexities,
        "n_augmented_coordinate_evaluations": n_aug,
        "sr_max_length": max_length,
        "split_data": split_data,
    })
    return result


# =============================================================================
# NPE arms
# =============================================================================

def make_runner(low: np.ndarray, high: np.ndarray, args: argparse.Namespace,
                device: str, model: str):
    """Build an ltu-ili ``InferenceRunner`` for an MDN (sbi) or MAF (lampe)."""
    import ili
    from ili.inference import InferenceRunner

    prior = ili.utils.Uniform(low=low.tolist(), high=high.tolist(), device=device)
    train_args = {
        "training_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_num_epochs": args.epochs,
        "stop_after_epochs": args.patience,
    }
    if model == "mdn":
        nets = [
            ili.utils.load_nde_sbi(
                engine="NPE", model="mdn",
                hidden_features=args.hidden_features,
                num_components=args.num_mdn_components,
            )
            for _ in range(max(1, int(args.repeats)))
        ]
        return InferenceRunner.load(
            backend="sbi", engine="NPE", prior=prior, nets=nets, device=device,
            train_args=train_args, proposal=None, out_dir=None,
        )
    if model == "maf":
        nets = [
            ili.utils.load_nde_lampe(
                engine="NPE", model="maf",
                hidden_features=args.hidden_features,
                num_transforms=args.num_transforms,
                repeats=args.repeats,
            )
        ]
        return InferenceRunner.load(
            backend="lampe", engine="NPE", prior=prior, nets=nets, device=device,
            train_args=train_args, proposal=None, out_dir=None,
        )
    raise ValueError(f"unknown NDE model: {model!r}")


def train_npe(
    target: np.ndarray, data: np.ndarray,
    low: np.ndarray, high: np.ndarray,
    args: argparse.Namespace, device: str, seed: int, model: str,
) -> tuple[float, np.ndarray, Any]:
    """Train one NPE arm. Returns best validation log-prob, curve, posterior."""
    import torch
    from ili.dataloaders import NumpyLoader

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    runner = make_runner(low, high, args, device, model)
    loader = NumpyLoader(
        x=np.asarray(data, dtype=np.float32),
        theta=np.asarray(target, dtype=np.float32),
    )
    posterior, summaries = runner(loader=loader)
    val = np.asarray(summaries[0]["validation_log_probs"])
    return float(np.max(val)), val, posterior


def marginal_log_probs_on_axis(
    posterior: Any,
    x_val: np.ndarray,
    axis_true: np.ndarray,
    projector: Callable[[np.ndarray], np.ndarray],
    n_samples: int,
    device: str,
) -> np.ndarray:
    """Gaussian-matched ``log p(axis_true | y)`` from posterior samples.

    Both arms of a comparison are pushed onto one common scalar axis, which
    is what makes their log-probabilities comparable. The Gaussian fit is
    appropriate here because ``p(y | eta)`` is exactly Gaussian for this
    simulator and the standardised prior is near-Gaussian by CLT.
    """
    import torch

    n_val = int(x_val.shape[0])
    out = np.empty(n_val, dtype=np.float32)
    x_t = torch.as_tensor(np.asarray(x_val, dtype=np.float32), device=device)
    log_2pi = float(np.log(2.0 * np.pi))

    with torch.no_grad():
        for i in range(n_val):
            try:
                s = posterior.sample((n_samples,), x=x_t[i], show_progress_bars=False)
            except TypeError:
                s = posterior.sample((n_samples,), x=x_t[i])
            s_np = s.detach().cpu().numpy().astype(np.float64)
            if s_np.ndim == 1:
                s_np = s_np[:, None]
            vals = np.asarray(projector(s_np), dtype=np.float64).reshape(-1)
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                out[i] = np.nan
                continue
            mu = float(vals.mean())
            var = float(vals.var()) + 1e-12
            out[i] = -0.5 * (log_2pi + np.log(var)) \
                     - 0.5 * (float(axis_true[i]) - mu) ** 2 / var
    return out


# =============================================================================
# Aggregation
# =============================================================================

def write_recovery_table(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-``d`` recovery counts with denominators -- the rebuttal table."""
    if df.empty:
        return
    rows = []
    for (nsims, d), grp in df.groupby(["nsims", "d"], sort=True):
        n = int(len(grp))
        ok = grp[grp["status"] == "ok"]
        rows.append({
            "nsims": int(nsims),
            "d": int(d),
            "trials": n,
            "completed": int(len(ok)),
            "rank_correct": int(ok["rank_correct"].sum()) if "rank_correct" in ok else 0,
            "symbolic_recovered": (
                int(ok["symbolic_recovered"].sum()) if "symbolic_recovered" in ok else 0
            ),
            "median_abs_spearman": (
                float(ok["spearman_abs"].median()) if "spearman_abs" in ok else np.nan
            ),
            # Level-set criterion, reported beside the Spearman one rather than
            # replacing it: the two disagree exactly where a coordinate is
            # rank-correlated with the analytic axis without being a function
            # of the product, which is the failure mode Spearman cannot see.
            "levelset_recovered": (
                int(ok["levelset_recovered"].sum())
                if "levelset_recovered" in ok else 0
            ),
            "median_levelset_U": (
                float(ok["levelset_unexplained"].median())
                if "levelset_unexplained" in ok else np.nan
            ),
            "median_levelset_U_linear": (
                float(ok["levelset_unexplained_linear"].median())
                if "levelset_unexplained_linear" in ok else np.nan
            ),
            "median_levelset_ratio_vs_linear": (
                float(ok["levelset_ratio_vs_linear"].median())
                if "levelset_ratio_vs_linear" in ok else np.nan
            ),
            "median_complexity": (
                float(ok["complexity"].median()) if "complexity" in ok else np.nan
            ),
            "failed_stages": ";".join(
                sorted({str(s) for s in df.loc[grp.index, "failed_stage"].dropna()})
            ),
        })
    pd.DataFrame(rows).to_csv(out_dir / "recovery_table.csv", index=False)


def write_aggregate(df: pd.DataFrame, out_dir: Path) -> None:
    """Mean/std/sem/n of the numeric NPE columns per (nsims, d)."""
    if df.empty:
        return
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return
    value_cols = [
        c for c in ok.columns
        if c.endswith("_log_prob") or c.endswith("_marg") or c == "spearman_abs"
    ]
    if not value_cols:
        return
    agg = ok.groupby(["nsims", "d"])[value_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    for c in value_cols:
        n = agg[f"{c}_count"].replace(0, np.nan)
        agg[f"{c}_sem"] = agg[f"{c}_std"] / np.sqrt(n)
    agg.reset_index().to_csv(out_dir / "metrics_aggregate.csv", index=False)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- sweep ---
    p.add_argument("--dims", nargs="+", type=int, default=[2, 3, 4, 6, 8, 10, 12],
                   help="Ambient dimensions to sweep. Intrinsic dim is always 1.")
    p.add_argument("--num-trials", type=int, default=10,
                   help="Independent end-to-end trials per (nsims, d).")
    p.add_argument("--nsims", type=int, default=1000,
                   help="Training simulations per run.")
    p.add_argument("--nsims-list", nargs="+", type=int, default=None,
                   help="Optional budget sweep; overrides --nsims.")
    p.add_argument("--seed", type=int, default=0, help="Master seed.")
    p.add_argument("--out-dir", type=Path, default=Path("heater_discovery_scaling"))
    p.add_argument("--resume", action="store_true",
                   help="Skip (nsims, d, trial) rows already in metrics.csv.")
    p.add_argument("--rank-only", action="store_true",
                   help="Stop after rank selection: no SR, no NPE, and no "
                        "torch/ltu-ili import. Cheap enough for a noise ablation.")
    p.add_argument("--skip-npe", action="store_true",
                   help="Run discovery and SR but skip the NPE arms.")
    p.add_argument("--keep-workdirs", action="store_true",
                   help="Keep per-run intermediate artefacts (large).")

    # --- simulator ---
    p.add_argument("--theta-min", type=float, default=1.0)
    p.add_argument("--theta-max", type=float, default=2.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--t-max", type=float, default=4.0)
    p.add_argument("--n-t", type=int, default=20)
    p.add_argument("--sigma", type=float, default=0.2,
                   help="Observation noise stdev. Note this does NOT change the "
                        "rank: F is exactly rank 1 for any sigma, so raising it "
                        "cannot improve conditioning.")
    p.add_argument("--n-test", type=int, default=1000,
                   help="Held-out simulations for evaluation, drawn separately "
                        "from the training budget.")
    p.add_argument("--independent-npe-sims", action="store_true",
                   help="Draw fresh simulations for the NPE arms instead of "
                        "reusing the discovery training set. Conservative "
                        "accounting; doubles the simulation cost.")

    # --- fishnets ---
    p.add_argument("--num-fishnets", type=int, default=10)
    p.add_argument("--fishnet-epochs", type=int, default=300)

    # --- flattening (rank-deficient defaults) ---
    p.add_argument("--loss-type",
                   choices=("log_frob", "frob", "squared_frob", "squared_frob_det"),
                   default="squared_frob",
                   help="Flattening loss. 'squared_frob' omits the inverse term "
                        "and is the cleaner choice for rank-deficient Fishers; "
                        "'squared_frob_det' is the fallback if the flow "
                        "collapses outright.")
    p.add_argument("--beta-det", type=float, default=0.1,
                   help="Weight of the (log det Q)^2 barrier "
                        "(only used by squared_frob_det).")
    p.add_argument("--flatten-noise", type=float, default=1e-3,
                   help="Cholesky noise added to F per sample. This both "
                        "regularises the singular Fisher and sets the floor the "
                        "null eigenvalues sit at, so it is the yardstick for the "
                        "rank rule. Keep it as low as conditioning allows.")
    p.add_argument("--no-invertibility-mlp", action="store_true", default=True,
                   help="Disable the forward/backward invertibility penalty. On "
                        "by default here: the penalty pulls J toward identity, "
                        "which is the opposite of what a rank-deficient problem "
                        "needs.")
    p.add_argument("--with-invertibility-mlp", dest="no_invertibility_mlp",
                   action="store_false",
                   help="Re-enable the invertibility penalty (not recommended).")
    p.add_argument("--invertibility-weight", type=float, default=1.0)
    p.add_argument("--flatten-batch-size", type=int, default=250,
                   help="Flattening batch size. Clamped at runtime to the "
                        "number of samples reaching the stage, since "
                        "fit_flattening drops every sample when the count is "
                        "below the batch size.")
    p.add_argument("--flatten-hidden-size", type=int, default=256,
                   help="Hidden width of the flattening residual MLP. The "
                        "shared module's default is 256 and it does NOT scale "
                        "with d, so the same capacity maps R^2 -> R^2 and "
                        "R^12 -> R^12. Exposed here to test whether the "
                        "falling per-restart success rate at high d is a "
                        "capacity limit. 256 reproduces previous behaviour.")
    p.add_argument("--flatten-n-layers", type=int, default=5,
                   help="Residual blocks in the flattening MLP (module default "
                        "is 3; this sweep has always passed 5).")
    p.add_argument("--flatten-restarts", type=int, default=1,
                   help="Independent flattening restarts per run, sharing one "
                        "fishnet ensemble. The attempt whose retained axes have "
                        "the highest Fisher-alignment score is kept. This is an "
                        "oracle-free selection rule (it reads only the Fisher "
                        "and the Jacobian) and it matters: at fixed settings the "
                        "flattening optimum is seed-dependent, and the alignment "
                        "score separates exact recoveries (~0.998) from "
                        "single-variable or affine coordinates (0.65-0.79). "
                        "1 reproduces the previous single-fit behaviour.")
    p.add_argument("--flatten-epochs-phase1", type=int, default=1000)
    p.add_argument("--flatten-epochs-phase2", type=int, default=500)
    p.add_argument("--flatten-finetune-epochs", type=int, default=200)
    p.add_argument("--align-mode", choices=("procrustes", "kabsch", "none"),
                   default="procrustes")

    # --- rank rule ---
    p.add_argument("--rank-method", choices=("eigengap", "floor"),
                   default="eigengap",
                   help="How to read the rank off the per-sample median "
                        "relative Fisher spectrum. 'eigengap' cuts at the "
                        "largest multiplicative gap and is the default: the "
                        "fishnet-estimated spectrum has a noise plateau near "
                        "1e-2 relative, so any fixed floor near there makes "
                        "the rank an artefact of the cutoff.")
    p.add_argument("--rank-min-gap", type=float, default=10.0,
                   help="Minimum multiplicative eigengap required to declare a "
                        "plateau. If no gap clears this, the model is treated "
                        "as full rank.")
    p.add_argument("--rank-floor-rel", type=float, default=1e-2,
                   help="Used only by --rank-method floor. Full spectra are "
                        "saved either way so this can be revisited without "
                        "rerunning.")
    p.add_argument("--max-rank", type=int, default=None,
                   help="Optional cap on retained rank.")
    p.add_argument("--expected-rank", type=int, default=1,
                   help="Ground-truth intrinsic dimension, used only to score "
                        "the rank rule.")

    # --- symbolic regression ---
    p.add_argument("--sr-time-limit", type=int, default=300,
                   help="PyOperon time budget in seconds, per component.")
    p.add_argument("--sr-n-aug", type=int, default=None,
                   help="Augmented coordinate evaluations. Default scales with "
                        "d as --sr-n-aug-per-dim * d, since a fixed pool gets "
                        "exponentially sparser as d grows.")
    p.add_argument("--sr-n-aug-per-dim", type=int, default=1000)
    p.add_argument("--sr-max-length", type=int, default=None,
                   help="Default max(25, 4d+8). A d-way product needs roughly "
                        "2d-1 tokens, so a fixed 25 makes high-d targets "
                        "unrepresentable.")
    p.add_argument("--sr-max-depth", type=int, default=None,
                   help="Default max(10, d+4).")
    p.add_argument("--levelset-n-pairs", type=int, default=20000,
                   help="Matched product-preserving pairs used for the "
                        "level-set recovery test. Pure numpy on the symbolic "
                        "expression, so this is cheap and never touches the "
                        "simulator; it is not part of the simulation budget.")
    p.add_argument("--recovery-corr-thresh", type=float, default=0.99,
                   help="Spearman |rho| against the analytic axis at or above "
                        "which a run counts as symbolic recovery.")

    # --- NPE ---
    p.add_argument("--nde-model", choices=("mdn", "maf"), default="mdn",
                   help="Density estimator for ALL arms. MDN by default: an "
                        "autoregressive flow on a 1-D target has nothing to "
                        "autoregress over and mis-localises sharp posteriors, "
                        "and holding the estimator fixed across arms keeps the "
                        "comparison about the coordinate.")
    p.add_argument("--raw-model", choices=("mdn", "maf"), default=None,
                   help="Override the estimator for the raw arm only.")
    p.add_argument("--num-mdn-components", type=int, default=4)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--hidden-features", type=int, default=50)
    p.add_argument("--num-transforms", type=int, default=5)
    p.add_argument("--repeats", type=int, default=2,
                   help="Ensemble size for the NDE.")
    p.add_argument("--n-marginal-val", type=int, default=200,
                   help="Held-out observations used for the common-axis "
                        "marginal evaluation.")
    p.add_argument("--n-marginal-samples", type=int, default=2000,
                   help="Posterior samples per observation for the Gaussian fit.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto" and not args.rank_only:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "auto":
        device = "cpu"
    print(f"device: {device}", flush=True)

    cfg = ChainHeaterCfg(
        theta_min=args.theta_min, theta_max=args.theta_max, tau=args.tau,
        t_max=args.t_max, n_t=args.n_t, sigma=args.sigma,
    )
    raw_model = args.raw_model or args.nde_model
    eta_model = args.nde_model
    nsims_values = list(args.nsims_list) if args.nsims_list else [int(args.nsims)]

    metrics_path = out_dir / "metrics.csv"
    rows: list[dict[str, Any]] = []
    done: set[tuple[int, int, int]] = set()
    if args.resume and metrics_path.exists():
        prev = pd.read_csv(metrics_path)
        rows = prev.to_dict("records")
        done = {
            (int(r["nsims"]), int(r["d"]), int(r["trial"]))
            for r in rows
            if str(r.get("status")) == "ok"
        }
        print(f"resuming: {len(done)} completed runs found", flush=True)

    spectra: dict[str, np.ndarray] = {}
    expressions: dict[str, Any] = {}

    manifest = {
        "script": "heater_discovery_dim_scaling_sweep.py",
        "description": (
            "Chain-product heater scaling sweep with the coordinate discovered "
            "independently at every ambient dimension, rather than supplied "
            "analytically."
        ),
        "simulator": {
            "theta_min": cfg.theta_min, "theta_max": cfg.theta_max,
            "tau": cfg.tau, "t_max": cfg.t_max, "n_t": cfg.n_t, "sigma": cfg.sigma,
        },
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
        "nde": {"raw_model": raw_model, "eta_model": eta_model,
                "num_mdn_components": args.num_mdn_components},
        "arms": ["raw", "analytic", "discovered"],
        "evaluation_axes": {
            "analytic": "raw (pushed) vs analytic (identity)",
            "discovered": "raw (pushed through discovered expression) vs discovered",
        },
        "success_criteria": {
            "rank_correct": f"retained rank == {args.expected_rank}",
            "symbolic_recovered": (
                f"Spearman |rho| vs analytic axis >= {args.recovery_corr_thresh}"
            ),
        },
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    for nsims_idx, n_sim in enumerate(nsims_values):
        for d in args.dims:
            for trial in range(args.num_trials):
                if (n_sim, d, trial) in done:
                    continue
                seed = (args.seed + 7919 * trial + 31 * d + 1_000_003 * nsims_idx)
                tag = f"N{n_sim}_d{d}_trial{trial}"
                print(f"\n=== {tag} (seed {seed}) ===", flush=True)

                row: dict[str, Any] = {
                    "nsims": int(n_sim), "d": int(d), "trial": int(trial),
                    "seed": int(seed), "status": "ok", "failed_stage": None,
                    "error": None,
                }
                runtimes: dict[str, float] = {}
                workdir = out_dir / "runs" / tag
                stage = "sim"
                try:
                    rng = np.random.default_rng(seed)
                    theta_tr, data_tr = chain_dataset(n_sim, d, cfg, rng)
                    theta_va, data_va = chain_dataset(args.n_test, d, cfg, rng)

                    stage = "discovery"
                    disc = run_discovery(
                        theta_tr, data_tr, theta_va, data_va,
                        workdir, args, seed, runtimes,
                    )
                    rank_info = disc["rank_info"]
                    row["retained_rank"] = int(rank_info["rank"])
                    row["rank_correct"] = bool(
                        int(rank_info["rank"]) == int(args.expected_rank)
                    )
                    row["discarded_jacobian_leakage"] = float(
                        rank_info["discarded_jacobian_leakage"]
                    )
                    row["keep_axes"] = ";".join(
                        str(int(i)) for i in rank_info["keep_axes"]
                    )
                    # Restart selection is part of the method, so it is
                    # recorded per run: how many restarts ran, how many
                    # survived, which was kept, and the full score vector, so
                    # the selection can be audited without rerunning.
                    row["alignment_score"] = disc.get("alignment_score", np.nan)
                    row["n_flatten_restarts"] = disc.get("n_flatten_restarts", 1)
                    row["n_flatten_restarts_ok"] = disc.get(
                        "n_flatten_restarts_ok", 1
                    )
                    row["selected_restart"] = disc.get("selected_restart", 0)
                    row["restart_scores"] = ";".join(
                        f"{s:.6f}" for s in disc.get("restart_scores", [])
                    )
                    spectra[f"{tag}_median_relative_spectrum"] = \
                        rank_info["median_relative_spectrum"]
                    spectra[f"{tag}_gap_ratios"] = rank_info["gap_ratios"]
                    spectra[f"{tag}_per_sample_relative_p90"] = \
                        rank_info["per_sample_relative_p90"]
                    spectra[f"{tag}_mean_fisher_eigvals_relative"] = \
                        rank_info["mean_fisher_eigvals_relative"]
                    spectra[f"{tag}_nonlinearity_spectrum"] = \
                        rank_info["nonlinearity_spectrum"]
                    spectra[f"{tag}_axis_scores"] = rank_info["axis_scores"]

                    if args.rank_only:
                        row["symbolic_recovered"] = False
                    else:
                        stage = "recovery_scoring"
                        mdl = disc["mdl_coords"]
                        expressions[tag] = {
                            "mdl": mdl,
                            "frob": disc["frob_coords"],
                            "complexities": disc["complexities"],
                            "keep_axes": [int(i) for i in rank_info["keep_axes"]],
                            "sr_max_length": disc["sr_max_length"],
                        }
                        row["expression"] = " | ".join(mdl)
                        row["complexity"] = (
                            float(np.sum(disc["complexities"]))
                            if disc["complexities"] else np.nan
                        )
                        row["n_augmented_coordinate_evaluations"] = int(
                            disc["n_augmented_coordinate_evaluations"]
                        )

                        # Score the discovered coordinate against the analytic
                        # axis on held-out draws. Spearman is primary: any
                        # monotone function of the product carries the same
                        # information.
                        from scipy.stats import pearsonr, spearmanr

                        project = symbolic_projector(mdl, d)
                        eta_disc_va = np.asarray(project(theta_va), dtype=np.float64)
                        primary = eta_disc_va[:, 0]
                        eta_an_va = analytic_eta(theta_va, cfg, d).astype(np.float64)
                        product_va = np.prod(theta_va, axis=1).astype(np.float64)
                        good = np.isfinite(primary)
                        if good.sum() < 10:
                            raise RuntimeError(
                                "discovered expression is non-finite on held-out draws"
                            )
                        rho = spearmanr(primary[good], eta_an_va[good]).correlation
                        row["spearman_abs"] = float(abs(rho))
                        row["pearson_abs_logprod"] = float(
                            abs(pearsonr(primary[good], eta_an_va[good])[0])
                        )
                        row["pearson_abs_product"] = float(
                            abs(pearsonr(primary[good], product_va[good])[0])
                        )
                        row["symbolic_recovered"] = bool(
                            abs(rho) >= args.recovery_corr_thresh
                        )

                        # Level-set test: does the coordinate depend on theta
                        # only through the product? Scored against two
                        # references computed on the same prior and the same
                        # pair draws -- the analytic axis (must come out ~0,
                        # so it doubles as a check that the estimator itself
                        # is working) and an additive surrogate carrying no
                        # multiplicative structure, which is the bar a genuine
                        # recovery has to clear.
                        ls_rng = np.random.default_rng(seed + 909_091)
                        n_pairs = int(args.levelset_n_pairs)
                        u_disc = level_set_unexplained(
                            lambda t: project(t)[:, 0], cfg, d, ls_rng, n_pairs,
                        )
                        u_lin = level_set_unexplained(
                            lambda t: t.sum(axis=1), cfg, d, ls_rng, n_pairs,
                        )
                        u_an = level_set_unexplained(
                            lambda t: np.log(t).sum(axis=1), cfg, d,
                            ls_rng, n_pairs,
                        )
                        row["levelset_unexplained"] = u_disc
                        row["levelset_unexplained_linear"] = u_lin
                        row["levelset_unexplained_analytic"] = u_an
                        row["levelset_ratio_vs_linear"] = (
                            float(u_disc / u_lin)
                            if np.isfinite(u_disc) and np.isfinite(u_lin) and u_lin > 0
                            else np.nan
                        )
                        # Baseline-relative, so there is no threshold to tune
                        # and no d-dependent cutoff to defend: U falls like 1/d
                        # for every coordinate, and dividing by the surrogate
                        # measured at the same d cancels that trend.
                        row["levelset_recovered"] = bool(
                            np.isfinite(u_disc) and np.isfinite(u_lin)
                            and u_disc < u_lin
                        )
                        print(
                            f"[recovery] |rho|={abs(rho):.4f}  "
                            f"rank_correct={row['rank_correct']}  "
                            f"symbolic_recovered={row['symbolic_recovered']}  "
                            f"levelset_U={u_disc:.5f} "
                            f"(linear {u_lin:.5f}, analytic {u_an:.2e})  "
                            f"levelset_recovered={row['levelset_recovered']}",
                            flush=True,
                        )

                    if not (args.rank_only or args.skip_npe):
                        stage = "npe"
                        if args.independent_npe_sims:
                            npe_rng = np.random.default_rng(seed + 555_557)
                            theta_np, data_np = chain_dataset(n_sim, d, cfg, npe_rng)
                        else:
                            theta_np, data_np = theta_tr, data_tr

                        marg_rng = np.random.default_rng(seed + 999_983)
                        n_marg = int(min(args.n_marginal_val, args.n_test))
                        theta_ev, data_ev = chain_dataset(n_marg, d, cfg, marg_rng)

                        # --- raw arm ---
                        low_raw = np.full(d, cfg.theta_min, dtype=np.float32)
                        high_raw = np.full(d, cfg.theta_max, dtype=np.float32)
                        raw_lp, _raw_curve, raw_post = train_npe(
                            theta_np, data_np, low_raw, high_raw,
                            args, device, seed, raw_model,
                        )
                        row["raw_log_prob"] = raw_lp

                        # --- analytic arm (oracle ceiling) ---
                        eta_an = analytic_eta(theta_np, cfg, d)[:, None]
                        low_an = np.array([-5.0], dtype=np.float32)
                        high_an = np.array([5.0], dtype=np.float32)
                        an_lp, _an_curve, an_post = train_npe(
                            eta_an, data_np, low_an, high_an,
                            args, device, seed + 1, eta_model,
                        )
                        row["analytic_log_prob"] = an_lp

                        # --- discovered arm ---
                        eta_disc_tr = np.asarray(project(theta_np), dtype=np.float64)
                        std = AffineStandardiser.fit(eta_disc_tr)
                        eta_disc_std = std(eta_disc_tr)
                        finite_rows = np.isfinite(eta_disc_std).all(axis=1)
                        n_eta = int(eta_disc_std.shape[1])
                        low_di = np.full(n_eta, -5.0, dtype=np.float32)
                        high_di = np.full(n_eta, 5.0, dtype=np.float32)
                        di_lp, _di_curve, di_post = train_npe(
                            eta_disc_std[finite_rows], data_np[finite_rows],
                            low_di, high_di, args, device, seed + 2, eta_model,
                        )
                        row["discovered_log_prob"] = di_lp
                        row["n_eta"] = n_eta

                        # --- fisher-linear arm (no flattening, no SR) ---
                        # theta projected onto the Fisher's mean informative
                        # direction. Deterministic: no seeds, no restarts, and
                        # it cannot "fail to recover", so it yields a usable
                        # column at every d -- including the dimensions where
                        # the discovered arm has no valid coordinate.
                        fisher_v = np.asarray(disc["fisher_v"], dtype=np.float64)
                        eta_fl_tr = (theta_np.astype(np.float64) @ fisher_v)[:, None]
                        std_fl = AffineStandardiser.fit(eta_fl_tr)
                        fl_lp, _fl_curve, fl_post = train_npe(
                            std_fl(eta_fl_tr), data_np,
                            np.array([-5.0], dtype=np.float32),
                            np.array([5.0], dtype=np.float32),
                            args, device, seed + 3, eta_model,
                        )
                        row["fisher_linear_log_prob"] = fl_lp
                        row["fisher_direction_spread"] = float(
                            disc["fisher_direction_spread"]
                        )

                        # --- common-axis evaluation ---
                        eta_an_ev = analytic_eta(theta_ev, cfg, d).astype(np.float64)
                        mu_log, var_log = log_theta_moments_uniform(
                            cfg.theta_min, cfg.theta_max,
                        )
                        an_scale = float(np.sqrt(d * var_log))
                        an_offset = float(d * mu_log)

                        def to_analytic(theta_s: np.ndarray) -> np.ndarray:
                            t = np.clip(np.asarray(theta_s, dtype=np.float64), 1e-12, None)
                            return (np.log(t).sum(axis=1) - an_offset) / an_scale

                        row["raw_on_analytic_marg"] = float(np.nanmean(
                            marginal_log_probs_on_axis(
                                raw_post, data_ev, eta_an_ev, to_analytic,
                                args.n_marginal_samples, device,
                            )
                        ))
                        row["analytic_on_analytic_marg"] = float(np.nanmean(
                            marginal_log_probs_on_axis(
                                an_post, data_ev, eta_an_ev,
                                lambda s: np.asarray(s).reshape(-1),
                                args.n_marginal_samples, device,
                            )
                        ))

                        # Discovered axis: the raw arm's theta samples are
                        # pushed through the discovered expression, so this
                        # comparison uses no oracle knowledge at all.
                        eta_di_ev = std(
                            np.asarray(project(theta_ev), dtype=np.float64)
                        )[:, 0].astype(np.float64)

                        def to_discovered(theta_s: np.ndarray) -> np.ndarray:
                            return std(
                                np.asarray(project(theta_s), dtype=np.float64)
                            )[:, 0]

                        row["raw_on_discovered_marg"] = float(np.nanmean(
                            marginal_log_probs_on_axis(
                                raw_post, data_ev, eta_di_ev, to_discovered,
                                args.n_marginal_samples, device,
                            )
                        ))
                        row["discovered_on_discovered_marg"] = float(np.nanmean(
                            marginal_log_probs_on_axis(
                                di_post, data_ev, eta_di_ev,
                                lambda s: np.asarray(s)[:, 0]
                                if np.asarray(s).ndim > 1
                                else np.asarray(s).reshape(-1),
                                args.n_marginal_samples, device,
                            )
                        ))
                        # Fisher-linear axis. theta -> v.theta is exact, so the
                        # raw arm's push-forward onto this axis is exact too --
                        # same footing as the discovered axis, and available at
                        # every d because this arm never fails.
                        eta_fl_ev = std_fl(
                            (theta_ev.astype(np.float64) @ fisher_v)[:, None]
                        )[:, 0].astype(np.float64)

                        def to_fisher_linear(theta_s: np.ndarray) -> np.ndarray:
                            t = np.asarray(theta_s, dtype=np.float64)
                            return std_fl((t @ fisher_v)[:, None])[:, 0]

                        row["raw_on_fisher_linear_marg"] = float(np.nanmean(
                            marginal_log_probs_on_axis(
                                raw_post, data_ev, eta_fl_ev, to_fisher_linear,
                                args.n_marginal_samples, device,
                            )
                        ))
                        row["fisher_linear_on_fisher_linear_marg"] = float(
                            np.nanmean(
                                marginal_log_probs_on_axis(
                                    fl_post, data_ev, eta_fl_ev,
                                    lambda s: np.asarray(s)[:, 0]
                                    if np.asarray(s).ndim > 1
                                    else np.asarray(s).reshape(-1),
                                    args.n_marginal_samples, device,
                                )
                            )
                        )

                        # Level-set score for the Fisher-linear coordinate, on
                        # the same footing as the discovered one, so the two
                        # reductions are directly comparable.
                        ls_rng2 = np.random.default_rng(seed + 909_093)
                        row["levelset_unexplained_fisher_linear"] = \
                            level_set_unexplained(
                                lambda t: t @ fisher_v, cfg, d, ls_rng2,
                                int(args.levelset_n_pairs),
                            )

                        print(
                            f"[npe] raw={raw_lp:.4f}  analytic={an_lp:.4f}  "
                            f"discovered={di_lp:.4f}  fisher_linear={fl_lp:.4f}  "
                            f"(n_eta={n_eta} of {d})",
                            flush=True,
                        )

                except Exception as exc:  # keep the sweep alive
                    row["status"] = "failed"
                    row["failed_stage"] = stage
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"!!! {tag} failed at stage {stage}: {exc}", flush=True)
                    traceback.print_exc()

                for k, v in runtimes.items():
                    row[f"runtime_{k}_s"] = float(v)
                row["runtime_total_s"] = float(sum(runtimes.values()))
                rows.append(row)

                # Persist after every run so a wall-clock kill loses nothing.
                df = pd.DataFrame(rows)
                df.to_csv(metrics_path, index=False)
                write_recovery_table(df, out_dir)
                write_aggregate(df, out_dir)
                if spectra:
                    np.savez_compressed(out_dir / "rank_spectra.npz", **spectra)
                if expressions:
                    with open(out_dir / "expressions.json", "w") as fh:
                        json.dump(expressions, fh, indent=2)

                if not args.keep_workdirs and workdir.exists():
                    import shutil
                    shutil.rmtree(workdir, ignore_errors=True)

    df = pd.DataFrame(rows)
    if not df.empty:
        np.savez_compressed(
            out_dir / "metrics.npz",
            **{c: df[c].to_numpy() for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c])},
        )
    manifest["completed_runs"] = int((df["status"] == "ok").sum()) if not df.empty else 0
    manifest["failed_runs"] = int((df["status"] == "failed").sum()) if not df.empty else 0
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nwrote {metrics_path}", flush=True)
    print(f"wrote {out_dir / 'recovery_table.csv'}", flush=True)
    if not df.empty:
        print(
            f"completed {manifest['completed_runs']}, "
            f"failed {manifest['failed_runs']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
