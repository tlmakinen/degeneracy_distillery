"""One-step adapter for the SIR R0 discovery experiment.

Wraps the published three-step driver ``scripts/sir_notebook_run.py`` rather
than reimplementing the simulator, in the same way
``problems/rayleigh_benard.py`` wraps the RB DNS. The simulator, the
supercriticality cut, the theta scaling convention and the success criterion
all come from that file so the one-step numbers can be read against
``follow_up_results/sir/rebuttal/aggregate_summary.md``.

Two conventions are inherited deliberately:

* ``theta`` is handed to the driver already scaled to ``(1.0, 2.0)``, matching
  ``fit_theta_scaler(..., feature_range=(1.0, 2.0))`` in the published script.
  The one-step driver feeds whatever ``sample``/``sample_prior`` return straight
  to Operon, so scaling here is what keeps expression complexity and MDL on the
  same footing as the published table.
* ``gates`` converts the discovered expressions back to physical
  ``(beta, gamma, I0_over_10)`` with ``expressions_to_physical`` and then calls
  the published ``r0_correlations_and_gradients`` unchanged, so
  ``physics_alignment`` is the identical statistic.

``use_log_features`` is left False on purpose. Scaled theta is strictly
positive so log features would be safe, but ``log(beta) - log(gamma)`` is
exactly ``log R0`` -- handing the map that feature would make the comparison
against a three-step flattener that never had it flattering rather than fair.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Matches SR_OFFSET / SR_LENGTH_PENALTY in scripts/sir_notebook_run.py.
SR_OFFSET = 0.0
FEATURE_RANGE = (1.0, 2.0)
THETA_NAMES = ("beta", "gamma", "I0_over_10")


def _load_sir_module():
    """Import the published driver as a library (it is __main__-guarded)."""
    name = "sir_notebook_run"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[2] / "scripts" / "sir_notebook_run.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the SIR simulator from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Dataclasses read sys.modules[cls.__module__] at class-body time.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class SIRProblem:
    """SIR epidemic curves; the informative coordinate is ``R0 = beta / gamma``."""

    def __init__(
        self,
        seed: int = 0,
        nsims: int = 500,
        mode: str = "rebuttal",
        min_r0_corr: float = 0.5,
        **_ignored,
    ):
        self.name = "sir"
        self.d = 3
        self.param_names = THETA_NAMES
        self.use_log_features = False
        self.m_probe = 3
        # Deliberately None. R0 = beta/gamma is *a* recoverable coordinate, not
        # evidence that the system is rank 1: the infection curve also pins the
        # timescale, so beta and gamma are separately informative. The published
        # three-step records agree -- SIR seeds report rank_deficient=False and
        # pick best_component 0 or 1, i.e. more than one coordinate is kept. The
        # success gate is the R0 correlation, as in the published criterion.
        self.expected_rank = None

        # Theta reaches the driver already scaled into FEATURE_RANGE.
        lo, hi = FEATURE_RANGE
        self.theta_lo = np.full(self.d, lo, dtype=np.float64)
        self.theta_hi = np.full(self.d, hi, dtype=np.float64)

        self._seed = int(seed)
        self._nsims = int(nsims)
        self._mode = str(mode)
        self.min_r0_corr = float(min_r0_corr)

        self._mod = None
        self._data: dict[str, np.ndarray] | None = None
        self._scaler = None
        self._calls = 0
        self.last_aux: dict[str, Any] = {}

    # -- simulator ---------------------------------------------------------
    def _sir(self):
        if self._mod is None:
            self._mod = _load_sir_module()
        return self._mod

    def _simulate(self) -> dict[str, np.ndarray]:
        """Run the published simulator once; it returns train *and* test."""
        if self._data is not None:
            return self._data
        mod = self._sir()
        base = mod.CONFIGS.get(self._mode) or mod.CONFIGS["rebuttal"]
        config = dataclasses.replace(base, nsims=self._nsims)
        data = mod.simulator_data(config, seed=self._seed)
        self._scaler = mod.fit_theta_scaler(
            data["theta_train"], feature_range=FEATURE_RANGE,
        )
        self._data = data
        return data

    def _scaled(self, theta: np.ndarray) -> np.ndarray:
        self._simulate()
        return np.asarray(
            self._scaler.transform(np.asarray(theta, dtype=np.float64)),
            dtype=np.float64,
        )

    def _physical(self, theta_scaled: np.ndarray) -> np.ndarray:
        self._simulate()
        return np.asarray(
            self._scaler.inverse_transform(np.asarray(theta_scaled, dtype=np.float64)),
            dtype=np.float64,
        )

    def sample(self, n: int, rng: np.random.Generator):
        """First call returns the train split, second the held-out split.

        ``simulator_data`` produces both in one call and seeds them exactly as
        the published run does, so splitting them here preserves that seeding
        rather than re-deriving it.
        """
        data = self._simulate()
        n = int(n)
        if n > self._nsims:
            raise ValueError(
                f"sir adapter was built for nsims={self._nsims} but was asked for "
                f"{n} rows; pass --problem-arg nsims={n} (or raise --nsims)."
            )
        split = "train" if self._calls == 0 else "test"
        self._calls += 1
        theta = np.asarray(data[f"theta_{split}"], dtype=np.float64)[:n]
        x = np.asarray(data[f"data_{split}"], dtype=np.float64)[:n]
        self.last_aux = {"split": split, "theta_physical": theta}
        return self._scaled(theta), x

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Prior draws for SR augmentation, on the same measure as training.

        Reproduces ``_generate``'s supercriticality cut and the Poisson initial
        condition; a plain uniform box would put the augmented rows on a
        different measure from the rows the map was fitted on.
        """
        mod = self._sir()
        n = int(n)
        keep_b: list[np.ndarray] = []
        keep_g: list[np.ndarray] = []
        have = 0
        # The cut retains ~60% of the box; oversample and loop so a short draw
        # cannot silently return fewer rows than asked for.
        while have < n:
            pool = max(4 * (n - have), 128)
            b = rng.uniform(mod.BETA_MIN, mod.BETA_MAX, pool)
            g = rng.uniform(mod.GAMMA_MIN, mod.GAMMA_MAX, pool)
            ok = (b / g) >= (1.0 + mod.DELTA)
            keep_b.append(b[ok])
            keep_g.append(g[ok])
            have += int(ok.sum())
        beta = np.concatenate(keep_b)[:n]
        gamma = np.concatenate(keep_g)[:n]
        i0 = rng.poisson(mod.I0_MEAN, size=n) / 10.0
        theta = np.stack([beta, gamma, i0], axis=1)
        return self._scaled(theta)

    # -- model -------------------------------------------------------------
    def estimator_factory(self, m: int):
        from degeneracy_distillery.oneshot import Estimator

        return Estimator(m=int(m), features=(256, 256, 128))

    def truth_coords(self, theta: np.ndarray) -> np.ndarray:
        """R0 = beta / gamma, recovered from the scaled theta handed to us."""
        phys = self._physical(theta)
        r0 = phys[:, 0] / phys[:, 1]
        return r0[:, None]

    # -- scoring -----------------------------------------------------------
    def _physical_exprs(self, expression: str | None):
        """Parse the driver's ' | '-joined X1..Xd stack into physical sympy."""
        if not expression:
            return None
        import sympy

        from degeneracy_distillery.sr_utils import expressions_to_physical

        parts = [p.strip() for p in str(expression).split("|") if p.strip()]
        if not parts:
            return None
        exprs = [sympy.sympify(p) for p in parts]
        return expressions_to_physical(
            exprs,
            self._scaler,
            sr_offset=SR_OFFSET,
            theta_names=THETA_NAMES,
            decimal=3,
        )

    def gates(self, payload: dict) -> dict[str, Any]:
        # No rank_correct key: there is no ground-truth rank to check against.
        out: dict[str, Any] = {}

        theta_scaled = payload.get("theta")
        exprs = None
        try:
            exprs = self._physical_exprs(payload.get("expression"))
        except Exception as exc:  # a malformed expression must not fail the run
            out["gate_error"] = f"parse: {type(exc).__name__}: {exc}"

        if exprs is None or theta_scaled is None:
            out["physics_alignment"] = float("nan")
            out["gate_ok"] = False
            return out

        try:
            mod = self._sir()
            theta_phys = self._physical(theta_scaled)
            corr = mod.r0_correlations_and_gradients(exprs, theta_phys)
            out["physics_alignment"] = float(corr["best_pearson_abs"])
            out["physics_alignment_spearman"] = float(corr["best_spearman_abs"])
            out["physics_alignment_grad_cosine"] = float(corr["best_grad_cosine"])
            out["gate_ok"] = bool(corr["best_pearson_abs"] >= self.min_r0_corr)
        except Exception as exc:
            out["physics_alignment"] = float("nan")
            out["gate_ok"] = False
            out["gate_error"] = f"score: {type(exc).__name__}: {exc}"
        return out

    def sr_hints(self) -> dict:
        # Matches run_symbolic_regression in scripts/sir_notebook_run.py.
        return {
            "allowed_symbols": "add,mul,div,pow,constant,variable",
            "max_length": 25,
            "max_depth": 10,
        }
