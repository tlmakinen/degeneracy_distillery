"""One-step adapters for the two gravitational-wave discovery experiments.

Both wrap a published three-step driver rather than reimplementing the
waveform model, in the same way ``problems/rayleigh_benard.py`` wraps the RB
DNS and ``problems/sir.py`` wraps the SIR simulator:

* ``GWTaylorF2Problem``  -> ``scripts/gw_notebook_run.py``
* ``GWIMRPhenomDProblem`` -> ``scripts/imrphenomd_notebook_run.py``

The two share their parameter space (``m1``, ``m2``), their PCA-compressed
observable and their theta scaling, and differ only in the simulator module and
the success criterion, so the common machinery lives in ``_GWProblem``.

As in the SIR adapter, ``theta`` reaches the driver already scaled to
``(1.0, 2.0)`` to match ``fit_theta_scaler`` in the published scripts, and
``gates`` converts the discovered expressions back to physical ``(m1, m2)``
with ``expressions_to_physical`` before calling the published correlation
function unchanged. ``use_log_features`` is left False for the same reason as
SIR: chirp mass is a power law in the masses, so log features would hand the
one-step map a linearising feature the three-step flattener never had.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SR_OFFSET = 0.0
FEATURE_RANGE = (1.0, 2.0)
THETA_NAMES = ("m1", "m2")


def _load_module(script_name: str):
    """Import a published driver as a library (they are __main__-guarded)."""
    if script_name in sys.modules:
        return sys.modules[script_name]
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the GW simulator from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Dataclasses read sys.modules[cls.__module__] at class-body time.
    sys.modules[script_name] = mod
    spec.loader.exec_module(mod)
    return mod


class _GWProblem:
    """Shared machinery for the two (m1, m2) waveform problems."""

    script_name: str = ""
    name: str = ""

    def __init__(
        self,
        seed: int = 0,
        nsims: int = 500,
        mode: str = "rebuttal",
        min_physics_corr: float = 0.75,
        **_ignored,
    ):
        self.d = 2
        self.param_names = THETA_NAMES
        self.use_log_features = False
        self.m_probe = 2
        # None for the same reason as SIR: the published records report
        # rank_deficient=False, so there is no ground-truth rank to score
        # against. The success gate is the published correlation criterion.
        self.expected_rank = None

        lo, hi = FEATURE_RANGE
        self.theta_lo = np.full(self.d, lo, dtype=np.float64)
        self.theta_hi = np.full(self.d, hi, dtype=np.float64)

        self._seed = int(seed)
        self._nsims = int(nsims)
        self._mode = str(mode)
        self.min_physics_corr = float(min_physics_corr)

        self._mod = None
        self._data: dict[str, Any] | None = None
        self._scaler = None
        self._calls = 0
        self._workdir: tempfile.TemporaryDirectory | None = None
        self.last_aux: dict[str, Any] = {}

    # -- simulator ---------------------------------------------------------
    def _gw(self):
        if self._mod is None:
            self._mod = _load_module(self.script_name)
        return self._mod

    def _simulate(self) -> dict[str, Any]:
        """Run the published simulator once; it returns train *and* test.

        The PCA basis is fitted inside that call on the noiseless waveforms
        that train and test already required, so the basis stays shared across
        the split exactly as the published run had it -- which is what keeps
        the PCA waveform accounting honest.
        """
        if self._data is not None:
            return self._data
        mod = self._gw()
        base = mod.CONFIGS.get(self._mode) or mod.CONFIGS["rebuttal"]
        config = dataclasses.replace(base, nsims=self._nsims)
        # outdir only receives an input-summary PNG; keep it out of the repo.
        self._workdir = tempfile.TemporaryDirectory(prefix=f"{self.name}_sim_")
        data = mod.simulator_data(config, seed=self._seed, outdir=Path(self._workdir.name))
        self._scaler = mod.fit_theta_scaler(
            np.asarray(data["theta_train"]), feature_range=FEATURE_RANGE,
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
        """First call returns the train split, second the held-out split."""
        data = self._simulate()
        n = int(n)
        if n > self._nsims:
            raise ValueError(
                f"{self.name} adapter was built for nsims={self._nsims} but was "
                f"asked for {n} rows; raise --nsims."
            )
        split = "train" if self._calls == 0 else "test"
        self._calls += 1
        theta = np.asarray(data[f"theta_{split}"], dtype=np.float64)[:n]
        x = np.asarray(data[f"data_{split}"], dtype=np.float64)[:n]
        self.last_aux = {
            "split": split,
            "theta_physical": theta,
            "n_pca_simulations": int(data.get("n_pca_basis_waveforms", 0) or 0),
        }
        return self._scaled(theta), x

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform on the mass box, matching simulator_data and the criterion.

        ``simulator_data`` draws ``2*nsims`` uniform masses and splits them into
        train and test -- there is no rejection cut -- and the published
        correlation functions sample the same box internally.
        """
        mod = self._gw()
        n = int(n)
        theta = rng.uniform(
            [mod.M1_MIN, mod.M2_MIN], [mod.M1_MAX, mod.M2_MAX], size=(n, 2),
        )
        return self._scaled(theta)

    # -- model -------------------------------------------------------------
    def estimator_factory(self, m: int):
        from degeneracy_distillery.oneshot import Estimator

        return Estimator(m=int(m), features=(256, 256, 128))

    def truth_coords(self, theta: np.ndarray) -> np.ndarray:
        """log chirp mass -- the coordinate the waveform phase is built on.

        Only the primary target is returned. ``r2_true_min`` takes the minimum
        over columns, so adding a weakly-determined second coordinate would
        drag that diagnostic down for reasons unrelated to the discovery.
        """
        mod = self._gw()
        phys = self._physical(theta)
        mc = mod.chirp_mass(phys[:, 0], phys[:, 1])
        return np.log(np.asarray(mc, dtype=np.float64))[:, None]

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

    def _score(self, exprs) -> dict[str, Any]:
        raise NotImplementedError

    def gates(self, payload: dict) -> dict[str, Any]:
        out: dict[str, Any] = {}
        aux = payload.get("aux") or self.last_aux
        if aux.get("n_pca_simulations"):
            out["n_pca_simulations"] = int(aux["n_pca_simulations"])

        exprs = None
        try:
            exprs = self._physical_exprs(payload.get("expression"))
        except Exception as exc:  # a malformed expression must not fail the run
            out["gate_error"] = f"parse: {type(exc).__name__}: {exc}"

        if exprs is None:
            out["physics_alignment"] = float("nan")
            out["gate_ok"] = False
            return out

        try:
            out.update(self._score(exprs))
        except Exception as exc:
            out["physics_alignment"] = float("nan")
            out["gate_ok"] = False
            out["gate_error"] = f"score: {type(exc).__name__}: {exc}"
        return out

    def sr_hints(self) -> dict:
        # Matches run_symbolic_regression in the published driver.
        return {
            "allowed_symbols": "add,mul,div,pow,constant,variable,exp",
            "max_length": 30,
            "max_depth": 20,
        }


class GWTaylorF2Problem(_GWProblem):
    """TaylorF2 inspiral waveforms; the target is chirp-mass-like structure."""

    script_name = "gw_notebook_run"
    name = "gw_taylorf2"

    def _score(self, exprs) -> dict[str, Any]:
        mod = self._gw()
        corr = mod.physics_correlations(exprs, self._seed)
        best = float(corr["best_abs_corr"])
        return {
            "physics_alignment": best,
            "chirp_mass_alignment": float(corr["best_chirp_mass_abs_corr"]),
            "gate_ok": bool(best >= self.min_physics_corr),
        }


class GWIMRPhenomDProblem(_GWProblem):
    """IMRPhenomD merger-ringdown waveforms; the target is the dual (M, Δm).

    The published criterion is a *conjunction*: total-mass |r| >= min_total and
    mass-difference |r| >= min_mass_diff (imrphenomd_notebook_run.py:843-845).
    Physics alignment is reported as the total-mass correlation and the
    mass-difference correlation is carried alongside as
    ``complementary_mass_diff_alignment``, matching the published
    ``run_record.json`` field names so
    ``scripts/recompute_success_at_threshold.EXPERIMENTS['gw_imrphenomd']``
    reads it directly.
    """

    script_name = "imrphenomd_notebook_run"
    name = "gw_imrphenomd"

    def __init__(
        self,
        seed: int = 0,
        nsims: int = 500,
        mode: str = "rebuttal",
        min_total_mass_corr: float = 0.75,
        min_mass_diff_corr: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            nsims=nsims,
            mode=mode,
            min_physics_corr=float(min_total_mass_corr),
            **kwargs,
        )
        self.min_total_mass_corr = float(min_total_mass_corr)
        self.min_mass_diff_corr = float(min_mass_diff_corr)

    def _score(self, exprs) -> dict[str, Any]:
        mod = self._gw()
        corr = mod.mass_targets_correlations(exprs, self._seed)
        total = float(corr["total_mass_abs_corr"])
        diff = float(corr["mass_difference_abs_corr"])
        return {
            "physics_alignment": total,
            "complementary_mass_diff_alignment": diff,
            "gate_ok": bool(
                total >= self.min_total_mass_corr
                and diff >= self.min_mass_diff_corr
            ),
        }
