"""Problem adapter protocol for the one-step sweep driver.

The driver owns the screen, the rank ladder, the K-member ensemble,
alignment, SR, and frozen NLL selection. A Problem supplies only the
parts that change between simulators.

Dataset-backed problems (CAMELS, QM7b, weak lensing) have no box prior.
``sample_prior`` must then draw from the empirical parameter measure.
The volume term in ``train_oneshot`` still averages ``log det JJ^T``
over the training pairs, not over a fresh prior draw. That is a known
limit of the current loss, not of this protocol.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Problem(Protocol):
    """Eight fields plus six methods. That is the adapter surface.

    Attributes
    ----------
    name
        Registry key written to ``metrics.csv``.
    d
        Ambient parameter dimension.
    param_names
        One name per coordinate of ``theta``.
    theta_lo, theta_hi
        Bounds used to standardise the flattener input.
    use_log_features
        If true, the flattener concatenates ``log(theta)``. Requires
        every coordinate to stay positive.
    m_probe
        Starting rank of ``whittle_ladder``.
    expected_rank
        Ground-truth rank, or None when it is unknown.
    """

    name: str
    d: int
    param_names: tuple[str, ...]
    theta_lo: np.ndarray
    theta_hi: np.ndarray
    use_log_features: bool
    m_probe: int
    expected_rank: int | None

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``n`` pairs ``(theta, x)``."""
        ...

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` prior ``theta`` rows. Used for SR augmentation."""
        ...

    def estimator_factory(self, m: int) -> Any:
        """Return a flax module ``x -> R^m`` for ``eta_psi``."""
        ...

    def truth_coords(self, theta: np.ndarray) -> np.ndarray | None:
        """Closed-form coordinates for ``R^2`` scoring, or None."""
        ...

    def gates(self, payload: dict) -> dict[str, bool]:
        """Problem-specific pass/fail checks on a finished run."""
        ...

    def sr_hints(self) -> dict:
        """Operon knobs: ``allowed_symbols``, ``max_length``, ``max_depth``."""
        ...


_REGISTRY: dict[str, Callable[..., Problem]] = {}


def register(name: str, factory: Callable[..., Problem]) -> None:
    _REGISTRY[name] = factory


def get_problem(name: str, **kwargs) -> Problem:
    """Build a registered problem. Imports the adapter on first use."""
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        _load_builtin(key)
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none loaded)"
        raise KeyError(f"unknown problem {name!r}. known: {known}")
    return _REGISTRY[key](**kwargs)


def _load_builtin(name: str) -> None:
    if name == "rosenbrock":
        from degeneracy_distillery.problems.rosenbrock import RosenbrockProblem
        register("rosenbrock", RosenbrockProblem)
        return
    if name in ("rayleigh_benard", "rb"):
        from degeneracy_distillery.problems.rayleigh_benard import RayleighBenardProblem
        register("rayleigh_benard", RayleighBenardProblem)
        register("rb", RayleighBenardProblem)
        return
    if name == "sir":
        from degeneracy_distillery.problems.sir import SIRProblem
        register("sir", SIRProblem)
        return
    if name in ("gw_taylorf2", "gw"):
        from degeneracy_distillery.problems.gw import GWTaylorF2Problem
        register("gw_taylorf2", GWTaylorF2Problem)
        register("gw", GWTaylorF2Problem)
        return
    if name in ("gw_imrphenomd", "imrphenomd", "imr"):
        from degeneracy_distillery.problems.gw import GWIMRPhenomDProblem
        register("gw_imrphenomd", GWIMRPhenomDProblem)
        register("imrphenomd", GWIMRPhenomDProblem)
        register("imr", GWIMRPhenomDProblem)
        return


def available_problems() -> tuple[str, ...]:
    for key in ("rosenbrock", "rayleigh_benard", "sir", "gw_taylorf2", "gw_imrphenomd"):
        try:
            _load_builtin(key)
        except Exception:
            pass
    return tuple(sorted(_REGISTRY))
