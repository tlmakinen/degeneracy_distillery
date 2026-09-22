"""Unit tests for ``degeneracy_distillery.dimensional.PiSystem``."""
from __future__ import annotations

import numpy as np
import pytest

from degeneracy_distillery.dimensional import PiSystem, rayleigh_benard_system

LOG_RA = (3.3, 4.3)
LOG_PR = (-0.5, 0.5)
LOG_G = (0.0, 0.3)


def _pi_box():
    lo = np.array([LOG_RA[0], LOG_PR[0], LOG_G[0]])
    hi = np.array([LOG_RA[1], LOG_PR[1], LOG_G[1]])
    return lo, hi


def _z_box(sys: PiSystem, half: float = 0.3):
    n = sys.n_nuisance
    return np.full(n, -half), np.full(n, half)


def test_null_space_exact():
    sys = rayleigh_benard_system()
    residual = sys.A @ sys.null_basis
    assert residual.shape == (3, 5)
    assert np.max(np.abs(residual)) < 1e-12


def test_pinv_left_inverse():
    sys = rayleigh_benard_system()
    assert np.allclose(sys.A @ sys.A_pinv, np.eye(3), atol=1e-12)


def test_rank_and_nuisance():
    sys = rayleigh_benard_system()
    assert sys.k == 3
    assert sys.d == 8
    assert sys.n_nuisance == 5
    assert sys.param_names[-1] == "c_p"
    assert np.allclose(sys.A[:, -1], 0.0)


def test_sampler_round_trip():
    sys = rayleigh_benard_system()
    rng = np.random.default_rng(0)
    theta, log_pi = sys.sample(64, _pi_box(), _z_box(sys), rng)
    recovered = sys.log_pi(theta, log_base=10.0)
    assert theta.shape == (64, 8)
    assert np.all(theta > 0)
    assert np.allclose(recovered, log_pi, atol=1e-10)


def test_sampler_hits_dns_box():
    sys = rayleigh_benard_system()
    rng = np.random.default_rng(1)
    theta_ref = np.array([9.81, 3.4e-3, 10.0, 0.05, 1e-6, 1.4e-7, 0.08, 4184.0])
    theta, log_pi = sys.sample(
        200, _pi_box(), _z_box(sys, 0.25), rng, theta_ref=theta_ref,
    )
    lo, hi = _pi_box()
    assert np.all(log_pi >= lo - 1e-12)
    assert np.all(log_pi <= hi + 1e-12)
    recovered = sys.log_pi(theta, log_base=10.0)
    assert np.allclose(recovered, log_pi, atol=1e-10)
    log_th = np.log10(theta)
    log_ref = np.log10(theta_ref)
    assert np.all(np.abs(log_th - log_ref) < 1.6)


def test_null_leakage_zero_on_pi_function():
    sys = rayleigh_benard_system()
    rng = np.random.default_rng(2)
    theta, log_pi = sys.sample(40, _pi_box(), _z_box(sys), rng)
    # eta = (log Ra, log Pr, log Gamma) = A log10 theta, so J = A.
    J = np.broadcast_to(sys.A[None, :, :], (40, 3, 8)).copy()
    assert sys.null_leakage(J) < 1e-12


def test_null_leakage_order_one_on_nuisance():
    sys = rayleigh_benard_system()
    N = sys.null_basis
    # Jacobian lives entirely in the null space.
    J = np.broadcast_to(N.T[None, :3, :], (20, 3, 8)).copy()
    leak = sys.null_leakage(J)
    assert leak > 0.9


def test_recover_exponents_identity():
    sys = rayleigh_benard_system()
    rec = sys.recover_exponents(sys.A)
    assert rec["integer_match"]
    assert np.allclose(rec["C_round"], np.eye(3))
    assert rec["rel_err"] < 1e-12


def test_reject_rank_deficient_A():
    A = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="full row rank"):
        PiSystem(A, param_names=("a", "b", "c"), pi_names=("p", "q"))
