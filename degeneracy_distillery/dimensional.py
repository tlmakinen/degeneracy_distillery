"""Buckingham Pi algebra for raw-unit sampling and scoring.

``PiSystem`` stores the exponent matrix ``A`` with ``log Pi = A log theta``.
Sampling inverts that map on the row space and adds an orthonormal null
component. Scoring asks whether a learned Jacobian lives in ``row(A)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PiSystem:
    """Linear map from ``log theta`` in ``R^d`` to ``log Pi`` in ``R^k``."""

    exponents: np.ndarray
    param_names: tuple[str, ...]
    pi_names: tuple[str, ...]

    def __post_init__(self) -> None:
        A = np.asarray(self.exponents, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError(f"exponents must be 2d, got shape {A.shape}")
        k, d = A.shape
        if len(self.param_names) != d:
            raise ValueError(
                f"param_names has {len(self.param_names)} entries, A has {d} columns"
            )
        if len(self.pi_names) != k:
            raise ValueError(
                f"pi_names has {len(self.pi_names)} entries, A has {k} rows"
            )
        rank = int(np.linalg.matrix_rank(A, tol=1e-10))
        if rank != k:
            raise ValueError(f"A must have full row rank {k}, got rank {rank}")
        self.exponents = A

    @property
    def A(self) -> np.ndarray:
        return self.exponents

    @property
    def k(self) -> int:
        return int(self.A.shape[0])

    @property
    def d(self) -> int:
        return int(self.A.shape[1])

    @property
    def n_nuisance(self) -> int:
        return self.d - self.k

    @property
    def A_pinv(self) -> np.ndarray:
        """Moore-Penrose right inverse. ``A @ A_pinv = I_k``."""
        return np.linalg.pinv(self.A)

    @property
    def null_basis(self) -> np.ndarray:
        """Orthonormal columns spanning ``null(A)``. Shape ``(d, d-k)``."""
        u, s, vh = np.linalg.svd(self.A, full_matrices=True)
        return vh[self.k :].T.copy()

    @property
    def row_projector(self) -> np.ndarray:
        """``P = A^+ A`` onto ``row(A)``. Shape ``(d, d)``."""
        return self.A_pinv @ self.A

    def log_pi(self, theta: np.ndarray, *, log_base: float = 10.0) -> np.ndarray:
        """``log_b Pi`` from raw ``theta``. ``theta`` is ``(n, d)`` or ``(d,)``."""
        th = np.asarray(theta, dtype=np.float64)
        single = th.ndim == 1
        if single:
            th = th[None, :]
        if np.any(th <= 0):
            raise ValueError("theta must be strictly positive")
        log_th = np.log(th) / np.log(log_base)
        out = log_th @ self.A.T
        return out[0] if single else out

    def sample(
        self,
        n: int,
        pi_box: tuple[np.ndarray, np.ndarray],
        z_box: tuple[np.ndarray, np.ndarray],
        rng: np.random.Generator,
        *,
        theta_ref: np.ndarray | None = None,
        log_base: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw raw ``theta`` whose Pi groups lie in ``pi_box``.

        Parameters
        ----------
        pi_box
            ``(lo, hi)`` for ``log_b Pi``, each of shape ``(k,)``.
        z_box
            ``(lo, hi)`` for the ``d-k`` null coordinates.
        theta_ref
            Particular solution centre. Default is the min-norm inverse.

        Returns
        -------
        theta_raw : (n, d)
        log_pi : (n, k)
            Induced ``log_b Pi = A log_b theta``.
        """
        n = int(n)
        lo_pi, hi_pi = (np.asarray(v, dtype=np.float64).reshape(self.k) for v in pi_box)
        lo_z, hi_z = (np.asarray(v, dtype=np.float64).reshape(self.n_nuisance) for v in z_box)
        log_pi = rng.uniform(lo_pi, hi_pi, size=(n, self.k))
        z = rng.uniform(lo_z, hi_z, size=(n, self.n_nuisance))
        N = self.null_basis
        if theta_ref is None:
            log_th = (self.A_pinv @ log_pi.T).T + (N @ z.T).T
        else:
            log_ref = np.log(np.asarray(theta_ref, dtype=np.float64).reshape(self.d))
            log_ref = log_ref / np.log(log_base)
            log_pi_ref = self.A @ log_ref
            log_th = (
                log_ref
                + (self.A_pinv @ (log_pi - log_pi_ref).T).T
                + (N @ z.T).T
            )
        theta = np.power(log_base, log_th)
        return theta, log_pi

    def null_leakage(self, J_log: np.ndarray) -> float:
        """Mean fraction of Jacobian energy outside ``row(A)``.

        ``J_log`` is ``d eta / d log theta``, shape ``(n, m, d)`` or
        ``(m, d)``. Zero means every axis is a function of the Pi groups.
        """
        J = np.asarray(J_log, dtype=np.float64)
        if J.ndim == 2:
            J = J[None, ...]
        if J.ndim != 3 or J.shape[-1] != self.d:
            raise ValueError(f"J_log must be (n, m, {self.d}), got {J.shape}")
        P = self.row_projector
        energy = np.einsum("nmi,nmi->nm", J, J)
        in_row = np.einsum("nmi,ij,nmj->nm", J, P, J)
        frac = in_row / np.maximum(energy, 1e-30)
        return float(1.0 - np.mean(frac))

    def recover_exponents(self, J_log: np.ndarray) -> dict:
        """Least-squares mix of ``A`` rows that matches the mean log-Jacobian.

        Writes ``J_bar ≈ C @ A``. Integer-rounded ``C`` is the SR verdict:
        a permutation-and-sign mix of the Pi groups.
        """
        J = np.asarray(J_log, dtype=np.float64)
        if J.ndim == 3:
            J_bar = J.mean(0)
        elif J.ndim == 2:
            J_bar = J
        else:
            raise ValueError(f"J_log must be (n, m, d) or (m, d), got {J.shape}")
        C, *_ = np.linalg.lstsq(self.A.T, J_bar.T, rcond=None)
        C = np.asarray(C, dtype=np.float64).T
        reconstructed = C @ self.A
        C_round = np.round(C)
        reconstructed_round = C_round @ self.A
        denom = np.linalg.norm(J_bar) + 1e-30
        rel_err = float(np.linalg.norm(J_bar - reconstructed) / denom)
        rel_err_round = float(np.linalg.norm(J_bar - reconstructed_round) / denom)
        return {
            "C": C,
            "C_round": C_round,
            "reconstructed": reconstructed,
            "reconstructed_round": reconstructed_round,
            "rel_err": rel_err,
            "rel_err_round": rel_err_round,
            "integer_match": bool(rel_err_round < 0.15),
        }


def rayleigh_benard_system() -> PiSystem:
    """Eight raw Boussinesq parameters and the three Pi groups.

    Parameters: ``g, alpha, dT, H, nu, kappa, Lx, c_p``.
    Groups: ``Ra, Pr, Gamma``. ``c_p`` has a zero column.
    """
    A = np.array(
        [
            [1.0, 1.0, 1.0, 3.0, -1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return PiSystem(
        exponents=A,
        param_names=("g", "alpha", "dT", "H", "nu", "kappa", "Lx", "c_p"),
        pi_names=("Ra", "Pr", "Gamma"),
    )
