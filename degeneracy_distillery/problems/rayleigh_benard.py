"""Rayleigh-Benard adapter: nondimensional gate and raw-unit discovery.

``n_params=3`` is the existing ``(log10 Ra, log10 Pr, log10 Gamma)`` setup.
``n_params=8`` samples raw Boussinesq units through ``PiSystem`` and feeds
the existing DNS ``single(key, log10 Pi)``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from degeneracy_distillery.dimensional import rayleigh_benard_system

THETA_REF = np.array(
    [9.81, 3.4e-3, 10.0, 0.05, 1.0e-6, 1.4e-7, 0.08, 4184.0],
    dtype=np.float64,
)
LOG_DECADE_HALF = 0.75
Z_HALF = 0.25
CP_INDEX = 7


def _load_rb_module():
    import sys

    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "rayleigh_benard_notebook_run.py"
    )
    spec = importlib.util.spec_from_file_location("rayleigh_benard_notebook_run", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load RB DNS from {path}")
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    mod = importlib.util.module_from_spec(spec)
    # Dataclasses read sys.modules[cls.__module__] at class-body time.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _nusselt_from_eta(eta: np.ndarray, log_pi: np.ndarray, nu: np.ndarray) -> dict:
    """Correlate neural axes with log Nu and match the (Ra, Pr) gradient."""
    eta = np.asarray(eta, dtype=np.float64)
    log_pi = np.asarray(log_pi, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    ok = np.isfinite(nu) & (nu > 0) & np.isfinite(eta).all(1) & np.isfinite(log_pi).all(1)
    if ok.sum() < 10:
        return {
            "best_nusselt_abs_corr": 0.0,
            "best_nusselt_rapr_cosine": 0.0,
        }
    eta, log_pi, nu = eta[ok], log_pi[ok], nu[ok]
    log_nu = np.log(nu)
    corrs = []
    for j in range(eta.shape[1]):
        if eta[:, j].std() < 1e-15:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(eta[:, j], log_nu)[0, 1]))
    design = np.column_stack([np.ones(ok.sum()), log_pi])
    coef_nu, *_ = np.linalg.lstsq(design, log_nu, rcond=None)
    ref = coef_nu[1:3]
    ref = ref / (np.linalg.norm(ref) + 1e-12)
    cosines = []
    for j in range(eta.shape[1]):
        coef, *_ = np.linalg.lstsq(design, eta[:, j], rcond=None)
        g = coef[1:3]
        nrm = np.linalg.norm(g)
        cosines.append(
            float(abs(np.dot(g, ref) / (nrm + 1e-12))) if nrm > 1e-12 else 0.0
        )
    return {
        "best_nusselt_abs_corr": float(max(abs(c) for c in corrs)),
        "best_nusselt_rapr_cosine": float(max(cosines) if cosines else 0.0),
        "axis_nusselt_corr": corrs,
        "axis_nusselt_cosine": cosines,
    }


class RayleighBenardProblem:
    """One-step adapter around the existing stress-free RB DNS."""

    def __init__(
        self,
        n_params: int = 3,
        mode: str = "smoke",
        seed: int = 0,
        min_nusselt_corr: float = 0.9,
        min_nusselt_cosine: float = 0.9,
        nsims: int | None = None,
        **_ignored,
    ):
        self.n_params = int(n_params)
        if self.n_params not in (3, 8):
            raise ValueError("n_params must be 3 (gate) or 8 (raw units)")
        self.mode = str(mode)
        self.min_nusselt_corr = float(min_nusselt_corr)
        self.min_nusselt_cosine = float(min_nusselt_cosine)
        self.name = "rayleigh_benard"
        self.expected_rank = 3
        self.m_probe = 5
        self.pi = rayleigh_benard_system()
        self.last_aux: dict[str, Any] = {}
        self._mod = None
        self._single = None
        self._config = None
        self._nsims_override = nsims

        if self.n_params == 3:
            self.d = 3
            self.param_names = ("logRa", "logPr", "logGamma")
            self.theta_lo = np.array([3.3, -0.5, 0.0], dtype=np.float64)
            self.theta_hi = np.array([4.3, 0.5, 0.3], dtype=np.float64)
            self.use_log_features = False
        else:
            self.d = 8
            self.param_names = self.pi.param_names
            span = 10.0 ** LOG_DECADE_HALF
            self.theta_lo = THETA_REF / span
            self.theta_hi = THETA_REF * span
            self.use_log_features = True

    def _rb(self):
        if self._mod is None:
            self._mod = _load_rb_module()
        return self._mod

    def _dns(self):
        if self._single is not None:
            return self._single, self._config
        import jax
        # Metal cannot legalize mhlo.fft. Pin CPU before the first jit.
        if any(d.platform.upper() == "METAL" for d in jax.devices()):
            jax.config.update("jax_default_device", jax.devices("cpu")[0])
        rb = self._rb()
        cfg = rb.CONFIGS[self.mode]
        if self._nsims_override is not None:
            from dataclasses import replace
            cfg = replace(cfg, nsims=int(self._nsims_override))
        phi_sin, phi_cos, kz, inv_norm = rb._vertical_basis(cfg.nz)
        k_edges = rb._shell_edges(cfg.nz, cfg.n_spectral_bins)
        single = rb._make_single_sim(cfg, phi_sin, phi_cos, kz, inv_norm, k_edges)
        self._single = single
        self._config = cfg
        return single, cfg

    def _pi_box(self):
        lo = np.array([3.3, -0.5, 0.0], dtype=np.float64)
        hi = np.array([4.3, 0.5, 0.3], dtype=np.float64)
        return lo, hi

    def _z_box(self):
        n = self.pi.n_nuisance
        return np.full(n, -Z_HALF), np.full(n, Z_HALF)

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        n = int(n)
        if self.n_params == 3:
            lo, hi = self._pi_box()
            return rng.uniform(lo, hi, size=(n, 3))
        theta, _ = self.pi.sample(
            n, self._pi_box(), self._z_box(), rng, theta_ref=THETA_REF,
        )
        return theta

    def _run_dns(self, log_pi: np.ndarray, rng: np.random.Generator):
        import jax
        import jax.numpy as jnp
        import jax.random as jr

        if bool(jax.config.jax_enable_x64):
            raise RuntimeError(
                "RB DNS must run before jax_enable_x64 is set. "
                "Call sample() before importing degeneracy_distillery.oneshot."
            )
        single, cfg = self._dns()
        batched = jax.jit(jax.vmap(single))
        log_pi = np.asarray(log_pi, dtype=np.float32)
        n = int(log_pi.shape[0])
        seed = int(rng.integers(0, 2**31 - 1))
        key = jr.PRNGKey(seed)
        obs_chunks, nu_chunks, re_chunks = [], [], []
        for start in range(0, n, cfg.sim_chunk):
            stop = min(start + cfg.sim_chunk, n)
            key, sub = jr.split(key)
            keys = jr.split(sub, stop - start)
            obs, nu, re = batched(keys, jnp.asarray(log_pi[start:stop]))
            obs = np.asarray(obs)
            if not np.isfinite(obs).all():
                raise RuntimeError("non-finite RB spectra")
            obs_chunks.append(obs)
            nu_chunks.append(np.asarray(nu))
            re_chunks.append(np.asarray(re))
        return (
            np.concatenate(obs_chunks, 0).astype(np.float32),
            np.concatenate(nu_chunks, 0).astype(np.float32),
            np.concatenate(re_chunks, 0).astype(np.float32),
        )

    def sample(self, n: int, rng: np.random.Generator):
        n = int(n)
        if self.n_params == 3:
            lo, hi = self._pi_box()
            theta = rng.uniform(lo, hi, size=(n, 3))
            log_pi = theta
        else:
            theta, log_pi = self.pi.sample(
                n, self._pi_box(), self._z_box(), rng, theta_ref=THETA_REF,
            )
        x, nu, re = self._run_dns(log_pi, rng)
        self.last_aux = {
            "nu": nu,
            "re": re,
            "log_pi": np.asarray(log_pi, dtype=np.float64),
            "theta": np.asarray(theta, dtype=np.float64),
        }
        return np.asarray(theta, dtype=np.float64), x

    def estimator_factory(self, m: int):
        from degeneracy_distillery.oneshot import Estimator
        return Estimator(m=int(m), features=(256, 256, 128))

    def truth_coords(self, theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=np.float64)
        if self.n_params == 3:
            return th
        return self.pi.log_pi(th, log_base=10.0)

    def gates(self, payload: dict) -> dict[str, bool]:
        r_hat = int(payload.get("r_hat", -1))
        rank_ok = bool(r_hat == 3)
        aux = payload.get("aux") or self.last_aux
        eta = payload.get("eta")
        out: dict[str, bool] = {"rank_correct": rank_ok}

        if eta is not None and aux.get("nu") is not None:
            nu_stats = _nusselt_from_eta(eta, aux["log_pi"], aux["nu"])
            out["nusselt_corr_ok"] = bool(
                nu_stats["best_nusselt_abs_corr"] >= self.min_nusselt_corr
            )
            out["nusselt_cosine_ok"] = bool(
                nu_stats["best_nusselt_rapr_cosine"] >= self.min_nusselt_cosine
            )
            out["best_nusselt_abs_corr"] = nu_stats["best_nusselt_abs_corr"]  # type: ignore[assignment]
            out["best_nusselt_rapr_cosine"] = nu_stats["best_nusselt_rapr_cosine"]  # type: ignore[assignment]
        else:
            out["nusselt_corr_ok"] = False
            out["nusselt_cosine_ok"] = False

        if self.n_params == 8:
            order = list(payload.get("screen_order", []))
            out["cp_ranked_last"] = bool(order and int(order[-1]) == CP_INDEX)
            J = payload.get("J")
            theta = payload.get("theta")
            leak = float("nan")
            integer_match = False
            if J is not None and theta is not None:
                J = np.asarray(J, dtype=np.float64)
                th = np.asarray(theta, dtype=np.float64)
                J_log = J * th[:, None, :] * np.log(10.0)
                leak = self.pi.null_leakage(J_log)
                rec = self.pi.recover_exponents(J_log)
                integer_match = bool(rec["integer_match"])
                out["exponent_rel_err"] = rec["rel_err_round"]  # type: ignore[assignment]
            out["null_leakage"] = leak  # type: ignore[assignment]
            out["null_leakage_ok"] = bool(np.isfinite(leak) and leak < 0.05)
            out["exponents_recovered"] = integer_match
            out["gate_ok"] = bool(
                rank_ok
                and out.get("null_leakage_ok")
                and out.get("cp_ranked_last")
                and integer_match
            )
        else:
            out["gate_ok"] = bool(
                rank_ok
                and out.get("nusselt_corr_ok")
                and out.get("nusselt_cosine_ok")
            )
        return out

    def sr_hints(self) -> dict:
        return {
            "allowed_symbols": "add,mul,div,pow,constant,variable,log,exp,sqrt",
            "max_length": 25 if self.n_params == 3 else 32,
            "max_depth": 10 if self.n_params == 3 else 12,
        }
