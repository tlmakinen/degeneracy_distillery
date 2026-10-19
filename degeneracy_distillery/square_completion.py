"""Square-completed rectangular loss for K < d.

The K < d rectangular loss uses ``sqrt(det(J J^T))`` and is not a density with
the right prior: by the coarea formula it puts mass uniformly in arc length
along each fibre of eta, so the prior never enters, and it carries an
intractable normaliser.

The fix is to square-complete eta. Take a bijection
``eta = (eta_a, eta_b): R^d -> R^K x R^(d-K)`` and pad the statistic with
zeros. Only ``eta_a`` sees the data::

    L = 0.5||eta_a(theta) - t(x)||^2 + 0.5||eta_b(theta)||^2
        - log|det d eta / d theta| + (d/2) log 2 pi

This is the K = d rectangular loss with ``t~ = (t, 0)``. It is normalised by
change of variables, so its expectation is ``E_x KL(p || q) + const`` and the
held-out value is a proper score. ``eta_b`` never sees data, so
``N(eta_b; 0, I) |det J|`` is a data-independent density along the fibres --
the prior conditional the rectangular loss could not express.

The bijection matters. A skip-plus-MLP eta can fold, which breaks the change
of variables and lets the loss fall below any proper posterior. This module
uses affine coupling layers, whose Jacobian is triangular by construction.

Reference: `square_completion_loss.md`, and `unified-sbi`
`src/unified_sbi/shared/nuisance.py`.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

#: logit scale, matching the reference EtaD
LOGIT_S = math.pi / math.sqrt(3.0)


class LogitBox(nn.Module):
    """theta on a box -> y on R^d, with the per-coordinate log Jacobian.

    eta must be onto R^d for the change of variables to hold, so a bounded
    prior needs logit coordinates. Returns ``(y, sum_i log|dy_i/dtheta_i|)``.
    """

    lo: jnp.ndarray
    hi: jnp.ndarray
    eps: float = 1e-6

    def __call__(self, theta):
        lo = jnp.asarray(self.lo)
        hi = jnp.asarray(self.hi)
        u = jnp.clip((theta - lo) / (hi - lo), self.eps, 1.0 - self.eps)
        y = (jnp.log(u) - jnp.log1p(-u)) / LOGIT_S
        ladj = -jnp.log(LOGIT_S * (hi - lo) * u * (1.0 - u))
        return y, jnp.sum(ladj)


class AffineCoupling(nn.Module):
    """One affine coupling layer. ``mask`` selects the pass-through half.

    The transformed half is ``y * exp(s(y_masked)) + t(y_masked)``, so the
    Jacobian is triangular and ``log|det|`` is ``sum(s)`` over the transformed
    coordinates. The scale is bounded by a tanh so training cannot run away.

    The mask is a call argument, not a field: flax treats dataclass fields as
    static, and an array there is not hashable.
    """

    features: Sequence[int] = (128, 128)
    scale_cap: float = 2.0
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, y, mask):
        m = jnp.asarray(mask, dtype=y.dtype)
        h = y * m
        for w in self.features:
            h = nn.gelu(nn.Dense(w, param_dtype=self.param_dtype)(h))
        st = nn.Dense(
            2 * y.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            param_dtype=self.param_dtype,
        )(h)
        s, t = jnp.split(st, 2, axis=-1)
        s = self.scale_cap * jnp.tanh(s) * (1.0 - m)
        t = t * (1.0 - m)
        out = y * m + (1.0 - m) * (y * jnp.exp(s) + t)
        return out, jnp.sum(s)


class CouplingFlow(nn.Module):
    """A stack of affine couplings with alternating masks. Bijective R^d -> R^d.

    ``n_layers`` should be even so every coordinate is transformed equally
    often. Each layer starts at the identity, because the scale and shift head
    is zero-initialised, so the flow begins as the logit map alone.
    """

    d: int
    n_layers: int = 8
    features: Sequence[int] = (128, 128)
    param_dtype: Any = jnp.float64
    permute: bool = True
    perm_seed: int = 0

    @nn.compact
    def __call__(self, y):
        ladj = jnp.zeros((), dtype=y.dtype)
        rng = np.random.default_rng(self.perm_seed)
        for i in range(self.n_layers):
            mask = jnp.asarray((jnp.arange(self.d) + i) % 2, dtype=y.dtype)
            y, lj = AffineCoupling(
                features=tuple(self.features), param_dtype=self.param_dtype
            )(y, mask)
            ladj = ladj + lj
            if self.permute and i < self.n_layers - 1:
                # A fixed permutation between couplings. Without it a
                # coordinate only ever mixes with the opposite parity class,
                # and the eta_a split eta[:K] then lands on transformed or
                # pass-through coordinates depending on the parity of K --
                # which puts a period-2 artefact in any scan over K.
                # |det| = 1, so the log-det is unchanged.
                y = y[jnp.asarray(rng.permutation(self.d))]
        return y, ladj


class SquareEta(nn.Module):
    """theta -> (eta, log|det d eta / d theta|). Bijective by construction."""

    d: int
    lo: jnp.ndarray
    hi: jnp.ndarray
    n_layers: int = 8
    features: Sequence[int] = (128, 128)
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, theta):
        y, ladj_box = LogitBox(lo=self.lo, hi=self.hi)(theta)
        z, ladj_flow = CouplingFlow(
            d=self.d, n_layers=self.n_layers, features=tuple(self.features),
            param_dtype=self.param_dtype,
        )(y)
        return z, ladj_box + ladj_flow


def square_loss(eta, ladj, t_hat, k_active: int, d: int):
    """Square-completed negative log density for one pair.

    ``eta`` is (d,), ``t_hat`` is (k_active,). The statistic is padded with
    ``d - k_active`` zeros, so the inactive coordinates are pulled to the
    standard normal and carry the prior along the fibres.
    """
    resid_a = eta[:k_active] - t_hat
    resid_b = eta[k_active:]
    quad = 0.5 * jnp.sum(resid_a ** 2) + 0.5 * jnp.sum(resid_b ** 2)
    return quad - ladj + 0.5 * d * math.log(2.0 * math.pi)
