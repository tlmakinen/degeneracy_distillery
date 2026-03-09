import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from typing import Callable, Literal

ScaleFn = Literal["exp", "softplus", "softmax"]


def _scale_and_log_scale(
    s_raw: jnp.ndarray, scale_fn: ScaleFn, mask: jnp.ndarray
):
    """Given raw scale logits `s_raw`, return (scale, log_scale).

    Masked dimensions get scale=1 and log_scale=0 so they don't affect
    the transformation or log-determinant.
    """
    unmask = (1 - mask).astype(bool)

    if scale_fn == "exp":
        scale_unmasked = jnp.exp(s_raw)
        log_scale_unmasked = jnp.log(scale_unmasked)
    elif scale_fn == "softplus":
        scale_unmasked = jax.nn.softplus(s_raw)
        log_scale_unmasked = jnp.log(scale_unmasked)
    else:  # "softmax"
        # Softmax over unmasked dimensions only, then convert to scale/log_scale.
        very_neg = jnp.array(-1e9, dtype=s_raw.dtype)
        s_for_softmax = jnp.where(unmask, s_raw, very_neg)
        log_scale_unmasked = jax.nn.log_softmax(s_for_softmax, axis=-1)
        scale_unmasked = jnp.exp(log_scale_unmasked)

    # For masked dimensions, force scale=1 and log_scale=0.
    scale = jnp.where(unmask, scale_unmasked, 1.0)
    log_scale = jnp.where(unmask, log_scale_unmasked, 0.0)
    return scale, log_scale


class AffineCoupling(nn.Module):
    hidden_dims: int
    mask: jnp.ndarray
    activation: Callable = nn.tanh  # Default activation
    scale_fn: ScaleFn = "exp"  # "exp", "softplus", or "softmax"

    @nn.compact
    def __call__(self, x, reverse: bool = False):
        x1 = x * self.mask

        # Internal layers use the flexible activation.
        # Output layer for 's' uses tanh to keep logits bounded.
        s_net = nn.Sequential(
            [
                nn.Dense(self.hidden_dims),
                self.activation,
                nn.Dense(x.shape[-1]),
                jnp.tanh,
            ]
        )

        t_net = nn.Sequential(
            [
                nn.Dense(self.hidden_dims),
                self.activation,
                nn.Dense(x.shape[-1]),
            ]
        )

        s_raw = s_net(x1)
        t = t_net(x1) * (1 - self.mask)

        scale, log_scale = _scale_and_log_scale(s_raw, self.scale_fn, self.mask)

        if not reverse:
            y = x1 + (x * (1 - self.mask)) * scale + t
            # Only unmasked dimensions contribute to the log-det.
            log_det = jnp.sum(log_scale * (1 - self.mask), axis=-1)
            return y, log_det
        else:
            inv_scale = 1.0 / scale
            y = (x - t) * inv_scale
            return x1 + y * (1 - self.mask)


class RealNVP(nn.Module):
    num_layers: int
    hidden_dims: int
    input_dim: int
    activation: Callable = nn.tanh
    scale_fn: ScaleFn = "exp"

    def setup(self):
        # Propagate the chosen activation and scale_fn to each coupling layer.
        self.masks = [
            jnp.arange(self.input_dim) % 2 == i % 2 for i in range(self.num_layers)
        ]
        self.layers = [
            AffineCoupling(self.hidden_dims, mask, self.activation, self.scale_fn)
            for mask in self.masks
        ]

    def __call__(self, x):
        log_det_total = 0.0
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer(z, reverse=True)
        return z