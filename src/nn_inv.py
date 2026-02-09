import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from typing import Callable

class AffineCoupling(nn.Module):
    hidden_dims: int
    mask: jnp.ndarray
    activation: Callable = nn.tanh  # Default activation

    @nn.compact
    def __call__(self, x, reverse=False):
        x1 = x * self.mask
        
        # Internal layers use the flexible activation
        # Output layer for 's' uses tanh to keep scaling from exploding
        s_net = nn.Sequential([
            nn.Dense(self.hidden_dims), self.activation,
            nn.Dense(x.shape[-1]), jnp.tanh 
        ])
        
        t_net = nn.Sequential([
            nn.Dense(self.hidden_dims), self.activation,
            nn.Dense(x.shape[-1])
        ])
        
        s = s_net(x1) * (1 - self.mask)
        t = t_net(x1) * (1 - self.mask)

        if not reverse:
            y = x1 + (x * (1 - self.mask)) * jnp.exp(s) + t
            log_det = jnp.sum(s, axis=-1)
            return y, log_det
        else:
            y = (x - t) * jnp.exp(-s)
            return x1 + y * (1 - self.mask)

class RealNVP(nn.Module):
    num_layers: int
    hidden_dims: int
    input_dim: int
    activation: Callable = nn.tanh

    def setup(self):
        # Propagate the chosen activation to each coupling layer
        self.masks = [jnp.arange(self.input_dim) % 2 == i % 2 for i in range(self.num_layers)]
        self.layers = [
            AffineCoupling(self.hidden_dims, mask, self.activation) 
            for mask in self.masks
        ]

    def __call__(self, x):
        log_det_total = 0
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer(z, reverse=True)
        return z