import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

import numpy as np
import math
from typing import Sequence, Callable, Any
#import tensorflow_probability.substrates.jax as tfp

import matplotlib.pyplot as plt


from functools import partial
Array = Any

# custom activation function
@jax.jit
def smooth_leaky(x: Array) -> Array:
  r"""Almost Leaky rectified linear unit activation function.
  Computes the element-wise function:
  .. math::
    \mathrm{almost\_leaky}(x) = \begin{cases}
      x, & x \leq -1\\
      - |x|^3/3, & -1 \leq x < 1\\
      3x & x > 1
    \end{cases}
  Args:
    x : input array
  """
  return jnp.where(x < -1, x, jnp.where((x < 1), ((-(jnp.abs(x)**3) / 3) + x*(x+2) + (1/3)), 3*x))

def safe_for_grad_log(x):
  return jnp.log(jnp.where(x > 0., x, 1.))

@jax.jit
def shifted_softplus(x: Array) -> Array:
  r"""shifted softplus activation function.
  Computes the element-wise function:
  .. math::
    \mathrm{shifted\_softplus}(x) = \ln(0.5 + 0.5\exp(x))
    \end{cases}
  Args:
    x : input array
  """
  return safe_for_grad_log(0.5 + 0.5*jnp.exp(x))


# drop-in replacement for tfp's fill_triangular function to remove substrates dependency
@jax.jit
def fill_lower_tri(v):
    m = v.shape[0]
    dim = int(math.sqrt((0.25 + 2 * m)) - 0.5)
    # we can use jax.ensure_compile_time_eval + jnp.tri to do mask indexing
    # but best practice is use numpy for static variable
    # and jnp.tril_indices is just a wrapper around np.tril_indices
    idx = np.tril_indices(dim)
    return jnp.zeros((dim, dim), dtype=v.dtype).at[idx].set(v)


# def fill_triangular(x):
#     m = x.shape[0] # should be n * (n+1) / 2
#     # solve for n
#     n = int(math.sqrt((0.25 + 2 * m)) - 0.5)
#     idx = (m - (n**2 - m))

#     x_tail = x[idx:]

#     return jnp.concatenate([x_tail, jnp.flip(x, [0])], 0).reshape(n, n)

def fill_diagonal(a, val):
    a = a.at[..., jnp.arange(0, a.shape[0]), jnp.arange(0, a.shape[0])].set(val)

    return a

def construct_fisher_matrix_single(outputs):
    Q = fill_lower_tri(outputs)
    middle = jnp.diag(jnp.tril(Q) - nn.softplus(jnp.tril(Q)))
    padding = jnp.zeros(Q.shape)

    L = Q - fill_diagonal(padding, middle)

    return jnp.einsum('...ij,...jk->...ik', L, jnp.transpose(L, (1, 0)))


def construct_fisher_matrix_multiple(outputs):
    Q = jax.vmap(fill_lower_tri)(outputs)
    # vmap the jnp.diag function for the batch
    _diag = jax.vmap(jnp.diag)

    middle = _diag(jnp.tril(Q) - nn.softplus(jnp.tril(Q)))
    padding = jnp.zeros(Q.shape)

    # vmap the fill_diagonal code
    L = Q - jax.vmap(fill_diagonal)(padding, middle)

    return jnp.einsum('...ij,...jk->...ik', L, jnp.transpose(L, (0, 2, 1)))


class MLP(nn.Module):
  features: Sequence[int]
  act: nn.activation = nn.swish

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = self.act(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x
  

class resMLP(nn.Module):
  features: Sequence[int]
  act: nn.activation = nn.swish

  @nn.compact
  def __call__(self, x):
    x = self.act(nn.Dense(self.features[0])(x))

    # residual connections
    for feat in self.features[1:-1]:
      z = self.act(nn.Dense(feat)(x))
      x += z
    
    x = nn.Dense(self.features[-1])(x)
    return x

# LAYER TO OBTAIN MLE AND FISHER FROM SPECIFIED EMBEDDING NETWORK

class Fishnet_from_embedding(nn.Module):
    n_p: int=2
    hidden: int=50
    act: nn.activation = nn.swish
    act_fisher: nn.activation = nn.leaky_relu
    
    @nn.compact
    def __call__(self, x):
        priorCinv = jnp.eye(self.n_p)
        t = self.act(nn.Dense(self.hidden)(x))
        fisher_cholesky = self.act_fisher(nn.Dense(self.hidden)(x))

        t = nn.Dense(self.n_p)(t)
        fisher_cholesky = nn.Dense((self.n_p * (self.n_p + 1) // 2))(fisher_cholesky)

        F = construct_fisher_matrix_single((fisher_cholesky)) + priorCinv
        #t = jnp.einsum("ij,j->i", jnp.linalg.inv(F), t)

        return t, F
    





