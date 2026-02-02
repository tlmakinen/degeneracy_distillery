import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np


import matplotlib.pyplot as plt

from typing import Sequence, Any, Callable
Array = Any

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn



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



class MLP(nn.Module):
  features: Sequence[int]
  act: Callable = nn.softplus

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = self.act(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x

    
    
class custom_MLP(nn.Module):
  features: Sequence[int]
  max_x: jnp.array
  min_x: jnp.array
  act: Callable = nn.softplus


  @nn.compact
  def __call__(self, x):
    
    # first adjust min-max
    x = (x - self.min_x) / (self.max_x - self.min_x)
    x += 1.0

    # small linear layer for coeffs
    x = nn.Dense(self.features[-1])(x)

    x = self.act(nn.Dense(self.features[0])(x))
    
    for feat in self.features[1:-1]:
      # residual connections
      x = self.act(nn.Dense(feat)(x))

    # linear layers to account for rotations
    x = nn.Dense(self.features[-1])(x)
    x = nn.Dense(self.features[-1])(x)
    x = nn.Dense(self.features[-1])(x)
    x = nn.Dense(self.features[-1])(x)
    return x




def kabsch(P: Array, Q: Array) -> Array:
    """
    For calculating how to rotated P onto Q.

    adapted from https://github.com/charnley/rmsd/

    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = jnp.dot(jnp.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = jnp.linalg.svd(C)
    d = (jnp.linalg.det(V) * jnp.linalg.det(W)) < 0.0

    if d:
        S = S.at[-1].set(-S[-1])
        V = V.at[:, -1].set(-V[:, -1])

    # Create Rotation matrix U
    U = jnp.dot(V, W)

    return U