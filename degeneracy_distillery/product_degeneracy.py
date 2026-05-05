"""JAX toy product-degeneracy simulators.

Two minimal simulators that exhibit an *exact* one-dimensional degeneracy of
the form

    eta_0 = theta_1 * theta_2.

They are intended as pedagogical examples for demonstrating dimensionality
reduction / coordinate flattening, sitting next to the more involved
Rosenbrock / SIR / GW examples in the rest of ``degeneracy_distillery``.

Both simulators share the same parameter prior
``theta = (theta_1, theta_2)`` on a positive box.  They differ only in the
form of the observation:

* :func:`simulator_scalar` -- the canonical "textbook" example.  The
  observation is a single noisy scalar ``y = theta_1 * theta_2 + noise``.
  The likelihood ``p(y | theta)`` is constant along every hyperbola
  ``theta_1 * theta_2 = const``, so the inference problem is exactly
  one-dimensional.

* :func:`simulator_heater` -- a slightly less trivial analog motivated by
  industrial process control.  Imagine identifying the steady heating
  power of a resistive heater attached to a first-order thermal plant,
  with applied voltage ``V`` and current ``I``.  The plant temperature
  follows a known step response and we observe a noisy time series

      y(t) = (V * I) * (1 - exp(-t / tau)) + noise(t).

  Even though the observation is high-dimensional in time, it depends on
  ``(V, I)`` only through the scalar dissipated power ``P = V * I``: a
  one-dimensional sufficient statistic for an exact rank-1 degeneracy.

The natural "good" coordinates are
``eta = (eta_0, eta_1) = (theta_1 * theta_2, log(theta_1 / theta_2))``.
The data depends only on ``eta_0``; ``eta_1`` is an unidentifiable nuisance
direction that the data cannot constrain.  These are exposed as
:func:`theta_to_eta` and :func:`eta_to_theta` so they can be plugged into
the same flattening / NPE pipelines used for the other examples.

All randomness is controlled via JAX PRNG keys, so a given ``key`` always
produces the same dataset.

Example
-------
>>> import jax
>>> import jax.random as jr
>>> from degeneracy_distillery.product_degeneracy import (
...     ToyConfig, make_dataset, simulator_heater, theta_to_eta,
... )
>>> cfg = ToyConfig()
>>> key = jr.PRNGKey(0)
>>> theta, y = make_dataset(2000, cfg, key, simulator=simulator_heater)
>>> eta = theta_to_eta(theta)
>>> y.shape, eta.shape
((2000, 50), (2000, 2))
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr


Array = jnp.ndarray


@dataclass
class ToyConfig:
    """Configuration for the product-degeneracy simulators.

    The prior box is restricted to positive ``theta`` so that the product is
    sign-stable and the log-ratio reparameterization is finite.
    """

    theta_min: float = 0.5
    theta_max: float = 3.0
    tau: float = 1.0
    t_max: float = 4.0
    n_t: int = 50
    sigma_scalar: float = 0.5
    sigma_heater: float = 0.2

    @property
    def t_grid(self) -> Array:
        return jnp.linspace(0.0, self.t_max, self.n_t, dtype=jnp.float32)

    @property
    def thermal_kernel(self) -> Array:
        """Known step-response kernel ``1 - exp(-t / tau)`` of shape ``(n_t,)``."""
        return (1.0 - jnp.exp(-self.t_grid / self.tau)).astype(jnp.float32)

    @property
    def prior_low(self) -> Array:
        return jnp.array([self.theta_min, self.theta_min], dtype=jnp.float32)

    @property
    def prior_high(self) -> Array:
        return jnp.array([self.theta_max, self.theta_max], dtype=jnp.float32)


def sample_prior(key: jax.Array, n: int, cfg: ToyConfig) -> Array:
    """Draw ``n`` samples of ``theta`` uniformly from the positive prior box."""
    return jr.uniform(
        key, shape=(n, 2), minval=cfg.theta_min, maxval=cfg.theta_max,
        dtype=jnp.float32,
    )


def theta_to_eta(theta: Array) -> Array:
    """Map ``(theta_1, theta_2) -> (theta_1 * theta_2, log(theta_1 / theta_2))``."""
    theta = jnp.asarray(theta, dtype=jnp.float32)
    eta_0 = theta[..., 0] * theta[..., 1]
    eta_1 = jnp.log(theta[..., 0] / theta[..., 1])
    return jnp.stack([eta_0, eta_1], axis=-1)


def eta_to_theta(eta: Array) -> Array:
    """Inverse of :func:`theta_to_eta`."""
    eta = jnp.asarray(eta, dtype=jnp.float32)
    p = eta[..., 0]
    r = eta[..., 1]
    root_p = jnp.sqrt(jnp.clip(p, 0.0, None))
    theta_1 = root_p * jnp.exp(r / 2.0)
    theta_2 = root_p * jnp.exp(-r / 2.0)
    return jnp.stack([theta_1, theta_2], axis=-1)


def simulator_scalar(key: jax.Array, theta: Array, cfg: ToyConfig) -> Array:
    """Trivial product simulator: ``y = theta_1 * theta_2 + N(0, sigma_scalar^2)``.

    Designed to be used with ``jax.vmap`` over leading dimensions of
    ``key`` and ``theta``.  Output shape: ``(1,)`` per sample.
    """
    theta = jnp.asarray(theta, dtype=jnp.float32)
    eta = theta[..., 0] * theta[..., 1]
    noise = jr.normal(key, shape=eta.shape, dtype=jnp.float32) * cfg.sigma_scalar
    return (eta + noise)[..., None]


def simulator_heater(key: jax.Array, theta: Array, cfg: ToyConfig) -> Array:
    """First-order heater step response forced by power ``P = V * I``.

    Returns observations of shape ``(n_t,)`` per sample.  Across the time
    axis the only ``theta``-dependence is through the scalar product, so the
    data is a rank-1 function of ``(theta_1, theta_2)`` -- a clean instance
    of high-dimensional data with a one-dimensional sufficient statistic.

    Designed to be used with ``jax.vmap`` over leading dimensions.
    """
    theta = jnp.asarray(theta, dtype=jnp.float32)
    power = theta[..., 0] * theta[..., 1]
    kernel = cfg.thermal_kernel
    mean = power[..., None] * kernel
    noise = jr.normal(key, shape=mean.shape, dtype=jnp.float32) * cfg.sigma_heater
    return mean + noise


def make_dataset(
    n: int,
    cfg: ToyConfig,
    key: jax.Array,
    *,
    simulator=simulator_heater,
) -> tuple[Array, Array]:
    """Sample ``n`` ``(theta, y)`` pairs from the chosen simulator.

    Splits ``key`` into independent prior- and noise- streams; both are
    vmapped, so changing ``n`` only re-uses the prior keys you already had.
    """
    prior_key, noise_key = jr.split(key)
    theta = sample_prior(prior_key, n, cfg)
    keys = jr.split(noise_key, n)
    data = jax.vmap(simulator, in_axes=(0, 0, None))(keys, theta, cfg)
    return theta, data


def _demo() -> None:  # pragma: no cover - illustrative only
    """Visualize the rank-1 degeneracy in ``theta`` and the flattened ``eta``."""
    import matplotlib.pyplot as plt
    import numpy as np

    cfg = ToyConfig()
    key = jr.PRNGKey(0)

    theta, y = make_dataset(1, cfg, key, simulator=simulator_heater)
    theta_true = np.asarray(theta[0])
    y_obs = np.asarray(y[0])
    p_true = float(theta_true[0] * theta_true[1])

    grid = np.linspace(cfg.theta_min, cfg.theta_max, 240, dtype=np.float32)
    t1, t2 = np.meshgrid(grid, grid, indexing="ij")
    kernel = np.asarray(cfg.thermal_kernel)
    mean_theta = (t1 * t2)[..., None] * kernel
    chi2_theta = ((mean_theta - y_obs) ** 2).sum(axis=-1) / cfg.sigma_heater ** 2
    log_post_theta = -0.5 * chi2_theta

    eta_p = np.linspace(cfg.theta_min ** 2, cfg.theta_max ** 2, 240, dtype=np.float32)
    eta_r = np.linspace(
        float(np.log(cfg.theta_min / cfg.theta_max)),
        float(np.log(cfg.theta_max / cfg.theta_min)),
        240,
        dtype=np.float32,
    )
    e0, e1 = np.meshgrid(eta_p, eta_r, indexing="ij")
    mean_eta = e0[..., None] * kernel
    chi2_eta = ((mean_eta - y_obs) ** 2).sum(axis=-1) / cfg.sigma_heater ** 2
    log_post_eta = -0.5 * chi2_eta

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)

    cs0 = axes[0].contourf(
        t1, t2, log_post_theta - log_post_theta.max(),
        levels=np.linspace(-25.0, 0.0, 26), cmap="viridis",
    )
    contour_levels = np.linspace(cfg.theta_min ** 2, cfg.theta_max ** 2, 9)
    axes[0].contour(
        t1, t2, t1 * t2, levels=contour_levels, colors="white", alpha=0.35, linewidths=0.6
    )
    axes[0].plot(theta_true[0], theta_true[1], "rx", ms=11, mew=2, label=r"true $\theta$")
    axes[0].set_xlabel(r"$\theta_1\;(\equiv V)$")
    axes[0].set_ylabel(r"$\theta_2\;(\equiv I)$")
    axes[0].set_title(r"log-posterior in $\theta$ (degenerate)")
    axes[0].legend(loc="lower right")
    fig.colorbar(cs0, ax=axes[0])

    cs1 = axes[1].contourf(
        e0, e1, log_post_eta - log_post_eta.max(),
        levels=np.linspace(-25.0, 0.0, 26), cmap="viridis",
    )
    axes[1].axvline(p_true, color="r", ls=":", label=r"true $\eta_0$")
    axes[1].set_xlabel(r"$\eta_0 = \theta_1\,\theta_2$  (power)")
    axes[1].set_ylabel(r"$\eta_1 = \log(\theta_1 / \theta_2)$")
    axes[1].set_title(r"log-posterior in $\eta$ (rank-1)")
    axes[1].legend(loc="upper right")
    fig.colorbar(cs1, ax=axes[1])

    plt.show()


if __name__ == "__main__":
    _demo()
