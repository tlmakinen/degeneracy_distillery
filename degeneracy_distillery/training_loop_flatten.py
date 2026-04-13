#!/usr/bin/env python
"""
This module defines fit_flattening(), a function that fits a flattening network 
to map from θ to a new coordinate system η. The function accepts two additional 
arguments: F_fishnets (the Fisher matrices, typically provided as an ensemble or 
aggregated from a fishnet procedure) and θs (the parameter values).

This is a merged version combining training_loop_flattening2.py and 
training_loop_flatten_inv.py, supporting both MLP and RealNVP architectures.
"""

import os, sys
import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax
import numpy as np
import scipy
import matplotlib.pyplot as plt
from typing import Sequence, Any, Callable, Optional, Literal
from tqdm import tqdm

# Import external modules (assumed to be provided)
# Support both package import and direct script execution
try:
    from .fishnets import *
    from .flatten_net import *
    from .nn_inv import *
    from .io_utils import FlexibleDict, create_results_dict
except ImportError:
    from fishnets import *
    from flatten_net import *
    from nn_inv import *
    from io_utils import FlexibleDict, create_results_dict

from jax import lax

def stable_sin_swish(x):
    """
    Composite activation: sin(swish(x))
    Numerical stability is handled by jax.nn.swish's internal 
    safe-sigmoid logic.
    """
    # jax.nn.swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    swish_x = nn.swish(x)
    
    # sin is generally stable for float inputs, but we use lax.sin 
    # for direct access to stablehlo operations if accuracy is critical.
    return lax.sin(swish_x)


# ---------------------- ROTATION UTILS -----------------------

def kabsch_jax(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    adapted from https://hunterheidenreich.com/posts/kabsch_algorithm/

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = jnp.mean(P, axis=0)
    centroid_Q = jnp.mean(Q, axis=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = jnp.dot(p.T, q)

    # SVD
    U, S, Vt = jnp.linalg.svd(H)

    # Validate right-handed coordinate system
    if jnp.linalg.det(jnp.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = jnp.dot(Vt.T, U.T)

    # RMSD
    rmsd = jnp.sqrt(jnp.sum(jnp.square(jnp.dot(p, R.T) - q)) / P.shape[0])

    return R, t, rmsd

def rotate_coords(y, theta, theta_fid=np.array([1.0,5.0])):
    """compute optimal global rotation with respect to a fiducial 
    theta value.

    Args:
        y (array_like): coordinates to rotate
        theta (array_like): input coordinates
        theta_fid (array_like, optional): fiducial value to align with. Defaults to np.array([1.0,5.0]).

    Returns:
        y_rotated, R (tuple): rotated coordinates and global rotation matrix
    """
    # find theta closest to central value
    argstar = np.argmin(np.sum((theta - theta_fid)** 2, -1))
    theta_star = theta[argstar]
    print("thetastar", theta_star)
    eta_star = y[argstar]
    rotmat,t_opt,_ = kabsch_jax(jnp.array([eta_star]), jnp.array([theta_star]))
    
    y = jnp.dot(y - t_opt, rotmat) + t_opt
    return y, rotmat

# ---------------------- WHITENING UTILITIES -----------------------
def compute_whitening_transform(F_ensemble, ensemble_weights):
    """
    Compute whitening transform from ensemble of Fisher matrices.
    
    The whitening is based on the GLOBAL mean Fisher (averaged over both 
    ensemble members and samples), so W is a single (n_params, n_params) matrix.
    
    Args:
        F_ensemble: Array of shape (n_ensemble, n_samples, n_params, n_params)
        ensemble_weights: Weights for each ensemble member, shape (n_ensemble,)
    
    Returns:
        W: whitening matrix (F_mean^{-1/2}), shape (n_params, n_params)
        W_inv: inverse whitening matrix (F_mean^{1/2}), shape (n_params, n_params)
        F_mean: the global mean Fisher, shape (n_params, n_params)
    """
    # First: weighted average over ensemble members -> (n_samples, n_params, n_params)
    F_ensemble_avg = jnp.average(F_ensemble, axis=0, weights=ensemble_weights)
    
    # Second: average over all samples -> (n_params, n_params)
    # This gives us the GLOBAL mean Fisher matrix
    F_mean = jnp.mean(F_ensemble_avg, axis=0)
    
    print(f"Global mean Fisher shape: {F_mean.shape}")
    print(f"Global mean Fisher:\n{F_mean}")
    
    # Eigendecomposition of the global mean Fisher
    eigvals, eigvecs = jnp.linalg.eigh(F_mean)
    
    print(f"Mean Fisher eigenvalues: {eigvals}")
    
    # Ensure numerical stability
    eigvals = jnp.maximum(eigvals, 1e-10)
    
    # W = F_mean^{-1/2} (whitening)
    W = eigvecs @ jnp.diag(1.0 / jnp.sqrt(eigvals)) @ eigvecs.T
    
    # W_inv = F_mean^{1/2} (inverse whitening)
    W_inv = eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T

    print(f"W_inv transformation: ", W_inv)
    
    return W, W_inv, F_mean

def whiten_fisher(F, W):
    """Apply whitening transform: F_white = W @ F @ W.T"""
    return W @ F @ W.T

def whiten_fisher_batch(F_batch, W):
    """Apply whitening to a batch of Fisher matrices."""
    return jax.vmap(lambda F: W @ F @ W.T)(F_batch)


# ---------------------- ROBUST NORMALIZATION -----------------------
def compute_robust_norm_factor(F_ensemble, method: str = "median_max_eig"):
    """
    Compute a robust normalization factor for Fisher matrices.
    
    This is more stable than using F.max() which can be dominated by outliers.
    
    Args:
        F_ensemble: Array of Fisher matrices, shape (n_ensemble, n_samples, n_params, n_params)
                    or (n_samples, n_params, n_params)
        method: Normalization method:
            - "median_max_eig": Median of maximum eigenvalues (default, most robust)
            - "median_trace": Median of traces / n_params
            - "median_det": Median of det^(1/n) (geometric mean of eigenvalues)
            - "percentile_90": 90th percentile of max eigenvalues
    
    Returns:
        norm_factor: Scalar normalization factor
    """
    # Flatten ensemble dimension if present
    if F_ensemble.ndim == 4:
        # Shape: (n_ensemble, n_samples, n_params, n_params)
        n_params = F_ensemble.shape[-1]
        F_flat = F_ensemble.reshape(-1, n_params, n_params)
    else:
        # Shape: (n_samples, n_params, n_params)
        n_params = F_ensemble.shape[-1]
        F_flat = F_ensemble
    
    if method == "median_max_eig":
        # Get all eigenvalues
        eigvals = jax.vmap(jnp.linalg.eigvalsh)(F_flat)
        # Maximum eigenvalue per sample
        max_eigvals = eigvals.max(axis=-1)
        # Use median (robust to outliers)
        norm_factor = jnp.median(max_eigvals)
        
    elif method == "median_trace":
        # Trace / n_params = average eigenvalue
        traces = jnp.trace(F_flat, axis1=-2, axis2=-1) / n_params
        norm_factor = jnp.median(traces)
        
    elif method == "median_det":
        # det^(1/n) = geometric mean of eigenvalues
        dets = jnp.linalg.det(F_flat)
        # Handle numerical issues with small/negative determinants
        dets = jnp.maximum(dets, 1e-20)
        geo_means = dets ** (1.0 / n_params)
        norm_factor = jnp.median(geo_means)
        
    elif method == "percentile_90":
        eigvals = jax.vmap(jnp.linalg.eigvalsh)(F_flat)
        max_eigvals = eigvals.max(axis=-1)
        norm_factor = jnp.percentile(max_eigvals, 90)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Ensure we don't divide by zero
    norm_factor = jnp.maximum(norm_factor, 1e-10)
    
    return float(norm_factor)


# ---------------------- INPUT AUGMENTATION -----------------------
def _log_augment(x):
    """Augment a single sample with log features computed from *raw* inputs.

    Returns a vector of extra features:
      - log(|x_i| + ε)  for each dimension i
      - log(|x_i + x_j| + ε)  for each unique pair i < j

    These give the network direct access to logarithmic combinations
    (e.g. log(m₁m₂), log(m₁+m₂)) without having to learn them from
    composed nonlinearities.  Must be called on raw (unscaled) inputs so
    that the log values carry physical meaning.
    """
    _eps = 1e-10
    n = x.shape[-1]
    log_x = jnp.log(jnp.abs(x) + _eps)
    parts = [log_x]
    if n > 1:
        idx_i, idx_j = jnp.triu_indices(n, k=1)
        pairwise_sums = x[idx_i] + x[idx_j]
        parts.append(jnp.log(jnp.abs(pairwise_sums) + _eps))
    return jnp.concatenate(parts)


def _poly_augment(x_scaled):
    """Augment a single sample with the product elementary symmetric polynomial.

    Must be called on *already min-max-scaled* inputs (x ∈ [1, 2]), so that
    the polynomial values are O(1) and do not dominate the first Dense layer.

    Returns a 1-element vector:
      - prod(x_i)  — last elementary symmetric polynomial (e.g. proportional
                     to m₁·m₂ after the affine scaling), computed in log-domain
                     to stay numerically stable for any input dimension.

    NOTE: sum(x_i) is intentionally omitted.  Its gradient ∂(sum)/∂θᵢ = 1/Δᵢ
    is a constant scalar multiple of [1, 1, …, 1] for every θ, which is exactly
    collinear with the ∂log(Σxᵢ)/∂θ direction already present from _log_augment.
    Adding it would make the augmented-input Jacobian J_aug near-rank-deficient
    near equal-input configurations (e.g. m₁ ≈ m₂), causing det(J) → 0 and
    det(Q) → ∞, which destabilises training via the nan_to_num safety clamps.

    prod(x_i) has gradient [x₂/Δ₁, x₁/Δ₂] which varies with θ, so it is NOT
    always collinear with the existing features and genuinely adds information
    about the bilinear combination (e.g. m₁·m₂ in the GW case).

    With x ∈ [1, 2]:  prod ∈ [1, 2ⁿ]  — safely O(1).
    """
    _eps = 1e-10
    log_x = jnp.log(jnp.abs(x_scaled) + _eps)
    sign = jnp.prod(jnp.sign(x_scaled + _eps))
    prod_val = sign * jnp.exp(jnp.sum(log_x))
    return jnp.stack([prod_val])


# ---------------------- CUSTOM NETWORK DEFINITIONS -----------------------
class custom_MLP(nn.Module):
    """MLP that outputs in whitened space (no inverse transform applied)."""
    features: Sequence[int]
    max_x: jnp.array
    min_x: jnp.array
    minmax_scale_inputs: bool = True
    augment_log_inputs: bool = False
    act: Callable = stable_sin_swish

    @nn.compact
    def __call__(self, x):
        if self.augment_log_inputs:
            log_feats = _log_augment(x)

        if self.minmax_scale_inputs:
            x = (x - self.min_x) / (self.max_x - self.min_x)
            x += 1.0

        if self.augment_log_inputs:
            poly_feats = _poly_augment(x)
            x = jnp.concatenate([x, log_feats, poly_feats])

        x = nn.Dense(self.features[-1])(x)

        x = self.act(nn.Dense(self.features[0])(x))
        for feat in self.features[1:-1]:
            z = self.act(nn.Dense(feat)(x))
            z = nn.Dense(feat)(z)
            x = self.act(x + z)

        x = nn.Dense(self.features[-1])(x)
        return x


class WhitenedMLP(nn.Module):
    """
    MLP with built-in inverse whitening transform.
    
    The network learns η_raw internally, then applies the inverse whitening
    W_inv = F_mean^{1/2} to get the final output:
    
        η(θ) = W_inv @ η_raw(θ)
    
    The Jacobian becomes:
        J = ∂η/∂θ = W_inv @ J_raw
    
    When computing the loss Q = J^{-T} @ F @ J^{-1} with ORIGINAL Fishers F:
        Q = J_raw^{-T} @ (W @ F @ W) @ J_raw^{-1}
          = J_raw^{-T} @ F_whitened @ J_raw^{-1}
    
    So training on ORIGINAL F with this network is equivalent to training
    on WHITENED F with a raw MLP. No need to pre-whiten the training data!
    
    The W_inv layer handles the whitening implicitly through the Jacobian.
    """
    features: Sequence[int]
    max_x: jnp.array
    min_x: jnp.array
    W_inv: jnp.array  # Inverse whitening matrix F_mean^{1/2}
    minmax_scale_inputs: bool = True
    augment_log_inputs: bool = False
    act: Callable = stable_sin_swish
    apply_inverse_whitening: bool = True  # Can disable for inspection

    @nn.compact
    def __call__(self, x):
        if self.augment_log_inputs:
            log_feats = _log_augment(x)

        if self.minmax_scale_inputs:
            x = (x - self.min_x) / (self.max_x - self.min_x)
            x += 1.0

        if self.augment_log_inputs:
            poly_feats = _poly_augment(x)
            x = jnp.concatenate([x, log_feats, poly_feats])

        x = nn.Dense(self.features[-1])(x)

        x = self.act(nn.Dense(self.features[0])(x))
        for feat in self.features[1:-1]:
            z = self.act(nn.Dense(feat)(x))
            z = nn.Dense(feat)(z)
            x = self.act(x + z)

        x = nn.Dense(self.features[-1])(x)

        if self.apply_inverse_whitening:
            x = self.W_inv @ x

        return x


class ReversePathMLP(nn.Module):
    """
    Maps η → θ̂ with the same residual MLP pattern as custom_MLP, but without θ min–max
    on the input (η lives in coordinate space).
    """

    features: Sequence[int]
    act: Callable = stable_sin_swish

    @nn.compact
    def __call__(self, y):
        x = y
        x = nn.Dense(self.features[-1])(x)
        x = self.act(nn.Dense(self.features[0])(x))
        for feat in self.features[1:-1]:
            z = self.act(nn.Dense(feat)(x))
            z = nn.Dense(feat)(z)
            x = self.act(x + z)
        x = nn.Dense(self.features[-1])(x)
        return x - 1.0 # include inductive bias


class ForwardBackwardMLP(nn.Module):
    """
    Forward map θ→η via ``custom_MLP``; learned inverse η→θ̂ via ``ReversePathMLP``.
    ``__call__`` is the forward map (for Jacobians / Fisher loss).
    """

    features: Sequence[int]
    max_x: jnp.ndarray
    min_x: jnp.ndarray
    minmax_scale_inputs: bool = True
    augment_log_inputs: bool = False
    act: Callable = stable_sin_swish

    def setup(self):
        self.forward_net = custom_MLP(
            features=self.features,
            max_x=self.max_x,
            min_x=self.min_x,
            minmax_scale_inputs=self.minmax_scale_inputs,
            augment_log_inputs=self.augment_log_inputs,
            act=self.act,
        )
        self.reverse_net = ReversePathMLP(features=self.features, act=self.act)

    def __call__(self, x):
        return self.forward_net(x)

    def inverse_path(self, y):
        return ((self.reverse_net(y) - 1.0) / (self.max_x - self.min_x)) + self.min_x

    def init_forward_and_reverse(self, x):
        """Initialize params for both nets (default ``init`` only runs ``__call__`` / forward)."""
        y = self.forward_net(x)
        return self.reverse_net(y)


class WhitenedForwardBackwardMLP(nn.Module):
    """Whitened forward (``WhitenedMLP``) plus learned ``ReversePathMLP`` on η."""

    features: Sequence[int]
    max_x: jnp.ndarray
    min_x: jnp.ndarray
    W_inv: jnp.ndarray
    minmax_scale_inputs: bool = True
    augment_log_inputs: bool = False
    act: Callable = stable_sin_swish
    apply_inverse_whitening: bool = True

    def setup(self):
        self.forward_net = WhitenedMLP(
            features=self.features,
            max_x=self.max_x,
            min_x=self.min_x,
            W_inv=self.W_inv,
            minmax_scale_inputs=self.minmax_scale_inputs,
            augment_log_inputs=self.augment_log_inputs,
            act=self.act,
            apply_inverse_whitening=self.apply_inverse_whitening,
        )
        self.reverse_net = ReversePathMLP(features=self.features, act=self.act)

    def __call__(self, x):
        return self.forward_net(x)

    def inverse_path(self, y):
        # y is η = W_inv @ η_raw (same as forward output); reverse_net is trained on η, not η_raw.
        return ((self.reverse_net(y) - 1.0) / (self.max_x - self.min_x)) + self.min_x

    def init_forward_and_reverse(self, x):
        """Initialize params for both nets (default ``init`` only runs ``__call__`` / forward)."""
        y = self.forward_net(x)
        return self.reverse_net(y)


class RealNVPWrapper(nn.Module):
    """
    Wrapper for RealNVP that applies input scaling and returns only the output
    (discarding log_det for the flattening task).
    """
    num_layers: int
    hidden_dims: int
    input_dim: int
    max_x: jnp.array
    min_x: jnp.array
    minmax_scale_inputs: bool = True
    act: Callable = stable_sin_swish

    def setup(self):
        self.real_nvp = RealNVP(
            num_layers=self.num_layers,
            hidden_dims=self.hidden_dims,
            input_dim=self.input_dim,
            activation=self.act
        )

    def __call__(self, x):
        if self.minmax_scale_inputs:
            x = (x - self.min_x) / (self.max_x - self.min_x)
            x += 1.0

        # Apply RealNVP (returns output and log_det)
        y, log_det = self.real_nvp(x)

        # Return only the output for flattening
        return y

    def inverse(self, y):
        y = self.real_nvp.inverse(y)
        if self.minmax_scale_inputs:
            y -= 1.0
            y = (y * (self.max_x - self.min_x)) + self.min_x
        return y


class WhitenedRealNVP(nn.Module):
    """
    RealNVP with built-in inverse whitening transform.
    
    Similar to WhitenedMLP, this network learns η_raw internally via RealNVP,
    then applies the inverse whitening W_inv = F_mean^{1/2} to get the final output:
    
        η(θ) = W_inv @ η_raw(θ)
    
    The Jacobian becomes:
        J = ∂η/∂θ = W_inv @ J_raw
    
    This implicitly whitens the Fishers through the Jacobian transformation,
    so no need to pre-whiten the training data.
    """
    num_layers: int
    hidden_dims: int
    input_dim: int
    max_x: jnp.array
    min_x: jnp.array
    W_inv: jnp.array  # Inverse whitening matrix F_mean^{1/2}
    minmax_scale_inputs: bool = True
    act: Callable = stable_sin_swish
    apply_inverse_whitening: bool = True  # Can disable for inspection

    def setup(self):
        self.real_nvp = RealNVP(
            num_layers=self.num_layers,
            hidden_dims=self.hidden_dims,
            input_dim=self.input_dim,
            activation=self.act
        )

    def __call__(self, x):
        if self.minmax_scale_inputs:
            x = (x - self.min_x) / (self.max_x - self.min_x)
            x += 1.0

        # Apply RealNVP (returns output and log_det)
        y, log_det = self.real_nvp(x)
        
        # Apply inverse whitening transform (fixed, non-trainable)
        # η_final = F_mean^{1/2} @ η_raw
        if self.apply_inverse_whitening:
            y = self.W_inv @ y
        
        return y
    
    def inverse(self, y):
        """
        Inverse transformation: η -> θ
        Reverses the forward pass operations in reverse order.
        First reverses the whitening, then the RealNVP, then the min-max scaling.
        """
        # Reverse inverse whitening transform (only if it was applied in forward)
        if self.apply_inverse_whitening:
            y = jnp.linalg.inv(self.W_inv) @ y
        
        # Reverse RealNVP transformation
        y = self.real_nvp.inverse(y)

        if self.minmax_scale_inputs:
            y -= 1.0
            y = (y * (self.max_x - self.min_x)) + self.min_x
        return y

# ---------------------- UTILITY FUNCTIONS -----------------------
Array = Any

def minmax(x, xmin, xmax, feature_range):
    minval, maxval = feature_range
    xstd = (x - xmin) / (xmax - xmin)
    return xstd * (maxval - minval) + minval

def minmax_inv(x, xmin, xmax, feature_range):
    minval, maxval = feature_range
    x = x - minval
    x /= (maxval - minval)
    x *= (xmax - xmin)
    return x + xmin

def weighted_std(values, weights, axis=0):
    """Return the weighted standard deviation."""
    average = jnp.average(values, weights=weights, axis=axis)
    variance = jnp.average((values - average)**2, weights=weights, axis=axis)
    return jnp.sqrt(variance)

# ---------------------- MAIN FUNCTION: fit_flattening -----------------------
def fit_flattening(F_network_ensemble, θs,
                   ensemble_weights,
                   hidden_size: int = 256,
                   n_layers: int = 3,
                   batch_size: int = 250,
                   epochs_phase1: int = 1000,
                   epochs_phase2: int = 1000,
                   finetune_epochs: int = 400,
                   min_epochs: int = 1200,
                   patience: int = 100,
                   lr_phase1: float = 2e-6,
                   lr_schedule_initial: float = 7e-5,
                   lr_decay: float = 0.3,
                   lr_finetune: float = 4e-6,
                   l1_alpha: float = 0.0,
                   noise: float = 1e-6,
                   seed: int = 0,
                   output_prefix: str = "flattened_coords_sr",
                   SCALE_THETA: bool = False,
                   do_average: bool = True,
                   Fisher_to_flatten: Literal["average", "best"] = "average",
                   F_avg: Any = None,
                   norm_factor: Any = None,
                   norm_method: str = "median_max_eig",
                   use_whitening: bool = True,
                   minmax_scale_inputs: bool = True,
                   augment_log_inputs: bool = False,
                   nn_inv: bool = False,
                   forward_backward_mlp: bool = False,
                   forward_backward_invertibility_weight: float = 1.0,
                   flattener_activation: Literal["sin_swish", "softplus"] = "sin_swish",
                   loss_type: Literal["log_frob", "frob", "squared_frob"] = "log_frob",
                   do_plot: bool = True,
                   loss_reweight_lambda: float = 100.0,
                   loss_reweight_epsilon: float = 1e-7,
                   loss_log_epsilon: float = 1e-12,
                   loss_log_tau: float = 0.1,
                   q_inv_jitter: float = 1e-8,
                   grad_clip_norm: Optional[float] = None,
                   lr_schedule_phase1: Optional[Any] = None,
                   lr_schedule_phase2: Optional[Any] = None,
                   lr_schedule_finetune: Optional[Any] = None,
                   update_pbar_every: int = 10):
    """
    Fits a flattening network to learn a mapping η = f(θ;w), based on matching 
    the neural-Fisher matrix with the identity. The function accepts F_fishnets and 
    θs (parameter values) as inputs along with various hyperparameters controlling the 
    training procedure.

    The user can optionally use an invertibl neural network (RealNVP) to learn the mapping
    for downstream tasks, although the default residual MLP usually returns flatter coordinates
    for the downstream symbolic regression fitting. 
    
    Args:
        F_network_ensemble: Ensemble of Fisher matrices, shape (n_ensemble, n_samples, n_params, n_params)
        θs: Parameter values, shape (n_samples, n_params)
        ensemble_weights: Weights for ensemble members, shape (n_ensemble,)
        hidden_size: Number of hidden units per layer
        n_layers: Number of hidden layers
        batch_size: Training batch size
        epochs_phase1: Epochs for phase 1 training
        epochs_phase2: Epochs for phase 2 training
        finetune_epochs: Epochs for ensemble fine-tuning
        min_epochs: Minimum epochs before early stopping
        patience: Patience for early stopping
        lr_phase1: Learning rate for phase 1 when lr_schedule_phase1 is None (default).
        lr_schedule_initial: Initial value for phase 2 exponential decay when lr_schedule_phase2 is None.
        lr_decay: Decay rate for the default phase 2 exponential schedule (ignored if lr_schedule_phase2 is set).
        lr_finetune: Learning rate for per-ensemble-member fine-tuning when lr_schedule_finetune is None.
        l1_alpha: Penalty on the forward Jacobian ``J = ∂η/∂θ``: adds
            ``l1_alpha * mean(|J|)`` to the objective (outside ``log``, same as legacy layout).
            Default 0 disables it. Alternatives: Frobenius ``‖J‖_F`` or ``‖J-I‖``, Huber on entries,
            or weight decay on network weights.
        noise: Scale of **additive** Gaussian noise on the **strict lower triangle**
            of each Cholesky factor ``L`` (diagonal of ``L`` unchanged); ``F`` is
            rebuilt as ``L_noisy @ L_noisy.T`` so matrices stay PSD. This avoids
            multiplicative noise on ``L_ii`` that inflates ``E[F]`` diagonals.
        seed: Random seed
        output_prefix: Prefix for output filename
        SCALE_THETA: Whether to scale theta (legacy parameter)
        do_average: Legacy: if False, flatten using the best member (equivalent to
            ``Fisher_to_flatten="best"``). Ignored when ``Fisher_to_flatten="best"``.
        Fisher_to_flatten: How to form the target Fisher per sample from the ensemble:
            ``"average"`` — weighted average using ``ensemble_weights`` (fishnets-style);
            ``"best"`` — Fisher from the member with largest weight (lowest val loss in fishnets).
            Whitening ``W`` uses the same choice (global mean over ensemble vs mean over samples
            of that member only).
        F_avg: Pre-computed Fisher targets (if provided, skips ensemble aggregation)
        norm_factor: Normalization factor for Fishers. If None (default), computed
                     automatically using robust_norm_factor with norm_method.
        norm_method: Method for computing norm_factor if not provided:
            - "median_max_eig": Median of maximum eigenvalues (default, most robust)
            - "median_trace": Median of traces / n_params  
            - "median_det": Median of det^(1/n)
            - "percentile_90": 90th percentile of max eigenvalues
        use_whitening: If True, use a whitened network (WhitenedMLP or WhitenedRealNVP) which 
                       has W_inv = F_mean^{1/2} as a fixed final layer. This implicitly whitens 
                       the Fishers through the Jacobian transformation (no need to pre-whiten 
                       training data). The network effectively learns to flatten 
                       F_whitened = W @ F @ W.
        minmax_scale_inputs: If True (default), map each θ dimension from
            ``[min_x, max_x]`` (from the training θ grid) to ``[0, 1]``, then add 1 so the
            network sees values in ``[1, 2]`` on that box. If False, pass θ through unchanged.
            RealNVP ``inverse`` reverses the shift and scaling only when this is True.
        augment_log_inputs: If True, concatenate logarithmic features to the
            (optionally scaled) inputs before the first dense layer.  Appended
            features are ``log(|θ_i| + ε)`` for each dimension and
            ``log(|θ_i + θ_j| + ε)`` for every unique pair *i < j*, computed from
            the **raw** (unscaled) parameters.  This gives the network direct
            access to log-space structure (e.g. ``log(m₁m₂)``, ``log(m₁+m₂)``)
            without having to learn it from composed activations.  Only supported
            for MLP-based architectures; ignored (with a warning) when
            ``nn_inv=True``.
        nn_inv: If True, use RealNVP (invertible normalizing flow) instead of MLP.
                The RealNVP is initialized with hidden_dims=hidden_size and 
                num_layers=n_layers. Can be combined with use_whitening=True for 
                WhitenedRealNVP. Incompatible with ``forward_backward_mlp``.
        forward_backward_mlp: If True, use a forward ``custom_MLP`` (or ``WhitenedMLP``)
            plus a separate ``ReversePathMLP`` η→θ̂ and add a mean-square cycle penalty
            ``‖θ - inverse(forward(θ))‖²`` (scaled by ``forward_backward_invertibility_weight``)
            to the training loss. Mutually exclusive with ``nn_inv``.
        forward_backward_invertibility_weight: Multiplier on the cycle-consistency term
            when ``forward_backward_mlp`` is True.
        flattener_activation: Nonlinearity for all flattener hidden units: ``sin_swish``
            (default, ``sin(swish(x))``) or ``softplus`` (``flax.linen.nn.softplus``), including
            RealNVP coupling nets when ``nn_inv`` is True.
        loss_type: Form of the per-sample flattening objective:
            ``"log_frob"`` (default) — reweighted ``‖Q−I‖_F + ‖Q⁻¹−I‖_F``, then outer
            ``log(·)``. Legacy behaviour.
            ``"frob"`` — same reweighted Frobenius + inverse term, **without** the outer log.
            Gives stronger gradients near the optimum.
            ``"squared_frob"`` — plain ``‖Q−I‖_F²`` with no reweighting, inverse term, or
            log. Simplest loss; most aggressive gradient signal near the optimum.
        do_plot: Whether to generate coordinate visualization plots
        loss_reweight_lambda: λ in the per-sample reweighting r = λ·loss / (loss + exp(-α·loss));
            larger values change how aggressively large residuals are up-weighted. Default matches
            former hardcoded λ=100.
        loss_reweight_epsilon: ϵ in the same reweighting (α derived from λ, ϵ). Default matches
            former hardcoded ϵ=1e-7.
        loss_log_epsilon: Unused (kept for backward compatibility).
        loss_log_tau: Crossover scale for ``log_frob`` loss. The loss is computed as
            ``τ · log1p(L / τ)``, whose gradient w.r.t. L is ``1 / (1 + L/τ)``.
            When ``L << τ`` this is ~1 (full raw-loss gradient); when ``L >> τ`` it
            decays as ``τ/L`` (log compression). Set ``τ ≈ exp(stall_log_loss)`` where
            ``stall_log_loss`` is the printed loss value at which training plateaus.
        q_inv_jitter: Added to Q as ε·I before jnp.linalg.inv(Q) in the loss, so singular or
            nearly singular Q (e.g. rank-deficient predicted Fisher) does not produce NaN grads.
            Set to 0 to restore the previous strict behavior (may NaN).
        grad_clip_norm: If set, apply global norm clipping to gradients before Adam. Default None:
            no clipping (legacy behavior).
        lr_schedule_phase1: Optional Optax learning rate schedule (or scalar) for phase 1. None means
            constant lr_phase1 (legacy behavior).
        lr_schedule_phase2: Optional schedule for phase 2. None means exponential_decay with
            lr_schedule_initial, lr_decay, and transition_steps derived from epochs_phase2 and batch layout
            (legacy behavior).
        lr_schedule_finetune: Optional schedule for ensemble fine-tuning. None means constant lr_finetune
            (legacy behavior).
        update_pbar_every: Refresh the tqdm bar description (and tqdm ``miniters``) at most every this
            many epochs; default 10 reduces log/Colab traffic. Use 1 to update every epoch.
    
    Returns:
        w: Trained network parameters
        ensemble_ws: List of trained parameters for each ensemble member
        output_dict: FlexibleDict containing training results with flexible naming:
                     - 'theta' (or 'X', 'params'): input parameters
                     - 'eta' (or 'y', 'coords'): learned coordinates
                     - 'Jacobians': Jacobian matrices
                     - 'F_ensemble': Fisher matrices
                     - 'eta_ensemble': Coordinate predictions per ensemble member
                     - Additional metrics and metadata
    """
    # ---------------------- CONSTANTS & SETUP -----------------------
    n_params = θs.shape[-1]

    key = jr.PRNGKey(seed)

    if Fisher_to_flatten not in ("average", "best"):
        raise ValueError(
            f"Fisher_to_flatten must be 'average' or 'best', got {Fisher_to_flatten!r}"
        )
    _fisher_mode: Literal["average", "best"] = (
        "best" if (Fisher_to_flatten == "best" or not do_average) else "average"
    )
    best_idx = int(jnp.argmax(jnp.asarray(ensemble_weights)))

    if F_avg is None:
        if _fisher_mode == "average":
            print("AVERAGING FISHERS (weighted ensemble)")
            F_fishnets = jnp.average(
                F_network_ensemble, axis=0, weights=ensemble_weights
            )
        else:
            print(
                f"USING BEST FISHER ENSEMBLE MEMBER (index {best_idx}, "
                "argmax ensemble_weights)"
            )
            F_fishnets = F_network_ensemble[best_idx]
    else:
        F_fishnets = F_avg

    # ---------------------- ROBUST NORMALIZATION -----------------------
    if norm_factor is None:
        print(f"COMPUTING ROBUST NORM FACTOR (method: {norm_method})")
        norm_factor = compute_robust_norm_factor(F_network_ensemble, method=norm_method)
    
    print(f'norm_factor = {norm_factor:.6g}')
    F_fishnets = F_fishnets / norm_factor

    # ---------------------- WHITENING TRANSFORM -----------------------
    W = None
    W_inv = None
    F_mean = None
    
    if use_whitening:
        print("COMPUTING WHITENING TRANSFORM")
        # Compute whitening from the (normalized) ensemble (or best member only)
        F_ensemble_normalized = F_network_ensemble / norm_factor
        if _fisher_mode == "average":
            W, W_inv, F_mean = compute_whitening_transform(
                F_ensemble_normalized, ensemble_weights
            )
        else:
            W, W_inv, F_mean = compute_whitening_transform(
                F_ensemble_normalized[best_idx : best_idx + 1],
                jnp.ones((1,), dtype=ensemble_weights.dtype),
            )
        
        # F_fishnets should be the weighted average (n_samples, n_params, n_params)
        # NOT the full ensemble. The WhitenedMLP's W_inv layer handles the whitening
        # implicitly through the Jacobian transformation.
        # (F_fishnets was already computed above as the weighted average)
        
        # Verify: W @ F_mean @ W should be ~I (sanity check)
        F_white_global_mean = W @ F_mean @ W
        print("Whitened Fisher global mean (should be ~I):")
        print(F_white_global_mean)
        
        # Check condition number of W (can cause gradient issues if too large)
        W_eigvals = jnp.linalg.eigvalsh(W_inv @ W_inv.T)
        W_cond = jnp.sqrt(W_eigvals.max() / W_eigvals.min())
        print(f"W_inv condition number: {W_cond:.2f}")
        if W_cond > 100:
            print("WARNING: High condition number may cause gradient scaling issues!")

    # Determine training input bounds from θs
    max_x = θs.max(0) + 1e-3
    min_x = θs.min(0) - 1e-3

    if nn_inv and forward_backward_mlp:
        raise ValueError("nn_inv and forward_backward_mlp cannot both be True.")

    if flattener_activation not in ("sin_swish", "softplus"):
        raise ValueError(
            "flattener_activation must be 'sin_swish' or 'softplus', "
            f"got {flattener_activation!r}"
        )
    if loss_type not in ("log_frob", "frob", "squared_frob"):
        raise ValueError(
            "loss_type must be 'log_frob', 'frob', or 'squared_frob', "
            f"got {loss_type!r}"
        )
    _flattener_act = (
        stable_sin_swish if flattener_activation == "sin_swish" else nn.softplus
    )
    print(f"Flattener activation: {flattener_activation}")
    print(f"Loss type: {loss_type}")
    print(f"Min-max input scaling: {minmax_scale_inputs}")
    if augment_log_inputs:
        if nn_inv:
            print("WARNING: augment_log_inputs is not supported with RealNVP (nn_inv); ignoring.")
            augment_log_inputs = False
        else:
            _n_log = n_params + n_params * (n_params - 1) // 2
            print(f"Log input augmentation: ON (+{_n_log} features)")

    _feat = [hidden_size] * n_layers + [n_params]

    # ---------------------- DEFINE THE MODEL -----------------------
    if forward_backward_mlp and use_whitening:
        print("USING WHITENED forward–backward MLP (forward WhitenedMLP + ReversePathMLP)")
        model = WhitenedForwardBackwardMLP(
            features=_feat,
            max_x=max_x,
            min_x=min_x,
            W_inv=W_inv,
            minmax_scale_inputs=minmax_scale_inputs,
            augment_log_inputs=augment_log_inputs,
            act=_flattener_act,
            apply_inverse_whitening=True,
        )
    elif forward_backward_mlp:
        print("USING forward–backward MLP (custom_MLP + ReversePathMLP)")
        model = ForwardBackwardMLP(
            features=_feat,
            max_x=max_x,
            min_x=min_x,
            minmax_scale_inputs=minmax_scale_inputs,
            augment_log_inputs=augment_log_inputs,
            act=_flattener_act,
        )
    elif nn_inv and use_whitening:
        print("USING WHITENED RealNVP (invertible normalizing flow with inverse whitening layer)")
        model = WhitenedRealNVP(
            num_layers=n_layers,
            hidden_dims=hidden_size,
            input_dim=n_params,
            max_x=max_x,
            min_x=min_x,
            W_inv=W_inv,
            minmax_scale_inputs=minmax_scale_inputs,
            act=_flattener_act,
            apply_inverse_whitening=True
        )
    elif nn_inv:
        print("USING RealNVP (invertible normalizing flow)")
        model = RealNVPWrapper(
            num_layers=n_layers,
            hidden_dims=hidden_size,
            input_dim=n_params,
            max_x=max_x,
            min_x=min_x,
            minmax_scale_inputs=minmax_scale_inputs,
            act=_flattener_act
        )
    elif use_whitening:
        print("USING WHITENED MLP (with inverse whitening layer)")
        model = WhitenedMLP(
            features=_feat,
            max_x=max_x,
            min_x=min_x,
            W_inv=W_inv,
            minmax_scale_inputs=minmax_scale_inputs,
            augment_log_inputs=augment_log_inputs,
            act=_flattener_act,
            apply_inverse_whitening=True
        )
    else:
        print("USING CUSTOM MLP (no whitening)")
        model = custom_MLP(
            features=_feat,
            max_x=max_x,
            min_x=min_x,
            minmax_scale_inputs=minmax_scale_inputs,
            augment_log_inputs=augment_log_inputs,
            act=_flattener_act
        )

    # ---------------------- LOSS & HELPER FUNCTIONS -----------------------
    @jax.jit
    def norm(A):
        return jnp.sqrt(jnp.einsum('ij,ij->', A, A))

    _loss_lam = loss_reweight_lambda
    _loss_eps = loss_reweight_epsilon
    _loss_alpha = float(
        -np.log(_loss_eps * (_loss_lam - 1.0) + _loss_eps**2.0 / (1.0 + _loss_eps))
        / _loss_eps
    )
    _log_eps = loss_log_epsilon
    _log_tau = loss_log_tau
    _q_jitter = q_inv_jitter
    _inv_pen_w = forward_backward_invertibility_weight
    _l1_alpha = l1_alpha
    _loss_type = loss_type


    if forward_backward_mlp:

        @jax.jit
        def info_loss(w, theta_batched, F_batched):
            def fn(theta, F):
                mymodel = lambda d: model.apply(w, d)
                eta = mymodel(theta)
                theta_rec = model.apply(w, eta, method="inverse_path")
                inv_pen = jnp.mean((theta - theta_rec) ** 2)

                J_eta = jax.jacrev(mymodel)(theta).squeeze()
                jac_l1 = _l1_alpha * jnp.mean(jnp.abs(J_eta))
                jac_l1 = jnp.nan_to_num(jac_l1, nan=0.0, posinf=0.0, neginf=0.0)
                Jeta_inv = jnp.linalg.pinv(J_eta)

                Q = Jeta_inv.T @ F @ Jeta_inv
                eye = jnp.eye(n_params)

                det_q = jnp.linalg.det(Q)
                det_q = jnp.nan_to_num(det_q, nan=0.0, posinf=0.0, neginf=0.0)

                if _loss_type == "squared_frob":
                    loss = jnp.sum((Q - eye) ** 2)
                    loss = jnp.where(jnp.isfinite(loss), loss, jnp.asarray(1e6, dtype=loss.dtype))
                else:
                    Q_reg = Q + _q_jitter * eye
                    inv_term = jnp.linalg.inv(Q_reg)
                    loss = norm(Q - eye) + norm(inv_term - eye)
                    loss = jnp.where(jnp.isfinite(loss), loss, jnp.asarray(1e6, dtype=loss.dtype))
                    r = _loss_lam * loss / (loss + jnp.exp(-_loss_alpha * loss))
                    loss *= r

                loss = loss + _inv_pen_w * inv_pen

                if _loss_type == "log_frob":
                    loss = _log_tau * jnp.log1p(loss / _log_tau)

                loss = jnp.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

                return loss, det_q, jac_l1

            log_losses, dets, l1_terms = jax.vmap(fn)(theta_batched, F_batched)
            return (jnp.mean(log_losses)) + l1_terms.mean(), jnp.mean(dets)

    else:

        @jax.jit
        def info_loss(w, theta_batched, F_batched):
            def fn(theta, F):
                mymodel = lambda d: model.apply(w, d)

                J_eta = jax.jacrev(mymodel)(theta).squeeze()
                jac_l1 = _l1_alpha * jnp.mean(jnp.abs(J_eta))
                jac_l1 = jnp.nan_to_num(jac_l1, nan=0.0, posinf=0.0, neginf=0.0)
                Jeta_inv = jnp.linalg.pinv(J_eta)

                Q = Jeta_inv.T @ F @ Jeta_inv
                eye = jnp.eye(n_params)

                det_q = jnp.linalg.det(Q)
                det_q = jnp.nan_to_num(det_q, nan=0.0, posinf=0.0, neginf=0.0)

                if _loss_type == "squared_frob":
                    loss = jnp.sum((Q - eye) ** 2)
                    loss = jnp.where(jnp.isfinite(loss), loss, jnp.asarray(1e6, dtype=loss.dtype))
                else:
                    Q_reg = Q + _q_jitter * eye
                    inv_term = jnp.linalg.inv(Q_reg)
                    loss = norm(Q - eye) + norm(inv_term - eye)
                    loss = jnp.where(jnp.isfinite(loss), loss, jnp.asarray(1e6, dtype=loss.dtype))
                    r = _loss_lam * loss / (loss + jnp.exp(-_loss_alpha * loss))
                    loss *= r

                if _loss_type == "log_frob":
                    loss = _log_tau * jnp.log1p(loss / _log_tau)

                loss = jnp.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

                return loss, det_q, jac_l1

            log_losses, dets, l1_terms = jax.vmap(fn)(theta_batched, F_batched)
            return (jnp.mean(log_losses)) + l1_terms.mean(), jnp.mean(dets)

    # ---------------------- PREPARE TRAINING DATA -----------------------
    # Shuffle data before batching to ensure proper train/val split randomization
    key, shuffle_key = jr.split(key)
    n_samples = θs.shape[0]
    shuffle_idx = jr.permutation(shuffle_key, jnp.arange(n_samples))
    θs_shuffled = θs[shuffle_idx]
    F_fishnets_shuffled = F_fishnets[shuffle_idx]
    
    # Expect θs and F_fishnets to be 2D or higher; here we reshape them in batch format.
    theta_true = θs_shuffled.reshape(-1, batch_size, n_params)
    F_fishnets = F_fishnets_shuffled.reshape(-1, batch_size, n_params, n_params)

    # ---------------------- TRAINING LOOP DEFINITION -----------------------
    _pbar_stride = max(1, int(update_pbar_every))
    def training_loop(key, w, theta_true, F_fishnets,
                      val_size: int = 5,
                      lr = 1e-5,
                      batch_size: int = batch_size,
                      patience: int = patience,
                      epochs: int = epochs_phase1,
                      min_epochs: int = min_epochs,
                      opt_type = None):
        best_w = w
        best_loss = jnp.inf
        base_opt = (
            optax.adam(learning_rate=lr)
            if opt_type is None
            else opt_type(learning_rate=lr)
        )
        if grad_clip_norm is not None:
            tx = optax.chain(
                optax.clip_by_global_norm(grad_clip_norm),
                base_opt,
            )
        else:
            tx = base_opt
        opt_state = tx.init(w)
        loss_grad_fn = jax.value_and_grad(info_loss, has_aux=True)

        def body_fun(i, inputs):
            w, loss_val, opt_state, detFeta, key, theta_true, F_fishnets = inputs
            theta_samples = theta_true[i]
            F_samples = F_fishnets[i]
            key, sk = jr.split(key)
            L = jax.scipy.linalg.cholesky(F_samples, lower=True)
            n_p = L.shape[-1]
            mask = jnp.tril(jnp.ones((n_p, n_p), dtype=L.dtype), k=-1)
            Z = jr.normal(sk, shape=L.shape)
            L_noisy = L + noise * Z * mask
            F_samples = jnp.einsum("bij,bkj->bik", L_noisy, L_noisy)

            (loss_val, detFeta), grads = loss_grad_fn(w, theta_samples, F_samples)
            updates, opt_state = tx.update(grads, opt_state)
            w = optax.apply_updates(w, updates)
            return w, loss_val, opt_state, detFeta, key, theta_true, F_fishnets

        num_sims = theta_true.reshape(-1, n_params).shape[0]
        lower = 0
        upper = theta_true.shape[0]

        losses = jnp.zeros(epochs)
        detFetas = jnp.zeros(epochs)
        val_losses = jnp.zeros(epochs)
        val_detFetas = jnp.zeros(epochs)
        loss = 0.
        detFeta = 0.
        best_detFeta = jnp.inf
        counter = 0

        pbar = tqdm(
            range(epochs),
            leave=True,
            position=0,
            miniters=_pbar_stride,
        )
        for j in pbar:
            if (counter > patience) and (j + 1 > min_epochs):
                print("\n patience reached. stopping training.")
                losses = losses[:j]
                detFetas = detFetas[:j]
                val_losses = val_losses[:j]
                val_detFetas = val_detFetas[:j]
                pbar.set_description(
                    "epoch %d loss: %.4f, det F(η): %.4f, val det F(η): %.4f"
                    % (j, loss, detFeta, val_detFeta)
                )
                break
            else:
                key, rng = jr.split(key)
                randidx = jr.permutation(key, jnp.arange(num_sims), independent=True)
                theta_train = theta_true.reshape(-1, n_params)[randidx].reshape(-1, batch_size, n_params)
                F_train = F_fishnets.reshape(-1, n_params, n_params)[randidx].reshape(-1, batch_size, n_params, n_params)
                
                init_vals = (w, loss, opt_state, detFeta, key, theta_train, F_train)
                w, loss, opt_state, detFeta, key, theta_train, F_train = jax.lax.fori_loop(lower, upper, body_fun, init_vals)
                theta_val = theta_true[-val_size:].reshape(-1, n_params)
                F_val = F_fishnets[-val_size:].reshape(-1, n_params, n_params)
                (val_loss, val_detFeta), _ = loss_grad_fn(w, theta_val, F_val)

                losses = losses.at[j].set(loss)
                detFetas = detFetas.at[j].set(detFeta)
                val_losses = val_losses.at[j].set(val_loss)
                val_detFetas = val_detFetas.at[j].set(val_detFeta)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_w = w
                    best_detFeta = val_detFeta
                    counter = 0
                else:
                    counter += 1

            if (j + 1) % _pbar_stride == 0 or j == epochs - 1:
                pbar.set_description(
                    "epoch %d loss: %.4f, det F(η): %.4f, val det F(η): %.4f"
                    % (j, loss, detFeta, val_detFeta)
                )
        return best_w, (losses, val_losses), (detFetas, val_detFetas)

    # ---------------------- TRAINING PHASE 1: INITIAL TRAINING -----------------------
    print("TRAINING FLATTENER NET")
    key, rng = jr.split(key)
    x_init = jnp.ones((n_params,))
    if forward_backward_mlp:
        # Default init only runs __call__ (forward_net); reverse_net must run once
        # so Flax creates Dense kernels under reverse_net (see init_forward_and_reverse).
        w = model.init(key, x_init, method="init_forward_and_reverse")
    else:
        w = model.init(key, x_init)
    lr1 = lr_schedule_phase1 if lr_schedule_phase1 is not None else lr_phase1
    w, all_loss, all_dets = training_loop(key, w, theta_true, F_fishnets,
                                          lr=lr1, opt_type=optax.adam)
    
    # ---------------------- PHASE 2: FINE-TUNING -----------------------
    print("FINE-TUNING FLATTENER NET")
    if lr_schedule_phase2 is not None:
        lr2 = lr_schedule_phase2
    else:
        total_steps = epochs_phase2 * (F_fishnets.shape[0]) + epochs_phase2
        lr2 = optax.schedules.exponential_decay(
            init_value=lr_schedule_initial,
            transition_begin=0,
            transition_steps=total_steps,
            decay_rate=lr_decay,
        )
    w, all_loss, all_dets = training_loop(key, w, theta_true, F_fishnets,
                                          lr=lr2,
                                          opt_type=optax.adam,
                                          epochs=epochs_phase2)
    
    # ---------------------- ENSEMBLE FINE-TUNING -----------------------
    # If F_fishnets represents an ensemble, perform fine-tuning per member.
    # Here we assume F_fishnets is an array of ensemble Fisher matrices.
    F_ensemble = jnp.array(F_network_ensemble) / norm_factor  # Normalized ensemble
    
    # If whitening, also whiten each ensemble member's Fishers
    if use_whitening:
        F_ensemble_for_training = F_ensemble # jnp.array([whiten_fisher_batch(f, W) for f in F_ensemble])
    else:
        F_ensemble_for_training = F_ensemble
    
    theta_true = θs.reshape(-1, batch_size, n_params)
    F_fishnets_ensemble = [f.reshape(-1, batch_size, n_params, n_params) for f in F_ensemble_for_training]

    print("FINE-TUNING EACH ENSEMBLE MEMBER")
    ensemble_ws = []
    init_anew = False
    for k, f in enumerate(F_fishnets_ensemble):
        print("fine-tuning for ensemble member %d" % (k))
        key, rng = jr.split(key)
        if init_anew:
            key, rng = jr.split(key)
            if forward_backward_mlp:
                _w = model.init(key, x_init, method="init_forward_and_reverse")
            else:
                _w = model.init(key, x_init)
        else:
            _w = w
        lr_ft = lr_schedule_finetune if lr_schedule_finetune is not None else lr_finetune
        _w, all_loss, all_dets = training_loop(key, _w, theta_true, f,
                                            lr=lr_ft,
                                            epochs=finetune_epochs,
                                            patience=20,
                                            opt_type=optax.adam)
        ensemble_ws.append(_w)


    # ---------------------- EVALUATION: GET JACOBIANS & ENSEMBLE OUTPUTS -----------------------
    @jax.jit
    def get_jacobian(θ, w=w):
        mymodel = lambda d: model.apply(w, d)
        return jax.jacobian(mymodel)(θ)

    # Gather ensemble outputs for η and corresponding Jacobians.
    η_ensemble = []
    Jbar_ensemble = []
    mymodel = lambda d: model.apply(w, d)
    for k, _w in enumerate(ensemble_ws):
        print("applying model to ensemble member %d" % (k))
        current_model = lambda d: model.apply(_w, d)
        ηs = jax.vmap(current_model)(θs)
        getjac = lambda d: get_jacobian(d, w=_w)
        η_ensemble.append(ηs)
        Jbar_ensemble.append(jnp.concatenate(jnp.array([jax.vmap(getjac)(t) 
                                                        for t in θs.reshape(-1, batch_size, n_params)])))
    
    # Compute Jacobians of the current flattening network.
    ηs = jax.vmap(mymodel)(θs)
    Jbar = jnp.concatenate(jnp.array([jax.vmap(get_jacobian)(t) 
                                       for t in θs.reshape(-1, batch_size, n_params)]))

    allFs = jnp.array(F_ensemble)
    dFs = weighted_std(allFs, jnp.ones(allFs.shape), axis=0)  # Here weights are uniform

    # ---------------------- ERROR PROPAGATION: δJ SOLVER -----------------------
    def get_δJ(F, δF, Jbar):
        """
        Propagate the error on a neural Fisher matrix estimate in θ 
        to the Jacobian for a flattened coordinate system.
        """
        J = np.linalg.inv(Jbar)
        Q = - np.einsum("bik,bkj,blj->bil", J, δF, J)
        X = J @ F
        A = np.einsum("bij,bkj->bik", X, X)
        S = jnp.array([scipy.linalg.solve_sylvester(a=A[i], b=A[i], q=Q[i])
                        for i in range(Q.shape[0])])
        δJ = S @ X
        return np.linalg.inv(J + δJ) - Jbar, δJ

    print("CALCULATING JACOBIAN ERROR")
    δJs, δinvJ = get_δJ(allFs.mean(0), dFs, Jbar)


    # ---------------------- GLOBAL ROTATION CORRECTION -----------------------
    print("ROTATING ENSEMBLE COORDINATES")
    ys = []
    dys = []
    F_ensemble = []
    weights = []
    theta_fid = θs.mean(0)

    for i,y in enumerate(η_ensemble):
        try:
            # y, rotmat = rotate_coords(y, theta=θs, theta_fid=theta_fid)
            ys.append(y)
            dy = Jbar_ensemble[i]
            #dys.append(np.dot(dy, rotmat))
            dys.append(dy)
            weights.append(ensemble_weights[i])
            F_ensemble.append(allFs[i])
        except:
            pass



    # ---------------------- SAVE RESULTS -----------------------
    outname = output_prefix
    if SCALE_THETA:
        outname += "_scaled"
    
    # Build output dictionary using FlexibleDict for flexible naming conventions
    output_dict = create_results_dict(
        theta=np.array(θs),              # Canonical: parameters (accessible as 'theta', 'X', 'params')
        eta=np.array(ηs),                # Canonical: coordinates (accessible as 'eta', 'y', 'coords')
        Jacobians=np.array(Jbar),
        deltaJ=np.array(δJs),
        delta_invJ=np.array(δinvJ),
        meanF=np.array(F_ensemble),
        dFs=np.array(dFs),
        F_ensemble=np.array(allFs),
        norm_factor=norm_factor,
        ensemble_weights=weights,
        eta_ensemble=np.array(ys),
        Jbar_ensemble=np.array(dys),
        use_whitening=use_whitening,
        nn_inv=nn_inv,
        forward_backward_mlp=np.array(forward_backward_mlp),
        forward_backward_invertibility_weight=np.array(
            forward_backward_invertibility_weight
        ),
        fisher_to_flatten=np.array(_fisher_mode),
        best_ensemble_member_index=np.array(best_idx),
        flattener_activation=np.array(flattener_activation),
    )
    
    # Add whitening matrices if used
    if use_whitening:
        output_dict['W'] = np.array(W)  # Whitening matrix F_mean^{-1/2}
        output_dict['W_inv'] = np.array(W_inv)  # Inverse whitening F_mean^{1/2}
        output_dict['F_mean'] = np.array(F_mean)  # Mean Fisher (in normalized space)

    # Save to npz file (converts FlexibleDict to regular dict for numpy)
    np.savez(outname, **dict(output_dict))
    print("Note: Load with io_utils.load_flattening_results(file) for alias support")

    # ---------------------- COORDINATE VISUALISATION -----------------------
    # visualise the first two components vs first two params

    if n_params > 3:
        num_pts = 5

    else:
        num_pts = 30

    xs = jnp.linspace(min_x[0], max_x[0], num_pts)
    ys = jnp.linspace(min_x[1], max_x[1], num_pts)

    # add in dummy last index
    if n_params > 2:
        extra = []
        for j in range(n_params - 2):
            zs = jnp.ones(num_pts) * ((max_x[2+j:3+j] - min_x[2+j:3+j]) / 2.) # middle dummy value
            extra.append(zs)
        
        grds = jnp.meshgrid(xs, ys, *extra)
        X = jnp.stack([g.flatten() for g in grds], axis=-1)

    else:
        xs, ys = jnp.meshgrid(xs, ys)
        X = jnp.stack([xs.flatten(), ys.flatten()], axis=-1)


    
    
    etas = jax.vmap(mymodel)(X)[:, :2]

    if do_plot:
        plt.figure(figsize=(10, 3))
        plt.subplot(121)
        data_plot = etas[:, 0].reshape(xs.shape)
        im = plt.contourf(xs, ys, data_plot, cmap='viridis', levels=20)
        plt.colorbar(im)
        plt.ylabel(r'$\theta_2$')
        plt.xlabel(r'$\theta_1$')
        plt.title(r'$\eta_1$')
        plt.legend(framealpha=0., loc='lower left')

        plt.subplot(122)
        data_plot = etas[:, 1].reshape(xs.shape)
        im = plt.contourf(xs, ys, data_plot, cmap='viridis', levels=20)
        plt.colorbar(im)
        plt.ylabel(r'$\theta_2$')
        plt.xlabel(r'$\theta_1$')
        plt.title(r'$\eta_2$')
        plt.legend(framealpha=0., loc='lower left')
        plt.tight_layout()
        plt.savefig("coordinate_visualisation.png")
        plt.close()

    print("EXPERIMENT COMPLETED & RESULTS SAVED TO:", outname + ".npz")
    return w, ensemble_ws, output_dict

# ---------------------- EXECUTION (for testing) -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit a flattening network to Fisher matrix estimates from fishnets."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="fishnets-log/fishnets_outputs",
        help="Path to fishnets output file (without .npz extension). Default: fishnets-log/fishnets_outputs"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="flattened_coords_sr2",
        help="Output filename prefix. Default: flattened_coords_sr2"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable coordinate visualization plot"
    )
    parser.add_argument(
        "--no-whitening",
        action="store_true",
        help="Disable Fisher whitening (not recommended for large dynamic range)"
    )
    parser.add_argument(
        "--flattener-activation",
        type=str,
        default="sin_swish",
        choices=["sin_swish", "softplus"],
        help="Hidden activation for the flattener (MLP / RealNVP). Default: sin_swish.",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="log_frob",
        choices=["log_frob", "frob", "squared_frob"],
        help=(
            "Flattening loss form. 'log_frob': reweighted Frobenius + inverse with "
            "outer log (legacy). 'frob': same without log. 'squared_frob': plain "
            "||Q-I||_F^2, no reweighting/inverse/log. Default: log_frob."
        ),
    )
    parser.add_argument(
        "--augment-log-inputs",
        action="store_true",
        help="Concatenate log(|θ_i|) and log(|θ_i+θ_j|) features to the network input.",
    )
    parser.add_argument(
        "--nn-inv",
        action="store_true",
        help="Use RealNVP (invertible normalizing flow) instead of MLP"
    )
    parser.add_argument(
        "--forward-backward-mlp",
        action="store_true",
        help="Train paired forward MLP + reverse MLP with cycle-consistency penalty (no RealNVP).",
    )
    parser.add_argument(
        "--forward-backward-invertibility-weight",
        type=float,
        default=1.0,
        help="Weight on ‖θ - reverse(forward(θ))‖² when --forward-backward-mlp is set.",
    )
    parser.add_argument(
        "--no-minmax-input-scaling",
        action="store_true",
        help="Pass θ into the flattener without min–max (+1) preprocessing (see fit_flattening).",
    )
    parser.add_argument(
        "--fisher-to-flatten",
        type=str,
        default="average",
        choices=["average", "best"],
        help="Target Fisher: weighted ensemble average, or single best member (argmax weight).",
    )
    parser.add_argument(
        "--norm-method",
        type=str,
        default="median_max_eig",
        choices=["median_max_eig", "median_trace", "median_det", "percentile_90"],
        help="Method for computing robust norm factor. Default: median_max_eig"
    )
    parser.add_argument(
        "--norm-factor",
        type=float,
        default=None,
        help="Manual normalization factor (overrides --norm-method if provided)"
    )
    args = parser.parse_args()
    if args.nn_inv and args.forward_backward_mlp:
        parser.error("--nn-inv and --forward-backward-mlp are mutually exclusive.")

    # ---------------------- LOAD DATA FROM FILE -----------------------
    fname = args.input
    fname_full = fname + ".npz"
    print(f"Loading fishnets data from: {fname_full}")
    
    data_npz = np.load(fname_full)
    thetas = jnp.array(data_npz["theta"])
    ensemble_weights = data_npz["ensemble_weights"]
    F_network_ensemble = jnp.array(data_npz["F_network_ensemble"])

    print("thetas shape:", thetas.shape)
    print("F_network_ensemble shape:", F_network_ensemble.shape)

    fit_flattening(F_network_ensemble, thetas,
                   ensemble_weights=ensemble_weights,
                   hidden_size=256,
                   n_layers=3,
                   batch_size=250,
                   epochs_phase1=10000,
                   epochs_phase2=250,
                   finetune_epochs=250,
                   min_epochs=1200,
                   patience=100,
                   lr_phase1=2e-6,
                   lr_schedule_initial=7e-5,
                   lr_decay=0.3,
                   lr_finetune=4e-6,
                   norm_factor=args.norm_factor,
                   norm_method=args.norm_method,
                   noise=1e-7,
                   seed=0,
                   output_prefix=args.output,
                   SCALE_THETA=False,
                   use_whitening=not args.no_whitening,
                   minmax_scale_inputs=not args.no_minmax_input_scaling,
                   augment_log_inputs=args.augment_log_inputs,
                   nn_inv=args.nn_inv,
                   forward_backward_mlp=args.forward_backward_mlp,
                   forward_backward_invertibility_weight=(
                       args.forward_backward_invertibility_weight
                   ),
                   flattener_activation=args.flattener_activation,
                   loss_type=args.loss_type,
                   do_plot=not args.no_plot,
                   Fisher_to_flatten=args.fisher_to_flatten)
