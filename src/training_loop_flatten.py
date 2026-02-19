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
from typing import Sequence, Any, Callable
from tqdm import tqdm

# Import external modules (assumed to be provided)
from fishnets import *
from flatten_net import *
from nn_inv import *

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


# ---------------------- CUSTOM NETWORK DEFINITIONS -----------------------
class custom_MLP(nn.Module):
    """MLP that outputs in whitened space (no inverse transform applied)."""
    features: Sequence[int]
    max_x: jnp.array
    min_x: jnp.array
    act: Callable = stable_sin_swish

    @nn.compact
    def __call__(self, x):
        # Adjust input by min-max scaling.
        x = (x - self.min_x) / (self.max_x - self.min_x)
        x += 1.0

        # Small dense layers for coefficients.
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
    act: Callable = stable_sin_swish
    apply_inverse_whitening: bool = True  # Can disable for inspection

    @nn.compact
    def __call__(self, x):
        # Adjust input by min-max scaling.
        x = (x - self.min_x) / (self.max_x - self.min_x)
        x += 1.0

        # Small dense layers for coefficients.
        x = nn.Dense(self.features[-1])(x)
        
        x = self.act(nn.Dense(self.features[0])(x))
        for feat in self.features[1:-1]:
            z = self.act(nn.Dense(feat)(x))
            z = nn.Dense(feat)(z)
            x = self.act(x + z)

        x = nn.Dense(self.features[-1])(x)
        
        # Apply inverse whitening transform (fixed, non-trainable)
        # η_final = F_mean^{1/2} @ η_raw
        if self.apply_inverse_whitening:
            x = self.W_inv @ x
        
        return x


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
    act: Callable = stable_sin_swish

    def setup(self):
        self.real_nvp = RealNVP(
            num_layers=self.num_layers,
            hidden_dims=self.hidden_dims,
            input_dim=self.input_dim,
            activation=self.act
        )

    def __call__(self, x):
        # Adjust input by min-max scaling
        x = (x - self.min_x) / (self.max_x - self.min_x)
        x += 1.0
        
        # Apply RealNVP (returns output and log_det)
        y, log_det = self.real_nvp(x)
        
        # Return only the output for flattening
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
        # Adjust input by min-max scaling
        x = (x - self.min_x) / (self.max_x - self.min_x)
        x += 1.0
        
        # Apply RealNVP (returns output and log_det)
        y, log_det = self.real_nvp(x)
        
        # Apply inverse whitening transform (fixed, non-trainable)
        # η_final = F_mean^{1/2} @ η_raw
        if self.apply_inverse_whitening:
            y = self.W_inv @ y
        
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
                   F_avg: Any = None,
                   norm_factor: Any = None,
                   norm_method: str = "median_max_eig",
                   use_whitening: bool = True,
                   nn_inv: bool = False,
                   do_plot: bool = True):
    """
    Fits a flattening network to learn a mapping η = f(θ;w), based on matching 
    the neural-Fisher matrix with the identity. The function accepts F_fishnets and 
    θs (theta values) as inputs along with various hyperparameters controlling the 
    training procedure.
    
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
        lr_phase1: Learning rate for phase 1
        lr_schedule_initial: Initial learning rate for phase 2 schedule
        lr_decay: Decay rate for learning rate schedule
        lr_finetune: Learning rate for fine-tuning
        l1_alpha: L1 regularization coefficient (currently disabled)
        noise: Noise level added to Fisher matrices during training
        seed: Random seed
        output_prefix: Prefix for output filename
        SCALE_THETA: Whether to scale theta (legacy parameter)
        do_average: Whether to average ensemble (legacy parameter)
        F_avg: Pre-computed averaged Fisher (if provided, skips averaging)
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
        nn_inv: If True, use RealNVP (invertible normalizing flow) instead of MLP.
                The RealNVP is initialized with hidden_dims=hidden_size and 
                num_layers=n_layers. Can be combined with use_whitening=True for 
                WhitenedRealNVP.
        do_plot: Whether to generate coordinate visualization plots
    
    Returns:
        w: Trained network parameters
        ensemble_ws: List of trained parameters for each ensemble member
    """
    # ---------------------- CONSTANTS & SETUP -----------------------
    n_params = θs.shape[-1]

    key = jr.PRNGKey(seed)

    if F_avg is None:
      print("AVERAGING FISHERS")
      # Compute weighted average of network Fisher matrices:
      F_fishnets = jnp.average(F_network_ensemble, axis=0, weights=ensemble_weights)
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
        # Compute whitening from the (normalized) ensemble
        F_ensemble_normalized = F_network_ensemble / norm_factor
        W, W_inv, F_mean = compute_whitening_transform(
            F_ensemble_normalized, ensemble_weights
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

    # ---------------------- DEFINE THE MODEL -----------------------
    if nn_inv and use_whitening:
        print("USING WHITENED RealNVP (invertible normalizing flow with inverse whitening layer)")
        model = WhitenedRealNVP(
            num_layers=n_layers,
            hidden_dims=hidden_size,
            input_dim=n_params,
            max_x=max_x,
            min_x=min_x,
            W_inv=W_inv,
            act=stable_sin_swish,
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
            act=stable_sin_swish
        )
    elif use_whitening:
        print("USING WHITENED MLP (with inverse whitening layer)")
        model = WhitenedMLP(
            features=[hidden_size]*n_layers + [n_params],
            max_x=max_x,
            min_x=min_x,
            W_inv=W_inv,
            act=stable_sin_swish,
            apply_inverse_whitening=True
        )
    else:
        print("USING CUSTOM MLP (no whitening)")
        model = custom_MLP(
            features=[hidden_size]*n_layers + [n_params],
            max_x=max_x,
            min_x=min_x,
            act=stable_sin_swish
        )

    # ---------------------- LOSS & HELPER FUNCTIONS -----------------------
    @jax.jit
    def norm(A):
        return jnp.sqrt(jnp.einsum('ij,ij->', A, A))

    def get_α(λ=100., ϵ=1e-7):
        return -jnp.log(ϵ*(λ - 1.) + ϵ**2. / (1. + ϵ)) / ϵ

    @jax.jit
    def l1_reg(x, alpha=l1_alpha):
        return alpha * jnp.abs(x).mean()

    theta_star = jnp.array([1.0, 1.0])

    @jax.jit
    def info_loss(w, theta_batched, F_batched):
        λ = 100.
        ϵ = 1e-7
        α = get_α(λ, ϵ)

        def fn(theta, F):
            mymodel = lambda d: model.apply(w, d)


            J_eta = jax.jacrev(mymodel)(theta).squeeze()
            Jeta_inv = jnp.linalg.pinv(J_eta)
            Q = Jeta_inv.T @ F @ Jeta_inv # <--- FLIP TRANSPOSE ??

            loss = norm(Q - jnp.eye(n_params)) + norm(jnp.linalg.inv(Q) - jnp.eye(n_params))
            r = λ * loss / (loss + jnp.exp(-α * loss))
            loss *= r
            
            l1_loss = 0.0 # l1_reg(J_eta)

            return jnp.log(loss), jnp.linalg.det(Q), l1_loss

        loss, Q, l1_loss = jax.vmap(fn)(theta_batched, F_batched)
        return (jnp.mean(loss)) + l1_loss.mean(), jnp.mean(Q)

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
        if opt_type is None:
            tx = optax.adam(learning_rate=lr)
        else:
            tx = opt_type(learning_rate=lr)
        opt_state = tx.init(w)
        loss_grad_fn = jax.value_and_grad(info_loss, has_aux=True)

        def body_fun(i, inputs):
            w, loss_val, opt_state, detFeta, key, theta_true, F_fishnets = inputs
            theta_samples = theta_true[i]
            F_samples = F_fishnets[i]
            # Add noise to Fisher matrices
            F_samples += jr.normal(key, shape=F_samples.shape)*noise*F_samples

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

        pbar = tqdm(range(epochs), leave=True, position=0)
        for j in pbar:
            if (counter > patience) and (j + 1 > min_epochs):
                print("\n patience reached. stopping training.")
                losses = losses[:j]
                detFetas = detFetas[:j]
                val_losses = val_losses[:j]
                val_detFetas = val_detFetas[:j]
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

            pbar.set_description('epoch %d loss: %.4f, detFeta: %.4f, val_detFeta: %.4f'
                                  % (j, loss, detFeta, val_detFeta))
        return best_w, (losses, val_losses), (detFetas, val_detFetas)

    # ---------------------- TRAINING PHASE 1: INITIAL TRAINING -----------------------
    print("TRAINING FLATTENER NET")
    key, rng = jr.split(key)
    w = model.init(key, jnp.ones((n_params,)))
    w, all_loss, all_dets = training_loop(key, w, theta_true, F_fishnets, 
                                          lr=lr_phase1, opt_type=optax.adam)
    
    # ---------------------- PHASE 2: FINE-TUNING -----------------------
    print("FINE-TUNING FLATTENER NET")
    total_steps = epochs_phase2*(F_fishnets.shape[0]) + epochs_phase2
    lr_schedule = optax.schedules.exponential_decay(init_value=lr_schedule_initial,
                                                    transition_begin=0,
                                                    transition_steps=total_steps,
                                                    decay_rate=lr_decay)
    w, all_loss, all_dets = training_loop(key, w, theta_true, F_fishnets,
                                          lr=lr_schedule,
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
            _w = model.init(key, jnp.ones((n_params,)))
        else:
            _w = w
        _w, all_loss, all_dets = training_loop(key, _w, theta_true, f, 
                                            lr=lr_finetune,
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
    
    # Build output dictionary
    output_dict = dict(
        theta=np.array(θs),
        eta=np.array(ηs),
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
        nn_inv=nn_inv
    )
    
    # Add whitening matrices if used
    if use_whitening:
        output_dict['W'] = np.array(W)  # Whitening matrix F_mean^{-1/2}
        output_dict['W_inv'] = np.array(W_inv)  # Inverse whitening F_mean^{1/2}
        output_dict['F_mean'] = np.array(F_mean)  # Mean Fisher (in normalized space)
    
    np.savez(outname, **output_dict)

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
    return w, ensemble_ws

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
        "--nn-inv",
        action="store_true",
        help="Use RealNVP (invertible normalizing flow) instead of MLP"
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
                   nn_inv=args.nn_inv,
                   do_plot=not args.no_plot)
