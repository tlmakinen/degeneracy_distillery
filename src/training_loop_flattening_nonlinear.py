#!/usr/bin/env python
"""
This module defines fit_flattening(), a function that fits a flattening network 
to map from θ to a new coordinate system η. The function accepts two additional 
arguments: F_fishnets (the Fisher matrices, typically provided as an ensemble or 
aggregated from a fishnet procedure) and θs (the parameter values).
"""

import os, sys
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
# from sr_functions import *


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

# ---------------------- CUSTOM NETWORK DEFINITIONS -----------------------
class custom_MLP(nn.Module):
    features: Sequence[int]
    max_x: jnp.array
    min_x: jnp.array
    act: Callable = nn.softplus

    @nn.compact
    def __call__(self, x):
        # Adjust input by min-max scaling.
        x = (x - self.min_x) / (self.max_x - self.min_x)
        x += 1.0

        # Small dense layers for coefficients.
        x = nn.Dense(self.features[-1])(x)
        x = self.act(nn.Dense(self.features[0])(x))
        for feat in self.features[1:-1]:
            x = self.act(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x



class custom_MLP(nn.Module):
    features: Sequence[int]
    max_x: jnp.array
    min_x: jnp.array
    act: Callable = nn.softplus
    num_nonlinear: int = 2

    @nn.compact
    def __call__(self, x, return_nonlinear=False):
        # Adjust input by min-max scaling.
        x = (x - self.min_x) / (self.max_x - self.min_x)
        x += 1.0

        x1 = x

        # Small dense layers for coefficients.
        x = nn.Dense(self.features[-1])(x)
        x = self.act(nn.Dense(self.features[0])(x))
        for feat in self.features[1:-1]:
            x = self.act(nn.Dense(feat)(x))
        #
        #x = self.act(x)
        # one nonlinear output
        x = nn.Dense(self.num_nonlinear)(x)

        xnonlinear = x

        x = jnp.concatenate([x, x1], -1)

        n_params = self.features[-1]

        #x = nn.Dense(n_params - self.num_nonlinear)(x)
        x = nn.Dense(n_params)(x)
        x = nn.Dense(n_params)(x)
        x = nn.Dense(n_params)(x)

        if return_nonlinear:
          return x, xnonlinear

        else:
          return x # output is number of params


class MLP(nn.Module):
  features: Sequence[int]
  act: nn.activation = nn.swish

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = self.act(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x

class bottleneck_MLP(nn.Module):
    features: Sequence[int]
    n_params: int 
    max_x: jnp.array
    min_x: jnp.array
    act: Callable = nn.softplus
    n_nonlinear: int = 2

    def setup(self):
      self.mlp = MLP((self.n_params,) + self.features + (self.n_nonlinear,), 
                   act = self.act)
      
      self.dense1 = nn.Dense(self.n_params)
      self.dense2 = nn.Dense(self.n_params)


    def __call__(self, x, return_nonlinear=False):
      x = (x - self.min_x) / (self.max_x - self.min_x)
      x += 1.0
      x1 = x # normalised params

      xnonlinear = self.mlp(x)

      x = jnp.concatenate([xnonlinear, x1], -1)

      x = self.dense1(x)
      x = self.dense2(x)

      if return_nonlinear:
        return x, xnonlinear

      else:
        return x


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
                   n_nonlinear: int = 2,
                   hidden_size: int = 256,
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
                   norm_factor: float = 1.0,
                   do_plot: bool = True):
    """
    Fits a flattening network to learn a mapping η = f(θ;w), based on matching 
    the neural-Fisher matrix with the identity. The function accepts F_fishnets and 
    θs (theta values) as inputs along with various hyperparameters controlling the 
    training procedure.
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

    # Normalize F_fishnets using its maximum value (as in the original code)
    # norm_factor = 1.0 # F_fishnets.max() / 100.
    print('norm factor', norm_factor)
    F_fishnets = F_fishnets / norm_factor

    # Determine training input bounds from θs
    max_x = θs.max(0) + 1e-3
    min_x = θs.min(0) - 1e-3

    # ---------------------- DEFINE THE MODEL -----------------------
    print("initialising model with %d nonlinear components"%(n_nonlinear))
    model = bottleneck_MLP(features=[hidden_size, 
                                  hidden_size, 
                                  hidden_size,
                                  hidden_size, 
                                  ],
                       n_nonlinear = n_nonlinear,
                       n_params = n_params,
                       max_x = max_x,
                       min_x = min_x,
                       act = nn.softplus)

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
            Q = Jeta_inv @ F @ Jeta_inv.T

            loss = norm(Q - jnp.eye(n_params)) + norm(jnp.linalg.inv(Q) - jnp.eye(n_params))
            r = λ * loss / (loss + jnp.exp(-α * loss))
            loss *= r
            
            loss += l1_reg(J_eta)

            return loss, jnp.linalg.det(Q)

        loss, Q = jax.vmap(fn)(theta_batched, F_batched)
        return jnp.log(jnp.mean(loss)), jnp.mean(Q)

    # ---------------------- PREPARE TRAINING DATA -----------------------
    # Expect θs and F_fishnets to be 2D or higher; here we reshape them in batch format.
    theta_true = θs.reshape(-1, batch_size, n_params)
    F_fishnets = F_fishnets.reshape(-1, batch_size, n_params, n_params)

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
    F_ensemble = jnp.array(F_network_ensemble) / norm_factor  # Adjust if F_fishnets is already aggregated differently.
    theta_true = θs.reshape(-1, batch_size, n_params)
    F_fishnets_ensemble = [f.reshape(-1, batch_size, n_params, n_params) for f in F_ensemble]

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
    y_nonlinear_ensemble = []
    Jbar_ensemble = []
    mymodel = lambda d: model.apply(w, d)
    for k, _w in enumerate(ensemble_ws):
        print("applying model to ensemble member %d" % (k))
        current_model = lambda d: model.apply(_w, d, return_nonlinear=True)
        ηs,y_nonlinear = jax.vmap(current_model)(θs)

        getjac = lambda d: get_jacobian(d, w=_w)
        η_ensemble.append(ηs)
        y_nonlinear_ensemble.append(y_nonlinear)
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
            y, rotmat = rotate_coords(y, theta=θs, theta_fid=theta_fid)
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
    np.savez(outname,
             theta=np.array(θs),
             eta=np.array(ηs),
             Jacobians=np.array(Jbar),
             deltaJ=np.array(δJs),
             delta_invJ=np.array(δinvJ),
             meanF=np.array(F_ensemble),
             dFs=np.array(dFs),
             F_ensemble=np.array(allFs),
             norm_factor=norm_factor,
             ensemble_weights=weights,  # Using uniform weights here
             eta_ensemble=np.array(ys),
             y_nonlinear_ensemble=np.array(y_nonlinear_ensemble),
             Jbar_ensemble=np.array(dys)
    )

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
    # Example: F_fishnets and θs should be provided externally.
    # For testing purposes, we create fake input arrays.
    #fake_theta = jnp.linspace(-3.0, 3.0, 1000).reshape(-1, 2)
    #fake_F = jnp.tile(jnp.eye(2), (fake_theta.shape[0], 1, 1))

    # ---------------------- LOAD DATA FROM FILE -----------------------
    fname = "fishnets-log/fishnets_outputs"
    fname_full = fname + ".npz"
    data_npz = np.load(fname_full)
    thetas = jnp.array(data_npz["theta"])
    ensemble_weights = data_npz["ensemble_weights"]
    F_network_ensemble = jnp.array(data_npz["F_network_ensemble"])
    # Compute weighted average of network Fisher matrices:
    # Fs = jnp.average(F_network_ensemble, axis=0, weights=ensemble_weights)

    print("thetas", thetas)


    fit_flattening(F_network_ensemble, thetas,
                   ensemble_weights=ensemble_weights,
                   hidden_size=256,
                   batch_size=250,
                   epochs_phase1=10000,
                   epochs_phase2=1000,
                   finetune_epochs=1000,
                   min_epochs=1200,
                   patience=100,
                   lr_phase1=2e-6,
                   lr_schedule_initial=7e-5,
                   lr_decay=0.3,
                   lr_finetune=4e-6,
                   noise=1e-7,
                   seed=0,
                   output_prefix="flattened_coords_sr2",
                   SCALE_THETA=False,
                   do_plot=True)