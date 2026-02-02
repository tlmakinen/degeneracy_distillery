import jax
import jax.numpy as jnp
import flax.linen as nn


import optax
import numpy as np
import jax.random as jr


import matplotlib.pyplot as plt

from typing import Sequence, Any
from tqdm import tqdm
import optax



import flax.linen as nn
import scipy

from fishnets import *
from flatten_net import *
import os,sys


#!/usr/bin/env python
"""
This script defines a function run_experiment(...) that wraps the entire training,
fine‐tuning, ensemble processing, evaluation, and plotting pipeline.
It assumes that the modules “fishnets” and “flatten_net” (and their functions/classes)
are available.
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

        # A small dense layer for coefficients.
        x = nn.Dense(self.features[-1])(x)
        x = self.act(nn.Dense(self.features[0])(x))
        for feat in self.features[1:-1]:
            x = self.act(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

# ---------------------- UTILITY FUNCTIONS -----------------------
Array = Any

def minmax(x, xmin, xmax, feature_range):
    minval, maxval = feature_range
    xstd = (x - xmin) / (xmax - xmin)
    return xstd * (maxval - minval) + minval

def minmax_inv(x, xmin, xmax, feature_range):
    minval, maxval = feature_range
    x -= minval
    x /= (maxval - minval)
    x *= (xmax - xmin)
    return x + xmin

def weighted_std(values, weights, axis=0):
    """
    Return the weighted standard deviation.
    """
    average = jnp.average(values, weights=weights, axis=axis)
    variance = jnp.average((values - average) ** 2, weights=weights, axis=axis)
    return jnp.sqrt(variance)

# ---------------------- MAIN EXPERIMENT FUNCTION -----------------------
def run_experiment(fname: str = "simple_test_regression_outputs",
                   hidden_size: int = 256,
                   batch_size: int = 250,
                   epochs_phase1: int = 5500,
                   epochs_phase2: int = 1000,
                   finetune_epochs: int = 400,
                   min_epochs: int = 1200,
                   patience: int = 100,
                   lr_phase1: float = 2e-6,
                   lr_schedule_initial: float = 7e-5,
                   lr_decay: float = 0.3,
                   lr_finetune: float = 4e-6,
                   noise: float = 1e-4,
                   seed: int = 0,
                   output_prefix: str = "flattened_coords_sr",
                   SCALE_THETA: bool = False,
                   do_plot: bool = True):
    """
    Runs the full pipeline including training a flattening network,
    fine-tuning, fine-tuning each ensemble member, computing Jacobian errors,
    saving the results, and producing a contour plot of the output coordinates.
    
    Hyperparameters (with default values) include file names, network sizes,
    batch size, epoch counts, learning rates, and others.
    """
    # ---------------------- CONSTANTS & SETUP -----------------------
    # Model input parameter boundaries
    MAX_VAR = 3.0
    MIN_VAR = -3.0
    MAX_MU = 3.0
    MIN_MU = -3.0
    n_params = 2

    # Initialize random seed
    key = jr.PRNGKey(seed)

    # Compute auxiliary dimension from parameter count (unused in custom_MLP here)
    n_outputs = int(n_params + int(n_params * (n_params + 1)) // 2)

    # ---------------------- LOAD DATA FROM FILE -----------------------
    fname_full = fname + ".npz"
    data_npz = np.load(fname_full)
    θs = jnp.array(data_npz["theta"])
    ensemble_weights = data_npz["ensemble_weights"]
    F_network_ensemble = jnp.array(data_npz["F_network_ensemble"])
    # Compute weighted average of network Fisher matrices:
    Fs = jnp.average(F_network_ensemble, axis=0, weights=ensemble_weights)

    # Determine scaling factor for Fisher matrices.
    try:
        norm_factor = data_npz["n_d"] / 2.
    except Exception:
        norm_factor = 50.
    norm_factor = Fs.max() / 10.
    print('norm factor', norm_factor)
    Fs = Fs / norm_factor

    # Compute the bounds on θ from training data.
    max_x = θs.max(0) + 1e-3
    min_x = θs.min(0) - 1e-3

    # ---------------------- DEFINE THE MODEL -----------------------
    # Build custom_MLP with a list of features.
    model = custom_MLP(features=[hidden_size, hidden_size, hidden_size, n_params],
                       max_x = max_x,
                       min_x = min_x,
                       act = nn.softplus)

    # ---------------------- DEFINE LOSS AND HELPER FUNCTIONS -----------------------
    @jax.jit
    def norm(A):
        return jnp.sqrt(jnp.einsum('ij,ij->', A, A))

    def get_α(λ=100., ϵ=1e-7):
        return - jnp.log(ϵ * (λ - 1.) + ϵ ** 2. / (1. + ϵ)) / ϵ

    @jax.jit
    def l1_reg(x, alpha=0.001):
        return alpha * (jnp.abs(x)).mean()

    theta_star = jnp.array([1.0, 1.0])

    @jax.jit
    def info_loss(w, theta_batched, F_batched):
        λ = 100.  # lambda parameter
        ϵ = 1e-7
        α = get_α(λ, ϵ)

        def fn(theta, F):
            # Define model function.
            mymodel = lambda d: model.apply(w, d)
            # Compute Jacobian of the network at theta.
            J_eta = jax.jacrev(mymodel)(theta).squeeze()
            Jeta_inv = jnp.linalg.pinv(J_eta)
            Q = Jeta_inv @ F @ Jeta_inv.T

            loss = norm(Q - jnp.eye(n_params)) + norm(jnp.linalg.inv(Q) - jnp.eye(n_params))
            # (Optionally add L1 regularization on the Jacobian here.)
            r = λ * loss / (loss + jnp.exp(-α * loss))
            loss *= r
            return loss, jnp.linalg.det(Q)

        loss, Q = jax.vmap(fn)(theta_batched, F_batched)
        return jnp.mean(loss), jnp.mean(Q)

    # ---------------------- PREPARE TRAINING DATA -----------------------
    num = 10000  # number of simulations (if needed)
    # Reshape training data in batches.
    theta_true = θs.reshape(-1, batch_size, n_params)
    F_fishnets = Fs.reshape(-1, batch_size, n_params, n_params)

    # ---------------------- DEFINE THE TRAINING LOOP -----------------------
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
            # add noise to the Fisher matrices
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
        best_detFeta = np.inf
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
                # Shuffle training data
                randidx = jr.permutation(key, jnp.arange(num_sims), independent=True)
                theta_train = theta_true.reshape(-1, n_params)[randidx].reshape(-1, batch_size, n_params)
                F_train = F_fishnets.reshape(-1, n_params, n_params)[randidx].reshape(-1, batch_size, n_params, n_params)
                
                init_vals = (w, loss, opt_state, detFeta, key, theta_train, F_train)
                w, loss, opt_state, detFeta, key, theta_train, F_train = jax.lax.fori_loop(lower, upper, body_fun, init_vals)
                # Validation loss on last few batches
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

            pbar.set_description('epoch %d loss: %.4f, detFeta: %.4f, val_detFeta: %.4f'%(j, loss, detFeta, val_detFeta))
        return best_w, (losses, val_losses), (detFetas, val_detFetas)

    # ---------------------- TRAINING PHASE 1: TRAIN FLATTENER NET -----------------------
    print("TRAINING FLATTENER NET")
    key, rng = jr.split(key)
    w = model.init(key, jnp.ones((n_params,)))
    w, all_loss, all_dets = training_loop(key, w, theta_true, F_fishnets, 
                                          lr=lr_phase1, opt_type=optax.adam)
    
    # ---------------------- PHASE 2: FINE-TUNE FLATTENER NET -----------------------
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
    # Read ensemble Fisher matrices from file and rescale.
    F_ensemble = jnp.array(data_npz["F_network_ensemble"]) / norm_factor
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
                                               opt_type=optax.adam)
        ensemble_ws.append(_w)

    # ---------------------- EVALUATION: GET JACOBIANS & ENSEMBLE OUTPUTS -----------------------
    @jax.jit
    def get_jacobian(θ, w=w):
        mymodel = lambda d: model.apply(w, d)
        return jax.jacobian(mymodel)(θ)

    # Compute ensemble outputs for η and the corresponding Jacobians.
    η_ensemble = []
    Jbar_ensemble = []
    mymodel = lambda d: model.apply(w, d)
    for k, _w in enumerate(ensemble_ws):
        current_model = lambda d: model.apply(_w, d)
        ηs = jax.vmap(current_model)(θs)
        getjac = lambda d: get_jacobian(d, w=_w)
        print("applying model to ensemble member %d" % (k))
        η_ensemble.append(ηs)
        # Concatenate Jacobians over batches (example reshaping as in script)
        Jbar_ensemble.append(jnp.concatenate(jnp.array([jax.vmap(getjac)(t) for t in θs.reshape(-1, batch_size, n_params)])))
    
    # Compute Jacobians for the current flattening network.
    ηs = jax.vmap(mymodel)(θs)
    Jbar = jnp.concatenate(jnp.array([jax.vmap(get_jacobian)(t) for t in θs.reshape(-1, batch_size, n_params)]))

    allFs = jnp.array(np.squeeze(data_npz["F_network_ensemble"])) / norm_factor
    dFs = weighted_std(allFs, ensemble_weights, axis=0)

    # ---------------------- ERROR PROPAGATION: δJ SOLVER -----------------------
    def get_δJ(F, δF, Jbar):
        """
        Propagate error on the neural Fisher matrix to the flattened Jacobian.
        """
        J = np.linalg.inv(Jbar)
        Q = - np.einsum("bik,bkj,blj->bil", J, δF, J)
        X = J @ F
        A = np.einsum("bij,bkj->bik", X, X)
        S = jnp.array([scipy.linalg.solve_sylvester(a=A[i], b=A[i], q=Q[i]) for i in range(Q.shape[0])])
        δJ = S @ X
        return np.linalg.inv(J + δJ) - Jbar, δJ

    print("CALCULATING JACOBIAN ERROR")
    δJs, δinvJ = get_δJ(allFs.mean(0), dFs, Jbar)

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
             meanF=np.array(Fs),
             dFs=np.array(dFs),
             F_ensemble=np.array(allFs),
             norm_factor=norm_factor,
             ensemble_weights=ensemble_weights,
             eta_ensemble=np.array(η_ensemble),
             Jbar_ensemble=np.array(Jbar_ensemble)
    )

    # ---------------------- COORDINATE VISUALISATION -----------------------
    lo = [MIN_MU, MIN_VAR]
    hi = [MAX_MU, MAX_VAR]
    num_pts = 30
    xs = jnp.linspace(MIN_MU, MAX_MU, num_pts)
    ys = jnp.linspace(MIN_VAR, MAX_VAR, num_pts)
    xs, ys = jnp.meshgrid(xs, ys)
    X = jnp.stack([xs.flatten(), ys.flatten()], axis=-1)
    etas = jax.vmap(mymodel)(X)

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
        plt.savefig("coordinate_visualisation.png")
        plt.close()

    print("EXPERIMENT COMPLETED & RESULTS SAVED TO:", outname + ".npz")
    return w, ensemble_ws

# ---------------------- EXECUTION -----------------------
if __name__ == '__main__':
    # You can override hyperparameters here if desired.
    run_experiment(fname="simple_test_regression_outputs",
                   hidden_size=256,
                   batch_size=250,
                   epochs_phase1=5500,
                   epochs_phase2=1000,
                   finetune_epochs=400,
                   min_epochs=1200,
                   patience=100,
                   lr_phase1=2e-6,
                   lr_schedule_initial=7e-5,
                   lr_decay=0.3,
                   lr_finetune=4e-6,
                   noise=1e-5,
                   seed=0,
                   output_prefix="flattened_coords_sr",
                   SCALE_THETA=False,
                   do_plot=True)