import os
import sys
import shutil
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax
import numpy as np
from tqdm import tqdm
import yaml
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Sequence

from sklearn.preprocessing import MinMaxScaler

from fishnets import MLP, Fishnet_from_embedding

# --- Helper functions for prediction ---
def predicted_fishers(model, w, data):
    def _getf(d):
        return model.apply(w, d)[1]
    return jax.vmap(_getf)(data)

def predicted_mle(model, w, data):
    def _getmle(d):
        return model.apply(w, d)[0]
    return jax.vmap(_getmle)(data)

# --- Main training function ---
def train_fishnets(theta,
                   data,
                   theta_test,
                   data_test,
                   data_shape=None,
                   hids_min: int = 10,
                   hids_max: int = 300,
                   n_layers: int = 3,
                   num_models: int = 20,
                   seed_model: int = 201,
                   seed_train: int = 999,
                   train_batch_size: int = 200,
                   train_epochs: int = 4000,
                   train_min_epochs: int = 100,
                   patience: int = 20,
                   lr: float = 5e-5,
                   acts: list = None,
                   outdir: str = "fishnets-log"):
    """
    Trains an ensemble of fishnet networks.

    Parameters:
      theta         : Training parameter array (e.g. shape [n_samples, n_params])
      data          : Training simulation data (e.g. shape [n_samples, data_dim])
      theta_test    : Test parameter array
      data_test     : Test simulation data
      data_shape    : If provided, the last dimension of data; if None, set from data.shape[-1]
      hids_min      : Minimum number of neurons for each hidden layer
      hids_max      : Maximum number of neurons for each hidden layer (hidden sizes will be sampled uniformly in this range)
      n_layers      : Number of MLP layers --> could randomise this as well
      seed_model    : Seed for model initialization
      seed_train    : Seed for training loop randomness
      train_batch_size : Batch size used for training loops
      train_epochs     : Total epochs for training loop
      train_min_epochs : Minimum epochs required before early stopping
      patience         : Patience before early stopping (calculated on validation set)
      lr               : Learning rate for the optimizer
      acts             : List of activation functions for diversity.
      outdir           : Directory where outputs will be saved. If it does not exist, it is created;
                         if it exists, it is emptied before saving.
    
    Returns:
      ws             : A list of trained parameters for each ensemble network.
      ensemble_weights : A vector of weights for each ensemble member (computed from the best validation loss).
    """
    # Ensure the output directory exists and is empty.
    print("saving to", outdir)
    if os.path.exists(outdir):
        # Remove all contents within outdir.
        for filename in os.listdir(outdir):
            file_path = os.path.join(outdir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(outdir)
    # -------------- PARAMETER SETUP --------------
    n_params = theta.shape[-1]
    if data_shape is None:
        data_shape = data.shape[-1]
    input_shape = (data_shape,)
    xmin = theta.min(0) - 1e-3
    xmax = theta.max(0) + 1e-3

    # -------------- RESCALE DATA --------------
    data_scaler = MinMaxScaler(feature_range=(0, 1))
    data = data_scaler.fit_transform(data.reshape(-1, data_shape)).reshape(data.shape)
    data_test = data_scaler.transform(data_test.reshape(-1, data_shape)).reshape(data_test.shape)
    print("data_test shape:", data_test.shape)
    print("theta_test shape:", theta_test.shape)

    # -------------- INITIALISE MODELS --------------
    key = jr.PRNGKey(seed_model)

    # Define a list of activation functions for diversity.
    mish = lambda x: x * nn.tanh(nn.softplus(x))
    if acts is None:
        acts = [nn.relu, nn.relu, nn.relu,
                nn.leaky_relu, nn.leaky_relu, nn.leaky_relu, nn.leaky_relu,
                nn.swish, nn.swish, nn.swish, mish, mish,
                nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu]

    
    #num_models = len(acts)
    idx_acts = np.random.choice(np.arange(len(acts)), size=(num_models,))
    acts = [acts[i] for i in idx_acts]

    hids_range = np.arange(hids_min, hids_max)
    all_n_hidden = []
    for n in range(num_models):
        key, rng = jr.split(key)
        hidden = int(jr.choice(key, hids_range, replace=True))
        print("Chosen hidden size for model", n+1, ":", hidden)
        all_n_hidden.append([hidden]*n_layers)  # three-layer network

    # Build ensemble: each network is a sequential composition of an MLP and a Fishnet_from_embedding.
    models = [
        nn.Sequential([
            MLP(all_n_hidden[i], act=acts[i]),
            Fishnet_from_embedding(n_p=n_params, act=acts[i], hidden=all_n_hidden[i][0])
        ])
        for i in range(num_models)
    ]

    data = jnp.squeeze(data)
    keys = jr.split(key, num=num_models)
    ws = [models[i].init(keys[i], data[0]) for i in range(num_models)]

    # -------------- SHUFFLE DATA BEFORE TRAINING --------------
    # Shuffle training data before batching to ensure proper randomization
    key = jr.PRNGKey(seed_train)
    key, shuffle_key = jr.split(key)
    n_train_samples = theta.shape[0]
    shuffle_idx = jr.permutation(shuffle_key, jnp.arange(n_train_samples))
    theta = theta[shuffle_idx]
    data = data[shuffle_idx]
    
    # -------------- DEFINE TRAINING LOOP --------------
    train_batch = train_batch_size
    train_epochs_val = train_epochs
    train_min_epochs_val = train_min_epochs

    def training_loop(key, model, w, data, theta, data_val, theta_val,
                      patience=patience, epochs=train_epochs_val, min_epochs=train_min_epochs_val):
        @jax.jit
        def kl_loss(w, x_batched, theta_batched):
            def fn(x, theta):
                mle, F = model.apply(w, x)
                return mle, F
            mle, F = jax.vmap(fn)(x_batched, theta_batched)
            res = theta_batched - mle
            return 0.5 * jnp.mean(jnp.einsum('ij,ij->i', res, jnp.einsum('ijk,ik->ij', F, res)) - jnp.log(jnp.linalg.det(F)), axis=0)

        tx = optax.adam(learning_rate=lr)
        opt_state = tx.init(w)
        loss_grad_fn = jax.value_and_grad(kl_loss)

        @jax.jit
        def body_fun(i, inputs):
            w, loss_val, opt_state, _data, _theta = inputs
            x_samples = _data[i]
            y_samples = _theta[i]
            loss, grads = loss_grad_fn(w, x_samples, y_samples)
            updates, opt_state = tx.update(grads, opt_state, w)
            w = optax.apply_updates(w, updates)
            return w, loss_val + loss, opt_state, _data, _theta

        losses = jnp.zeros(epochs)
        val_losses = jnp.zeros(epochs)
        loss_val = 0.0
        n_train = 10000  # fixed number of training samples (could be parameterized)
        lower = 0
        upper = n_train // train_batch
        counter = 0
        patience_counter = 0
        best_loss = jnp.inf
        best_w = w
        pbar = tqdm(range(epochs), desc="Training Epochs", leave=True, position=0)

        for j in pbar:
            key, rng = jr.split(key)
            # Shuffle training data
            randidx = jr.permutation(key, jnp.arange(theta.reshape(-1, n_params).shape[0]), independent=True)
            _data = data.reshape(-1, data_shape)[randidx].reshape(-1, train_batch, data_shape)
            _theta = theta.reshape(-1, n_params)[randidx].reshape(-1, train_batch, n_params)
            inits = (w, loss_val, opt_state, _data, _theta)
            w, loss_val, opt_state, _, _theta = jax.lax.fori_loop(lower, upper, body_fun, inits)
            loss_val /= _data.shape[0]
            losses = losses.at[j].set(loss_val)

            # Evaluate validation loss on provided data_val and theta_val 
            val_loss, _ = loss_grad_fn(w, data_val, theta_val)
            val_losses = val_losses.at[j].set(val_loss)
            pbar.set_description('Epoch %d loss: %.5f ; val_loss: %.5f' % (j, loss_val, val_loss))

            counter += 1
            if val_loss < best_loss:
                best_loss = val_loss
                best_w = w
                patience_counter = 0
            else:
                patience_counter += 1

            if (patience_counter - min_epochs > patience) and (j + 1 > min_epochs):
                print("\nEarly stopping triggered at epoch %d" % j)
                break

        return losses[:j], val_losses[:j], best_loss, best_w

    # -------------- TRAIN EACH ENSEMBLE MODEL --------------
    print("STARTING TRAINING LOOP")
    all_losses = []
    all_val_losses = []
    best_val_losses = []
    keys = jr.split(key, num=num_models)

    for i, w in enumerate(ws):
        print("\nTraining model %d of %d" % (i+1, num_models))
        loss, val_loss, best_val_loss, wtrained = training_loop(
            keys[i],
            models[i],
            w,
            data,
            theta,
            data_test.squeeze(),
            theta_test.squeeze(),
            patience=patience,
            epochs=train_epochs_val,
            min_epochs=train_min_epochs_val)
        all_losses.append(loss)
        all_val_losses.append(val_loss)
        best_val_losses.append(best_val_loss)
        ws[i] = wtrained



    # Compute ensemble weights from best validation losses.
    ensemble_weights_arr = jnp.array([1. / jnp.exp(best_val_losses[i]) for i in range(num_models)])
    print("Ensemble weights:", ensemble_weights_arr)

    # -------------- PREDICTION ON TEST DATASET --------------
    data_test = data_test.reshape(-1, data_shape)
    ensemble_F_predictions = jnp.array([predicted_fishers(models[i], ws[i], data_test) for i in range(num_models)])
    ensemble_mle_predictions = jnp.array([predicted_mle(models[i], ws[i], data_test) for i in range(num_models)])

    outname = os.path.join(outdir, "fishnets_outputs")
    np.savez(outname,
             theta=theta_test,
             F_network_ensemble=ensemble_F_predictions,
             mle_network_ensemble=ensemble_mle_predictions,
             ensemble_weights=ensemble_weights_arr)
    print("Training completed. Outputs saved to:", outname + ".npz")

    return ws, ensemble_weights_arr

# -------------- EXAMPLE USAGE --------------
if __name__ == '__main__':
    # For demonstration, load synthetic data or precomputed arrays.
    # Here we create fake arrays for theta, data, theta_test, and data_test.
    n_samples = 10000
    n_test = 5000
    n_params = 2
    n_d = 50  # e.g., simulation flattening dimension

    MAX_VAR=10.0
    MIN_VAR=0.0666666666666667

    MAX_VAR=20.0
    MIN_VAR=0.2

    MAX_MU=5.0
    MIN_MU=-5.0

    @jax.jit
    def Fisher(θ, n_d=n_d):
        A = θ[1]
        return jnp.array([[n_d * A, 0.], [0., n_d * (0.75 * jnp.sqrt(2*jnp.pi) * A**(-5./2.))]])

    @jax.jit
    def simulator(key, θ):
        return θ[0] + jr.normal(key, shape=(n_d,)) * jnp.sqrt((1./θ[1]))


    @jax.jit
    def simulator(key, θ):
        return θ[0] + jr.normal(key, shape=(n_d,)) * jnp.sqrt((θ[1]))



    # -------------- MAKE SOME DATA --------------
    print('making data')
    key = jr.PRNGKey(0)
    key1,key2 = jr.split(key)

    mu_ = jr.uniform(key1, shape=(n_samples,), minval=MIN_MU, maxval=MAX_MU)
    sigma_ = jr.uniform(key2, shape=(n_samples,), minval=MIN_VAR, maxval=MAX_VAR)
    theta_ = jnp.stack([mu_, sigma_], axis=-1)

    # make test set
    key1,key2 = jr.split(key1)
    mu_test = jr.uniform(key1, shape=(n_test,), minval=MIN_MU, maxval=MAX_MU)
    sigma_test = jr.uniform(key2, shape=(n_test,), minval=MIN_VAR, maxval=MAX_VAR)
    theta_test = jnp.stack([mu_test, sigma_test], axis=-1)

    # create data
    keys = jr.split(key, num=n_samples)
    data = jax.vmap(simulator)(keys, theta_) #[:, :, jnp.newaxis]
    keys = jr.split(key2, num=n_test)
    data_test = jax.vmap(simulator)(keys, theta_test) #[:, :, jnp.newaxis]

    print('data test shape', data_test.shape)
    theta = theta_.copy()


    # Call train_fishnets with hyperparameters as desired.
    ws, ens_weights = train_fishnets(theta, data, theta_test, data_test,
                                     data_shape=n_d,
                                     hids_min=10,
                                     hids_max=300,
                                     n_layers=3,
                                     seed_model=201,
                                     seed_train=999,
                                     num_models=20,
                                     train_batch_size=200,
                                     train_epochs=4000,
                                     train_min_epochs=100,
                                     patience=20,
                                     lr=5e-5,
                                     outdir="fishnets-log")