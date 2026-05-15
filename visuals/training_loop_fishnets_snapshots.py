"""
Modified version of `degeneracy_distillery.training_loop_fishnets.train_fishnets`
that saves per-model snapshots of the predicted Fisher matrix on the
validation set every `save_every` epochs.

The companion script `make_detF_gif.py` consumes these snapshots and
produces a .gif of the ensemble-averaged val_detF evolving over training,
plotted as a 2D scatter colormap over two parameters (in the same style
as `degeneracy_distillery.diagnostics.diagnose_low_information`).

Because the ensemble is trained sequentially (one member at a time), we
cannot ensemble-average at training time. Instead we store per-model
Fisher snapshots; the GIF builder reconstructs the ensemble-averaged
val_detF on a common epoch axis (members that stopped early are held at
their last snapshot).

Snapshot layout under ``outdir/snapshots/``::

    snapshots/
        metadata.json         # num_models, ensemble_weights, save_every, ...
        theta_val.npy         # (n_val, n_params)
        model_00/
            epoch_00000.npz   # F_val: (n_val, n_params, n_params)
            epoch_00005.npz
            ...
        model_01/
            ...
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from degeneracy_distillery.fishnets import (  # noqa: E402
    Fishnet_from_embedding,
    optimized_smooth_leaky,
    resMLP,
)
from degeneracy_distillery.io_utils import create_results_dict  # noqa: E402


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def predicted_fishers(model, w, data):
    def _getf(d):
        return model.apply(w, d)[1]

    return jax.vmap(_getf)(data)


def predicted_mle(model, w, data):
    def _getmle(d):
        return model.apply(w, d)[0]

    return jax.vmap(_getmle)(data)


def _save_snapshot(snap_dir: str, model_idx: int, epoch: int, F_val: np.ndarray) -> None:
    mdir = os.path.join(snap_dir, f"model_{model_idx:02d}")
    os.makedirs(mdir, exist_ok=True)
    np.savez_compressed(
        os.path.join(mdir, f"epoch_{epoch:05d}.npz"),
        F_val=F_val,
    )


def _clear_outdir(outdir: str) -> None:
    if os.path.exists(outdir):
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


# ----------------------------------------------------------------------------
# Main training function with snapshots
# ----------------------------------------------------------------------------

def train_fishnets_with_snapshots(
    theta,
    data,
    theta_test,
    data_test,
    data_shape: Optional[int] = None,
    hids_min: int = 10,
    hids_max: int = 300,
    n_layers: Union[int, Sequence[int]] = 3,
    num_models: int = 20,
    seed_model: int = 201,
    seed_train: int = 999,
    train_batch_size: int = 200,
    train_epochs: int = 4000,
    train_min_epochs: int = 100,
    patience: int = 20,
    lr: float = 5e-5,
    acts: Optional[list] = None,
    scaler_type: str = "minmax",
    embedding_net: Optional[nn.Module] = None,
    outdir: str = "fishnets-log-snapshots",
    snapshots_subdir: str = "snapshots",
    save_every: int = 5,
    save_initial: bool = True,
    save_final: bool = True,
    update_pbar_every: int = 10,
    param_names: Optional[Sequence[str]] = None,
):
    """
    Train an ensemble of fishnet networks while writing per-model
    snapshots of the predicted Fisher on the validation set every
    ``save_every`` epochs (and at epoch 0 if ``save_initial=True``).

    Parameters mirror :func:`degeneracy_distillery.training_loop_fishnets.train_fishnets`
    except for the snapshot controls:

    snapshots_subdir : str
        Subdirectory of ``outdir`` where snapshots are written.
    save_every : int
        Save a Fisher snapshot every this many epochs.
    save_initial : bool
        If True, save a snapshot at epoch 0 (immediately after init,
        before any gradient steps) so the GIF starts from the untrained
        ensemble.
    save_final : bool
        If True, also save a snapshot at the final epoch of each model
        (after early stopping) even if it isn't on the ``save_every``
        grid.
    param_names : sequence of str, optional
        Display names for the parameters; recorded in ``metadata.json``
        and used by the GIF generator's axis labels.

    Returns the same tuple as ``train_fishnets``::

        ws, ensemble_weights, models, data_scaler, outputs
    """
    print("saving to", outdir)
    _clear_outdir(outdir)
    snap_dir = os.path.join(outdir, snapshots_subdir)
    os.makedirs(snap_dir, exist_ok=True)

    # -------------- PARAMETER SETUP --------------
    n_params = theta.shape[-1]
    if data_shape is None:
        data_shape = data.shape[-1]

    if param_names is None:
        param_names = [f"theta_{i + 1}" for i in range(n_params)]
    elif len(param_names) != n_params:
        raise ValueError(
            f"param_names has length {len(param_names)}; expected {n_params}"
        )

    # -------------- RESCALE DATA --------------
    if scaler_type.lower() == "minmax":
        data_scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaler_type.lower() == "standard":
        data_scaler = StandardScaler()
    else:
        raise ValueError(
            f"Unknown scaler_type: '{scaler_type}'. Options are 'minmax' or 'standard'."
        )

    data = data_scaler.fit_transform(data.reshape(-1, data_shape)).reshape(data.shape)
    data_test = data_scaler.transform(data_test.reshape(-1, data_shape)).reshape(
        data_test.shape
    )
    print("data_test shape:", data_test.shape)
    print("theta_test shape:", theta_test.shape)

    # Save validation parameters once (canonical theta-space).
    np.save(os.path.join(snap_dir, "theta_val.npy"), np.asarray(theta_test))

    # -------------- INITIALISE MODELS --------------
    key = jr.PRNGKey(seed_model)

    mish = lambda x: x * nn.tanh(nn.softplus(x))
    if acts is None:
        acts = [
            nn.relu, nn.relu, nn.relu,
            nn.leaky_relu, nn.leaky_relu, nn.leaky_relu, nn.leaky_relu,
            nn.swish, nn.swish, nn.swish, mish, mish,
            optimized_smooth_leaky, optimized_smooth_leaky, optimized_smooth_leaky,
            nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu,
        ]

    idx_acts = np.random.choice(np.arange(len(acts)), size=(num_models,))
    acts = [acts[i] for i in idx_acts]

    hids_range = np.arange(hids_min, hids_max)
    if isinstance(n_layers, (list, tuple, np.ndarray)):
        if len(n_layers) != 2:
            raise ValueError("n_layers range must have exactly two values: [min_layers, max_layers].")
        min_layers, max_layers = int(n_layers[0]), int(n_layers[1])
        if min_layers <= 0 or max_layers <= 0:
            raise ValueError("n_layers values must be positive.")
        if min_layers > max_layers:
            raise ValueError("n_layers range must satisfy min_layers <= max_layers.")
        sample_n_layers = True
    else:
        fixed_layers = int(n_layers)
        if fixed_layers <= 0:
            raise ValueError("n_layers must be a positive integer.")
        sample_n_layers = False

    all_n_hidden = []
    all_sharpness = []
    all_threshold = []
    for n in range(num_models):
        key, hidden_key = jr.split(key)
        hidden = int(jr.choice(hidden_key, hids_range, replace=True))
        if sample_n_layers:
            key, layers_key = jr.split(key)
            n_layers_model = int(
                jr.randint(layers_key, shape=(), minval=min_layers, maxval=max_layers + 1)
            )
        else:
            n_layers_model = fixed_layers
        print("Chosen hidden size for model", n + 1, ":", hidden, "| layers:", n_layers_model)
        all_n_hidden.append([hidden] * n_layers_model)

        key, rng1, rng2 = jr.split(key, 3)
        sharpness_val = jr.normal(rng1, shape=(1,)) * 0.7 + 5.0
        threshold_val = jr.normal(rng2, shape=(1,)) * 0.7 + 1.0
        all_sharpness.append(sharpness_val)
        all_threshold.append(threshold_val)

    if embedding_net is not None:
        models = [
            nn.Sequential([
                embedding_net,
                resMLP(all_n_hidden[i], act=acts[i]),
                Fishnet_from_embedding(
                    n_p=n_params,
                    act=acts[i],
                    hidden=all_n_hidden[i][0],
                    act_fisher=nn.gelu,
                    sharpness=all_sharpness[i],
                    threshold=all_threshold[i],
                ),
            ])
            for i in range(num_models)
        ]
    else:
        models = [
            nn.Sequential([
                resMLP(all_n_hidden[i], act=acts[i]),
                Fishnet_from_embedding(
                    n_p=n_params,
                    act=acts[i],
                    hidden=all_n_hidden[i][0],
                    act_fisher=nn.gelu,
                    sharpness=all_sharpness[i],
                    threshold=all_threshold[i],
                ),
            ])
            for i in range(num_models)
        ]

    data = jnp.squeeze(data)
    keys = jr.split(key, num=num_models)
    ws = [models[i].init(keys[i], data[0]) for i in range(num_models)]

    # -------------- SHUFFLE TRAINING DATA --------------
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
    _pbar_stride = max(1, int(update_pbar_every))

    def training_loop(
        key,
        model,
        w,
        data,
        theta,
        data_val,
        theta_val,
        model_idx: int,
        patience=patience,
        epochs=train_epochs_val,
        min_epochs=train_min_epochs_val,
    ):
        @jax.jit
        def kl_loss(w, x_batched, theta_batched):
            def fn(x, theta):
                mle, F = model.apply(w, x)
                return mle, F

            mle, F = jax.vmap(fn)(x_batched, theta_batched)
            res = theta_batched - mle
            sign, logdet = jnp.linalg.slogdet(F)
            logdet = jnp.clip(logdet, -50, 50)
            return 0.5 * jnp.mean(
                jnp.einsum("ij,ij->i", res, jnp.einsum("ijk,ik->ij", F, res)) - logdet,
                axis=0,
            )

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

        # Snapshot epoch 0 (untrained state) if requested.
        saved_epochs: list[int] = []
        if save_initial:
            F0 = np.asarray(predicted_fishers(model, w, data_val))
            _save_snapshot(snap_dir, model_idx, 0, F0)
            saved_epochs.append(0)

        losses = jnp.zeros(epochs)
        val_losses = jnp.zeros(epochs)
        loss_val = 0.0
        n_train = (theta.reshape(-1, n_params).shape[0] // train_batch) * train_batch
        lower = 0
        upper = n_train // train_batch
        counter = 0
        patience_counter = 0
        best_loss = jnp.inf
        best_w = w
        pbar = tqdm(
            range(epochs),
            desc="Training Epochs",
            leave=True,
            position=0,
            miniters=_pbar_stride,
        )

        j = 0
        for j in pbar:
            key, rng = jr.split(key)
            randidx = jr.permutation(
                key, jnp.arange(theta.reshape(-1, n_params).shape[0]), independent=True
            )[:n_train]
            _data = data.reshape(-1, data_shape)[randidx].reshape(-1, train_batch, data_shape)
            _theta = theta.reshape(-1, n_params)[randidx].reshape(-1, train_batch, n_params)
            inits = (w, loss_val, opt_state, _data, _theta)
            w, loss_val, opt_state, _, _theta = jax.lax.fori_loop(lower, upper, body_fun, inits)
            loss_val /= _data.shape[0]
            losses = losses.at[j].set(loss_val)

            val_loss, _ = loss_grad_fn(w, data_val, theta_val)
            val_losses = val_losses.at[j].set(val_loss)

            counter += 1
            if val_loss < best_loss:
                best_loss = val_loss
                best_w = w
                patience_counter = 0
            else:
                patience_counter += 1

            # Save snapshot on the save_every grid (using 1-indexed epoch
            # number so the first post-training snapshot is at j+1=save_every).
            epoch_one_indexed = j + 1
            if save_every > 0 and (epoch_one_indexed % save_every == 0):
                F_val = np.asarray(predicted_fishers(model, w, data_val))
                _save_snapshot(snap_dir, model_idx, epoch_one_indexed, F_val)
                saved_epochs.append(epoch_one_indexed)

            will_stop = (patience_counter - min_epochs > patience) and (j + 1 > min_epochs)
            if (j + 1) % _pbar_stride == 0 or j == epochs - 1 or will_stop:
                pbar.set_description(
                    "Epoch %d loss: %.5f ; val_loss: %.5f" % (j, loss_val, val_loss)
                )
            if will_stop:
                print("\nEarly stopping triggered at epoch %d" % j)
                break

        # Optional final snapshot if the last epoch wasn't on the grid.
        final_epoch = j + 1
        if save_final and (len(saved_epochs) == 0 or saved_epochs[-1] != final_epoch):
            F_val = np.asarray(predicted_fishers(model, w, data_val))
            _save_snapshot(snap_dir, model_idx, final_epoch, F_val)
            saved_epochs.append(final_epoch)

        return losses[:j], val_losses[:j], best_loss, best_w, saved_epochs, final_epoch

    # -------------- TRAIN EACH ENSEMBLE MODEL --------------
    print("STARTING TRAINING LOOP")
    all_losses = []
    all_val_losses = []
    best_val_losses = []
    final_epochs = []
    per_model_saved_epochs = []
    keys = jr.split(key, num=num_models)

    for i, w in enumerate(ws):
        print("\nTraining model %d of %d" % (i + 1, num_models))
        loss, val_loss, best_val_loss, wtrained, saved_epochs, final_epoch = training_loop(
            keys[i],
            models[i],
            w,
            data,
            theta,
            data_test.squeeze(),
            theta_test.squeeze(),
            model_idx=i,
            patience=patience,
            epochs=train_epochs_val,
            min_epochs=train_min_epochs_val,
        )
        all_losses.append(loss)
        all_val_losses.append(val_loss)
        best_val_losses.append(best_val_loss)
        final_epochs.append(final_epoch)
        per_model_saved_epochs.append(saved_epochs)
        ws[i] = wtrained

        # Write metadata incrementally so a killed run still produces a
        # usable snapshot directory.
        _write_metadata(
            snap_dir=snap_dir,
            num_models=num_models,
            n_params=n_params,
            param_names=list(param_names),
            save_every=save_every,
            save_initial=save_initial,
            save_final=save_final,
            best_val_losses=[float(b) for b in best_val_losses],
            final_epochs=[int(e) for e in final_epochs],
            per_model_saved_epochs=[list(map(int, e)) for e in per_model_saved_epochs],
            n_models_completed=i + 1,
        )

    # Ensemble weights from best validation losses (same recipe as the
    # original training loop).
    ensemble_weights_arr = jnp.array(
        [1.0 / jnp.exp(best_val_losses[i]) for i in range(num_models)]
    )
    print("Ensemble weights:", ensemble_weights_arr)

    # Final metadata write including ensemble weights.
    _write_metadata(
        snap_dir=snap_dir,
        num_models=num_models,
        n_params=n_params,
        param_names=list(param_names),
        save_every=save_every,
        save_initial=save_initial,
        save_final=save_final,
        best_val_losses=[float(b) for b in best_val_losses],
        final_epochs=[int(e) for e in final_epochs],
        per_model_saved_epochs=[list(map(int, e)) for e in per_model_saved_epochs],
        n_models_completed=num_models,
        ensemble_weights=[float(x) for x in np.asarray(ensemble_weights_arr).tolist()],
    )

    # -------------- PREDICTION ON TEST DATASET --------------
    data_test_flat = data_test.reshape(-1, data_shape)
    ensemble_F_predictions = jnp.array(
        [predicted_fishers(models[i], ws[i], data_test_flat) for i in range(num_models)]
    )
    ensemble_mle_predictions = jnp.array(
        [predicted_mle(models[i], ws[i], data_test_flat) for i in range(num_models)]
    )

    outputs = create_results_dict(
        theta=theta_test,
        Fs=ensemble_F_predictions,
        mle=ensemble_mle_predictions,
        ensemble_weights=ensemble_weights_arr,
        x=data_test_flat,
    )

    outname = os.path.join(outdir, "fishnets_outputs")
    np.savez(outname, **dict(outputs))
    print("Training completed. Outputs saved to:", outname + ".npz")
    print("Snapshots saved to:", snap_dir)

    return ws, ensemble_weights_arr, models, data_scaler, outputs


def _write_metadata(
    snap_dir: str,
    num_models: int,
    n_params: int,
    param_names,
    save_every: int,
    save_initial: bool,
    save_final: bool,
    best_val_losses,
    final_epochs,
    per_model_saved_epochs,
    n_models_completed: int,
    ensemble_weights=None,
) -> None:
    meta = {
        "num_models": int(num_models),
        "n_params": int(n_params),
        "param_names": list(param_names),
        "save_every": int(save_every),
        "save_initial": bool(save_initial),
        "save_final": bool(save_final),
        "best_val_losses": list(best_val_losses),
        "final_epochs": list(final_epochs),
        "per_model_saved_epochs": list(per_model_saved_epochs),
        "n_models_completed": int(n_models_completed),
    }
    if ensemble_weights is not None:
        meta["ensemble_weights"] = list(ensemble_weights)
    with open(os.path.join(snap_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ----------------------------------------------------------------------------
# Example usage: mu/sigma 2D toy problem (mirrors the original script's
# __main__ block so the snapshot pipeline is runnable end-to-end).
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    n_samples = 10_000
    n_test = 5_000
    n_params = 2
    n_d = 50

    MAX_VAR = 20.0
    MIN_VAR = 0.2
    MAX_MU = 5.0
    MIN_MU = -5.0

    @jax.jit
    def simulator(key, theta):
        return theta[0] + jr.normal(key, shape=(n_d,)) * jnp.sqrt(theta[1])

    print("making data")
    key = jr.PRNGKey(0)
    key1, key2 = jr.split(key)
    mu_ = jr.uniform(key1, shape=(n_samples,), minval=MIN_MU, maxval=MAX_MU)
    sigma_ = jr.uniform(key2, shape=(n_samples,), minval=MIN_VAR, maxval=MAX_VAR)
    theta_ = jnp.stack([mu_, sigma_], axis=-1)

    key1, key2 = jr.split(key1)
    mu_test = jr.uniform(key1, shape=(n_test,), minval=MIN_MU, maxval=MAX_MU)
    sigma_test = jr.uniform(key2, shape=(n_test,), minval=MIN_VAR, maxval=MAX_VAR)
    theta_test = jnp.stack([mu_test, sigma_test], axis=-1)

    keys = jr.split(key, num=n_samples)
    data = jax.vmap(simulator)(keys, theta_)
    keys = jr.split(key2, num=n_test)
    data_test = jax.vmap(simulator)(keys, theta_test)

    print("data test shape", data_test.shape)
    theta = theta_.copy()

    train_fishnets_with_snapshots(
        theta,
        data,
        theta_test,
        data_test,
        data_shape=n_d,
        hids_min=10,
        hids_max=300,
        n_layers=3,
        num_models=20,
        seed_model=201,
        seed_train=999,
        train_batch_size=200,
        train_epochs=500,
        train_min_epochs=50,
        patience=20,
        lr=5e-5,
        outdir=os.path.join(_THIS_DIR, "fishnets-log-snapshots"),
        save_every=5,
        save_initial=True,
        save_final=True,
        param_names=["mu", "sigma^2"],
    )
