"""
Dataset-based training loop for Fishnet ensembles.

This is an adaptation of ``training_loop_fishnets.train_fishnets`` that consumes
indexable / iterable datasets (PyTorch- or PyG-style) instead of pre-loaded
``(theta, data)`` arrays. The training loop fetches batches lazily from the
dataset, so the data never has to fit into a single JAX array.

Typical usage with a PyG-style dataset
--------------------------------------

    import torch_geometric as pyg

    train_ds = pyg.datasets.QM7b(root='./toy')
    test_ds  = pyg.datasets.QM7b(root='./toy_test')

    def collate_fn(samples):
        # samples: list of pyg.data.Data
        # convert into whatever batched representation your embedding net wants.
        x_batch     = my_pyg_to_jraph(samples)            # e.g. jraph.GraphsTuple
        theta_batch = jnp.stack([jnp.asarray(s.y) for s in samples])
        return x_batch, theta_batch

    def model_apply_fn(model, w, x_batch):
        # model directly accepts the batched graph -> per-graph (mle, F).
        return model.apply(w, x_batch)

    def model_init_fn(model, key, x_batch):
        # init on the batched graph rather than a single element of x_batch.
        return model.init(key, x_batch)

    ws, ens_w, models, _, outputs = train_fishnets_dataset(
        train_ds, test_ds,
        n_params=14,
        collate_fn=collate_fn,
        model_apply_fn=model_apply_fn,
        model_init_fn=model_init_fn,
        embedding_net=MyGNN(),
        ...
    )

For tensor-only datasets (``dataset[i] -> (x, theta)`` with fixed-shape ``x``)
the defaults work out of the box.
"""

import os
import shutil
from typing import Any, Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Support both package import and direct script execution
try:
    from .fishnets import resMLP, Fishnet_from_embedding, optimized_smooth_leaky
    from .io_utils import create_results_dict
except ImportError:
    from fishnets import resMLP, Fishnet_from_embedding, optimized_smooth_leaky
    from io_utils import create_results_dict


# =============================================================================
# Default helpers (work for tensor datasets; override for graph datasets)
# =============================================================================

def _default_sample_to_xy(sample: Any):
    """Extract (x, theta) from a single dataset sample.

    Supports the following formats out of the box:
      - tuple/list:        ``(x, theta)``
      - PyG-like object:   ``sample`` with attribute ``.y`` (x = sample, theta = sample.y)
      - dict-like:         ``{'x': x, 'y': theta}`` or ``{'x': x, 'theta': theta}``
    """
    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        return sample[0], sample[1]
    if isinstance(sample, dict):
        x = sample.get('x', sample.get('data'))
        theta = sample.get('y', sample.get('theta'))
        if x is None or theta is None:
            raise ValueError(
                "dict samples must contain 'x'/'data' and 'y'/'theta' keys."
            )
        return x, theta
    if hasattr(sample, 'y'):
        return sample, sample.y
    raise ValueError(
        "Could not infer (x, theta) from dataset sample. "
        "Pass a custom `sample_to_xy_fn`."
    )


def _to_jnp(x):
    """Best-effort conversion to a jax array (handles torch tensors)."""
    if isinstance(x, jnp.ndarray):
        return x
    # torch.Tensor support without forcing torch as a dependency
    if hasattr(x, 'detach') and hasattr(x, 'cpu') and hasattr(x, 'numpy'):
        return jnp.asarray(x.detach().cpu().numpy())
    return jnp.asarray(np.asarray(x))


def make_default_collate(sample_to_xy_fn: Callable):
    """Build a default collate function that stacks tensor-like samples."""
    def collate_fn(samples):
        xs, ths = zip(*[sample_to_xy_fn(s) for s in samples])
        x_batch = jnp.stack([_to_jnp(x) for x in xs], axis=0)
        theta_batch = jnp.stack([_to_jnp(t) for t in ths], axis=0)
        return x_batch, theta_batch
    return collate_fn


def _default_model_apply(model, w, x_batch):
    """Default per-batch apply: vmap over the leading axis of ``x_batch``.

    Assumes ``x_batch`` is a single jax array of shape ``(batch, ...)`` and that
    the model takes a single sample. Override ``model_apply_fn`` if your
    embedding network already accepts a batched input (e.g. a jraph GNN).
    """
    def fn(x):
        return model.apply(w, x)
    return jax.vmap(fn)(x_batch)


def _default_model_init(model, key, x_batch):
    """Default init.

    Tries the vmap-friendly path first by initialising on ``x_batch[0]``
    (a single sample) to match the default ``_default_model_apply`` which
    vmaps ``model.apply`` over the leading batch axis.

    If ``x_batch`` is a structured batched container that can't be subscripted
    (e.g. a ``jraph.GraphsTuple`` or PyG-style ``GraphBatch`` coming from a
    custom ``collate_fn``), we fall back to initialising on the whole batched
    input -- which is what a model that consumes batched graphs directly
    expects anyway.

    You can bypass this heuristic entirely by passing a custom
    ``model_init_fn``.
    """
    try:
        single = x_batch[0]
    except (TypeError, KeyError, IndexError):
        return model.init(key, x_batch)
    return model.init(key, single)


# =============================================================================
# Dataset utilities
# =============================================================================

def _materialise_dataset(dataset):
    """Return a list-like wrapper that supports ``len`` and ``__getitem__``.

    If the input already supports both, returns it unchanged. Otherwise we
    consume the iterator into a list (which is required for shuffling).
    """
    if hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__'):
        return dataset
    return list(dataset)


def _iter_batches(dataset, indices, batch_size, collate_fn, drop_last=True):
    """Yield collated batches for the given index order."""
    n = len(indices)
    n_full = (n // batch_size) * batch_size
    last = n_full if drop_last else n
    for start in range(0, last, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        samples = [dataset[int(i)] for i in idx]
        yield collate_fn(samples)


# =============================================================================
# Prediction helpers
# =============================================================================

def _predict_dataset(model, w, dataset, batch_size, collate_fn, model_apply_fn):
    """Run the model over an entire dataset in batches and stack outputs.

    Returns ``(mle, F, theta)`` stacked along the leading axis.
    """
    n = len(dataset)
    all_mle, all_F, all_theta = [], [], []
    indices = np.arange(n)
    for x_batch, theta_batch in _iter_batches(
        dataset, indices, batch_size, collate_fn, drop_last=False
    ):
        mle, F = model_apply_fn(model, w, x_batch)
        all_mle.append(np.asarray(mle))
        all_F.append(np.asarray(F))
        all_theta.append(np.asarray(theta_batch))
    return (
        jnp.asarray(np.concatenate(all_mle, axis=0)),
        jnp.asarray(np.concatenate(all_F, axis=0)),
        jnp.asarray(np.concatenate(all_theta, axis=0)),
    )


# =============================================================================
# Main training function
# =============================================================================

def train_fishnets_dataset(
    train_dataset,
    test_dataset,
    n_params: Optional[int] = None,
    *,
    sample_to_xy_fn: Optional[Callable] = None,
    collate_fn: Optional[Callable] = None,
    model_apply_fn: Optional[Callable] = None,
    model_init_fn: Optional[Callable] = None,
    init_input: Any = None,
    hids_min: int = 10,
    hids_max: int = 300,
    n_layers: Union[int, Sequence[int]] = 3,
    num_models: int = 20,
    seed_model: int = 201,
    seed_train: int = 999,
    train_batch_size: int = 200,
    val_batch_size: Optional[int] = None,
    train_epochs: int = 4000,
    train_min_epochs: int = 100,
    patience: int = 20,
    lr: float = 5e-5,
    acts: Optional[list] = None,
    scaler_type: str = 'none',
    scaler_fit_max_samples: int = 5000,
    embedding_net: Optional[nn.Module] = None,
    outdir: str = "fishnets-log",
    update_pbar_every: int = 10,
    drop_last: bool = True,
):
    """Train an ensemble of fishnet networks from an iterator-style dataset.

    Parameters
    ----------
    train_dataset, test_dataset
        Indexable datasets. Each item should be convertible to ``(x, theta)``
        either by being a ``(x, theta)`` tuple, a dict with ``'x'``/``'y'``
        keys, or an object with a ``.y`` attribute (PyG-style). For arbitrary
        formats, supply a custom ``sample_to_xy_fn``.
    n_params
        Dimensionality of theta. If ``None`` it is inferred from the first
        sample.
    sample_to_xy_fn
        Optional ``sample -> (x, theta)`` function used by the default
        ``collate_fn``. Ignored if a custom ``collate_fn`` is supplied.
    collate_fn
        ``list[sample] -> (x_batch, theta_batch)``. Default stacks tensor-like
        samples along axis 0. Override for graph data, ragged data, etc.
    model_apply_fn
        ``(model, w, x_batch) -> (mle_batch, F_batch)``. Default vmaps
        ``model.apply`` over the batch axis of a stacked array. Override when
        the model already accepts a batched input (e.g. a jraph GNN with a
        batched ``GraphsTuple``).
    model_init_fn
        ``(model, key, x_batch) -> w``. Default calls
        ``model.init(key, x_batch[0])`` (matches the default vmap-based apply).
        Override when your embedding network expects a batched input directly
        -- e.g. for graph data pass
        ``lambda model, key, x_batch: model.init(key, x_batch)``.
    init_input
        Explicit escape hatch: if provided, ``model.init(key, init_input)`` is
        called directly and both ``model_init_fn`` and ``collate_fn`` are
        bypassed for initialisation. If ``None``, the init input is built by
        feeding ``[train_dataset[0]]`` through ``collate_fn`` and passing the
        resulting ``x_batch`` to ``model_init_fn``.
    scaler_type
        ``'minmax'``, ``'standard'`` or ``'none'``. When not ``'none'`` we
        iterate up to ``scaler_fit_max_samples`` items of ``train_dataset`` and
        fit a scaler on the flattened ``x``. The scaler is only applicable when
        the default tensor collate is used; for graph data, leave as
        ``'none'`` and pre-normalise upstream.
    scaler_fit_max_samples
        Maximum number of training samples to pull when fitting the scaler.
    val_batch_size
        Batch size used when computing validation loss / predictions on the
        test set. Defaults to ``train_batch_size``.
    drop_last
        Whether to drop a partial trailing batch during training. Validation
        always uses every sample.

    Returns
    -------
    ws, ensemble_weights, models, data_scaler, outputs
        ``data_scaler`` is ``None`` when ``scaler_type='none'``. ``outputs``
        follows the same FlexibleDict format as ``train_fishnets``.
    """
    # ---------------- output directory ----------------
    print("saving to", outdir)
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

    # ---------------- dataset prep ----------------
    train_dataset = _materialise_dataset(train_dataset)
    test_dataset = _materialise_dataset(test_dataset)
    n_train_samples = len(train_dataset)
    n_test_samples = len(test_dataset)
    if n_train_samples == 0:
        raise ValueError("train_dataset is empty.")
    if n_test_samples == 0:
        raise ValueError("test_dataset is empty.")

    if sample_to_xy_fn is None:
        sample_to_xy_fn = _default_sample_to_xy
    if collate_fn is None:
        collate_fn = make_default_collate(sample_to_xy_fn)
    if model_apply_fn is None:
        model_apply_fn = _default_model_apply
    if model_init_fn is None:
        model_init_fn = _default_model_init

    # Probe one sample to infer n_params.
    first_x, first_theta = sample_to_xy_fn(train_dataset[0])
    first_theta_arr = _to_jnp(first_theta)
    if n_params is None:
        if first_theta_arr.ndim == 0:
            n_params = 1
        else:
            n_params = int(first_theta_arr.shape[-1])

    # ---------------- optional input scaling ----------------
    data_scaler = None
    scaler_type_l = scaler_type.lower()
    if scaler_type_l != 'none':
        if scaler_type_l == 'minmax':
            data_scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type_l == 'standard':
            data_scaler = StandardScaler()
        else:
            raise ValueError(
                f"Unknown scaler_type: '{scaler_type}'. "
                "Use 'minmax', 'standard' or 'none'."
            )
        n_fit = min(scaler_fit_max_samples, n_train_samples)
        print(f"Fitting {scaler_type_l} scaler on {n_fit} training samples...")
        feat = []
        for i in range(n_fit):
            x, _ = sample_to_xy_fn(train_dataset[i])
            arr = np.asarray(_to_jnp(x))
            feat.append(arr.reshape(-1, arr.shape[-1]))
        feat = np.concatenate(feat, axis=0)
        data_scaler.fit(feat)

        # Wrap collate_fn to apply the scaler to ``x_batch`` inline.
        _user_collate = collate_fn

        def _scaling_collate(samples):
            x_batch, theta_batch = _user_collate(samples)
            arr = np.asarray(x_batch)
            shape = arr.shape
            arr = data_scaler.transform(arr.reshape(-1, shape[-1])).reshape(shape)
            return jnp.asarray(arr), theta_batch

        collate_fn = _scaling_collate

    # ---------------- model setup ----------------
    key = jr.PRNGKey(seed_model)

    mish = lambda x: x * nn.tanh(nn.softplus(x))
    if acts is None:
        acts = [nn.relu, nn.relu, nn.relu,
                nn.leaky_relu, nn.leaky_relu, nn.leaky_relu, nn.leaky_relu,
                nn.swish, nn.swish, nn.swish, mish, mish,
                optimized_smooth_leaky, optimized_smooth_leaky,
                optimized_smooth_leaky,
                nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu, nn.gelu]

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

    all_n_hidden, all_sharpness, all_threshold = [], [], []
    for n in range(num_models):
        key, hidden_key = jr.split(key)
        hidden = int(jr.choice(hidden_key, hids_range, replace=True))
        if sample_n_layers:
            key, layers_key = jr.split(key)
            n_layers_model = int(jr.randint(layers_key, shape=(), minval=min_layers, maxval=max_layers + 1))
        else:
            n_layers_model = fixed_layers
        print("Chosen hidden size for model", n + 1, ":", hidden, "| layers:", n_layers_model)
        all_n_hidden.append([hidden] * n_layers_model)

        key, rng1, rng2 = jr.split(key, 3)
        all_sharpness.append(jr.normal(rng1, shape=(1,)) * 0.7 + 5.0)
        all_threshold.append(jr.normal(rng2, shape=(1,)) * 0.7 + 1.0)

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

    keys = jr.split(key, num=num_models)
    if init_input is not None:
        # Explicit override: call model.init directly on the user-supplied input.
        ws = [models[i].init(keys[i], init_input) for i in range(num_models)]
    else:
        # Route through the user's collate_fn so any torch->jax conversion,
        # graph batching, or input scaling that happens at training time is
        # also applied at init time.
        x_batch_init, _ = collate_fn([train_dataset[0]])
        ws = [
            model_init_fn(models[i], keys[i], x_batch_init)
            for i in range(num_models)
        ]

    # ---------------- training loop ----------------
    train_batch = train_batch_size
    val_batch = val_batch_size if val_batch_size is not None else train_batch
    _pbar_stride = max(1, int(update_pbar_every))

    def training_loop(key, model, w):
        def kl_loss(w, x_batch, theta_batch):
            mle, F = model_apply_fn(model, w, x_batch)
            res = theta_batch - mle
            sign, logdet = jnp.linalg.slogdet(F)
            logdet = jnp.clip(logdet, -50, 50)
            return 0.5 * jnp.mean(
                jnp.einsum('ij,ij->i', res, jnp.einsum('ijk,ik->ij', F, res)) - logdet,
                axis=0,
            )

        tx = optax.adam(learning_rate=lr)
        opt_state = tx.init(w)
        loss_grad_fn = jax.value_and_grad(kl_loss)

        @jax.jit
        def train_step(w, opt_state, x_batch, theta_batch):
            loss, grads = loss_grad_fn(w, x_batch, theta_batch)
            updates, opt_state = tx.update(grads, opt_state, w)
            w = optax.apply_updates(w, updates)
            return w, opt_state, loss

        @jax.jit
        def eval_loss(w, x_batch, theta_batch):
            return kl_loss(w, x_batch, theta_batch)

        losses = []
        val_losses = []
        best_loss = jnp.inf
        best_w = w
        patience_counter = 0

        pbar = tqdm(
            range(train_epochs),
            desc="Training Epochs",
            leave=True,
            position=0,
            miniters=_pbar_stride,
        )

        for j in pbar:
            key, rng = jr.split(key)
            perm = np.asarray(jr.permutation(rng, jnp.arange(n_train_samples)))
            epoch_loss = 0.0
            n_batches = 0
            for x_batch, theta_batch in _iter_batches(
                train_dataset, perm, train_batch, collate_fn, drop_last=drop_last
            ):
                w, opt_state, loss = train_step(w, opt_state, x_batch, theta_batch)
                epoch_loss = epoch_loss + loss
                n_batches += 1
            if n_batches == 0:
                raise RuntimeError(
                    "Training produced 0 batches. "
                    "Check `train_batch_size` vs dataset size."
                )
            epoch_loss = float(epoch_loss / n_batches)
            losses.append(epoch_loss)

            # Validation: average loss across the entire test set.
            val_indices = np.arange(n_test_samples)
            v_total = 0.0
            v_count = 0
            for x_batch, theta_batch in _iter_batches(
                test_dataset, val_indices, val_batch, collate_fn, drop_last=False
            ):
                v = eval_loss(w, x_batch, theta_batch)
                bs = int(x_batch.shape[0]) if hasattr(x_batch, 'shape') else int(theta_batch.shape[0])
                v_total = v_total + float(v) * bs
                v_count += bs
            val_loss = v_total / max(v_count, 1)
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_w = w
                patience_counter = 0
            else:
                patience_counter += 1

            will_stop = (patience_counter - train_min_epochs > patience) and (
                j + 1 > train_min_epochs
            )
            if (j + 1) % _pbar_stride == 0 or j == train_epochs - 1 or will_stop:
                pbar.set_description(
                    "Epoch %d loss: %.5f ; val_loss: %.5f"
                    % (j, epoch_loss, val_loss)
                )
            if will_stop:
                print("\nEarly stopping triggered at epoch %d" % j)
                break

        return jnp.asarray(losses), jnp.asarray(val_losses), best_loss, best_w

    # ---------------- train ensemble ----------------
    print("STARTING TRAINING LOOP")
    all_losses, all_val_losses, best_val_losses = [], [], []
    train_keys = jr.split(jr.PRNGKey(seed_train), num=num_models)

    for i, w in enumerate(ws):
        print("\nTraining model %d of %d" % (i + 1, num_models))
        loss, val_loss, best_val_loss, wtrained = training_loop(
            train_keys[i], models[i], w
        )
        all_losses.append(loss)
        all_val_losses.append(val_loss)
        best_val_losses.append(best_val_loss)
        ws[i] = wtrained

    ensemble_weights_arr = jnp.array(
        [1.0 / jnp.exp(best_val_losses[i]) for i in range(num_models)]
    )
    print("Ensemble weights:", ensemble_weights_arr)

    # ---------------- predict on test set ----------------
    test_mles, test_Fs, test_thetas, test_xs = [], [], [], None
    for i in range(num_models):
        mle, F, theta_stacked = _predict_dataset(
            models[i], ws[i], test_dataset, val_batch, collate_fn, model_apply_fn
        )
        test_mles.append(mle)
        test_Fs.append(F)
        if i == 0:
            test_thetas = theta_stacked
            # Try to also stack inputs for the ``x`` field of the outputs dict;
            # fall back to ``None`` when collated x is not a plain array.
            try:
                xs = []
                for x_batch, _ in _iter_batches(
                    test_dataset,
                    np.arange(n_test_samples),
                    val_batch,
                    collate_fn,
                    drop_last=False,
                ):
                    xs.append(np.asarray(x_batch))
                test_xs = jnp.asarray(np.concatenate(xs, axis=0))
            except Exception:
                test_xs = None

    ensemble_F_predictions = jnp.stack(test_Fs, axis=0)
    ensemble_mle_predictions = jnp.stack(test_mles, axis=0)

    out_kwargs = dict(
        theta=test_thetas,
        Fs=ensemble_F_predictions,
        mle=ensemble_mle_predictions,
        ensemble_weights=ensemble_weights_arr,
    )
    if test_xs is not None:
        out_kwargs['x'] = test_xs
    outputs = create_results_dict(**out_kwargs)

    outname = os.path.join(outdir, "fishnets_outputs")
    np.savez(outname, **dict(outputs))
    print("Training completed. Outputs saved to:", outname + ".npz")
    print("Note: Load with io_utils.load_fishnets_results(file) for alias support")

    return ws, ensemble_weights_arr, models, data_scaler, outputs


# =============================================================================
# Example usage
# =============================================================================

if __name__ == '__main__':
    # Tiny end-to-end smoke test using a list-of-tuples "dataset" so we don't
    # need to install pyg / torch just to exercise the code path.
    import jax.random as jr

    n_samples = 2000
    n_test = 500
    n_params = 2
    n_d = 50

    @jax.jit
    def simulator(key, theta):
        return theta[0] + jr.normal(key, shape=(n_d,)) * jnp.sqrt(theta[1])

    key = jr.PRNGKey(0)
    k1, k2, k3, k4 = jr.split(key, 4)
    theta_train = jnp.stack(
        [
            jr.uniform(k1, (n_samples,), minval=-5.0, maxval=5.0),
            jr.uniform(k2, (n_samples,), minval=0.2, maxval=20.0),
        ],
        axis=-1,
    )
    theta_test = jnp.stack(
        [
            jr.uniform(k3, (n_test,), minval=-5.0, maxval=5.0),
            jr.uniform(k4, (n_test,), minval=0.2, maxval=20.0),
        ],
        axis=-1,
    )
    keys = jr.split(key, num=n_samples)
    x_train = jax.vmap(simulator)(keys, theta_train)
    keys = jr.split(k4, num=n_test)
    x_test = jax.vmap(simulator)(keys, theta_test)

    train_ds = [(x_train[i], theta_train[i]) for i in range(n_samples)]
    test_ds = [(x_test[i], theta_test[i]) for i in range(n_test)]

    ws, ens_weights, models, scaler, outputs = train_fishnets_dataset(
        train_ds,
        test_ds,
        n_params=n_params,
        hids_min=10,
        hids_max=64,
        n_layers=3,
        num_models=2,
        train_batch_size=200,
        train_epochs=10,
        train_min_epochs=2,
        patience=5,
        lr=5e-4,
        scaler_type='minmax',
        outdir="fishnets-log-dataset",
    )
    print("\nFishers shape:", outputs['Fs'].shape)
    print("MLE shape:", outputs['mle'].shape)
