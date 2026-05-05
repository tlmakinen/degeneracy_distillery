"""PyTorch port of
:mod:`degeneracy_distillery.training_loop_fishnets_dataset`.

This is a drop-in replacement for the JAX dataset training loop, intended for
``torch`` and ``torch_geometric`` (PyG) data pipelines. It keeps the same
public arguments (``hids_min/max``, ``n_layers``, ``num_models``,
``train_batch_size``, ``train_epochs``, ``patience``, ``lr``, ``acts``,
``embedding_net``, ``outdir``, ``update_pbar_every``, ...) so callers can
swap the backend with minimal changes.

PyTorch-specific niceties:

  * ``embedding_net`` is taken to be a *prototype* nn.Module. We
    :func:`copy.deepcopy` it once per ensemble member so each network has
    independent weights, mirroring the per-member ``model.init`` calls in the
    Flax version. (Pass ``embedding_net_factory`` to construct fresh
    embedding networks instead.)
  * Network heads (``resMLP`` and ``Fishnet_from_embedding``) use
    :class:`torch.nn.LazyLinear` for their first projection so the embedding
    dimension does not need to be known at construction time. We trigger
    lazy-init by running a single forward on one batch before training.
  * ``model_apply_fn`` and ``model_init_fn`` are *PyTorch-flavoured* (no
    ``w`` argument):
        ``model_apply_fn(model, x_batch) -> (mle, F)``
        ``model_init_fn(model, x_batch) -> None``  (one forward to init lazy params)
  * The returned ``outputs`` FlexibleDict holds plain numpy arrays, so they
    can be moved to JAX with ``jax.numpy.asarray(outputs['Fs'])`` and fed into
    the JAX-only flattening loop.

Typical usage with a PyG dataset
--------------------------------

    import torch
    from torch_geometric.data import Batch

    train_ds, test_ds = MyPyGDataset(...), MyPyGDataset(...)

    def collate_fn(samples):
        # samples: list[pyg.data.Data]
        batch = Batch.from_data_list(samples)
        theta = torch.stack([s.y for s in samples], dim=0).float()
        return batch, theta

    ws, ens_w, models, _, outputs = train_fishnets_dataset(
        train_ds, test_ds,
        n_params=14,
        collate_fn=collate_fn,
        embedding_net=MyGAT(),       # produces (B, embed_dim) from a Batch
        device='cuda',
    )

    # Hand off to the JAX flattening loop:
    import jax.numpy as jnp
    Fs_jax  = jnp.asarray(outputs['Fs'])
    mle_jax = jnp.asarray(outputs['mle'])
"""

from __future__ import annotations

import copy
import os
import shutil
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Support both package import and direct script execution.
try:
    from .fishnets import (
        Fishnet_from_embedding,
        optimized_smooth_leaky,
        resMLP,
    )
except ImportError:  # pragma: no cover - fallback for ``python this_file.py``
    from fishnets import Fishnet_from_embedding, optimized_smooth_leaky, resMLP

try:
    from ..io_utils import create_results_dict
except ImportError:  # pragma: no cover
    try:
        from degeneracy_distillery.io_utils import create_results_dict
    except ImportError:
        from io_utils import create_results_dict


# =============================================================================
# Default helpers (work for tensor datasets; override for graph datasets)
# =============================================================================


def _default_sample_to_xy(sample: Any):
    """Extract ``(x, theta)`` from a single dataset sample.

    Supported formats out of the box:
      - tuple/list:        ``(x, theta)``
      - PyG-like object:   ``sample`` with attribute ``.y``
        (``x = sample``, ``theta = sample.y``)
      - dict-like:         ``{'x': x, 'y': theta}`` or
        ``{'x': x, 'theta': theta}``
    """
    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        return sample[0], sample[1]
    if isinstance(sample, dict):
        x = sample.get("x", sample.get("data"))
        theta = sample.get("y", sample.get("theta"))
        if x is None or theta is None:
            raise ValueError(
                "dict samples must contain 'x'/'data' and 'y'/'theta' keys."
            )
        return x, theta
    if hasattr(sample, "y"):
        return sample, sample.y
    raise ValueError(
        "Could not infer (x, theta) from dataset sample. "
        "Pass a custom `sample_to_xy_fn`."
    )


def _to_tensor(x) -> torch.Tensor:
    """Best-effort conversion to a CPU torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        # other framework tensors that quack like torch / jax
        try:
            return torch.as_tensor(np.asarray(x.detach().cpu().numpy()))
        except Exception:
            pass
    return torch.as_tensor(np.asarray(x))


def make_default_collate(sample_to_xy_fn: Callable) -> Callable:
    """Build a default collate function that stacks tensor-like samples."""

    def collate_fn(samples):
        xs, ths = zip(*[sample_to_xy_fn(s) for s in samples])
        x_batch = torch.stack([_to_tensor(x) for x in xs], dim=0)
        theta_batch = torch.stack([_to_tensor(t) for t in ths], dim=0)
        return x_batch, theta_batch

    return collate_fn


def _default_model_apply(model: nn.Module, x_batch: Any):
    """Default per-batch apply: just call the model.

    PyTorch ``nn.Linear`` already broadcasts over leading batch dims, so we
    don't need a ``vmap``. Override when your embedding network needs more
    careful handling -- for example, to unpack a custom container before
    passing it through the network.
    """
    return model(x_batch)


def _default_model_init(model: nn.Module, x_batch: Any) -> None:
    """Default lazy-param initialiser: one forward pass under ``no_grad``.

    Required because ``Fishnet_from_embedding`` and ``resMLP`` use
    :class:`torch.nn.LazyLinear` for their first projection. After this call,
    those layers are materialised regular ``nn.Linear``s.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(x_batch)
    if was_training:
        model.train()


# =============================================================================
# Loss functions
# =============================================================================


def default_kl_loss(
    mle: torch.Tensor,
    F_mat: torch.Tensor,
    theta_batch: torch.Tensor,
    x_batch: Any,
) -> torch.Tensor:
    """Default Gaussian-KL loss used by :func:`train_fishnets_dataset`.

    Parameters
    ----------
    mle
        Predicted MLE, shape ``(B, n_params)``.
    F_mat
        Predicted Fisher information, shape ``(B, n_params, n_params)``.
    theta_batch
        True parameters, shape ``(B, n_params)``.
    x_batch
        Collated input batch. Unused by the default loss; kept in the signature
        so user-supplied losses can read fields such as ``x_batch.graph_mask``.

    Returns
    -------
    torch.Tensor
        Scalar loss with autograd attached.
    """
    del x_batch
    res = theta_batch - mle
    sign, logdet = torch.linalg.slogdet(F_mat)
    logdet = torch.clamp(logdet, -50.0, 50.0)
    quad = torch.einsum("ij,ij->i", res, torch.einsum("ijk,ik->ij", F_mat, res))
    return 0.5 * torch.mean(quad - logdet)


def masked_kl_loss(
    mle: torch.Tensor,
    F_mat: torch.Tensor,
    theta_batch: torch.Tensor,
    x_batch: Any,
) -> torch.Tensor:
    """Graph-padding-aware KL loss.

    Reads ``x_batch.graph_mask`` (a boolean / float tensor of shape ``(B,)``
    with ``True`` for real graphs) and computes the mean over real graphs
    only. Pad-graph contributions are zeroed out and the normaliser is
    ``sum(graph_mask)`` rather than ``B``.
    """
    mask = x_batch.graph_mask.to(mle.dtype)
    res = theta_batch - mle
    sign, logdet = torch.linalg.slogdet(F_mat)
    logdet = torch.clamp(logdet, -50.0, 50.0)
    quad = torch.einsum("ij,ij->i", res, torch.einsum("ijk,ik->ij", F_mat, res))
    per_graph = quad - logdet
    denom = torch.clamp(mask.sum(), min=1.0)
    return 0.5 * (per_graph * mask).sum() / denom


# =============================================================================
# Dataset utilities
# =============================================================================


def _is_dataloader(obj: Any) -> bool:
    """True for ``torch.utils.data.DataLoader`` and PyG's ``DataLoader``.

    Both expose a ``.dataset`` attribute and are iterable but not subscriptable.
    Detected duck-typed so we don't have to import torch_geometric here.
    """
    return (
        obj is not None
        and hasattr(obj, "dataset")
        and hasattr(obj, "__iter__")
        and not hasattr(obj, "__getitem__")
    )


def _underlying_dataset(source: Any) -> Any:
    """Return the underlying indexable dataset behind a source.

    For a DataLoader this is ``source.dataset``. For an already-indexable
    dataset (or anything else) we return it unchanged.
    """
    if _is_dataloader(source):
        return source.dataset
    return source


def _materialise_dataset(dataset):
    """Return a list-like wrapper that supports ``len`` and ``__getitem__``.

    If the input already supports both, returns it unchanged. Otherwise we
    consume the iterator into a list (which is required for shuffling).
    """
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        return dataset
    return list(dataset)


def _prefetch_iter(iterable, n_prefetch):
    """Overlap a Python producer with the consumer using a daemon thread.

    Buffers up to ``n_prefetch`` items. ``n_prefetch <= 0`` returns the
    iterable unchanged. Exceptions raised inside the worker are re-raised on
    the consumer side.
    """
    if n_prefetch is None or n_prefetch <= 0:
        yield from iterable
        return

    import queue
    import threading

    q: "queue.Queue" = queue.Queue(maxsize=int(n_prefetch))
    _SENTINEL = object()
    _ERROR = object()

    def _worker():
        try:
            for item in iterable:
                q.put(item)
        except BaseException as exc:  # pragma: no cover
            q.put((_ERROR, exc))
            return
        q.put(_SENTINEL)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is _SENTINEL:
            return
        if isinstance(item, tuple) and len(item) == 2 and item[0] is _ERROR:
            raise item[1]
        yield item


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


def _move_to_device(obj: Any, device: Optional[torch.device]) -> Any:
    """Recursively move a (nested) batched container onto ``device``.

    Handles tensors, PyG ``Data`` / ``Batch`` (anything with ``.to``), tuples,
    lists and dicts. Other objects are returned unchanged. ``device=None`` is
    a no-op.
    """
    if device is None or obj is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, tuple):
        return type(obj)(_move_to_device(o, device) for o in obj)
    if isinstance(obj, list):
        return [_move_to_device(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    to = getattr(obj, "to", None)
    if callable(to):
        try:
            return obj.to(device)
        except Exception:
            return obj
    return obj


def _infer_batch_size(x_batch: Any, theta_batch: torch.Tensor) -> int:
    """Infer the per-graph (or per-sample) batch size from a collated input."""
    if isinstance(x_batch, torch.Tensor):
        return int(x_batch.shape[0])
    # PyG Batch
    if hasattr(x_batch, "num_graphs"):
        try:
            return int(x_batch.num_graphs)
        except Exception:
            pass
    if hasattr(x_batch, "shape"):
        try:
            return int(x_batch.shape[0])
        except (TypeError, IndexError):
            pass
    return int(theta_batch.shape[0])


# =============================================================================
# Prediction helper
# =============================================================================


def _autocast_apply(
    model_apply_fn: Callable,
    model: nn.Module,
    x_batch: Any,
    amp_dtype: Optional[torch.dtype],
    device_type: str,
):
    """Run ``model_apply_fn`` under an optional autocast region.

    When ``amp_dtype`` is not ``None`` the forward runs in mixed precision
    (e.g. ``torch.bfloat16`` on Ampere/Ada/L4 GPUs) and the returned
    ``(mle, F)`` are cast back to fp32 so downstream ops (notably
    ``torch.linalg.slogdet`` in :func:`default_kl_loss`) stay well-conditioned.
    When ``amp_dtype`` is ``None`` this is a thin pass-through.
    """
    if amp_dtype is None:
        return model_apply_fn(model, x_batch)
    with torch.autocast(device_type=device_type, dtype=amp_dtype):
        mle, F_mat = model_apply_fn(model, x_batch)
    return mle.float(), F_mat.float()


def _predict_dataset(
    model: nn.Module,
    source,
    dataset,
    batch_size: int,
    collate_fn: Callable,
    model_apply_fn: Callable,
    device: Optional[torch.device],
    prefetch_batches: int = 0,
    scale_iter: Optional[Callable] = None,
    amp_dtype: Optional[torch.dtype] = None,
):
    """Run ``model`` over an entire dataset in batches and stack outputs.

    ``source`` may be either a ``DataLoader`` (in which case it is iterated
    directly) or an indexable ``dataset`` (in which case ``collate_fn`` and
    ``batch_size`` are used). The user-facing ordering of test predictions
    follows whatever the source produces; for reproducible test indexing,
    pass a DataLoader with ``shuffle=False`` or an indexable dataset.

    Returns ``(mle, F, theta)`` as numpy arrays stacked along the leading
    axis. Always runs under ``torch.no_grad`` in ``eval`` mode. When
    ``amp_dtype`` is provided the forward runs under
    :func:`torch.autocast` and outputs are cast back to fp32 before being
    materialised on the host.
    """
    model.eval()
    device_type = device.type if device is not None else "cpu"
    all_mle, all_F, all_theta = [], [], []
    with torch.no_grad():
        if _is_dataloader(source):
            base = iter(source)
        else:
            base = _iter_batches(
                dataset, np.arange(len(dataset)), batch_size, collate_fn,
                drop_last=False,
            )
        if scale_iter is not None:
            base = scale_iter(base)
        for x_batch, theta_batch in _prefetch_iter(base, prefetch_batches):
            x_batch = _move_to_device(x_batch, device)
            theta_batch_dev = _move_to_device(theta_batch, device)
            mle, F_mat = _autocast_apply(
                model_apply_fn, model, x_batch, amp_dtype, device_type,
            )
            all_mle.append(mle.detach().cpu().numpy())
            all_F.append(F_mat.detach().cpu().numpy())
            # Use the original theta (CPU is fine; we asked the user not to
            # rely on a particular device for their target tensor).
            all_theta.append(
                theta_batch_dev.detach().cpu().numpy()
                if isinstance(theta_batch_dev, torch.Tensor)
                else np.asarray(theta_batch)
            )
    return (
        np.concatenate(all_mle, axis=0),
        np.concatenate(all_F, axis=0),
        np.concatenate(all_theta, axis=0),
    )


# =============================================================================
# Activation roster (mirrors the Flax version's defaults)
# =============================================================================


def _mish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(F.softplus(x))


def _default_acts():
    return [
        F.relu, F.relu, F.relu,
        F.leaky_relu, F.leaky_relu, F.leaky_relu, F.leaky_relu,
        F.silu, F.silu, F.silu, _mish, _mish,
        optimized_smooth_leaky, optimized_smooth_leaky, optimized_smooth_leaky,
        F.gelu, F.gelu, F.gelu, F.gelu, F.gelu, F.gelu, F.gelu, F.gelu,
    ]


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
    loss_fn: Optional[Callable] = None,
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
    scaler_type: str = "none",
    scaler_fit_max_samples: int = 5000,
    embedding_net: Optional[nn.Module] = None,
    embedding_net_factory: Optional[Callable[[], nn.Module]] = None,
    outdir: str = "fishnets-log",
    update_pbar_every: int = 10,
    drop_last: bool = True,
    prefetch_batches: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    amp_dtype: Optional[torch.dtype] = None,
):
    """Train an ensemble of fishnet networks from datasets or DataLoaders.

    Both ``train_dataset`` and ``test_dataset`` may be either:

      * an indexable dataset (anything with ``__len__`` and ``__getitem__``;
        in this case the function batches via the supplied ``collate_fn``);
      * a ``torch.utils.data.DataLoader`` (or ``torch_geometric.loader.
        DataLoader``); in this case batching, shuffling and collation are
        handled by the loader itself and the supplied ``collate_fn`` /
        ``train_batch_size`` / ``drop_last`` are ignored for that stream.
        The loader is iterated fresh once per epoch.

    See :func:`degeneracy_distillery.training_loop_fishnets_dataset.
    train_fishnets_dataset` for the JAX version's full argument docs; only the
    PyTorch-specific differences are highlighted below.

    Parameters (PyTorch-specific notes)
    -----------------------------------
    sample_to_xy_fn, collate_fn
        Same contract as the JAX version, but values flow through PyTorch
        tensors (and, for graph data, PyG ``Batch`` objects) instead of
        ``jnp.ndarray``. When you pass a ``DataLoader``, set its own
        ``collate_fn`` to produce ``(x_batch, theta_batch)`` tuples (matching
        the contract here); the ``collate_fn`` argument to this function is
        then only used for the lazy-init forward pass on a single
        ``dataset[0]`` sample, so it can be left as the default.
    model_apply_fn
        ``(model, x_batch) -> (mle, F)``. PyTorch ``nn.Module``s carry their
        own weights, so there is no separate ``w`` argument like in JAX.
        Default: ``model(x_batch)``.
    model_init_fn
        ``(model, x_batch) -> None``. Runs a single forward to materialise any
        :class:`torch.nn.LazyLinear` layers. Default does this under
        ``torch.no_grad`` in ``eval`` mode.
    loss_fn
        ``(mle, F, theta_batch, x_batch) -> scalar tensor``. Default is
        :func:`default_kl_loss`.
    embedding_net
        Prototype embedding ``nn.Module``. We :func:`copy.deepcopy` it once
        per ensemble member so each network has independent weights. Pass
        ``embedding_net_factory`` (zero-arg callable returning a fresh
        ``nn.Module``) for full independent re-initialisation per member.
    embedding_net_factory
        Optional. If provided, takes precedence over ``embedding_net`` and is
        called once per ensemble member to build a fresh embedding network.
    device
        ``'cuda'`` / ``'cpu'`` / ``torch.device``. Default ``None`` selects
        ``cuda`` if available, else ``cpu``.
    amp_dtype
        Optional :class:`torch.dtype` to enable mixed-precision forward
        passes via :func:`torch.autocast`. Set to ``torch.bfloat16`` on
        Ampere/Ada/L4 GPUs for a typical 1.5--2.5x speedup; the
        ``(mle, F)`` outputs are cast back to fp32 before the loss so
        ``torch.linalg.slogdet`` stays well-conditioned. Default ``None``
        (autocast disabled). bf16 does not require a ``GradScaler``;
        for fp16 you would need to add one yourself via a custom
        ``model_apply_fn`` / ``loss_fn`` combo.
    drop_last, prefetch_batches
        Same semantics as the JAX version.
    init_input
        Explicit override. If provided, ``model_init_fn(model, init_input)``
        is called directly instead of routing the first batch through
        ``collate_fn`` and ``model_init_fn``.

    Returns
    -------
    state_dicts, ensemble_weights, models, data_scaler, outputs
        ``state_dicts`` is a list of ``model.state_dict()`` mappings (parallel
        to ``ws`` in the JAX version). ``models`` are the trained
        ``nn.Module``s on the chosen device, with their best-validation-loss
        weights loaded. ``data_scaler`` is ``None`` when ``scaler_type='none'``.
        ``outputs`` is a FlexibleDict of numpy arrays.
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

    # ---------------- device ----------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # ---------------- dataset prep ----------------
    # We accept either DataLoader-style sources (iterable, with .dataset) or
    # indexable datasets. ``*_source`` is what we *iterate* per epoch;
    # ``*_dataset`` is the underlying indexable thing used for one-off probes
    # (sample inspection, scaler fitting, lazy-init forward).
    train_source = train_dataset
    test_source = test_dataset
    train_dataset = _underlying_dataset(train_source)
    test_dataset = _underlying_dataset(test_source)
    if not _is_dataloader(train_source):
        train_dataset = _materialise_dataset(train_dataset)
    if not _is_dataloader(test_source):
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
    if loss_fn is None:
        loss_fn = default_kl_loss

    # Probe one sample to infer n_params.
    first_x, first_theta = sample_to_xy_fn(train_dataset[0])
    first_theta_arr = _to_tensor(first_theta)
    if n_params is None:
        if first_theta_arr.ndim == 0:
            n_params = 1
        else:
            n_params = int(first_theta_arr.shape[-1])

    # ---------------- optional input scaling ----------------
    data_scaler = None
    scaler_type_l = scaler_type.lower()
    if scaler_type_l != "none":
        if scaler_type_l == "minmax":
            data_scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type_l == "standard":
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
            arr = _to_tensor(x).cpu().numpy()
            feat.append(arr.reshape(-1, arr.shape[-1]))
        feat = np.concatenate(feat, axis=0)
        data_scaler.fit(feat)

    # Iteration-time scaler: applied to whatever (x_batch, theta_batch) tuples
    # come out of the source (indexable or DataLoader). For graph batches we
    # raise -- pre-normalise upstream and leave ``scaler_type='none'``.
    def _scale_iter(iterable):
        if data_scaler is None:
            yield from iterable
            return
        for x_batch, theta_batch in iterable:
            if not isinstance(x_batch, torch.Tensor):
                raise TypeError(
                    "scaler_type != 'none' is only supported when the source "
                    "yields a torch.Tensor x_batch. For graph data, "
                    "pre-normalise upstream and leave scaler_type='none'."
                )
            arr = x_batch.detach().cpu().numpy()
            shape = arr.shape
            arr = data_scaler.transform(arr.reshape(-1, shape[-1])).reshape(shape)
            yield torch.as_tensor(arr, dtype=x_batch.dtype), theta_batch

    # ---------------- model setup ----------------
    rng = np.random.default_rng(seed_model)

    if acts is None:
        acts = _default_acts()
    idx_acts = rng.choice(np.arange(len(acts)), size=(num_models,))
    acts = [acts[i] for i in idx_acts]

    hids_range = np.arange(hids_min, hids_max)
    if isinstance(n_layers, (list, tuple, np.ndarray)):
        if len(n_layers) != 2:
            raise ValueError(
                "n_layers range must have exactly two values: [min_layers, max_layers]."
            )
        min_layers, max_layers = int(n_layers[0]), int(n_layers[1])
        if min_layers <= 0 or max_layers <= 0:
            raise ValueError("n_layers values must be positive.")
        if min_layers > max_layers:
            raise ValueError(
                "n_layers range must satisfy min_layers <= max_layers."
            )
        sample_n_layers = True
    else:
        fixed_layers = int(n_layers)
        if fixed_layers <= 0:
            raise ValueError("n_layers must be a positive integer.")
        sample_n_layers = False

    all_n_hidden, all_sharpness, all_threshold = [], [], []
    for n in range(num_models):
        hidden = int(rng.choice(hids_range))
        if sample_n_layers:
            n_layers_model = int(rng.integers(min_layers, max_layers + 1))
        else:
            n_layers_model = fixed_layers
        print(
            "Chosen hidden size for model", n + 1, ":", hidden,
            "| layers:", n_layers_model,
        )
        all_n_hidden.append([hidden] * n_layers_model)
        all_sharpness.append(float(rng.normal() * 0.7 + 5.0))
        all_threshold.append(float(rng.normal() * 0.7 + 1.0))

    # Build embedding network instances per ensemble member.
    if embedding_net_factory is not None:
        embedding_nets = [embedding_net_factory() for _ in range(num_models)]
    elif embedding_net is not None:
        embedding_nets = [copy.deepcopy(embedding_net) for _ in range(num_models)]
    else:
        embedding_nets = [None] * num_models

    torch.manual_seed(int(seed_model))

    def _make_model(i: int) -> nn.Module:
        head = nn.Sequential(
            resMLP(all_n_hidden[i], act=acts[i]),
            Fishnet_from_embedding(
                n_p=n_params,
                act=acts[i],
                hidden=all_n_hidden[i][0],
                act_fisher=F.gelu,
                sharpness=all_sharpness[i],
                threshold=all_threshold[i],
            ),
        )
        if embedding_nets[i] is not None:
            return nn.Sequential(embedding_nets[i], head)
        return head

    models = [_make_model(i) for i in range(num_models)]

    # ---------------- lazy-init forward pass ----------------
    if init_input is not None:
        for m in models:
            model_init_fn(m, init_input)
    else:
        # For DataLoader sources, pull a real first batch so the init mirrors
        # exactly the shapes / containers the user's loader produces. For
        # indexable sources, route a single sample through the user's
        # collate_fn so any torch->device conversion or graph batching is
        # also exercised at init time.
        if _is_dataloader(train_source):
            x_batch_init, _ = next(iter(train_source))
        else:
            x_batch_init, _ = collate_fn([train_dataset[0]])
        for m in models:
            model_init_fn(m, x_batch_init)

    # Move models onto the target device after lazy init has materialised
    # all parameters.
    for m in models:
        m.to(device)

    # ---------------- training loop ----------------
    train_batch = train_batch_size
    val_batch = val_batch_size if val_batch_size is not None else train_batch
    _pbar_stride = max(1, int(update_pbar_every))
    _device_type = device.type
    if amp_dtype is not None:
        print(
            f"Mixed precision: forward passes will run under "
            f"torch.autocast(device_type={_device_type!r}, "
            f"dtype={amp_dtype}); outputs cast back to fp32 for the loss."
        )

    def _train_iter(rng_local):
        """Yield one epoch of training batches (DataLoader- or dataset-aware)."""
        if _is_dataloader(train_source):
            # The loader handles its own shuffling/batching/collation; we
            # just walk it once per epoch.
            base = iter(train_source)
        else:
            perm = rng_local.permutation(n_train_samples)
            base = _iter_batches(
                train_dataset, perm, train_batch, collate_fn,
                drop_last=drop_last,
            )
        return _prefetch_iter(_scale_iter(base), prefetch_batches)

    def _val_iter():
        """Yield one pass of validation batches in deterministic order."""
        if _is_dataloader(test_source):
            base = iter(test_source)
        else:
            base = _iter_batches(
                test_dataset, np.arange(n_test_samples), val_batch,
                collate_fn, drop_last=False,
            )
        return _prefetch_iter(_scale_iter(base), prefetch_batches)

    def training_loop(seed: int, model: nn.Module):
        rng_local = np.random.default_rng(int(seed))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses, val_losses = [], []
        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0

        pbar = tqdm(
            range(train_epochs),
            desc="Training Epochs",
            leave=True,
            position=0,
            miniters=_pbar_stride,
        )

        for j in pbar:
            # ---- train pass ----
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for x_batch, theta_batch in _train_iter(rng_local):
                x_batch = _move_to_device(x_batch, device)
                theta_batch = _move_to_device(theta_batch, device)
                optimizer.zero_grad(set_to_none=True)
                mle, F_mat = _autocast_apply(
                    model_apply_fn, model, x_batch, amp_dtype, _device_type,
                )
                loss_t = loss_fn(mle, F_mat, theta_batch, x_batch)
                loss_t.backward()
                optimizer.step()
                epoch_loss += float(loss_t.detach())
                n_batches += 1
            if n_batches == 0:
                raise RuntimeError(
                    "Training produced 0 batches. "
                    "Check `train_batch_size` vs dataset size."
                )
            epoch_loss /= n_batches
            losses.append(epoch_loss)

            # ---- validation pass ----
            model.eval()
            v_total = 0.0
            v_count = 0
            with torch.no_grad():
                for x_batch, theta_batch in _val_iter():
                    x_batch = _move_to_device(x_batch, device)
                    theta_batch = _move_to_device(theta_batch, device)
                    mle, F_mat = _autocast_apply(
                        model_apply_fn, model, x_batch, amp_dtype, _device_type,
                    )
                    v = loss_fn(mle, F_mat, theta_batch, x_batch)
                    bs = _infer_batch_size(x_batch, theta_batch)
                    v_total += float(v.detach()) * bs
                    v_count += bs
            val_loss = v_total / max(v_count, 1)
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
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

        # Restore best weights.
        model.load_state_dict(best_state)
        return (
            np.asarray(losses),
            np.asarray(val_losses),
            float(best_loss),
        )

    # ---------------- train ensemble ----------------
    print("STARTING TRAINING LOOP")
    all_losses, all_val_losses, best_val_losses = [], [], []
    train_seed_rng = np.random.default_rng(int(seed_train))
    train_seeds = [int(s) for s in train_seed_rng.integers(0, 2**31 - 1, size=num_models)]

    for i in range(num_models):
        print("\nTraining model %d of %d" % (i + 1, num_models))
        loss, val_loss, best_val_loss = training_loop(train_seeds[i], models[i])
        all_losses.append(loss)
        all_val_losses.append(val_loss)
        best_val_losses.append(best_val_loss)

    ensemble_weights_arr = np.asarray(
        [1.0 / np.exp(best_val_losses[i]) for i in range(num_models)]
    )
    print("Ensemble weights:", ensemble_weights_arr)

    # ---------------- predict on test set ----------------
    test_mles, test_Fs, test_thetas, test_xs = [], [], [], None
    for i in range(num_models):
        mle, F_mat, theta_stacked = _predict_dataset(
            models[i], test_source, test_dataset, val_batch, collate_fn,
            model_apply_fn, device=device, prefetch_batches=prefetch_batches,
            scale_iter=_scale_iter, amp_dtype=amp_dtype,
        )
        test_mles.append(mle)
        test_Fs.append(F_mat)
        if i == 0:
            test_thetas = theta_stacked
            # Try to also stack inputs for the ``x`` field of the outputs dict;
            # fall back to ``None`` when collated x is not a plain tensor
            # (e.g. a PyG Batch).
            try:
                xs = []
                if _is_dataloader(test_source):
                    base = iter(test_source)
                else:
                    base = _iter_batches(
                        test_dataset, np.arange(n_test_samples), val_batch,
                        collate_fn, drop_last=False,
                    )
                for x_batch, _ in _scale_iter(base):
                    if not isinstance(x_batch, torch.Tensor):
                        raise TypeError("x_batch is not a torch.Tensor")
                    xs.append(x_batch.cpu().numpy())
                test_xs = np.concatenate(xs, axis=0)
            except Exception:
                test_xs = None

    ensemble_F_predictions = np.stack(test_Fs, axis=0)
    ensemble_mle_predictions = np.stack(test_mles, axis=0)

    out_kwargs = dict(
        theta=test_thetas,
        Fs=ensemble_F_predictions,
        mle=ensemble_mle_predictions,
        ensemble_weights=ensemble_weights_arr,
    )
    if test_xs is not None:
        out_kwargs["x"] = test_xs
    outputs = create_results_dict(**out_kwargs)

    outname = os.path.join(outdir, "fishnets_outputs")
    np.savez(outname, **dict(outputs))
    print("Training completed. Outputs saved to:", outname + ".npz")
    print("Note: Load with io_utils.load_fishnets_results(file) for alias support")

    state_dicts = [m.state_dict() for m in models]
    return state_dicts, ensemble_weights_arr, models, data_scaler, outputs


# =============================================================================
# Example usage
# =============================================================================


if __name__ == "__main__":  # pragma: no cover
    # Tiny end-to-end smoke test using a list-of-tuples "dataset" so we don't
    # need to install pyg just to exercise the code path.
    n_samples = 2000
    n_test = 500
    n_params = 2
    n_d = 50

    rng = np.random.default_rng(0)

    def simulator(theta_row):
        return theta_row[0] + rng.standard_normal(n_d) * np.sqrt(theta_row[1])

    theta_train = np.stack(
        [
            rng.uniform(-5.0, 5.0, size=n_samples),
            rng.uniform(0.2, 20.0, size=n_samples),
        ],
        axis=-1,
    ).astype(np.float32)
    theta_test = np.stack(
        [
            rng.uniform(-5.0, 5.0, size=n_test),
            rng.uniform(0.2, 20.0, size=n_test),
        ],
        axis=-1,
    ).astype(np.float32)
    x_train = np.stack([simulator(t) for t in theta_train], axis=0).astype(np.float32)
    x_test = np.stack([simulator(t) for t in theta_test], axis=0).astype(np.float32)

    train_ds = [
        (torch.as_tensor(x_train[i]), torch.as_tensor(theta_train[i]))
        for i in range(n_samples)
    ]
    test_ds = [
        (torch.as_tensor(x_test[i]), torch.as_tensor(theta_test[i]))
        for i in range(n_test)
    ]

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
        scaler_type="minmax",
        outdir="fishnets-log-dataset-torch",
    )
    print("\nFishers shape:", outputs["Fs"].shape)
    print("MLE shape:", outputs["mle"].shape)
