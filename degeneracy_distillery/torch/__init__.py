"""PyTorch (and PyG)-friendly versions of the fishnets head and the
dataset-based training loop.

These modules mirror the public interface of
:mod:`degeneracy_distillery.fishnets` and
:mod:`degeneracy_distillery.training_loop_fishnets_dataset` but are written in
PyTorch so they can be plugged into ``torch`` / ``torch_geometric`` data
pipelines without going through JAX.

After training, the contents of the returned ``outputs`` dict are plain numpy
arrays, so they can be moved to JAX with ``jax.numpy.asarray(...)`` and
fed straight into ``training_loop_flatten`` (which is JAX-only).
"""

from .fishnets import (
    SoftSquelch,
    SparseActivation,
    smooth_leaky,
    optimized_smooth_leaky,
    safe_for_grad_log,
    shifted_softplus,
    fill_lower_tri,
    fill_diagonal,
    construct_fisher_matrix_log_cholesky,
    construct_fisher_matrix_single,
    construct_fisher_matrix_multiple,
    MLP,
    resMLP,
    Fishnet_from_embedding,
)

from .training_loop_fishnets_dataset import (
    train_fishnets_dataset,
    default_kl_loss,
    masked_kl_loss,
    make_default_collate,
)

__all__ = [
    "SoftSquelch",
    "SparseActivation",
    "smooth_leaky",
    "optimized_smooth_leaky",
    "safe_for_grad_log",
    "shifted_softplus",
    "fill_lower_tri",
    "fill_diagonal",
    "construct_fisher_matrix_log_cholesky",
    "construct_fisher_matrix_single",
    "construct_fisher_matrix_multiple",
    "MLP",
    "resMLP",
    "Fishnet_from_embedding",
    "train_fishnets_dataset",
    "default_kl_loss",
    "masked_kl_loss",
    "make_default_collate",
]
