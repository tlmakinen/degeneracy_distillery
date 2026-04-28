"""PyTorch port of :mod:`degeneracy_distillery.fishnets`.

The public API mirrors the JAX/Flax version as closely as possible:

  - :class:`SoftSquelch`, :class:`SparseActivation`
  - :func:`smooth_leaky`, :func:`optimized_smooth_leaky`,
    :func:`safe_for_grad_log`, :func:`shifted_softplus`
  - :func:`fill_lower_tri`, :func:`fill_diagonal`
  - :func:`construct_fisher_matrix_log_cholesky`,
    :func:`construct_fisher_matrix_single`,
    :func:`construct_fisher_matrix_multiple`
  - :class:`MLP`, :class:`resMLP`, :class:`Fishnet_from_embedding`

All modules operate over a leading batch dimension, which is the natural
convention for PyTorch (and PyTorch Geometric) workflows. There is no need
to ``vmap`` over the batch axis: ``nn.Linear`` already broadcasts and the
Cholesky construction supports arbitrary leading batch dimensions.

Every ``nn.Module`` defined here that consumes an "embedding" accepts an
optional ``in_features`` constructor argument; pass it to bypass the
:class:`torch.nn.LazyLinear` placeholders if you know the upstream embedding
size at construction time.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Activations
# =============================================================================


class SoftSquelch(nn.Module):
    """Symmetric soft gate that suppresses values close to zero.

    Identical in behaviour to the Flax version:
        gate(x) = sigmoid(sharpness * (|x| - threshold))
        out(x)  = x * gate(x)

    ``threshold`` and ``sharpness`` are stored as plain (non-learnable) floats
    to match the JAX version. Pass scalars or 0/1-d tensors -- they are coerced
    to floats at construction time.
    """

    threshold: float
    sharpness: float

    def __init__(self, threshold: float = 1.0, sharpness: float = 5.0) -> None:
        super().__init__()
        self.threshold = float(_to_python_scalar(threshold))
        self.sharpness = float(_to_python_scalar(sharpness))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.sharpness * (torch.abs(x) - self.threshold))
        return x * gate


class SparseActivation(nn.Module):
    """Learnable soft-thresholding / squelch activation.

    Mirrors the Flax ``SparseActivation`` module: a learnable threshold
    (kept positive via softplus) and either a hard soft-threshold or a
    smooth squelch with a learnable sharpness.
    """

    def __init__(
        self,
        mode: str = "squelch",
        init_threshold: float = 0.5,
        init_sharpness: float = 10.0,
    ) -> None:
        super().__init__()
        if mode not in ("squelch", "threshold"):
            raise ValueError(f"Unknown SparseActivation mode: {mode!r}")
        self.mode = mode
        self.t_param = nn.Parameter(torch.full((1,), float(init_threshold)))
        if mode == "squelch":
            self.s_param = nn.Parameter(torch.full((1,), float(init_sharpness)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        threshold = F.softplus(self.t_param)
        if self.mode == "threshold":
            return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)
        sharpness = F.softplus(self.s_param)
        gate = torch.sigmoid(sharpness * (torch.abs(x) - threshold))
        return x * gate


def smooth_leaky(x: torch.Tensor) -> torch.Tensor:
    r"""Almost-leaky ReLU that's smooth across thresholds.

    Mirrors the JAX implementation:

    .. math::
        \mathrm{smooth\_leaky}(x) = \begin{cases}
            x, & x \le -1 \\
            -|x|^3 / 3 + x(x + 2) + 1/3, & -1 < x < 1 \\
            3x, & x \ge 1
        \end{cases}
    """
    out_left = x
    out_mid = -(torch.abs(x) ** 3) / 3.0 + x * (x + 2.0) + (1.0 / 3.0)
    out_right = 3.0 * x
    return torch.where(x < -1.0, out_left, torch.where(x < 1.0, out_mid, out_right))


def optimized_smooth_leaky(x: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
    r"""C-infinity smooth version of leaky ReLU.

    Combines linear growth with a smooth transition near the origin via the
    log-sigmoid. Identical to the JAX version:

    .. math::
        f(x) = \alpha\, x + (1 - \alpha)\, \log\sigma(x)\, x.
    """
    return alpha * x + (1.0 - alpha) * F.logsigmoid(x) * x


def safe_for_grad_log(x: torch.Tensor) -> torch.Tensor:
    """``log(x)`` if ``x > 0`` else ``log(1) = 0``, with a gradient-safe form."""
    return torch.log(torch.where(x > 0.0, x, torch.ones_like(x)))


def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    r""":math:`\ln(0.5 + 0.5\, e^{x})`."""
    return safe_for_grad_log(0.5 + 0.5 * torch.exp(x))


# =============================================================================
# Triangular helpers
# =============================================================================


def fill_lower_tri(v: torch.Tensor) -> torch.Tensor:
    """Build a lower-triangular matrix from its lower-triangular elements.

    Supports arbitrary leading batch dimensions: an input of shape
    ``(..., n*(n+1)/2)`` returns a tensor of shape ``(..., n, n)``.
    """
    m = v.shape[-1]
    dim = int(math.sqrt(0.25 + 2.0 * m) - 0.5)
    if dim * (dim + 1) // 2 != m:
        raise ValueError(
            f"fill_lower_tri: input last dim {m} is not a triangular number."
        )
    out = torch.zeros(*v.shape[:-1], dim, dim, dtype=v.dtype, device=v.device)
    idx = torch.tril_indices(dim, dim, device=v.device)
    out[..., idx[0], idx[1]] = v
    return out


def fill_diagonal(a: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    """Out-of-place fill of the (last 2 dims of) ``a``'s diagonal with ``val``.

    Returns a new tensor; the original is not modified. Supports leading
    batch dims, mirroring the JAX implementation.
    """
    n = a.shape[-1]
    out = a.clone()
    idx = torch.arange(n, device=a.device)
    out[..., idx, idx] = val
    return out


# =============================================================================
# Fisher matrix construction
# =============================================================================


def construct_fisher_matrix_log_cholesky(
    outputs: torch.Tensor, n_p: int
) -> torch.Tensor:
    """Construct ``F = L L^T`` from log-Cholesky parameters.

    The first ``n_p`` entries of the trailing axis are the (log) diagonal of
    the Cholesky factor (passed through ``softplus`` plus a small floor for
    positivity). The remaining ``n_p * (n_p - 1) / 2`` entries are the
    unconstrained strictly lower-triangular entries.

    Supports arbitrary leading batch dimensions: input of shape
    ``(..., n_p*(n_p+1)/2)`` returns ``(..., n_p, n_p)``. With a 1-D input
    this matches the original single-sample behaviour.
    """
    n_diag = n_p
    batch_shape = outputs.shape[:-1]

    log_diag = outputs[..., :n_diag]
    diag_elements = F.softplus(log_diag) + 1e-4

    off_diag = outputs[..., n_diag:]

    L = torch.zeros(
        *batch_shape, n_p, n_p, dtype=outputs.dtype, device=outputs.device
    )
    diag_idx = torch.arange(n_p, device=outputs.device)
    L[..., diag_idx, diag_idx] = diag_elements
    if n_p > 1:
        lower_idx = torch.tril_indices(n_p, n_p, offset=-1, device=outputs.device)
        L[..., lower_idx[0], lower_idx[1]] = off_diag

    return L @ L.transpose(-1, -2)


def construct_fisher_matrix_single(outputs: torch.Tensor) -> torch.Tensor:
    """Single-sample Fisher construction matching the original Flax helper.

    Operates on a 1-D ``outputs`` (or any leading batch dim plus a final
    triangular-numbered axis) and returns a (..., n, n) PSD matrix.
    """
    Q = fill_lower_tri(outputs)
    Q_lower = torch.tril(Q)
    middle = torch.diagonal(Q_lower - F.softplus(Q_lower), dim1=-2, dim2=-1)
    padding = torch.zeros_like(Q)
    L = Q - fill_diagonal(padding, middle)
    return L @ L.transpose(-1, -2)


def construct_fisher_matrix_multiple(outputs: torch.Tensor) -> torch.Tensor:
    """Batched form. Identical to :func:`construct_fisher_matrix_single` since
    the latter already supports leading batch dims in PyTorch -- exposed under
    this name purely for parity with the JAX module's API.
    """
    return construct_fisher_matrix_single(outputs)


# =============================================================================
# MLPs
# =============================================================================


class MLP(nn.Module):
    """Plain MLP. Mirrors the Flax ``MLP``.

    All hidden layers use ``act``; the final layer is linear. Pass
    ``in_features`` to avoid the lazy-init forward pass that
    :class:`torch.nn.LazyLinear` requires for the first layer.
    """

    def __init__(
        self,
        features: Sequence[int],
        in_features: Optional[int] = None,
        act: Callable[[torch.Tensor], torch.Tensor] = F.silu,
    ) -> None:
        super().__init__()
        if len(features) == 0:
            raise ValueError("MLP requires at least one feature size.")
        self.features = list(features)
        self.act = act

        layers = []
        prev = in_features
        for feat in self.features:
            if prev is None:
                layers.append(nn.LazyLinear(feat))
            else:
                layers.append(nn.Linear(prev, feat))
            prev = feat
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        return x


class resMLP(nn.Module):
    """Residual MLP, matching the Flax ``resMLP``.

    All hidden sizes ``features[:-1]`` must be equal so the residual
    connection ``x + z`` is well-defined; this matches the original
    implementation's contract (and the way the training loop builds it as
    ``[hidden] * n_layers``).

    The first projection is a :class:`torch.nn.LazyLinear` unless
    ``in_features`` is supplied; the remaining layers are eagerly sized.
    """

    def __init__(
        self,
        features: Sequence[int],
        in_features: Optional[int] = None,
        act: Callable[[torch.Tensor], torch.Tensor] = F.silu,
    ) -> None:
        super().__init__()
        if len(features) == 0:
            raise ValueError("resMLP requires at least one feature size.")
        self.features = list(features)
        self.act = act

        if in_features is None:
            self.first_linear: nn.Module = nn.LazyLinear(self.features[0])
        else:
            self.first_linear = nn.Linear(in_features, self.features[0])

        # Residual blocks: each block requires its output shape to equal the
        # first hidden width so that x + z broadcasts correctly. This matches
        # the original implementation, where the training loop always builds
        # features = [hidden] * n_layers.
        self.res_blocks = nn.ModuleList()
        for feat in self.features[1:-1]:
            if feat != self.features[0]:
                raise ValueError(
                    "resMLP residual blocks require features[i] == features[0] "
                    f"for the residual to broadcast (got features={self.features})."
                )
            self.res_blocks.append(
                nn.ModuleList(
                    [
                        nn.Linear(self.features[0], feat),
                        nn.Linear(feat, feat),
                    ]
                )
            )

        self.final_linear = nn.Linear(self.features[0], self.features[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.first_linear(x))
        for block in self.res_blocks:
            lin1, lin2 = block[0], block[1]
            z = self.act(lin1(x))
            z = lin2(z)
            x = self.act(x + z)
        x = self.final_linear(x)
        return x


# =============================================================================
# Fishnet head
# =============================================================================


class Fishnet_from_embedding(nn.Module):
    """Two-headed module turning an embedding into ``(MLE, Fisher)``.

    Identical in spirit to the Flax ``Fishnet_from_embedding``: a one-layer
    "regression" branch produces the MLE and a one-layer "Fisher" branch
    produces the strict log-Cholesky parameters; the latter passes through
    :class:`SoftSquelch` before being projected to the
    ``n_p * (n_p + 1) / 2`` Cholesky degrees of freedom.

    Operates over a leading batch dimension. For an input of shape
    ``(B, embed_dim)`` returns ``mle`` of shape ``(B, n_p)`` and Fisher of
    shape ``(B, n_p, n_p)``. Higher-rank leading batches also work.
    """

    def __init__(
        self,
        n_p: int = 2,
        hidden: int = 50,
        in_features: Optional[int] = None,
        act: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        act_fisher: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        sharpness: float = 5.0,
        threshold: float = 1.0,
        add_prior: bool = False,
    ) -> None:
        super().__init__()
        if n_p < 1:
            raise ValueError(f"n_p must be >= 1, got {n_p}.")
        self.n_p = int(n_p)
        self.hidden = int(hidden)
        self.act = act
        self.act_fisher = act_fisher
        self.add_prior = bool(add_prior)

        self.softsquelch = SoftSquelch(
            threshold=_to_python_scalar(threshold),
            sharpness=_to_python_scalar(sharpness),
        )

        if in_features is None:
            self.t_linear1: nn.Module = nn.LazyLinear(self.hidden)
            self.fisher_linear1: nn.Module = nn.LazyLinear(self.hidden)
        else:
            self.t_linear1 = nn.Linear(in_features, self.hidden)
            self.fisher_linear1 = nn.Linear(in_features, self.hidden)

        n_chol = self.n_p * (self.n_p + 1) // 2
        self.t_linear2 = nn.Linear(self.hidden, self.n_p)
        self.fisher_linear2 = nn.Linear(self.hidden, n_chol)

    def forward(self, x: torch.Tensor):
        t = self.act(self.t_linear1(x))
        fisher_chol = self.fisher_linear1(x)
        fisher_chol = self.softsquelch(fisher_chol)

        t = self.t_linear2(t)
        fisher_chol = self.fisher_linear2(fisher_chol)

        F_mat = construct_fisher_matrix_log_cholesky(fisher_chol, self.n_p)
        if self.add_prior:
            eye = torch.eye(self.n_p, device=F_mat.device, dtype=F_mat.dtype)
            F_mat = F_mat + eye
        return t, F_mat


# =============================================================================
# Internal helpers
# =============================================================================


def _to_python_scalar(x):
    """Coerce a 0-d tensor / numpy scalar / python number to a Python float.

    Useful for accepting the same ``sharpness`` / ``threshold`` argument
    shapes the JAX training loop produces (e.g. ``jax.random.normal(..., (1,))``
    converted via numpy)."""
    if hasattr(x, "item") and callable(x.item):
        try:
            return x.item()
        except Exception:
            pass
    if hasattr(x, "__len__"):
        # 1-element array-likes
        try:
            if len(x) == 1:
                return float(x[0])
        except Exception:
            pass
    return float(x)
