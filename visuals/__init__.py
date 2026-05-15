"""visuals/ — visualisation utilities for the degeneracy distillery release.

Public API
----------
- :func:`train_fishnets_with_snapshots`: drop-in replacement for
  :func:`degeneracy_distillery.training_loop_fishnets.train_fishnets`
  that additionally writes per-model Fisher snapshots on the validation
  set every ``save_every`` epochs. Same positional signature, same
  return tuple ``(ws, ensemble_weights, models, data_scaler, outputs)``.
- :func:`make_gif`: render a .gif of the ensemble-averaged
  ``val_detF`` evolving over training, and (by default) also write a
  self-contained ``.npz`` for local replotting.
- :func:`display_gif`: notebook helper that returns an
  ``IPython.display.Image`` for inline display.
- :func:`save_timeseries_npz`, :func:`load_snapshots`,
  :func:`build_frames`: lower-level building blocks reused by
  :func:`make_gif`.

The JAX-dependent training symbols are loaded lazily (PEP 562), so the
GIF / replotting half of the API stays usable on a machine without
JAX installed (e.g. a laptop where you just want to restyle the
``.npz`` timeseries).

Notebook usage
--------------
From a notebook in ``notebooks/`` (sibling of ``visuals/``)::

    import sys, os
    sys.path.insert(0, os.path.abspath(".."))   # put repo root on sys.path

    from visuals import (
        train_fishnets_with_snapshots, make_gif, display_gif,
    )

    ws, ew, models, scaler, outputs = train_fishnets_with_snapshots(
        theta, data, theta_test, data_test,
        data_shape=n_d, num_models=20,
        train_epochs=4000, save_every=5,
        outdir="runs/my_experiment", param_names=["mu", "sigma^2"],
    )

    gif_path, epochs, data_path = make_gif(
        "runs/my_experiment/snapshots",
        out_path="runs/my_experiment/val_detF.gif",
        fps=12,
    )
    display_gif(gif_path)   # inline display
"""
from __future__ import annotations

# Eager imports: these have no JAX/flax/optax dependency, so they're
# safe to load anywhere (including a "plotting-only" environment).
from .make_detF_gif import (
    build_frames,
    load_snapshots,
    make_gif,
    save_timeseries_npz,
)


def display_gif(gif_path: str, embed: bool = True):
    """Return an IPython display object for inline .gif display in a notebook.

    Parameters
    ----------
    gif_path : str
        Path to the .gif file (typically the first element of the tuple
        returned by :func:`make_gif`).
    embed : bool, default True
        Embed the gif bytes in the notebook output (so the rendered
        notebook is self-contained). Set to ``False`` to reference the
        file by path only.

    Returns
    -------
    IPython.display.Image
        Place this as the last expression in a cell to display inline;
        or pass to ``IPython.display.display(...)``.

    Examples
    --------
    >>> from visuals import make_gif, display_gif
    >>> gif_path, _, _ = make_gif("runs/exp/snapshots")
    >>> display_gif(gif_path)   # inline in the next cell output
    """
    from IPython.display import Image  # lazy import; only needed in notebooks

    return Image(filename=gif_path, embed=embed)


# Lazy-loaded names. Listed here for tab-completion and for the
# `__all__` contract; actually imported on first attribute access.
_LAZY_TRAIN_NAMES = (
    "train_fishnets_with_snapshots",
    "predicted_fishers",
    "predicted_mle",
)


def __getattr__(name: str):
    """PEP 562 lazy loader for JAX-dependent symbols.

    Keeps ``import visuals`` cheap and JAX-free until the user actually
    asks for a training symbol.
    """
    if name in _LAZY_TRAIN_NAMES:
        from . import training_loop_fishnets_snapshots as _t

        for attr in _LAZY_TRAIN_NAMES:
            globals()[attr] = getattr(_t, attr)
        return globals()[name]
    raise AttributeError(f"module 'visuals' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_TRAIN_NAMES) | set(__all__))


__all__ = [
    "train_fishnets_with_snapshots",
    "predicted_fishers",
    "predicted_mle",
    "make_gif",
    "save_timeseries_npz",
    "load_snapshots",
    "build_frames",
    "display_gif",
]
