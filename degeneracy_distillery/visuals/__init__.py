"""visuals/ — visualisation utilities for the degeneracy distillery release.

Public API
----------
Fisher network (val_detF over training)
    - :func:`train_fishnets_with_snapshots`: drop-in for
      :func:`degeneracy_distillery.training_loop_fishnets.train_fishnets`
      that writes per-model Fisher snapshots on the validation set every
      ``save_every`` epochs. Same positional signature, same return tuple.
    - :func:`make_gif`: render the ensemble-averaged val_detF over
      ``(theta_1, theta_2)`` evolving over training, plus a portable
      ``.npz`` for local replotting.

Flattener network (learned eta over training)
    - :func:`fit_flattening_with_snapshots`: drop-in for
      :func:`degeneracy_distillery.training_loop_flatten.fit_flattening`
      that writes grid-based snapshots of the **main model** ``eta``
      every ``save_every`` epochs after a ``burn_in``. Same return
      contract as ``fit_flattening``.
    - :func:`make_eta_grid_gif`: render a 2-panel contour gif of
      ``eta_1`` and ``eta_2`` over ``(theta_1, theta_2)`` evolving over
      training, plus a portable ``.npz``.

Shared
    - :func:`display_gif`: notebook helper returning an
      ``IPython.display.Image`` for inline display.
    - Lower-level building blocks: :func:`load_snapshots`,
      :func:`build_frames`, :func:`save_timeseries_npz` (Fisher side);
      :func:`load_eta_snapshots`, :func:`save_eta_timeseries_npz`
      (flattener side).

The JAX-dependent training symbols are loaded lazily (PEP 562), so the
GIF / replotting half of the API stays usable on a machine without
JAX installed (e.g. a laptop where you just want to restyle the
``.npz`` timeseries).

Notebook usage
--------------
After ``pip install -e .`` (or any install of ``degeneracy_distillery``)
the visuals are a real subpackage::

    from degeneracy_distillery.visuals import (
        train_fishnets_with_snapshots,
        make_gif,
        fit_flattening_with_snapshots,
        make_eta_grid_gif,
        display_gif,
    )

    ws, ew, models, scaler, outputs = train_fishnets_with_snapshots(
        theta, data, theta_test, data_test,
        data_shape=n_d, num_models=20,
        train_epochs=4000, save_every=5,
        outdir="runs/fisher_exp", param_names=["mu", "sigma^2"],
    )
    gif_path, epochs, data_path = make_gif(
        "runs/fisher_exp/snapshots", out_path="runs/fisher_exp/val_detF.gif", fps=12,
    )
    display_gif(gif_path)

    w, ensemble_ws, output_dict = fit_flattening_with_snapshots(
        outputs["Fs"], outputs["theta"], ew,
        snapshots_outdir="runs/flat_exp/snapshots",
        save_every=5, burn_in=500, grid_num_pts=30,
        param_names=["mu", "sigma^2"],
        output_prefix="runs/flat_exp/flattened_coords",
    )
    gif_path, epochs, data_path = make_eta_grid_gif(
        "runs/flat_exp/snapshots", out_path="runs/flat_exp/eta_grid.gif", fps=12,
    )
    display_gif(gif_path)

The legacy ``from visuals import ...`` spelling is also kept alive by
a thin top-level shim in the repo root, so existing notebooks /
scripts continue to work unchanged.
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
from .make_eta_grid_gif import (
    align_eta_stack,
    load_eta_snapshots,
    make_eta_grid_gif,
    save_eta_timeseries_npz,
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
_LAZY_FISHNETS_NAMES = (
    "train_fishnets_with_snapshots",
    "predicted_fishers",
    "predicted_mle",
)
_LAZY_FLATTEN_NAMES = (
    "fit_flattening_with_snapshots",
)


def __getattr__(name: str):
    """PEP 562 lazy loader for JAX-dependent symbols.

    Keeps ``import visuals`` cheap and JAX-free until the user actually
    asks for a training symbol.
    """
    if name in _LAZY_FISHNETS_NAMES:
        from . import training_loop_fishnets_snapshots as _t

        for attr in _LAZY_FISHNETS_NAMES:
            globals()[attr] = getattr(_t, attr)
        return globals()[name]
    if name in _LAZY_FLATTEN_NAMES:
        from . import training_loop_flatten_snapshots as _t

        for attr in _LAZY_FLATTEN_NAMES:
            globals()[attr] = getattr(_t, attr)
        return globals()[name]
    raise AttributeError(f"module 'visuals' has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals())
        | set(_LAZY_FISHNETS_NAMES)
        | set(_LAZY_FLATTEN_NAMES)
        | set(__all__)
    )


__all__ = [
    # Fisher side
    "train_fishnets_with_snapshots",
    "predicted_fishers",
    "predicted_mle",
    "make_gif",
    "save_timeseries_npz",
    "load_snapshots",
    "build_frames",
    # Flattener side
    "fit_flattening_with_snapshots",
    "make_eta_grid_gif",
    "save_eta_timeseries_npz",
    "load_eta_snapshots",
    "align_eta_stack",
    # Shared
    "display_gif",
]
