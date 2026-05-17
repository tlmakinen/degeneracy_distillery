"""Backward-compatibility shim: ``visuals`` is now a subpackage of
``degeneracy_distillery``.

This top-level ``visuals/`` directory only exists so that legacy
notebooks/scripts written as

    from visuals import make_eta_grid_gif, fit_flattening_with_snapshots, ...

keep working without having to be rewritten. New code should use the
canonical path::

    from degeneracy_distillery.visuals import ...

The shim simply re-imports the real package and forwards every public
attribute. Lazy attribute access (PEP 562 ``__getattr__``) inside
``degeneracy_distillery.visuals`` is preserved so JAX is only imported
when a training symbol is actually requested.
"""
from __future__ import annotations

import warnings as _warnings

from degeneracy_distillery import visuals as _real

# Re-export every name the real subpackage declares as public, but
# WITHOUT forcing lazy (JAX-dependent) attributes to resolve at import
# time — that would defeat the PEP 562 lazy loading inside
# `degeneracy_distillery.visuals.__getattr__`. Only copy names that
# are already in the real package's namespace (eager exports); the
# rest are resolved on demand by this module's own ``__getattr__``
# below.
__all__ = list(getattr(_real, "__all__", []))
_real_namespace = vars(_real)
for _name in __all__:
    if _name in _real_namespace:
        globals()[_name] = _real_namespace[_name]


def __getattr__(name: str):
    """Forward attribute access to the canonical subpackage.

    This makes lazy-loaded symbols (e.g. JAX-dependent training
    helpers) resolve through ``degeneracy_distillery.visuals.__getattr__``
    on first reference.
    """
    try:
        return getattr(_real, name)
    except AttributeError as exc:
        raise AttributeError(
            f"module 'visuals' (compat shim) has no attribute {name!r}; "
            f"the canonical path is 'degeneracy_distillery.visuals'."
        ) from exc


def __dir__():
    return sorted(set(__all__) | set(dir(_real)))


# One-time deprecation hint — quiet, doesn't interrupt notebooks.
_warnings.warn(
    "Importing from the top-level `visuals` package is supported for "
    "backward compatibility; prefer `from degeneracy_distillery.visuals "
    "import ...` in new code.",
    DeprecationWarning,
    stacklevel=2,
)
