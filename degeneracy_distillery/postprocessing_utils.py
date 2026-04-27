"""
Postprocessing utilities for SR expression optimization and flattening.

This module provides:
- SR expression parsing and manipulation
- Optimization loss functions for coordinate rotation
- Expression pruning and simplification
- Flattening quality checks

For preprocessing (rotations, alignment, data loading), see preprocessing_utils.py

Author: Consolidated from postprocessing.ipynb
"""

import numpy as np
import jax
import jax.numpy as jnp
import sympy
from typing import Optional, Tuple, List, Dict, Any, Callable
from tqdm import tqdm
from copy import deepcopy

# Import preprocessing utilities
# Support both package import and direct script execution
try:
    from .preprocessing_utils import (
        flatten_with_numerical_jacobian,
        batch_flatten_fisher,
        weighted_std,
    )
    from .sr_utils import promote_jacrev_output, safe_lambdify
except ImportError:
    from preprocessing_utils import (
        flatten_with_numerical_jacobian,
        batch_flatten_fisher,
        weighted_std,
    )
    from sr_utils import promote_jacrev_output, safe_lambdify

# Try importing ESR (required for complexity calculations)
try:
    import esr.generation.generator
    ESR_AVAILABLE = True
except ImportError:
    ESR_AVAILABLE = False


# =============================================================================
# SR EXPRESSION UTILITIES
# =============================================================================

def substitute_names(expr, name_map: dict[str, str]) -> str:
    """Replace keys in `name_map` inside the string form of `expr`."""
    s = str(expr)
    for key in sorted(name_map.keys(), key=len, reverse=True):
        s = s.replace(key, name_map[key])
    return s


def print_discovered_expressions(
    pruned_exprs,
    name_map: dict[str, str] | None = None,
    ) -> None:
    """Print discovered expressions with optional name substitution."""
    name_map = name_map or {}
    print("\n" + "=" * 60)
    print("DISCOVERED EXPRESSIONS")
    print("=" * 60)
    for i, expr in enumerate(pruned_exprs):
        print(f"  η_{i+1} = {substitute_names(expr, name_map)}")


def split_by_punctuation(s: str) -> List[str]:
    """
    Split string by mathematical punctuation while preserving operators.
    
    Parameters
    ----------
    s : str
        Mathematical expression string
        
    Returns
    -------
    List[str]
        List of tokens
    """
    punctuation = '+-*/^(),'
    result = []
    current = ''
    
    for char in s:
        if char in punctuation:
            if current:
                result.append(current)
                current = ''
            result.append(char)
        elif char.isspace():
            if current:
                result.append(current)
                current = ''
        else:
            current += char
            
    if current:
        result.append(current)
        
    return result


def is_float(s: str) -> bool:
    """Check if string represents a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def replace_floats(s: str) -> Tuple[str, List[float]]:
    """
    Replace floats in expression with parameter names b0, b1, ...
    
    Parameters
    ----------
    s : str
        Mathematical expression
        
    Returns
    -------
    replaced : str
        Expression with parameters
    values : List[float]
        Parameter values in order
    """
    split_str = split_by_punctuation(s)
    values = []
    
    for i in range(len(split_str)):
        if is_float(split_str[i]) and "." in split_str[i]:
            values.append(float(split_str[i]))
            split_str[i] = f'b{len(values)-1}'
        elif len(split_str[i]) > 1 and split_str[i][-1] == 'e' and is_float(split_str[i][:-1]):
            if i + 1 < len(split_str) and split_str[i+1] in ['+', '-']:
                values.append(float(''.join(split_str[i:i+3])))
                split_str[i] = f'b{len(values)-1}'
                split_str[i+1] = ''
                split_str[i+2] = ''
    
    # Handle negative parameters
    for i in range(len(values)):
        idx = split_str.index(f'b{i}')
        if (idx == 1) and (split_str[0] == '-'):
            split_str[0] = ''
            values[i] *= -1
        elif (split_str[idx-1] == '-') and (split_str[idx-2] in ['+', '-', '*', '/', '(', '^']):
            values[i] *= -1
            split_str[idx-1] = ''
    
    replaced = ''.join(split_str)
    return replaced, values


def replace_floats_nonlinear(s: str) -> Tuple[str, List[float], List[float], List[str], List[int]]:
    """
    Replace floats and identify linear vs nonlinear parameters.
    
    Parameters
    ----------
    s : str
        Mathematical expression
        
    Returns
    -------
    replaced : str
        Expression with parameters
    values : List[float]
        All parameter values
    linear_values : List[float]
        Linear parameter values
    linear_labels : List[str]
        Linear parameter names
    linear_indexes : List[int]
        Indices of linear parameters
    """
    split_str = split_by_punctuation(s)
    values = []
    param_labels = []
    
    for i in range(len(split_str)):
        if is_float(split_str[i]) and "." in split_str[i]:
            values.append(float(split_str[i]))
            split_str[i] = f'b{len(values)-1}'
            param_labels.append(split_str[i])
        elif len(split_str[i]) > 1 and split_str[i][-1] == 'e' and is_float(split_str[i][:-1]):
            if i + 1 < len(split_str) and split_str[i+1] in ['+', '-']:
                values.append(float(''.join(split_str[i:i+3])))
                split_str[i] = f'b{len(values)-1}'
                param_labels.append(split_str[i])
                split_str[i+1] = ''
                split_str[i+2] = ''
    
    # Handle negative parameters
    for i in range(len(values)):
        idx = split_str.index(f'b{i}')
        if (idx == 1) and (split_str[0] == '-'):
            split_str[0] = ''
            values[i] *= -1
        elif (split_str[idx-1] == '-') and (split_str[idx-2] in ['+', '-', '*', '/', '(', '^']):
            values[i] *= -1
            split_str[idx-1] = ''
    
    replaced = ''.join(split_str)
    
    # Identify linear parameters
    linear_values = []
    linear_labels = []
    linear_indexes = []
    
    # print(param_labels)  # Debug
    
    for i, v in enumerate(values):
        ind = split_str.index(f'b{i}')
        p = split_str[ind]
        
        # Calculate second derivative - if zero, parameter is linear
        derv = sympy.Derivative(sympy.Derivative(str(replaced), p, evaluate=True), p, evaluate=True)
        
        if str(derv) == "0":
            linear_values.append(values[i])
            linear_labels.append(p)
            linear_indexes.append(i)
    
    return replaced, values, linear_values, linear_labels, linear_indexes


def _lambdify_ordered_symbols(
    expr: sympy.Expr, n_params: int, n_b_params: int
) -> Tuple[List[sympy.Symbol], List[sympy.Symbol]]:
    """
    Build (b0..b{n-1}, X1..X{n_params}) symbol lists for lambdify.

    Parser-created symbols must be reused: ``Symbol('b0')`` and
    ``Symbol('b0', real=True)`` are not equal in SymPy, so passing fresh
    ``symbols(..., real=True)`` to lambdify leaves ``b*`` unbound in the
    generated code (globals), which breaks ``jax.jacrev`` (tracers hit SymPy).
    """
    by_name = {s.name: s for s in expr.free_symbols}
    bs: List[sympy.Symbol] = []
    for i in range(n_b_params):
        name = f"b{i}"
        if name not in by_name:
            raise ValueError(
                f"Expression missing parameter {name!r} after parse; "
                f"have symbols {sorted(by_name)}"
            )
        bs.append(by_name[name])
    xs: List[sympy.Symbol] = []
    for i in range(1, n_params + 1):
        name = f"X{i}"
        if name in by_name:
            xs.append(by_name[name])
        else:
            xs.append(sympy.Symbol(name))
    return bs, xs


# =============================================================================
# MATRIX UTILITIES FOR OPTIMIZATION
# =============================================================================

def get_Q(A: np.ndarray) -> np.ndarray:
    """
    Get orthogonal Q from QR decomposition with positive diagonal.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix
        
    Returns
    -------
    Q : np.ndarray
        Orthogonal matrix
    """
    Q, R = np.linalg.qr(A)
    D = np.diag(np.sign(np.diag(R)))
    Q = Q @ D
    return Q


def get_Q_jax(A: jnp.ndarray) -> jnp.ndarray:
    """
    JAX version of get_Q.
    
    Parameters
    ----------
    A : jnp.ndarray
        Input matrix
        
    Returns
    -------
    Q : jnp.ndarray
        Orthogonal matrix

    Notes
    -----
    ``sign(diag(R))`` has zero-gradient cliffs at ``R_ii = 0`` — the map is
    only piecewise-smooth.  For optimization, prefer :func:`cayley_rotation`
    or :func:`expm_rotation`, which are smooth everywhere on their domain
    (the latter is in fact a global surjection onto ``SO(n)``).
    """
    Q, R = jnp.linalg.qr(A)
    D = jnp.diag(jnp.sign(jnp.diag(R)))
    Q = Q @ D
    return Q


# ---------------------------------------------------------------------------
# Smooth SO(n) parameterisations (no QR+sign cliffs).
#
# These are drop-in alternatives to get_Q_jax in optimizer inner loops.  They
# operate on a flat vector ``w`` of length ``n*(n-1)/2`` (the strict
# upper-triangular entries of a skew-symmetric matrix).  Use
# :func:`skew_param_count` to size the optimizer state.
# ---------------------------------------------------------------------------

def skew_param_count(n: int) -> int:
    """Number of free parameters in an ``n x n`` skew-symmetric matrix."""
    return n * (n - 1) // 2


def _skew_from_params_jax(w: jnp.ndarray, n: int) -> jnp.ndarray:
    """Build ``n x n`` skew-symmetric matrix from its ``n(n-1)/2`` free
    parameters (strict upper triangle)."""
    S = jnp.zeros((n, n))
    iu = jnp.triu_indices(n, k=1)
    S = S.at[iu].set(w)
    return S - S.T


def cayley_rotation(w: jnp.ndarray, n: int) -> jnp.ndarray:
    r"""Cayley map :math:`R = (I - S)(I + S)^{-1}` with ``S`` skew-symmetric.

    Smooth on all of :math:`\\mathbb{R}^{n(n-1)/2}` and has image dense in
    ``SO(n)`` (misses only the measure-zero set where ``-1`` is an
    eigenvalue).  Strictly better gradients than the QR + sign parameterisation
    used by :func:`get_Q_jax` — that is the primary reason to use it.
    """
    S = _skew_from_params_jax(w, n)
    I = jnp.eye(n)
    # S has pure-imaginary eigenvalues, so I + S is always invertible.
    return jnp.linalg.solve(I + S, I - S)


def expm_rotation(w: jnp.ndarray, n: int) -> jnp.ndarray:
    """Matrix-exponential ``R = expm(S)`` with ``S`` skew-symmetric.

    Globally surjective onto ``SO(n)``; slightly more expensive than Cayley
    but with no missing sub-manifold.
    """
    S = _skew_from_params_jax(w, n)
    return jax.scipy.linalg.expm(S)


def rotation_from_param(w: jnp.ndarray, n: int, param: str) -> jnp.ndarray:
    """Dispatch helper: choose the orthogonal parameterisation by name."""
    if param == "cayley":
        return cayley_rotation(w, n)
    if param == "expm":
        return expm_rotation(w, n)
    raise ValueError(f"Unknown param={param!r}; use 'cayley' or 'expm'.")


@jax.jit
def norm(A: jnp.ndarray) -> float:
    """Frobenius norm of a matrix."""
    return jnp.linalg.norm(A, ord='fro')


def get_linear_par_index(linear_pars: List[List[float]]) -> Tuple[List[float], List[List[int]]]:
    """
    Get flattened linear parameter indices.
    
    Parameters
    ----------
    linear_pars : List[List[float]]
        Nested list of linear parameters per component
        
    Returns
    -------
    linear_pars_flat : List[float]
        Flattened parameter list
    idx_par_shaped : List[List[int]]
        Index mapping back to original structure
    """
    linear_pars_flat = [p for ps in linear_pars for p in ps]
    idx_par = np.arange(len(linear_pars_flat))
    lengths = [len(ps) for ps in linear_pars]
    
    idx_par_shaped = []
    total_length = 0
    for i, ps in enumerate(linear_pars):
        idx = total_length
        idx_par_shaped.append(list(idx_par[idx:idx+lengths[i]]))
        total_length += lengths[i]
    
    return linear_pars_flat, idx_par_shaped


def construct_M(linear_pars: List[List[float]], n_params: int) -> np.ndarray:
    """
    Construct transformation matrix from linear parameters.
    
    Parameters
    ----------
    linear_pars : List[List[float]]
        Linear parameters per component
    n_params : int
        Number of parameters
        
    Returns
    -------
    M : np.ndarray
        Transformation matrix
    """
    linear_pars_flat, idx_par_shaped = get_linear_par_index(linear_pars)
    M = np.zeros((n_params, len(linear_pars_flat)))
    
    for i, p in enumerate(linear_pars):
        for j, l in enumerate(p):
            M[i, idx_par_shaped[i][j]] = l
    
    return M


# =============================================================================
# CHECK FLATTENING FUNCTION
# =============================================================================

def check_flattening(coordinates: List[str], X: np.ndarray, Fs: np.ndarray,
                     return_J: bool = True, A: Optional[jnp.ndarray] = None,
                     transpose: bool = False) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Check flattening quality of SR coordinate expressions.
    
    Parameters
    ----------
    coordinates : List[str]
        List of SR expressions for each coordinate
    X : np.ndarray
        Input data of shape (n_samples, n_params)
    Fs : np.ndarray
        Fisher matrices of shape (n_samples, n_params, n_params)
    return_J : bool
        Whether to return Jacobian
    A : jnp.ndarray, optional
        Transformation matrix
    transpose : bool
        Whether to transpose Jacobian
        
    Returns
    -------
    flats : jnp.ndarray
        Flattened Fisher matrices
    Jpred : jnp.ndarray, optional
        Predicted Jacobian (if return_J=True)
    """
    if A is None:
        A = jnp.eye(X.shape[1])
    
    # Setup sympy functions
    basis_functions = [
        ["X", "b"],  # type0
        ["square", "exp", "inv", "sqrt", "log", "cos"],  # type1
        ["+", "*", "-", "/", "^"]  # type2
    ]
    
    a, b = sympy.symbols('a b', real=True)
    inv = sympy.Lambda(a, 1/a)
    square = sympy.Lambda(a, a*a)
    sqrt = sympy.Lambda(a, sympy.sqrt(a))
    log = sympy.Lambda(a, sympy.log(a))
    power = sympy.Lambda((a, b), sympy.Pow(a, b))
    
    sympy_locs = {
        "inv": inv, "square": square, "cos": sympy.cos,
        "^": power, "Abs": sympy.Abs, "sqrt": sqrt, "log": log
    }
    
    jac_rows = []
    
    for eq in coordinates:
        expr, pars = replace_floats(eq)
        expr, nodes, c = esr.generation.generator.string_to_node(
            expr, basis_functions, evalf=True, allow_eval=True, 
            check_ops=True, locs=sympy_locs
        )
        
        all_b, all_x = _lambdify_ordered_symbols(expr, X.shape[1], len(pars))
        eq_jax = safe_lambdify(all_b + all_x, expr)
        
        def get_jac_row(p):
            def myeq(*args):
                return promote_jacrev_output(eq_jax(*p, *args))

            yjac = jax.jacrev(myeq, argnums=list(range(0, X.shape[1])))
            Jpred = jnp.array(jax.vmap(yjac)(*X.T)).T
            return Jpred
        
        jac_rows.append(get_jac_row(pars))
    
    Jpred = jnp.stack(jac_rows, axis=-1).transpose((0, 2, 1))
    # print(Jpred.shape)  # Debug
    
    # Use canonical flatten_fisher
    flats = batch_flatten_fisher(Jpred, Fs, A if not jnp.allclose(A, jnp.eye(A.shape[0])) else None)
    
    if return_J:
        return flats, Jpred
    else:
        return flats


def make_check_flattening_fn(X: np.ndarray, Fs: np.ndarray) -> Callable:
    """
    Create a check_flattening function with bound X and Fs.
    
    This is useful for passing to get_pruned_expressions_final.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    Fs : np.ndarray
        Fisher matrices
        
    Returns
    -------
    check_fn : Callable
        Function that takes coordinates and returns flattening results
    """
    def check_fn(coordinates, return_J=True, A=None):
        return check_flattening(coordinates, X, Fs, return_J=return_J, A=A)
    return check_fn


def get_y_sr(coordinates: List[str], X: np.ndarray) -> np.ndarray:
    """
    Evaluate symbolic-regression coordinate strings on ``X`` (NumPy only).

    Uses the same ESR parse, ``replace_floats``, ``_lambdify_ordered_symbols``,
    and argument order as :func:`check_flattening` (constants ``pars`` then
    ``X1``..``X{n_params}``), but returns the coordinate values instead of a
    Jacobian.

    Parameters
    ----------
    coordinates : list of str
        One SR expression per output dimension.
    X : np.ndarray, shape (n_samples, n_params)
        Parameter samples.

    Returns
    -------
    y : np.ndarray, shape (n_samples, len(coordinates))
        Predicted coordinates, column ``k`` is expression ``coordinates[k]``.
    """
    if len(coordinates) == 0:
        return np.empty((X.shape[0], 0), dtype=float)

    basis_functions = [
        ["X", "b"],
        ["square", "exp", "inv", "sqrt", "log", "cos"],
        ["+", "*", "-", "/", "^"],
    ]

    a, b = sympy.symbols("a b", real=True)
    inv = sympy.Lambda(a, 1 / a)
    square = sympy.Lambda(a, a * a)
    sqrt = sympy.Lambda(a, sympy.sqrt(a))
    log = sympy.Lambda(a, sympy.log(a))
    power = sympy.Lambda((a, b), sympy.Pow(a, b))

    sympy_locs = {
        "inv": inv,
        "square": square,
        "cos": sympy.cos,
        "^": power,
        "Abs": sympy.Abs,
        "sqrt": sqrt,
        "log": log,
    }

    cols: List[np.ndarray] = []
    for eq in coordinates:
        expr_str, pars = replace_floats(eq)
        expr, nodes, c = esr.generation.generator.string_to_node(
            expr_str,
            basis_functions,
            evalf=True,
            allow_eval=True,
            check_ops=True,
            locs=sympy_locs,
        )
        all_b, all_x = _lambdify_ordered_symbols(expr, X.shape[1], len(pars))
        eq_fn = safe_lambdify(all_b + all_x, expr, ["numpy"])
        y = np.asarray(eq_fn(*pars, *X.T))
        y = np.reshape(y, (X.shape[0],))
        cols.append(y)

    return np.column_stack(cols)


def pruned_coordinate_is_degenerate(
    expr: str,
    X: np.ndarray,
    y_atol: float = 1e-10,
    const_rel_atol: float = 1e-8,
) -> bool:
    """
    True if a pruned coordinate has no nontrivial θ-signal.

    A coordinate is considered degenerate when its Jacobian row in θ is
    (numerically) zero, which happens for **both**:

    * **Identically zero** expressions (``y ≈ 0`` everywhere on ``X``).
    * **Constant** expressions (``y ≈ c`` everywhere on ``X``, ``c`` independent
      of θ).

    Either case makes :math:`Q = (A J^+)^T F (A J^+)` rank-deficient downstream
    and breaks the flattening operation, so we treat them the same — the
    repair routine then patches both with a linear :math:`c X_j` term so
    every coordinate has at least a linear dependence on one θ component.

    Detection strategy (short-circuited):
      1. Cheap string match for literal ``0``.
      2. ``sympy.simplify`` reduction to ``0``.
      3. **Symbolic constancy check**: ``∂expr/∂X_j == 0`` for every
         ``X_j ∈ {X1, …, X_n}``.
      4. Numerical fallback via :func:`get_y_sr`:

         * ``max|y| < y_atol``                                      → zero, or
         * ``range(y) < max(y_atol, const_rel_atol · max|y|)``      → constant.

    Parameters
    ----------
    expr : str
        Coordinate expression in ``X1..Xn``.
    X : np.ndarray
        ``(n_samples, n_params)`` parameter samples used for the numerical
        fallback.  ``n_params`` is read from ``X.shape[1]``.
    y_atol : float
        Absolute tolerance for treating a coordinate as identically zero
        (also a floor for the constant check).
    const_rel_atol : float
        Relative tolerance for treating a non-zero coordinate as constant:
        ``range(y) < const_rel_atol · max|y|`` triggers a repair.  Default
        ``1e-8``.  Set to ``0`` to disable constant detection (legacy
        behaviour: only literal-zero coordinates are repaired).
    """
    s = str(expr).strip()
    if s in ("0", "0.0", "-0", "-0.0"):
        return True

    n_pars = int(X.shape[1]) if X.ndim == 2 else 0
    detect_constants = const_rel_atol > 0.0
    try:
        simp = sympy.simplify(s)
        if simp == 0:
            return True
        if detect_constants and n_pars > 0:
            # Match by symbol *name* rather than constructing fresh symbols:
            # the parsed X1..Xn carry no assumptions while ``sympy.symbols``
            # may attach ``real=True`` etc., so identity comparison is unsafe.
            theta_names = {f"X{j}" for j in range(1, n_pars + 1)}
            free_names = {sym.name for sym in simp.free_symbols}
            if not (free_names & theta_names):
                return True
    except Exception:
        pass

    if not ESR_AVAILABLE:
        return False
    try:
        y = get_y_sr([s], X)
        if y.size == 0:
            return False
        y_abs_max = float(np.max(np.abs(y)))
        if y_abs_max < y_atol:
            return True
        if detect_constants:
            y_range = float(np.max(y) - np.min(y))
            if y_range < max(y_atol, const_rel_atol * y_abs_max):
                return True
    except Exception:
        pass
    return False


def _mean_flattening_score(
    coordinates: List[str],
    check_flattening_fn: Callable,
    A: np.ndarray,
    n_params: int,
) -> float:
    """Scalar score used in pruning: mean_s ||flat_s - I||_F with same convention as :func:`check_flattening`."""
    eye = jnp.eye(n_params)
    flats, _ = check_flattening_fn(coordinates, A=jnp.array(A))
    return float(jax.vmap(norm)(flats - eye).mean())


def repair_degenerate_pruned_expressions(
    coordinates: List[str],
    X: np.ndarray,
    n_params: int,
    A: np.ndarray,
    check_flattening_fn: Callable,
    flat_score_ref: float,
    y_atol: float = 1e-10,
    const_rel_atol: float = 1e-8,
    rel_deviation_threshold: float = 0.05,
    scale: float = 1.0,
    verbose: bool = False,
) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    """
    If importance pruning drives a full coordinate to a θ-independent value
    (zero **or** any constant), restore a *single* linear :math:`c X_j` term in
    :math:`\\theta` (notation ``X1..Xn``) so the Jacobian is no longer
    rank-degenerate.

    The flattening map uses :math:`Q = (A J^+)^T F (A J^+)`; any coordinate
    whose Jacobian row is identically zero — whether the expression is
    literally ``0`` or a non-trivial constant — makes the pseudoinverse
    pathological, so neither is acceptable.  Both cases are detected by
    :func:`pruned_coordinate_is_degenerate` and patched here.

    **Invariance:** There is in general *no* linear :math:`c^\\top X` for which the
    flatness score is *exactly* unchanged after replacing a zero/constant row
    — :math:`J^+` is global. This routine picks :math:`(j, c=\\text{scale})`
    that *minimizes* the change from ``flat_score_ref`` (the same idea as the
    pruning :math:`\\Delta / \\text{ref}` test). If the best result still
    deviates by more than ``rel_deviation_threshold``, it is kept anyway to
    avoid a dead output; you can set ``scale`` small (e.g. ``1e-3``) to
    reduce impact on the score.

    Parameters
    ----------
    flat_score_ref
        Reference from the un-pruned (or pre-repair) pipeline, e.g. from
        :func:`get_pruned_expressions_final` before the fix.
    y_atol
        Absolute tolerance for treating a coordinate as identically zero
        (forwarded to :func:`pruned_coordinate_is_degenerate`).
    const_rel_atol
        Relative tolerance for treating a non-zero coordinate as constant in
        ``θ`` (forwarded to :func:`pruned_coordinate_is_degenerate`).  Set to
        ``0`` to disable constant detection (legacy zero-only behaviour).
    rel_deviation_threshold
        Maximum acceptable relative change ``|S - ref| / ref`` for logging only.
    scale
        Coefficient in ``+ scale * Xj``; choose ``<< 1`` if a gentler nudge
        to the score is required.

    Returns
    -------
    fixed_expressions
        A copy of ``coordinates`` with degenerate entries patched.
    repaired_indices
        Indices of coordinates that were modified.
    details
        One dict per repair with keys ``output_index``, ``X_index``, ``coeff``,
        ``rel_delta`` (float or ``"inf"``).
    """
    if not ESR_AVAILABLE:
        return list(coordinates), [], []

    out: List[str] = list(map(str, coordinates))
    ref = float(flat_score_ref) if not np.isnan(flat_score_ref) else 0.0
    A_np = np.asarray(A)
    if ref <= 0.0 or np.isclose(ref, 0.0):
        ref = max(ref, 1e-30)  # avoid div by zero in rel_delta

    repaired: List[int] = []
    details: List[Dict[str, Any]] = []

    for i in range(len(out)):
        if not pruned_coordinate_is_degenerate(
            out[i], X, y_atol=y_atol, const_rel_atol=const_rel_atol
        ):
            continue
        if n_params < 1:
            continue

        best_j = 1
        best_score: Optional[float] = None
        best_cand: Optional[str] = None
        e0 = str(out[i])
        cstr = f"{float(scale):.12g}"
        for j in range(1, n_params + 1):
            # Parenthesize so the parser sees one expression (same convention as the rest of the module).
            candidate = f"({e0})+{cstr}*X{j}"
            try:
                simp = str(sympy.simplify(candidate))
            except Exception:
                simp = candidate
            try:
                s_test = _mean_flattening_score(
                    [simp if k == i else out[k] for k in range(len(out))],
                    check_flattening_fn,
                    A_np,
                    n_params,
                )
            except Exception:
                continue
            if best_score is None or abs(s_test - ref) < abs(best_score - ref):
                best_score = s_test
                best_j = j
                best_cand = simp
        if best_cand is None and n_params >= 1:
            if verbose:
                print(
                    f"degenerate output {i}: no candidate linear term could be "
                    f"scored; defaulting to {cstr}*X1"
                )
            best_j = 1
            j = 1
            try:
                best_cand = str(sympy.simplify(f"({e0})+{cstr}*X{j}"))
            except Exception:
                best_cand = f"({e0})+{cstr}*X{j}"
        if best_cand is None:
            if verbose:
                print(f"degenerate output {i}: repair skipped (n_params={n_params})")
            continue

        tmp_coords = [str(best_cand) if k == i else out[k] for k in range(len(out))]
        try:
            s_after = _mean_flattening_score(
                tmp_coords, check_flattening_fn, A_np, n_params
            )
        except Exception:
            s_after = best_score
        if s_after is not None and not (isinstance(s_after, float) and np.isnan(s_after)):
            rel_delta = abs(float(s_after) - ref) / ref
        else:
            rel_delta = float("inf")
        if verbose and rel_delta is not None:
            if rel_delta > rel_deviation_threshold and rel_delta < float("inf"):
                print(
                    f"repair: output {i} -> + {cstr}*X{best_j}  "
                    f"(rel |Δflat|/ref = {rel_delta:.6f} > {rel_deviation_threshold})"
                )
            else:
                print(
                    f"repair: output {i} was degenerate; set to include ~{cstr}*X{best_j} "
                    f"(rel |Δflat|/ref = {rel_delta if rel_delta < float('inf') else 'inf'})"
                )
        out[i] = str(best_cand)
        repaired.append(i)
        details.append(
            {
                "output_index": i,
                "X_index": best_j,
                "coeff": float(scale),
                "rel_delta": float(rel_delta) if rel_delta < float("inf") else "inf",
            }
        )
    return out, repaired, details


def get_missing_vars(coordinates, n_params, n_appearances=2):
    pars_to_append = []

    for j in range(6):
        parstring = "X%d"%(j+1)

        _lens = np.array([len(m.split(parstring)) for m in coordinates])

        par_present = (_lens < n_appearances).sum()

        if par_present == n_params:
            pars_to_append.append(" + (%.4f * "%(np.random.randn()) + parstring + ")")

    return pars_to_append


# =============================================================================
# SR OPTIMIZATION FUNCTIONS
# =============================================================================

def get_alpha_scaling(lambda_: float = 10., epsilon: float = 0.1) -> float:
    """
    Compute adaptive scaling factor for loss weighting.
    
    Uses sigmoid-like scaling to balance loss terms.
    
    Parameters
    ----------
    lambda_ : float
        Maximum scaling factor
    epsilon : float
        Smoothness parameter
        
    Returns
    -------
    float
        Scaling factor alpha
    """
    return -jnp.log(epsilon * (lambda_ - 1.) + epsilon ** 2. / (1 + epsilon)) / epsilon


def smooth_l1_loss(x: jnp.ndarray, delta: float = 0.1) -> jnp.ndarray:
    """
    Smooth L1 loss using log-cosh (differentiable everywhere).
    
    This is preferred over Huber loss for JAX optimization as it's
    smooth everywhere with well-defined gradients.
    
    Parameters
    ----------
    x : jnp.ndarray
        Input values
    delta : float
        Unused, kept for API compatibility
        
    Returns
    -------
    jnp.ndarray
        Sum of smooth L1 losses
    """
    return jnp.log(jnp.cosh(x)).sum()


def smooth_l1_huber(x: jnp.ndarray, delta: float = 0.1) -> jnp.ndarray:
    """
    Huber loss (piecewise smooth L1).
    
    Parameters
    ----------
    x : jnp.ndarray
        Input values
    delta : float
        Transition point between quadratic and linear
        
    Returns
    -------
    jnp.ndarray
        Sum of Huber losses
    """
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    linear = abs_x - quadratic
    return (0.5 * quadratic**2 + delta * linear).sum()


def lossfn_jac_jax(A: jnp.ndarray, 
                   all_pars: List[np.ndarray],
                   all_fns: List[Callable],
                   linear_pars: List[List[float]],
                   linear_indexes: List[List[int]],
                   X: np.ndarray,
                   Fs: np.ndarray,
                   n_params: int,
                   dy_sr: Optional[np.ndarray] = None,
                   parts: bool = False,
                   smoothl1: bool = True,
                   delta: float = 0.5,
                   alpha: float = 1.0,
                   compare_jacs: bool = False,
                   lambda_flat: float = 10.0,
                   verbose: bool = False) -> jnp.ndarray:
    """
    Compute loss for SR coordinate optimization with Jacobian-based flattening.
    
    This is the PRIMARY optimization function that should be used. It optimizes
    the rotation matrix A to:
    1. Minimize L1 norm of coefficients (sparsity)
    2. Match transformed predictions to original
    3. Ensure Fisher matrices are well-flattened (close to identity)
    
    Parameters
    ----------
    A : jnp.ndarray
        Rotation matrix (flattened or square)
    all_pars : List[np.ndarray]
        List of parameter arrays for each component expression
    all_fns : List[Callable]
        List of callable functions for each component
    linear_pars : List[List[float]]
        Linear parameters for each component
    linear_indexes : List[List[int]]
        Indices of linear parameters in all_pars
    X : np.ndarray
        Input data of shape (n_samples, n_params)
    Fs : np.ndarray
        Fisher matrices of shape (n_samples, n_params, n_params)
    n_params : int
        Number of parameters
    dy_sr : np.ndarray, optional
        Reference Jacobians for comparison mode
    parts : bool
        If True, return individual loss components
    smoothl1 : bool
        Use smooth L1 (log-cosh) instead of absolute value
    delta : float
        Smoothness parameter for Huber loss
    alpha : float
        Weight for sparsity term
    compare_jacs : bool
        If True, compare Jacobians instead of predictions
    lambda_flat : float
        Weight scaling for flattening loss
    verbose : bool
        Print debug information
        
    Returns
    -------
    loss : jnp.ndarray
        Total loss (or tuple of parts if parts=True)
    """
    A = A.reshape((n_params, n_params))
    A = get_Q_jax(A)
    
    # Construct coefficient matrix
    M = construct_M(linear_pars, n_params)
    linear_pars_flat, idx_par_shaped = get_linear_par_index(linear_pars)
    ooft = A @ M
    
    # Update parameters with new linear coefficients
    altered_pars = []
    for i, pm in enumerate(all_pars):
        pm2 = jnp.array(deepcopy(pm))
        for j, _ in enumerate(linear_pars[i]):
            pm2 = pm2.at[linear_indexes[i][j]].set(ooft[i, idx_par_shaped[i][j]])
        altered_pars.append(pm2)
    
    # Compute predictions and Jacobians
    n_samples = X.shape[0]
    ypreds_prime = jnp.zeros((n_params, n_samples))
    dypreds_prime = jnp.zeros((n_params, n_samples, n_params))
    
    X_jnp = jnp.array(X)
    
    for l in range(n_params):
        y_l = jnp.zeros(n_samples)
        dy_l = jnp.zeros((n_samples, n_params))
        
        for i in range(n_params):
            p = jnp.array(altered_pars[i])
            y_l = y_l + all_fns[i](*p, *X_jnp.T)
            
            # Compute Jacobian via autodiff
            def myeq(*args):
                return promote_jacrev_output(all_fns[i](*p, *args))

            yjac = jax.jacrev(myeq, argnums=list(range(X.shape[1])))
            Jpred_i = jnp.array(jax.vmap(yjac)(*X_jnp.T)).T
            dy_l = dy_l + Jpred_i
        
        ypreds_prime = ypreds_prime.at[l].set(y_l)
        dypreds_prime = dypreds_prime.at[l].set(dy_l)
    
    Jpred = dypreds_prime.transpose((1, 0, 2))
    
    if verbose:
        print(f"Jpred shape: {Jpred.shape}")
    
    # Compute flattened Fisher matrices
    fn = lambda j, f: flatten_with_numerical_jacobian(j, f, A=A)
    flats = jax.vmap(fn)(Jpred, jnp.array(Fs))
    
    # Original predictions
    ypreds = jnp.array([all_fns[i](*p, *X.T) for i, p in enumerate(all_pars)])
    
    # Compute reconstruction
    if compare_jacs:
        yprime_inv = jnp.einsum("ij,bjk->bik", jnp.linalg.pinv(A), Jpred)
        ypreds_cmp = jnp.array(deepcopy(dy_sr)).T if dy_sr is not None else ypreds
    else:
        yprime_inv = jnp.einsum("ij,jb->bi", jnp.linalg.pinv(A), ypreds_prime)
        ypreds_cmp = ypreds
    
    # Loss components
    _L1 = smooth_l1_loss if smoothl1 else lambda x, d=None : jnp.abs(x).sum()
    
    # Part 1: Sparsity (L1 on coefficients)
    part1 = alpha * _L1(ooft, delta)
    part1 = part1 + alpha * jnp.abs(ooft).sum(1).mean()  # Row-wise sparsity
    
    # Part 2: Reconstruction loss
    lam_ = 1.0 / (2.0 * yprime_inv.shape[0])
    part2 = lam_ * jnp.linalg.norm(yprime_inv - ypreds_cmp.T)
    
    # Part 3: Flattening quality (Fisher close to identity)
    eye = jnp.eye(n_params)
    flat_fn = lambda q: norm(q - eye) + norm(jnp.linalg.pinv(q) - eye)
    part3 = jax.vmap(flat_fn)(flats).mean()
    
    # Adaptive scaling for flattening loss
    alpha_scale = get_alpha_scaling(lambda_flat)
    r = lambda_flat * part3 / (part3 + jnp.exp(-alpha_scale * part3))
    part3 = part3 * r
    
    if parts:
        return part1, part2, part3
    else:
        return part1 + part2 + part3


def lossfn_jac_jax_simple(A: jnp.ndarray,
                          all_pars: List[np.ndarray],
                          all_fns: List[Callable],
                          linear_pars: List[List[float]],
                          linear_indexes: List[List[int]],
                          X: np.ndarray,
                          n_params: int,
                          parts: bool = False,
                          delta: float = 0.5,
                          alpha: float = 1.0) -> jnp.ndarray:
    """
    Simplified loss function without Jacobian/flattening computation.
    
    Use this for faster optimization when flattening quality is not critical.
    Only optimizes for sparsity and reconstruction.
    
    Parameters
    ----------
    A : jnp.ndarray
        Rotation matrix
    all_pars : List[np.ndarray]
        Parameter arrays for each component
    all_fns : List[Callable]
        Callable functions for each component
    linear_pars : List[List[float]]
        Linear parameters
    linear_indexes : List[List[int]]
        Indices of linear parameters
    X : np.ndarray
        Input data
    n_params : int
        Number of parameters
    parts : bool
        Return individual loss components
    delta : float
        Smoothness parameter
    alpha : float
        Sparsity weight
        
    Returns
    -------
    loss : jnp.ndarray
        Total loss
    """
    A = A.reshape((n_params, n_params))
    A = get_Q_jax(A)
    
    M = construct_M(linear_pars, n_params)
    linear_pars_flat, idx_par_shaped = get_linear_par_index(linear_pars)
    ooft = A @ M
    
    # Update parameters
    altered_pars = []
    for i, pm in enumerate(all_pars):
        pm2 = jnp.array(deepcopy(pm))
        for j, _ in enumerate(linear_pars[i]):
            pm2 = pm2.at[linear_indexes[i][j]].set(ooft[i, idx_par_shaped[i][j]])
        altered_pars.append(pm2)
    
    # Compute predictions
    n_samples = X.shape[0]
    ypreds_prime = jnp.zeros((n_params, n_samples))
    X_jnp = jnp.array(X)
    
    for l in range(n_params):
        y_l = jnp.zeros(n_samples)
        for i in range(n_params):
            p = jnp.array(altered_pars[i])
            y_l = y_l + all_fns[i](*p, *X_jnp.T)
        ypreds_prime = ypreds_prime.at[l].set(y_l)
    
    ypreds = jnp.array([all_fns[i](*p, *X.T) for i, p in enumerate(all_pars)])
    yprime_inv = jnp.einsum("ij,jb->bi", jnp.linalg.inv(A), ypreds_prime)
    
    part1 = alpha * smooth_l1_loss(ooft, delta=delta)
    lam_ = 1.0 / (2.0 * yprime_inv.shape[0])
    part2 = lam_ * jnp.linalg.norm(yprime_inv - ypreds_prime.T)
    part3 = 0.0
    
    if parts:
        return part1, part2, part3
    else:
        return part1 + part2 + part3


def loss_and_grad_jax(A: jnp.ndarray,
                      all_pars: List[np.ndarray],
                      all_fns: List[Callable],
                      linear_pars: List[List[float]],
                      linear_indexes: List[List[int]],
                      X: np.ndarray,
                      Fs: np.ndarray,
                      n_params: int,
                      **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute loss and gradient for optimization.
    
    Parameters
    ----------
    A : jnp.ndarray
        Rotation matrix
    **kwargs
        Additional arguments passed to lossfn_jac_jax
        
    Returns
    -------
    loss : jnp.ndarray
        Loss value
    grads : jnp.ndarray
        Gradients with respect to A
    """
    fn = lambda a: lossfn_jac_jax(a, all_pars, all_fns, linear_pars, 
                                   linear_indexes, X, Fs, n_params, **kwargs)
    loss = fn(A)
    grads = jax.grad(fn)(A)
    return loss, grads


# =============================================================================
# SR EXPRESSION PROCESSING
# =============================================================================

def get_component(eq: str, 
                  idx: int,
                  X: np.ndarray,
                  module: str = "numpy") -> Tuple:
    """
    Parse SR expression and extract component information.
    
    Parameters
    ----------
    eq : str
        SR expression string
    idx : int
        Component index
    X : np.ndarray
        Input data (used for determining number of variables)
    module : str
        Module for lambdify ("numpy" or "jax")
        
    Returns
    -------
    labels : list
        Expression labels
    expr : sympy.Expr
        Parsed sympy expression
    pars : np.ndarray
        All parameter values
    linear_pars : np.ndarray
        Linear parameter values
    all_x : list
        Input variable symbols
    all_b : list
        Parameter symbols
    eq_fn : callable
        Compiled function
    param_dict : dict
        Dictionary with all parsed information
    linear_indexes : list
        Indices of linear parameters
    """
    basis_functions = [
        ["X", "b"],
        ["square", "exp", "inv", "sqrt", "log", "cos", "logAbs"],
        ["+", "*", "-", "/", "^"]
    ]
    
    a, b = sympy.symbols('a b', real=True)
    inv = sympy.Lambda(a, 1/a)
    square = sympy.Lambda(a, a*a)
    cube = sympy.Lambda(a, a*a*a)
    sqrt = sympy.Lambda(a, sympy.sqrt(a))
    log = sympy.Lambda(a, sympy.log(a))
    logAbs = sympy.Lambda(a, sympy.log(sympy.Abs(a)))
    power = sympy.Lambda((a, b), sympy.Pow(a, b))
    
    sympy_locs = {
        "inv": inv, "square": square, "cube": cube,
        "cos": sympy.cos, "^": power, "Abs": sympy.Abs,
        "sqrt": sqrt, "log": log, "logAbs": logAbs
    }
    
    expr_str = str(eq)
    expr, pars, linear_pars, linear_par_names, linear_indexes = replace_floats_nonlinear(expr_str)
    
    expr, nodes, c = esr.generation.generator.string_to_node(
        expr, basis_functions, evalf=True, allow_eval=True,
        check_ops=True, locs=sympy_locs
    )
    
    labels = nodes.to_list(basis_functions)
    
    all_b, all_x = _lambdify_ordered_symbols(expr, X.shape[1], len(pars))
    eq_fn = safe_lambdify(all_b + all_x, expr, preferred_modules=[module])
    
    by_name = {s.name: s for s in expr.free_symbols}
    linear_b = [by_name[nm] for nm in linear_par_names]
    
    param_dict = dict(
        labels=labels,
        expr=expr,
        linear_b=linear_b,
        linear_pars=list(linear_pars),
        pars=list(pars),
        all_b=all_b,
        all_x=all_x,
        eq_numpy=eq_fn,
        linear_indexes=linear_indexes
    )
    
    return labels, expr, np.array(pars), np.array(linear_pars), all_x, all_b, eq_fn, param_dict, linear_indexes


def get_pruned_coeffs(A: np.ndarray,
                      all_pars: List[np.ndarray],
                      linear_pars: List[List[float]],
                      n_params: int,
                      threshold: float = 1e-3) -> np.ndarray:
    """
    Get pruned coefficient matrix after rotation.
    
    Parameters
    ----------
    A : np.ndarray
        Rotation matrix
    all_pars : List[np.ndarray]
        Parameter arrays
    linear_pars : List[List[float]]
        Linear parameters
    n_params : int
        Number of parameters
    threshold : float
        Threshold below which coefficients are zeroed
        
    Returns
    -------
    ooft : np.ndarray
        Transformed coefficient matrix
    """
    A = A.reshape((n_params, n_params))
    M = construct_M(linear_pars, n_params)
    ooft = A @ M
    
    for i, p in enumerate(all_pars):
        upto = len(p)
        ooft[i, upto:] = 0
    
    return ooft


def get_pruned_expressions(A: np.ndarray,
                           param_dicts: List[Dict],
                           all_pars: List[np.ndarray],
                           linear_pars: List[List[float]],
                           all_expressions: List[str],
                           linear_labels: List[List],
                           n_params: int,
                           remove_floats: bool = True,
                           decimal: int = 3,
                           rational: Optional[bool] = False,
                           threshold: float = 1e-2) -> Tuple[List[str], List[List[float]]]:
    """
    Generate pruned SR expressions after rotation.
    
    Parameters
    ----------
    A : np.ndarray
        Rotation matrix
    param_dicts : List[Dict]
        Parameter dictionaries for each component
    all_pars : List[np.ndarray]
        All parameters
    linear_pars : List[List[float]]
        Linear parameters
    all_expressions : List[str]
        Original expressions
    linear_labels : List[List]
        Labels for linear parameters
    n_params : int
        Number of parameters
    remove_floats : bool
        Replace floats with parameter names
    decimal : int
        Decimal places for rounding
    rational : bool, optional
        Use rational simplification
    threshold : float
        Threshold for zeroing coefficients
        
    Returns
    -------
    new_expressions : List[str]
        Pruned expressions
    new_constants : List[List[float]]
        Constants in expressions
    """
    A = A.reshape((n_params, n_params))
    
    _, idx_par_shaped = get_linear_par_index(linear_pars)
    M = construct_M(linear_pars, n_params)
    ooft = A @ M
    
    new_expressions = []
    new_constants = []
    
    for l in range(A.shape[0]):
        replaced = []
        
        for i, eq in enumerate(all_expressions):
            eq = str(eq)
            split_str = split_by_punctuation(eq)
            all_labels = param_dicts[i]["all_b"]
            
            # Replace linear parameters
            for j, p in enumerate(linear_labels[i]):
                ind = split_str.index(str(p))
                value = ooft[l, idx_par_shaped[i][j]]
                
                if np.abs(value) < threshold:
                    value = 0.0
                split_str[ind] = str(value)
            
            # Replace non-linear parameters
            for j, p in enumerate(all_labels):
                if p not in linear_labels[i]:
                    ind = split_str.index(str(p))
                    value = all_pars[i][j]
                    split_str[ind] = str(value)
            
            split_str = ''.join(split_str)
            split_str = str(sympy.simplify(str(split_str)))
            replaced.append(split_str)
        
        replaced = '+'.join(replaced)
        replaced = sympy.simplify(str(replaced))
        
        # Round floats
        replaced1 = replaced
        for a in sympy.preorder_traversal(replaced1):
            if isinstance(a, sympy.Float):
                replaced = replaced.subs(a, round(a, decimal))
        
        if remove_floats:
            replaced = sympy.simplify(str(replaced), rational=rational)
            replaced, new_values = replace_floats(str(replaced))
        else:
            replaced = sympy.simplify(str(replaced), rational=rational)
            _, new_values = replace_floats(str(replaced))
            replaced = str(replaced)
        
        new_expressions.append(replaced)
        new_constants.append(new_values)
    
    return new_expressions, new_constants


def get_pruned_expressions_final(A: np.ndarray,
                                  param_dicts: List[Dict],
                                  all_pars: List[np.ndarray],
                                  linear_pars: List[List[float]],
                                  all_expressions: List[str],
                                  linear_labels: List[List],
                                  X: np.ndarray,
                                  Fs: np.ndarray,
                                  n_params: int,
                                  check_flattening_fn: Optional[Callable] = None,
                                  remove_floats: bool = True,
                                  decimal: int = 3,
                                  rational: bool = False,
                                  threshold: float = 1e-2,
                                  verbose: bool = True,
                                  update: bool = False,
                                  importance_based: bool = True,
                                  perturbation: float = 1e-4,
                                  batch_removal: bool = False,
                                  batch_size: int = 5,
                                  repair_degenerate_linear: bool = True,
                                  repair_linear_scale: float = 1.0,
                                  repair_y_atol: float = 1e-10,
                                  repair_const_rel_atol: float = 1e-8) -> Tuple[List[str], List[List[float]]]:
    """
    Generate final pruned expressions with loss-based coefficient removal.
    
    This function removes coefficients that don't significantly affect the 
    flattening quality. Three modes available:
    
    1. importance_based=True, batch_removal=True (fastest):
       - Computes importance scores for all coefficients first
       - Attempts to remove multiple low-importance terms at once
       - Falls back to individual removal if batch fails
       
    2. importance_based=True, batch_removal=False (default, reliable):
       - Computes importance scores for all coefficients first
       - Removes in order of importance (least to most important)
       - Permutation-independent and more principled
       
    3. importance_based=False (legacy):
       - Sequential removal in index order
       - Permutation-dependent but slightly faster for importance computation
    
    Parameters
    ----------
    A : np.ndarray
        Rotation matrix
    param_dicts : List[Dict]
        Parameter dictionaries
    all_pars : List[np.ndarray]
        All parameters
    linear_pars : List[List[float]]
        Linear parameters
    all_expressions : List[str]
        Original expressions
    linear_labels : List[List]
        Linear parameter labels
    X : np.ndarray
        Input data
    Fs : np.ndarray
        Fisher matrices
    n_params : int
        Number of parameters
    check_flattening_fn : Callable, optional
        Function to check flattening quality. If None, creates one from X, Fs.
    remove_floats : bool
        Replace floats with parameter names
    decimal : int
        Decimal places for rounding
    rational : bool
        Use rational simplification
    threshold : float
        Relative loss threshold for removing coefficients
    verbose : bool
        Print progress
    update : bool
        Update flattening score as we progress (only used in legacy mode)
    importance_based : bool
        Use importance-based ordering (recommended for better results)
    perturbation : float
        Finite difference step size for computing importance scores
    batch_removal : bool
        Attempt to remove multiple low-importance coefficients simultaneously
    batch_size : int
        Number of coefficients to attempt removing in each batch
    repair_degenerate_linear : bool
        If True, any coordinate whose Jacobian row is identically zero — i.e.
        the expression collapses to ``0`` *or* to any θ-independent constant —
        is patched with a single :math:`c X_j` term; see
        :func:`repair_degenerate_pruned_expressions`.
    repair_linear_scale : float
        Coefficient ``c`` in that term (try ``1e-3`` if the score moves too much).
    repair_y_atol : float
        Absolute tolerance for the "coordinate is identically zero" test in
        :func:`pruned_coordinate_is_degenerate`.
    repair_const_rel_atol : float
        Relative tolerance for the "coordinate is constant in θ" test (default
        ``1e-8``).  Set to ``0`` to disable constant detection (legacy
        zero-only behaviour).
        
    Returns
    -------
    new_expr : List[str]
        Final pruned expressions
    consts : List[List[float]]
        Constants
    """
    
    # Create check_flattening function if not provided
    if check_flattening_fn is None:
        check_flattening_fn = make_check_flattening_fn(X, Fs)
    
    A = A.reshape((n_params, n_params))
    
    # Get reference expressions and score
    new_expr, consts = get_pruned_expressions(
        A=A, param_dicts=param_dicts, all_pars=all_pars,
        linear_pars=linear_pars, all_expressions=all_expressions,
        linear_labels=linear_labels, n_params=n_params,
        remove_floats=False, decimal=3, rational=rational, threshold=0.0
    )
    
    flats, _ = check_flattening_fn(new_expr, A=jnp.array(A))
    eye = jnp.eye(n_params)
    flat_score_reference = jax.vmap(norm)(flats - eye).mean()
    
    if importance_based:
        # === IMPORTANCE-BASED PRUNING (PERMUTATION-INDEPENDENT) ===
        
        if verbose:
            print(f"initial flattening score: {flat_score_reference:.6f}")
            print("computing importance scores for all coefficients...")
        
        # Flatten linear_pars for easier indexing
        linear_pars_flat, _ = get_linear_par_index(linear_pars)
        
        # Compute importance score for each coefficient
        importance_scores = []
        coeff_indices = []  # Track (i, j) tuples
        
        for i, pararr in enumerate(linear_pars):
            for j in range(len(pararr)):
                coeff_val = linear_pars[i][j]
                
                # Skip if already near zero
                if np.abs(coeff_val) < 1e-12:
                    importance_scores.append(0.0)
                    coeff_indices.append((i, j))
                    continue
                
                # Compute finite difference: perturb coefficient slightly
                linear_pars_perturbed = deepcopy(linear_pars)
                linear_pars_perturbed[i][j] = coeff_val + perturbation
                
                try:
                    prop_expr, _ = get_pruned_expressions(
                        A=A, param_dicts=param_dicts, all_pars=all_pars,
                        linear_pars=linear_pars_perturbed, all_expressions=all_expressions,
                        linear_labels=linear_labels, n_params=n_params,
                        remove_floats=False, decimal=3, rational=rational, threshold=0.0
                    )
                    
                    flats_temp, _ = check_flattening_fn(prop_expr, A=jnp.array(A))
                    flat_score_temp = jax.vmap(norm)(flats_temp - eye).mean()
                    
                    # Importance = |gradient| * |coefficient value|
                    grad_approx = (flat_score_temp - flat_score_reference) / perturbation
                    importance = np.abs(grad_approx) * np.abs(coeff_val)
                    importance_scores.append(float(importance))
                    
                except Exception as e:
                    # If perturbation causes error, mark as high importance (don't remove)
                    importance_scores.append(float('inf'))
                
                coeff_indices.append((i, j))
        
        if verbose:
            finite_scores = [s for s in importance_scores if s != float('inf')]
            if finite_scores:
                print(f"Importance scores - min: {np.min(finite_scores):.6e}, "
                      f"max: {np.max(finite_scores):.6e}, "
                      f"median: {np.median(finite_scores):.6e}")
        
        # Sort coefficients by importance (ascending = least important first)
        sorted_order = np.argsort(importance_scores)
        
        # Try removing coefficients in order of increasing importance
        linear_pars_export = deepcopy(linear_pars)
        n_removed = 0
        n_total = len(importance_scores)
        
        if batch_removal:
            # === BATCH REMOVAL MODE ===
            # Try removing multiple low-importance coefficients at once
            
            if verbose:
                print(f"using batch removal with batch_size={batch_size}")
            
            batch_start = 0
            pbar = tqdm(total=n_total, desc="batch pruning") if verbose else None
            
            while batch_start < n_total:
                # Get next batch of candidates
                batch_end = min(batch_start + batch_size, n_total)
                batch_indices = sorted_order[batch_start:batch_end]
                
                # Filter out already-zero coefficients
                active_batch = []
                for sort_idx in batch_indices:
                    i, j = coeff_indices[sort_idx]
                    if np.abs(linear_pars_export[i][j]) >= 1e-12:
                        active_batch.append((sort_idx, i, j))
                
                if len(active_batch) == 0:
                    batch_start = batch_end
                    if pbar:
                        pbar.update(len(batch_indices))
                    continue
                
                # Try removing entire batch at once
                linear_pars_batch = deepcopy(linear_pars_export)
                for sort_idx, i, j in active_batch:
                    linear_pars_batch[i][j] = 0.0
                
                try:
                    prop_expr, _ = get_pruned_expressions(
                        A=A, param_dicts=param_dicts, all_pars=all_pars,
                        linear_pars=linear_pars_batch, all_expressions=all_expressions,
                        linear_labels=linear_labels, n_params=n_params,
                        remove_floats=False, decimal=3, rational=rational, threshold=0.0
                    )
                    
                    flats_temp, _ = check_flattening_fn(prop_expr, A=jnp.array(A))
                    flat_score_temp = jax.vmap(norm)(flats_temp - eye).mean()
                    delta = (flat_score_temp - flat_score_reference) / flat_score_reference
                    
                    if delta < threshold:
                        # Accept entire batch removal
                        linear_pars_export = linear_pars_batch
                        n_removed += len(active_batch)
                        
                        if verbose:
                            print(f"batch {batch_start}-{batch_end}: removed {len(active_batch)} coeffs (delta: {delta:.6f})")
                        if pbar:
                            pbar.update(len(batch_indices))
                        
                        batch_start = batch_end
                        continue
                    
                except Exception as e:
                    if verbose:
                        print(f"batch {batch_start}-{batch_end} caused error, trying individually")
                
                # Batch failed - try removing individually
                for sort_idx, i, j in active_batch:
                    linear_pars_temp = deepcopy(linear_pars_export)
                    linear_pars_temp[i][j] = 0.0
                    
                    try:
                        prop_expr, _ = get_pruned_expressions(
                            A=A, param_dicts=param_dicts, all_pars=all_pars,
                            linear_pars=linear_pars_temp, all_expressions=all_expressions,
                            linear_labels=linear_labels, n_params=n_params,
                            remove_floats=False, decimal=3, rational=rational, threshold=0.0
                        )
                        
                        flats_temp, _ = check_flattening_fn(prop_expr, A=jnp.array(A))
                        flat_score_temp = jax.vmap(norm)(flats_temp - eye).mean()
                        delta = (flat_score_temp - flat_score_reference) / flat_score_reference
                        
                        if delta < threshold:
                            linear_pars_export[i][j] = 0.0
                            n_removed += 1
                            if verbose:
                                print(f"  individual: removed coeff ({i},{j}), delta: {delta:.6f}")
                        
                    except Exception as e:
                        if verbose:
                            print(f"  zeroing coeff ({i},{j}) caused error - keeping it")
                
                if pbar:
                    pbar.update(len(batch_indices))
                batch_start = batch_end
            
            if pbar:
                pbar.close()
        
        else:
            # === INDIVIDUAL REMOVAL MODE ===
            
            iterator = tqdm(sorted_order, desc="pruning coefficients") if verbose else sorted_order
            
            for sort_idx in iterator:
                i, j = coeff_indices[sort_idx]
                
                # Skip if already zero
                if np.abs(linear_pars_export[i][j]) < 1e-12:
                    continue
                
                # Try zeroing this coefficient
                linear_pars_temp = deepcopy(linear_pars_export)
                linear_pars_temp[i][j] = 0.0
                
                try:
                    prop_expr, _ = get_pruned_expressions(
                        A=A, param_dicts=param_dicts, all_pars=all_pars,
                        linear_pars=linear_pars_temp, all_expressions=all_expressions,
                        linear_labels=linear_labels, n_params=n_params,
                        remove_floats=False, decimal=3, rational=rational, threshold=0.0
                    )
                    
                    flats_temp, _ = check_flattening_fn(prop_expr, A=jnp.array(A))
                    flat_score_temp = jax.vmap(norm)(flats_temp - eye).mean()
                    delta = (flat_score_temp - flat_score_reference) / flat_score_reference
                    
                    if delta < threshold:
                        # Accept removal
                        linear_pars_export[i][j] = 0.0
                        n_removed += 1
                        
                        if verbose and hasattr(iterator, 'set_postfix'):
                            iterator.set_postfix({'removed': n_removed, 'delta': f'{delta:.6f}'})
                    elif verbose:
                        if hasattr(iterator, 'write'):
                            iterator.write(f"coeff ({i},{j}) - importance {importance_scores[sort_idx]:.6e}: "
                                          f"delta {delta:.6f} exceeds threshold, keeping")
                        
                except Exception as e:
                    if verbose:
                        msg = f"zeroing coeff ({i},{j}) caused error - keeping it"
                        if hasattr(iterator, 'write'):
                            iterator.write(msg)
                        else:
                            print(msg)
        
        if verbose:
            print(f"\npruning summary: removed {n_removed}/{n_total} coefficients")
    
    else:
        # === LEGACY SEQUENTIAL PRUNING (PERMUTATION-DEPENDENT) ===
        
        linear_pars_export = deepcopy(linear_pars)
        iterator = tqdm(enumerate(linear_pars)) if verbose else enumerate(linear_pars)
        
        for i, pararr in iterator:
            linear_pars2 = deepcopy(linear_pars)
            if verbose:
                print(f"Looking at component {i}")
            
            for j in range(len(pararr)):
                try:
                    linear_pars2[i][j] = 0.0
                    
                    prop_expr, _ = get_pruned_expressions(
                        A=A, param_dicts=param_dicts, all_pars=all_pars,
                        linear_pars=linear_pars2, all_expressions=all_expressions,
                        linear_labels=linear_labels, n_params=n_params,
                        remove_floats=False, decimal=3, rational=rational, threshold=0.0
                    )
                    
                    flats_j, _ = check_flattening_fn(prop_expr)
                    flat_score_j = jax.vmap(norm)(flats_j - eye).mean()
                    delta = (flat_score_j - flat_score_reference) / flat_score_reference
                    
                    if verbose:
                        print(f'  delta: {delta:.6f}')
                    
                    if delta < threshold:
                        linear_pars_export[i][j] = 0.0
                        if update:
                            flat_score_reference = flat_score_j
                        
                except Exception as e:
                    if verbose:
                        print(f'  zeroed component -> skip!')
                
                linear_pars2 = deepcopy(linear_pars_export)
    
    # Final pruned expressions
    new_expr, consts = get_pruned_expressions(
        A=A, param_dicts=param_dicts, all_pars=all_pars,
        linear_pars=linear_pars_export, all_expressions=all_expressions,
        linear_labels=linear_labels, n_params=n_params,
        remove_floats=remove_floats, decimal=decimal,
        rational=rational, threshold=0.0
    )

    if repair_degenerate_linear and ESR_AVAILABLE:
        new_expr, _repaired, _rinfo = repair_degenerate_pruned_expressions(
            new_expr,
            X,
            n_params,
            A,
            check_flattening_fn,
            float(flat_score_reference),
            y_atol=repair_y_atol,
            const_rel_atol=repair_const_rel_atol,
            rel_deviation_threshold=threshold,
            scale=repair_linear_scale,
            verbose=verbose,
        )
        if _repaired:
            if remove_floats:
                paired = [
                    replace_floats(str(sympy.simplify(str(e)))) for e in new_expr
                ]
                new_expr = [p[0] for p in paired]
                consts = [p[1] for p in paired]
            else:
                new_expr = [str(sympy.simplify(str(e))) for e in new_expr]
                consts = [
                    replace_floats(s)[1] for s in new_expr
                ]
    
    return new_expr, consts


def prune_all_constants(expressions: List[str],
                        X: np.ndarray,
                        Fs: np.ndarray,
                        n_params: int,
                        check_flattening_fn: Optional[Callable] = None,
                        threshold: float = 0.05,
                        perturbation: float = 1e-4,
                        A: Optional[np.ndarray] = None,
                        verbose: bool = True) -> Tuple[List[str], List[List[float]], List[Dict]]:
    """
    Prune ALL constants in symbolic expressions (including nonlinear ones).
    
    Unlike get_pruned_expressions_final which only handles linear coefficients,
    this function operates on the final expression strings and can remove any
    numeric constant - including those inside exp(), log(), cos(), etc.
    
    For a constant c in the expression, zeroing it means substituting c=0 and
    simplifying via sympy. For example:
        exp(a*X1 + b*X3 + c*X5) with c=0  ->  exp(a*X1 + b*X3)
    
    Parameters
    ----------
    expressions : List[str]
        Symbolic expression strings (one per coordinate)
    X : np.ndarray
        Input data of shape (n_samples, n_params)
    Fs : np.ndarray
        Fisher matrices of shape (n_samples, n_params, n_params)
    n_params : int
        Number of parameters
    check_flattening_fn : Callable, optional
        Function to check flattening quality. If None, creates from X, Fs.
    threshold : float, default=0.05
        Relative loss increase tolerance for removing a constant
    perturbation : float, default=1e-4
        Finite difference step for importance scoring
    A : np.ndarray, optional
        Rotation matrix passed to check_flattening_fn. Defaults to identity.
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    pruned_expressions : List[str]
        Simplified expressions with unimportant constants removed
    pruned_constants : List[List[float]]
        Remaining constants in each expression
    importance_info : List[Dict]
        Per-constant importance scores for diagnostics. Each dict has keys:
        'expr_idx', 'const_idx', 'value', 'importance', 'removed'
    """
    if check_flattening_fn is None:
        check_flattening_fn = make_check_flattening_fn(X, Fs)
    
    if A is None:
        A = np.eye(n_params)
    A_jnp = jnp.array(A)
    
    eye = jnp.eye(n_params)
    
    # Get reference flattening score
    flats_ref, _ = check_flattening_fn(expressions, A=A_jnp)
    flat_score_ref = float(jax.vmap(norm)(flats_ref - eye).mean())
    
    if verbose:
        print(f"Pruning all constants (including nonlinear)")
        print(f"Reference flattening score: {flat_score_ref:.6f}")
    
    # Catalog all constants across all expressions
    all_const_info = []
    for expr_idx, expr_str in enumerate(expressions):
        _, values = replace_floats(str(expr_str))
        for const_idx, val in enumerate(values):
            all_const_info.append({
                'expr_idx': expr_idx,
                'const_idx': const_idx,
                'value': val,
                'importance': 0.0,
                'removed': False
            })
    
    if verbose:
        print(f"Found {len(all_const_info)} total constants across {len(expressions)} expressions")
    
    if len(all_const_info) == 0:
        return list(expressions), [[] for _ in expressions], []
    
    # Phase 1: Compute importance scores for all constants
    if verbose:
        print("Computing importance scores...")
    
    importance_iter = tqdm(all_const_info, desc="Scoring") if verbose else all_const_info
    
    for info in importance_iter:
        expr_idx = info['expr_idx']
        const_idx = info['const_idx']
        val = info['value']
        
        if abs(val) < 1e-12:
            info['importance'] = 0.0
            continue
        
        # Build modified expressions with this constant perturbed
        modified_exprs = list(expressions)
        expr_str = str(expressions[expr_idx])
        
        # Replace floats, perturb one, reconstruct
        template, values = replace_floats(expr_str)
        values_perturbed = list(values)
        values_perturbed[const_idx] = val + perturbation
        
        # Substitute perturbed values back into template
        perturbed_str = template
        for k in range(len(values_perturbed)):
            perturbed_str = perturbed_str.replace(f'b{k}', f'({values_perturbed[k]})', 1)
        
        modified_exprs[expr_idx] = str(sympy.simplify(perturbed_str))
        
        try:
            flats_p, _ = check_flattening_fn(modified_exprs, A=A_jnp)
            flat_score_p = float(jax.vmap(norm)(flats_p - eye).mean())
            
            grad_approx = (flat_score_p - flat_score_ref) / perturbation
            info['importance'] = abs(grad_approx) * abs(val)
        except Exception:
            info['importance'] = float('inf')
    
    # Phase 2: Sort by importance and attempt removal (least important first)
    sorted_info = sorted(all_const_info, key=lambda x: x['importance'])
    
    if verbose:
        finite_scores = [c['importance'] for c in sorted_info if c['importance'] != float('inf')]
        if finite_scores:
            print(f"Importance scores - min: {np.min(finite_scores):.6e}, "
                  f"max: {np.max(finite_scores):.6e}, "
                  f"median: {np.median(finite_scores):.6e}")
    
    current_exprs = list(expressions)
    n_removed = 0
    
    removal_iter = tqdm(sorted_info, desc="Pruning") if verbose else sorted_info
    
    for info in removal_iter:
        expr_idx = info['expr_idx']
        val = info['value']
        
        if abs(val) < 1e-12:
            continue
        
        # Build candidate expression with this constant set to 0
        expr_str = str(current_exprs[expr_idx])
        template, values = replace_floats(expr_str)
        
        # Find which constant index in the CURRENT expression matches
        # (indices may shift as expressions get simplified)
        if info['const_idx'] >= len(values):
            continue
        
        # Match by value proximity since indices can shift after simplification
        best_match = None
        best_dist = float('inf')
        for k, v in enumerate(values):
            dist = abs(v - val)
            if dist < best_dist:
                best_dist = dist
                best_match = k
        
        if best_match is None or best_dist > abs(val) * 0.1 + 1e-8:
            continue
        
        # Zero this constant and simplify
        values_zeroed = list(values)
        values_zeroed[best_match] = 0.0
        
        zeroed_str = template
        for k in range(len(values_zeroed)):
            zeroed_str = zeroed_str.replace(f'b{k}', f'({values_zeroed[k]})', 1)
        
        try:
            simplified = str(sympy.simplify(zeroed_str))
            
            # Check if simplification is valid
            candidate_exprs = list(current_exprs)
            candidate_exprs[expr_idx] = simplified
            
            flats_c, _ = check_flattening_fn(candidate_exprs, A=A_jnp)
            flat_score_c = float(jax.vmap(norm)(flats_c - eye).mean())
            delta = (flat_score_c - flat_score_ref) / flat_score_ref
            
            if delta < threshold:
                current_exprs[expr_idx] = simplified
                info['removed'] = True
                n_removed += 1
                
                if verbose and hasattr(removal_iter, 'set_postfix'):
                    removal_iter.set_postfix({
                        'removed': n_removed,
                        'expr': expr_idx,
                        'val': f'{val:.4f}',
                        'delta': f'{delta:.6f}'
                    })
                    
        except Exception as e:
            if verbose and hasattr(removal_iter, 'write'):
                removal_iter.write(f"Error zeroing const {val:.4f} in expr {expr_idx}: {e}")
    
    if verbose:
        print(f"\nConstant pruning summary: removed {n_removed}/{len(all_const_info)} constants")
    
    # Extract final constants
    pruned_constants = []
    for expr_str in current_exprs:
        _, vals = replace_floats(str(expr_str))
        pruned_constants.append(vals)
    
    return current_exprs, pruned_constants, all_const_info


def optimize_sparse_rotation(M: np.ndarray,
                             lambda_ortho: float = 1.0,
                             alpha: float = 1.0,
                             maxiter: int = 1000,
                             verbose: bool = True,
                             use_jax: bool = True,
                             enforce_orthogonal: bool = True,
                             param: str = "qr",
                             loss: str = "logcosh",
                             n_reweight: int = 5,
                             reweight_eps: float = 1e-3,
                             learning_rate: float = 0.01,
                             momentum: float = 0.9,
                             seed: Optional[int] = None) -> np.ndarray:
    """
    Find orthogonal rotation matrix A that makes A @ M sparse.
    
    This is useful for finding better coordinate representations by rotating
    the coefficient matrix to maximize sparsity.
    
    Parameters
    ----------
    M : np.ndarray
        Coefficient matrix of shape (n_params, n_coefficients)
    lambda_ortho : float, default=1.0
        Weight for orthogonality penalty (only used when ``param='qr'`` and
        ``enforce_orthogonal=False``).  Ignored for ``'cayley'`` / ``'expm'``.
    alpha : float, default=1.0
        Scaling factor for log-cosh sparsity loss (higher = closer to L1).
        Ignored when ``loss='reweighted_l1'``.
    maxiter : int, default=1000
        Maximum inner-loop gradient-descent iterations.  For reweighted-L1
        the total work is ``n_reweight * maxiter``.
    verbose : bool, default=True
        Print optimization progress.
    use_jax : bool, default=True
        Use JAX (recommended).  Ignored for ``param`` other than ``'qr'`` —
        the Cayley / expm parameterisations are JAX-only.
    enforce_orthogonal : bool, default=True
        When ``param='qr'``: project to orthogonal manifold at each step.
        Ignored for ``'cayley'`` / ``'expm'``, which are orthogonal by
        construction.
    param : {'qr', 'cayley', 'expm'}, default='qr'
        Orthogonal parameterisation.

        - ``'qr'``: legacy path — optimise ``A_flat`` and project with
          :func:`get_Q_jax`.  ``sign(diag(R))`` has gradient cliffs; when
          the optimiser lands near ``R_ii = 0`` it can stall.
        - ``'cayley'``: optimise the ``n(n-1)/2`` strictly-upper-triangular
          parameters ``w`` and build ``R = (I - S)(I + S)^{-1}`` via
          :func:`cayley_rotation`.  Smooth gradients; recommended.
        - ``'expm'``: ``R = expm(S)`` via :func:`expm_rotation`.  Slightly
          more expensive but globally surjective onto ``SO(n)``.
    loss : {'logcosh', 'reweighted_l1'}, default='logcosh'
        Sparsity-inducing loss on ``A @ M``.

        - ``'logcosh'``: smooth L1 (``sum log cosh(alpha * x) / alpha``).
          Does not drive entries to exact zero; some residual always
          remains.
        - ``'reweighted_l1'``: iteratively-reweighted L1 (Candès-Wakin-Boyd)
          ``sum w_ij sqrt(x_ij^2 + eps^2)`` with ``w_{k+1} = 1 /
          sqrt(x_k^2 + eps^2)``.  Converges to a ``||.||_0``-like solution
          with much cleaner zero structure.  Prefer with
          ``param='cayley'``.
    n_reweight : int, default=5
        Number of outer reweighting iterations when
        ``loss='reweighted_l1'`` (ignored otherwise).
    reweight_eps : float, default=1e-3
        Smoothing parameter in the reweighted-L1 surrogate.
    learning_rate, momentum : float
        Inner-loop optimiser settings.  Defaults match the legacy values.
    seed : int, optional
        Seed for the random orthogonal initialisation.  ``None`` uses
        :func:`numpy.random`'s global state (legacy behaviour).

    Returns
    -------
    A_opt : np.ndarray
        Optimized rotation matrix of shape (n_params, n_params).

    Notes
    -----
    The default combination ``param='qr'`` and ``loss='logcosh'`` reproduces
    the original behaviour exactly.  For best results on realistic SR
    output, use ``param='cayley'`` and ``loss='reweighted_l1'``.

    Examples
    --------
    Legacy call (unchanged)::

        A = optimize_sparse_rotation(M)

    Recommended::

        A = optimize_sparse_rotation(M, param='cayley',
                                     loss='reweighted_l1',
                                     n_reweight=8, maxiter=400)
    """
    n_dim = M.shape[0]
    if param not in ("qr", "cayley", "expm"):
        raise ValueError(f"param must be one of 'qr','cayley','expm'; got {param!r}")
    if loss not in ("logcosh", "reweighted_l1"):
        raise ValueError(f"loss must be 'logcosh' or 'reweighted_l1'; got {loss!r}")

    # Cayley / expm are JAX-only.
    if param in ("cayley", "expm") and not use_jax:
        raise ValueError(f"param={param!r} requires use_jax=True")

    if seed is not None:
        rng = np.random.default_rng(seed)

        def _randn(*shape):
            return rng.standard_normal(shape)
    else:
        def _randn(*shape):
            return np.random.randn(*shape)

    # ---------------------------------------------------------------
    # QR parameterisation (legacy path)
    # ---------------------------------------------------------------
    if param == "qr":
        if use_jax:
            from jax import grad, jit

            M_j = jnp.array(M)

            def _sparsity(RM, weights):
                if loss == "logcosh":
                    return jnp.sum(jnp.log(jnp.cosh(alpha * RM))) / alpha
                # reweighted_l1
                return jnp.sum(weights * jnp.sqrt(RM * RM + reweight_eps ** 2))

            if enforce_orthogonal:
                def _raw_loss(A_flat, weights):
                    A = A_flat.reshape((n_dim, n_dim))
                    A = get_Q_jax(A)
                    RM = A @ M_j
                    row_sparsity = jnp.abs(RM).sum(axis=1).mean()
                    return _sparsity(RM, weights) + 0.5 * row_sparsity
            else:
                def _raw_loss(A_flat, weights):
                    A = A_flat.reshape((n_dim, n_dim))
                    RM = A @ M_j
                    row_sparsity = jnp.abs(RM).sum(axis=1).mean()
                    I = jnp.eye(n_dim)
                    ortho_loss = jnp.linalg.norm(A.T @ A - I, ord='fro') ** 2
                    return (_sparsity(RM, weights) + 0.5 * row_sparsity
                            + lambda_ortho * ortho_loss)

            loss_fn = jit(_raw_loss)
            grad_fn = jit(grad(_raw_loss))

            A_init = get_Q_jax(
                jnp.array(_randn(n_dim, n_dim) * (2.0 / n_dim ** 2))
            )
            x = A_init.flatten()
            weights = jnp.ones_like(M_j)

            best_loss = float("inf")
            best_x = x

            if verbose:
                print(
                    f"Optimizing rotation (JAX, param=qr, loss={loss}, "
                    f"{'orthogonal' if enforce_orthogonal else 'soft'})..."
                )

            outer = n_reweight if loss == "reweighted_l1" else 1
            for r in range(outer):
                velocity = jnp.zeros_like(x)
                patience = 50
                no_improve = 0
                for i in range(maxiter):
                    g = grad_fn(x, weights)
                    velocity = momentum * velocity - learning_rate * g
                    x = x + velocity
                    cur = float(loss_fn(x, weights))
                    if cur < best_loss:
                        best_loss = cur
                        best_x = x
                        no_improve = 0
                    else:
                        no_improve += 1
                    if no_improve > patience:
                        if verbose:
                            print(f"  outer {r+1}/{outer}: early stop at inner {i}")
                        break
                    if verbose and i % 100 == 0:
                        print(f"  outer {r+1}/{outer} iter {i}: loss = {cur:.6f}")
                if loss == "reweighted_l1":
                    A_cur = get_Q_jax(x.reshape((n_dim, n_dim))) if enforce_orthogonal \
                        else x.reshape((n_dim, n_dim))
                    RM_cur = A_cur @ M_j
                    weights = 1.0 / jnp.sqrt(RM_cur * RM_cur + reweight_eps ** 2)
                    if verbose:
                        print(f"  reweight {r+1}/{outer}: ||R M||_1 "
                              f"= {float(jnp.abs(RM_cur).sum()):.4f}")

            A_opt = np.array(best_x).reshape((n_dim, n_dim))
            if enforce_orthogonal:
                A_opt = np.array(get_Q_jax(jnp.array(A_opt)))

        else:
            # NumPy/SciPy version (logcosh only — reweighted_l1 would add
            # complexity with little upside outside JAX).
            if loss == "reweighted_l1":
                raise ValueError(
                    "loss='reweighted_l1' requires use_jax=True"
                )
            from scipy.optimize import minimize

            def loss_fn_numpy(A_flat):
                A = A_flat.reshape((n_dim, n_dim))
                if enforce_orthogonal:
                    A = np.array(get_Q(A))
                RM = A @ M
                sparsity = np.sum(np.log(np.cosh(alpha * RM))) / alpha
                row_sparsity = np.abs(RM).sum(axis=1).mean()
                if enforce_orthogonal:
                    return sparsity + 0.5 * row_sparsity
                I = np.eye(n_dim)
                ortho = np.linalg.norm(A.T @ A - I, ord='fro') ** 2
                return sparsity + 0.5 * row_sparsity + lambda_ortho * ortho

            A_init = get_Q(_randn(n_dim, n_dim) * (2.0 / n_dim ** 2))
            result = minimize(
                fun=loss_fn_numpy, x0=A_init.flatten(),
                method="L-BFGS-B",
                options={"disp": verbose, "maxiter": maxiter},
            )
            A_opt = result.x.reshape((n_dim, n_dim))
            if enforce_orthogonal:
                A_opt = get_Q(A_opt)

    # ---------------------------------------------------------------
    # Cayley / expm parameterisation (recommended path)
    # ---------------------------------------------------------------
    else:
        from jax import grad, jit

        M_j = jnp.array(M)
        n_w = skew_param_count(n_dim)
        rot_fn = cayley_rotation if param == "cayley" else expm_rotation

        def _sparsity(RM, weights):
            if loss == "logcosh":
                return jnp.sum(jnp.log(jnp.cosh(alpha * RM))) / alpha
            return jnp.sum(weights * jnp.sqrt(RM * RM + reweight_eps ** 2))

        def _raw_loss(w, weights):
            R = rot_fn(w, n_dim)
            RM = R @ M_j
            row_sparsity = jnp.abs(RM).sum(axis=1).mean()
            return _sparsity(RM, weights) + 0.5 * row_sparsity

        loss_fn = jit(_raw_loss)
        grad_fn = jit(grad(_raw_loss))

        w = jnp.array(_randn(n_w) * 1e-3)
        weights = jnp.ones_like(M_j)

        best_loss = float("inf")
        best_w = w

        if verbose:
            print(f"Optimizing rotation (JAX, param={param}, loss={loss})...")

        outer = n_reweight if loss == "reweighted_l1" else 1
        for r in range(outer):
            velocity = jnp.zeros_like(w)
            patience = 50
            no_improve = 0
            for i in range(maxiter):
                g = grad_fn(w, weights)
                velocity = momentum * velocity - learning_rate * g
                w = w + velocity
                cur = float(loss_fn(w, weights))
                if cur < best_loss:
                    best_loss = cur
                    best_w = w
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve > patience:
                    if verbose:
                        print(f"  outer {r+1}/{outer}: early stop at inner {i}")
                    break
                if verbose and i % 100 == 0:
                    print(f"  outer {r+1}/{outer} iter {i}: loss = {cur:.6f}")
            if loss == "reweighted_l1":
                R_cur = rot_fn(w, n_dim)
                RM_cur = R_cur @ M_j
                weights = 1.0 / jnp.sqrt(RM_cur * RM_cur + reweight_eps ** 2)
                if verbose:
                    print(f"  reweight {r+1}/{outer}: ||R M||_1 "
                          f"= {float(jnp.abs(RM_cur).sum()):.4f}")

        A_opt = np.array(rot_fn(best_w, n_dim))

    # ---------------------------------------------------------------
    # Diagnostics (common)
    # ---------------------------------------------------------------
    if verbose:
        M_sparse = A_opt @ M
        from scipy.stats import kurtosis

        ortho_check = A_opt.T @ A_opt
        ortho_error = np.linalg.norm(ortho_check - np.eye(n_dim), ord="fro")

        kurt_original = np.mean(kurtosis(M, axis=1, nan_policy="omit"))
        kurt_rotated = np.mean(kurtosis(M_sparse, axis=1, nan_policy="omit"))

        sparsity_original = np.mean(np.abs(M) < 0.01)
        sparsity_rotated = np.mean(np.abs(M_sparse) < 0.01)

        print("\nRotation optimization complete:")
        print(f"  param / loss        : {param} / {loss}")
        print(f"  Orthogonality error : {ortho_error:.6e}")
        print(f"  Kurtosis (original) : {kurt_original:.2f}")
        print(f"  Kurtosis (rotated)  : {kurt_rotated:.2f} (higher = sparser)")
        print(f"  Sparsity (original) : {sparsity_original:.2%}")
        print(f"  Sparsity (rotated)  : {sparsity_rotated:.2%}")

    return A_opt


def optimize_rotation_with_flattening(all_pars: List[np.ndarray],
                                      all_fns: List[Callable],
                                      linear_pars: List[List[float]],
                                      linear_indexes: List[List[int]],
                                      X: np.ndarray,
                                      Fs: np.ndarray,
                                      n_params: int,
                                      lambda_sparse: float = 1.0,
                                      lambda_flat: float = 10.0,
                                      alpha: float = 1.0,
                                      maxiter: int = 500,
                                      verbose: bool = True,
                                      param: str = "qr",
                                      loss: str = "logcosh",
                                      n_reweight: int = 5,
                                      reweight_eps: float = 1e-3,
                                      learning_rate: float = 0.01,
                                      momentum: float = 0.9,
                                      seed: Optional[int] = None) -> np.ndarray:
    """
    Optimize rotation matrix considering BOTH sparsity and flattening quality.
    
    This is the recommended approach as it finds A that:
    1. Makes A @ M sparse (fewer terms in expressions)
    2. Ensures Fisher matrices remain well-flattened

    Parameters
    ----------
    all_pars : List[np.ndarray]
        Parameter arrays for each component.
    all_fns : List[Callable]
        Callable functions for each component.
    linear_pars : List[List[float]]
        Linear parameters.
    linear_indexes : List[List[int]]
        Indices of linear parameters.
    X : np.ndarray
        Input data.
    Fs : np.ndarray
        Fisher matrices.
    n_params : int
        Number of parameters.
    lambda_sparse : float, default=1.0
        Weight for the sparsity term.
    lambda_flat : float, default=10.0
        Weight for the flattening term (passed through to
        :func:`lossfn_jac_jax`).
    alpha : float, default=1.0
        Scaling for the log-cosh sparsity loss (ignored when
        ``loss='reweighted_l1'``).
    maxiter : int, default=500
        Inner gradient-descent iterations (per outer reweight).
    verbose : bool, default=True
        Print progress.
    param : {'qr', 'cayley', 'expm'}, default='qr'
        Orthogonal parameterisation.  See :func:`optimize_sparse_rotation`
        for a full discussion.  ``'qr'`` reproduces the legacy behaviour
        exactly; ``'cayley'`` or ``'expm'`` give smoother gradients.
    loss : {'logcosh', 'reweighted_l1'}, default='logcosh'
        Sparsity surrogate.  ``'reweighted_l1'`` converges to an
        ``L0``-like solution over ``n_reweight`` outer iterations.
    n_reweight : int, default=5
        Outer reweighting iterations for ``loss='reweighted_l1'``.
    reweight_eps : float, default=1e-3
        Smoothing in the reweighted-L1 surrogate.
    learning_rate, momentum : float
        Inner-loop optimiser settings.
    seed : int, optional
        Seed for the random initialisation (``param='qr'`` starts at
        ``I``; Cayley/expm start at a tiny-random ``w``).

    Returns
    -------
    A_opt : np.ndarray
        Optimized rotation matrix.

    Notes
    -----
    The default ``param='qr'`` / ``loss='logcosh'`` combination reproduces
    the original behaviour exactly.  Prefer ``param='cayley'`` and
    ``loss='reweighted_l1'`` for realistic SR output.
    """
    if param not in ("qr", "cayley", "expm"):
        raise ValueError(f"param must be 'qr','cayley','expm'; got {param!r}")
    if loss not in ("logcosh", "reweighted_l1"):
        raise ValueError(f"loss must be 'logcosh' or 'reweighted_l1'; got {loss!r}")

    from jax import grad, jit

    M = construct_M(linear_pars, n_params)
    M_j = jnp.array(M)

    rng = np.random.default_rng(seed) if seed is not None else np.random

    def _sparsity(RM, weights):
        if loss == "logcosh":
            return jnp.sum(jnp.log(jnp.cosh(alpha * RM))) / alpha
        return jnp.sum(weights * jnp.sqrt(RM * RM + reweight_eps ** 2))

    # ---------- pick parameterisation ----------
    if param == "qr":
        def _build_A(x):
            return x.reshape((n_params, n_params))

        def _build_A_ortho(x):
            return get_Q_jax(x.reshape((n_params, n_params)))

        x0 = jnp.eye(n_params).flatten()  # legacy init
    else:
        rot_fn = cayley_rotation if param == "cayley" else expm_rotation
        n_w = skew_param_count(n_params)

        def _build_A(x):
            return rot_fn(x, n_params)

        def _build_A_ortho(x):
            return rot_fn(x, n_params)

        x0 = jnp.array(rng.standard_normal(n_w) * 1e-3)

    # ---------- combined loss ----------
    def _raw_loss(x, weights):
        A = _build_A(x)
        flat_loss = lossfn_jac_jax(
            A=A,
            all_pars=all_pars,
            all_fns=all_fns,
            linear_pars=linear_pars,
            linear_indexes=linear_indexes,
            X=X, Fs=Fs, n_params=n_params,
            alpha=0.5,
            lambda_flat=lambda_flat,
            smoothl1=True,
        )
        A_ortho = _build_A_ortho(x)
        RM = A_ortho @ M_j
        sparsity_loss = _sparsity(RM, weights)
        return flat_loss + lambda_sparse * sparsity_loss

    loss_fn = jit(_raw_loss)
    grad_fn = jit(grad(_raw_loss))

    x = x0
    weights = jnp.ones_like(M_j)
    best_loss = float("inf")
    best_x = x

    if verbose:
        print("Optimizing rotation with both sparsity and flattening constraints...")
        print(f"  lambda_sparse={lambda_sparse}, lambda_flat={lambda_flat}, "
              f"param={param}, loss={loss}")

    outer = n_reweight if loss == "reweighted_l1" else 1
    for r in range(outer):
        velocity = jnp.zeros_like(x)
        patience = 50
        no_improve = 0
        for i in range(maxiter):
            g = grad_fn(x, weights)
            velocity = momentum * velocity - learning_rate * g
            x = x + velocity
            cur = float(loss_fn(x, weights))
            if cur < best_loss:
                best_loss = cur
                best_x = x
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > patience:
                if verbose:
                    print(f"  outer {r+1}/{outer}: early stop at inner {i}")
                break
            if verbose and i % 50 == 0:
                print(f"  outer {r+1}/{outer} iter {i}: loss = {cur:.6f}")
        if loss == "reweighted_l1":
            A_cur = _build_A_ortho(x)
            RM_cur = A_cur @ M_j
            weights = 1.0 / jnp.sqrt(RM_cur * RM_cur + reweight_eps ** 2)
            if verbose:
                print(f"  reweight {r+1}/{outer}: ||R M||_1 "
                      f"= {float(jnp.abs(RM_cur).sum()):.4f}")

    A_opt = np.array(_build_A_ortho(best_x))

    if verbose:
        M_sparse = A_opt @ M
        from scipy.stats import kurtosis

        kurt_original = np.mean(kurtosis(M, axis=1, nan_policy='omit'))
        kurt_rotated = np.mean(kurtosis(M_sparse, axis=1, nan_policy='omit'))
        ortho_error = float(np.linalg.norm(
            A_opt.T @ A_opt - np.eye(n_params), ord='fro'
        ))

        print("\nOptimization complete:")
        print(f"  Final loss            : {best_loss:.6f}")
        print(f"  Orthogonality error   : {ortho_error:.3e}")
        print(f"  Kurtosis improvement  : {kurt_original:.2f} -> {kurt_rotated:.2f}")

    return A_opt


def postprocess_eqs(coordinates: List[str],
                    X: np.ndarray,
                    Fs: np.ndarray,
                    n_params: int,
                    A_rotation: Optional[np.ndarray] = None,
                    optimize_rotation: str = "none",
                    rotation_params: Optional[Dict] = None,
                    threshold: float = 0.05,
                    importance_based: bool = True,
                    batch_removal: bool = False,
                    batch_size: int = 5,
                    prune_constants: bool = False,
                    constant_threshold: Optional[float] = None,
                    remove_floats: bool = False,
                    decimal: int = 3,
                    rational: bool = False,
                    verbose: bool = True,
                    perturbation: float = 1e-4,
                    check_flattening_fn: Optional[Callable] = None,
                    module: str = "jax",
                    repair_degenerate_linear: bool = True,
                    repair_linear_scale: float = 1.0,
                    repair_y_atol: float = 1e-10,
                    repair_const_rel_atol: float = 1e-8) -> Tuple[List[str], List[List[float]], Optional[np.ndarray]]:
    """
    High-level wrapper for postprocessing symbolic expressions.
    
    This function provides a simplified interface that:
    1. Parses symbolic expression strings into components
    2. (Optional) Optimizes rotation for sparsity and/or flattening
    3. Applies importance-based pruning of linear coefficients
    4. (Optional) Prunes ALL constants including nonlinear ones (e.g. inside exp/log)
    5. Returns simplified expressions
    
    Parameters
    ----------
    coordinates : List[str]
        List of symbolic expression strings (one per coordinate)
    X : np.ndarray
        Input data of shape (n_samples, n_params)
    Fs : np.ndarray
        Fisher matrices of shape (n_samples, n_params, n_params)
    n_params : int
        Number of parameters
    A_rotation : np.ndarray, optional
        Pre-computed rotation matrix. If provided, optimize_rotation is ignored.
        Shape should be (n_params, n_params).
    optimize_rotation : str, default="none"
        How to optimize rotation matrix:
        - "none": Use identity or provided A_rotation
        - "sparse": Optimize for sparsity only (fast)
        - "full": Optimize for both sparsity and flattening (recommended but slower)
    rotation_params : Dict, optional
        Parameters for rotation optimization. Keys depend on optimize_rotation:

        - For "sparse": ``lambda_ortho``, ``alpha``, ``maxiter``, ``use_jax``,
          ``enforce_orthogonal``, plus the parameterisation / loss knobs
          ``param`` (``'qr'|'cayley'|'expm'``), ``loss``
          (``'logcosh'|'reweighted_l1'``), ``n_reweight``, ``reweight_eps``,
          ``learning_rate``, ``momentum``, ``seed``.  See
          :func:`optimize_sparse_rotation` for the full parameter list.
        - For "full": ``lambda_sparse``, ``lambda_flat``, ``alpha``,
          ``maxiter`` plus the same extended knobs as above.  See
          :func:`optimize_rotation_with_flattening`.

        The defaults (``param='qr'``, ``loss='logcosh'``) reproduce the
        legacy behaviour exactly.  For realistic SR output, prefer
        ``rotation_params={"param": "cayley", "loss": "reweighted_l1"}``.
    threshold : float, default=0.05
        Relative loss threshold for removing linear coefficients.
    importance_based : bool, default=True
        Use importance-based ordering (recommended). If False, uses legacy
        sequential pruning which is permutation-dependent.
    batch_removal : bool, default=False
        Attempt to remove multiple low-importance coefficients simultaneously.
    batch_size : int, default=5
        Number of coefficients to attempt removing in each batch.
    prune_constants : bool, default=False
        If True, run a second pass that prunes ALL constants in the output
        expressions, including nonlinear ones (e.g. inside exp, log, cos).
        This is the key option for simplifying expressions like
        exp(X1 + b*X3 + c*X5) down to exp(X1 + b*X3).
    constant_threshold : float, optional
        Threshold for constant pruning. If None, uses same value as threshold.
    remove_floats : bool, default=False
        Replace numeric floats with parameter names (b0, b1, etc.)
    decimal : int, default=3
        Number of decimal places for rounding
    rational : bool, default=False
        Use rational simplification in sympy
    verbose : bool, default=True
        Print progress and diagnostics
    perturbation : float, default=1e-4
        Finite difference step size for computing importance scores
    check_flattening_fn : Callable, optional
        Custom function to check flattening quality. If None, creates
        one automatically from X and Fs.
    module : str, default="jax"
        Module for lambdify ("jax" or "numpy")
    repair_degenerate_linear : bool, default=True
        If a pruned output is identically zero **or** collapses to a θ-independent
        constant, append a linear :math:`c X_j` term so it has at least a linear
        dependence on one θ component (see
        :func:`repair_degenerate_pruned_expressions`). Set False to disable.
    repair_linear_scale : float, default=1.0
        Coefficient :math:`c` in the repair term; use a small value (e.g. ``1e-3``)
        if the flatness score moves too much.
    repair_y_atol : float, default=1e-10
        Absolute tolerance for the "coordinate is identically zero" test on ``X``.
    repair_const_rel_atol : float, default=1e-8
        Relative tolerance for the "coordinate is constant in θ" test on ``X``.
        Set to ``0`` to disable constant detection (legacy zero-only behaviour).
        
    Returns
    -------
    pruned_expressions : List[str]
        Pruned and simplified symbolic expressions
    constants : List[List[float]]
        Constants in the pruned expressions
    A_rotation : np.ndarray or None
        The rotation matrix used (either provided or optimized)
        
    Examples
    --------
    >>> # Basic usage (linear pruning only)
    >>> pruned_exprs, consts, _ = postprocess_eqs(
    ...     coordinates=mdl_coordinates,
    ...     X=X_test, Fs=Fs_test, n_params=6
    ... )
    
    >>> # With nonlinear constant pruning (recommended for complex expressions)
    >>> pruned_exprs, consts, _ = postprocess_eqs(
    ...     coordinates=mdl_coordinates,
    ...     X=X_test, Fs=Fs_test, n_params=6,
    ...     prune_constants=True,
    ...     threshold=0.05
    ... )
    
    >>> # Full pipeline: rotation + linear pruning + constant pruning
    >>> pruned_exprs, consts, A_opt = postprocess_eqs(
    ...     coordinates=mdl_coordinates,
    ...     X=X_test, Fs=Fs_test, n_params=6,
    ...     optimize_rotation="full",
    ...     threshold=0.1,
    ...     prune_constants=True,
    ...     constant_threshold=0.05,
    ...     batch_removal=True
    ... )
    """
    if not ESR_AVAILABLE:
        raise ImportError(
            "ESR package is required for postprocessing. "
            "Please install it to use this function."
        )
    
    if len(coordinates) != n_params:
        raise ValueError(
            f"Number of coordinates ({len(coordinates)}) must match "
            f"n_params ({n_params})"
        )
    
    if verbose:
        print(f"Parsing {n_params} symbolic expressions...")
    
    # Parse all components
    all_pars = []
    all_linear_pars = []
    all_fns = []
    all_linear_inds = []
    all_param_dicts = []
    all_xs = []
    all_bs = []
    all_expr = []
    all_linear_labels = []
    
    for i in range(n_params):
        try:
            lab, expr, prs, linear_prs, _x, _b, eq_fn, param_dict, linear_inds = get_component(
                coordinates[i], i, X=X, module=module
            )
            
            all_pars.append(prs)
            all_linear_pars.append(linear_prs)
            all_fns.append(eq_fn)
            all_xs.append(_x)
            all_bs.append(_b)
            all_linear_inds.append(linear_inds)
            all_param_dicts.append(param_dict)
            all_expr.append(expr)
            all_linear_labels.append([_b[l] for l in linear_inds])
            
        except Exception as e:
            raise ValueError(
                f"Failed to parse coordinate {i}: {coordinates[i]}\n"
                f"Error: {str(e)}"
            )
    
    if verbose:
        total_linear = sum(len(lp) for lp in all_linear_pars)
        total_params = sum(len(p) for p in all_pars)
        print(f"Found {total_params} total parameters ({total_linear} linear)")
    
    # Determine rotation matrix
    if A_rotation is None and optimize_rotation != "none":
        # Optimize rotation
        if verbose:
            print(f"\nOptimizing rotation matrix (mode: {optimize_rotation})...")
        
        rotation_params = rotation_params or {}
        
        if optimize_rotation == "sparse":
            # Sparsity-only optimization
            M = construct_M(all_linear_pars, n_params)
            default_params = {
                'lambda_ortho': 1.0,
                'alpha': 1.0,
                'maxiter': 1000,
                'verbose': verbose,
                'use_jax': True,
                'enforce_orthogonal': True
            }
            default_params.update(rotation_params)
            A_rotation = optimize_sparse_rotation(M, **default_params)
            
        elif optimize_rotation == "full":
            # Joint sparsity + flattening optimization
            default_params = {
                'lambda_sparse': 1.0,
                'lambda_flat': 10.0,
                'alpha': 1.0,
                'maxiter': 500,
                'verbose': verbose
            }
            default_params.update(rotation_params)
            A_rotation = optimize_rotation_with_flattening(
                all_pars=all_pars,
                all_fns=all_fns,
                linear_pars=all_linear_pars,
                linear_indexes=all_linear_inds,
                X=X,
                Fs=Fs,
                n_params=n_params,
                **default_params
            )
        else:
            raise ValueError(
                f"Unknown optimize_rotation mode: {optimize_rotation}. "
                f"Must be 'none', 'sparse', or 'full'"
            )
    
    elif A_rotation is None:
        # Default to identity
        A_rotation = np.eye(n_params)
    else:
        # Use provided rotation
        A_rotation = np.array(A_rotation).reshape((n_params, n_params))
    
    if verbose:
        print(f"\nStarting pruning with threshold={threshold}...")
        if batch_removal:
            print(f"Using batch removal with batch_size={batch_size}")
    
    # Call the main pruning function
    pruned_expressions, constants = get_pruned_expressions_final(
        A=A_rotation,
        param_dicts=all_param_dicts,
        all_pars=all_pars,
        linear_pars=all_linear_pars,
        all_expressions=all_expr,
        linear_labels=all_linear_labels,
        X=X,
        Fs=Fs,
        n_params=n_params,
        check_flattening_fn=check_flattening_fn,
        remove_floats=remove_floats,
        decimal=decimal,
        rational=rational,
        threshold=threshold,
        verbose=verbose,
        importance_based=importance_based,
        perturbation=perturbation,
        batch_removal=batch_removal,
        batch_size=batch_size,
        repair_degenerate_linear=repair_degenerate_linear,
        repair_linear_scale=repair_linear_scale,
        repair_y_atol=repair_y_atol,
        repair_const_rel_atol=repair_const_rel_atol
    )
    
    # Optional second pass: prune ALL constants (including nonlinear)
    if prune_constants:
        if verbose:
            print("\n--- Second pass: pruning all constants (including nonlinear) ---")
        
        c_threshold = constant_threshold if constant_threshold is not None else threshold
        
        pruned_expressions, constants, _ = prune_all_constants(
            expressions=pruned_expressions,
            X=X,
            Fs=Fs,
            n_params=n_params,
            check_flattening_fn=check_flattening_fn,
            threshold=c_threshold,
            perturbation=perturbation,
            A=A_rotation,
            verbose=verbose
        )
    
    if verbose:
        print("\nPostprocessing complete!")
        print(f"Input expressions: {len(coordinates)}")
        print(f"Output expressions: {len(pruned_expressions)}")
    
    return pruned_expressions, constants, A_rotation



if __name__ == "__main__":
    # Run basic tests
    print("testing postprocessing utilities...")
    
    # Test smooth_l1_loss
    test_x = jnp.array([0.5, -0.5, 1.0, -1.0])
    loss = smooth_l1_loss(test_x)
    assert loss > 0, "Smooth L1 loss should be positive"
    print("  smooth_l1_loss: OK")
    
    # Test get_Q_jax
    A_test = jnp.array([[1.0, 0.5], [0.3, 1.0]])
    Q_test = get_Q_jax(A_test)
    assert jnp.allclose(Q_test @ Q_test.T, jnp.eye(2), atol=1e-5), "Q should be orthogonal"
    print("  get_Q_jax: OK")
    
    # Test split_by_punctuation
    tokens = split_by_punctuation("1.0 + 2.0 * X1")
    assert "+" in tokens and "*" in tokens, "should split by operators"
    print("  split_by_punctuation: OK")
    
    # Test replace_floats
    expr, vals = replace_floats("1.5 + 2.3 * X1")
    assert "b0" in expr and "b1" in expr, "should replace floats with parameters"
    assert len(vals) == 2, "should extract two float values"
    print("  replace_floats: OK")
    
    print("\nall tests passed!")
