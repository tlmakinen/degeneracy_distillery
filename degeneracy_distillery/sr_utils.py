"""
Symbolic Regression utilities for flattening coordinate discovery.

This module provides functions for:
- Fitting symbolic regression models to network outputs
- Computing description length (MDL) for equations
- Analyzing and ranking equations by multiple criteria
- Computing flattening metrics for symbolic expressions

Example Usage
-------------
>>> from src.sr_utils import fit_and_analyze_sr
>>> 
>>> # Data from preprocessing
>>> # X: parameters, y: network outputs, dy_sr: Jacobians, Fs: Fisher matrices
>>> 
>>> # Run complete SR pipeline with automatic train/test split
>>> # Fits all components by default
>>> mdl_coords, frob_coords, analysis, split_data = fit_and_analyze_sr(
...     X, y, y_std, dy_sr, Fs,
...     n_params=2,
...     parent_dir='./sr_results/',
...     test_size=0.5,         # 50% for validation
...     random_state=42,        # reproducible split
...     shuffle=True,           # shuffle before splitting
...     # SR hyperparameters (all optional, showing defaults)
...     time_limit=120,         # 2 minutes per component
...     max_length=25,          # max equation complexity
...     max_depth=10,           # max tree depth
...     allowed_symbols='add,mul,pow,constant,variable,exp,logabs,sqrt',
...     objectives=['r2', 'length']
... )
>>> 
>>> # Access test data for further analysis
>>> X_test = split_data['X_test']
>>> y_test = split_data['y_test']
>>> Fs_test = split_data['Fs_test']
>>> 
>>> # Or fit specific components only
>>> mdl_coords, frob_coords, analysis, split_data = fit_and_analyze_sr(
...     X, y, y_std, dy_sr, Fs,
...     n_params=2,
...     components_to_fit=[0, 1],  # fit only components 0 and 1
...     parent_dir='./sr_results/'
... )
>>> 
>>> # Advanced: slice Fisher matrices to subspace (assumes 1-to-1 component-parameter mapping)
>>> mdl_coords, frob_coords, analysis, split_data = fit_and_analyze_sr(
...     X, y, y_std, dy_sr, Fs,
...     n_params=3,
...     components_to_fit=[0, 2],  # fit components 0 and 2
...     slice_fisher=True           # work in 2D subspace [params 0, 2]
... )
>>> 
>>> # Use different equation sets for analysis
>>> mdl_coords, frob_coords, analysis, split_data = fit_and_analyze_sr(
...     X, y, y_std, dy_sr, Fs,
...     n_params=2,
...     equation_set='both'         # analyze both Pareto and full population
... )
>>> 
>>> print("Best MDL coordinates:", mdl_coords)
>>> print("Best Frobenius loss coordinates:", frob_coords)

Author: Ported from sr_dummy_functions.ipynb
"""

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import csv
import sys
import os
import string
import multiprocessing
from typing import List, Tuple, Dict, Optional, Callable, Any
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from pyoperon.sklearn import SymbolicRegressor
from scipy.optimize import least_squares, root
import sympy
import inspect

# Try importing ESR (required for MDL calculations)
try:
    import esr.generation.generator
    ESR_AVAILABLE = True
except ImportError:
    ESR_AVAILABLE = False


# =============================================================================
# PYOPERON VERSION COMPATIBILITY
# =============================================================================

def get_time_limit_param_name():
    """
    Determine the correct parameter name for time limit in PyOperon.
    Different versions use either 'time_limit' or 'max_time'.
    
    Returns
    -------
    str
        Either 'time_limit' or 'max_time'
    """
    sig = inspect.signature(SymbolicRegressor.__init__)
    if 'time_limit' in sig.parameters:
        return 'time_limit'
    elif 'max_time' in sig.parameters:
        return 'max_time'
    else:
        # Default to time_limit
        return 'time_limit'

# Cache the parameter name to avoid repeated inspection
_TIME_PARAM_NAME = get_time_limit_param_name()


# =============================================================================
# PARAMETER SCALING UTILITIES
# =============================================================================
#
# Pre-fishnet MinMax scaling of theta is the recommended convention: it
# stabilises Fisher conditioning, flow training, and SR constants when the
# physical parameters span very different magnitudes.  These helpers fit a
# scaler before training and convert the SR-discovered expressions back to
# physical-theta expressions afterwards, in one line each.
#
# Typical workflow:
#
#     # before fishnets
#     scaler = fit_theta_scaler(theta_train, feature_range=(1.0, 2.0))
#     theta_train_s = scaler.transform(theta_train)
#     theta_test_s  = scaler.transform(theta_test)
#     # ... fishnets / flattening / load_and_process_data_v2 / fit_and_analyze_sr ...
#     # SR runs in scaled coordinates and returns expressions in (X1, X2, ...)
#     # Optional `+sr_offset` (e.g. `+1`) is what was passed to PyOperon.
#
#     # after SR
#     physical_exprs = expressions_to_physical(
#         pruned_exprs, scaler, sr_offset=1.0,
#         theta_names=("beta", "gamma", "I0_over_10"),
#     )
# =============================================================================

def fit_theta_scaler(
    theta: np.ndarray,
    feature_range: Tuple[float, float] = (1.0, 2.0),
):
    """Fit a ``sklearn.preprocessing.MinMaxScaler`` on physical theta.

    Parameters
    ----------
    theta : array of shape (n_samples, n_params)
        Physical-theta training samples.  Fit only on training data and
        reuse the returned scaler on test / SR pools.
    feature_range : (low, high), default (1.0, 2.0)
        Target box for the scaled theta.  ``(1.0, 2.0)`` keeps every
        component strictly positive, of order unity, and avoids the zero
        branch that confuses log-friendly SR symbol sets.

    Returns
    -------
    sklearn.preprocessing.MinMaxScaler
        A fitted scaler exposing the affine map
        ``theta_scaled = scaler.scale_ * theta + scaler.min_``.
    """
    from sklearn.preprocessing import MinMaxScaler  # local import to keep top-level deps light
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(np.asarray(theta))
    return scaler


def expressions_to_physical(
    exprs,
    scaler,
    sr_offset: float = 0.0,
    theta_names: Optional[Tuple[str, ...]] = None,
    simplify: bool = True,
    decimal: Optional[int] = None,
):
    """Convert SR expressions in scaled-theta coordinates back to physical theta.

    Symbolic regression typically runs on ``X = scaler.transform(theta) + sr_offset``
    (the optional ``+1`` shift used in several reference notebooks keeps PyOperon
    inputs strictly positive after Procrustes alignment).  The returned sympy
    expressions are then in the variables ``X1, X2, ...``.  This helper
    substitutes

        X_i  ->  scaler.scale_[i] * theta_i + scaler.min_[i] + sr_offset

    and returns expressions in the physical theta symbols, ready for printing,
    further simplification, or evaluation.

    Parameters
    ----------
    exprs : Sequence[str | sympy.Expr]
        SR-discovered coordinate expressions, one per learned eta component.
    scaler : object with ``scale_`` and ``min_`` attributes
        A fitted scaler (e.g. from :func:`fit_theta_scaler`).
    sr_offset : float, default 0.0
        Constant shift that was added to ``X`` before SR (commonly ``1.0``).
    theta_names : tuple of str, optional
        Symbol names for the physical thetas.  Default: ``("theta1", ..., "thetaN")``.
    simplify : bool, default True
        Run ``sympy.simplify`` on each substituted expression.
    decimal : int, optional
        If given, round numerical coefficients to this many decimals via
        ``sympy.nsimplify`` after substitution.

    Returns
    -------
    list of sympy.Expr
        Same length as ``exprs``, each in the symbols named by ``theta_names``.
    """
    a = np.asarray(scaler.scale_, dtype=float)
    b = np.asarray(scaler.min_, dtype=float)
    n = len(a)
    if theta_names is None:
        theta_names = tuple(f"theta{i+1}" for i in range(n))
    if len(theta_names) != n:
        raise ValueError(
            f"theta_names has length {len(theta_names)} but scaler implies n={n}"
        )

    X_syms = sympy.symbols(" ".join(f"X{i+1}" for i in range(n)))
    if isinstance(X_syms, sympy.Symbol):
        X_syms = (X_syms,)
    th_syms = sympy.symbols(" ".join(theta_names))
    if isinstance(th_syms, sympy.Symbol):
        th_syms = (th_syms,)

    subs = {
        X_syms[i]: float(a[i]) * th_syms[i] + float(b[i]) + float(sr_offset)
        for i in range(n)
    }

    out = []
    for e in exprs:
        ex = sympy.sympify(e) if not isinstance(e, sympy.Expr) else e
        ex = ex.subs(subs)
        if simplify:
            ex = sympy.simplify(ex)
        if decimal is not None:
            ex = ex.xreplace({
                num: round(float(num), int(decimal))
                for num in ex.atoms(sympy.Float)
            })
        out.append(ex)
    return out


# =============================================================================
# SAFE LAMBDIFY HELPER
# =============================================================================

def safe_lambdify(args, expr, preferred_modules=None):
    """Wrapper around sympy.lambdify with fallback module handling.

    Tries multiple module backends in sequence to handle expressions that
    sympy can't convert with certain backends (e.g., JAX).

    Parameters
    ----------
    args : list
        Symbols to use as function arguments
    expr : sympy.Expr
        Expression to compile
    preferred_modules : list or str, optional
        First module backend to try. Default: ["jax"]

    Returns
    -------
    callable
        Compiled function from sympy.lambdify

    Raises
    ------
    Exception
        If all module backends fail, re-raises the last exception.
    """
    if preferred_modules is None:
        preferred_modules = ["jax"]

    attempts = [preferred_modules, ["numpy"], "numpy"]
    last_exc = None

    for modules in attempts:
        try:
            return sympy.lambdify(args, expr, modules=modules)
        except Exception as e:
            last_exc = e

    raise last_exc


def promote_jacrev_output(y):
    """Cast SR lambdify output to a floating dtype so ``jax.jacrev`` accepts it.

    SymPy/JAX lambdify sometimes yields integer dtypes (e.g. from integer literals
    in the parse tree), which triggers ``jacrev``'s real-floating requirement.
    """
    y = jnp.asarray(y)
    return jnp.asarray(y, dtype=jnp.promote_types(jnp.float32, y.dtype))


def vmapped_jacobian_wrt_X(myeq, X: jnp.ndarray) -> jnp.ndarray:
    """Jacobian :math:`\\partial y / \\partial \\theta` with shape ``(n_samples, n_params)``.

    For scalar ``myeq(θ₁,…,θₚ)`` evaluated along rows of ``X`` we use a single
    vector argument ``z ∈ ℝᵖ`` and ``jax.jacrev`` on ``z ↦ myeq(*split(z))``.
    This avoids JAX-version differences in how ``jacrev(..., argnums=(0,…,p-1))``
    combined with ``vmap`` returns either a tuple of partials, a stacked array,
    or other layouts — which previously led to accidental transposes and bogus
    flattening scores.
    """
    X = jnp.asarray(X)
    _, n_params = X.shape

    def one_row(z: jnp.ndarray) -> jnp.ndarray:
        def f(t: jnp.ndarray) -> jnp.ndarray:
            t_parts = jnp.split(t.reshape(-1), int(n_params))
            t_parts = tuple(p.reshape(()) for p in t_parts)
            return myeq(*t_parts)

        out = jax.jacrev(f)(z.reshape(-1))
        return jnp.ravel(out)

    return jax.vmap(one_row)(X)


def print_sr_utils_location() -> None:
    """Print the loaded ``sr_utils`` path (use in notebooks to confirm editable installs)."""
    import degeneracy_distillery.sr_utils as _m

    print(_m.__file__)


# =============================================================================
# STRING PARSING UTILITIES
# =============================================================================

def split_by_punctuation(s: str) -> List[str]:
    """
    Convert a string into a list, where the string is split by punctuation,
    excluding underscores or full stops.
    
    For example, the string 'he_ll*o.w0%rl^d' becomes
    ['he_ll', '*', 'o.w0', '%', 'rl', '^', 'd']
    
    Parameters
    ----------
    s : str
        The string to split up
        
    Returns
    -------
    split_str : list[str]
        The string split by punctuation
    """
    pun = string.punctuation.replace('_', '')  # allow underscores in variable names
    pun = pun.replace('.', '')  # allow full stops
    pun = pun + ' '
    where_pun = [i for i in range(len(s)) if s[i] in pun]
    
    if len(where_pun) > 0:
        split_str = [s[:where_pun[0]]]
        for i in range(len(where_pun) - 1):
            split_str += [s[where_pun[i]]]
            split_str += [s[where_pun[i] + 1:where_pun[i + 1]]]
        split_str += [s[where_pun[-1]]]
        if where_pun[-1] != len(s) - 1:
            split_str += [s[where_pun[-1] + 1:]]
    else:
        split_str = [s]
        
    # Remove spaces
    split_str = [s.strip() for s in split_str if len(s) > 0 and (not s.isspace())]
    
    return split_str


def is_float(s: str) -> bool:
    """
    Function to determine whether a string has a numeric value.
    
    Parameters
    ----------
    s : str
        The string of interest
        
    Returns
    -------
    bool
        True if s has a numeric value, False otherwise
    """
    try:
        float(eval(s))
        return True
    except:
        return False


def replace_floats(s: str) -> Tuple[str, List[float]]:
    """
    Replace the floats in a string by parameters named b0, b1, ...
    where each float (even if they have the same value) is assigned a
    different b.
    
    Parameters
    ----------
    s : str
        The string to consider
        
    Returns
    -------
    replaced : str
        The same string, but with floats replaced by parameter names
    values : list[float]
        The values of the parameters in order [b0, b1, ...]
    """
    split_str = split_by_punctuation(s)
    values = []
    
    # Initial pass at replacing floats
    for i in range(len(split_str)):
        if is_float(split_str[i]) and "." in split_str[i]:
            values.append(float(split_str[i]))
            split_str[i] = f'b{len(values) - 1}'
        elif len(split_str[i]) > 1 and split_str[i][-1] == 'e' and is_float(split_str[i][:-1]):
            if split_str[i + 1] in ['+', '-']:
                values.append(float(''.join(split_str[i:i + 3])))
                split_str[i] = f'b{len(values) - 1}'
                split_str[i + 1] = ''
                split_str[i + 2] = ''
            else:
                assert split_str[i + 1].isdigit()
                values.append(float(''.join(split_str[i:i + 2])))
                split_str[i] = f'b{len(values) - 1}'
                split_str[i + 1] = ''
    
    # Now check for negative parameters
    for i in range(len(values)):
        idx = split_str.index(f'b{i}')
        if (idx == 1) and (split_str[0] == '-'):
            split_str[0] = ''
            values[i] *= -1
        elif (split_str[idx - 1] == '-') and (split_str[idx - 2] in ['+', '-', '*', '/', '(', '^']):
            values[i] *= -1
            split_str[idx - 1] = ''

    # Rejoin string
    replaced = ''.join(split_str)

    return replaced, values


def check_symbolic_invertibility(
    expressions: List[str],
    input_symbols: Optional[List[Any]] = None,
    output_symbols: Optional[List[Any]] = None,
    input_prefix: str = "X",
    output_prefix: str = "Y",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Try to symbolically invert SR coordinate expressions with SymPy.

    Parameters
    ----------
    expressions : list[str]
        Coordinate expressions, e.g. ``["X1 * X2", "5.6 * X2"]``.
    input_symbols : list, optional
        Symbols to solve for. If omitted, symbols with names like ``X1``,
        ``X2``, ... are inferred from the expressions.
    output_symbols : list, optional
        Symbols for the transformed coordinates. If omitted, uses ``Y1``,
        ``Y2``, ... with one output per expression.
    input_prefix, output_prefix : str
        Prefixes used when inferring/generating symbols.
    verbose : bool
        If True, print the forward equations and any inverse solutions.

    Returns
    -------
    dict
        Contains parsed expressions, equations, solutions, inverse coordinate
        expressions, Jacobian, Jacobian determinant, and boolean summaries for
        square/local/global checks.
    """
    local_dict = {
        "exp": sympy.exp,
        "log": sympy.log,
        "sqrt": sympy.sqrt,
        "cos": sympy.cos,
        "Abs": sympy.Abs,
        "abs": sympy.Abs,
        "logAbs": lambda x: sympy.log(sympy.Abs(x)),
        "inv": lambda x: 1 / x,
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
    }
    parsed_exprs = [sympy.sympify(expr, locals=local_dict) for expr in expressions]

    def _symbol_sort_key(symbol):
        name = str(symbol)
        prefix = "".join(ch for ch in name if not ch.isdigit())
        suffix = name[len(prefix):]
        return prefix, int(suffix) if suffix.isdigit() else suffix

    if input_symbols is None:
        input_symbols = sorted(
            {
                symbol
                for expr in parsed_exprs
                for symbol in expr.free_symbols
                if str(symbol).startswith(input_prefix)
            },
            key=_symbol_sort_key,
        )
    else:
        input_symbols = [sympy.Symbol(symbol) if isinstance(symbol, str) else symbol for symbol in input_symbols]

    if output_symbols is None:
        output_symbols = list(sympy.symbols(f"{output_prefix}1:{len(parsed_exprs) + 1}"))
    else:
        output_symbols = [sympy.Symbol(symbol) if isinstance(symbol, str) else symbol for symbol in output_symbols]

    if len(output_symbols) != len(parsed_exprs):
        raise ValueError(
            f"Expected {len(parsed_exprs)} output symbols, got {len(output_symbols)}."
        )

    equations = [
        sympy.Eq(output_symbol, expr)
        for output_symbol, expr in zip(output_symbols, parsed_exprs)
    ]
    solutions = sympy.solve(
        equations,
        input_symbols,
        dict=True,
        simplify=False,
        rational=False,
    )

    jacobian = None
    jacobian_det = None
    is_square = len(parsed_exprs) == len(input_symbols)
    if is_square:
        jacobian = sympy.Matrix(parsed_exprs).jacobian(input_symbols)
        jacobian_det = sympy.simplify(jacobian.det())

    is_locally_invertible = bool(is_square and jacobian_det is not None and jacobian_det != 0)
    is_symbolically_invertible = bool(
        is_square
        and len(solutions) == 1
        and all(symbol in solutions[0] for symbol in input_symbols)
    )
    complete_solutions = [
        solution
        for solution in solutions
        if all(symbol in solution for symbol in input_symbols)
    ]
    inv_coord_solutions = [
        inverse_solution_to_sr_coords(solution, input_symbols, output_symbols)
        for solution in complete_solutions
    ]
    inv_coords = inv_coord_solutions[0] if inv_coord_solutions else None

    if verbose:
        print("Forward map:")
        for equation in equations:
            print(f"  {equation.lhs} = {equation.rhs}")

        print(f"\nSolving for {input_symbols} in terms of {output_symbols}:")
        if solutions:
            for i, solution in enumerate(solutions, start=1):
                label = f"Solution {i}" if len(solutions) > 1 else "Solution"
                print(f"  {label}:")
                for symbol in input_symbols:
                    print(f"    {symbol} = {solution.get(symbol, '<unsolved>')}")
        else:
            print("  No symbolic inverse found by sympy.solve.")

        if inv_coord_solutions:
            label = "Inverse coordinates for get_y_sr/in terms of X"
            if len(inv_coord_solutions) > 1:
                label += " (first branch shown; see inv_coord_solutions for all branches)"
            print(f"\n{label}:")
            for i, coord in enumerate(inv_coords, start=1):
                print(f"  X{i} = {coord}")

        if jacobian_det is not None:
            print(f"\nJacobian determinant: {jacobian_det}")
            print(f"Locally invertible where determinant != 0: {is_locally_invertible}")

    return {
        "expressions": parsed_exprs,
        "input_symbols": input_symbols,
        "output_symbols": output_symbols,
        "equations": equations,
        "solutions": solutions,
        "inv_coords": inv_coords,
        "inv_coord_solutions": inv_coord_solutions,
        "jacobian": jacobian,
        "jacobian_det": jacobian_det,
        "is_square": is_square,
        "is_locally_invertible": is_locally_invertible,
        "has_symbolic_inverse": bool(inv_coord_solutions),
        "is_symbolically_invertible": is_symbolically_invertible,
    }


def inverse_solution_to_sr_coords(
    solution: Dict[Any, Any],
    input_symbols: List[Any],
    output_symbols: List[Any],
    output_prefix: str = "X",
    log_to_logabs: bool = True,
    precision: int = 12,
) -> List[str]:
    """Convert one SymPy inverse solution to ``get_y_sr``-style expressions.

    ``check_symbolic_invertibility`` solves ``Y = f(X)`` and SymPy returns
    expressions for the original inputs in terms of output symbols ``Y1``,
    ``Y2``, ... . This helper rewrites those output symbols as ``X1``, ``X2``,
    ... so the inverse can be passed directly to ``get_y_sr(inv_coords, eta)``.
    By default, plain ``log(...)`` terms are emitted as ``logAbs(...)`` for
    better portability on sampled coordinates that may cross sign boundaries.
    """
    input_symbols = [sympy.Symbol(symbol) if isinstance(symbol, str) else symbol for symbol in input_symbols]
    output_symbols = [sympy.Symbol(symbol) if isinstance(symbol, str) else symbol for symbol in output_symbols]
    coord_symbols = list(sympy.symbols(f"{output_prefix}1:{len(output_symbols) + 1}"))
    substitutions = dict(zip(output_symbols, coord_symbols))
    logAbs = sympy.Function("logAbs")

    def _format_expr(expr: sympy.Expr) -> str:
        expr = sympy.factor(expr)
        if log_to_logabs:
            expr = expr.replace(
                lambda node: node.func == sympy.log and len(node.args) == 1,
                lambda node: logAbs(node.args[0]),
            )
        return str(expr.evalf(precision))

    coords = []
    for symbol in input_symbols:
        if symbol not in solution:
            raise ValueError(f"Solution is missing inverse expression for {symbol}.")
        coords.append(_format_expr(solution[symbol].subs(substitutions)))
    return coords


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
        ["square", "exp", "inv", "sqrt", "log", "cos", "logAbs"],
        ["+", "*", "-", "/", "^"],
    ]

    a, b = sympy.symbols("a b", real=True)
    inv = sympy.Lambda(a, 1 / a)
    square = sympy.Lambda(a, a * a)
    sqrt = sympy.Lambda(a, sympy.sqrt(a))
    log = sympy.Lambda(a, sympy.log(a))
    logAbs = sympy.Lambda(a, sympy.log(sympy.Abs(a)))
    power = sympy.Lambda((a, b), sympy.Pow(a, b))

    sympy_locs = {
        "inv": inv,
        "square": square,
        "cos": sympy.cos,
        "^": power,
        "Abs": sympy.Abs,
        "sqrt": sqrt,
        "log": log,
        "logAbs": logAbs,
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


def get_inv_y_sr(
    expressions: List[str],
    y: np.ndarray,
    initial_guess: Optional[np.ndarray] = None,
    input_symbols: Optional[List[Any]] = None,
    input_prefix: str = "X",
    method: str = "hybr",
    bounds: Optional[Tuple[Any, Any]] = None,
    warm_start: bool = True,
    solver_options: Optional[Dict[str, Any]] = None,
    residual_tol: Optional[float] = None,
    raise_on_fail: bool = True,
    full_output: bool = False,
) -> np.ndarray:
    """Numerically invert symbolic-regression coordinates row-by-row.

    This is the inverse analogue of ``get_y_sr(expressions, data)``: given
    target coordinates ``y``, it solves ``expressions(X) = y`` for ``X`` using
    SciPy's nonlinear root finder. This is useful when SymPy cannot provide a
    closed-form inverse, e.g. for transcendental SR expressions.

    Parameters
    ----------
    expressions : list[str]
        One SR expression per output dimension, written in terms of ``X1``,
        ``X2``, ...
    y : np.ndarray, shape (n_samples, n_outputs) or (n_outputs,)
        Target coordinate values to invert.
    initial_guess : np.ndarray, optional
        Initial guess for the original coordinates. May be shape ``(n_inputs,)``
        or ``(n_samples, n_inputs)``. If omitted, zeros are used.
    input_symbols : list, optional
        Symbols to solve for. If omitted, they are inferred from expressions by
        ``input_prefix`` and sorted as ``X1, X2, ...``.
    input_prefix : str
        Prefix used when inferring input symbols.
    method : str
        Method passed to ``scipy.optimize.root``. Use ``"least_squares"`` for
        bounded nonlinear least-squares via ``scipy.optimize.least_squares``.
    bounds : tuple, optional
        Lower and upper bounds for ``method="least_squares"``.
    warm_start : bool
        If True and ``initial_guess`` is one-dimensional, each row uses the
        previous row's solution as its next initial guess.
    solver_options : dict, optional
        Options passed to ``scipy.optimize.root`` or keyword arguments passed
        to ``scipy.optimize.least_squares``.
    residual_tol : float, optional
        If provided, mark solutions with residual norm above this value as
        failed even when the optimizer reports convergence.
    raise_on_fail : bool
        If True, raise ``RuntimeError`` when any row does not converge.
    full_output : bool
        If True, return ``(x, diagnostics)`` instead of just ``x``.

    Returns
    -------
    x : np.ndarray, shape (n_samples, n_inputs)
        Numerically recovered input coordinates.
    diagnostics : list[dict], optional
        Returned only when ``full_output=True``. Contains solver status per row.
    """
    target = np.asarray(y, dtype=float)
    if target.ndim == 1:
        if len(expressions) == 1:
            target = target.reshape(-1, 1)
        else:
            target = target.reshape(1, -1)
    elif target.ndim != 2:
        raise ValueError("y must have shape (n_samples, n_outputs) or (n_outputs,).")

    if len(expressions) != target.shape[1]:
        raise ValueError(
            f"Expected y to have {len(expressions)} columns, got {target.shape[1]}."
        )

    local_dict = {
        "exp": sympy.exp,
        "log": sympy.log,
        "sqrt": sympy.sqrt,
        "cos": sympy.cos,
        "sin": sympy.sin,
        "tan": sympy.tan,
        "Abs": sympy.Abs,
        "abs": sympy.Abs,
        "logAbs": lambda x: sympy.log(sympy.Abs(x)),
        "inv": lambda x: 1 / x,
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
    }
    parsed_exprs = [
        sympy.sympify(expr, locals=local_dict, convert_xor=True)
        for expr in expressions
    ]

    def _symbol_sort_key(symbol):
        name = str(symbol)
        prefix = "".join(ch for ch in name if not ch.isdigit())
        suffix = name[len(prefix):]
        return prefix, int(suffix) if suffix.isdigit() else suffix

    if input_symbols is None:
        input_symbols = sorted(
            {
                symbol
                for expr in parsed_exprs
                for symbol in expr.free_symbols
                if str(symbol).startswith(input_prefix)
            },
            key=_symbol_sort_key,
        )
    else:
        input_symbols = [sympy.Symbol(symbol) if isinstance(symbol, str) else symbol for symbol in input_symbols]

    n_inputs = len(input_symbols)
    if n_inputs == 0:
        raise ValueError("No input symbols found to solve for.")
    if len(expressions) != n_inputs:
        raise ValueError(
            "Numerical inversion requires a square system: "
            f"{len(expressions)} expressions for {n_inputs} inputs."
        )

    if initial_guess is None:
        guesses = np.zeros((target.shape[0], n_inputs), dtype=float)
        use_single_warm_guess = True
    else:
        guesses = np.asarray(initial_guess, dtype=float)
        if guesses.ndim == 1:
            if guesses.shape[0] != n_inputs:
                raise ValueError(
                    f"initial_guess must have length {n_inputs}, got {guesses.shape[0]}."
                )
            guesses = np.repeat(guesses.reshape(1, -1), target.shape[0], axis=0)
            use_single_warm_guess = True
        elif guesses.ndim == 2:
            if guesses.shape != (target.shape[0], n_inputs):
                raise ValueError(
                    "2D initial_guess must have shape "
                    f"{(target.shape[0], n_inputs)}, got {guesses.shape}."
                )
            use_single_warm_guess = False
        else:
            raise ValueError("initial_guess must be one- or two-dimensional.")

    forward_fn = sympy.lambdify(input_symbols, parsed_exprs, modules="numpy")
    solver_options = {} if solver_options is None else dict(solver_options)
    if bounds is None:
        lower_bounds = np.full(n_inputs, -np.inf, dtype=float)
        upper_bounds = np.full(n_inputs, np.inf, dtype=float)
    else:
        lower_bounds = np.asarray(bounds[0], dtype=float)
        upper_bounds = np.asarray(bounds[1], dtype=float)
        if lower_bounds.shape != (n_inputs,) or upper_bounds.shape != (n_inputs,):
            raise ValueError(
                "bounds must be a tuple of arrays with shape "
                f"{(n_inputs,)}, got {lower_bounds.shape} and {upper_bounds.shape}."
            )

    results = np.empty((target.shape[0], n_inputs), dtype=float)
    diagnostics = []
    current_guess = guesses[0].copy()

    def _evaluate(x):
        value = forward_fn(*x)
        value = np.asarray(value, dtype=float)
        return np.reshape(value, (len(expressions),))

    for i, target_row in enumerate(target):
        if not (warm_start and use_single_warm_guess):
            current_guess = guesses[i].copy()

        def residuals(x):
            return _evaluate(x) - target_row

        if method == "least_squares":
            current_guess = np.clip(current_guess, lower_bounds, upper_bounds)
            sol = least_squares(
                residuals,
                current_guess,
                bounds=(lower_bounds, upper_bounds),
                **solver_options,
            )
        else:
            sol = root(residuals, current_guess, method=method, options=solver_options)

        residual_norm = float(np.linalg.norm(residuals(sol.x)))
        success = bool(sol.success)
        if residual_tol is not None and residual_norm > residual_tol:
            success = False

        results[i] = sol.x
        diagnostics.append(
            {
                "success": success,
                "status": sol.status,
                "message": sol.message,
                "nfev": getattr(sol, "nfev", None),
                "residual_norm": residual_norm,
            }
        )

        if raise_on_fail and not success:
            raise RuntimeError(
                f"Failed to invert row {i}: {sol.message}. "
                f"Residual norm={diagnostics[-1]['residual_norm']:.3e}."
            )

        if warm_start:
            current_guess = sol.x

    if full_output:
        return results, diagnostics
    return results


# =============================================================================
# DESCRIPTION LENGTH AND FLATTENING
# =============================================================================

def _parse_sr_equation(eq: str) -> Tuple[sympy.Expr, Any, float, List[float]]:
    """Parse an Operon/ESR equation string (same grammar as :func:`compute_DL`)."""
    if not ESR_AVAILABLE:
        raise ImportError(
            "ESR is required for parsing equations but not installed.\n"
            "Install it with:\n"
            "  !git clone https://github.com/DeaglanBartlett/ESR.git\n"
            "  !pip install -e ESR\n"
            "See COLAB_SETUP.md for full installation instructions."
        )

    basis_functions = [
        ["X", "b"],  # type0
        ["square", "exp", "inv", "sqrt", "log", "cos", "logAbs"],  # type1
        ["+", "*", "-", "/", "^"],  # type2
    ]

    a, b = sympy.symbols('a b', real=True)
    sympy.init_printing(use_unicode=True)
    inv = sympy.Lambda(a, 1 / a)
    square = sympy.Lambda(a, a * a)
    cube = sympy.Lambda(a, a * a * a)
    sqrt = sympy.Lambda(a, sympy.sqrt(a))
    log = sympy.Lambda(a, sympy.log(a))
    logAbs = sympy.Lambda(a, sympy.log(sympy.Abs(a)))
    power = sympy.Lambda((a, b), sympy.Pow(a, b))

    sympy_locs = {
        "inv": inv,
        "square": square,
        "cube": cube,
        "cos": sympy.cos,
        "^": power,
        "Abs": sympy.Abs,
        "sqrt": sqrt,
        "log": log,
        "logAbs": logAbs,
    }

    expr_str, par_values = replace_floats(eq)
    expr, nodes, complexity = esr.generation.generator.string_to_node(
        expr_str,
        basis_functions,
        evalf=True,
        allow_eval=True,
        check_ops=True,
        locs=sympy_locs,
    )
    return expr, nodes, complexity, par_values


def max_exp_nesting_depth(expr: sympy.Expr) -> int:
    """
    Longest chain exp(exp(...arg)) along any branch of the expression tree.

    Examples
    --------
    ``exp(X1)`` → 1; ``exp(exp(X1))`` → 2; ``exp(X1)+exp(X2)`` → 1.
    """
    if expr.func == sympy.exp:
        return 1 + max_exp_nesting_depth(expr.args[0])
    if not expr.args:
        return 0
    return max(max_exp_nesting_depth(a) for a in expr.args)


def max_pow_nesting_depth(expr: sympy.Expr) -> int:
    """
    Longest chain of nested ``Pow`` nodes along any branch (base or exponent).

    Examples
    --------
    ``X1**2`` → 1; ``(X1**2)**3`` → 2; ``X1**(X2**2)`` → 2.
    """
    if isinstance(expr, sympy.Pow):
        base, ex = expr.as_base_exp()
        return 1 + max(max_pow_nesting_depth(base), max_pow_nesting_depth(ex))
    if not expr.args:
        return 0
    return max(max_pow_nesting_depth(a) for a in expr.args)


def _is_sympy_log_abs(expr: sympy.Expr) -> bool:
    """True for ``log(Abs(u))``, i.e. the SR ``logAbs`` unary as parsed in :func:`_parse_sr_equation`."""
    return (
        expr.func == sympy.log
        and len(expr.args) == 1
        and expr.args[0].func == sympy.Abs
    )


def max_log_nesting_depth(expr: sympy.Expr) -> int:
    """
    Longest chain ``log(log(...arg))`` along any branch of the expression tree.

    Examples
    --------
    ``log(X1)`` → 1; ``log(log(X1))`` → 2; ``log(X1)+log(X2)`` → 1.
    """
    if expr.func == sympy.log:
        return 1 + max_log_nesting_depth(expr.args[0])
    if not expr.args:
        return 0
    return max(max_log_nesting_depth(a) for a in expr.args)


def max_logabs_nesting_depth(expr: sympy.Expr) -> int:
    """
    Longest chain of nested ``log(Abs(...))`` (SR ``logAbs``) along any branch.

    Only counts when the outer node is exactly ``log`` with a single ``Abs``
    child, matching the grammar's ``logAbs`` operator — not plain ``log(u)``.

    Examples
    --------
    ``log(Abs(X1))`` → 1; ``log(Abs(log(Abs(X1))))`` → 2.
    """
    if _is_sympy_log_abs(expr):
        inner = expr.args[0].args[0]
        return 1 + max_logabs_nesting_depth(inner)
    if not expr.args:
        return 0
    return max(max_logabs_nesting_depth(a) for a in expr.args)


def _is_sr_coordinate_symbol(s: sympy.Basic) -> bool:
    """True for symbols named ``X1``, ``X2``, … as produced by the SR/ESR grammar."""
    if not isinstance(s, sympy.Symbol):
        return False
    name = s.name
    return len(name) >= 2 and name.startswith("X") and name[1:].isdigit()


def _expr_depends_on_sr_coordinate(e: sympy.Expr) -> bool:
    return any(_is_sr_coordinate_symbol(s) for s in e.free_symbols)


def _mul_remove_one_factor(term: sympy.Expr, factor: sympy.Expr) -> sympy.Expr:
    """Remove one instance of ``factor`` from a product (``term`` is ``expand_mul``'d)."""
    if isinstance(term, sympy.Mul):
        args = list(term.args)
    else:
        if term == factor:
            return sympy.S.One
        return term
    removed = False
    kept: List[sympy.Expr] = []
    for a in args:
        if not removed and a == factor:
            removed = True
            continue
        kept.append(a)
    if not removed:
        return term
    if not kept:
        return sympy.S.One
    if len(kept) == 1:
        return kept[0]
    return sympy.Mul(*kept)


def _term_is_exp_log_form_of_x_power(term: sympy.Expr) -> bool:
    """
    True if ``term`` is (after ``expand_mul``) a product ``c * log(u)`` encoding
    ``u**c`` via ``exp(term)``, with coordinate dependence in both ``u`` and ``c``.

    Rejects only decompositions where the cofactor of a single top-level ``log``
    has no ``log`` (so ``exp(X1*log(X2)*log(X3))`` is not treated as a plain
    two-variable power).
    """
    term = sympy.expand_mul(term)
    factors = list(term.args) if isinstance(term, sympy.Mul) else [term]
    for fac in factors:
        if not isinstance(fac, sympy.log):
            continue
        u = fac.args[0]
        c = sympy.expand_mul(_mul_remove_one_factor(term, fac))
        if c.has(sympy.log):
            continue
        if _expr_depends_on_sr_coordinate(u) and _expr_depends_on_sr_coordinate(c):
            return True
    return False


def _expr_has_exp_log_x_power_form(expr: sympy.Expr) -> bool:
    """Detect ``exp(… + …)`` shapes equivalent to ``Pow`` with ``Xi`` in exponent."""
    for node in sympy.preorder_traversal(expr):
        if node.func != sympy.exp:
            continue
        arg = sympy.expand_mul(sympy.expand(node.args[0]))
        terms = sympy.Add.make_args(arg) if isinstance(arg, sympy.Add) else (arg,)
        for term in terms:
            if _term_is_exp_log_form_of_x_power(term):
                return True
    return False


def expr_has_pow_with_x_in_exponent(expr: sympy.Expr) -> bool:
    """
    True if the expression encodes a coordinate-dependent exponent on a
    coordinate-dependent base (forbidden ``Xi**Xj``-style structure).

    This covers:

    - Any explicit ``Pow`` (including ``^`` from SR) whose exponent depends on
      some ``Xi`` (symbol names ``X`` + digits).
    - The common SR encoding ``exp(c * log(u))`` (and sums thereof) when both
      ``u`` and ``c`` depend on coordinates — e.g. ``(a*X2)**(b*X1)`` as
      ``exp((b*X1)*log(a*X2))``, which has no ``Pow`` node.

    Matching is by **symbol name** (``X`` + digits), not set intersection with
    separately constructed ``Symbol`` objects, so this still works when
    ``n_params`` passed to :func:`sr_structure_predicate` is wrong or when SymPy
    uses distinct instances for the same ``X2`` label.
    """
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.Pow):
            _base, ex = node.as_base_exp()
            if any(_is_sr_coordinate_symbol(t) for t in ex.free_symbols):
                return True
    return _expr_has_exp_log_x_power_form(expr)


def _sr_coordinate_symbols(e: sympy.Expr) -> set[sympy.Symbol]:
    """Coordinate symbols named ``X1``, ``X2``, ... used by an expression."""
    return {s for s in e.free_symbols if _is_sr_coordinate_symbol(s)}


def expr_has_self_transcendental(expr: sympy.Expr) -> bool:
    """
    True for same-coordinate nonlinear self-couplings that are hard to invert.

    This intentionally keeps cross-coordinate terms such as ``X2*exp(X1)`` and
    ``X2**X1`` while rejecting terms such as ``X1*exp(X1)``, ``X1**X1``, and
    ``(a*X1)**(b*X1)``.
    """
    transcendental_funcs = {sympy.exp, sympy.log, sympy.sin, sympy.cos, sympy.tan}

    # Same coordinate in both base and exponent, e.g. X1**X1 or
    # (a*X1)**(b*X1). Constant exponents like X1**2 are kept.
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.Pow):
            base, exponent = node.as_base_exp()
            if _sr_coordinate_symbols(base) & _sr_coordinate_symbols(exponent):
                return True

    # Multiplicative terms where a factor outside f(...) and the argument of
    # f(...) depend on the same coordinate, e.g. (1 + X1)*exp(X1).
    expanded = sympy.expand_mul(expr)
    terms = sympy.Add.make_args(expanded) if isinstance(expanded, sympy.Add) else (expanded,)
    for term in terms:
        factors = sympy.Mul.make_args(term) if isinstance(term, sympy.Mul) else (term,)
        factor_symbols = [_sr_coordinate_symbols(factor) for factor in factors]
        for idx, factor in enumerate(factors):
            other_symbols = set().union(
                *(symbols for j, symbols in enumerate(factor_symbols) if j != idx)
            )
            if not other_symbols:
                continue
            for node in sympy.preorder_traversal(factor):
                if node.func in transcendental_funcs and len(node.args) == 1:
                    if _sr_coordinate_symbols(node.args[0]) & other_symbols:
                        return True

    return False


def sr_structure_predicate(
    n_params: int,
    *,
    check_nested_exp: bool = True,
    max_exp_nesting: Optional[int] = 2,
    check_nested_pow: bool = False,
    max_pow_nesting: Optional[int] = 1,
    check_nested_log: bool = False,
    max_log_nesting: Optional[int] = 1,
    check_nested_logabs: bool = False,
    max_logabs_nesting: Optional[int] = 1,
    forbid_x_in_pow_exponent: bool = True,
    forbid_self_transcendental: bool = False,
) -> Callable[[str], bool]:
    """
    Build a string predicate for :func:`analyze_equations` / ``equation_predicate``.

    Parses each equation with the same ESR grammar as ``compute_DL``. On parse
    failure the predicate returns False (equation is skipped).

    Parameters
    ----------
    n_params
        Number of parameter dimensions (kept for API consistency; the
        variable-in-exponent rule uses symbol **names** ``X1``, ``X2``, …, not
        this count, so a mismatched ``n_params`` no longer disables that check).
    check_nested_exp
        If True, apply the ``max_exp_nesting`` bound using :func:`max_exp_nesting_depth`.
    max_exp_nesting
        Reject if nested ``exp`` depth exceeds this value when ``check_nested_exp``
        is True. ``None`` means no cap (nested ``exp`` never rejected by this rule).
        Default ``2`` allows ``exp(exp(...))`` but not a third nested ``exp``.
    check_nested_pow
        If True, apply the ``max_pow_nesting`` bound using :func:`max_pow_nesting_depth`.
    max_pow_nesting
        Reject if nested ``Pow`` depth exceeds this value when ``check_nested_pow``
        is True. ``None`` means no cap on ``Pow`` nesting. Default ``1`` keeps at
        most one ``Pow`` along any ``Pow`` chain (e.g. ``X1**2`` ok, ``(X1**2)**3`` out).
    check_nested_log
        If True, apply the ``max_log_nesting`` bound using :func:`max_log_nesting_depth`.
    max_log_nesting
        Reject if nested ``log`` depth exceeds this value when ``check_nested_log``
        is True. ``None`` means no cap. Default ``1`` allows ``log(X1)`` but not
        ``log(log(X1))``.
    check_nested_logabs
        If True, apply the ``max_logabs_nesting`` bound using
        :func:`max_logabs_nesting_depth` (chains of ``log(Abs(...))``, i.e. SR ``logAbs``).
    max_logabs_nesting
        Reject if nested ``log(Abs(...))`` depth exceeds this value when
        ``check_nested_logabs`` is True. ``None`` means no cap. Default ``1``
        allows one ``logAbs`` wrapper but not ``logAbs(logAbs(...))``.
    forbid_x_in_pow_exponent
        If True, reject ``Pow`` nodes whose exponent depends on any ``Xi``.
    forbid_self_transcendental
        If True, reject same-coordinate self-couplings such as ``X1*exp(X1)``
        and ``X1**X1`` while keeping cross-coordinate terms such as
        ``X2*exp(X1)`` and ``X2**X1``.

    Returns
    -------
    Callable[[str], bool]
        ``True`` means the equation is kept for MDL / Frobenius analysis.
    """
    def predicate(eq: str) -> bool:
        try:
            expr, _, _, _ = _parse_sr_equation(eq)
        except Exception:
            return False
        if forbid_x_in_pow_exponent and expr_has_pow_with_x_in_exponent(expr):
            return False
        if forbid_self_transcendental and expr_has_self_transcendental(expr):
            return False
        if check_nested_exp and max_exp_nesting is not None:
            if max_exp_nesting_depth(expr) > max_exp_nesting:
                return False
        if check_nested_pow and max_pow_nesting is not None:
            if max_pow_nesting_depth(expr) > max_pow_nesting:
                return False
        if check_nested_log and max_log_nesting is not None:
            if max_log_nesting_depth(expr) > max_log_nesting:
                return False
        if check_nested_logabs and max_logabs_nesting is not None:
            if max_logabs_nesting_depth(expr) > max_logabs_nesting:
                return False
        return True

    return predicate


def filter_equation_csv(
    csv_path: str,
    equation_predicate: Callable[[str], bool],
    *,
    model_column: str = "model",
    delimiter: str = ";",
    backup_suffix: str = ".unfiltered",
) -> Dict[str, int | str]:
    """
    Rewrite an SR equation CSV after removing equations rejected by a predicate.

    This is useful for making the persisted Pareto front match the structural
    constraints used later by :func:`analyze_equations`, rather than only
    skipping rejected equations in memory.
    """
    data = pd.read_csv(csv_path, delimiter=delimiter)
    if model_column not in data.columns:
        raise ValueError(f"{csv_path} does not contain model column {model_column!r}")

    keep = np.array([bool(equation_predicate(str(eq))) for eq in data[model_column]], dtype=bool)
    removed = int((~keep).sum())
    original = int(len(data))

    backup_path = f"{csv_path}{backup_suffix}"
    if removed and not os.path.exists(backup_path):
        data.to_csv(backup_path, sep=delimiter, index=False)
    if removed:
        data.loc[keep].to_csv(csv_path, sep=delimiter, index=False)

    return {
        "path": csv_path,
        "backup_path": backup_path if removed else "",
        "original": original,
        "kept": int(keep.sum()),
        "removed": removed,
    }


def filter_pareto_fronts(
    parent_dir: str,
    n_components: int,
    equation_predicate: Callable[[str], bool],
    *,
    backup_suffix: str = ".unfiltered",
) -> List[Dict[str, int | str]]:
    """Remove predicate-rejected equations from component ``pareto.csv`` files."""
    summaries = []
    for component_idx in range(1, n_components + 1):
        pareto_path = os.path.join(parent_dir, f"component_{component_idx}", "pareto.csv")
        summaries.append(
            filter_equation_csv(
                pareto_path,
                equation_predicate,
                backup_suffix=backup_suffix,
            )
        )
    return summaries


@jax.jit
def norm(A: jnp.ndarray) -> float:
    """Frobenius norm of a matrix."""
    return jnp.sqrt(jnp.einsum('ij,ij->', A, A))


def compute_DL(eq: str, component_idx: int, X: np.ndarray, y: np.ndarray, 
               y_std: np.ndarray, dy_sr: np.ndarray, Fs: np.ndarray,
               n_params: int, length_penalty: float = 2.0) -> Tuple[float, str, float, float, float]:
    """
    Compute description length (MDL) and flattening metrics for a symbolic equation.
    
    This function:
    1. Parses the equation string and extracts parameters
    2. Computes complexity using Aifeyn criterion
    3. Computes negative log-likelihood
    4. Computes parameter encoding cost via Fisher information
    5. Computes Frobenius norm loss for flattening
    
    Parameters
    ----------
    eq : str
        String representation of the equation
    component_idx : int
        Index of the component being fitted (position in y vector)
    X : np.ndarray
        Input parameters of shape (n_samples, n_params)
    y : np.ndarray
        Network outputs of shape (n_samples, n_components)
    y_std : np.ndarray
        Standard deviations of outputs of shape (n_samples, n_components)
    dy_sr : np.ndarray
        Jacobians of shape (n_samples, n_components, n_params)
    Fs : np.ndarray
        Fisher matrices of shape (n_samples, n_params, n_params)
    n_params : int
        Number of parameters
    length_penalty : float
        Multiplier on the Fisher-based parameter codelength term in ``DL``. Default 2.0
        gives ``DL = neglogL + aifeyn + length_penalty * param_codelen``.

    Returns
    -------
    complexity : float
        Equation complexity (number of nodes)
    latex_expr : str
        LaTeX representation of the equation
    neglogL : float
        Negative log-likelihood
    DL : float
        Description length (MDL criterion)
    frobloss : float
        Frobenius norm flattening loss
    """
    basis_functions = [
        ["X", "b"],  # type0
        ["square", "exp", "inv", "sqrt", "log", "cos", "logAbs"],  # type1
        ["+", "*", "-", "/", "^"],  # type2
    ]

    expr, nodes, complexity, pars = _parse_sr_equation(eq)

    param_list = [f"b{i}" for i in range(len(pars))]
    latex_expr = sympy.latex(expr)
    
    # Compute Aifeyn complexity: k*log(n) + sum_i log|c_i|
    aifeyn = esr.generation.generator.aifeyn_complexity(nodes.to_list(basis_functions), param_list)
    
    # Turn function into callable object
    all_x = ' '.join([f'X{i}' for i in range(1, X.shape[1] + 1)])
    all_x = list(sympy.symbols(all_x, real=True))
    all_b = list(sympy.symbols(param_list, real=True))
    eq_jax = safe_lambdify(all_b + all_x, expr)

    # Define loss function (negative log-likelihood)
    def myloss(p):
        ypred = eq_jax(*p, *X.T)
        result = jnp.sum((y[:, component_idx] - ypred)**2 / 2 / y_std[:, component_idx]**2)
        return result
    
    # Define flattening loss
    def frob_loss(p):
        def get_jac_row(p):
            myeq = lambda *args: promote_jacrev_output(eq_jax(*p, *args))
            return vmapped_jacobian_wrt_X(myeq, jnp.asarray(X))

        jac_row = get_jac_row(pars)
        
        # Assign the SR expression's jacobian row to a copy of the network Jac
        jacobian = dy_sr.copy()
        jacobian[:, component_idx, :] = np.array(jac_row)

        # Flatten Fisher matrices into η-space.  For J shape (m, d) the
        # pullback invJ.T @ F @ invJ is (m, m), so compare to I_m — not I_d.
        # (The old eye(n_params) only worked when m == d.)
        def flatten_fisher(J, F):
            invJ = jnp.linalg.pinv(J)
            return invJ.T @ F @ invJ

        flats = jax.vmap(flatten_fisher)(jacobian, Fs)
        nn_flats = jax.vmap(flatten_fisher)(dy_sr, Fs)

        n_out = int(dy_sr.shape[1])
        eye_out = jnp.eye(n_out)
        fn = lambda q: norm((q - eye_out)) + norm((jnp.linalg.pinv(q) - eye_out))

        return np.mean(jax.vmap(fn)(flats) - jax.vmap(fn)(nn_flats))
    
    # Compute negative log-likelihood and Frobenius loss
    neglogL = myloss(pars)
    frobloss = frob_loss(pars)
    
    # Compute parameter encoding cost
    if len(pars) == 0:
        param_codelen = 0
    else:
        theta_ML = np.array(pars)

        # Compute Hessian
        hessian_myloss = jax.hessian(myloss)
        I_ii = np.diag(np.array(hessian_myloss(pars)))
            
        # Remove parameters which do not affect the likelihood or zero parameters
        kept_mask = (I_ii > 0) & (theta_ML != 0)
        theta_ML = theta_ML[kept_mask]
        I_ii = I_ii[kept_mask]
        
        # If the error is bigger than the parameter value, set precision to parameter value
        Delta = np.sqrt(12. / I_ii)
        nsteps = np.abs(np.array(theta_ML)) / Delta
        m = nsteps < 1
        I_ii[m] = 12 / theta_ML[m] ** 2
        
        # Compute parameter part of codelength
        p = len(theta_ML) - np.sum(m)  # subtract out sum of mask => params == 0
        param_codelen = -p / 2. * np.log(3.) + np.sum(
            0.5 * np.log(I_ii) + np.log(np.abs(np.array(theta_ML)))
        )

    # Combine the terms (matches docstring: scaled Fisher parameter codelength only)
    DL = neglogL + aifeyn + float(length_penalty) * param_codelen

    return complexity, latex_expr, neglogL, DL, frobloss


# =============================================================================
# SYMBOLIC REGRESSION FITTING
# =============================================================================

def fit_symbolic_regression(
    X: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
    components_to_fit: Optional[List[int]] = None,
    parent_dir: str = './sr_results/',
    allowed_symbols: str = 'add,mul,pow,constant,variable,exp,logabs,sqrt',
    epsilon: float = 1e-5,
    max_length: int = 25,
    max_depth: int = 10,
    time_limit: int = 120,
    objectives: List[str] = ['r2', 'length'],
    max_evaluations: float = 1e8,
    generations: float = 1e8,
    random_state: int = 2345,
    optimizer_iterations: int = 10,
    verbose: bool = True
) -> None:
    """
    Fit symbolic regression models to multiple components.
    
    This function trains a symbolic regression model for each component
    specified in `components_to_fit`, saving the Pareto front and final
    population to CSV files.
    
    Parameters
    ----------
    X : np.ndarray
        Input parameters of shape (n_samples, n_params)
    y : np.ndarray
        Network outputs of shape (n_samples, n_components)
    y_std : np.ndarray
        Standard deviations of outputs
    components_to_fit : list[int], optional
        Indices of components to fit. If None, fits all components.
    parent_dir : str
        Output directory for results
    allowed_symbols : str
        Comma-separated list of allowed operators. 
        Default: 'add,mul,pow,constant,variable,exp,logabs,sqrt'
        Other options: 'square', 'inv', 'cos', 'sin', 'tan', etc.
    epsilon : float
        Threshold for constant simplification. Default: 1e-5
    max_length : int
        Maximum equation length (number of nodes). Default: 25
    max_depth : int
        Maximum tree depth. Default: 10
    time_limit : int
        Time limit per component in seconds. Default: 120 (2 minutes)
    objectives : list[str]
        Optimization objectives. Default: ['r2', 'length']
        Options: 'r2', 'rmse', 'mae', 'length', etc.
    max_evaluations : float
        Maximum number of evaluations. Default: 1e8
    generations : float
        Maximum number of generations. Default: 1e8
    random_state : int
        Random seed
    optimizer_iterations : int
        Number of optimizer iterations for constant refinement
    verbose : bool
        Whether to print progress
    """
    # Default to fitting all components
    if components_to_fit is None:
        components_to_fit = list(range(y.shape[1]))
    
    if not os.path.isdir(parent_dir):
        os.mkdir(parent_dir)

    n_components = len(components_to_fit)
    # halfx = X.shape[0] // 2

    for i in range(n_components):
        comp_idx = components_to_fit[i]
        out_dir = os.path.join(parent_dir, f"component_{i + 1}")
        
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        Xfit = X
        yfit = y[:, comp_idx]
        y_std_fit = y_std[:, comp_idx]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fitting component {i + 1} of {n_components} (index {comp_idx})")
            print(f"{'='*60}")
            print(f"X train shape: {Xfit.shape}, y train shape: {yfit.shape}")
        
        # Build kwargs with version-compatible time parameter name
        reg_kwargs = {
            'allowed_symbols': allowed_symbols,
            'offspring_generator': 'basic',
            'optimizer_iterations': optimizer_iterations,
            'max_length': max_length,
            'max_depth': max_depth,
            'initialization_method': 'btc',
            'n_threads': multiprocessing.cpu_count(),
            'objectives': objectives,
            'epsilon': epsilon,
            'random_state': random_state,
            'reinserter': 'keep-best',
            'max_evaluations': int(max_evaluations),
            'symbolic_mode': False,
            _TIME_PARAM_NAME: int(time_limit),  # Compatible with both time_limit and max_time
            'generations': int(generations),
            'add_model_scale_term': True,
            'add_model_intercept_term': True,
        }
        
        reg = SymbolicRegressor(**reg_kwargs)

        if verbose:
            print('Fitting...')
        
        reg.fit(Xfit, yfit)
        
        if verbose:
            print('Done!')
            print(f"Best model: {reg.get_model_string(reg.model_, 2)}")
            print(f"Stats: {reg.stats_}")

        # Save Pareto front
        pareto_path = os.path.join(out_dir, 'pareto.csv')
        with open(pareto_path, 'w') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['length', 'mse', 'model'])
            
            if verbose:
                print(f'Outputting {len(reg.pareto_front_)} individuals on Pareto front')
            
            for individual in reg.pareto_front_:
                writer.writerow([
                    individual['tree'].Length,
                    individual['mean_squared_error'],
                    individual['model'],
                ])
        
        # Save population
        pop_path = os.path.join(out_dir, 'final_population.csv')
        if verbose:
            print(f'Outputting {len(reg.individuals_)} individuals in population')
        
        with open(pop_path, 'w') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['length', 'mse', 'model'])
            
            for ind in reg.individuals_[:reg.population_size]:
                tree = ind.Genotype
                
                # Get name but block printing to sys.stderr
                sys.stderr = open(os.devnull, 'w')
                s = reg.get_model_string(ind.Genotype, 10)
                sys.stderr = sys.__stderr__
                
                y_pred = reg.evaluate_model(tree, Xfit)
                mse = np.mean((yfit - y_pred)**2)
                
                writer.writerow([tree.Length, mse, s])


# =============================================================================
# EQUATION ANALYSIS
# =============================================================================

def analyze_equations(
    X: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
    dy_sr: np.ndarray,
    Fs: np.ndarray,
    n_params: int,
    components_to_fit: Optional[List[int]] = None,
    parent_dir: str = './sr_results/',
    max_complexity_thresh: int = 14,
    equation_set: str = 'pareto',
    verbose: bool = True,
    length_penalty: float = 2.0,
    equation_predicate: Optional[Callable[[str], bool]] = None,
) -> Tuple[List[str], List[str], Dict[str, List]]:
    """
    Analyze symbolic regression results and rank equations.
    
    This function:
    1. Loads equations from saved CSV files
    2. Computes DL and flattening metrics for each equation
    3. Finds best equations according to MDL and Frobenius loss
    4. Returns coordinates and analysis data
    
    Parameters
    ----------
    X : np.ndarray
        Input parameters (validation set) of shape (n_samples, n_params)
    y : np.ndarray
        Network outputs (validation set) of shape (n_samples, n_components)
    y_std : np.ndarray
        Standard deviations of outputs
    dy_sr : np.ndarray
        Jacobians of shape (n_samples, n_components, n_params)
    Fs : np.ndarray
        Fisher matrices of shape (n_samples, n_params, n_params)
    n_params : int
        Number of parameters
    components_to_fit : list[int], optional
        Indices of components that were fitted. If None, analyzes all components.
    parent_dir : str
        Directory containing SR results
    max_complexity_thresh : int
        Maximum complexity to consider
    equation_predicate : callable, optional
        If set, ``equation_predicate(eq_str)`` must return True for an equation
        to be scored with ``compute_DL``. Use e.g. :func:`sr_structure_predicate`
        to drop variable–variable powers and optionally nested ``exp``, ``Pow``,
        ``log``, or ``log(Abs(...))`` (SR ``logAbs``).
        Parse errors
        should be handled inside the predicate (return False to skip).
    equation_set : str
        Which equation set to use for analysis. Options:
        - 'pareto': Use only equations from pareto.csv (default)
        - 'full_population': Use only equations from final_population.csv
        - 'both': Concatenate equations from both pareto.csv and final_population.csv
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    mdl_coordinates : list[str]
        Best equations according to MDL criterion
    frob_coordinates : list[str]
        Best equations according to Frobenius loss
    analysis_data : dict
        Dictionary containing:
        - 'latex': LaTeX representations for each component
        - 'equations': String equations for each component
        - 'frobloss': Frobenius losses for each component
        - 'DL': Description lengths for each component
        - 'logL': Negative log-likelihoods for each component
        - 'complexity': Complexities for each component
        - 'ibest_mdl': Index of best MDL equation for each component
        - 'ibest_frob': Index of best Frobenius loss equation for each component
    """
    # Default to analyzing all components
    if components_to_fit is None:
        components_to_fit = list(range(y.shape[1]))
    
    n_components = len(components_to_fit)
    
    mdl_coordinates = []
    frob_coordinates = []
    
    both_comp_latex = []
    both_comp_eqs = []
    both_comp_frobloss = []
    both_comp_logL = []
    both_comp_DL = []
    both_comp_complexity = []
    both_comp_ibest_mdl = []
    both_comp_ibest_frob = []

    for i in range(n_components):
        idx = components_to_fit[i]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Analysing component {i + 1} of {n_components} (index {idx})")
            print(f"{'='*60}")

        outdir = os.path.join(parent_dir, f"component_{i + 1}")
        
        # Load equations based on equation_set parameter
        if equation_set == 'pareto':
            data_path = os.path.join(outdir, 'pareto.csv')
            data = pd.read_csv(data_path, delimiter=";")
            if verbose:
                print(f"Loading equations from: pareto.csv")
        elif equation_set == 'full_population':
            data_path = os.path.join(outdir, 'final_population.csv')
            data = pd.read_csv(data_path, delimiter=";")
            if verbose:
                print(f"Loading equations from: final_population.csv")
        elif equation_set == 'both':
            pareto_path = os.path.join(outdir, 'pareto.csv')
            pop_path = os.path.join(outdir, 'final_population.csv')
            data_pareto = pd.read_csv(pareto_path, delimiter=";")
            data_pop = pd.read_csv(pop_path, delimiter=";")
            # Concatenate both dataframes
            data = pd.concat([data_pareto, data_pop], ignore_index=True)
            if verbose:
                print(f"Loading equations from: both pareto.csv and final_population.csv")
                print(f"  Pareto: {len(data_pareto)} equations, Population: {len(data_pop)} equations")
        else:
            raise ValueError(f"equation_set must be 'pareto', 'full_population', or 'both', got '{equation_set}'")
        
        complexity = np.array(data["length"])
        mse_mask = (complexity < max_complexity_thresh)
        
        if verbose:
            print(f"{mse_mask.sum()} equations below complexity threshold {max_complexity_thresh}")

        complexity = complexity[mse_mask]
        eqs = [str(eq) for eq in np.array(data['model'])[mse_mask]]

        if equation_predicate is not None:
            keep = np.array([bool(equation_predicate(eq)) for eq in eqs], dtype=bool)
            if verbose:
                print(
                    f"{keep.sum()} / {len(eqs)} equations pass equation_predicate "
                    f"(skipped {len(eqs) - keep.sum()})"
                )
            complexity = complexity[keep]
            eqs = [eq for eq, k in zip(eqs, keep) if k]

        # Compute metrics for all equations
        all_DL = np.ones(len(eqs)) * np.inf
        all_logL = np.ones(len(eqs)) * np.inf
        all_frobloss = np.ones(len(eqs)) * np.inf
        all_latex = [None] * len(eqs)
        all_complexity = np.ones(len(eqs)) * np.inf

        for j, eq in enumerate(tqdm(eqs, desc=f"Component {i+1}")):
            try:
                c, latex, logL, DL, frobloss = compute_DL(
                    eq, idx, X, y, y_std, dy_sr, Fs, n_params,
                    length_penalty=length_penalty,
                )
                all_complexity[j] = c
                all_latex[j] = latex
                all_logL[j] = logL
                all_DL[j] = DL
                all_frobloss[j] = frobloss
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to process equation {j}: {eq}")
                    print(f"  Error: {e}")
                continue

        # Set nans to infs
        all_DL[np.isnan(all_DL)] = np.inf
        all_logL[np.isnan(all_logL)] = np.inf
        all_frobloss[np.isnan(all_frobloss)] = np.inf

        # Get best model at each complexity level (Pareto front in complexity-DL space)
        pareto_DL = []
        pareto_logL = [] 
        pareto_latex = []
        pareto_eqs = []
        pareto_frobloss = []
        comps = []
        
        for comp in np.unique(complexity):
            if comp > 1:  # Skip trivial constants
                mask = (complexity == comp)
                ibest_model_at_complexity = np.argmin(all_DL[mask])
                pareto_DL.append(all_DL[mask][ibest_model_at_complexity])
                pareto_logL.append(all_logL[mask][ibest_model_at_complexity])
                pareto_frobloss.append(all_frobloss[mask][ibest_model_at_complexity])
                pareto_latex.append(str(np.array(all_latex)[mask][ibest_model_at_complexity]))
                pareto_eqs.append(str(np.array(eqs)[mask][ibest_model_at_complexity]))
                comps.append(comp)

        complexity = np.array(comps)

        # Normalize metrics (relative to minimum)
        pareto_DL = np.array(pareto_DL)
        pareto_DL -= np.amin(pareto_DL)

        pareto_logL = np.array(pareto_logL)
        pareto_logL -= np.amin(pareto_logL)

        pareto_frobloss = np.array(pareto_frobloss)
        pareto_frobloss -= np.amin(pareto_frobloss)

        # Store results
        both_comp_latex.append(pareto_latex)
        both_comp_eqs.append(pareto_eqs)
        both_comp_frobloss.append(pareto_frobloss)
        both_comp_DL.append(pareto_DL)
        both_comp_logL.append(pareto_logL)
        both_comp_complexity.append(complexity)

        # Find best equations
        ibest = np.argmin(pareto_DL)
        ibest_frob = np.argmin(pareto_frobloss)
        
        # Store best indices
        both_comp_ibest_mdl.append(ibest)
        both_comp_ibest_frob.append(ibest_frob)
        
        if verbose:
            print(f'\nBest MDL equation (complexity={complexity[ibest]}):')
            print(f'  {pareto_eqs[ibest]}')
            print(f'  LaTeX: ${pareto_latex[ibest]}$')
            print(f'\nBest Frob loss equation (complexity={complexity[ibest_frob]}):')
            print(f'  {pareto_eqs[ibest_frob]}')
            print(f'  LaTeX: ${pareto_latex[ibest_frob]}$')
        
        mdl_coordinates.append(str(pareto_eqs[ibest]))
        frob_coordinates.append(str(pareto_eqs[ibest_frob]))

    analysis_data = {
        'latex': both_comp_latex,
        'equations': both_comp_eqs,
        'frobloss': both_comp_frobloss,
        'DL': both_comp_DL,
        'logL': both_comp_logL,
        'complexity': both_comp_complexity,
        'ibest_mdl': both_comp_ibest_mdl,
        'ibest_frob': both_comp_ibest_frob,
    }

    return mdl_coordinates, frob_coordinates, analysis_data


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def fit_and_analyze_sr(
    X: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
    dy_sr: np.ndarray,
    Fs: np.ndarray,
    n_params: int,
    components_to_fit: Optional[List[int]] = None,
    parent_dir: str = './sr_results/',
    test_size: float = 0.5,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    slice_fisher: bool = False,
    save_split_data: bool = True,
    split_data_npz_name: str = "split_data.npz",
    flatten_model=None,
    ensemble_w=None,
    rotmats=None,
    ensemble_weights: Optional[np.ndarray] = None,
    n_sr_samples: int = 2000,
    key=None,
    **sr_kwargs
) -> Tuple[List[str], List[str], Dict[str, List], Dict]:
    """
    Complete pipeline: split data, fit SR models, and analyze results.
    
    This function automatically splits the data into train/validation sets using
    scikit-learn's train_test_split, then fits symbolic regression models on the
    training set and evaluates them on the validation set.
    
    Parameters
    ----------
    X : np.ndarray
        Input parameters of shape (n_samples, n_params)
    y : np.ndarray
        Network outputs of shape (n_samples, n_components)
    y_std : np.ndarray
        Standard deviations of outputs of shape (n_samples, n_components)
    dy_sr : np.ndarray
        Jacobians of shape (n_samples, n_components, n_params)
    Fs : np.ndarray
        Fisher matrices of shape (n_samples, n_params, n_params)
    n_params : int
        Number of parameters (dimension of full parameter space)
    components_to_fit : list[int], optional
        Which components to fit. If None, fits all components.
    parent_dir : str
        Output directory for results
    save_split_data : bool
        If True (default), writes ``split_data`` arrays (and split metadata) to
        ``os.path.join(parent_dir, split_data_npz_name)`` after analysis.
    split_data_npz_name : str
        Filename for the compressed NumPy archive (default ``split_data.npz``).
    test_size : float
        Proportion of dataset to include in validation split (default: 0.5)
    random_state : int, optional
        Random seed for reproducible splits. If None, uses random split.
    shuffle : bool
        Whether to shuffle data before splitting (default: True)
    slice_fisher : bool
        If True, slices X, dy_sr, and Fs to only include the dimensions
        corresponding to components_to_fit. This assumes components map to
        parameters 1-to-1 (i.e., component i corresponds to parameter i).
        Default: False (uses full Fisher matrices and parameter space)
    **sr_kwargs
        Additional arguments for fit_symbolic_regression and analyze_equations.
        ``verbose`` applies to both the fitter and ``analyze_equations``.
        Common fit_symbolic_regression options:
        - allowed_symbols: str = 'add,mul,pow,constant,variable,exp,logabs,sqrt'
        - epsilon: float = 1e-5
        - max_length: int = 25
        - max_depth: int = 10
        - time_limit: int = 120 (seconds)
        - objectives: List[str] = ['r2', 'length']
        - max_evaluations: float = 1e8
        - generations: float = 1e8
        - random_state: int = 2345
        - optimizer_iterations: int = 10
        
        Common analyze_equations options:
        - equation_set: str = 'pareto' ('pareto', 'full_population', or 'both')
        - max_complexity_thresh: int = 14
        - length_penalty: float = 2.0 (``compute_DL`` parameter codelength weight)
        - equation_predicate: Optional[Callable[[str], bool]] = None
          (e.g. ``sr_structure_predicate(n_params, max_exp_nesting=1)`` or
          ``check_nested_pow=True, max_pow_nesting=1`` to cap ``Pow`` nesting only;
          use ``check_nested_log=True`` / ``check_nested_logabs=True`` to cap
          nested ``log`` and ``logAbs`` chains)

    Returns
    -------
    mdl_coordinates : list[str]
        Best MDL equations
    frob_coordinates : list[str]
        Best Frobenius loss equations
    analysis_data : dict
        Analysis results containing complexity curves and metrics.
        Keys: 'latex', 'equations', 'frobloss', 'DL', 'logL', 'complexity',
        'ibest_mdl', 'ibest_frob' (one entry per component)
    split_data : dict
        Dictionary containing train/test split data:
        - 'X_train', 'X_test': Input parameters
        - 'y_train', 'y_test': Network outputs
        - 'y_std_train', 'y_std_test': Output uncertainties
        - 'dy_sr_train', 'dy_sr_test': Jacobians
        - 'Fs_train', 'Fs_test': Fisher matrices
        - 'n_params': Number of parameters (updated if slice_fisher=True)

        When ``save_split_data`` is True, the same arrays are stored under the
        same keys in ``parent_dir/split_data_npz_name``, plus ``components_to_fit``
        (int64), ``slice_fisher`` (bool), ``test_size``, ``shuffle``, and
        ``random_state`` (int64, ``-1`` if the call used ``random_state=None``).
        
    Examples
    --------
    >>> # Fit all components with full Fisher matrices
    >>> mdl_coords, frob_coords, analysis, split_data = fit_and_analyze_sr(
    ...     X, y, y_std, dy_sr, Fs,
    ...     n_params=2,
    ...     test_size=0.5,
    ...     random_state=42
    ... )
    >>> # Access test set data and analysis results
    >>> X_test = split_data['X_test']
    >>> y_test = split_data['y_test']
    >>> 
    >>> # Access best equation indices
    >>> ibest_mdl = analysis['ibest_mdl'][0]  # best MDL index for component 0
    >>> ibest_frob = analysis['ibest_frob'][0]  # best Frob index for component 0
    >>> best_mdl_equation = analysis['equations'][0][ibest_mdl]
    >>> best_frob_equation = analysis['equations'][0][ibest_frob]
    >>> 
    >>> # Fit specific components with full parameter space (default)
    >>> mdl_coords, frob_coords, analysis, split_data = fit_and_analyze_sr(
    ...     X, y, y_std, dy_sr, Fs,
    ...     n_params=3,
    ...     components_to_fit=[0, 2],  # fit components 0 and 2 only
    ...     slice_fisher=False          # use full 3D parameter space
    ... )
    >>> 
    >>> # Fit specific components with sliced Fisher matrices
    >>> # (assumes component i corresponds to parameter i)
    >>> mdl_coords, frob_coords, analysis, split_data = fit_and_analyze_sr(
    ...     X, y, y_std, dy_sr, Fs,
    ...     n_params=3,
    ...     components_to_fit=[0, 2],  # fit components 0 and 2
    ...     slice_fisher=True           # slice to 2D subspace [params 0, 2]
    ... )
    >>> 
    >>> # Customize SR hyperparameters
    >>> mdl_coords, frob_coords, analysis, split_data = fit_and_analyze_sr(
    ...     X, y, y_std, dy_sr, Fs,
    ...     n_params=2,
    ...     allowed_symbols='add,mul,pow,constant,variable,exp,sqrt',  # no log
    ...     max_length=20,              # shorter equations
    ...     time_limit=300,             # 5 minutes per component
    ...     objectives=['rmse', 'length']  # RMSE instead of R2
    ... )
    >>> 
    >>> # Use full population instead of just Pareto front
    >>> mdl_coords, frob_coords, analysis, split_data = fit_and_analyze_sr(
    ...     X, y, y_std, dy_sr, Fs,
    ...     n_params=2,
    ...     equation_set='full_population'  # analyze all equations
    ... )
    >>> 
    >>> # Analyze both Pareto front and full population
    >>> mdl_coords, frob_coords, analysis, split_data = fit_and_analyze_sr(
    ...     X, y, y_std, dy_sr, Fs,
    ...     n_params=2,
    ...     equation_set='both',         # concatenate both datasets
    ...     max_complexity_thresh=20     # allow more complex equations
    ... )
    """
    if components_to_fit is None:
        components_to_fit = list(range(y.shape[1]))

    # ── Augmentation validation ──────────────────────────────────────────────
    _aug_args = (flatten_model, ensemble_w, rotmats, ensemble_weights)
    _aug_provided = [a is not None for a in _aug_args]
    if any(_aug_provided) and not all(_aug_provided):
        raise ValueError(
            "flatten_model, ensemble_w, rotmats, and ensemble_weights must all be "
            "provided together to enable the uniform-grid augmentation."
        )
    use_augmentation = all(_aug_provided)

    if use_augmentation:
        # ── Build augmented SR grid ──────────────────────────────────────────
        from degeneracy_distillery.postprocessing_utils import weighted_std as _weighted_std
        import jax.random as jr

        if key is None:
            key = jr.PRNGKey(0)

        X_arr = np.asarray(X)
        X_sr = np.array(jr.uniform(
            key,
            minval=jnp.array(X_arr.min(0)),
            maxval=jnp.array(X_arr.max(0)),
            shape=(n_sr_samples, X_arr.shape[1]),
        ))

        ys_sr = jnp.array([
            jax.vmap(lambda x: flatten_model.apply(w_i, x))(jnp.array(X_sr))
            for w_i in ensemble_w
        ])

        ys_sr_rot = np.array([
            np.einsum("ij,bj->bi", np.asarray(rotmats[i]),
                      np.array(ys_sr[i]) - np.array(ys_sr[i]).mean(0))
            for i in range(len(ensemble_w))
        ])

        ew = np.asarray(ensemble_weights)
        y_sr = np.average(ys_sr_rot, axis=0, weights=ew)
        y_std_sr = _weighted_std(ys_sr_rot, ew)

        # Shift minimum to zero (matches notebook convention)
        ys_sr_rot -= y_sr.min(0)
        y_sr -= y_sr.min(0)

        # Split augmented set: train → SR fitting, val → held out
        (X_sr_train, X_sr_val,
         y_sr_train, y_sr_val,
         y_std_sr_train, y_std_sr_val) = train_test_split(
            X_sr, y_sr, y_std_sr,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        X_fit, y_fit, y_std_fit = X_sr_train, y_sr_train, y_std_sr_train

        # Analysis uses the full original data: Frobenius loss requires matched Fishers
        X_eval     = np.asarray(X)
        y_eval     = np.asarray(y)
        y_std_eval = np.asarray(y_std)
        dy_sr_eval = np.asarray(dy_sr)
        Fs_eval    = np.asarray(Fs)

    else:
        # ── Original train/test split ────────────────────────────────────────
        split_result = train_test_split(
            X, y, y_std, dy_sr, Fs,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        X_fit,     X_eval     = split_result[0], split_result[1]
        y_fit,     y_eval     = split_result[2], split_result[3]
        y_std_fit, y_std_eval = split_result[4], split_result[5]
        dy_sr_eval            = split_result[7]
        Fs_eval               = split_result[9]
        X_sr_train = X_sr_val = y_sr_train = y_sr_val = y_std_sr_train = y_std_sr_val = None

    # ── Optional Fisher / parameter-space slicing ────────────────────────────
    if slice_fisher:
        X_fit      = X_fit[:, components_to_fit]
        X_eval     = X_eval[:, components_to_fit]
        dy_sr_eval = dy_sr_eval[:, components_to_fit, :][:, :, components_to_fit]
        Fs_eval    = Fs_eval[:, components_to_fit, :][:, :, components_to_fit]
        if X_sr_train is not None:
            X_sr_train = X_sr_train[:, components_to_fit]
            X_sr_val   = X_sr_val[:, components_to_fit]
        n_params = len(components_to_fit)

    # ── Separate kwargs for fitting vs. analysis ─────────────────────────────
    _analysis_only = frozenset({
        'equation_set', 'max_complexity_thresh', 'length_penalty', 'equation_predicate',
    })
    fit_kwargs = {k: v for k, v in sr_kwargs.items() if k not in _analysis_only}

    equation_set          = sr_kwargs.get('equation_set', 'pareto')
    max_complexity_thresh = sr_kwargs.get('max_complexity_thresh', 14)
    length_penalty        = float(sr_kwargs.get('length_penalty', 2.0))
    equation_predicate    = sr_kwargs.get('equation_predicate', None)
    verbose_sr            = bool(sr_kwargs.get('verbose', True))

    # ── Fit SR models ────────────────────────────────────────────────────────
    fit_symbolic_regression(
        X_fit, y_fit, y_std_fit,
        components_to_fit, parent_dir,
        **fit_kwargs,
    )

    # ── Analyse / select from Pareto front ───────────────────────────────────
    # Always uses the original (X, y, dy_sr, Fs): Frobenius loss requires
    # Fisher matrices matched to the evaluation points.
    mdl_coords, frob_coords, analysis = analyze_equations(
        X_eval, y_eval, y_std_eval, dy_sr_eval, Fs_eval,
        n_params, components_to_fit, parent_dir,
        max_complexity_thresh=max_complexity_thresh,
        equation_set=equation_set,
        verbose=verbose_sr,
        length_penalty=length_penalty,
        equation_predicate=equation_predicate,
    )

    # ── Package split data ───────────────────────────────────────────────────
    split_data = {
        'X_train':      X_fit,
        'y_train':      y_fit,
        'y_std_train':  y_std_fit,
        'X_test':       X_eval,
        'y_test':       y_eval,
        'y_std_test':   y_std_eval,
        'dy_sr_test':   dy_sr_eval,
        'Fs_test':      Fs_eval,
        'X_sr_val':     X_sr_val,
        'y_sr_val':     y_sr_val,
        'y_std_sr_val': y_std_sr_val,
        'n_params':     n_params,
        'components_to_fit': components_to_fit,
        'slice_fisher': slice_fisher,
        'augmented':    use_augmentation,
    }

    if save_split_data and split_data_npz_name:
        os.makedirs(parent_dir, exist_ok=True)
        split_npz_path = os.path.join(parent_dir, split_data_npz_name)
        rs_save = np.int64(-1) if random_state is None else np.int64(random_state)
        save_kwargs = dict(
            X_train=X_fit,
            X_test=X_eval,
            y_train=y_fit,
            y_test=y_eval,
            y_std_train=y_std_fit,
            y_std_test=y_std_eval,
            dy_sr_test=dy_sr_eval,
            Fs_test=Fs_eval,
            n_params=np.int64(n_params),
            components_to_fit=np.asarray(components_to_fit, dtype=np.int64),
            slice_fisher=np.bool_(slice_fisher),
            test_size=np.float64(test_size),
            shuffle=np.bool_(shuffle),
            random_state=rs_save,
            augmented=np.bool_(use_augmentation),
        )
        if X_sr_val is not None:
            save_kwargs.update(
                X_sr_val=X_sr_val,
                y_sr_val=y_sr_val,
                y_std_sr_val=y_std_sr_val,
            )
        np.savez_compressed(split_npz_path, **save_kwargs)

    return mdl_coords, frob_coords, analysis, split_data
