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
except ImportError:
    from preprocessing_utils import (
        flatten_with_numerical_jacobian,
        batch_flatten_fisher,
        weighted_std,
    )

# Try importing ESR (required for complexity calculations)
try:
    import esr.generation.generator
    ESR_AVAILABLE = True
except ImportError:
    ESR_AVAILABLE = False


# =============================================================================
# SR EXPRESSION UTILITIES
# =============================================================================

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
    """
    Q, R = jnp.linalg.qr(A)
    D = jnp.diag(jnp.sign(jnp.diag(R)))
    Q = Q @ D
    return Q


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
        
        param_list = [f"b{i}" for i in range(len(pars))]
        all_x = ' '.join([f'X{i}' for i in range(1, X.shape[1] + 1)])
        all_x = list(sympy.symbols(all_x, real=True))
        all_b = list(sympy.symbols(param_list, real=True))
        eq_jax = sympy.lambdify(all_b + all_x, expr, modules=["jax"])
        
        def get_jac_row(p):
            myeq = lambda *args: eq_jax(*p, *args)
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
            myeq = lambda *args: all_fns[i](*p, *args)
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
    
    param_list = [f"b{i}" for i in range(len(pars))]
    labels = nodes.to_list(basis_functions)
    
    all_x = ' '.join([f'X{i}' for i in range(1, X.shape[1] + 1)])
    all_x = list(sympy.symbols(all_x, real=True))
    all_b = list(sympy.symbols(param_list, real=True))
    eq_fn = sympy.lambdify(all_b + all_x, expr, modules=[module])
    
    linear_b = list(sympy.symbols(linear_par_names, real=True))
    
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
                                  batch_size: int = 5) -> Tuple[List[str], List[List[float]]]:
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
    
    return new_expr, consts


def postprocess_eqs(coordinates: List[str],
                    X: np.ndarray,
                    Fs: np.ndarray,
                    n_params: int,
                    A_rotation: Optional[np.ndarray] = None,
                    threshold: float = 0.05,
                    importance_based: bool = True,
                    batch_removal: bool = False,
                    batch_size: int = 5,
                    remove_floats: bool = False,
                    decimal: int = 3,
                    rational: bool = False,
                    verbose: bool = True,
                    perturbation: float = 1e-4,
                    check_flattening_fn: Optional[Callable] = None,
                    module: str = "jax") -> Tuple[List[str], List[List[float]]]:
    """
    High-level wrapper for postprocessing symbolic expressions.
    
    This function provides a simplified interface that:
    1. Parses symbolic expression strings into components
    2. Extracts linear and nonlinear parameters
    3. Applies importance-based pruning with optional rotation
    4. Returns simplified expressions
    
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
        Rotation matrix for coordinate transformation. If None, uses identity
        (no rotation). Shape should be (n_params, n_params).
    threshold : float, default=0.05
        Relative loss threshold for removing coefficients. Higher values
        lead to more aggressive pruning.
    importance_based : bool, default=True
        Use importance-based ordering (recommended). If False, uses legacy
        sequential pruning which is permutation-dependent.
    batch_removal : bool, default=False
        Attempt to remove multiple low-importance coefficients simultaneously
        for faster pruning. Recommended for large problems.
    batch_size : int, default=5
        Number of coefficients to attempt removing in each batch
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
        
    Returns
    -------
    pruned_expressions : List[str]
        Pruned and simplified symbolic expressions
    constants : List[List[float]]
        Constants in the pruned expressions
        
    Examples
    --------
    >>> # Basic usage with default settings
    >>> pruned_exprs, consts = postprocess_eqs(
    ...     coordinates=mdl_coordinates,
    ...     X=X_test,
    ...     Fs=Fs_test,
    ...     n_params=6
    ... )
    
    >>> # Aggressive pruning with batch removal for speed
    >>> pruned_exprs, consts = postprocess_eqs(
    ...     coordinates=mdl_coordinates,
    ...     X=X_test,
    ...     Fs=Fs_test,
    ...     n_params=6,
    ...     threshold=0.1,
    ...     batch_removal=True,
    ...     batch_size=10
    ... )
    
    >>> # With coordinate rotation
    >>> pruned_exprs, consts = postprocess_eqs(
    ...     coordinates=mdl_coordinates,
    ...     X=X_test,
    ...     Fs=Fs_test,
    ...     n_params=6,
    ...     A_rotation=optimized_rotation_matrix
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
    
    # Default to identity rotation if not provided
    if A_rotation is None:
        A_rotation = np.eye(n_params)
    else:
        A_rotation = np.array(A_rotation).reshape((n_params, n_params))
    
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
        batch_size=batch_size
    )
    
    if verbose:
        print("\nPostprocessing complete!")
        print(f"Input expressions: {len(coordinates)}")
        print(f"Output expressions: {len(pruned_expressions)}")
    
    return pruned_expressions, constants


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
