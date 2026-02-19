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
from typing import List, Tuple, Dict, Optional, Callable
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from pyoperon.sklearn import SymbolicRegressor
import sympy
import esr.generation.generator


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


# =============================================================================
# DESCRIPTION LENGTH AND FLATTENING
# =============================================================================

@jax.jit
def norm(A: jnp.ndarray) -> float:
    """Frobenius norm of a matrix."""
    return jnp.sqrt(jnp.einsum('ij,ij->', A, A))


def compute_DL(eq: str, component_idx: int, X: np.ndarray, y: np.ndarray, 
               y_std: np.ndarray, dy_sr: np.ndarray, Fs: np.ndarray,
               n_params: int) -> Tuple[float, str, float, float, float]:
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
        ["+", "*", "-", "/", "^"]  # type2
    ]

    # Define sympy functions
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
        "logAbs": logAbs
    }
    
    # Parse equation
    expr, pars = replace_floats(eq)
    expr, nodes, complexity = esr.generation.generator.string_to_node(
        expr, 
        basis_functions, 
        evalf=True, 
        allow_eval=True, 
        check_ops=True, 
        locs=sympy_locs
    )
    
    param_list = [f"b{i}" for i in range(len(pars))]
    latex_expr = sympy.latex(expr)
    
    # Compute Aifeyn complexity: k*log(n) + sum_i log|c_i|
    aifeyn = esr.generation.generator.aifeyn_complexity(nodes.to_list(basis_functions), param_list)
    
    # Turn function into callable object
    all_x = ' '.join([f'X{i}' for i in range(1, X.shape[1] + 1)])
    all_x = list(sympy.symbols(all_x, real=True))
    all_b = list(sympy.symbols(param_list, real=True))
    eq_jax = sympy.lambdify(all_b + all_x, expr, modules=["jax"])

    # Define loss function (negative log-likelihood)
    def myloss(p):
        ypred = eq_jax(*p, *X.T)
        result = jnp.sum((y[:, component_idx] - ypred)**2 / 2 / y_std[:, component_idx]**2)
        return result
    
    # Define flattening loss
    def frob_loss(p):
        def get_jac_row(p):
            myeq = lambda *args: eq_jax(*p, *args)
            # Compute Jacobian for this component
            yjac = jax.jacrev(myeq, argnums=list(range(0, X.shape[1])))
            Jpred = jnp.array(jax.vmap(yjac)(*X.T)).T
            return Jpred

        jac_row = get_jac_row(pars)
        
        # Assign the SR expression's jacobian row to a copy of the network Jac
        jacobian = dy_sr.copy()
        jacobian[:, component_idx, :] = np.array(jac_row)

        # Flatten Fisher matrices
        def flatten_fisher(J, F):
            invJ = jnp.linalg.pinv(J)
            return invJ.T @ F @ invJ

        flats = jax.vmap(flatten_fisher)(jacobian, Fs)
        nn_flats = jax.vmap(flatten_fisher)(dy_sr, Fs)
        
        fn = lambda q: norm((q - jnp.eye(n_params))) + norm((jnp.linalg.pinv(q) - jnp.eye(n_params)))

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
    
    # Combine the terms
    DL = neglogL + aifeyn + param_codelen
    
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
        
        reg = SymbolicRegressor(
            allowed_symbols=allowed_symbols,
            offspring_generator='basic',
            optimizer_iterations=optimizer_iterations,
            max_length=max_length,
            max_depth=max_depth,
            initialization_method='btc',
            n_threads=multiprocessing.cpu_count(),
            objectives=objectives,
            epsilon=epsilon,
            random_state=random_state,
            reinserter='keep-best',
            max_evaluations=int(max_evaluations),
            symbolic_mode=False,
            time_limit=int(time_limit),
            generations=int(generations),
            add_model_scale_term=True,
            add_model_intercept_term=True,
        )

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
    verbose: bool = True
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
        eqs = list(np.array(data['model'])[mse_mask])

        # Compute metrics for all equations
        all_DL = np.ones(len(eqs)) * np.inf
        all_logL = np.ones(len(eqs)) * np.inf
        all_frobloss = np.ones(len(eqs)) * np.inf
        all_latex = [None] * len(eqs)
        all_complexity = np.ones(len(eqs)) * np.inf

        for j, eq in enumerate(tqdm(eqs, desc=f"Component {i+1}")):
            try:
                c, latex, logL, DL, frobloss = compute_DL(
                    eq, idx, X, y, y_std, dy_sr, Fs, n_params
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
                pareto_latex.append(np.array(all_latex)[mask][ibest_model_at_complexity])
                pareto_eqs.append(np.array(eqs)[mask][ibest_model_at_complexity])
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
        
        mdl_coordinates.append(pareto_eqs[ibest])
        frob_coordinates.append(pareto_eqs[ibest_frob])

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
    
    # Split data using train_test_split
    # Note: we need to keep all arrays aligned, so we split them together
    arrays_to_split = [X, y, y_std, dy_sr, Fs]
    
    split_result = train_test_split(
        *arrays_to_split,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )
    
    # Unpack split results
    X_train, X_val = split_result[0], split_result[1]
    y_train, y_val = split_result[2], split_result[3]
    y_std_train, y_std_val = split_result[4], split_result[5]
    dy_sr_train, dy_sr_val = split_result[6], split_result[7]
    Fs_train, Fs_val = split_result[8], split_result[9]
    
    # Optionally slice Fisher matrices and parameters to match components_to_fit
    if slice_fisher:
        # Slice parameter space to only include dimensions for components being fitted
        # Assumes 1-to-1 mapping: component i corresponds to parameter i
        X_train = X_train[:, components_to_fit]
        X_val = X_val[:, components_to_fit]
        
        # Slice Jacobians: (n_samples, n_components, n_params) -> (n_samples, n_components_fit, n_params_fit)
        # First slice rows (output dimensions), then columns (parameter dimensions)
        dy_sr_train = dy_sr_train[:, components_to_fit, :][:, :, components_to_fit]
        dy_sr_val = dy_sr_val[:, components_to_fit, :][:, :, components_to_fit]
        
        # Slice Fisher matrices: (n_samples, n_params, n_params) -> (n_samples, n_params_fit, n_params_fit)
        # Slice both rows and columns to extract the submatrix
        Fs_train = Fs_train[:, components_to_fit, :][:, :, components_to_fit]
        Fs_val = Fs_val[:, components_to_fit, :][:, :, components_to_fit]
        
        # Update n_params to reflect sliced dimension
        n_params = len(components_to_fit)
    
    # Separate kwargs for fitting and analysis
    fit_kwargs = {k: v for k, v in sr_kwargs.items() 
                  if k not in ['equation_set', 'max_complexity_thresh']}
    
    # Extract analysis-specific kwargs
    equation_set = sr_kwargs.get('equation_set', 'pareto')
    max_complexity_thresh = sr_kwargs.get('max_complexity_thresh', 14)
    
    # Fit SR models on training set
    fit_symbolic_regression(
        X_train, y_train, y_std_train,
        components_to_fit, parent_dir,
        **fit_kwargs
    )
    
    # Analyze results on validation set
    mdl_coords, frob_coords, analysis = analyze_equations(
        X_val, y_val, y_std_val, dy_sr_val, Fs_val,
        n_params, components_to_fit, parent_dir,
        max_complexity_thresh=max_complexity_thresh,
        equation_set=equation_set
    )
    
    # Package split data for return
    split_data = {
        'X_train': X_train,
        'X_test': X_val,
        'y_train': y_train,
        'y_test': y_val,
        'y_std_train': y_std_train,
        'y_std_test': y_std_val,
        'dy_sr_train': dy_sr_train,
        'dy_sr_test': dy_sr_val,
        'Fs_train': Fs_train,
        'Fs_test': Fs_val,
        'n_params': n_params,
        'components_to_fit': components_to_fit,
        'slice_fisher': slice_fisher,
    }
    
    return mdl_coords, frob_coords, analysis, split_data
