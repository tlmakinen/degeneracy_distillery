"""
Preprocessing utilities for data loading, rotations, and coordinate alignment.

This module provides:
- Linear algebra helpers (rotation, reflection, alignment)
- Coordinate transformation functions (PCA, scaling)
- Fisher matrix flattening functions
- Ensemble rotation and processing

Author: Consolidated from postprocessing.ipynb
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List, Dict, Any

# =============================================================================
# LINEAR ALGEBRA HELPERS
# =============================================================================

def weighted_std(values: jnp.ndarray, weights: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """
    Compute weighted standard deviation.
    
    Parameters
    ----------
    values : jnp.ndarray
        Input values
    weights : jnp.ndarray
        Weights (normalized internally to sum to 1)
    axis : int
        Axis along which to compute
        
    Returns
    -------
    jnp.ndarray
        Weighted standard deviation
    """
    average = jnp.average(values, weights=weights, axis=axis)
    variance = jnp.average((values - average)**2, weights=weights, axis=axis)
    return jnp.sqrt(variance)


def reflection(u: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Reflection of u on hyperplane with normal vector n.
    
    Parameters
    ----------
    u : np.ndarray
        Input vector or matrix of shape (m, k)
    n : np.ndarray
        Normal vector of shape (m,)
    
    Returns
    -------
    np.ndarray
        Reflected vector or matrix of same shape as u
    """
    n = n.reshape(-1, 1)
    return u - (2 * n @ (n.T @ u) / (n.T @ n))


def rotate_x_to_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix R such that R @ (x/|x|) = y/|y|.
    
    Uses double reflection method: first reflect over (v+u), then over v.
    
    Parameters
    ----------
    x : np.ndarray
        Source vector to be rotated
    y : np.ndarray
        Target vector
    
    Returns
    -------
    np.ndarray
        Rotation matrix R
    """
    u = x / np.linalg.norm(x)
    v = y / np.linalg.norm(y)
    N = u.shape[0]
    S = reflection(np.eye(N), v + u)
    R = reflection(S, v)
    return R


def reflection_rotation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix using reflection formula.
    
    R = I - (u+v)(u+v)^T / (1 + <u,v>) + 2 v u^T
    
    Parameters
    ----------
    x : np.ndarray
        Source vector
    y : np.ndarray
        Target vector

    Returns
    -------
    np.ndarray
        Rotation matrix R such that R @ u = v (where u,v are unit vectors)
    """
    u = x / np.linalg.norm(x)
    v = y / np.linalg.norm(y)
    
    I = np.eye(u.size)
    w = u + v
    dot = np.dot(u, v)
    R = I - np.outer(w, w) / (1.0 + dot) + 2.0 * np.outer(v, u)
    return R


def kabsch_jax(P: jnp.ndarray, Q: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Kabsch algorithm: compute optimal rotation and translation to align P -> Q.
    
    Parameters
    ----------
    P : jnp.ndarray
        Source points of shape (N, d)
    Q : jnp.ndarray
        Target points of shape (N, d)
    
    Returns
    -------
    R : jnp.ndarray
        Optimal rotation matrix
    t : jnp.ndarray
        Optimal translation vector
    rmsd : float
        Root mean square deviation after alignment
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    centroid_P = jnp.mean(P, axis=0)
    centroid_Q = jnp.mean(Q, axis=0)
    t = centroid_Q - centroid_P

    p = P - centroid_P
    q = Q - centroid_Q

    H = jnp.dot(p.T, q)
    U, S, Vt = jnp.linalg.svd(H)

    # Ensure right-handed coordinate system
    if jnp.linalg.det(jnp.dot(Vt.T, U.T)) < 0.0:
        Vt = Vt.at[-1, :].mul(-1.0)

    R = jnp.dot(Vt.T, U.T)
    rmsd = jnp.sqrt(jnp.sum(jnp.square(jnp.dot(p, R.T) - q)) / P.shape[0])

    return R, t, rmsd


def ortho_rotation(components: np.ndarray, method: str = "quartimax", 
                   tol: float = 1e-6, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orthogonal rotation (varimax or quartimax).
    
    Parameters
    ----------
    components : np.ndarray
        Component matrix of shape (n_samples, n_components)
    method : str
        "varimax" or "quartimax"
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    rotation_matrix : np.ndarray
        The rotation matrix
    rotated_components : np.ndarray
        Rotated components
    """
    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    var = 0

    for _ in range(max_iter):
        comp_rot = np.dot(components, rotation_matrix)
        if method == "varimax":
            tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / nrow)
        elif method == "quartimax":
            tmp = 0
        else:
            raise ValueError(f"Unknown method: {method}")
            
        u, s, v = np.linalg.svd(np.dot(components.T, comp_rot**3 - tmp))
        
        # Ensure right-handed coordinate system
        if np.linalg.det(np.dot(v.T, u.T)) < 0.0:
            v[-1, :] *= -1.0
            
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new

    return rotation_matrix.T, np.dot(components, rotation_matrix).T


# =============================================================================
# DATA TRANSFORMATION FUNCTIONS
# =============================================================================

def my_standard_scale(X: np.ndarray, dx: Optional[np.ndarray] = None, 
                      mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """
    Apply standard scaling.
    
    Parameters
    ----------
    X : np.ndarray
        Data to scale
    dx : np.ndarray, optional
        If provided, scale dx instead of X
    mean : float
        Mean to subtract
    std : float
        Standard deviation to divide by
        
    Returns
    -------
    np.ndarray
        Scaled data
    """
    if dx is None:
        return (X - mean) / std
    else:
        return dx / std


def vanilla_righthanded_pca(X: np.ndarray, cov: Optional[np.ndarray] = None, 
                            components: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA with right-handed coordinate system enforcement.
    
    Parameters
    ----------
    X : np.ndarray
        Data of shape (n_samples, n_features)
    cov : np.ndarray, optional
        Precomputed covariance matrix
    components : int, optional
        Number of components to keep
        
    Returns
    -------
    Vt : np.ndarray
        Principal components (transposed)
    X_transformed : np.ndarray
        Transformed data
    """
    n_samples, n_features = X.shape
    mean_ = X.mean(0)
    
    if cov is None:
        C = X.T @ X
        C -= n_samples * np.reshape(mean_, (-1, 1)) * np.reshape(mean_, (1, -1))
        C /= n_samples - 1
    else:
        C = cov

    eigenvals, eigenvecs = np.linalg.eigh(C)
    eigenvals = np.flip(eigenvals, axis=0)
    eigenvecs = np.flip(eigenvecs, axis=1)

    if components is not None:
        eigenvecs[:, components:] = 0.0

    Vt = eigenvecs.T
    X_centered = X - mean_

    return Vt.T, np.einsum("bj,jk->bk", X_centered, Vt.T)


# =============================================================================
# FISHER MATRIX FLATTENING FUNCTIONS
# =============================================================================

def flatten_with_numerical_jacobian(J_eta: jnp.ndarray, F: jnp.ndarray, 
                                     A: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Flatten Fisher matrix using numerical Jacobian.
    
    This is the CANONICAL implementation that should be used throughout.
    Computes Q = (A @ J^+)^T @ F @ (A @ J^+)
    
    Parameters
    ----------
    J_eta : jnp.ndarray
        Jacobian matrix of shape (n_outputs, n_params) or (n_params, n_outputs)
    F : jnp.ndarray
        Fisher information matrix of shape (n_params, n_params)
    A : jnp.ndarray, optional
        Optional transformation matrix. If None, identity is used.
        
    Returns
    -------
    Q : jnp.ndarray
        Flattened Fisher matrix
        
    Notes
    -----
    The flattening transforms the Fisher matrix from parameter space to 
    coordinate space via the pseudoinverse of the Jacobian.
    """
    Jeta_inv = jnp.linalg.pinv(J_eta)
    
    if A is not None:
        Jeta_inv = A @ Jeta_inv
        
    Q = Jeta_inv.T @ F @ Jeta_inv
    return Q


def flatten_fisher(J: jnp.ndarray, F: jnp.ndarray, 
                   A: Optional[jnp.ndarray] = None,
                   transpose: bool = False) -> jnp.ndarray:
    """
    Flatten Fisher matrix - wrapper with transpose option.
    
    This provides backward compatibility with different calling conventions.
    
    Parameters
    ----------
    J : jnp.ndarray
        Jacobian matrix
    F : jnp.ndarray
        Fisher information matrix
    A : jnp.ndarray, optional
        Transformation matrix
    transpose : bool
        If True, transpose J before computing pseudoinverse
        
    Returns
    -------
    Q : jnp.ndarray
        Flattened Fisher matrix
    """
    if transpose:
        J = J.T
    return flatten_with_numerical_jacobian(J, F, A)


def batch_flatten_fisher(Js: jnp.ndarray, Fs: jnp.ndarray,
                         A: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Batch flatten Fisher matrices.
    
    Parameters
    ----------
    Js : jnp.ndarray
        Batch of Jacobians of shape (batch, n_outputs, n_params)
    Fs : jnp.ndarray
        Batch of Fisher matrices of shape (batch, n_params, n_params)
    A : jnp.ndarray, optional
        Transformation matrix
        
    Returns
    -------
    Qs : jnp.ndarray
        Batch of flattened Fisher matrices
    """
    flatten_fn = lambda J, F: flatten_with_numerical_jacobian(J, F, A)
    return jax.vmap(flatten_fn)(Js, Fs)


def get_eigenvalues(M: jnp.ndarray) -> jnp.ndarray:
    """
    Get sorted eigenvalues of a matrix.
    
    Parameters
    ----------
    M : jnp.ndarray
        Input matrix
        
    Returns
    -------
    eigenvalues : jnp.ndarray
        Sorted eigenvalues (ascending)
    """
    eigenvalues, _ = jnp.linalg.eigh(M)
    return eigenvalues[eigenvalues.argsort()]


# =============================================================================
# COORDINATE ROTATION
# =============================================================================

def rotate_coords(y: np.ndarray, theta: np.ndarray, Fs: np.ndarray, 
                  dy: np.ndarray, y_reference: Optional[np.ndarray] = None,
                  theta_fid: Optional[np.ndarray] = None, 
                  use_var: bool = False, smallest: bool = False,
                  tol: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate coordinates to align with reference and apply PCA-based rotation.
    
    This function performs:
    1. Centering of y
    2. Kabsch alignment to reference
    3. Fisher-based eigenvalue decomposition
    4. Optional rotation to align with eigenvector
    
    Parameters
    ----------
    y : np.ndarray
        Network outputs of shape (n_samples, n_outputs)
    theta : np.ndarray
        Parameters of shape (n_samples, n_params)
    Fs : np.ndarray
        Fisher matrices (averaged)
    dy : np.ndarray
        Jacobians of shape (n_samples, n_outputs, n_params)
    y_reference : np.ndarray, optional
        Reference outputs for Kabsch alignment
    theta_fid : np.ndarray, optional
        Fiducial parameter values
    use_var : bool
        Use variance-based covariance
    smallest : bool
        Align to smallest eigenvalue direction
    tol : float
        Tolerance for eigenvalue cutoff
        
    Returns
    -------
    y_rotated : np.ndarray
        Rotated outputs
    dy : np.ndarray
        Original Jacobians
    dy_sr : np.ndarray
        Rotated Jacobians
    rotmat : np.ndarray
        Combined rotation matrix
    A : np.ndarray
        Transformation matrix (identity in current implementation)
    """
    theta = theta.copy()
    
    # Center y
    ybar = y.mean(0)
    y = y - ybar

    # zero-out reference point
    y_reference -= y_reference.mean(0)

    
    # Find fiducial point
    if theta_fid is None:
        theta_fid = theta.mean(0)
    
    argstar = np.argmin(np.sum((theta - theta_fid)**2, -1))
    theta_star = theta[argstar]
    eta_star = y_reference[argstar]
    dy_star = dy[argstar]
    
    print("thetastar", theta_star)
    
    # Eigenvalue calculation with prior normalization
    delta = jnp.abs(theta.max(0) - theta.min(0))
    prior_norm = jnp.outer(delta, delta)
    F_norm = Fs / prior_norm
    
    if use_var:
        C = F_norm.std(0) / (F_norm[argstar] + prior_norm)
    else:
        C = jnp.linalg.pinv(F_norm[argstar])
    
    eigenval, eigenvec = np.linalg.eigh(C)
    idx = eigenval.argsort()
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]
    
    A_eig = eigenvec[:, :]
    S = np.matmul(A_eig.T, theta_star)
    
    if smallest:
        eigidx = np.min(np.arange(eigenval.shape[0])[eigenval > tol])
        print(f'smallest evalue idx above tolerance {tol:.3f}: {eigidx}')
        eigidx = 0
    else:
        eigidx = -1
    
    # Kabsch alignment to reference
    if y_reference is not None:

        # set theta_star to first (best constrained) eigenvalue
        theta_star = eigenvec[:, 0]
        
        # first align reference to theta star
        rotmat0 = rotate_x_to_y(eta_star, theta_star)
        y_reference = np.einsum("ij,bj->bi", rotmat0, y_reference)

        # then kabsch rotate
        rotmat, *_ = kabsch_jax(y, y_reference)
        
        # y += eigenvec[:, 0]

    else:
        rotmat = rotate_x_to_y(eta_star, theta_star)
    
    # print("rotmat", rotmat)
    y = np.einsum("ij,bj->bi", rotmat, y)
    A = np.eye(y.shape[-1])
    
    # Rotate Jacobian
    dy_sr = jnp.einsum("ij,bjk->bik", rotmat, dy)

    y -= y_reference.min(0)
    
    return y, dy, dy_sr, rotmat, A


def process_ensemble_rotation(datafile: Dict[str, Any], 
                               randidx: np.ndarray,
                               Favg: np.ndarray,
                               best_model_idx: int,
                               n_d: float = 1.0,
                               verbose: bool = True) -> Dict[str, Any]:
    """
    Process and rotate ensemble members to a common reference frame.
    
    This function:
    1. Loops through all ensemble members
    2. Rotates each to align with the best model using Kabsch alignment
    3. Computes weighted averages across the ensemble
    4. Masks out samples with zero standard deviation
    
    Parameters
    ----------
    datafile : Dict[str, Any]
        Loaded npz datafile containing:
        - 'eta_ensemble': Network outputs (n_nets, n_samples, n_outputs)
        - 'Jbar_ensemble': Jacobians (n_nets, n_samples, n_outputs, n_params)
        - 'F_ensemble': Fisher matrices (n_nets, n_samples, n_params, n_params)
        - 'theta': Parameters (n_samples, n_params)
        - 'ensemble_weights': Weights for each network
    randidx : np.ndarray
        Indices of samples to use
    Favg : np.ndarray
        Averaged Fisher matrix for rotation computation
    best_model_idx : int
        Index of the best model to use as reference
    n_d : float
        Normalization factor (e.g., number of data points)
    verbose : bool
        Whether to print progress information
        
    Returns
    -------
    result : Dict[str, Any]
        Dictionary containing:
        - 'y': Weighted average of rotated outputs (n_samples, n_outputs)
        - 'y_std': Weighted std of outputs (n_samples, n_outputs)
        - 'dy': Weighted average of Jacobians (n_samples, n_outputs, n_params)
        - 'dy_sr': Weighted average of rotated Jacobians
        - 'Fs': Weighted average of Fisher matrices
        - 'X': Parameters (masked)
        - 'ys': All rotated outputs (n_nets, n_samples, n_outputs)
        - 'dys': All Jacobians (n_nets, n_samples, n_outputs, n_params)
        - 'dys_sr': All rotated Jacobians
        - 'ensemble_Fs': All Fisher matrices
        - 'ensemble_weights': Network weights
        - 'rotmats': Rotation matrices for each network
        - 'rotmat_avg': Weighted average rotation matrix
        - 'mask': Boolean mask for valid samples
        - 'Jbar': Copy of weighted average Jacobian
    """
    ensemble_weights_raw = datafile['ensemble_weights']
    num_nets = len(ensemble_weights_raw)
    
    # Get reference for Kabsch alignment
    y_reference = datafile['eta_ensemble'][best_model_idx][randidx]
    
    ys = []
    dys = []
    dys_sr = []
    Fs = []
    ensemble_weights = []
    rotmats = []
    
    for i in range(num_nets):
        y = datafile["eta_ensemble"][i][randidx]
        dy = datafile["Jbar_ensemble"][i][randidx]
        _F = datafile["F_ensemble"][i][randidx]
        X = datafile['theta'][randidx]
        
        if verbose:
            print(f"Network {i}: y.min() = {y.min():.6f}, weight = {ensemble_weights_raw[i]:.1f}")
        
        # Rotate to align with reference
        y_rot, dy_orig, dy_sr_rot, rotmat, A = rotate_coords(
            y, theta=X, Fs=Favg, dy=dy, 
            y_reference=y_reference
        )
        
        if verbose:
            print(f"  Shapes: y={y_rot.shape}, X={X.shape}")
        
        ys.append(y_rot)
        dys_sr.append(dy_sr_rot)
        dys.append(dy_orig)
        rotmats.append(rotmat)
        ensemble_weights.append(ensemble_weights_raw[i])
        Fs.append(_F)
    
    # Convert to arrays and apply n_d normalization
    ys = np.array(ys) / np.sqrt(n_d)
    ensemble_weights = np.array(ensemble_weights)
    dys = np.array(dys) / np.sqrt(n_d)  # DIVIDE BY SQRT(N_D) IN JACOBIAN
    dys_sr = np.array(dys_sr) / np.sqrt(n_d)
    rotmats = np.array(rotmats)
    Fs = np.array(Fs) / n_d
    
    ensemble_Fs = Fs.copy()
    
    if verbose:
        print(f"\nEnsemble shapes: dys={dys.shape}, ys={ys.shape}")
    
    # Compute weighted averages
    y = np.average(ys, axis=0, weights=ensemble_weights)
    y_std = weighted_std(jnp.array(ys), weights=jnp.array(ensemble_weights), axis=0)
    y_std = np.array(y_std)  # Convert back to numpy
    
    rotmat_avg = np.average(rotmats, weights=ensemble_weights, axis=0)
    
    # Mask out samples with zero std (numerical issues)
    mask = (y_std[:, 0] != 0)
    
    y = y[mask]
    y_std = y_std[mask]
    
    # Average Jacobians with masking
    dy = np.average(dys, axis=0, weights=ensemble_weights)[mask]
    dy_sr = np.average(dys_sr, axis=0, weights=ensemble_weights)[mask]
    
    # Apply mask to individual ensemble members
    dys = np.array([j[mask] for j in dys])
    dys_sr = np.array([j[mask] for j in dys_sr])
    
    Jbar = dy.copy()
    
    # Get masked parameters and Fisher matrices
    X = datafile["theta"][randidx][mask]
    Fs = np.array([f[mask] for f in Fs])
    ensemble_Fs = Fs.copy()
    
    Fs_avg = np.average(Fs, axis=0, weights=ensemble_weights)
    
    # Apply randidx and mask to eta_ensemble
    eta_ensemble_masked = datafile['eta_ensemble'][:, randidx, :][:, mask, :]
    
    return {
        'y': y,
        'y_std': y_std,
        'dy': dy,
        'dy_sr': dy_sr,
        'Fs': Fs_avg,
        'X': X,
        'ys': ys,
        'dys': dys,
        'dys_sr': dys_sr,
        'ensemble_Fs': ensemble_Fs,
        'ensemble_weights': ensemble_weights,
        'rotmats': rotmats,
        'rotmat_avg': rotmat_avg,
        'mask': mask,
        'Jbar': Jbar,
        'n_d': n_d,
        'eta_ensemble': eta_ensemble_masked,
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_and_process_data(datapath: str, filename: str, 
                          num_samps: int = 4000, seed: int = 44,
                          process_ensemble: bool = False,
                          n_d: float = 1.0,
                          y_reference_index: int = None,
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Load and process flattening data file.
    
    Parameters
    ----------
    datapath : str
        Path to data directory
    filename : str
        Data filename
    num_samps : int
        Number of samples to use
    seed : int
        Random seed
    process_ensemble : bool
        If True, also run process_ensemble_rotation
    n_d : float
        Normalization factor for ensemble processing
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    data : Dict[str, Any]
        Processed data dictionary. If process_ensemble=True, includes
        all outputs from process_ensemble_rotation.
    """
    np.random.seed(seed)
    
    datafile = np.load(datapath + filename)
    
    X = datafile["theta"]
    ensemble_weights = datafile["ensemble_weights"]
    best_model_idx = int(np.argmax(ensemble_weights)) if y_reference_index is None else y_reference_index
    
    if verbose:
        print(f"best model {best_model_idx}")
    
    randidx = np.arange(num_samps)
    X = datafile["theta"][randidx]
    
    Favg = np.average(datafile['F_ensemble'], weights=ensemble_weights, axis=0)
    
    result = {
        'X': X,
        'ensemble_weights': ensemble_weights,
        'best_model_idx': best_model_idx,
        'randidx': randidx,
        'Favg': Favg,
        'datafile': datafile,
        'eta_ensemble': datafile['eta_ensemble'],
    }
    
    if process_ensemble:
        # Run full ensemble processing
        ensemble_result = process_ensemble_rotation(
            datafile=datafile,
            randidx=randidx,
            Favg=Favg,
            best_model_idx=best_model_idx,
            n_d=n_d,
            verbose=verbose
        )
        # Merge results
        result.update(ensemble_result)
    
    return result


if __name__ == "__main__":
    # Run basic tests
    print("Testing preprocessing utilities...")
    
    # Test flatten_with_numerical_jacobian
    J = jnp.array([[1.0, 0.5], [0.3, 1.0]])
    F = jnp.eye(2)
    
    Q1 = flatten_with_numerical_jacobian(J, F)
    Q2 = flatten_with_numerical_jacobian(J, F, A=jnp.eye(2))
    
    assert jnp.allclose(Q1, Q2), "Flattening with identity A should match no A"
    print("  flatten_with_numerical_jacobian: OK")
    
    # Test eigenvalues
    M = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    evals = get_eigenvalues(M)
    assert evals[0] < evals[1], "Eigenvalues should be sorted"
    print("  get_eigenvalues: OK")
    
    # Test rotation functions
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    R = rotate_x_to_y(x, y)
    assert np.allclose(R @ x, y), "Rotation should align x to y"
    print("  rotate_x_to_y: OK")
    
    print("\nAll tests passed!")
