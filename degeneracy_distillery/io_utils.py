"""
I/O utilities with FlexibleDict - flexible naming convention support.

This module provides:
- FlexibleDict: Dictionary with multiple naming convention support
- I/O functions for loading and saving results with flexible access
- Functions for working with saved results

Author: Degeneracy Distillery Team
"""

import numpy as np
from typing import Any, Dict, Union, Optional, KeysView, ValuesView, ItemsView
from pathlib import Path


# =============================================================================
# FlexibleDict - Dictionary with Multiple Naming Conventions
# =============================================================================

class FlexibleDict(dict):
    """
    Dictionary that maps common aliases to canonical keys for flexible notation.
    
    This class allows users to access data using their preferred notation style:
    - Statistical/Mathematical: theta, eta, x
    - Machine Learning: X, y, params, features
    - Descriptive: observations, coordinates
    
    Examples
    --------
    >>> results = FlexibleDict({'theta': params_array, 'eta': coords_array})
    >>> # All of these access the same data:
    >>> params1 = results['theta']  # Canonical Greek notation
    >>> params2 = results['X']       # ML convention (for symbolic regression)
    >>> params3 = results['params']  # Descriptive name
    >>> # All three variables point to the same array
    
    >>> coords1 = results['eta']    # Canonical Greek notation
    >>> coords2 = results['y']       # ML convention (for symbolic regression)
    >>> coords3 = results['coords']  # Descriptive name
    
    Canonical Keys
    --------------
    The following are the canonical (preferred) keys that should be used when
    storing data:
    - 'theta' : Model parameters (θ)
    - 'eta' : Learned coordinates/embeddings (η)
    - 'x' : Observations/data
    - 'Fs' : Fisher information matrices (shape: n_simulations × n_params × n_params)
    - 'mle' : Maximum likelihood estimates
    
    Note: Use 'Fs' (plural) for the canonical key. Single 'F' is an alias for backwards compatibility.
    
    All other names are aliases that map to these canonical keys.
    """
    
    # Define the canonical key mappings
    _aliases: Dict[str, str] = {
        # For theta (parameters)
        'X': 'theta',           # ML convention for SR input
        'params': 'theta',      # Descriptive
        'parameters': 'theta',  # Fully descriptive
        
        # For eta (learned coordinates)
        'y': 'eta',             # ML convention for SR target
        'coords': 'eta',        # Descriptive
        'coordinates': 'eta',   # Fully descriptive
        'embeddings': 'eta',    # Alternative descriptive
        
        # For x (observations/data)
        'observations': 'x',    # Descriptive
        'obs': 'x',             # Short descriptive
        'data': 'x',            # Common alternative
        'features': 'x',        # ML convention
        
        # For Fisher matrices (plural - multiple per simulation)
        'F': 'Fs',              # Backwards compatibility (singular -> plural)
        'fisher': 'Fs',         # Descriptive
        'fisher_matrices': 'Fs', # Fully descriptive
        'F_matrices': 'Fs',     # Alternative
        
        # For MLE
        'mle_predictions': 'mle',
        'theta_hat': 'mle',
        'estimates': 'mle',
    }
    
    def __getitem__(self, key: str) -> Any:
        """
        Get item using either canonical key or alias.
        
        Parameters
        ----------
        key : str
            Key or alias to access
            
        Returns
        -------
        Any
            The value associated with the canonical key
            
        Raises
        ------
        KeyError
            If neither the key nor its canonical mapping exists
        """
        # First check if it's an alias and map to canonical key
        canonical_key = self._aliases.get(key, key)
        return super().__getitem__(canonical_key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set item using canonical key (aliases are not stored).
        
        Parameters
        ----------
        key : str
            Key to set (will be converted to canonical if it's an alias)
        value : Any
            Value to store
        """
        # Map to canonical key if it's an alias
        canonical_key = self._aliases.get(key, key)
        super().__setitem__(canonical_key, value)
    
    def __contains__(self, key: str) -> bool:
        """
        Check if key or its alias exists in the dictionary.
        
        Parameters
        ----------
        key : str
            Key or alias to check
            
        Returns
        -------
        bool
            True if the key (or its canonical mapping) exists
        """
        canonical_key = self._aliases.get(key, key)
        return super().__contains__(canonical_key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get item with default value if key doesn't exist.
        
        Parameters
        ----------
        key : str
            Key or alias to access
        default : Any, optional
            Default value if key doesn't exist
            
        Returns
        -------
        Any
            The value or default
        """
        try:
            return self[key]
        except KeyError:
            return default
    
    def keys(self) -> KeysView:
        """Return view of canonical keys only."""
        return super().keys()
    
    def values(self) -> ValuesView:
        """Return view of values."""
        return super().values()
    
    def items(self) -> ItemsView:
        """Return view of (canonical_key, value) pairs."""
        return super().items()
    
    @classmethod
    def get_aliases(cls, canonical_key: str = None) -> Dict[str, str]:
        """
        Get all aliases or aliases for a specific canonical key.
        
        Parameters
        ----------
        canonical_key : str, optional
            If provided, return only aliases that map to this key
            
        Returns
        -------
        Dict[str, str]
            Dictionary of {alias: canonical_key} mappings
            
        Examples
        --------
        >>> FlexibleDict.get_aliases('theta')
        {'X': 'theta', 'params': 'theta', 'parameters': 'theta'}
        
        >>> FlexibleDict.get_aliases()  # Returns all aliases
        """
        if canonical_key is None:
            return cls._aliases.copy()
        else:
            return {alias: canon for alias, canon in cls._aliases.items() 
                    if canon == canonical_key}
    
    @classmethod
    def add_alias(cls, alias: str, canonical_key: str) -> None:
        """
        Add a new alias mapping (affects all FlexibleDict instances).
        
        Parameters
        ----------
        alias : str
            New alias to add
        canonical_key : str
            Canonical key it should map to
            
        Examples
        --------
        >>> FlexibleDict.add_alias('my_params', 'theta')
        >>> results = FlexibleDict({'theta': [1, 2, 3]})
        >>> results['my_params']  # Returns [1, 2, 3]
        """
        cls._aliases[alias] = canonical_key
    
    def __repr__(self) -> str:
        """String representation showing canonical keys only."""
        return f"FlexibleDict({super().__repr__()})"


def create_results_dict(**kwargs) -> FlexibleDict:
    """
    Create a FlexibleDict with common result keys.
    
    This is a convenience function for creating output dictionaries with
    the canonical keys. It automatically converts keys to their canonical
    form if aliases are provided.
    
    Parameters
    ----------
    **kwargs
        Key-value pairs to store. Keys will be converted to canonical form.
        
    Returns
    -------
    FlexibleDict
        Dictionary with flexible key access
        
    Examples
    --------
    >>> results = create_results_dict(
    ...     theta=theta_array,
    ...     eta=eta_array,
    ...     x=observations,
    ...     F=fisher_matrices
    ... )
    >>> # Can access with any notation:
    >>> results['theta']  # or results['X'] or results['params']
    """
    result = FlexibleDict()
    for key, value in kwargs.items():
        result[key] = value
    return result


# =============================================================================
# I/O Functions for Loading and Saving Results
# =============================================================================


def load_fishnets_results(filepath: Union[str, Path], 
                          use_flexible_dict: bool = True) -> Union[FlexibleDict, dict]:
    """
    Load fishnets training results from .npz file.
    
    This function loads results saved by train_fishnets and optionally wraps
    them in a FlexibleDict for convenient access with multiple naming conventions.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .npz file (with or without extension)
    use_flexible_dict : bool, default=True
        If True, return FlexibleDict for flexible key access.
        If False, return standard dict.
        
    Returns
    -------
    results : FlexibleDict or dict
        Dictionary containing:
        - 'theta' : Test parameters (also accessible as 'X', 'params')
        - 'Fs' : Fisher matrix ensemble predictions (also accessible as 'F', 'fisher')
          Shape: (num_models, n_test, n_params, n_params)
        - 'mle' : MLE predictions (also accessible as 'theta_hat')
        - 'x' : Test observations (also accessible as 'data', 'obs')
        - 'ensemble_weights' : Model weights
        
    Examples
    --------
    >>> # Load with flexible access
    >>> results = load_fishnets_results('fishnets-log/fishnets_outputs.npz')
    >>> theta = results['theta']  # Statistical notation
    >>> X = results['X']          # ML notation (same array)
    >>> params = results['params'] # Descriptive (same array)
    >>> Fs = results['Fs']        # Fisher matrices (canonical, plural)
    >>> Fs = results['F']         # Same as above (backwards compatible)
    >>> fisher = results['fisher'] # Same as above (descriptive)
    
    >>> # Load as regular dict
    >>> results = load_fishnets_results('output.npz', use_flexible_dict=False)
    >>> theta = results['theta']  # Only canonical keys work
    >>> Fs = results['Fs']        # Canonical plural form
    """
    # Handle path without extension
    filepath = str(filepath)
    if not filepath.endswith('.npz'):
        filepath += '.npz'
    
    # Load the npz file
    loaded = np.load(filepath)
    
    # Convert to dict
    data_dict = {key: loaded[key] for key in loaded.files}
    
    if use_flexible_dict:
        return FlexibleDict(data_dict)
    else:
        return data_dict


def load_sr_results(filepath: Union[str, Path],
                   use_flexible_dict: bool = True) -> Union[FlexibleDict, dict]:
    """
    Load symbolic regression results from .npz file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .npz file containing SR results
    use_flexible_dict : bool, default=True
        If True, return FlexibleDict for flexible key access
        
    Returns
    -------
    results : FlexibleDict or dict
        Dictionary containing SR results. Common keys include:
        - 'theta' : Original parameters (also 'X', 'params')
        - 'eta' : Learned coordinates (also 'y', 'coords')
        - 'expressions' : SR expressions
        - 'A' : Rotation matrix
        - 'complexity' : Expression complexities
        
    Examples
    --------
    >>> results = load_sr_results('sr_output.npz')
    >>> # Access using preferred notation
    >>> X_train = results['X']      # ML notation
    >>> y_train = results['y']      # ML notation
    >>> # Or use statistical notation
    >>> theta = results['theta']    # Statistical
    >>> eta = results['eta']        # Statistical
    """
    filepath = str(filepath)
    if not filepath.endswith('.npz'):
        filepath += '.npz'
    
    loaded = np.load(filepath, allow_pickle=True)
    data_dict = {key: loaded[key] for key in loaded.files}
    
    if use_flexible_dict:
        return FlexibleDict(data_dict)
    else:
        return data_dict


def load_flattening_results(filepath: Union[str, Path],
                            use_flexible_dict: bool = True) -> Union[FlexibleDict, dict]:
    """
    Load flattening network training results from .npz file.
    
    This function loads results saved by fit_flattening and optionally wraps
    them in a FlexibleDict for convenient access with multiple naming conventions.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .npz file (with or without extension)
    use_flexible_dict : bool, default=True
        If True, return FlexibleDict for flexible key access.
        If False, return standard dict.
        
    Returns
    -------
    results : FlexibleDict or dict
        Dictionary containing:
        - 'theta' : Input parameters (also accessible as 'X', 'params')
        - 'eta' : Learned coordinates (also accessible as 'y', 'coords')
        - 'Jacobians' : Jacobian matrices
        - 'F_ensemble' : Fisher matrices
        - 'eta_ensemble' : Coordinate predictions per ensemble member
        - Additional training metrics and metadata
        
    Examples
    --------
    >>> # Load with flexible access
    >>> results = load_flattening_results('flattened_coords_sr.npz')
    >>> # Access using preferred notation
    >>> X = results['X']          # ML notation (parameters as features)
    >>> y = results['y']          # ML notation (coordinates as targets)
    >>> # Or use statistical notation
    >>> theta = results['theta']  # Same as X
    >>> eta = results['eta']      # Same as y
    >>> 
    >>> # Perfect for symbolic regression
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     results['X'], results['y'], test_size=0.2
    ... )
    """
    filepath = str(filepath)
    if not filepath.endswith('.npz'):
        filepath += '.npz'
    
    loaded = np.load(filepath, allow_pickle=True)
    data_dict = {key: loaded[key] for key in loaded.files}
    
    if use_flexible_dict:
        return FlexibleDict(data_dict)
    else:
        return data_dict


def save_sr_results(filepath: Union[str, Path],
                   theta: np.ndarray,
                   eta: np.ndarray,
                   expressions: Optional[list] = None,
                   A: Optional[np.ndarray] = None,
                   **kwargs) -> None:
    """
    Save symbolic regression results to .npz file with canonical keys.
    
    This function ensures consistent naming conventions across saved files.
    
    Parameters
    ----------
    filepath : str or Path
        Output path for .npz file
    theta : np.ndarray
        Parameter array (will be saved as 'theta')
    eta : np.ndarray
        Coordinate array (will be saved as 'eta')
    expressions : list, optional
        List of SR expression strings
    A : np.ndarray, optional
        Rotation matrix
    **kwargs
        Additional arrays to save
        
    Examples
    --------
    >>> save_sr_results(
    ...     'sr_results.npz',
    ...     theta=theta_array,
    ...     eta=eta_array,
    ...     expressions=['X1 + X2', 'X1 * X2'],
    ...     A=rotation_matrix,
    ...     complexity=[3, 3]
    ... )
    """
    filepath = str(filepath)
    if not filepath.endswith('.npz'):
        filepath += '.npz'
    
    # Build save dict with canonical keys
    save_dict = {
        'theta': theta,
        'eta': eta,
    }
    
    if expressions is not None:
        save_dict['expressions'] = np.array(expressions, dtype=object)
    
    if A is not None:
        save_dict['A'] = A
    
    # Add any additional kwargs
    save_dict.update(kwargs)
    
    np.savez(filepath, **save_dict)
    print(f"Results saved to: {filepath}")
    print(f"Load with: load_sr_results('{filepath}')")


def print_access_examples(results: FlexibleDict, key: str = 'theta') -> None:
    """
    Print examples of different ways to access a key in FlexibleDict.
    
    Useful for documentation and helping users understand the flexible access.
    
    Parameters
    ----------
    results : FlexibleDict
        Results dictionary
    key : str, default='theta'
        Canonical key to show examples for
        
    Examples
    --------
    >>> results = load_fishnets_results('output.npz')
    >>> print_access_examples(results, 'theta')
    """
    if key not in results:
        print(f"Key '{key}' not found in results")
        return
    
    # Get all aliases for this key
    aliases = FlexibleDict.get_aliases(key)
    
    print(f"\n{'='*60}")
    print(f"Multiple ways to access '{key}' data:")
    print(f"{'='*60}")
    print(f"  results['{key}']  # Canonical (Greek notation)")
    
    for alias in aliases:
        if alias == 'X' or alias == 'y':
            print(f"  results['{alias}']  # ML/SR convention")
        else:
            print(f"  results['{alias}']  # Descriptive")
    
    print(f"\nAll return the same array with shape: {results[key].shape}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test FlexibleDict
    print("Testing FlexibleDict...")
    
    # Create test data
    theta_data = np.array([[1, 2], [3, 4]])
    eta_data = np.array([[0.5, 0.6], [0.7, 0.8]])
    
    # Test 1: Basic creation and access
    results = FlexibleDict({
        'theta': theta_data,
        'eta': eta_data
    })
    
    assert np.array_equal(results['theta'], theta_data)
    assert np.array_equal(results['X'], theta_data)  # Alias
    assert np.array_equal(results['params'], theta_data)  # Alias
    print("✓ Basic access with aliases works")
    
    assert np.array_equal(results['eta'], eta_data)
    assert np.array_equal(results['y'], eta_data)  # Alias
    assert np.array_equal(results['coords'], eta_data)  # Alias
    print("✓ Eta/coords access with aliases works")
    
    # Test 2: Contains
    assert 'theta' in results
    assert 'X' in results
    assert 'params' in results
    print("✓ Contains check works with aliases")
    
    # Test 3: Get with default
    assert results.get('nonexistent', 'default') == 'default'
    print("✓ Get with default works")
    
    # Test 4: Create with convenience function
    results2 = create_results_dict(
        theta=theta_data,
        eta=eta_data
    )
    assert np.array_equal(results2['X'], theta_data)
    print("✓ Convenience function works")
    
    # Test 5: Get aliases
    theta_aliases = FlexibleDict.get_aliases('theta')
    assert 'X' in theta_aliases
    assert 'params' in theta_aliases
    print("✓ Get aliases works")
    
    # Test 6: Keys only show canonical keys
    assert list(results.keys()) == ['theta', 'eta']
    print("✓ Keys returns only canonical keys")
    
    print("\n✅ FlexibleDict tests passed!\n")
    
    # Test I/O utilities
    print("Testing I/O utilities...")
    
    # Create a temporary test file
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test_output.npz")
        
        # Create test data
        theta_test = np.random.randn(100, 2)
        eta_test = np.random.randn(100, 2)
        
        # Save with canonical keys
        save_sr_results(
            test_file,
            theta=theta_test,
            eta=eta_test,
            expressions=['X1 + X2', 'X1 * X2'],
            complexity=[3, 3]
        )
        
        # Load with FlexibleDict
        results = load_sr_results(test_file)
        
        # Test flexible access
        assert np.array_equal(results['theta'], results['X'])
        assert np.array_equal(results['theta'], results['params'])
        assert np.array_equal(results['eta'], results['y'])
        assert np.array_equal(results['eta'], results['coords'])
        
        print("✓ Save and load with FlexibleDict works")
        print("✓ All aliases access the same data")
        
        # Test print examples
        print_access_examples(results, 'theta')
        print_access_examples(results, 'eta')
    
    print("\n✅ All tests passed!")
    print("\nExample usage:")
    print(f"  results['theta'] shape: {results['theta'].shape}")
    print(f"  results['X'] shape: {results['X'].shape}  # Same as theta")
    print(f"  results['params'] shape: {results['params'].shape}  # Same as theta")
    print(f"  results['eta'] shape: {results['eta'].shape}")
    print(f"  results['y'] shape: {results['y'].shape}  # Same as eta")
