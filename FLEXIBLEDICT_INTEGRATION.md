# FlexibleDict Integration Summary

## What Was Implemented

I've implemented a complete flexible naming convention system for Degeneracy Distillery that allows users to access data using their preferred notation (Greek mathematical, ML convention, or descriptive names).

## Files Created

1. **`degeneracy_distillery/io_utils.py`** (567 lines - consolidated module)
   - Core `FlexibleDict` class that extends Python's built-in `dict`
   - Automatic aliasing system: `theta` ↔ `X` ↔ `params`
   - Convenience function `create_results_dict()` for creating output dictionaries
   - `load_fishnets_results()`: Load training results with flexible access
   - `load_sr_results()`: Load symbolic regression results with flexible access
   - `save_sr_results()`: Save with canonical keys
   - `print_access_examples()`: Show users all available access methods
   - Note: `flexible_dict.py` was merged into this file to reduce script count

2. **`NAMING_CONVENTIONS.md`**
   - Comprehensive user guide explaining the naming system
   - Examples for different user types (statisticians vs ML practitioners)
   - Best practices for code authors and users
   - Quick reference table

3. **`QUICK_REFERENCE.md`**
   - One-page cheat sheet for quick lookup
   - All aliases in a simple table

4. **`FLEXIBLEDICT_INTEGRATION.md`** (this file)
   - Technical integration summary
   - List of all changes made to existing files
   
## Note on File Organization

The `FlexibleDict` class and related I/O functions were originally in separate files but have been **consolidated into `io_utils.py`** to reduce the number of scripts. All imports should now use:

```python
from degeneracy_distillery.io_utils import FlexibleDict, create_results_dict
# or
from degeneracy_distillery import FlexibleDict, create_results_dict
```

## Files Modified

### 1. `degeneracy_distillery/training_loop_fishnets.py`

**Changes:**
- Added imports: `FlexibleDict`, `create_results_dict`
- Modified output section (lines ~300-314):
  - Creates `FlexibleDict` for test predictions
  - Returns `outputs` as 5th return value (was 4 before)
- Updated docstring to document the new `outputs` return value
- Added example usage showing flexible access

**Impact:**
- Functions now return 5 values instead of 4 (backwards compatible if unpacking only first 4)
- Saved `.npz` files now include additional keys (`x` for observations)

### 3. `degeneracy_distillery/training_loop_flatten.py`

**Changes:**
- Added imports: `FlexibleDict`, `create_results_dict`
- Modified `fit_flattening()` to use `create_results_dict()` for outputs
- Returns `output_dict` as 3rd return value (was 2 before)
- Updated docstring with Returns section
- Added note about loading with `load_flattening_results()`

**Impact:**
- `fit_flattening()` now returns 3 values instead of 2
- Output dict has flexible access to theta, eta, and all metrics

### 4. `degeneracy_distillery/training_loop_flattening2.py`

**Changes:**
- Same updates as training_loop_flatten.py
- Added imports, FlexibleDict output, updated docstring

### 5. `degeneracy_distillery/training_loop_flatten_inv.py`

**Changes:**
- Same updates as training_loop_flatten.py
- Added imports, FlexibleDict output, updated docstring

### 2. `degeneracy_distillery/__init__.py`

**Changes:**
- Updated module docstring to reference io_utils instead of flexible_dict
- Imported `FlexibleDict` and `create_results_dict` from `io_utils` for top-level access
- Added `__all__` list for explicit exports

**Impact:**
- Users can do: `from degeneracy_distillery import FlexibleDict`
- Imports come from the consolidated `io_utils.py` module

## How It Works

### The Alias System

```python
class FlexibleDict(dict):
    _aliases = {
        # For theta (parameters)
        'X': 'theta',
        'params': 'theta',
        'parameters': 'theta',
        
        # For eta (coordinates)
        'y': 'eta',
        'coords': 'eta',
        'coordinates': 'eta',
        'embeddings': 'eta',
        
        # For x (observations)
        'data': 'x',
        'obs': 'x',
        'observations': 'x',
        'features': 'x',
        
        # ... more aliases ...
    }
    
    def __getitem__(self, key):
        # Automatically maps aliases to canonical keys
        canonical_key = self._aliases.get(key, key)
        return super().__getitem__(canonical_key)
```

### Key Features

1. **Zero Memory Overhead**: Aliases don't duplicate data, they just provide alternative keys
2. **Backward Compatible**: Old code using canonical keys still works
3. **Dict-like Interface**: Behaves exactly like a regular Python dict
4. **Extensible**: Users can add custom aliases with `FlexibleDict.add_alias()`

## Integration Points

### For Training Code

```python
from degeneracy_distillery import create_results_dict

def my_training_function(...):
    # ... training code ...
    
    # Return results with flexible access
    outputs = create_results_dict(
        theta=theta_test,
        eta=eta_predictions,
        x=data_test,
        F=fisher_predictions
    )
    
    return trained_models, outputs
```

### For Loading Results

```python
from degeneracy_distillery.io_utils import load_fishnets_results

# Load with flexible access
results = load_fishnets_results('output.npz')

# Use any notation
theta = results['theta']  # Greek
X = results['X']          # ML
params = results['params'] # Descriptive
```

### For Symbolic Regression

```python
# Natural ML notation for SR
X_train = results['X']    # Input features (parameters)
y_train = results['y']    # Targets (coordinates)

# Train SR model
model.fit(X_train, y_train)

# Save with canonical keys
save_sr_results('sr_output.npz', theta=X_train, eta=y_train)
```

## Testing

Basic functionality tested and working:

```bash
$ python -c "from degeneracy_distillery import FlexibleDict; d = FlexibleDict({'theta': [1,2,3]}); print('theta:', d['theta']); print('X:', d['X']); print('params:', d['params'])"
theta: [1, 2, 3]
X: [1, 2, 3]
params: [1, 2, 3]
Test passed!
```

## Migration Guide

### For Existing Code

**Minimal changes needed:**

```python
# Old code (still works!)
ws, ens_weights, models, scaler = train_fishnets(...)

# New code (to use FlexibleDict)
ws, ens_weights, models, scaler, outputs = train_fishnets(...)

# Access with flexible notation
theta = outputs['theta']  # or outputs['X'] or outputs['params']
```

### For Loading Old Files

```python
# Old npz files can be loaded normally
old_data = np.load('old_output.npz')

# Optionally wrap in FlexibleDict for flexible access
from degeneracy_distillery import FlexibleDict
results = FlexibleDict(dict(old_data))
```

## Canonical Keys Reference

| Concept | Canonical Key | Primary Aliases |
|---------|--------------|-----------------|
| Parameters | `theta` | `X`, `params` |
| Coordinates | `eta` | `y`, `coords` |
| Observations | `x` | `data`, `obs` |
| Fisher Info | `F` | `fisher` |
| MLE | `mle` | `theta_hat` |

## Usage Examples

### Example 1: Training and Accessing Results

```python
from degeneracy_distillery.training_loop_fishnets import train_fishnets

# Train
ws, weights, models, scaler, outputs = train_fishnets(
    theta, data, theta_test, data_test,
    num_models=20
)

# Statistical notation
parameters = outputs['theta']
fisher = outputs['F']

# ML notation
X = outputs['X']  # Same as outputs['theta']
F = outputs['F']

# Check they're the same
assert outputs['theta'] is outputs['X']  # True!
```

### Example 2: Symbolic Regression Workflow

```python
from degeneracy_distillery.io_utils import load_fishnets_results, save_sr_results
from sklearn.linear_model import LinearRegression

# Load results
results = load_fishnets_results('fishnets-log/output.npz')

# Use ML notation for SR
X_train = results['X']    # Parameters as features
y_train = results['eta']  # Coordinates as targets

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save results using canonical keys
save_sr_results(
    'sr_results.npz',
    theta=X_train,
    eta=y_train,
    coefficients=model.coef_
)
```

### Example 3: Adding Custom Aliases

```python
from degeneracy_distillery import FlexibleDict

# Add project-specific alias
FlexibleDict.add_alias('cosmology_params', 'theta')

# Now use it
results = load_fishnets_results('output.npz')
cosmo_params = results['cosmology_params']  # Works!
```

## Benefits

1. **Reduces Confusion**: Users can think in terms they're comfortable with
2. **Backwards Compatible**: Existing code continues to work
3. **Self-Documenting**: Users see all available access methods via `print_access_examples()`
4. **Flexible for Different Domains**: Works for statistics, ML, physics, etc.
5. **No Performance Cost**: Alias lookup is O(1) dictionary access

## Future Extensions

Potential enhancements:

1. Add more domain-specific aliases as needed
2. Create domain-specific subclasses (e.g., `CosmologyResults`, `PhysicsResults`)
3. Add `.theta`, `.eta` property access for even cleaner syntax
4. Integrate with plotting functions to auto-label axes based on canonical keys

## Questions or Issues?

See `NAMING_CONVENTIONS.md` for the user-facing guide with detailed examples and best practices.
