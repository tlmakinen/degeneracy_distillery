# Fisher Matrix Naming Update: F → Fs

## Summary

Successfully updated the canonical key for Fisher matrices from `F` (singular) to `Fs` (plural) to better distinguish between different shapes and improve code clarity.

## What Changed

### Canonical Key
- **Old**: `F` (singular)
- **New**: `Fs` (plural)
- **Backwards Compatibility**: `F` remains as an alias

### Rationale

The plural form `Fs` better represents the data structure:
- **Not** a single Fisher matrix (shape: n_params × n_params)
- **But** multiple Fisher matrices (shape: n_simulations × n_params × n_params)

This makes it easier to distinguish:
```python
ensemble_Fs    # Shape: (num_models, n_simulations, n_params, n_params)
averaged_Fs    # Shape: (n_simulations, n_params, n_params)
single_F       # Shape: (n_params, n_params) - one matrix
```

## Files Updated (6)

### Code Files (2)
1. ✅ **`io_utils.py`**
   - Updated FlexibleDict aliases: `'F': 'Fs'`
   - `Fs` is now canonical
   - `F` maps to `Fs` for backwards compatibility
   - Updated all docstrings

2. ✅ **`training_loop_fishnets.py`**
   - Changed `create_results_dict` to use `Fs` key
   - Added shape documentation
   - Updated example usage code

### Documentation Files (4)
3. ✅ **`NAMING_CONVENTIONS.md`**
   - Updated canonical key to `Fs`
   - Added shape conventions section
   - Updated all code examples

4. ✅ **`QUICK_REFERENCE.md`**
   - Updated table: Fs canonical, F alias
   - Updated code examples

5. ✅ **`COMPREHENSIVE_UPDATE_SUMMARY.md`**
   - Updated all Fisher matrix references
   - Added note about plural canonical form

6. ✅ **`FLEXIBLEDICT_CHEATSHEET.md`**
   - Updated to show Fs (canonical) and F (backwards compatible)

### New Documentation (1)
7. ✅ **`FISHER_NAMING.md`** (NEW)
   - Complete guide to Fs vs F distinction
   - Shape conventions and examples
   - Migration guide
   - Best practices
   - Rationale explanation

## Usage

### All These Work (Same Data)

```python
results = load_fishnets_results('output.npz')

# All equivalent - access the same data:
Fs = results['Fs']              # ✅ Canonical (recommended)
F = results['F']                # ✅ Backwards compatible alias
fisher = results['fisher']      # ✅ Descriptive alias
matrices = results['fisher_matrices']  # ✅ Fully descriptive
```

### Recommended Variable Naming

```python
# ✅ GOOD - Plural indicates multiple matrices
Fs_ensemble = outputs['Fs']
Fs_averaged = np.average(Fs_ensemble, axis=0, weights=weights)
ensemble_Fs = outputs['Fs']

# ❌ AVOID - Singular is misleading
F = outputs['Fs']  # Implies one matrix, but it's many!
```

## Migration

### Do I Need to Update My Code?

**No!** Your existing code continues to work:

```python
# Old code (still works)
F_matrices = results['F']          # ✅ Works via alias

# New code (recommended)
Fs_matrices = results['Fs']        # ✅ More descriptive

# Both access the same data
assert np.array_equal(F_matrices, Fs_matrices)  # True
```

### When Saving New Results

Use the canonical plural form for new code:

```python
# Recommended
outputs = create_results_dict(
    theta=parameters,
    Fs=fisher_matrices,  # Canonical plural ✅
    eta=coordinates
)

# Also works (F is aliased to Fs)
outputs = create_results_dict(
    theta=parameters,
    F=fisher_matrices,   # Alias (less descriptive)
    eta=coordinates
)
```

## Examples

### Training and Accessing Results

```python
from degeneracy_distillery.training_loop_fishnets import train_fishnets

# Train Fisher networks
ws, weights, models, scaler, outputs = train_fishnets(theta, data, ...)

# Access Fisher matrices - all work
Fs = outputs['Fs']         # Canonical ✅
F = outputs['F']           # Alias (backwards compatible) ✅
fisher = outputs['fisher'] # Descriptive ✅

# Check shape
print(Fs.shape)  # e.g., (20, 5000, 2, 2)
# 20 models, 5000 test samples, 2×2 Fisher matrices per sample
```

### Averaging Over Ensemble

```python
import numpy as np

# Get ensemble Fisher matrices
Fs_ensemble = outputs['Fs']  # Shape: (num_models, n_test, n_params, n_params)

# Average over models
Fs_averaged = np.average(
    Fs_ensemble, 
    axis=0,  # Average over model dimension
    weights=outputs['ensemble_weights']
)
# Shape: (n_test, n_params, n_params)

# Use in flattening
w, ens_ws, flatten_out = fit_flattening(
    Fs_averaged,
    outputs['theta'],
    outputs['ensemble_weights']
)
```

## Benefits

1. **Clarity**: Plural `Fs` clearly indicates multiple matrices
2. **Disambiguation**: Easy to distinguish:
   - `ensemble_Fs` (full 4D array)
   - `averaged_Fs` (averaged 3D array)
   - Single Fisher matrix (2D array)
3. **Self-Documenting**: Code with `Fs` is more descriptive
4. **Backwards Compatible**: Old code using `F` still works
5. **Mathematical Consistency**: Matches subscript notation (F₀, F₁, ..., Fₙ)

## Verification

All modified files compile successfully:
```bash
python -m py_compile degeneracy_distillery/training_loop_fishnets.py  ✅
python -m py_compile degeneracy_distillery/io_utils.py                ✅
```

## Summary

| Aspect | Details |
|--------|---------|
| **Canonical Key** | `Fs` (plural) |
| **Backwards Compatibility** | `F` → `Fs` (alias) |
| **Breaking Changes** | None |
| **Migration Required** | No (optional for clarity) |
| **Files Updated** | 6 files |
| **New Documentation** | FISHER_NAMING.md |

---

**Key Takeaway**: Use `Fs` for new code (more descriptive), but `F` still works (backwards compatible). Zero breaking changes!
