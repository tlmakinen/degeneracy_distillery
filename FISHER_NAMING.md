# Fisher Matrix Naming Convention - Fs vs F

## Summary

**Canonical key is now `Fs` (plural)** to represent that it's multiple Fisher matrices (one per simulation/sample).

## Key Distinction

### `Fs` - Fisher Matrices (Canonical)
The **canonical** (preferred) key for storing Fisher information matrices.

**Why plural?** Because it's typically an array of multiple Fisher matrices:
- Shape when saved from fishnets: `(num_models, n_simulations, n_params, n_params)`
- Shape after averaging: `(n_simulations, n_params, n_params)`

Each simulation/sample has its own Fisher information matrix, so plural `Fs` makes this clear.

### `F` - Backwards Compatible Alias
Single `F` is maintained as an **alias** pointing to `Fs` for backwards compatibility.

**Usage:**
```python
results['Fs']      # Canonical (recommended)
results['F']       # Alias (backwards compatible)
results['fisher']  # Descriptive alias
```

All three access the **same data** with zero overhead.

---

## Shape Conventions

### Ensemble Fisher Matrices
```python
# From train_fishnets() - full ensemble
outputs['Fs']  # Shape: (num_models, n_test, n_params, n_params)

# Each model in the ensemble predicts a Fisher matrix for each test point
ensemble_Fs = outputs['Fs']
model_0_predictions = ensemble_Fs[0]  # Shape: (n_test, n_params, n_params)
```

### Averaged Fisher Matrices
```python
# After averaging over ensemble (e.g., in fit_flattening)
import numpy as np

averaged_Fs = np.average(
    outputs['Fs'],                   # (num_models, n_test, n_params, n_params)
    axis=0,                          # Average over models
    weights=outputs['ensemble_weights']
)
# Shape: (n_test, n_params, n_params)
```

### Variable Naming Recommendations

When working with Fisher matrices in your code:

```python
# RECOMMENDED - Use plural to indicate multiple matrices
Fs_ensemble = outputs['Fs']           # Full ensemble
Fs_averaged = average_over_models(Fs_ensemble)  # Averaged

# DESCRIPTIVE - Even more explicit
ensemble_Fs = outputs['Fs']
averaged_Fs = average_over_models(ensemble_Fs)

# AVOID - Singular suggests a single matrix
F = outputs['Fs']  # Misleading - it's actually multiple matrices!
```

---

## Migration Guide

### If you have existing code using `F`

**Good news:** Your code still works! `F` is an alias for `Fs`.

```python
# Old code (still works)
F_matrices = results['F']

# New code (recommended - more descriptive)
Fs_matrices = results['Fs']

# Both access the same data
assert np.array_equal(F_matrices, Fs_matrices)  # True
```

### When saving new results

Use the canonical plural form:

```python
# RECOMMENDED
output_dict = create_results_dict(
    theta=parameters,
    Fs=fisher_matrices,  # Canonical plural
    eta=coordinates
)

# ALSO WORKS (but less descriptive)
output_dict = create_results_dict(
    theta=parameters,
    F=fisher_matrices,   # Alias - will be stored as 'Fs'
    eta=coordinates
)
```

---

## Examples

### Fisher Networks Training

```python
from degeneracy_distillery.training_loop_fishnets import train_fishnets
from degeneracy_distillery.io_utils import load_fishnets_results

# Train
ws, weights, models, scaler, outputs = train_fishnets(theta, data, ...)

# Access Fisher matrices - all equivalent
Fs_ensemble = outputs['Fs']        # Canonical (recommended)
F_ensemble = outputs['F']          # Alias (backwards compatible)
fisher = outputs['fisher']         # Descriptive alias

print(Fs_ensemble.shape)  
# Output: (20, 5000, 2, 2)
# Interpretation: 20 models, 5000 test samples, 2×2 Fisher matrices

# Load saved results
results = load_fishnets_results('fishnets-log/fishnets_outputs.npz')
Fs = results['Fs']  # Canonical key
```

### Flattening Network Training

```python
from degeneracy_distillery.training_loop_flatten import fit_flattening
import numpy as np

# Average Fisher matrices over ensemble
Fs_ensemble = fishnets_out['Fs']  # Shape: (num_models, n_test, n_params, n_params)
weights = fishnets_out['ensemble_weights']

Fs_averaged = np.average(Fs_ensemble, axis=0, weights=weights)
# Shape: (n_test, n_params, n_params)

# Train flattening network
w, ensemble_ws, flatten_out = fit_flattening(
    Fs_averaged,                    # Pass averaged Fs
    fishnets_out['theta'],
    weights
)
```

### In Your Own Analysis

```python
# Load results
results = load_fishnets_results('output.npz')

# GOOD - Clear variable names
Fs_full = results['Fs']                        # (models, samples, params, params)
Fs_model_0 = Fs_full[0]                       # (samples, params, params)
Fs_sample_0_model_0 = Fs_full[0, 0]          # (params, params) - single matrix

# BETTER - Use descriptive variable names based on context
ensemble_Fs = results['Fs']
model_predictions = ensemble_Fs[model_idx]
single_fisher = model_predictions[sample_idx]
```

---

## Rationale

### Why change from `F` to `Fs`?

1. **Clarity**: Plural form (`Fs`) makes it clear you're dealing with multiple Fisher matrices
2. **Consistency**: Matches the mathematical convention of using subscripts (F₀, F₁, ..., Fₙ)
3. **Disambiguation**: Clear distinction between:
   - `Fs` (or `ensemble_Fs`): Full ensemble shape (models, samples, params, params)
   - `Fs` (or `averaged_Fs`): Averaged shape (samples, params, params)
   - A single Fisher matrix: shape (params, params)

4. **Self-documenting**: Code like `Fs_averaged` is more self-explanatory than `F_averaged`

### Backwards Compatibility

We maintain `F` as an alias so existing code doesn't break:
- `results['F']` automatically maps to `results['Fs']`
- No data duplication - just aliasing
- Gradual migration path for users

---

## Best Practices Summary

✅ **DO:**
- Use `Fs` when saving/storing Fisher matrices (canonical)
- Use plural variable names: `Fs_ensemble`, `Fs_averaged`, `model_Fs`
- Use `F` as an alias for backwards compatibility in existing code

❌ **DON'T:**
- Use singular `F` for new variable names when storing multiple matrices
- Mix singular/plural naming inconsistently in the same codebase

🔄 **MIGRATION:**
- Existing code using `F` continues to work (it's an alias)
- Update new code to use `Fs` for clarity
- No rush to update old code - both work!

---

## Technical Details

### In FlexibleDict

```python
class FlexibleDict(dict):
    _aliases = {
        'F': 'Fs',              # Singular -> Plural mapping
        'fisher': 'Fs',         # Descriptive -> Canonical
        'fisher_matrices': 'Fs', # Fully descriptive -> Canonical
        # ... other aliases ...
    }
```

When you access:
- `results['Fs']` → returns the data stored under canonical key 'Fs'
- `results['F']` → aliased to 'Fs', returns the same data
- `results['fisher']` → aliased to 'Fs', returns the same data

All three are **views** of the same data, no copying!

---

## Questions?

**Q: Do I need to update my old code?**  
A: No! `F` still works as an alias. Update at your convenience.

**Q: What should I use in new code?**  
A: Use `Fs` (plural) as the canonical key. It's more descriptive.

**Q: Will old `.npz` files still load?**  
A: Yes! If they have `F` as a key, `FlexibleDict(np.load('old.npz'))` will make it accessible via both `F` and `Fs`.

**Q: What about a single Fisher matrix?**  
A: Use a descriptive variable name like `fisher_matrix` or `F_single`. The key `Fs` is for collections of matrices.

---

**Remember: `Fs` is plural because you typically have multiple Fisher matrices, one per simulation/sample!**
