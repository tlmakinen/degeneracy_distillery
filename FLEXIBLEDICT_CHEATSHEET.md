# FlexibleDict Quick Cheat Sheet

## One-Line Summary
**All training functions now return FlexibleDict with flexible naming: theta↔X↔params, eta↔y↔coords**

---

## Import
```python
from degeneracy_distillery.io_utils import (
    FlexibleDict, create_results_dict,
    load_fishnets_results, load_flattening_results
)
```

---

## Training Functions - NEW Return Signatures

### Fisher Networks
```python
ws, weights, models, scaler, outputs = train_fishnets(...)
#                               ^^^^^ NEW (FlexibleDict)
```

### Flattening Networks (all 3 versions)
```python
w, ensemble_ws, output_dict = fit_flattening(...)
#               ^^^^^^^^^^^ NEW (FlexibleDict)
```

---

## Access Patterns (ALL equivalent)

```python
# Parameters
outputs['theta']      # Statistical notation
outputs['X']          # ML/SR notation ← USE THIS FOR SYMBOLIC REGRESSION
outputs['params']     # Descriptive

# Coordinates
outputs['eta']        # Statistical notation
outputs['y']          # ML/SR notation ← USE THIS FOR SYMBOLIC REGRESSION
outputs['coords']     # Descriptive

# Observations
outputs['x']          # Canonical
outputs['data']       # Common
outputs['obs']        # Short

# Fisher matrices
outputs['Fs']         # Canonical (plural)
outputs['F']          # Backwards compatible (singular)
outputs['fisher']     # Descriptive
```

---

## Complete Pipeline Example

```python
# 1. Train Fisher networks
ws, weights, models, scaler, fish_out = train_fishnets(theta, data, ...)

# 2. Train flattening
w, ens_ws, flat_out = fit_flattening(
    fish_out['Fs'],             # Fisher matrices (canonical plural)
                                # Same as fish_out['F'] or ['fisher']
    fish_out['theta'],          # Same as fish_out['X'] or ['params']
    fish_out['ensemble_weights']
)

# 3. Symbolic Regression (natural ML notation!)
X_train = flat_out['X']    # Input features = parameters
y_train = flat_out['y']    # Targets = coordinates

from pysr import PySRRegressor
model = PySRRegressor()
model.fit(X_train, y_train)
```

---

## Loading Saved Results

```python
# Load with flexible access
fish_results = load_fishnets_results('fishnets_outputs.npz')
flat_results = load_flattening_results('flattened_coords.npz')

# All notation works
theta = flat_results['theta']   # Statistical
X = flat_results['X']           # ML (same data!)
params = flat_results['params'] # Descriptive (same data!)
```

---

## Migration Guide

### Old Code (still works)
```python
ws, weights, models, scaler = train_fishnets(...)        # Works
w, ensemble_ws = fit_flattening(...)                     # Works
```

### New Code (gets FlexibleDict)
```python
ws, weights, models, scaler, outputs = train_fishnets(...)     # NEW
w, ensemble_ws, output_dict = fit_flattening(...)              # NEW
```

---

## Key Benefits
- ✅ Consistent naming across entire pipeline
- ✅ Works with any notation preference
- ✅ Perfect for symbolic regression (X→y)
- ✅ No memory overhead (aliases = references)
- ✅ Backwards compatible

---

## Files Modified
1. `training_loop_fishnets.py` - Fisher networks
2. `training_loop_flatten.py` - Main flattening
3. `training_loop_flattening2.py` - Alt flattening
4. `training_loop_flatten_inv.py` - Invertible flattening
5. `io_utils.py` - Added load_flattening_results()

---

## Documentation
- `COMPREHENSIVE_UPDATE_SUMMARY.md` - Full technical details
- `NAMING_CONVENTIONS.md` - User guide with examples
- `QUICK_REFERENCE.md` - Quick lookup
- `FLEXIBLEDICT_INTEGRATION.md` - Implementation details

---

**Remember: θ, X, and params all access the SAME data. Choose your preferred notation!**
