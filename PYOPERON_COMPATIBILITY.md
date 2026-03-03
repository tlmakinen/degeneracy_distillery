# PyOperon Version Compatibility

This document explains how the code handles different versions of PyOperon that have incompatible parameter names.

## The Problem

Different versions of PyOperon's `SymbolicRegressor` class use different parameter names for time limits:
- **Newer versions** (e.g., in Google Colab): `max_time`  
- **Older versions**: `time_limit`

This caused import errors when running code across different environments.

## The Solution

The code now automatically detects which parameter name is available using Python's `inspect` module and uses the correct one.

### Implementation

Added to `degeneracy_distillery/sr_utils.py`:

```python
import inspect

def get_time_limit_param_name():
    """Determine the correct parameter name for time limit in PyOperon."""
    sig = inspect.signature(SymbolicRegressor.__init__)
    if 'time_limit' in sig.parameters:
        return 'time_limit'
    elif 'max_time' in sig.parameters:
        return 'max_time'
    else:
        return 'time_limit'  # Default

# Cache the parameter name
_TIME_PARAM_NAME = get_time_limit_param_name()
```

Then when creating the `SymbolicRegressor`:

```python
reg_kwargs = {
    'allowed_symbols': allowed_symbols,
    'max_length': max_length,
    # ... other parameters ...
    _TIME_PARAM_NAME: int(time_limit),  # Uses correct param name
}

reg = SymbolicRegressor(**reg_kwargs)
```

## Benefits

✅ Works with both old and new PyOperon versions  
✅ No manual version checking required  
✅ Automatically adapts to the installed version  
✅ Your API remains consistent (always use `time_limit` parameter)  
✅ Works across Colab, local installations, and different conda environments  

## Usage

The `time_limit` parameter in all wrapper functions like `fit_and_analyze_sr()` remains the same:

```python
mdl_coords, frob_coords, analysis, split_data = fit_and_analyze_sr(
    X, y, y_std, dy_sr, Fs,
    n_params=2,
    time_limit=120,  # Still use this name in your code
    # ...other params...
)
```

The code automatically translates it to the correct parameter name internally!

## Testing

To verify compatibility:

```python
from degeneracy_distillery.sr_utils import _TIME_PARAM_NAME
print(f"Using parameter: {_TIME_PARAM_NAME}")
# Output: "Using parameter: max_time" (in Colab)
# or "Using parameter: time_limit" (in older versions)
```

No more version-specific errors! 🎉
