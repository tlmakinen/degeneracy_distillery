# Google Colab Setup Guide

This guide explains how to install and use `degeneracy-distillery` in Google Colab.

## Quick Installation

```python
# Clone the repository
!git clone https://github.com/yourusername/degeneracy_distillery.git
%cd degeneracy_distillery

# Install the package
!pip install -e .

# Install ESR (required dependency)
!git clone https://github.com/DeaglanBartlett/ESR.git
!pip install -e ESR

# Verify installation
import degeneracy_distillery
print(f"✓ Package version: {degeneracy_distillery.__version__}")
```

## What Works in Colab

The package will install successfully in Colab with all functionality:

✅ Neural network training and flattening
✅ JAX/Flax models
✅ Preprocessing and postprocessing utilities
✅ PyOperon symbolic regression
✅ ESR symbolic regression complexity metrics
✅ All analysis tools

## Dependencies

Most dependencies are automatically installed when you run `!pip install -e .`

However, the **ESR package** ([DeaglanBartlett/ESR](https://github.com/DeaglanBartlett/ESR)) for symbolic regression complexity metrics (MDL criterion, Aifeyn complexity) **must be installed separately**:

```python
!git clone https://github.com/DeaglanBartlett/ESR.git
!pip install -e ESR
```

This is required because pip cannot reliably clone git repositories during dependency resolution.

## Importing Modules

Both import methods work seamlessly!

### Method 1: Package Import (Recommended after installation)

```python
import degeneracy_distillery
from degeneracy_distillery.training_loop_flatten import *
from degeneracy_distillery.preprocessing_utils import *
from degeneracy_distillery.sr_utils import *
```

### Method 2: Direct Import (Alternative)

```python
import sys
sys.path.insert(0, 'degeneracy_distillery')

from training_loop_flatten import *
from preprocessing_utils import *
from sr_utils import *
```

The code automatically uses the correct import method based on your setup.

## Troubleshooting

### Import Errors: "No module named 'degeneracy_distillery'"

If you encounter this error, check:

1. **Did you run `!pip install -e .`?**
   ```python
   %cd degeneracy_distillery
   !pip install -e .
   ```

2. **Check if installation succeeded:**
   ```python
   import degeneracy_distillery
   print(degeneracy_distillery.__file__)  # Should show path to __init__.py
   ```

3. **Alternative: Use sys.path (no installation needed):**
   ```python
   import sys
   sys.path.insert(0, 'degeneracy_distillery')
   from training_loop_flatten import *
   ```

### Other Import Errors

If you encounter other import errors, make sure:
1. You're in the correct directory (`degeneracy_distillery`)
2. The package installed successfully (check for errors during pip install)
3. All dependencies are installed

### GPU Support

To use JAX with GPU in Colab:

```python
# Colab usually has JAX with GPU pre-installed, but if needed:
!pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Memory Issues

If you run out of memory:
- Use smaller batch sizes in training
- Reduce network size
- Use Colab Pro for more RAM

## Example Notebook

See `notebooks/` for example usage. Most notebooks should work in Colab with minimal modifications.
