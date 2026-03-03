# Google Colab Setup Guide

This guide explains how to install and use `degeneracy-distillery` in Google Colab.

## Installation Steps

### Step 1: Install the Package

Run this in a Colab cell:

```python
# Clone the repository
!git clone https://github.com/yourusername/degeneracy_distillery.git
%cd degeneracy_distillery

# Install the package (this installs all dependencies including pyoperon, jax, flax, etc.)
!pip install -e .

# Install ESR (required dependency - must be installed separately)
!git clone https://github.com/DeaglanBartlett/ESR.git
!pip install -e ESR

print("\n" + "="*60)
print("✓ Installation complete!")
print("✓ Now you can import and use the package")
print("="*60)
```

### Step 2: Verify Installation

Run this in the same or a new cell:

```python
# Verify installation
import degeneracy_distillery
from degeneracy_distillery.training_loop_flatten import fit_flattening
from degeneracy_distillery.sr_utils import fit_and_analyze_sr
print(f"✓ Package version: {degeneracy_distillery.__version__}")
print("✓ All imports successful!")
```

**If you see `ModuleNotFoundError`:** The package may have updated core Colab packages. Simply **restart the runtime** (Runtime → Restart runtime), navigate back (`%cd degeneracy_distillery`), and try the import again.

## What Works in Colab

The package installs successfully in Colab with all functionality:

✅ Neural network training and flattening  
✅ JAX/Flax models  
✅ Preprocessing and postprocessing utilities  
✅ PyOperon symbolic regression  
✅ ESR symbolic regression complexity metrics  
✅ All analysis tools  

**Note:** Jupyter/notebook packages (ipython, ipykernel, notebook, etc.) are now optional dependencies and won't be installed by default, avoiding conflicts with Colab's environment. Colab already has these packages pre-installed.

## Dependencies

**All core dependencies** (including `pyoperon`, `jax`, `flax`, etc.) are automatically installed when you run `!pip install -e .`

However, the **ESR package** ([DeaglanBartlett/ESR](https://github.com/DeaglanBartlett/ESR)) for symbolic regression complexity metrics (MDL criterion, Aifeyn complexity) **must be installed separately**:

```python
!git clone https://github.com/DeaglanBartlett/ESR.git
!pip install -e ESR
```

This is required because pip cannot reliably clone git repositories during dependency resolution.

### Required Packages
- **pyoperon** - Symbolic regression (auto-installed with `pip install -e .`)
- **ESR** - Complexity metrics (must install separately as shown above)
- **JAX/Flax** - Neural networks (auto-installed with `pip install -e .`)
- **NumPy, SciPy, scikit-learn** - Scientific computing (auto-installed)

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

If you encounter this error:

1. **Did you restart the runtime after installation?**
   - Go to: **Runtime → Restart runtime**
   - Then navigate back: `%cd degeneracy_distillery`
   - Try the import again

2. **Make sure you ran `!pip install -e .` in the correct directory:**
   ```python
   %cd degeneracy_distillery
   !pip install -e .
   ```

3. **Check if installation succeeded:**
   ```python
   import degeneracy_distillery
   print(degeneracy_distillery.__file__)  # Should show path to __init__.py
   ```

4. **Alternative: Use sys.path (no installation needed):**
   ```python
   import sys
   sys.path.insert(0, 'degeneracy_distillery')
   from training_loop_flatten import *
   ```

### Import Errors: "No module named 'pyoperon'" or "No module named 'jax'"

This means the `pip install -e .` command didn't complete successfully or some dependencies failed to install.

**Diagnosis:** Check the pip install output carefully for error messages. Look for lines like:
- `ERROR: Failed building wheel for pyoperon`
- `Successfully installed ...` (should list all packages)

**Solution 1 - Install dependencies manually first:**

```python
# Install core dependencies explicitly
!pip install pyoperon jax jaxlib flax optax orbax-checkpoint numpy scipy matplotlib pandas scikit-learn

# Then install the package
!pip install -e .
```

**Solution 2 - If pyoperon specifically fails:**

PyOperon is a C++ library with Python bindings that sometimes has compilation issues. Try:

```python
# Upgrade build tools
!pip install --upgrade pip setuptools wheel

# Install pyoperon
!pip install pyoperon

# Verify it works
import pyoperon
print("✓ PyOperon installed successfully")

# Then install the main package
!pip install -e .
```

**Solution 3 - Fresh restart:**

If all else fails, restart the Colab runtime and run the installation from scratch:
- Runtime → Restart runtime
- Run the Quick Installation commands again

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
