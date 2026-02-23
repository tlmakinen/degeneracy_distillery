# Google Colab Setup Guide

This guide explains how to install and use `degeneracy-distillery` in Google Colab.

## Quick Installation

```python
# Clone the repository
!git clone https://github.com/yourusername/degeneracy_distillery.git
%cd degeneracy_distillery

# Install the package
!pip install -e .
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

All dependencies, including the ESR package for symbolic regression complexity metrics (MDL criterion, Aifeyn complexity), are automatically installed when you run `!pip install -e .`

The ESR package is automatically cloned from [DeaglanBartlett/ESR](https://github.com/DeaglanBartlett/ESR) and installed as part of the setup process.

## Importing Modules

Both import methods work seamlessly!

### Method 1: Package Import (Recommended)

```python
from degeneracy_distillery.src.training_loop_flatten import *
from degeneracy_distillery.src.preprocessing_utils import *
from degeneracy_distillery.src.sr_utils import *
```

### Method 2: Direct Import

```python
import sys
sys.path.insert(0, 'src')

from training_loop_flatten import *
from preprocessing_utils import *
from sr_utils import *
```

The code automatically uses the correct import method based on your setup.

## Troubleshooting

### Import Errors

If you encounter import errors, make sure:
1. You're in the correct directory (`degeneracy_distillery`)
2. The package installed successfully
3. You've added the `src` path to `sys.path`

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
