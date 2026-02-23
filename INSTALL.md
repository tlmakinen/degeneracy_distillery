# Installation Guide

This guide provides instructions for installing the `degeneracy-distillery` package from GitHub.

## Prerequisites

- Git
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- Python 3.10 or higher

## Installation Methods

### Option 0: Automated Setup Script (Easiest)

Use the provided setup script for automated installation:

```bash
git clone https://github.com/yourusername/degeneracy_distillery.git
cd degeneracy_distillery
./setup.sh
```

The script will guide you through the installation process with interactive prompts.

### Option 1: Using Conda Environment (Recommended)

This method recreates the exact conda environment used during development, including all system-level dependencies.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/degeneracy_distillery.git
   cd degeneracy_distillery
   ```

2. **Create the conda environment from the exported environment file:**
   ```bash
   conda env create -f degen_env.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate degen
   ```

4. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```
   
   This will automatically install all dependencies, including ESR from GitHub.

### Option 2: Using pip only (Quick Install)

If you already have a Python environment and want to quickly install just the Python dependencies:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/degeneracy_distillery.git
   cd degeneracy_distillery
   ```

2. **Create a new conda environment (or use an existing one):**
   ```bash
   conda create -n degen python=3.12
   conda activate degen
   ```

3. **Install system dependencies (macOS with conda-forge):**
   ```bash
   conda install -c conda-forge eigen cmake
   ```

4. **Install the package with pip:**
   ```bash
   pip install -e .
   ```

### Option 3: Development Installation

For contributors who want to install with development tools:

1. **Follow steps 1-3 from Option 1 or Option 2**

2. **Install with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

   This includes additional tools like pytest, black, isort, flake8, and mypy.

## Verifying Installation

After installation, verify that the package is correctly installed:

```python
import sys
sys.path.insert(0, 'src')

from training_loop_flatten import *
from preprocessing_utils import *
from sr_utils import *
```

Or if you're using the package as installed:

```python
from src import training_loop_flatten
from src import preprocessing_utils
from src import sr_utils
```

## Special Notes

### PyOperon Dependency

The package requires `pyoperon>=0.4.0`, which is a symbolic regression library. If you encounter installation issues with PyOperon:

- On macOS, ensure you have the necessary build tools installed via conda:
  ```bash
  conda install -c conda-forge cmake eigen pybind11
  ```

- On Linux, you may need to install system dependencies:
  ```bash
  sudo apt-get install cmake libeigen3-dev
  ```

### ESR Dependency

The package requires the `esr` package ([ESR by DeaglanBartlett](https://github.com/DeaglanBartlett/ESR)) for computing symbolic regression complexity metrics (MDL criterion, Aifeyn complexity).

**ESR is automatically installed** when you install `degeneracy-distillery` via pip, as it's listed in the dependencies. The installation process will clone it from GitHub and install it for you.

### JAX with GPU Support

If you want to use JAX with GPU acceleration:

```bash
# For CUDA 12
pip install --upgrade "jax[cuda12]"

# For CUDA 11
pip install --upgrade "jax[cuda11_local]"
```

See the [JAX installation guide](https://github.com/google/jax#installation) for more details.

## Troubleshooting

### Import Errors

If you encounter import errors, make sure you've activated the correct conda environment:
```bash
conda activate degen  # or your environment name
```

### Missing System Dependencies

Some packages (like pyoperon) require system-level libraries. Use conda to install them:
```bash
conda install -c conda-forge eigen cmake pybind11
```

<!-- ### Version Conflicts

If you encounter version conflicts, you can try creating a fresh environment:
```bash
conda deactivate
conda env remove -n degen
conda env create -f degen_env.yml
``` -->

<!-- ## Updating the Package

To update to the latest version from GitHub:

```bash
cd degeneracy_distillery
git pull origin main
pip install -e . --upgrade
```

## Uninstallation

To remove the package:

```bash
pip uninstall degeneracy-distillery
```

To remove the entire conda environment:

```bash
conda deactivate
conda env remove -n degen
``` -->
