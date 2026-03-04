# Degeneracy Distillery

A research package for analyzing degeneracy in neural networks and performing symbolic regression analysis using network flattening techniques.

## Features

- Neural network flattening and degeneracy analysis
- Symbolic regression integration with PyOperon
- JAX/Flax-based neural network training
- Preprocessing and postprocessing utilities for network analysis
- Support for various network architectures (FishNets, inverted architectures)

## Installation

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

**Quick start:**

```bash
# Clone the repository
git clone https://github.com/yourusername/degeneracy_distillery.git
cd degeneracy_distillery

# Create conda environment (use minimal for better compatibility)
conda env create -f degen_env_minimal.yml
conda activate degen

# Install package
pip install -e .

# Install ESR (REQUIRED - must be done separately)
git clone https://github.com/DeaglanBartlett/ESR.git
pip install -e ESR

# Optional: Install Jupyter for local notebook development
pip install -e ".[jupyter]"
```

## Usage

### Option 1: Import as a Package (Recommended for Colab/External Use)

```python
import degeneracy_distillery
from degeneracy_distillery.training_loop_flatten import *
from degeneracy_distillery.preprocessing_utils import *
from degeneracy_distillery.sr_utils import *
```

### Option 2: Direct Import (For Working in Repository)

When working directly in the repository (e.g., from notebooks/):

```python
import sys
sys.path.insert(0, '../degeneracy_distillery')  # from notebooks/
# or sys.path.insert(0, 'degeneracy_distillery') if at repo root
from training_loop_flatten import *
from preprocessing_utils import *
from sr_utils import *
```

**Both methods work seamlessly!** The code automatically detects which import method to use.

See the `notebooks/` directory for example usage and analysis workflows.

### Google Colab Installation

For Google Colab, use:

```python
# Clone and install
!git clone https://github.com/yourusername/degeneracy_distillery.git
%cd degeneracy_distillery

# Install the package (this installs all dependencies including pyoperon, jax, etc.)
!pip install -e .

# Install ESR (required - must be installed separately)
!git clone https://github.com/DeaglanBartlett/ESR.git
!pip install -e ESR

# Verify installation
import degeneracy_distillery
from degeneracy_distillery.training_loop_flatten import fit_flattening
from degeneracy_distillery.sr_utils import fit_and_analyze_sr
print(f"✓ Package version: {degeneracy_distillery.__version__}")
print("✓ All imports successful!")
```

See `COLAB_SETUP.md` for detailed troubleshooting and step-by-step instructions.

## Project Structure

```
degeneracy_distillery/
├── degeneracy_distillery/  # Main source code package
│   ├── training_loop_*.py  # Training loops for various architectures
│   ├── preprocessing_utils.py
│   ├── postprocessing_utils.py
│   ├── sr_utils.py         # Symbolic regression utilities
│   └── ...
├── notebooks/              # Jupyter notebooks with examples
├── data/                   # Data files
├── degen_env_minimal.yml   # Conda environment (recommended)
└── degen_env.yml           # Full environment export (may have conflicts)
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
