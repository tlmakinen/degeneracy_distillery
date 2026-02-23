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

# Create conda environment
conda env create -f degen_env.yml
conda activate degen

# Install package
pip install -e .
```

## Usage

Import modules from the `src` package:

```python
from src.training_loop_flatten import *
from src.preprocessing_utils import *
from src.sr_utils import *
```

See the `notebooks/` directory for example usage and analysis workflows.

### Google Colab Installation

For Google Colab, use:

```python
# Clone and install
!git clone https://github.com/yourusername/degeneracy_distillery.git
%cd degeneracy_distillery
!pip install -e .
```

**Note:** The package will automatically install the `esr` package ([ESR by DeaglanBartlett](https://github.com/DeaglanBartlett/ESR)) for symbolic regression complexity metrics.

## Project Structure

```
degeneracy_distillery/
├── src/                    # Main source code
│   ├── training_loop_*.py  # Training loops for various architectures
│   ├── preprocessing_utils.py
│   ├── postprocessing_utils.py
│   ├── sr_utils.py         # Symbolic regression utilities
│   └── ...
├── notebooks/              # Jupyter notebooks with examples
├── data/                   # Data files
└── degen_env.yml           # Conda environment specification
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
