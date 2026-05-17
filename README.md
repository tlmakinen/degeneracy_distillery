# Degeneracy Distillery

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/degeneracy_distillery/blob/main/notebooks/example.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research package for analyzing degeneracy in neural networks and performing symbolic regression analysis using network flattening techniques.


### Distill in three steps from simulations or labelled data

**Gist:** simulate $(\theta, x)$, learn local Fisher geometry with a network, flatten it, then fit **closed-form** expressions for identifiable vs. degenerate directions.

```python
# Degeneracy Distillery — conceptual API (each block ≈ one main library call)

# Step 1 — Fisher networks (ensemble)
theta, x = sample_prior_and_simulate(nsims)          # your simulator
W = train_fishnets(theta, x)                         # θ̂ and F(θ|x) per datum

# Step 2 — flatten Fisher geometry (normalizing flow / whitening)
F_ens = predicted_fisher_ensemble(W, x_test)         # shape: (n_models, n, p, p)
eta, J = fit_flattening(F_ens, theta_test, ensemble_weights)  # η, J = ∂η/∂θ

# Step 3 — symbolic regression (+ optional postprocess)
X, y, F_sr = preprocess_for_sr(eta, J, F_ens)      # align / subsample / rotate
exprs = fit_symbolic_regression(X, y, F_sr)          # e.g. PyOperon + MDL
exprs = postprocess(exprs)                           # prune / rotate for sparsity

# exprs → interpretable formulas, e.g. η₂ = θ₂ + θ₁², η₁ = β/γ, …
```



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
git clone https://github.com/tlmakinen/degeneracy_distillery.git
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

For Google Colab:

```python
# 1. Install degeneracy_distillery
!git clone https://github.com/yourusername/degeneracy_distillery.git
%cd /content/degeneracy_distillery
!pip install -e .

# 2. Install ESR (REQUIRED!)
%cd /content
!git clone https://github.com/DeaglanBartlett/ESR.git
%cd /content/ESR
!pip install -e .
```

**Then restart the runtime:** Runtime → Restart runtime

After restarting, verify:
```python
%cd /content/degeneracy_distillery
import degeneracy_distillery
from degeneracy_distillery.sr_utils import fit_and_analyze_sr
print(f"✓ Package version: {degeneracy_distillery.__version__}")
```

See `COLAB_SETUP.md` for detailed step-by-step instructions.

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

