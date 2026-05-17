"""
Degeneracy Distillery - Neural network degeneracy analysis and symbolic regression

Main modules:
    - training_loop_fishnets: Fisher network training
    - training_loop_flatten: Flattening network training
    - preprocessing_utils: Data preprocessing and rotation utilities
    - postprocessing_utils: Symbolic regression postprocessing
    - sr_utils: Symbolic regression utilities
    - fishnets: Neural network architectures
    - plot_utils: Plotting utilities for Fisher matrices and visualizations
    - io_utils: I/O utilities with FlexibleDict for flexible naming conventions
    - diagnostics: Fisher-information diagnostics for low-information regions
    - visuals: Snapshot-based training visualisations and gif builders
      (Fisher det F + flattener eta over training; see
      :mod:`degeneracy_distillery.visuals` for the public API)
"""

__version__ = "0.1.0"

# Import key utilities for convenient access
from .io_utils import FlexibleDict, create_results_dict
from .diagnostics import diagnose_low_information

__all__ = ['FlexibleDict', 'create_results_dict', 'diagnose_low_information']

# Note: Modules are available for direct import:
# from degeneracy_distillery.training_loop_flatten import fit_flattening
# from degeneracy_distillery.preprocessing_utils import load_and_process_data
# from degeneracy_distillery import FlexibleDict
# etc.
