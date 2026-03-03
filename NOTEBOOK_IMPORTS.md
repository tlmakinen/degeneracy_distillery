# Notebook Import Guide

This document explains how imports work in the notebooks after the package restructuring.

## Package Renamed: `src/` → `degeneracy_distillery/`

The source code directory has been renamed from `src` to `degeneracy_distillery` to match the package name.

## Notebooks Updated

All notebooks have been updated with the following changes:

### Updated Notebooks
✅ `notebooks/test_ex1.ipynb`
✅ `notebooks/test_ex2.ipynb`  
✅ `notebooks/test_ex3_sbi.ipynb`  
✅ `notebooks/postprocess_test_ex1.ipynb`  
✅ `notebooks/postprocess_test_ex2.ipynb`  
✅ `notebooks/postprocess_test_ex0_hard.ipynb`  
✅ `notebooks/sr_dummy_functions.ipynb`  
✅ `notebooks/postprocessing.ipynb`  
✅ `notebooks/postprocessing_testing.ipynb`

### Changes Made

1. **sys.path updated:**
   - ❌ Old: `sys.path.append('/Users/lucas/repositories/degeneracy_distillery/src/')`
   - ✅ New: `sys.path.insert(0, '../degeneracy_distillery')`

2. **Imports updated:**
   - ❌ Old: `from src.preprocessing_utils import ...`
   - ✅ New: `from preprocessing_utils import ...`
   
   - ❌ Old: `from src.postprocessing_utils import ...`
   - ✅ New: `from postprocessing_utils import ...`
   
   - ❌ Old: `from src.sr_utils import ...`
   - ✅ New: `from sr_utils import ...`

## How Imports Work Now

With `sys.path.insert(0, '../degeneracy_distillery')`, Python adds the package directory to the path, allowing direct imports:

```python
from preprocessing_utils import load_and_process_data
from postprocessing_utils import check_flattening
from sr_utils import fit_and_analyze_sr
from training_loop_flatten import fit_flattening
```

## Alternative: Package Import (if installed)

If you've installed the package with `pip install -e .`, you can also use:

```python
import degeneracy_distillery
from degeneracy_distillery.preprocessing_utils import load_and_process_data
from degeneracy_distillery.postprocessing_utils import check_flattening
```

## ESR Note

The notebooks will warn you if ESR is not installed but will not crash. To install ESR:

```bash
cd /Users/lucas/repositories/degeneracy_distillery
git clone https://github.com/DeaglanBartlett/ESR.git
pip install -e ESR
```

## Testing

To verify your notebooks work correctly:

1. Open any notebook (e.g., `test_ex1.ipynb`)
2. Run the import cell
3. You should see no `ModuleNotFoundError` for `fishnets`, `preprocessing_utils`, etc.
4. You may see a warning about ESR if not installed (this is OK)

All notebooks should now run without import errors!
