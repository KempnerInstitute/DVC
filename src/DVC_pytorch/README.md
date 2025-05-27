# DVC PyTorch Implementation

A pure PyTorch implementation of Deep Vine Copulas (DVC) for improved GPU optimization and scalability.

## Current Status

### ✅ Completed Components

1. **Core Classes & Objects** (`classes/`)
   - `vine_obj_bin`: Main vine copula object with fitting and structure
   - `margin_obj`: Marginal distribution objects
   - `grid_obj`: Grid objects for numerical integration

2. **Parametric Copulas** (`param/`)
   - Gaussian, Student-t, Clayton, and rotated Clayton copulas
   - Conditional CDF (h-functions) and inverse conditional CDF
   - PyTorch-compatible versions: `copulaccdf_torch`, `copulainvccdf_torch`

3. **Vine Tree Operations** (`vine_tree/`)
   - C-vine and D-vine structure generation
   - Parent-child relationships and edge definitions
   - R-matrix generation for vine structure

4. **Probability Operations** (`utils/prob_op.py`)
   - `kernel_cdf_torch`: Empirical CDF computation
   - `kernel_pdf2`: Kernel density estimation
   - Numerical integration utilities

5. **Preprocessing** (`pre_proc/`)
   - Data transformation utilities
   - Bijector operations for copula transformations

6. **Fitting Procedures** (`optim/`)
   - Maximum likelihood estimation for copula parameters
   - Tree-by-tree fitting algorithm
   - Marginal distribution fitting

7. **Sampling** (`sampling/`)
   - `VineSampler`: Main sampling class following TensorFlow logic
   - Parametric vine sampling with proper h-function computation
   - Marginal transformations for converting uniform to original scale

### 🔧 Recent Fixes

1. **Sampling Correlation Preservation**
   - Fixed uniform sample generation to properly preserve correlations
   - Corrected h-function computation in higher trees
   - Proper extraction of samples using R-matrix ordering

2. **Marginal Transformations**
   - Fixed interpretation of `Mar_G` (grid values and CDF values)
   - Proper inverse CDF interpolation for marginal distributions

3. **Import Structure**
   - Removed all dependencies on `DVC_pyolder`
   - Clean module structure with proper `__init__.py` files

### 📊 Test Results

The implementation now successfully:
- Fits vine copulas to correlated data
- Preserves correlation structure in sampling (with small differences ~0.01-0.15)
- Generates uniform samples with correct means (~0.5)
- Handles 3D+ dimensional data

### 🚧 Known Issues

1. **Sampling Accuracy**: While correlations are preserved, there's still some discrepancy compared to original data
2. **Non-parametric Copulas**: Implementation exists but needs more testing
3. **GPU Optimization**: Currently uses CPU for compatibility; GPU support needs testing

## Usage Example

```python
import torch
import numpy as np
from classes.objects import vine_obj_bin, margin_obj
from sampling import VineSampler

# Generate correlated data
n_samples = 1000
d = 3
mean = np.zeros(d)
cov = np.array([[1.0, 0.8, 0.6],
                [0.8, 1.0, 0.7],
                [0.6, 0.7, 1.0]])
data = np.random.multivariate_normal(mean, cov, n_samples)
data_torch = torch.tensor(data, dtype=torch.float32)

# Create and fit vine
margins = [margin_obj('norm', [0, 1], True) for _ in range(d)]
vine = vine_obj_bin(
    vine_family='c-vine',
    families=['gaussian'],
    vine_depth=d-1,
    margin=margins,
    knots=50,
    method='matrix'
)

# Fitting options
gen_dict = {"parallel": False, "param": True, "binning": False, "fitted": False, "vine_depth": d-1}
par_dict = {"param_families": ["gaussian"]}
npc_dict = {"opt_method": "LL1", "batch_paral": False}
bin_dict = {"n_bin": 1}

# Fit the vine
vine.fit(data_torch, gen_dict, npc_dict, par_dict, bin_dict)

# Sample from the fitted vine
sampler = VineSampler(vine)
samples, u_samples = sampler.sample(1000)
```

## Directory Structure

```
DVC_pytorch/
├── classes/          # Core vine copula and margin objects
├── param/           # Parametric copula families and fitting
├── vine_tree/       # Vine structure and tree operations
├── utils/           # Probability operations and utilities
├── pre_proc/        # Data preprocessing and transformations
├── optim/           # Optimization and fitting procedures
├── sampling/        # Vine copula sampling
├── evalu/           # Evaluation metrics (in progress)
├── info/            # Information measures (in progress)
├── plot/            # Plotting utilities (in progress)
└── experiments/     # Experiment scripts (in progress)
```

## Requirements

- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- Matplotlib >= 3.3.0

## Future Work

1. Complete GPU optimization for all operations
2. Implement more copula families (Frank, Gumbel, etc.)
3. Add vine structure selection algorithms
4. Improve non-parametric copula implementation
5. Add comprehensive unit tests 