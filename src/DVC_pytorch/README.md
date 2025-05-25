# PyTorch Implementation of Deep Vine Copula (DVC)

This is a PyTorch port of the TensorFlow-based Deep Vine Copula implementation, designed for GPU-accelerated computation and better scalability.

## Overview

Deep Vine Copulas (DVC) are a flexible framework for modeling complex multivariate dependencies. This implementation provides:

- **Parametric copula families**: Gaussian, Student-t, Clayton, and rotated Clayton copulas
- **Vine structures**: R-vine, C-vine, and D-vine constructions
- **Optimal tree selection**: Using Prim's algorithm for maximum spanning trees
- **GPU acceleration**: Full PyTorch tensor operations for efficient computation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from classes.objects import vine_obj_bin, margin_obj

# Create margin objects
margins = [margin_obj(dist='empirical', theta=None, is_cont=True) for _ in range(d)]

# Create vine copula
vine = vine_obj_bin(
    vine_family='r-vine',
    families=['gaussian', 'clayton', 'student'],
    vine_depth=d-1,
    margin=margins,
    knots=32,
    method='optimal'
)

# Fit to data
gen_dict = {'binning': False, 'parallel': False, 'param': True, 'vine_depth': d-1}
par_dict = {'param_families': ['gaussian', 'clayton', 'student', 'ind']}
vine.fit(data, gen_dict, {}, par_dict, {})
```

## Implementation Status

### ✅ Completed Modules

#### Core Utilities (`utils/`)
- **tensor_op.py**: Tensor operations, boundary checking, NaN/Inf handling
- **interpolation.py**: 2D nearest neighbor and 1D linear interpolation
- **dataset_op.py**: K-fold CV, data splitting, binning
- **bijector.py**: Normal and Gamma CDF bijectors
- **prob_op.py**: Bivariate normal PDF, kernel density estimation, Kendall's tau

#### Preprocessing (`pre_proc/`)
- **transformation.py**: Uniform to normal space transformations with PCA
- **preparation.py**: Data preparation with optional correlation-based sorting

#### Grid Operations (`grid/`)
- **grid_op.py**: Grid creation functions
- **grid_class.py**: Grid object with axis, diff, step methods

#### Evaluation (`evalu/`)
- **cop_eval.py**: Copula PDF normalization and CDF computation
- **vine_eval.py**: Vine evaluation structure (partial)

#### Parametric Components (`param/`)
- **margin_pdf.py**: PDF functions for all copula families
- **margin_cost.py**: Negative log-likelihood cost functions
- **cond_copula.py**: Conditional CDF and inverse conditional CDF
- **copula_fit.py**: Adam optimizer-based fitting for all families

#### Vine Tree Operations (`vine_tree/`)
- **tree_op.py**: Complete tree operations including optimal tree construction

#### Optimization (`optim/`)
- **vine_fit.py**: Parametric fitting with AIC-based family selection

#### Classes (`classes/`)
- **objects.py**: Main copula and vine objects with basic fitting framework

### ⚠️ Partially Implemented

- Non-parametric copula fitting (requires bandwidth selection and local likelihood)
- Full vine evaluation and likelihood computation
- Binning support for conditional copulas

### ❌ Not Yet Implemented

- **optim/**: bandwidth.py, local_lik.py, nadam.py, MISE.py
- **sampling/**: Vine copula sampling functions
- **pred/**: Prediction functions
- **info/**: Information measures (mutual information, entropy)
- **plot/**: Visualization utilities

## Key Differences from TensorFlow Version

1. **Tensor Operations**: Direct indexing instead of `tf.tensor_scatter_nd_update`
2. **Distributions**: `torch.distributions` instead of `tensorflow_probability`
3. **Device Management**: Explicit device placement for GPU computation
4. **Gradient Computation**: Manual finite differences (can be improved with autograd)
5. **Mixed Operations**: Some functions use numpy for scipy.stats compatibility

## Architecture

```
src/DVC_pytorch/
├── classes/          # Main copula and vine objects
├── utils/            # Core utility functions
├── pre_proc/         # Data preprocessing
├── grid/             # Grid operations
├── param/            # Parametric copula functions
├── vine_tree/        # Vine tree construction
├── evalu/            # Evaluation functions
├── optim/            # Optimization routines
├── sampling/         # Sampling functions (placeholder)
├── pred/             # Prediction functions (placeholder)
├── info/             # Information measures (placeholder)
└── plot/             # Plotting utilities (placeholder)
```

## Performance Considerations

1. **GPU Acceleration**: All tensor operations support GPU computation
2. **Batch Processing**: Functions designed for efficient batch operations
3. **Memory Management**: Careful handling of large correlation matrices
4. **Mixed Precision**: Ready for `torch.cuda.amp` integration

## Future Enhancements

1. Complete non-parametric copula implementation
2. Full vine likelihood evaluation
3. Implement sampling from fitted vines
4. Add pure PyTorch alternatives to scipy.stats functions
5. Implement torch.jit.script decorators for performance
6. Add distributed training support

## Example Usage

See `example_usage.py` for a complete workflow demonstrating:
- Data generation and preparation
- Vine copula construction
- Parametric fitting with family selection
- Basic evaluation structure

## Contributing

This implementation is designed to maintain API compatibility with the TensorFlow version while leveraging PyTorch's advantages for GPU computation and automatic differentiation.

## References

The implementation follows the methodology described in the original Deep Vine Copula papers and maintains compatibility with the TensorFlow implementation structure. 