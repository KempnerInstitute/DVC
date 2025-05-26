# Deep Vine Copula (DVC) - PyTorch Implementation

A GPU-accelerated PyTorch implementation of Deep Vine Copula for multivariate dependence modeling, converted from the original TensorFlow implementation.

## Features

- ✅ **GPU Acceleration**: Native PyTorch GPU support with automatic device management
- ✅ **All Vine Structures**: C-vine, D-vine, and R-vine with optimal tree construction
- ✅ **Parametric Copulas**: Gaussian, Student-t, Clayton, and rotated variants
- ✅ **Non-parametric Copulas**: With bandwidth optimization
- ✅ **Numerical Stability**: Enhanced stability fixes for log-likelihood calculations
- ✅ **Performance**: 1.37x average speedup over TensorFlow implementation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd DVC/src/DVC_pytorch

# Install dependencies
pip install torch numpy scipy matplotlib
```

## Quick Start

```python
import torch
from classes.objects import vine_obj_bin, margin_obj
from grid.grid_op import create_grids

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate or load data (should be in uniform [0,1] margins)
data = torch.rand(1000, 5, device=device)

# Create margins
margins = [margin_obj('empirical', None, True) for _ in range(5)]

# Create vine copula
vine = vine_obj_bin(
    vine_family='r-vine',    # Options: 'c-vine', 'd-vine', 'r-vine'
    families=['gaussian'],    # Copula families to consider
    vine_depth=4,            # Depth of vine (max: n_dims - 1)
    margin=margins,
    knots=32,
    method='optimal'         # For r-vine: 'optimal' or 'random'
)

# Create grids
vine.grid_u, vine.grid_s, vine.grid_x = create_grids(vine.knots, device=device)

# Set parameters
gen_dict = {
    'binning': False,
    'parallel': False,
    'param': True,           # True for parametric, False for non-parametric
    'vine_depth': 4
}

par_dict = {
    'param_families': ['gaussian', 'clayton', 'student', 'ind']
}

bin_dict = {'n_bin': 1}     # For binning support

# Fit the vine copula
vine.fit(data, gen_dict, {}, par_dict, bin_dict)

# Evaluate on new data
test_data = torch.rand(100, 5, device=device)
p, p_cop, log_p = vine.evaluation(test_data)

print(f"Mean log-likelihood: {log_p.mean().item():.3f}")
```

## Performance Comparison

| Test Case            | PyTorch (GPU) | TensorFlow (CPU) | Speedup |
|---------------------|---------------|------------------|---------|
| C-vine (3D Gaussian)| 13.1s         | 18.2s            | 1.39x   |
| D-vine (4D Clayton) | 26.3s         | 35.6s            | 1.35x   |
| R-vine (5D Mixed)   | 36.0s         | 48.8s            | 1.36x   |

## Project Structure

```
DVC_pytorch/
├── classes/          # Core vine copula classes
│   └── objects.py    # Main vine_obj_bin class
├── param/            # Parametric copula functions
│   ├── margin_pdf.py # PDF functions
│   ├── margin_cost.py # Cost functions
│   ├── cond_copula.py # Conditional CDFs
│   └── copula_fit.py  # Fitting routines
├── utils/            # Utility functions
│   ├── tensor_op.py   # Tensor operations
│   ├── prob_op.py     # Probability operations
│   ├── interpolation.py # Interpolation functions
│   └── bijector.py    # Bijector transformations
├── vine_tree/        # Vine tree operations
│   └── tree_op.py     # Tree construction and manipulation
├── grid/             # Grid operations
│   ├── grid_op.py     # Grid creation
│   └── grid_class.py  # Grid class definitions
├── optim/            # Optimization routines
│   ├── bandwidth.py   # Bandwidth selection
│   ├── vine_fit.py    # Vine fitting optimization
│   └── nadam.py       # Nadam optimizer
├── evalu/            # Evaluation functions
│   └── vine_eval.py   # Vine evaluation routines
├── sampling/         # Sampling functions
│   └── vine_sample.py # Vine sampling
├── pred/             # Prediction functions
│   └── prediction.py  # Conditional prediction
├── info/             # Information measures
│   └── info_estimation.py # Entropy estimation
└── plot/             # Plotting functions
    └── plot_vine.py   # Visualization tools
```

## Key Improvements Over TensorFlow

1. **GPU Acceleration**: Native CUDA support for all operations
2. **Performance**: 1.37x average speedup 
3. **Memory Efficiency**: Better memory management with PyTorch
4. **Modern Ecosystem**: Integration with PyTorch ecosystem
5. **Automatic Differentiation**: Native autograd support

## Known Limitations

1. **Marginal Density Estimation**: KDE implementation needs improvement
2. **Entropy Calculation**: Some numerical stability issues remain
3. **Bandwidth Selection**: Sometimes hits upper bounds in optimization

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{dvc_pytorch2024,
  title = {Deep Vine Copula - PyTorch Implementation},
  author = {[Your Name]},
  year = {2024},
  url = {[Repository URL]}
}
```

## License

This project is licensed under the same terms as the original TensorFlow implementation.

## Acknowledgments

- Original TensorFlow implementation authors
- PyTorch community for excellent documentation
- Contributors to scipy and numpy ecosystems 