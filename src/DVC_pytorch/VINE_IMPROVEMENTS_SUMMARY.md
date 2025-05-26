# Vine Copula Improvements Summary

This document summarizes the five major improvements implemented for the PyTorch vine copula system.

## 1. Proper Sampling Methods from Fitted Vines

**Module**: `sampling/vine_sampling.py`

### Features Implemented:
- **VineSampler class**: Comprehensive sampling from fitted vine copula models
- **Inverse transform method**: Proper conditional CDF inversion for accurate sampling
- **Support for all vine types**: C-vine, D-vine, and R-vine structures
- **Uniform and original scale sampling**: Sample in [0,1]^d or transform to original margins

### Key Methods:
- `sample_uniform()`: Generate samples in uniform copula space
- `sample()`: Generate samples in original data scale
- Tree-specific sampling: `_sample_cvine()`, `_sample_dvine()`, `_sample_rvine()`
- h-function inversions for conditional sampling

## 2. Entropy Calculation for Vine Copulas

**Module**: `evalu/vine_entropy.py`

### Features Implemented:
- **VineEntropyCalculator class**: Complete entropy and information metrics
- **Total entropy**: H(X) using Monte Carlo integration
- **Copula entropy**: Mutual information I(X1,...,Xd)
- **Conditional entropy**: H(Y|X) for any variable subsets
- **Mutual information**: I(X;Y) between variable groups

### Key Methods:
- `total_entropy()`: Full joint entropy of the vine model
- `copula_entropy()`: Dependence information captured by copula
- `conditional_entropy()`: Entropy of target given conditioning variables
- `mutual_information()`: Information shared between variable groups
- k-NN entropy estimation for non-parametric cases

## 3. Improved Conditional Distribution Prediction

**Module**: `pred/vine_conditional.py`

### Features Implemented:
- **VineConditionalPredictor class**: Predict conditional distributions
- **Conditional sampling**: P(X_target | X_given = values)
- **Conditional statistics**: Mean, quantiles, and density estimation
- **h-function caching**: Pre-compute for efficiency

### Key Methods:
- `predict_conditional()`: Generate conditional samples
- `conditional_mean()`: E[X_target | X_given]
- `conditional_quantiles()`: Quantiles of conditional distribution
- `conditional_density()`: Evaluate conditional PDF at points
- Proper vine structure traversal for conditioning

## 4. More Parametric Copula Families

**Module**: `param/parametric_copulas.py`

### Copula Families Implemented:
1. **Gaussian (Normal) Copula**: Elliptical with correlation parameter
2. **Student-t Copula**: Elliptical with correlation and degrees of freedom
3. **Clayton Copula**: Archimedean with lower tail dependence
4. **Gumbel Copula**: Archimedean with upper tail dependence
5. **Frank Copula**: Archimedean with symmetric dependence
6. **Joe Copula**: Archimedean with upper tail dependence

### Features:
- Analytical PDF, CDF, and h-functions where possible
- Kendall's tau conversions for all families
- Maximum likelihood estimation (MLE) fitting
- Proper device handling for GPU support

## 5. Optimized Non-parametric Bandwidth Selection

**Module**: `optim/bandwidth_selection.py`

### Bandwidth Selection Methods:
1. **Cross-validation (CV)**: Leave-one-out likelihood maximization
2. **Maximum likelihood (ML)**: Direct likelihood optimization
3. **Plug-in methods**: Sheather-Jones optimal bandwidth
4. **Adaptive bandwidth**: Local density-based adjustment

### Advanced Features:
- **BandwidthSelector class**: Unified interface for all methods
- **BandwidthOptimizer class**: Vine-specific optimization
- **Copula-specific methods**: Transform to handle bounded [0,1] support
- **Adaptive bandwidth matrices**: Full covariance bandwidth selection

### Special Optimizations:
- Boundary correction for copula densities
- Transformation methods for better performance
- Vine structure-aware bandwidth selection
- Handles both marginal and copula density estimation

## Usage Example

```python
import torch
from classes.objects import vine_obj_bin, margin_obj
from sampling.vine_sampling import VineSampler
from evalu.vine_entropy import VineEntropyCalculator
from pred.vine_conditional import VineConditionalPredictor
from param.parametric_copulas import create_copula, fit_copula_mle
from optim.bandwidth_selection import BandwidthSelector, BandwidthOptimizer

# Fit a vine copula model
data = torch.randn(1000, 3)
margins = [margin_obj('kernel', None, True) for _ in range(3)]
vine = vine_obj_bin('c-vine', ['gaussian', 'clayton'], 2, margins, 25, 'matrix')
vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)

# 1. Sample from the fitted vine
sampler = VineSampler(vine)
new_samples = sampler.sample(500)

# 2. Calculate entropy
entropy_calc = VineEntropyCalculator(vine)
total_entropy = entropy_calc.total_entropy()
mutual_info = entropy_calc.mutual_information([0], [1, 2])

# 3. Conditional prediction
predictor = VineConditionalPredictor(vine)
cond_mean = predictor.conditional_mean([2], [0, 1], torch.tensor([[0.0, 1.0]]))

# 4. Use parametric copulas
clayton = create_copula('clayton', 2.0)
density = clayton.pdf(u, v)

# 5. Optimize bandwidth
selector = BandwidthSelector(method='cv')
optimal_h = selector.select_bandwidth(data[:, 0])
```

## Performance Improvements

1. **GPU Support**: All operations support CUDA tensors
2. **Numerical Stability**: Improved handling of edge cases and bounds
3. **Efficiency**: Caching and vectorized operations where possible
4. **Flexibility**: Multiple methods and options for each task
5. **Robustness**: Better error handling and fallback options

## Testing

All improvements are tested in `test_vine_improvements.py`, which includes:
- Correlation preservation in sampling
- Entropy estimation accuracy
- Conditional distribution correctness
- Parametric copula properties
- Bandwidth selection performance

These improvements make the PyTorch vine copula implementation more complete, efficient, and ready for practical applications in high-dimensional dependence modeling. 