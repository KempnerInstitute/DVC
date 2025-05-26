# Bounded KDE Implementation for Deep Vine Copula

## Overview

The bounded KDE implementation ensures that marginal distributions have **zero probability outside the data range**. This is crucial for:
- Preventing extrapolation beyond observed values
- Handling naturally bounded data (e.g., probabilities, correlations)
- Improving numerical stability in copula calculations

## Key Features

### 1. Boundary Correction Methods

#### Truncation (Simple)
```python
from utils.kde_bounded import bounded_kde_gaussian

density, mesh = bounded_kde_gaussian(
    data, 
    n=128,
    bounds=None,  # Uses data range
    boundary_correction='truncate'
)
```
- Sets density to 0 outside bounds
- Fast but may create discontinuities

#### Reflection
```python
density, mesh = bounded_kde_gaussian(
    data,
    n=128,
    bounds=(-1, 1),  # Explicit bounds
    boundary_correction='reflect'
)
```
- Reflects kernels at boundaries
- Smoother near edges
- Better for data concentrated near bounds

#### Renormalization
```python
density, mesh = bounded_kde_gaussian(
    data,
    n=128,
    bounds=None,
    boundary_correction='renormalize'
)
```
- Truncates and renormalizes to ensure ∫pdf = 1
- Recommended for most use cases

### 2. Transform Methods

For data with natural bounds, transform to unbounded domain:

```python
from utils.kde_bounded import transform_bounded_kde

# Logit transform for [0,1] bounded data
density, mesh = transform_bounded_kde(
    data,
    n=128,
    transform='logit',  # or 'probit', 'log'
    bounds=(0, 1)
)
```

### 3. Adaptive Bounds

Handle outliers by using percentile-based bounds:

```python
from utils.kde_bounded import adaptive_bounded_kde

# Exclude 5% of extreme values
density, mesh = adaptive_bounded_kde(
    data,
    n=128,
    alpha=0.05  # 95% coverage
)
```

## Integration with Deep Vine Copula

### Using in Marginal PDF Estimation

Update your marginal distribution code:

```python
# In your vine copula fitting
from utils.prob_op import kde_wrapper

# Old way (unbounded)
density, mesh = kde_wrapper(data, n=128, method='fft')

# New way (bounded)
density, mesh = kde_wrapper(
    data, 
    n=128, 
    method='fft_bounded',  # Add '_bounded' suffix
    bounded=True,
    bounds=None  # Auto-detect from data
)
```

### For Specific Variable Types

#### Correlation Coefficients [-1, 1]
```python
density, mesh = bounded_kde_gaussian(
    correlation_data,
    bounds=(-1, 1),
    boundary_correction='reflect'
)
```

#### Probabilities [0, 1]
```python
density, mesh = transform_bounded_kde(
    probability_data,
    transform='logit',
    bounds=(0, 1)
)
```

#### Count Data [0, ∞)
```python
density, mesh = bounded_kde_gaussian(
    count_data,
    bounds=(0, None),  # Lower bound only
    boundary_correction='renormalize'
)
```

## Performance Considerations

1. **Computational Cost**: Bounded KDE adds minimal overhead
2. **Memory Usage**: Same as standard KDE
3. **Accuracy**: Improved for bounded data, equivalent for unbounded

## Example: Complete Workflow

```python
import torch
from utils.kde_bounded import bounded_kde_wrapper
from utils.prob_op import kde_wrapper

# Generate example data
data = torch.distributions.Beta(2, 5).sample((1000,))

# Method 1: Direct bounded KDE
density, mesh = bounded_kde_wrapper(
    data,
    method='fft_bounded_renorm',
    enforce_bounds=True,
    bounds=(0, 1)
)

# Method 2: Through prob_op wrapper
density, mesh = kde_wrapper(
    data,
    method='fft',
    bounded=True,
    bounds=(0, 1)
)

# Method 3: Adaptive bounds for robustness
from utils.kde_bounded import adaptive_bounded_kde
density, mesh = adaptive_bounded_kde(data, alpha=0.01)
```

## Recommendations

1. **Default Usage**: Use `boundary_correction='renormalize'`
2. **Known Bounds**: Always specify explicit bounds when known
3. **Unknown Bounds**: Use adaptive bounds with α=0.05
4. **Transform Method**: Use for strictly bounded domains (e.g., [0,1])
5. **Reflection Method**: Use when data concentrates near boundaries

## Validation

The bounded KDE ensures:
- `density[mesh < data.min()] = 0`
- `density[mesh > data.max()] = 0`
- `∫ density = 1.0` (when renormalized)

This prevents the copula from assigning probability mass to impossible values, improving both accuracy and interpretability of the Deep Vine Copula model. 