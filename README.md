# DVC: Dynamic Vine Copula Library

A comprehensive Python library for vine copula estimation and time-dependent modeling using normalizing flows.

## Overview

DVC provides implementations for:
- **Vine Copula Models**: C-vine, D-vine, and R-vine with multiple copula families
- **Time-Dependent Modeling**: Normalizing flows for time-varying dependencies
- **Information Theory**: Entropy estimation and mutual information analysis
- **Experiment Framework**: YAML-configurable experiments for research

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate dvc-env

# Or use the automated setup script
./setup_environment.sh
```

### 2. Run Examples

```bash
# Basic vine copula example
python examples/basic_vine_example.py

# Time-dependent modeling example  
python examples/time_dependent_example.py

# Entropy analysis example
python examples/entropy_analysis_example.py
```

### 3. Run Experiments

```bash
# Create example configurations
python scripts/run_experiment.py --create-examples

# Run probability analysis
python scripts/run_experiment.py configs/probability_analysis.yaml

# Run time-dependent analysis
python scripts/run_experiment.py configs/time_dependent.yaml
```

## Repository Structure

```
DVC/
├── src/dvc_package/           # Main source code
│   ├── core/                  # Core vine copula implementations
│   ├── time/                  # Time-dependent modeling
│   ├── experiments/           # Experiment framework
│   └── utils/                 # Utility functions
├── examples/                  # Usage examples
├── scripts/                   # Utility scripts
├── configs/                   # Experiment configurations
├── tests/                     # Unit tests
├── docs/                      # Documentation
└── archive/                   # Legacy implementations
```

## Features

### Vine Copula Models
- **Multiple vine types**: C-vine, D-vine, R-vine with structure optimization
- **Copula families**: Gaussian, Student-t, Clayton, Frank, Gumbel, Independence
- **Advanced fitting**: Gradient-based parameter estimation with numerical stability
- **Correlation preservation**: Specialized algorithms for maintaining dependencies

### Time-Dependent Modeling
- **Normalizing flows**: Neural networks for time-varying bandwidth parameters
- **Temporal analysis**: Correlation evolution and entropy dynamics
- **Model evaluation**: Performance metrics and temporal smoothness assessment

### Analysis Tools
- **Information theory**: Multiple entropy estimation methods (Gaussian, kernel, k-NN)
- **Mutual information**: Pairwise and higher-order dependency analysis
- **Model comparison**: Systematic evaluation across vine types and copula families

### Experiment Framework
- **YAML configuration**: Easy experiment setup and reproducibility
- **Automated analysis**: Comprehensive statistical analysis and visualization
- **Extensible design**: Simple framework for adding new experiment types

## Documentation

- **[Setup Guide](docs/setup.md)**: Environment setup and installation
- **[User Guide](docs/user-guide/)**: Detailed usage instructions
- **[API Reference](src/dvc_package/)**: Code documentation and examples

## Examples

### Basic Vine Copula Usage

```python
from dvc_package.core.vine_factory import create_vine
import numpy as np

# Generate data
data = np.random.multivariate_normal([0, 0, 0], np.eye(3), 1000)

# Create and fit vine
vine = create_vine('c-vine', vine_depth=3, data=data)

# Analyze structure
print(f"Vine structure: {vine.ind_vine}")
```

### Time-Dependent Analysis

```python
from dvc_package.time.data import generate_synthetic_time_series
from dvc_package.time.model import TimeDependentVineCopula

# Generate time-dependent data
time_data, time_indices = generate_synthetic_time_series(
    n_time_steps=100, n_samples=200, n_variables=3,
    correlation_evolution='sinusoidal'
)

# Analyze temporal correlations
from dvc_package.time.data import compute_time_varying_correlations
correlations = compute_time_varying_correlations(time_data)
```

### Experiment Configuration

```yaml
# configs/my_experiment.yaml
name: "custom_analysis"
description: "Custom vine copula analysis"
output_dir: "results/custom"

data_config:
  type: "synthetic"
  n_samples: 2000
  n_variables: 4

vine_config:
  types: ["c-vine", "r-vine"]
  families: ["gaussian", "clayton", "frank"]

analysis_config:
  experiment_type: "probability_analysis"
  compute_information_measures: true
```

## Testing

```bash
# Run tests
python -m pytest tests/

# Test installation
python scripts/test_installation.py
```
