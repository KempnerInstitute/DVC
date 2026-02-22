# DVC Project Structure

This document describes the organization of the DVC (Dynamic Vine Copula) codebase.

## Repository Layout

```
DVC/
├── README.md                      # Project overview and quick start
├── LICENSE                        # License information
├── environment.yml               # Conda environment specification
├── requirements.txt              # Pip requirements
├── setup_environment.sh          # Automated environment setup
├── pyproject.toml               # Python project configuration
├── CONTRIBUTING.md              # Contribution guidelines
│
├── src/dvc_package/             # Main source code package
│   ├── __init__.py
│   ├── core/                    # Core vine copula implementations
│   │   ├── __init__.py
│   │   ├── objects.py           # Basic data structures
│   │   ├── vine_factory.py      # Vine creation factory
│   │   ├── vine_model.py        # Main vine fitting/sampling
│   │   ├── param_copula.py      # Parametric copula families
│   │   ├── d_vine_fix.py        # D-vine specific algorithms
│   │   ├── cop_eval.py          # Copula evaluation functions
│   │   ├── vine_eval.py         # Vine evaluation
│   │   ├── vine_tree.py         # Vine tree structures
│   │   ├── transformation.py    # Data transformations
│   │   ├── preparation.py       # Data preparation
│   │   ├── prediction.py        # Prediction functions
│   │   ├── sampling.py          # Sampling algorithms
│   │   ├── grid_ops.py          # Grid operations
│   │   ├── dataset_ops.py       # Dataset operations
│   │   ├── info_estimation.py   # Information estimation
│   │   ├── config.py            # Configuration management
│   │   ├── utils_*.py           # Various utility modules
│   │   └── ...
│   │
│   ├── time/                    # Time-dependent modeling
│   │   ├── __init__.py
│   │   ├── flows.py             # Normalizing flows
│   │   ├── nf.py                # Flow utilities and bijectors
│   │   ├── data.py              # Time-series data utilities
│   │   ├── trainer.py           # Training utilities
│   │   ├── eval.py              # Evaluation functions
│   │   ├── metrics.py           # Time-dependent metrics
│   │   └── models.py            # Canonical time-dependent vine models
│   │
│   ├── experiments/             # Experiment framework
│   │   ├── __init__.py
│   │   ├── experiment_framework.py  # Main experiment classes
│   │   ├── benchmarks.py        # Benchmarking utilities
│   │   ├── runner.py            # Experiment runner
│   │   └── comparison.py        # Comparison utilities
│   │
│   ├── optimization/            # Structure optimization
│   │   ├── __init__.py
│   │   ├── criteria.py          # Optimization criteria
│   │   └── structure.py         # Structure optimization
│   │
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── data_utils.py        # Data utilities
│   │   └── utils_tensor.py      # Tensor utilities
│   │
│   └── cli/                     # Command-line interface
│       ├── __init__.py
│       └── main.py              # Main CLI entry point
│
├── examples/                    # Usage examples
│   ├── basic_vine_example.py    # Basic vine copula usage
│   ├── time_dependent_example.py # Time-dependent modeling
│   └── entropy_analysis_example.py # Entropy analysis
│
├── scripts/                     # Utility scripts
│   ├── run_experiment.py        # Experiment runner script
│   ├── test_installation.py     # Installation verification
│   └── install_package.sh       # Package installation script
│
├── configs/                     # Experiment configurations
│   ├── probability_analysis.yaml
│   ├── entropy_analysis.yaml
│   ├── time_dependent.yaml
│   └── comprehensive_comparison.yaml
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── conftest.py              # Test configuration
│   ├── test_vine_factory.py     # Test vine factory
│   └── test_optimization.py     # Test optimization
│
├── docs/                        # Documentation
│   ├── index.md                 # Documentation index
│   ├── setup.md                 # Environment setup guide
│   ├── structure.md             # Project structure (this file)
│   ├── user-guide/              # User documentation
│   │   ├── fitting.md           # Vine fitting guide
│   │   ├── evaluation.md        # Model evaluation
│   │   ├── time-dependent.md    # Time-dependent modeling
│   │   └── experiments.md       # Experiment framework
│   ├── research/                # Research documentation
│   │   ├── overview.md          # Research overview
│   │   └── metrics.md           # Metrics and evaluation
│   └── howto/                   # How-to guides
│       └── runner.md            # Running experiments
│
└── archive/                     # Legacy implementations
    ├── DVC_NF/                  # Original TensorFlow time-dependent implementation
    ├── src_legacy/              # Legacy source code
    │   ├── DVC_tensorflow/      # TensorFlow reference implementation
    │   └── ...
    ├── scripts_legacy/          # Legacy scripts
    ├── md_notes/                # Development notes (internal)
    └── ...
```

## Key Components

### Core Package (`src/dvc_package/core/`)

The core package contains the main vine copula implementations:

- **`vine_factory.py`**: Factory functions for creating different vine types
- **`vine_model.py`**: Main vine fitting, evaluation, and sampling functions
- **`param_copula.py`**: Parametric copula family implementations
- **`objects.py`**: Core data structures (vine_obj_bin, copula_obj, etc.)
- **`d_vine_fix.py`**: Specialized D-vine algorithms with correlation preservation

### Time-Dependent Module (`src/dvc_package/time/`)

Time-dependent modeling using normalizing flows:

- **`flows.py`**: Normalizing flow implementations
- **`nf.py`**: Flow utilities and bijectors
- **`data.py`**: Time-series data generation and preprocessing
- **`models.py`**: Canonical `TimeDependentVine` APIs for fitting/evaluation

### Experiment Framework (`src/dvc_package/experiments/`)

YAML-configurable experiment system:

- **`experiment_framework.py`**: Main experiment classes and runner
- **`benchmarks.py`**: Benchmarking and comparison utilities

### Examples (`examples/`)

Working examples demonstrating library usage:

- **`basic_vine_example.py`**: Basic vine copula operations
- **`time_dependent_example.py`**: Time-dependent modeling
- **`entropy_analysis_example.py`**: Information-theoretic analysis

### Scripts (`scripts/`)

Utility scripts for running experiments and testing:

- **`run_experiment.py`**: Main experiment runner CLI
- **`test_installation.py`**: Environment verification

### Configurations (`configs/`)

YAML configuration files for experiments:

- **`probability_analysis.yaml`**: Probability distribution analysis
- **`entropy_analysis.yaml`**: Entropy and information estimation
- **`time_dependent.yaml`**: Time-dependent vine modeling
- **`comprehensive_comparison.yaml`**: Large-scale comparison study

## Usage Patterns

### Basic Usage

1. **Environment Setup**: `./setup_environment.sh`
2. **Run Examples**: `python examples/basic_vine_example.py`
3. **Run Experiments**: `python scripts/run_experiment.py configs/probability_analysis.yaml`

### Development Workflow

1. **Modify Source**: Edit files in `src/dvc_package/`
2. **Test Changes**: Run tests with `python -m pytest tests/`
3. **Validate**: Run examples to ensure functionality
4. **Document**: Update relevant documentation in `docs/`

### Research Workflow

1. **Create Config**: Copy and modify YAML files in `configs/`
2. **Run Experiments**: Use `scripts/run_experiment.py`
3. **Analyze Results**: Check output in `results/` directory
4. **Compare Methods**: Use comprehensive comparison configs


## Import Structure

### Internal Imports
```python
# Core functionality
from dvc_package.core.vine_factory import create_vine
from dvc_package.core.param_copula import fit_gaussian

# Time-dependent modeling
from dvc_package.time.models import create_time_dependent_vine
from dvc_package.time.data import generate_synthetic_time_series

# Experiments
from dvc_package.experiments.experiment_framework import ExperimentRunner
```

### External Dependencies
```python
# Scientific computing
import numpy as np
import scipy
import torch

# Configuration and I/O
import yaml
import json

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
```
