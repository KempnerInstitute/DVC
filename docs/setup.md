# DVC Environment Setup Instructions

This guide provides step-by-step instructions to set up a conda environment for the DVC (Dynamic Vine Copula) project.

## Prerequisites

- Conda or Miniconda installed on your system
- Python 3.8 or higher

## Quick Setup (Recommended)

### 1. Create Conda Environment

```bash
# Navigate to your DVC project directory
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/DVC

# Create new conda environment with Python 3.9
conda create -n dvc-env python=3.9 -y

# Activate the environment
conda activate dvc-env
```

### 2. Install Core Dependencies

```bash
# Install PyTorch (CPU version - adjust for your CUDA version if needed)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Alternative: For CUDA 11.8 (if you have CUDA GPU)
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install scientific computing packages
conda install numpy scipy matplotlib seaborn pandas -c conda-forge -y

# Install additional dependencies
conda install scikit-learn pyyaml jupyter notebook -c conda-forge -y

# Install via pip for packages not available in conda
pip install tensorboard
```

### 3. Install Optional Dependencies

```bash
# For enhanced plotting and analysis
pip install plotly kaleido

# For statistical analysis
conda install statsmodels -c conda-forge -y

# For optimization
pip install optuna

# For progress bars and utilities
pip install tqdm rich

# For memory profiling (optional)
pip install memory-profiler
```

## Detailed Package List

Here's the complete list of packages and their purposes:

### Core Scientific Computing
- **numpy** (>=1.21.0): Numerical computations
- **scipy** (>=1.7.0): Statistical functions, optimization
- **matplotlib** (>=3.5.0): Basic plotting
- **pandas** (>=1.3.0): Data manipulation and analysis

### Machine Learning
- **torch** (>=1.12.0): PyTorch for neural networks and tensor operations
- **torchvision**: Computer vision utilities (comes with PyTorch)
- **scikit-learn** (>=1.0.0): Additional ML utilities and metrics

### Vine Copula Specific
- **scipy.stats**: Statistical distributions and tests (kendalltau, multivariate_normal, etc.)
- **numpy.linalg**: Linear algebra operations for correlation matrices

### Configuration and I/O
- **pyyaml**: YAML configuration file parsing
- **json**: Built-in JSON handling

### Visualization
- **seaborn** (>=0.11.0): Enhanced statistical plotting
- **plotly** (optional): Interactive plots
- **matplotlib**: Static plots

### Development and Analysis
- **jupyter**: Notebook interface for interactive development
- **tensorboard**: Monitoring training progress
- **tqdm**: Progress bars
- **rich**: Enhanced terminal output

## Environment File Method (Alternative)

You can also create an environment using a YAML file:

### 1. Create environment.yml file

```yaml
# environment.yml
name: dvc-env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch>=1.12.0
  - torchvision
  - torchaudio
  - numpy>=1.21.0
  - scipy>=1.7.0
  - matplotlib>=3.5.0
  - seaborn>=0.11.0
  - pandas>=1.3.0
  - scikit-learn>=1.0.0
  - pyyaml
  - jupyter
  - notebook
  - statsmodels
  - pip
  - pip:
    - tensorboard
    - tqdm
    - rich
    - plotly
    - kaleido
    - optuna
```

### 2. Create environment from file

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate environment
conda activate dvc-env
```

## Verification

### Test Your Installation

```bash
# Activate environment
conda activate dvc-env

# Test core packages
python -c "
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
print('All core packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Test DVC Framework

```bash
# Test the experiment framework
python run_experiment.py --create-examples

# If successful, you should see:
# "Example configuration files created in 'example_configs/' directory"
```

## GPU Support (Optional)

If you have NVIDIA GPU and want CUDA acceleration:

### Check CUDA Version
```bash
nvidia-smi  # Check your CUDA version
```

### Install PyTorch with CUDA
```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Package Versions (Tested)

Here are the specific versions that have been tested with the DVC framework:

```
python=3.9.18
pytorch=2.1.0
numpy=1.24.3
scipy=1.10.1
matplotlib=3.7.2
seaborn=0.12.2
pandas=2.0.3
scikit-learn=1.3.0
pyyaml=6.0.1
jupyter=1.0.0
statsmodels=0.14.0
```

## Troubleshooting

### Common Issues

**1. PyTorch Installation Issues**
```bash
# If PyTorch installation fails, try:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**2. CUDA Issues**
```bash
# Force CPU-only PyTorch if CUDA causes problems:
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

**3. Package Conflicts**
```bash
# If you encounter package conflicts, create a fresh environment:
conda deactivate
conda env remove -n dvc-env
conda create -n dvc-env python=3.9 -y
conda activate dvc-env
# Then reinstall packages
```

**4. Import Errors**
```bash
# Make sure you're in the DVC directory and environment is activated:
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/DVC
conda activate dvc-env
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Memory Issues

For large experiments, you might need to adjust memory settings:

```bash
# Increase memory limits (if needed)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Or force CPU usage
export CUDA_VISIBLE_DEVICES=""
```

## Development Setup

If you plan to modify the code:

```bash
# Install development packages
conda install pytest black flake8 mypy -c conda-forge -y

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Complete Setup Script

Here's a complete script to set up everything:

```bash
#!/bin/bash
# setup_dvc_environment.sh

echo "Setting up DVC environment..."

# Navigate to project directory
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/DVC

# Create conda environment
echo "Creating conda environment..."
conda create -n dvc-env python=3.9 -y

# Activate environment
echo "Activating environment..."
conda activate dvc-env

# Install core packages
echo "Installing core packages..."
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
conda install numpy scipy matplotlib seaborn pandas -c conda-forge -y
conda install scikit-learn pyyaml jupyter notebook statsmodels -c conda-forge -y

# Install additional packages via pip
echo "Installing additional packages..."
pip install tensorboard tqdm rich plotly kaleido

# Test installation
echo "Testing installation..."
python -c "
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import yaml
print('Installation successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Create example configs
echo "Creating example configurations..."
python run_experiment.py --create-examples

echo "Setup complete!"
echo "Activate environment with: conda activate dvc-env"
echo "Run experiments with: python run_experiment.py example_configs/<config>.yaml"
```

Save this as `setup_dvc_environment.sh`, make it executable with `chmod +x setup_dvc_environment.sh`, and run it with `./setup_dvc_environment.sh`.

## Quick Start After Setup

Once your environment is set up:

```bash
# Activate environment
conda activate dvc-env

# Navigate to DVC directory
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/DVC

# Run a quick test
python run_experiment.py example_configs/probability_analysis.yaml

# Check results
ls -la results/probability_analysis/
```

This setup will give you everything needed to run vine copula experiments, time-dependent modeling, entropy analysis, and all the features we've implemented.