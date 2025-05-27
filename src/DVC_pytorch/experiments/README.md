# DVC PyTorch Experiments

This directory contains experiments and examples for the Deep Vine Copula (DVC) PyTorch implementation.

## Available Experiments

### 1. Simple Vine Example (`simple_vine_example.py`)
A basic demonstration of vine copula usage:
- Generates synthetic data with different marginal distributions
- Fits a C-vine copula model
- Generates samples and compares correlations
- Plots the vine structure and data comparisons

Run with:
```bash
python simple_vine_example.py
```

### 2. Comprehensive Gaussian Vine Test (`comprehensive_gaussian_vine_test.py`)
A thorough test of the implementation:
- Tests multiple dimensions (3D and 5D)
- Different correlation structures (Toeplitz, block diagonal)
- Various marginal distributions (normal, exponential, uniform, student-t, gamma)
- Compares C-vine, D-vine, and R-vine models
- Evaluates correlation preservation and entropy estimation
- Generates comprehensive plots and statistics

Run with:
```bash
python comprehensive_gaussian_vine_test.py
```

## Output Files

The experiments generate several output files:

- **Plots**:
  - `vine_structure.png` - Visualization of the fitted vine structure
  - `data_comparison.png` - Scatter plots comparing original and sampled data
  - `correlation_comparison_*.png` - Heatmaps of correlation matrices
  - `comprehensive_test_summary.png` - Summary statistics and comparisons

- **Data**:
  - `comprehensive_vine_test_results.csv` - Detailed results from all test configurations

## Key Features Demonstrated

1. **Data Generation**: Creating synthetic data with known copula structure
2. **Model Fitting**: Fitting different vine types (C, D, R) with various copula families
3. **Sampling**: Generating new samples from fitted models
4. **Evaluation**: Computing correlations, entropies, and other statistics
5. **Visualization**: Plotting vine structures and data comparisons

## Usage Tips

- Adjust `n_samples` to control the amount of training data
- Modify `dimensions` to test different data dimensions
- Change `marginal_configs` to experiment with different marginal distributions
- Set `vine_types` to focus on specific vine structures

## Requirements

The experiments require the following packages:
- numpy
- torch
- matplotlib
- seaborn
- scipy
- pandas

All DVC modules should be properly installed and accessible. 