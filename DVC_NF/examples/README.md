# DVC-NF Examples

This directory contains example scripts demonstrating the usage of DVC-NF framework components.

## 🚀 Available Examples

### Quick Start

**[time_dependent_demo.py](time_dependent_demo.py)** - Interactive demonstration script
```bash
# Quick demo (small parameters)
python time_dependent_demo.py --quick

# Comprehensive analysis (multiple scenarios) 
python time_dependent_demo.py --comprehensive
```
- Demonstrates basic time-dependent vine copula workflow
- Includes both quick testing and full analysis modes
- Shows data generation, model fitting, and result visualization

### Core Analysis Examples

**[multivariate_gaussian_analysis.py](multivariate_gaussian_analysis.py)** - Multivariate Gaussian vine copula analysis
- Comprehensive framework for Gaussian distribution modeling
- Correlation estimation and entropy analysis
- Comparison between empirical and vine-estimated correlations
- Publication-quality visualizations

**[entropy_comparison.py](entropy_comparison.py)** - R-vine optimization comparison
- Compares different R-vine structure optimization methods
- Entropy-based vs Kendall's tau-based optimization
- Performance metrics and computational cost analysis

**[block_switching_demo.py](block_switching_demo.py)** - Advanced block-structured correlation modeling
- Demonstrates sophisticated block-structured correlation matrices
- Dynamic regime switching with multiple correlation patterns  
- Entropy evolution tracking and analysis
- Vine copula adaptation to complex temporal structures

**[beyond_pairwise_demo.py](beyond_pairwise_demo.py)** - Beyond-pairwise interactions modeling 🔗
- Demonstrates triple interactions: X[k] += strength * X[i] * X[j]
- Tests vine copula's ability to capture higher-order dependencies
- Pairwise correlations with regime switching
- Empirical triple interaction detection and analysis

## 🎯 Usage Patterns

### Basic Workflow
```