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

## 🎯 Usage Patterns

### Basic Workflow
```python
# 1. Import the framework
from dvc_nf import TimeDependentVineCopula, TimeDependentDataGenerator

# 2. Generate or load data
generator = TimeDependentDataGenerator(dim=3)
data, times, metadata = generator.generate_sinusoidal_correlation_data(...)

# 3. Initialize and fit model
model = TimeDependentVineCopula(dim=3, vine_type='c-vine')
model.initialize_vine_structure()
model.initialize_flows()
model.fit(data, times)

# 4. Analyze results
predictions = model.predict_bandwidth_evolution()
```

### Advanced Analysis
```python
# Comprehensive multi-scenario analysis
from dvc_nf.analysis.comprehensive import ComprehensiveTimeDependentAnalysis

analyzer = ComprehensiveTimeDependentAnalysis(dim=3)
analyzer.run_complete_analysis(
    test_scenarios=['piecewise', 'sinusoidal', 'financial']
)
```

## 📊 Example Data

The examples use various synthetic datasets:

- **Piecewise correlations**: Sudden structural breaks
- **Sinusoidal patterns**: Smooth periodic changes  
- **Financial scenarios**: Volatility clustering and correlation breaks
- **Regime switching**: Markov chain driven changes

## 🔧 Configuration

Most examples can be customized by modifying parameters at the top of each script:

```python
# Common parameters
dim = 3                    # Data dimensionality
n_time_steps = 100        # Number of time points
n_samples_per_time = 150  # Samples per time point
vine_type = 'c-vine'      # Vine structure type
num_epochs = 500          # Training epochs
```

## 📈 Expected Outputs

Examples generate:
- **Plots**: Bandwidth evolution, correlation patterns, training curves
- **Data files**: NumPy arrays (.npz), JSON results, model checkpoints
- **Analysis reports**: Comprehensive performance metrics

All outputs are saved to the `../results/` directory with timestamps and scenario names.

## 🐛 Troubleshooting

**Common issues:**
- Memory errors: Reduce `n_samples_per_time` or `dim`
- Training instability: Lower learning rate or increase regularization
- Import errors: Ensure DVC-NF is in Python path

**Performance tips:**
- Start with small dimensions (3-4) for testing
- Use `--quick` mode for initial exploration
- C-vine structures are typically faster than R-vine

## 🤝 Contributing Examples

To add new examples:
1. Follow the existing naming convention
2. Include comprehensive docstrings
3. Add parameter configuration section at the top
4. Update this README with description
5. Test with both small and realistic parameter sets 