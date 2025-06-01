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

**[advanced_scenarios_demo.py](advanced_scenarios_demo.py)** - Advanced simulation scenarios 🚀
- Demonstrates four cutting-edge simulation scenarios
- Ising-like model with time-varying couplings (MCMC-based)
- Hidden Markov regime switching with higher-order patterns
- Log-linear synergy model with triple interactions (Gibbs sampling)
- Spatiotemporal image blocks for spatial-temporal analysis
- Comprehensive comparative analysis across scenarios
- Professional publication-quality visualizations

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

### Advanced Scenarios Usage
```python
# 1. Ising-like model with time-varying couplings
from dvc_nf.data.generators import TimeDependentDataGenerator

generator = TimeDependentDataGenerator(dim=4)
data, times, metadata = generator.generate_ising_time_series(
    n_time_steps=50,
    n_samples_per_time=200,
    mcmc_sweeps=100  # MCMC sweeps for quality sampling
)

# 2. Hidden Markov regime switching
data, times, metadata = generator.generate_hmm_regimes(
    n_time_steps=60,
    n_samples_per_time=150,
    n_regimes=4,  # Multiple hidden regimes
    regime_transition=0.12  # Transition probability
)

# 3. Log-linear synergy model
data, times, metadata = generator.generate_loglinear_synergy(
    n_time_steps=50,
    n_samples_per_time=200,
    triple_synergy=True,  # Include triple interactions
    gibbs_sweeps=100  # Gibbs sampling sweeps
)

# 4. Spatiotemporal image blocks
data, times, metadata = generator.generate_spatiotemporal_image_blocks(
    height=16, width=16,  # Image dimensions
    n_time_steps=30,
    block_rows=2, block_cols=2,  # Block grid structure
    n_frames_per_time=100
)
```

### Advanced Analysis
```python
# Comprehensive multi-scenario analysis including advanced scenarios
from dvc_nf.analysis.comprehensive import ComprehensiveTimeDependentAnalysis

analyzer = ComprehensiveTimeDependentAnalysis(dim=4)
analyzer.run_complete_analysis(
    test_scenarios=['piecewise', 'sinusoidal', 'financial', 'block_switching', 
                   'beyond_pairwise', 'ising', 'hmm', 'loglinear', 'spatiotemporal']
)
```

### Block-Structured Analysis
```python
# Advanced block-structured correlation modeling
from dvc_nf.data.generators import TimeDependentDataGenerator

generator = TimeDependentDataGenerator(dim=6)
data, times, metadata = generator.generate_block_switching_correlation_data(
    n_time_steps=120,
    n_samples_per_time=150,
    block_sizes=[2, 2, 2],  # Three blocks
    n_regimes=5,
    switch_probability=0.08,
    within_block_corr_range=(0.6, 0.9),
    between_block_corr_range=(-0.7, -0.3)
)
```

### Beyond-Pairwise Analysis
```python
# Advanced beyond-pairwise interactions modeling
from dvc_nf.data.generators import TimeDependentDataGenerator

generator = TimeDependentDataGenerator(dim=4)
data, times, metadata = generator.generate_beyond_pairwise_interactions(
    n_time_steps=60,
    n_samples_per_time=100,
    switch_times=[0.3, 0.7],  # Regime switch points
    corr_low=0.2,
    corr_high=0.8,
    beyond_pairwise_strength=0.4  # Triple interaction strength
)
```

## 📊 Example Data

The examples use various synthetic datasets:

**Standard Scenarios:**
- **Piecewise correlations**: Sudden structural breaks
- **Sinusoidal patterns**: Smooth periodic changes  
- **Financial scenarios**: Volatility clustering and correlation breaks
- **Regime switching**: Markov chain driven changes
- **Block switching**: Complex block-structured correlation matrices with regime changes
- **Beyond-pairwise**: Triple interactions X[k] += strength * X[i] * X[j] with regime switching

**Advanced Scenarios:**
- **Ising-like model**: Magnetic spin interactions with time-varying couplings J_ij(t) and K_ijk(t)
- **HMM regimes**: Hidden Markov model with regime-specific correlation structures
- **Log-linear synergy**: Information-theoretic synergy with triple interactions θ_ij(t) and α_ijk(t)
- **Spatiotemporal blocks**: Spatial-temporal image patterns with wave and swirl dynamics

## 🔧 Configuration

Most examples can be customized by modifying parameters at the top of each script:

```python
# Common parameters
dim = 4                    # Data dimensionality (4D for rich interactions)
n_time_steps = 100        # Number of time points
n_samples_per_time = 150  # Samples per time point
vine_type = 'c-vine'      # Vine structure type
num_epochs = 500          # Training epochs

# Block switching specific
block_sizes = [2, 2, 2]   # Block structure definition
n_regimes = 4             # Number of correlation regimes
switch_probability = 0.05  # Regime switching rate

# Beyond-pairwise specific
beyond_pairwise_strength = 0.3  # Triple interaction strength
switch_times = [0.3, 0.7]       # Regime switch points

# Advanced scenarios specific
mcmc_sweeps = 100         # MCMC sweeps for Ising model
gibbs_sweeps = 100        # Gibbs sweeps for log-linear model
regime_transition = 0.12  # HMM transition probability
block_rows, block_cols = 2, 2  # Spatiotemporal block grid
```

## 📈 Expected Outputs

Examples generate:
- **Plots**: Bandwidth evolution, correlation patterns, training curves
- **Data files**: NumPy arrays (.npz), JSON results, model checkpoints
- **Analysis reports**: Comprehensive performance metrics
- **Block analysis**: Correlation matrix evolution, entropy tracking
- **Triple interaction analysis**: Higher-order dependency detection
- **Advanced visualizations**: 
  - Ising: Spin dynamics, coupling evolution, magnetization
  - HMM: Regime sequences, transition matrices, regime-specific correlations
  - Log-linear: Synergy evolution, activity patterns, binary state dynamics
  - Spatiotemporal: Block intensity evolution, spatial patterns, wave dynamics

All outputs are saved to the `../results/` directory with timestamps and scenario names.

## 🐛 Troubleshooting

**Common issues:**
- Memory errors: Reduce `n_samples_per_time` or `dim`
- Training instability: Lower learning rate or increase regularization
- Import errors: Ensure DVC-NF is in Python path
- MCMC/Gibbs slow: Reduce sweep counts for testing

**Performance tips:**
- Start with small dimensions (3-4) for testing
- Use `--quick` mode for initial exploration
- C-vine structures are typically faster than R-vine
- For block switching: Use dimensions ≥4 for clear block structure
- For beyond-pairwise: Use dimensions ≥3 for triple interactions
- For Ising: Reduce MCMC sweeps for faster generation (30-50 for testing)
- For HMM: Start with 3-4 regimes for manageable complexity
- For log-linear: Reduce Gibbs sweeps for faster sampling (30-50 for testing)
- For spatiotemporal: Use moderate image sizes (8x8 to 16x16)

## 🤝 Contributing Examples

To add new examples:
1. Follow the existing naming convention
2. Include comprehensive docstrings
3. Add parameter configuration section at the top
4. Update this README with description
5. Test with both small and realistic parameter sets