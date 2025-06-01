# DVC-NF: Deep Vine Copulas with Normalizing Flows

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**DVC-NF** is a comprehensive framework for **time-dependent vine copula modeling** using **normalizing flows** for dynamic bandwidth estimation. This extends traditional vine copulas to handle temporal evolution of interaction structures through learnable neural networks.

## 🚀 Key Features

- **Time-Dependent Vine Copulas**: Neural networks learn time-varying bandwidth parameters
- **Multiple Vine Types**: Support for R-vine, C-vine, and D-vine structures  
- **Advanced Optimization**: Entropy-based R-vine structure optimization
- **Comprehensive Analysis**: Full pipeline for model comparison and evaluation
- **Synthetic Data Generation**: Multiple temporal scenarios for testing
- **Professional Codebase**: Clean, modular, and well-documented architecture

## 🏗️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd DVC_NF

# Install dependencies
pip install tensorflow numpy scipy matplotlib seaborn pandas

# Add to Python path (if needed)
export PYTHONPATH=$PYTHONPATH:/path/to/DVC_NF
```

## 📁 Project Structure

```
DVC_NF/
├── README.md                    # Main project documentation
├── dvc_nf/                      # Main library package
│   ├── __init__.py             # Package initialization
│   ├── core/                   # Core functionality
│   │   ├── __init__.py
│   │   └── flows.py            # Time-dependent vine copulas with flows
│   ├── data/                   # Data generation utilities
│   │   ├── __init__.py
│   │   └── generators.py       # Synthetic time-dependent data
│   ├── analysis/               # Analysis frameworks
│   │   ├── __init__.py
│   │   └── comprehensive.py    # Complete analysis pipeline
│   └── optimization/           # Advanced optimization methods
│       ├── __init__.py
│       └── entropy.py          # Entropy-based R-vine optimization
├── examples/                   # Example scripts and demos
│   ├── time_dependent_demo.py  # Quick demonstration
│   ├── multivariate_gaussian_analysis.py
│   └── entropy_comparison.py
├── tests/                      # Test files
│   ├── test_entropy_optimization.py
│   ├── lightweight_test.py
│   └── test_script.py
├── docs/                       # Documentation
│   ├── time_dependent.md       # Time-dependent vine copula guide
│   └── entropy_optimization.md # Entropy optimization details
└── results/                    # Output directory (generated)
```

## 🎯 Quick Start

### Basic Usage

```python
import numpy as np
from dvc_nf import TimeDependentVineCopula, TimeDependentDataGenerator

# 1. Generate synthetic time-dependent data
generator = TimeDependentDataGenerator(dim=3, random_seed=42)
data, times, metadata = generator.generate_sinusoidal_correlation_data(
    n_time_steps=100, 
    n_samples_per_time=150,
    base_correlation=0.4,
    amplitude=0.3, 
    frequency=2.0
)

# 2. Initialize time-dependent vine copula
model = TimeDependentVineCopula(
    dim=3,
    vine_type='c-vine',
    optimization_method='tau',
    n_time_steps=100
)

# 3. Initialize structure and flows
model.initialize_vine_structure()
model.initialize_flows(hidden_dim=32)

# 4. Fit the model
model.fit(data, times, num_epochs=500, learning_rate=1e-3)

# 5. Analyze bandwidth evolution
predictions = model.predict_bandwidth_evolution()
for edge_id, pred in predictions.items():
    bw_range = (pred['bandwidths'].min(), pred['bandwidths'].max())
    print(f"{edge_id}: bandwidth range [{bw_range[0]:.3f}, {bw_range[1]:.3f}]")
```

### Command Line Demo

```bash
# Quick demonstration (small parameters)
cd examples
python time_dependent_demo.py --quick

# Comprehensive analysis (all scenarios)
python time_dependent_demo.py --comprehensive
```

## 🔬 Core Components

### 1. TimeBandwidthFlow

Neural network that maps time indices to positive bandwidth parameters:

```python
from dvc_nf.core.flows import TimeBandwidthFlow

# Create flow for specific edge
flow = TimeBandwidthFlow(hidden_dim=64)

# Map time to bandwidth
time_indices = np.linspace(0, 1, 100)
bandwidths = flow(time_indices)  # Shape: (100, 1)
```

**Features:**
- Multi-layer perceptron with batch normalization
- Softplus activation ensures positivity
- Dropout regularization for generalization
- Separate flow for each vine copula edge

### 2. TimeDependentVineCopula

Main class integrating normalizing flows with vine copula structures:

```python
from dvc_nf.core.flows import TimeDependentVineCopula

model = TimeDependentVineCopula(
    dim=4,                        # Data dimensionality
    vine_type='r-vine',          # 'r-vine', 'c-vine', 'd-vine'
    optimization_method='tau',    # 'tau', 'entropy', 'random'
    n_time_steps=200             # Number of time steps
)
```

**Integration with existing DVC framework:**
- ✅ Compatible with `optimal_tree()` R-vine optimization
- ✅ Uses existing `vine_obj_bin` class infrastructure  
- ✅ Maintains all parametric/non-parametric copula support

### 3. TimeDependentDataGenerator

Comprehensive synthetic data generation for testing:

```python
from dvc_nf.data.generators import TimeDependentDataGenerator

generator = TimeDependentDataGenerator(dim=4, random_seed=42)

# Piecewise correlation changes
data1, times1, meta1 = generator.generate_piecewise_correlation_data(
    n_time_steps=100, n_samples_per_time=200,
    breakpoints=[0.3, 0.7], correlations=[0.2, 0.8, 0.3]
)

# Financial market inspired (volatility clustering + correlation breaks)
data2, times2, meta2 = generator.generate_financial_inspired_data(
    n_time_steps=100, n_samples_per_time=200,
    volatility_clustering=True, correlation_breaks=True
)
```

**Available scenarios:**
- **Piecewise**: Structural breaks at specified time points
- **Sinusoidal**: Smooth periodic correlation changes
- **Regime Switching**: Markov chain driven transitions
- **Financial**: Volatility clustering with stress correlations
- **Changing Structure**: Evolving vine tree structures

## 📊 Analysis Pipeline

### Comprehensive Analysis Framework

```python
from dvc_nf.analysis.comprehensive import ComprehensiveTimeDependentAnalysis

# Initialize analyzer
analyzer = ComprehensiveTimeDependentAnalysis(dim=3, random_seed=42)

# Run complete analysis across multiple scenarios
analyzer.run_complete_analysis(
    n_time_steps=100,
    n_samples_per_time=150, 
    test_scenarios=['piecewise', 'sinusoidal', 'financial']
)
```

**Analysis includes:**
- Time-dependent vs static vine copula comparison
- Model fitting performance and computational cost
- Bandwidth evolution visualization and interpretation
- Cross-scenario performance evaluation
- Comprehensive reporting with plots and metrics

### Advanced R-vine Optimization

```python
from dvc_nf.optimization.entropy import EntropyBasedRVineOptimizer

# Initialize entropy-based optimizer
optimizer = EntropyBasedRVineOptimizer(dim=4)

# Find optimal R-vine structure
optimal_structure = optimizer.optimize_rvine_structure(
    data=your_data,
    method='entropy',  # or 'tau', 'random'
    max_iterations=100
)
```

## 🎨 Visualization

**Comprehensive plotting capabilities:**
- ✅ Bandwidth evolution over time (per edge)
- ✅ True vs learned correlation patterns  
- ✅ Training convergence and loss curves
- ✅ Comparative performance metrics
- ✅ Time-dependent correlation heatmaps
- ✅ Distribution comparisons across time periods

**Output formats:**
- High-resolution PNG plots (300 DPI)
- PDF figures for publication
- NumPy data arrays (.npz format)
- JSON results for further analysis

## ⚡ Performance & Scalability

**Computational Characteristics:**
- **Training time**: O(T × E × H) where T=time steps, E=edges, H=hidden units
- **Memory usage**: O(T × N × D) for time series storage
- **Tested scales**: Up to 6 dimensions, 200 time steps, 1000 samples

**Optimization features:**
- Gradient clipping for numerical stability
- Early stopping with patience-based convergence
- Adaptive learning rate scheduling
- Efficient tensor operations with TensorFlow

## 🔗 Integration with DVC Framework

**Full compatibility maintained:**
- ✅ `vine_tree/tree_op.py` - R-vine optimization algorithms
- ✅ `classes/objects.py` - Vine object interfaces  
- ✅ Existing copula fitting and bandwidth selection
- ✅ All parametric and non-parametric copula families

**Extension points:**
- Flow-based bandwidth in `loclik_batch()` functions
- Time-dependent fitting in vine object methods
- Enhanced `optimization_method` parameter support

## 📚 Documentation

- **[Time-Dependent Guide](docs/time_dependent.md)**: Complete guide to time-dependent vine copulas
- **[Entropy Optimization](docs/entropy_optimization.md)**: Advanced R-vine structure optimization
- **[API Reference](docs/api.md)**: Detailed API documentation (coming soon)

## 🧪 Testing

Run the test suite:

```bash
cd tests
python test_entropy_optimization.py     # Test entropy optimization
python lightweight_test.py              # Quick functionality test
python test_script.py                   # Basic integration test
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DVC Framework**: Built on the foundation of the Deep Vine Copulas framework
- **TensorFlow**: Neural network implementation and optimization
- **Research Community**: Vine copula and normalizing flow research

## 📧 Contact

DVC Analysis Team - [contact@dvc-analysis.org](mailto:contact@dvc-analysis.org)

Project Link: [https://github.com/your-org/DVC_NF](https://github.com/your-org/DVC_NF)

---

**⭐ Star this repository if you find it useful!** 