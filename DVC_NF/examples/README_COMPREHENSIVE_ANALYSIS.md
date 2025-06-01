# Comprehensive Advanced Analysis for DVC-NF

## 🚀 **Unified Framework for Advanced Testing**

The `comprehensive_advanced_analysis.py` script provides a complete framework for testing all advanced simulations, visualizations, and experiments with configurable parameters suitable for large-scale model testing.

## ✨ **Integrated Capabilities**

### 📊 **All Advanced Simulation Scenarios**
- **Ising-like model** with MCMC sampling and time-varying couplings
- **Hidden Markov Model** with regime switching and correlation structures  
- **Log-linear synergy** with triple interactions and Gibbs sampling
- **Spatiotemporal image blocks** with wave patterns and block aggregation
- **Block switching correlations** with dynamic regime changes
- **Sinusoidal patterns** with smooth periodic correlation changes

### 🎨 **Complete Visualization Suite**
- **R-vine structure graphs** with enhanced styling and time-dependent edges
- **2D copula visualizations** with KDE contours and independence lines
- **Temporal interaction analysis** with line plots and heatmaps
- **Executive summary dashboards** with multi-panel layouts
- **Scenario-specific deep dives** with detailed analysis
- **Comparative analysis matrices** for cross-scenario comparisons

### 🧠 **Advanced Analysis Methods**
- **Time-dependent vine copula modeling** with normalizing flows
- **Entropy-based R-vine optimization** vs traditional tau-based methods
- **Performance benchmarking** and scalability analysis
- **Comprehensive reporting** with JSON and text outputs

## 🔧 **Usage**

### **Quick Start**

```bash
# Quick test (3D, 10 time steps, 50 samples)
python comprehensive_advanced_analysis.py --config quick_test

# Standard analysis (4D, 30 time steps, 100 samples)  
python comprehensive_advanced_analysis.py --config standard_analysis

# Large scale test (8D, 50 time steps, 200 samples)
python comprehensive_advanced_analysis.py --config comprehensive_large

# Scalability test (16D, 100 time steps, 500 samples)
python comprehensive_advanced_analysis.py --config scalability_test
```

### **Custom Configuration**

```bash
# Custom dimensions and parameters
python comprehensive_advanced_analysis.py --dim 6 --time_steps 40 --samples 150

# Specific scenarios only
python comprehensive_advanced_analysis.py --scenarios ising hmm loglinear

# Large scale with custom settings
python comprehensive_advanced_analysis.py --config comprehensive_large --dim 12 --time_steps 80

# Disable advanced analysis for speed
python comprehensive_advanced_analysis.py --config standard_analysis --no_plots
```

## ⚙️ **Configuration Presets**

### **1. Quick Test**
```python
dim=3, n_time_steps=10, n_samples_per_time=50
scenarios=['ising', 'hmm', 'sinusoidal']
run_time_dependent_vines=False
run_entropy_optimization=False
```
**Use case:** Fast testing and debugging

### **2. Standard Analysis** 
```python
dim=4, n_time_steps=30, n_samples_per_time=100
scenarios=['ising', 'hmm', 'loglinear', 'spatiotemporal', 'block_switching', 'sinusoidal']
run_time_dependent_vines=True
run_entropy_optimization=True
```
**Use case:** Regular comprehensive analysis

### **3. Comprehensive Large**
```python
dim=8, n_time_steps=50, n_samples_per_time=200
scenarios=['ising', 'hmm', 'loglinear', 'spatiotemporal', 'block_switching', 'sinusoidal']
run_time_dependent_vines=True
run_entropy_optimization=True
```
**Use case:** Large-scale modeling and publication-quality analysis

### **4. Scalability Test**
```python
dim=16, n_time_steps=100, n_samples_per_time=500
scenarios=['ising', 'hmm', 'sinusoidal']  # Reduced for computational efficiency
run_time_dependent_vines=False  # Too computationally expensive
run_entropy_optimization=False
```
**Use case:** Testing computational limits and scalability

## 📊 **Analysis Phases**

### **Phase 1: Data Generation**
- Generates all configured simulation scenarios
- Measures generation times and memory usage
- Validates data shapes and properties

### **Phase 2: Comprehensive Visualization**
- Creates R-vine structure demonstrations
- Generates 2D copula plots for all scenarios
- Builds temporal interaction analysis
- Produces executive summary dashboards

### **Phase 3: Time-Dependent Vine Analysis**
- Fits time-dependent vine copulas with normalizing flows
- Analyzes bandwidth evolution over time
- Creates training loss visualizations

### **Phase 4: Entropy-Based Optimization**
- Compares entropy-based vs tau-based R-vine optimization
- Generates structure comparison visualizations
- Analyzes optimization performance differences

### **Phase 5: Performance Analysis**
- Computes generation rates and scalability projections
- Analyzes computational complexity for different scenarios
- Projects performance for larger problem sizes

### **Phase 6: Comprehensive Reporting**
- Generates detailed text reports
- Saves JSON results with metadata
- Creates performance visualization dashboards

## 📈 **Output Structure**

```
results/comprehensive_advanced_analysis/
├── comprehensive_analysis_report.txt          # Detailed text report
├── comprehensive_analysis_results.json        # Machine-readable results
├── performance_dashboard.png                  # Performance visualization
├── comprehensive_simulation/                  # Advanced visualization outputs
│   ├── executive_summary.png
│   ├── scenario_deep_dives/
│   └── comparative_analysis/
└── entropy_optimization/                      # Entropy analysis outputs
    └── entropy_vs_tau_optimization.png
```

## 🎯 **Scalability Projections**

The script automatically computes scalability projections for larger problem sizes:

| Scale | Dimensions | Time Steps | Samples | Est. Time |
|-------|------------|------------|---------|-----------|
| 1     | 8D         | 50         | 200     | ~2-5 hours |
| 2     | 16D        | 100        | 500     | ~8-20 hours |
| 3     | 32D        | 200        | 1000    | ~50+ hours |

## 🚀 **Advanced Features**

### **Configurable Analysis**
```python
# Example custom configuration
from comprehensive_advanced_analysis import AnalysisConfig, run_comprehensive_analysis_with_config

custom_config = AnalysisConfig(
    dim=6,
    n_time_steps=40,
    n_samples_per_time=150,
    scenarios_to_run=['ising', 'hmm', 'loglinear'],
    run_time_dependent_vines=True,
    run_entropy_optimization=False,
    mcmc_sweeps=60,
    gibbs_sweeps=60
)

results, metrics, analysis = run_comprehensive_analysis_with_config(custom_config=custom_config)
```

### **Integration with Existing Code**
```python
# Use within existing analysis pipelines
from comprehensive_advanced_analysis import ComprehensiveAdvancedAnalysis

analysis = ComprehensiveAdvancedAnalysis(config)
results, metrics = analysis.run_comprehensive_analysis()

# Access individual components
data_generator = analysis.generator
visualizer = analysis.visualizer
```

## 💡 **Performance Tips**

### **For Large-Scale Testing:**
- Use `scalability_test` config for very large dimensions
- Disable vine analysis and entropy optimization for speed
- Reduce MCMC/Gibbs sweeps for faster generation
- Use `--no_plots` to disable visualization for batch runs

### **For Development:**
- Use `quick_test` config for rapid iteration
- Focus on specific scenarios with `--scenarios`
- Enable interactive plots with `--show_plots` for debugging

### **For Publication:**
- Use `comprehensive_large` config for final analysis
- Enable all advanced features for complete analysis
- Set high DPI for publication-quality plots

## 🎉 **Example Workflows**

### **Research Development**
```bash
# Quick iteration during development
python comprehensive_advanced_analysis.py --config quick_test

# Test specific scenarios
python comprehensive_advanced_analysis.py --scenarios ising hmm --dim 5 --time_steps 20
```

### **Production Analysis**
```bash
# Comprehensive analysis for publication
python comprehensive_advanced_analysis.py --config comprehensive_large

# Custom large-scale analysis
python comprehensive_advanced_analysis.py --dim 10 --time_steps 60 --samples 300
```

### **Computational Benchmarking**
```bash
# Test computational limits
python comprehensive_advanced_analysis.py --config scalability_test

# Performance profiling
python comprehensive_advanced_analysis.py --config standard_analysis --no_plots
```

---

## 🔗 **Integration Points**

This comprehensive script integrates with:
- **DVC-NF data generators** (`dvc_nf.data.generators`)
- **Advanced visualization suite** (`dvc_nf.visualization`)
- **Time-dependent vine copulas** (`dvc_nf.core.flows`)
- **Entropy-based optimization** (`dvc_nf.optimization.entropy`)

**✨ The ultimate tool for comprehensive time-dependent vine copula analysis!** 