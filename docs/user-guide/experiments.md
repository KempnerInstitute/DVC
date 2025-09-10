# DVC Experiment Framework

A comprehensive experiment framework for vine copula analysis, entropy estimation, and time-dependent modeling using YAML configuration files.

## Features

- **YAML Configuration**: Experiment setup using human-readable YAML files
- **Multiple Experiment Types**: Probability analysis, entropy estimation, time-dependent modeling
- **Vine Copula Support**: C-vine, D-vine, and R-vine with multiple copula families
- **Time-Dependent Modeling**: Normalizing flows for time-varying bandwidth parameters
- **Comprehensive Analysis**: Information theory, correlation preservation, model evaluation
- **Automated Plotting**: Visualization and analysis plots
- **CLI Interface**: Command-line interface for running experiments

## Quick Start

### 1. Create Example Configuration Files

```bash
python run_experiment.py --create-examples
```

This creates example YAML configuration files in the `example_configs/` directory.

### 2. Run an Experiment

```bash
# Run probability analysis
python run_experiment.py example_configs/probability_analysis.yaml

# Run entropy analysis
python run_experiment.py example_configs/entropy_analysis.yaml

# Run time-dependent analysis
python run_experiment.py example_configs/time_dependent.yaml
```

### 3. View Results

Results are saved in the directory specified in your YAML config (default: `results/`). Each experiment creates:
- **JSON results**: Detailed numerical results
- **Plots**: Visualizations and analysis plots  
- **Logs**: Detailed execution logs
- **Models**: Saved model states (when applicable)

## Experiment Types

### 1. Probability Analysis (`probability_analysis`)

Analyzes probability distributions and vine copula structures:

- **Vine Fitting**: Fits C-vine, D-vine, and R-vine models
- **Copula Family Comparison**: Tests multiple copula families (Gaussian, Clayton, Frank, Gumbel, etc.)
- **Correlation Analysis**: Analyzes correlation preservation and structure
- **Information Measures**: Computes entropy, mutual information, and other information-theoretic measures

**Example Configuration**:
```yaml
name: "probability_analysis_example"
description: "Comprehensive probability analysis"
output_dir: "results/probability_analysis"
seed: 42

data_config:
  type: "synthetic"
  n_samples: 2000
  n_variables: 4
  correlation_type: "moderate"

vine_config:
  types: ["c-vine", "d-vine", "r-vine"]
  families: ["independence", "gaussian", "clayton", "frank", "gumbel"]

analysis_config:
  experiment_type: "probability_analysis"
  compute_information_measures: true
```

### 2. Entropy Analysis (`entropy_analysis`)

Focuses on entropy estimation and information-theoretic analysis:

- **Multiple Entropy Estimators**: Gaussian, kernel density, k-nearest neighbors
- **Method Comparison**: Compares different entropy estimation approaches
- **Information Decomposition**: Analyzes marginal vs. joint entropy
- **Mutual Information**: Estimates pairwise and higher-order dependencies

**Example Configuration**:
```yaml
name: "entropy_analysis_example"
description: "Entropy and information estimation"
output_dir: "results/entropy_analysis"
seed: 123

analysis_config:
  experiment_type: "entropy_analysis"
  entropy_methods: ["gaussian", "kernel", "knn"]
```

### 3. Time-Dependent Analysis (`time_dependent`)

Models time-varying dependencies using normalizing flows:

- **Time-Varying Correlations**: Analyzes how correlations change over time
- **Normalizing Flows**: Uses neural networks to model time-dependent bandwidth parameters
- **Temporal Dynamics**: Studies correlation evolution patterns (sinusoidal, linear, piecewise)
- **Model Evaluation**: Assesses model performance and temporal smoothness

**Example Configuration**:
```yaml
name: "time_dependent_example"
description: "Time-dependent vine copula modeling"
output_dir: "results/time_dependent"
seed: 456

time_config:
  n_time_steps: 100
  n_samples_per_time: 200
  n_variables: 3
  correlation_evolution: "sinusoidal"
  hidden_dim: 64

analysis_config:
  experiment_type: "time_dependent"
```

## Configuration Reference

### General Settings

```yaml
name: "experiment_name"              # Experiment identifier
description: "Experiment description" # Human-readable description
output_dir: "results/experiment"     # Output directory
seed: 42                            # Random seed for reproducibility
```

### Data Configuration (`data_config`)

```yaml
data_config:
  type: "synthetic"                 # "synthetic" or "file"
  n_samples: 2000                   # Number of samples
  n_variables: 4                    # Number of variables
  correlation_type: "moderate"      # "moderate", "block", "independent"
  path: "data/mydata.npy"          # Path to data file (if type="file")
```

### Vine Configuration (`vine_config`)

```yaml
vine_config:
  types:                           # Vine types to analyze
    - "c-vine"
    - "d-vine" 
    - "r-vine"
  families:                        # Copula families to test
    - "independence"
    - "gaussian"
    - "student"
    - "clayton"
    - "frank"
    - "gumbel"
  structure_optimization: "tau"    # R-vine structure selection method
```

### Time-Dependent Configuration (`time_config`)

```yaml
time_config:
  n_time_steps: 100               # Number of time points
  n_samples_per_time: 200         # Samples per time point
  n_variables: 3                  # Number of variables
  correlation_evolution: "sinusoidal"  # Evolution pattern
  hidden_dim: 64                  # Neural network hidden dimension
  noise_level: 0.1               # Noise level in correlation evolution
```

### Analysis Configuration (`analysis_config`)

```yaml
analysis_config:
  experiment_type: "probability_analysis"  # Experiment type
  compute_information_measures: true       # Compute entropy/MI
  compare_vine_types: true                # Compare different vine types
  evaluate_copula_families: true          # Evaluate copula families
```

### Plot Configuration (`plot_config`)

```yaml
plot_config:
  create_plots: true              # Whether to create plots
  plot_style: "seaborn"          # Plotting style
  save_format: "png"             # Plot format
  dpi: 300                       # Plot resolution
```

## Advanced Usage

### Custom Data

To use your own data, set `data_config.type: "file"` and provide the path:

```yaml
data_config:
  type: "file"
  path: "path/to/your/data.npy"  # NumPy array with shape (n_samples, n_variables)
```

### Multiple Experiments

You can run multiple experiments in batch:

```bash
# Run all example experiments
for config in example_configs/*.yaml; do
    python run_experiment.py "$config"
done
```

### Custom Logging

Control logging level for debugging:

```bash
python run_experiment.py config.yaml --log-level DEBUG
```

### Output Directory Override

Override the output directory from command line:

```bash
python run_experiment.py config.yaml --output-dir custom_results/
```

## Understanding Results

### Result Structure

Each experiment creates the following output structure:

```
results/experiment_name/
├── experiment_name_results.json    # Detailed numerical results
├── experiment_name_results.pkl     # Full Python objects
├── plots/                          # Visualization plots
│   ├── marginal_distributions.png
│   ├── correlation_heatmap.png
│   ├── pairwise_plots.png
│   └── ...
├── data/                          # Generated or processed data
├── models/                        # Saved model states
└── logs/                          # Detailed execution logs
```

### Key Result Metrics

**Probability Analysis**:
- `correlation_preservation.error`: How well vine preserves correlations
- `information_measures`: Entropy and mutual information estimates
- `vine_analysis`: Results for each vine type tested

**Entropy Analysis**:
- `entropy_analysis`: Entropy estimates for each dataset
- `method_comparison`: Comparison across estimation methods
- `mutual_information`: Pairwise dependency measures

**Time-Dependent Analysis**:
- `correlation_analysis`: Time-varying correlation statistics
- `model_results`: Normalizing flow model performance
- `evaluation`: Model evaluation metrics

## Extending the Framework

### Adding New Experiment Types

1. Create a new class inheriting from `BaseExperiment`
2. Implement the `run()` method
3. Add the experiment type to `ExperimentRunner.run_from_config()`

### Custom Analysis Functions

You can extend the analysis by adding methods to existing experiment classes or creating new analysis modules in the `experiments/` directory.

### Custom Plotting

Plotting functions can be customized by modifying the `_create_*_plots()` methods in each experiment class.

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the DVC root directory
2. **Memory Issues**: Reduce `n_samples` or `n_variables` for large experiments
3. **CUDA Issues**: Set `CUDA_VISIBLE_DEVICES=""` to force CPU usage if needed

### Getting Help

- Check the experiment logs in `experiment.log`
- Use `--log-level DEBUG` for detailed debugging information
- Examine the JSON results for numerical outputs

### Performance Tips

- Use smaller datasets for initial testing
- Enable GPU acceleration for time-dependent experiments
- Use `correlation_type: "independent"` for faster vine fitting

## Examples Gallery

The framework includes several pre-configured examples:

- **`probability_analysis.yaml`**: Standard probability analysis with multiple vine types
- **`entropy_analysis.yaml`**: Information-theoretic analysis with multiple estimators  
- **`time_dependent.yaml`**: Time-varying correlation modeling
- **`comprehensive_comparison.yaml`**: Large-scale comparison study

Each example demonstrates different aspects of the framework and can serve as starting points for your own experiments.

---

## API Reference

For detailed API documentation, see the docstrings in:
- `src/dvc_package/experiments/experiment_framework.py`
- Individual experiment classes and methods

