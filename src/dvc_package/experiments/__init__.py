"""
Experimental Framework for DVC Package

Provides tools for running comprehensive experiments, benchmarks,
and comparative analyses of vine copula methods.

Key Features:
- Experiment configuration management
- Automated experiment runners
- Performance benchmarking
- Comparative analysis tools
- Result visualization and reporting

Main modules:
- runner: Main experiment execution engine
- benchmarks: Standard benchmark datasets and tests
- comparison: Methods for comparing different vine approaches
- visualization: Plotting and analysis tools
"""

from .runner import (
    run_experiment,
    ExperimentRunner,
    ExperimentConfig
)

from .benchmarks import (
    generate_benchmark_data,
    run_benchmark_suite,
    BenchmarkDataset
)

from .comparison import (
    compare_vine_methods,
    compare_optimization_algorithms,
    VineComparison
)

__all__ = [
    # Main experiment runner
    "run_experiment",
    "ExperimentRunner", 
    "ExperimentConfig",
    
    # Benchmarking
    "generate_benchmark_data",
    "run_benchmark_suite",
    "BenchmarkDataset",
    
    # Comparison tools
    "compare_vine_methods",
    "compare_optimization_algorithms", 
    "VineComparison",
]
