"""Experiment runners, benchmark datasets, and method comparison utilities.

Two runner layers are exposed:

- :class:`ExperimentRunner` (from :mod:`dvc_package.experiments.experiment_framework`):
  YAML-configured dispatcher for the bundled experiment types
  (``probability_analysis``, ``entropy_analysis``, ``time_dependent``,
  ``simulation_benchmarks``). This is the entry point used by
  ``scripts/run_experiment.py``.
- :class:`BenchmarkRunner` (from :mod:`dvc_package.experiments.runner`):
  programmatic Cartesian sweep runner used by the ``dvc-experiment`` CLI and by
  callers that want to assemble a :class:`BenchmarkConfig` in Python.
"""

from .benchmarks import (
    BenchmarkDataset,
    generate_benchmark_data,
    run_benchmark_suite,
)
from .comparison import (
    VineComparison,
    compare_optimization_algorithms,
    compare_vine_methods,
)
from .experiment_framework import ExperimentConfig, ExperimentRunner
from .runner import BenchmarkConfig, BenchmarkRunner, run_experiment

__all__ = [
    # YAML-configured dispatcher.
    "ExperimentRunner",
    "ExperimentConfig",
    # Programmatic benchmark grid runner.
    "BenchmarkRunner",
    "BenchmarkConfig",
    "run_experiment",
    # Benchmark datasets.
    "generate_benchmark_data",
    "run_benchmark_suite",
    "BenchmarkDataset",
    # Comparison helpers.
    "compare_vine_methods",
    "compare_optimization_algorithms",
    "VineComparison",
]
