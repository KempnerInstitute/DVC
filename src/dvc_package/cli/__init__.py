"""
Command Line Interface for DVC Package

Provides command-line tools for vine copula modeling, optimization,
and time-dependent analysis.

Available commands:
- dvc-fit: Fit vine copula models to data
- dvc-entropy: Estimate entropy and mutual information
- dvc-time: Time-dependent modeling and analysis
- dvc-experiment: Run experimental configurations
"""

from .main import (
    fit_vine_cli,
    estimate_entropy_cli,
    time_model_cli,
    run_experiment_cli
)

__all__ = [
    "fit_vine_cli",
    "estimate_entropy_cli", 
    "time_model_cli",
    "run_experiment_cli",
]
