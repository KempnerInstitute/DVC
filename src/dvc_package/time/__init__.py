"""
Time-Dependent Vine Copula Modeling

This module provides capabilities for modeling time-dependent dependencies
using normalizing flows and dynamic vine copula structures.
"""

from .flows import TimeBandwidthFlow, MLPEdgeFlow

from .data import (
    TimeSeriesVineDataset,
    generate_synthetic_time_series,
    create_data_loader,
    preprocess_real_data,
    compute_time_varying_correlations,
)

from .models import (
    TimeDependentVine,
    DynamicEntropyEstimator,
    create_time_dependent_vine,
)

__all__ = [
    # Flows
    "TimeBandwidthFlow",
    "MLPEdgeFlow",
    # Data
    "TimeSeriesVineDataset",
    "generate_synthetic_time_series",
    "create_data_loader",
    "preprocess_real_data",
    "compute_time_varying_correlations",
    # Models
    "TimeDependentVine",
    "DynamicEntropyEstimator",
    "create_time_dependent_vine",
]
