"""
Time-Dependent Vine Copula Modeling

This module provides capabilities for modeling time-dependent dependencies
using normalizing flows and dynamic vine copula structures.

Key Features:
- TimeBandwidthFlow: Neural networks for time-dependent bandwidth parameters
- TimeDependentVine: Vine copulas with time-varying parameters
- Dynamic entropy estimation for time series data
- Time-dependent mutual information computation

Main Classes:
- TimeBandwidthFlow: MLP that maps time to bandwidth parameters
- TimeDependentVine: Wrapper for time-varying vine copulas
- TimeSeriesDataset: Data handling for time series vine modeling
- DynamicEntropyEstimator: Entropy estimation for time-dependent systems
"""

from .models import (
    TimeBandwidthFlow,
    TimeDependentVine,
    DynamicEntropyEstimator
)

from .data import (
    TimeSeriesDataset,
    generate_time_dependent_data,
    preprocess_time_series
)

from .flows import (
    MaskedAffineCouplingLayer,
    RealNVPFlow,
    create_normalizing_flow
)

from .trainer import (
    TimeVineTrainer,
    train_time_dependent_vine
)

from .eval import (
    evaluate_time_dependent_vine,
    compute_dynamic_entropy,
    compute_time_dependent_mi
)

__all__ = [
    # Main models
    "TimeBandwidthFlow",
    "TimeDependentVine", 
    "DynamicEntropyEstimator",
    
    # Data handling
    "TimeSeriesDataset",
    "generate_time_dependent_data",
    "preprocess_time_series",
    
    # Normalizing flows
    "MaskedAffineCouplingLayer",
    "RealNVPFlow", 
    "create_normalizing_flow",
    
    # Training
    "TimeVineTrainer",
    "train_time_dependent_vine",
    
    # Evaluation
    "evaluate_time_dependent_vine",
    "compute_dynamic_entropy",
    "compute_time_dependent_mi",
]