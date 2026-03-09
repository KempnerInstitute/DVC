"""
Time-Dependent Vine Copula Modeling

This module provides capabilities for modeling time-dependent dependencies
using normalizing flows and dynamic vine copula structures.
"""

from .flows import TimeBandwidthFlow, MLPEdgeFlow
from .trajectory_models import (
    BasisTrajectory,
    MLPTrajectory,
    StateSpaceTrajectory,
    TimeTrajectoryBase,
    create_trajectory_model,
)

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
from .regularized_cvine import (
    RegularizedDynamicCVine,
    RegularizedDynamicCVineResult,
    DynamicCVineWindowFit,
    EdgeCandidate,
    EdgeDiagnostics,
    SelectedEdgeFit,
    enumerate_edge_candidates,
    mean_copula_nll,
    parameter_distance,
    select_edge_candidate,
    solve_root_sequence,
)
from .joint_dynamic_cvine import (
    JointDynamicCVine,
    JointDynamicCVineResult,
    JointDynamicEdgeFit,
)
from .latent_state_dynamic_cvine import (
    LatentStateDynamicCVine,
    LatentStateDynamicCVineResult,
    LatentStateEdgeFit,
)
from .nonparametric_dynamic_cvine import (
    WindowedNonparametricCVine,
    WindowedNonparametricCVineResult,
    WindowedDynamicNonparametricVine,
    JointDynamicNonparametricCVine,
    JointDynamicNonparametricCVineResult,
    JointDynamicNonparametricVine,
    DynamicNonparametricEdgeFit,
)

__all__ = [
    # Flows
    "TimeBandwidthFlow",
    "MLPEdgeFlow",
    "BasisTrajectory",
    "MLPTrajectory",
    "StateSpaceTrajectory",
    "TimeTrajectoryBase",
    "create_trajectory_model",
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
    # Temporally regularized dynamic C-vine
    "RegularizedDynamicCVine",
    "RegularizedDynamicCVineResult",
    "DynamicCVineWindowFit",
    "EdgeCandidate",
    "EdgeDiagnostics",
    "SelectedEdgeFit",
    "enumerate_edge_candidates",
    "mean_copula_nll",
    "parameter_distance",
    "select_edge_candidate",
    "solve_root_sequence",
    # Joint dynamic C-vine
    "JointDynamicCVine",
    "JointDynamicCVineResult",
    "JointDynamicEdgeFit",
    "LatentStateDynamicCVine",
    "LatentStateDynamicCVineResult",
    "LatentStateEdgeFit",
    "WindowedNonparametricCVine",
    "WindowedNonparametricCVineResult",
    "WindowedDynamicNonparametricVine",
    "JointDynamicNonparametricCVine",
    "JointDynamicNonparametricCVineResult",
    "JointDynamicNonparametricVine",
    "DynamicNonparametricEdgeFit",
]
