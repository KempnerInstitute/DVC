"""
Utility Functions for DVC Package

Shared utilities and helper functions used across the package.
"""

from .data_utils import (
    load_data,
    save_data,
    validate_data,
    preprocess_data
)

from .math_utils import (
    ensure_positive_definite,
    compute_correlation_matrix,
    numerical_gradient
)

from .plotting_utils import (
    plot_vine_structure,
    plot_entropy_evolution,
    plot_correlation_heatmap
)

__all__ = [
    # Data utilities
    "load_data",
    "save_data", 
    "validate_data",
    "preprocess_data",
    
    # Math utilities
    "ensure_positive_definite",
    "compute_correlation_matrix",
    "numerical_gradient",
    
    # Plotting utilities
    "plot_vine_structure",
    "plot_entropy_evolution",
    "plot_correlation_heatmap",
]
