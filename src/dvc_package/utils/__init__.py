"""
Utility Functions for DVC Package

Shared utilities and helper functions used across the package.
"""

from .utils_tensor import (
    replace_nan_inf,
    check_finite,
    safe_log,
    safe_exp,
    safe_div,
    clamp_preserve_gradients,
    handle_small_sample_size,
    robust_correlation,
    check_positive_definite,
    adaptive_bandwidth_selection,
    robust_matrix_inverse,
)

# Lazy import for data_utils (depends on pandas)
def _get_data_utils():
    from . import data_utils
    return data_utils

__all__ = [
    "replace_nan_inf",
    "check_finite",
    "safe_log",
    "safe_exp",
    "safe_div",
    "clamp_preserve_gradients",
    "handle_small_sample_size",
    "robust_correlation",
    "check_positive_definite",
    "adaptive_bandwidth_selection",
    "robust_matrix_inverse",
]
