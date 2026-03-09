"""
DVC (Dynamic Vine Copula) Package

A GPU-accelerated Python library for vine copula modeling, multivariate
dependency estimation, and time-dependent entropy analysis using PyTorch.

Key Features:
- Multiple vine types: C-vine, D-vine, R-vine with structure optimization
- Parametric copula families: Gaussian, Student-t, Clayton, Frank, Gumbel, Joe
- Nonparametric local-likelihood copula estimation
- Time-dependent dependency modeling using normalizing flows
- Monte Carlo entropy and mutual information estimation

Main modules:
- core: Complete vine copula implementation
- time: Time-dependent modeling with normalizing flows
- optimization: Vine structure optimization algorithms
- cli: Command-line interface tools
- experiments: Experiment runners and templates
- utils: Shared utilities and helpers
"""

__version__ = "0.1.0"

# Import main classes and functions for easy access
from .core.objects import vine_obj_bin, copula_obj, margin_obj, cop_par_obj
from .core.vine_model import fit_vine, sample_vine, evaluate_vine
from .core.nonparametric_vine import fit_nonparametric_vine, evaluate_nonparametric_vine
from .core.info_estimation import vine_entropy, mutual_information, cond_vine_entropy

# Import vine factory functions
from .core.vine_factory import create_vine, VineType, optimize_vine_type

# Lazy imports for optional heavy modules
def optimize_vine_structure(*args, **kwargs):
    from .optimization.structure import optimize_vine_structure
    return optimize_vine_structure(*args, **kwargs)

def _get_time_dependent_vine():
    from .time.models import TimeDependentVine, TimeBandwidthFlow
    return TimeDependentVine, TimeBandwidthFlow

__all__ = [
    # Core classes
    "vine_obj_bin",
    "copula_obj",
    "margin_obj",
    "cop_par_obj",

    # Main functions
    "fit_vine",
    "fit_nonparametric_vine",
    "sample_vine",
    "evaluate_vine",
    "evaluate_nonparametric_vine",
    "create_vine",
    "optimize_vine_type",
    "optimize_vine_structure",

    # Information theory
    "vine_entropy",
    "mutual_information",
    "cond_vine_entropy",

    # Enums
    "VineType",
]
