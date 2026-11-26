"""Dynamic Vine Copulas.

A Python package for vine copula modeling, multivariate dependence estimation,
and time-indexed copula diagnostics.

Key Features:
- Multiple vine types: C-vine, D-vine, R-vine with structure optimization
- Parametric copula families: Gaussian, Student-t, Clayton, Frank, Gumbel, Joe
- Nonparametric local-likelihood copula estimation
- Time-dependent C-vine estimators and normalizing-flow helpers
- Monte Carlo entropy and mutual information estimation

Main modules:
- core: Complete vine copula implementation
- time: Time-dependent vine-copula models and flow helpers
- optimization: Vine structure optimization algorithms
- cli: Command-line interface tools
- experiments: Experiment runners and templates
- utils: Shared utilities and helpers
"""

__version__ = "0.1.0"

# Resolve the established public API when accessed so the NumPy modulation
# estimator can also be used without initializing the PyTorch vine backend.
from importlib import import_module

_CORE = {
    **dict.fromkeys(("vine_obj_bin", "copula_obj", "margin_obj", "cop_par_obj"), "objects"),
    **dict.fromkeys(("fit_vine", "sample_vine", "evaluate_vine"), "vine_model"),
    **dict.fromkeys(("fit_nonparametric_vine", "evaluate_nonparametric_vine"), "nonparametric_vine"),
    **dict.fromkeys(("vine_entropy", "mutual_information", "cond_vine_entropy"), "info_estimation"),
    **dict.fromkeys(("create_vine", "VineType", "optimize_vine_type"), "vine_factory"),
}


def __getattr__(name):
    if name not in _CORE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    symbol = getattr(import_module(f".core.{_CORE[name]}", __name__), name)
    globals()[name] = symbol
    return symbol


def __dir__():
    return sorted(set(globals()) | set(_CORE))


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
