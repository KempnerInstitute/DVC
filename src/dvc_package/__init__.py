"""
DVC (Distributed Vine Copula) Package

A comprehensive Python library for vine copula modeling, multivariate dependency estimation,
and time-dependent entropy analysis.

Key Features:
- Multiple vine types: C-vine, D-vine, R-vine with optimization
- Parametric and nonparametric copula estimation  
- Time-dependent dependency modeling using normalizing flows
- Entropy and mutual information estimation
- Professional CLI tools and experiment runners

Main modules:
- core: Complete vine copula implementation
- time: Time-dependent modeling with normalizing flows
- optimization: Vine structure optimization algorithms
- cli: Command-line interface tools
- experiments: Experiment runners and templates
- utils: Shared utilities and helpers
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes and functions for easy access
from .core.objects import vine_obj_bin, copula_obj, margin_obj, cop_par_obj
from .core.vine_model import fit_vine, sample_vine, evaluate_vine
from .core.info_estimation import vine_entropy, mutual_information, cond_vine_entropy

# Import vine factory functions
from .core.vine_factory import create_vine, VineType

# Import optimization functions
from .optimization.structure import optimize_vine_structure

# Import time-dependent classes
from .time.models import TimeDependentVine, TimeBandwidthFlow

__all__ = [
    # Core classes
    "vine_obj_bin",
    "copula_obj", 
    "margin_obj",
    "cop_par_obj",
    
    # Main functions
    "fit_vine",
    "sample_vine", 
    "evaluate_vine",
    "create_vine",
    
    # Information theory
    "vine_entropy",
    "mutual_information",
    "cond_vine_entropy",
    
    # Optimization
    "optimize_vine_structure",
    
    # Time-dependent
    "TimeDependentVine",
    "TimeBandwidthFlow",
    
    # Enums
    "VineType",
]