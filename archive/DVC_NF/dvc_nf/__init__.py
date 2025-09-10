"""
DVC-NF: Deep Vine Copulas with Normalizing Flows

A comprehensive framework for time-dependent vine copula modeling using 
normalizing flows for dynamic bandwidth estimation.

Author: DVC Analysis Team
Date: 2025
"""

__version__ = "1.0.0"
__author__ = "DVC Analysis Team"

# Core imports
from .core.flows import TimeBandwidthFlow, TimeDependentVineCopula
from .data.generators import TimeDependentDataGenerator
from .analysis.comprehensive import ComprehensiveTimeDependentAnalysis
from .optimization.entropy import EntropyBasedRVineOptimizer

__all__ = [
    'TimeBandwidthFlow',
    'TimeDependentVineCopula', 
    'TimeDependentDataGenerator',
    'ComprehensiveTimeDependentAnalysis',
    'EntropyBasedRVineOptimizer'
] 