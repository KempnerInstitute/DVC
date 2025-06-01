"""
Optimization methods for vine copula structure selection.

This module provides advanced optimization algorithms for determining
optimal vine structures using information-theoretic criteria.
"""

from .entropy import EntropyBasedRVineOptimizer

__all__ = ['EntropyBasedRVineOptimizer'] 