"""Baselines for dependency modeling and time-varying structure.

This package contains simple, reproducible baselines used in the NeurIPS draft
experiments (e.g., TVGL-like Gaussian structure learning).
"""

from .tvgl import tvgl_frobenius  # noqa: F401
from .gaussian_state_space import gaussian_copula_state_space_nll_fit_eval  # noqa: F401
