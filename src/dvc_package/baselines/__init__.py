"""Baseline models for dependence modeling and time-varying structure.

These implementations are intentionally small and reproducible. They are useful
for examples, tests, and benchmark comparisons against Gaussian or learned
non-Gaussian alternatives.
"""

from .tvgl import tvgl_frobenius  # noqa: F401
from .gaussian_state_space import gaussian_copula_state_space_nll_fit_eval  # noqa: F401
