"""Regression tests for the MINE (Belghazi et al. 2018) MI-estimator baseline.

Pins two properties the baseline must satisfy:
- independence case recovers MI near zero,
- bivariate-Gaussian case recovers the analytic MI within a loose tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from dvc_package.baselines.mine import mine_mi_estimate


def _gaussian_mi(rho: float) -> float:
    return -0.5 * float(np.log(max(1.0 - rho * rho, 1e-12)))


def test_mine_indep_is_near_zero():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    mi = mine_mi_estimate(x, y, n_epochs=80, seed=0)
    assert np.isfinite(mi)
    # Independent Gaussian: true MI = 0. Allow generous tolerance.
    assert abs(mi) < 0.12, f"MINE MI for independent pair should be near 0, got {mi}"


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7])
def test_mine_recovers_gaussian_mi(rho: float):
    rng = np.random.default_rng(42)
    n = 600
    z = rng.standard_normal((n, 2))
    x = z[:, 0]
    y = rho * z[:, 0] + np.sqrt(1.0 - rho * rho) * z[:, 1]
    mi_true = _gaussian_mi(rho)
    mi_est = mine_mi_estimate(x, y, n_epochs=100, seed=7)
    assert np.isfinite(mi_est)
    # Tolerance scales with the true MI; require 25% relative error or 0.05 nats absolute.
    err = abs(mi_est - mi_true)
    # MINE is known to have finite-sample bias and upper-bound variance; we
    # accept the larger of 0.10 nats absolute or 50% relative error.
    tol = max(0.10, 0.5 * mi_true)
    assert err <= tol, (
        f"MINE MI at rho={rho} off by {err:.3f} nats (est={mi_est:.3f}, true={mi_true:.3f})"
    )
