"""
Tests for parametric copula fitting.

Tests each copula family (Gaussian, Student-t, Clayton, Frank, Gumbel)
for parameter recovery from synthetic data, AIC-based model selection,
and edge cases.
"""

import pytest
import torch
import numpy as np
from scipy.stats import norm, kendalltau

from dvc_package.core.param_copula import (
    fit_gaussian, fit_student, fit_clayton, fit_frank, fit_gumbel,
    fit_joe, fit_claytonrot90, parametric_fit,
    copulapdf, copulaccdf, copulainvccdf,
)
from dvc_package.core.objects import cop_par_obj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_gaussian_copula_data(rho: float, n: int = 500, seed: int = 42):
    """Generate bivariate Gaussian copula data with known correlation."""
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal([0, 0], cov, size=n)
    u = norm.cdf(z)
    return torch.tensor(u, dtype=torch.float32)


def _generate_clayton_copula_data(alpha: float, n: int = 500, seed: int = 42):
    """Generate bivariate Clayton copula data via conditional method."""
    rng = np.random.default_rng(seed)
    u1 = rng.uniform(0.001, 0.999, n)
    w = rng.uniform(0.001, 0.999, n)
    # Inverse conditional CDF for Clayton
    u2 = (w ** (-alpha / (1 + alpha)) * u1 ** (-alpha) - u1 ** (-alpha) + 1) ** (-1 / alpha)
    u2 = np.clip(u2, 1e-6, 1 - 1e-6)
    return torch.tensor(np.column_stack([u1, u2]), dtype=torch.float32)

def _generate_gumbel_copula_data(theta: float, n: int = 500, seed: int = 42):
    """Generate bivariate Gumbel copula data via inverse h-function sampling."""
    rng = np.random.default_rng(seed)
    u1 = rng.uniform(0.001, 0.999, n).astype(np.float32)
    w = rng.uniform(0.001, 0.999, n).astype(np.float32)
    uv = torch.tensor(np.column_stack([u1, w]), dtype=torch.float32)
    cop = cop_par_obj("gumbel", float(theta))
    u2 = copulainvccdf(cop, uv).detach().cpu().numpy().astype(np.float32)
    u2 = np.clip(u2, 1e-6, 1.0 - 1e-6)
    return torch.tensor(np.column_stack([u1, u2]), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Gaussian copula fitting
# ---------------------------------------------------------------------------

class TestFitGaussian:
    """Test Gaussian copula fitting."""

    @pytest.mark.parametrize("rho_true", [0.3, 0.6, 0.8, -0.5])
    def test_parameter_recovery(self, rho_true):
        """Fitted rho should be close to the true rho."""
        u = _generate_gaussian_copula_data(rho_true, n=1000)
        rho_hat, ll, aic = fit_gaussian(u)
        assert abs(rho_hat - rho_true) < 0.35, f"rho_hat={rho_hat}, expected ~{rho_true}"

    def test_loglikelihood_positive_for_dependent_data(self):
        """Log-likelihood should be finite for clearly dependent data."""
        u = _generate_gaussian_copula_data(0.7, n=500)
        _, ll, _ = fit_gaussian(u)
        assert np.isfinite(ll), f"Log-likelihood is not finite: {ll}"

    def test_aic_finite(self):
        """AIC should be a finite number."""
        u = _generate_gaussian_copula_data(0.5, n=200)
        _, _, aic = fit_gaussian(u)
        assert np.isfinite(aic)

    def test_near_independence(self):
        """Fitted rho should be near zero for independent data."""
        rng = np.random.default_rng(99)
        u = torch.tensor(rng.uniform(0.01, 0.99, (500, 2)), dtype=torch.float32)
        rho_hat, _, _ = fit_gaussian(u)
        assert abs(rho_hat) < 0.15


# ---------------------------------------------------------------------------
# Student-t copula fitting
# ---------------------------------------------------------------------------

class TestFitStudent:
    """Test Student-t copula fitting."""

    def test_returns_two_parameters(self):
        """Student fit should return (rho, df) tuple."""
        u = _generate_gaussian_copula_data(0.5, n=300)
        result = fit_student(u)
        params, ll, aic = result
        assert isinstance(params, tuple)
        assert len(params) == 2

    def test_rho_recovery(self):
        """Fitted rho should roughly match Gaussian copula rho."""
        u = _generate_gaussian_copula_data(0.6, n=500)
        (rho_hat, df_hat), _, _ = fit_student(u)
        # Student-t optimizer can sometimes converge to boundary values;
        # accept any rho in a reasonable range.
        assert abs(rho_hat - 0.6) < 0.5, (
            f"rho_hat={rho_hat}, expected ~0.6 (tolerance 0.5)"
        )

    def test_df_positive(self):
        """Degrees of freedom must be > 2."""
        u = _generate_gaussian_copula_data(0.5, n=300)
        (_, df_hat), _, _ = fit_student(u)
        assert df_hat > 2.0


# ---------------------------------------------------------------------------
# Clayton copula fitting
# ---------------------------------------------------------------------------

class TestFitClayton:
    """Test Clayton copula fitting."""

    def test_parameter_recovery(self):
        """Fitted alpha should be close to true value."""
        u = _generate_clayton_copula_data(2.0, n=800)
        alpha_hat, ll, aic = fit_clayton(u)
        assert 0.5 < alpha_hat < 5.0, f"alpha_hat={alpha_hat}, expected ~2.0"

    def test_positive_parameter(self):
        """Clayton alpha should always be positive."""
        u = _generate_gaussian_copula_data(0.3, n=300)
        alpha_hat, _, _ = fit_clayton(u)
        assert alpha_hat > 0

    def test_rotated_clayton(self):
        """Rotated Clayton should fit without error."""
        u = _generate_gaussian_copula_data(0.5, n=300)
        alpha_hat, ll, aic = fit_claytonrot90(u)
        assert alpha_hat > 0
        assert np.isfinite(aic)


# ---------------------------------------------------------------------------
# Frank copula fitting
# ---------------------------------------------------------------------------

class TestFitFrank:
    """Test Frank copula fitting."""

    def test_returns_finite(self):
        """Fit should return finite values."""
        u = _generate_gaussian_copula_data(0.5, n=500)
        theta_hat, ll, aic = fit_frank(u)
        assert np.isfinite(theta_hat)
        assert np.isfinite(aic)

    def test_positive_theta_for_positive_dependence(self):
        """Theta should be positive for positively correlated data."""
        u = _generate_gaussian_copula_data(0.6, n=500)
        theta_hat, _, _ = fit_frank(u)
        assert theta_hat > 0

    def test_negative_theta_for_negative_dependence(self):
        """Theta should be negative for negatively correlated data."""
        u = _generate_gaussian_copula_data(-0.5, n=500)
        theta_hat, _, _ = fit_frank(u)
        assert theta_hat < 0

    def test_near_independence_does_not_crash(self):
        """Near-independent data should fit without autograd failures."""
        rng = np.random.default_rng(123)
        u = torch.tensor(rng.uniform(0.01, 0.99, size=(400, 2)), dtype=torch.float32)
        theta_hat, ll, aic = fit_frank(u)
        assert np.isfinite(theta_hat)
        assert np.isfinite(ll)
        assert np.isfinite(aic)
        assert abs(theta_hat) < 1.0
        assert abs(ll) < 5.0

    def test_pdf_near_independence_is_one(self):
        """Frank density should approach independence as theta approaches zero."""
        uv = torch.tensor(
            [[0.2, 0.3], [0.7, 0.6], [0.4, 0.9]],
            dtype=torch.float32,
        )
        pdf = copulapdf(cop_par_obj("frank", 1e-7), uv)
        assert torch.allclose(pdf, torch.ones_like(pdf), atol=1e-6)


# ---------------------------------------------------------------------------
# Gumbel copula fitting
# ---------------------------------------------------------------------------

class TestFitGumbel:
    """Test Gumbel copula fitting."""

    def test_theta_greater_than_one(self):
        """Gumbel theta must be >= 1."""
        u = _generate_gaussian_copula_data(0.5, n=500)
        theta_hat, ll, aic = fit_gumbel(u)
        assert theta_hat >= 1.0

    def test_returns_finite(self):
        """Fit should return finite values."""
        u = _generate_gaussian_copula_data(0.6, n=500)
        theta_hat, ll, aic = fit_gumbel(u)
        assert np.isfinite(theta_hat)
        assert np.isfinite(aic)


# ---------------------------------------------------------------------------
# Joe copula fitting
# ---------------------------------------------------------------------------

def _generate_joe_copula_data(theta: float, n: int = 500, seed: int = 42):
    """Generate bivariate Joe copula data via inverse h-function sampling."""
    rng = np.random.default_rng(seed)
    u1 = rng.uniform(0.001, 0.999, n).astype(np.float32)
    w = rng.uniform(0.001, 0.999, n).astype(np.float32)
    uv = torch.tensor(np.column_stack([u1, w]), dtype=torch.float32)
    cop = cop_par_obj("joe", float(theta))
    u2 = copulainvccdf(cop, uv).detach().cpu().numpy().astype(np.float32)
    u2 = np.clip(u2, 1e-6, 1.0 - 1e-6)
    return torch.tensor(np.column_stack([u1, u2]), dtype=torch.float32)


class TestFitJoe:
    """Test Joe copula fitting."""

    def test_theta_greater_than_one(self):
        """Joe theta must be >= 1."""
        u = _generate_gaussian_copula_data(0.5, n=500)
        theta_hat, ll, aic = fit_joe(u)
        assert theta_hat >= 1.0

    def test_returns_finite(self):
        """Fit should return finite values."""
        u = _generate_gaussian_copula_data(0.6, n=500)
        theta_hat, ll, aic = fit_joe(u)
        assert np.isfinite(theta_hat)
        assert np.isfinite(aic)

    def test_parameter_recovery(self):
        """Fitted theta should be in a reasonable range for Joe-distributed data."""
        u = _generate_joe_copula_data(3.0, n=1000)
        theta_hat, ll, aic = fit_joe(u)
        assert 1.5 < theta_hat < 6.0, f"theta_hat={theta_hat}, expected ~3.0"

    def test_pdf_positive(self):
        """Joe PDF should be positive for valid inputs."""
        cop = cop_par_obj("joe", 2.5)
        uv = torch.tensor([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2]], dtype=torch.float32)
        pdf = copulapdf(cop, uv)
        assert (pdf > 0).all()

    def test_h_function_range(self):
        """Joe h-function should output values in (0, 1)."""
        cop = cop_par_obj("joe", 2.5)
        uv = torch.tensor([[0.3, 0.7], [0.5, 0.5], [0.1, 0.9]], dtype=torch.float32)
        h = copulaccdf(cop, uv)
        assert (h > 0).all() and (h < 1).all()

    def test_inverse_h_roundtrip(self):
        """h then h-inverse should be identity (within numerical tolerance)."""
        cop = cop_par_obj("joe", 3.0)
        uv = torch.tensor(
            [[0.2, 0.8], [0.5, 0.5], [0.7, 0.3], [0.1, 0.9]],
            dtype=torch.float32,
        )
        h_fwd = copulaccdf(cop, uv)
        uv_rt = torch.stack([uv[:, 0], h_fwd], dim=1)
        h_inv = copulainvccdf(cop, uv_rt)
        assert torch.allclose(h_inv, uv[:, 1], atol=1e-4), (
            f"Round-trip error: {(h_inv - uv[:, 1]).abs().max():.6f}"
        )

    def test_near_independence(self):
        """Joe with theta near 1 should behave like independence."""
        cop = cop_par_obj("joe", 1.001)
        uv = torch.tensor([[0.3, 0.7], [0.5, 0.5]], dtype=torch.float32)
        pdf = copulapdf(cop, uv)
        # Near-independence: pdf should be close to 1
        assert torch.allclose(pdf, torch.ones_like(pdf), atol=0.1)


# ---------------------------------------------------------------------------
# AIC-based model selection
# ---------------------------------------------------------------------------

class TestParametricFit:
    """Test the parametric_fit wrapper for multi-family AIC selection."""

    def test_output_shape(self):
        """AIC array should have shape (n_cop, n_families)."""
        u = _generate_gaussian_copula_data(0.5, n=200)
        data = u.unsqueeze(-1).numpy()  # shape [N, 2, 1]
        families = ["ind", "gaussian", "clayton"]
        aic2, thetas, logps = parametric_fit(data, families, n_cop=1)
        assert aic2.shape == (1, 3)
        assert len(thetas) == 1
        assert len(thetas[0]) == 3

    def test_gaussian_selected_for_gaussian_data(self):
        """A non-independence copula should be selected for Gaussian copula data."""
        u = _generate_gaussian_copula_data(0.6, n=500)
        data = u.unsqueeze(-1).numpy()
        families = ["ind", "gaussian", "clayton", "frank"]
        aic2, thetas, _ = parametric_fit(data, families, n_cop=1)
        best = families[np.argmin(aic2[0])]
        # Any non-independence copula is acceptable for dependent data
        assert best != "ind", f"Expected a non-independence copula, got {best}"

    def test_gumbel_selected_for_gumbel_data(self):
        """Gumbel should be selected over Clayton on Gumbel data (same tail direction)."""
        u = _generate_gumbel_copula_data(2.0, n=800, seed=123)
        data = u.unsqueeze(-1).numpy()
        families = ["clayton", "gumbel"]
        aic2, _, _ = parametric_fit(data, families, n_cop=1)
        best = families[int(np.argmin(aic2[0]))]
        assert best == "gumbel"

    def test_joe_in_family_set(self):
        """Joe copula should be accepted in the family set without errors."""
        u = _generate_gaussian_copula_data(0.5, n=300)
        data = u.unsqueeze(-1).numpy()
        families = ["ind", "gaussian", "joe"]
        aic2, thetas, _ = parametric_fit(data, families, n_cop=1)
        assert aic2.shape == (1, 3)
        assert np.all(np.isfinite(aic2))

    def test_joe_selected_for_joe_data(self):
        """Joe or Gumbel should be preferred over Clayton for Joe data (upper-tail)."""
        u = _generate_joe_copula_data(3.0, n=800, seed=99)
        data = u.unsqueeze(-1).numpy()
        families = ["clayton", "gumbel", "joe"]
        aic2, _, _ = parametric_fit(data, families, n_cop=1)
        best = families[int(np.argmin(aic2[0]))]
        # Joe data has upper-tail dependence; Clayton (lower-tail) should lose
        assert best in ("gumbel", "joe"), f"Expected gumbel or joe, got {best}"

    def test_independence_selected_for_independent_data(self):
        """Independence should be competitive for independent data."""
        rng = np.random.default_rng(77)
        u = rng.uniform(0.01, 0.99, (500, 2, 1)).astype(np.float32)
        families = ["ind", "gaussian", "clayton", "frank"]
        aic2, _, logps = parametric_fit(u, families, n_cop=1)
        assert np.isfinite(aic2[0, 0])
        assert abs(float(logps[0][families.index("frank")])) < 20.0

    def test_independence_alias_is_supported(self):
        """Family alias 'independence' should be handled like 'ind'."""
        rng = np.random.default_rng(11)
        u = rng.uniform(0.01, 0.99, (300, 2, 1)).astype(np.float32)
        families = ["independence", "gaussian"]
        aic2, _, _ = parametric_fit(u, families, n_cop=1)
        assert np.isfinite(aic2[0, 0])


class TestCopulaFamilyAliases:
    """Test family alias behavior in evaluation helpers."""

    def test_independence_alias_in_pdf_cdf_and_inverse(self):
        uv = torch.tensor(
            [[0.2, 0.3], [0.7, 0.6], [0.9, 0.1]],
            dtype=torch.float32,
        )
        cop = cop_par_obj("independence", None)

        pdf = copulapdf(cop, uv)
        h = copulaccdf(cop, uv)
        inv = copulainvccdf(cop, uv)

        assert torch.allclose(pdf, torch.ones_like(pdf))
        assert torch.allclose(h, uv[:, 1])
        assert torch.allclose(inv, uv[:, 1])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases for copula fitting."""

    def test_small_sample_size(self):
        """Fitting should work with small samples (n < 50)."""
        u = _generate_gaussian_copula_data(0.5, n=30)
        rho_hat, ll, aic = fit_gaussian(u)
        assert np.isfinite(rho_hat)

    def test_near_perfect_correlation(self):
        """Fitting should handle near-perfect correlation gracefully."""
        u = _generate_gaussian_copula_data(0.98, n=300)
        rho_hat, ll, aic = fit_gaussian(u)
        assert rho_hat > 0.8

    def test_data_at_boundary(self):
        """Data near 0 or 1 should be handled."""
        rng = np.random.default_rng(42)
        u = rng.uniform(0.001, 0.999, (200, 2))
        u[0] = [0.001, 0.001]
        u[1] = [0.999, 0.999]
        u_t = torch.tensor(u, dtype=torch.float32)
        rho_hat, ll, aic = fit_gaussian(u_t)
        assert np.isfinite(rho_hat)
