"""
Tests for the full vine copula pipeline: fit → evaluate → sample.

Verifies:
- Fitting produces a valid vine structure
- Evaluation returns proper PDF values
- Sampling produces data with similar correlation structure
"""

import pytest
import torch
import numpy as np
from scipy.stats import norm

from dvc_package.core.vine_factory import create_vine, optimize_vine_type
from dvc_package.core.objects import vine_obj_bin, margin_obj, cop_par_obj
from dvc_package.core.vine_model import fit_vine, evaluate_vine, sample_vine
from dvc_package.core.sampling import vine_cop_par_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_correlated_gaussian(rho_matrix, n=500, seed=42):
    """Generate multivariate normal data from a given correlation matrix."""
    rng = np.random.default_rng(seed)
    d = rho_matrix.shape[0]
    z = rng.multivariate_normal(np.zeros(d), rho_matrix, size=n)
    return z.astype(np.float32)


def _make_default_dicts(families=None):
    """Return the standard dict quartet for fit_vine."""
    if families is None:
        families = ["ind", "gaussian", "clayton"]
    gen_dict = {"param": True, "binning": False, "fitted": False}
    npc_dict = {}
    par_dict = {"param_families": families}
    bin_dict = {}
    return gen_dict, npc_dict, par_dict, bin_dict


# ---------------------------------------------------------------------------
# Fitting tests
# ---------------------------------------------------------------------------

class TestVineFitting:
    """Test vine copula fitting pipeline."""

    def test_fit_3d_cvine(self):
        """Fit a 3-dimensional C-vine and verify structure."""
        rho = np.array([[1, 0.6, 0.3], [0.6, 1, 0.4], [0.3, 0.4, 1]])
        x = _make_correlated_gaussian(rho, n=400)
        vine = create_vine("c-vine", vine_depth=3)
        dicts = _make_default_dicts()
        fit_vine(vine, x, *dicts)

        assert vine.fitted is True
        assert vine.param is True
        assert len(vine.copulas) == 2  # 3D → 2 levels
        assert len(vine.copulas[0]) == 2  # level 0: edges (0,1), (0,2)
        assert len(vine.copulas[1]) == 1  # level 1: edge (1,2|0)

    def test_fit_4d_cvine(self):
        """Fit a 4-dimensional C-vine."""
        rho = np.eye(4)
        for i in range(4):
            for j in range(i + 1, 4):
                rho[i, j] = rho[j, i] = 0.5 / (1 + abs(i - j))
        x = _make_correlated_gaussian(rho, n=500)
        vine = create_vine("c-vine", vine_depth=4)
        dicts = _make_default_dicts()
        fit_vine(vine, x, *dicts)

        assert vine.fitted
        assert len(vine.copulas) == 3  # 4D → 3 levels

    @pytest.mark.parametrize(
        ("vine_type", "create_kwargs"),
        [
            ("d-vine", {"variable_order": [0, 2, 1, 3]}),
            ("r-vine", {}),
        ],
    )
    def test_fit_generic_parametric_vines(self, vine_type, create_kwargs):
        """Fit non-C vine types through the same public parametric pipeline."""
        rho = np.array(
            [
                [1.0, 0.6, 0.4, 0.2],
                [0.6, 1.0, 0.5, 0.3],
                [0.4, 0.5, 1.0, 0.4],
                [0.2, 0.3, 0.4, 1.0],
            ]
        )
        x = _make_correlated_gaussian(rho, n=350)
        vine = create_vine(vine_type, vine_depth=4, **create_kwargs)
        dicts = _make_default_dicts()
        fit_vine(vine, x, *dicts)

        assert vine.fitted is True
        assert vine.param is True
        assert len(vine.copulas) == 3
        assert all(len(level) > 0 for level in vine.copulas)

    def test_copula_families_are_valid(self):
        """All fitted copulas should have a recognized family."""
        rho = np.array([[1, 0.7, 0.3], [0.7, 1, 0.5], [0.3, 0.5, 1]])
        x = _make_correlated_gaussian(rho, n=400)
        vine = create_vine("c-vine", vine_depth=3)
        dicts = _make_default_dicts()
        fit_vine(vine, x, *dicts)

        valid_families = {"ind", "gaussian", "student", "clayton", "claytonrot90",
                          "frank", "gumbel"}
        for level_cops in vine.copulas:
            for cop in level_cops:
                assert cop.family in valid_families, f"Unknown family: {cop.family}"


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestVineEvaluation:
    """Test vine copula evaluation."""

    @pytest.fixture
    def fitted_vine_3d(self):
        """A pre-fitted 3D vine."""
        rho = np.array([[1, 0.6, 0.3], [0.6, 1, 0.4], [0.3, 0.4, 1]])
        x = _make_correlated_gaussian(rho, n=400)
        vine = create_vine("c-vine", vine_depth=3)
        dicts = _make_default_dicts()
        fit_vine(vine, x, *dicts)
        return vine

    def test_evaluation_returns_three_tensors(self, fitted_vine_3d):
        """evaluate_vine should return (pdf, copula_pdf, log_marginals)."""
        pts = torch.randn(50, 3)
        p, p_cop, log_m = evaluate_vine(fitted_vine_3d, pts)
        assert p.shape == (50,)
        assert p_cop.shape == (50,)
        assert log_m.shape == (50,)

    def test_pdf_non_negative(self, fitted_vine_3d):
        """PDF values should be non-negative."""
        pts = torch.randn(100, 3)
        p, _, _ = evaluate_vine(fitted_vine_3d, pts)
        assert (p >= 0).all()

    def test_copula_pdf_non_negative(self, fitted_vine_3d):
        """Copula PDF values should be non-negative."""
        pts = torch.randn(100, 3)
        _, p_cop, _ = evaluate_vine(fitted_vine_3d, pts)
        assert (p_cop >= 0).all()

    def test_pdf_higher_at_mode(self, fitted_vine_3d):
        """PDF near the mode (0,0,0) should be higher than far away."""
        near_mode = torch.zeros(10, 3) + 0.01 * torch.randn(10, 3)
        far_away = torch.ones(10, 3) * 3.0 + 0.01 * torch.randn(10, 3)
        p_near, _, _ = evaluate_vine(fitted_vine_3d, near_mode)
        p_far, _, _ = evaluate_vine(fitted_vine_3d, far_away)
        assert p_near.mean() > p_far.mean()

    @pytest.mark.parametrize(
        ("vine_type", "create_kwargs"),
        [
            ("d-vine", {"variable_order": [0, 2, 1, 3]}),
            ("r-vine", {}),
        ],
    )
    def test_generic_vine_evaluation_is_finite(self, vine_type, create_kwargs):
        """D-vine and R-vine fits should evaluate without NaNs."""
        rho = np.array(
            [
                [1.0, 0.6, 0.4, 0.2],
                [0.6, 1.0, 0.5, 0.3],
                [0.4, 0.5, 1.0, 0.4],
                [0.2, 0.3, 0.4, 1.0],
            ]
        )
        x = _make_correlated_gaussian(rho, n=300, seed=7)
        vine = create_vine(vine_type, vine_depth=4, **create_kwargs)
        dicts = _make_default_dicts()
        fit_vine(vine, x, *dicts)

        pts = torch.tensor(x[:24])
        p, p_cop, log_m = evaluate_vine(vine, pts)
        assert torch.isfinite(p).all()
        assert torch.isfinite(p_cop).all()
        assert torch.isfinite(log_m).all()


# ---------------------------------------------------------------------------
# Sampling tests
# ---------------------------------------------------------------------------

class TestVineSampling:
    """Test vine copula sampling."""

    @pytest.fixture
    def fitted_vine_3d(self):
        """A pre-fitted 3D vine with known correlation."""
        rho = np.array([[1, 0.6, 0.3], [0.6, 1, 0.4], [0.3, 0.4, 1]])
        x = _make_correlated_gaussian(rho, n=500)
        vine = create_vine("c-vine", vine_depth=3)
        dicts = _make_default_dicts()
        fit_vine(vine, x, *dicts)
        return vine, rho

    def test_sample_shape(self, fitted_vine_3d):
        """Samples should have correct shape."""
        vine, _ = fitted_vine_3d
        samples = sample_vine(vine, 200)
        assert samples.shape == (200, 3)

    def test_samples_finite(self, fitted_vine_3d):
        """All samples should be finite."""
        vine, _ = fitted_vine_3d
        samples = sample_vine(vine, 200)
        assert np.all(np.isfinite(samples))

    def test_correlation_recovery(self, fitted_vine_3d):
        """Samples should approximately recover the original correlations."""
        vine, rho_true = fitted_vine_3d
        samples = sample_vine(vine, 2000)
        rho_hat = np.corrcoef(samples.T)
        # Check off-diagonal elements are in the right ballpark
        for i in range(3):
            for j in range(i + 1, 3):
                err = abs(rho_hat[i, j] - rho_true[i, j])
                assert err < 0.6, (
                    f"Correlation ({i},{j}): true={rho_true[i,j]:.2f}, "
                    f"hat={rho_hat[i,j]:.2f}"
                )

    def test_marginal_roughly_normal(self, fitted_vine_3d):
        """Each marginal should be roughly standard normal."""
        vine, _ = fitted_vine_3d
        samples = sample_vine(vine, 1000)
        for j in range(3):
            col = samples[:, j]
            assert abs(np.mean(col)) < 0.5, f"Mean of col {j} = {np.mean(col):.2f}"
            assert 0.3 < np.std(col) < 3.0, f"Std of col {j} = {np.std(col):.2f}"

    @pytest.mark.parametrize(
        ("vine_type", "create_kwargs"),
        [
            ("d-vine", {"variable_order": [0, 2, 1, 3]}),
            ("r-vine", {}),
        ],
    )
    def test_generic_vine_sampling(self, vine_type, create_kwargs):
        """D-vine and R-vine sampling should return finite arrays."""
        rho = np.array(
            [
                [1.0, 0.6, 0.4, 0.2],
                [0.6, 1.0, 0.5, 0.3],
                [0.4, 0.5, 1.0, 0.4],
                [0.2, 0.3, 0.4, 1.0],
            ]
        )
        x = _make_correlated_gaussian(rho, n=300, seed=11)
        vine = create_vine(vine_type, vine_depth=4, **create_kwargs)
        dicts = _make_default_dicts()
        fit_vine(vine, x, *dicts)

        samples = sample_vine(vine, 128)
        assert samples.shape == (128, 4)
        assert np.isfinite(samples).all()

    @pytest.mark.parametrize(
        ("vine_type", "create_kwargs"),
        [
            ("c-vine", {}),
            ("d-vine", {"variable_order": [0, 2, 1, 3]}),
            ("r-vine", {}),
        ],
    )
    def test_fast_parametric_sampler_generates_uniform_copula_draws(self, vine_type, create_kwargs):
        """The direct inverse-Rosenblatt sampler should stay inside (0,1) without collapse."""
        rho = np.array(
            [
                [1.0, 0.6, 0.4, 0.2],
                [0.6, 1.0, 0.5, 0.3],
                [0.4, 0.5, 1.0, 0.4],
                [0.2, 0.3, 0.4, 1.0],
            ]
        )
        x = _make_correlated_gaussian(rho, n=350, seed=17)
        vine = create_vine(vine_type, vine_depth=4, **create_kwargs)
        dicts = _make_default_dicts()
        fit_vine(vine, x, *dicts)

        _samples, u, _, _ = vine_cop_par_sample(vine, 1024)
        assert u.shape == (1024, 4)
        assert np.isfinite(u).all()
        assert np.all((u > 0.0) & (u < 1.0))
        assert np.all(np.abs(u.mean(axis=0) - 0.5) < 0.08)
        assert np.all((u.std(axis=0) > 0.24) & (u.std(axis=0) < 0.33))
        assert np.all(np.mean((u <= 1e-5) | (u >= 1.0 - 1e-5), axis=0) == 0.0)

    def test_fast_parametric_sampler_handles_mixed_family_cvine(self):
        """A manually specified mixed-family C-vine should sample finite values."""
        vine = create_vine("c-vine", vine_depth=3)
        vine.param = True
        vine.fitted = True
        vine.copulas = [
            [cop_par_obj("clayton", 1.2), cop_par_obj("gaussian", 0.45)],
            [cop_par_obj("frank", 2.5)],
        ]

        samples, u, _, _ = vine_cop_par_sample(vine, 512)
        assert samples.shape == (512, 3)
        assert np.isfinite(samples).all()
        assert np.isfinite(u).all()
        assert np.all((u > 0.0) & (u < 1.0))


# ---------------------------------------------------------------------------
# Vine methods on vine_obj_bin
# ---------------------------------------------------------------------------

class TestVineObjectMethods:
    """Test that vine_obj_bin methods work correctly."""

    def test_evaluation_method(self):
        """vine.evaluation() should delegate to evaluate_vine."""
        rho = np.array([[1, 0.5], [0.5, 1]])
        x = _make_correlated_gaussian(rho, n=200)
        vine = create_vine("c-vine", vine_depth=2)
        dicts = _make_default_dicts()
        fit_vine(vine, x, *dicts)

        pts = np.random.randn(30, 2).astype(np.float32)
        p, p_cop, log_m = vine.evaluation(pts)
        assert p.shape == (30,)

    def test_logpdf_method(self):
        """vine.logpdf() should return log-probabilities."""
        rho = np.array([[1, 0.5], [0.5, 1]])
        x = _make_correlated_gaussian(rho, n=200)
        vine = create_vine("c-vine", vine_depth=2)
        dicts = _make_default_dicts()
        fit_vine(vine, x, *dicts)

        pts = torch.randn(30, 2)
        logp = vine.logpdf(pts)
        assert logp.shape == (30,)
        assert torch.isfinite(logp).all()

    def test_sample_method(self):
        """vine.sample() should return numpy array."""
        rho = np.array([[1, 0.5], [0.5, 1]])
        x = _make_correlated_gaussian(rho, n=200)
        vine = create_vine("c-vine", vine_depth=2)
        dicts = _make_default_dicts()
        fit_vine(vine, x, *dicts)

        samples = vine.sample(50)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (50, 2)


class TestVineTypeSelection:
    """Test selecting the best vine family from fitted candidates."""

    def test_optimize_vine_type_returns_fitted_model(self):
        """The vine-type selector should fit and score real candidate models."""
        rho = np.array(
            [
                [1.0, 0.65, 0.45, 0.25],
                [0.65, 1.0, 0.55, 0.35],
                [0.45, 0.55, 1.0, 0.4],
                [0.25, 0.35, 0.4, 1.0],
            ]
        )
        x = _make_correlated_gaussian(rho, n=320, seed=13)
        vine = optimize_vine_type(
            x,
            selection_criterion="aic",
            optimize_structure=True,
            optimization_method="sequential",
            optimization_criterion="kendall_tau",
            par_dict={"param_families": ["ind", "gaussian", "clayton"]},
        )

        assert vine.fitted is True
        assert vine.vine_family in {"c-vine", "d-vine", "r-vine"}
        assert hasattr(vine, "selection_score")
        assert np.isfinite(vine.selection_score)
        pts = torch.tensor(x[:20])
        logp = vine.logpdf(pts)
        assert torch.isfinite(logp).all()
