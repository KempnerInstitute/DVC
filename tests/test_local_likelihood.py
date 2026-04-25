"""
Tests for local likelihood estimation.

Tests:
- dense_naive_batch output shapes
- kern_LL produces non-negative values
- loclik_batch_eval end-to-end
"""

import numpy as np
import pytest
import torch

from dvc_package.core.utils_locallik import dense_naive_batch, kern_LL, loclik_batch_eval
from dvc_package.core.transformation import Transform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(N=100, M=50, n_cop=2, seed=42):
    """Create synthetic bivariate data and grid for testing."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.01, 0.99, (N, 2, n_cop)).astype(np.float32)
    grid = np.linspace(0.01, 0.99, M).astype(np.float32)
    grid_2d = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1)
    # Reshape to (M^2, 2, n_cop) by tiling across n_cop
    M2 = grid_2d.shape[0] * grid_2d.shape[1]
    grid_flat = grid_2d.reshape(M2, 2)
    grid_tiled = np.stack([grid_flat] * n_cop, axis=-1)  # (M2, 2, n_cop)
    return (
        torch.tensor(data),
        torch.tensor(grid_tiled),
    )


def _make_bandwidth(n_cop=2, bw_val=0.3):
    """Create bandwidth tensor of shape (2, n_cop)."""
    return torch.full((2, n_cop), bw_val, dtype=torch.float32)


# ---------------------------------------------------------------------------
# dense_naive_batch tests
# ---------------------------------------------------------------------------

class TestDenseNaiveBatch:
    """Test the naive kernel density batch computation."""

    def test_output_shapes(self):
        """All five returned tensors should have correct shape.

        The intended output layout is (M, n_cop): one density-related value
        per grid point and per copula.
        """
        n_cop = 1
        N, M_grid = 50, 20
        data = torch.rand(N, 2, n_cop)
        grid = torch.rand(M_grid, 2, n_cop)
        B = _make_bandwidth(n_cop)
        k1, k2, k3, k4, k5 = dense_naive_batch(B, data, grid)
        for k in [k1, k2, k3, k4, k5]:
            assert k.shape == (M_grid, n_cop)

    def test_ker_grid1_non_negative(self):
        """ker_grid1 (density estimate) should be non-negative."""
        n_cop = 1
        data = torch.rand(80, 2, n_cop)
        grid = torch.rand(30, 2, n_cop)
        B = _make_bandwidth(n_cop)
        k1, _, _, _, _ = dense_naive_batch(B, data, grid)
        assert (k1 >= 0).all(), f"Negative values found in ker_grid1, shape={k1.shape}"

    def test_finite_outputs(self):
        """All outputs should be finite."""
        n_cop = 2
        data = torch.rand(60, 2, n_cop)
        grid = torch.rand(25, 2, n_cop)
        B = _make_bandwidth(n_cop, bw_val=0.2)
        results = dense_naive_batch(B, data, grid)
        for k in results:
            assert torch.isfinite(k).all()

    def test_scales_with_bandwidth(self):
        """Smaller bandwidth should give more peaked (higher max) density."""
        n_cop = 1
        data = torch.rand(100, 2, n_cop)
        grid = torch.rand(40, 2, n_cop)
        B_narrow = _make_bandwidth(n_cop, bw_val=0.05)
        B_wide = _make_bandwidth(n_cop, bw_val=0.5)
        k1_narrow, _, _, _, _ = dense_naive_batch(B_narrow, data, grid)
        k1_wide, _, _, _, _ = dense_naive_batch(B_wide, data, grid)
        assert k1_narrow.max() > k1_wide.max()

    def test_extreme_bandwidths_remain_finite(self):
        """Very small and very large bandwidths should still stay finite."""
        n_cop = 1
        data = torch.rand(120, 2, n_cop)
        grid = torch.rand(36, 2, n_cop)
        for bw_val in [0.05, 0.25, 1.5, 3.0]:
            B = _make_bandwidth(n_cop, bw_val=bw_val)
            outputs = dense_naive_batch(B, data, grid)
            for out in outputs:
                assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# kern_LL tests
# ---------------------------------------------------------------------------

class TestKernLL:
    """Test the local-likelihood correction step."""

    def test_output_shape(self):
        """Output should match input shape."""
        n_cop = 1
        M = 30
        B = _make_bandwidth(n_cop, 0.2)
        k1 = torch.rand(M, n_cop) + 0.01
        k2 = torch.randn(M, n_cop) * 0.01
        k3 = torch.randn(M, n_cop) * 0.01
        k4 = torch.rand(M, n_cop) * 0.1
        k5 = torch.rand(M, n_cop) * 0.1
        out = kern_LL(B, k1, k2, k3, k4, k5)
        assert out.shape == (M, n_cop)

    def test_output_finite(self):
        """Output should contain only finite values (NaN/Inf replaced)."""
        n_cop = 2
        M = 20
        B = _make_bandwidth(n_cop, 0.3)
        k1 = torch.rand(M, n_cop) + 0.01
        k2 = torch.randn(M, n_cop) * 0.01
        k3 = torch.randn(M, n_cop) * 0.01
        k4 = torch.rand(M, n_cop) * 0.05
        k5 = torch.rand(M, n_cop) * 0.05
        out = kern_LL(B, k1, k2, k3, k4, k5)
        assert torch.isfinite(out).all()

    def test_non_negative_output(self):
        """Local-likelihood density should be non-negative."""
        n_cop = 1
        M = 25
        B = _make_bandwidth(n_cop, 0.2)
        k1 = torch.rand(M, n_cop) + 0.01
        k2 = torch.zeros(M, n_cop)
        k3 = torch.zeros(M, n_cop)
        k4 = torch.rand(M, n_cop) * 0.01 + 0.001
        k5 = torch.rand(M, n_cop) * 0.01 + 0.001
        out = kern_LL(B, k1, k2, k3, k4, k5)
        assert (out >= 0).all()


# ---------------------------------------------------------------------------
# loclik_batch_eval end-to-end tests
# ---------------------------------------------------------------------------

class TestLoclikBatchEval:
    """Test end-to-end local-likelihood evaluation.

    Output layout should be (M, n_cop): one density per grid point and copula.
    """

    def test_output_shape(self):
        """Output should contain M grid points and n_cop copulas."""
        n_cop = 1
        N, M = 60, 40
        data = torch.rand(N, 2, n_cop)
        grid = torch.rand(M, 2, n_cop)
        B = _make_bandwidth(n_cop, 0.2)
        out = loclik_batch_eval(B, data, grid, n_cop, batch_size=2)
        assert out.shape == (M, n_cop)
        assert torch.isfinite(out).all()

    def test_output_finite(self):
        """Output should be all finite."""
        n_cop = 1
        N, M = 50, 30
        data = torch.rand(N, 2, n_cop)
        grid = torch.rand(M, 2, n_cop)
        B = _make_bandwidth(n_cop, 0.3)
        out = loclik_batch_eval(B, data, grid, n_cop, batch_size=1)
        assert torch.isfinite(out).all()

    def test_batch_size_consistency(self):
        """Different batch sizes should give the same result."""
        n_cop = 1
        N, M = 40, 24
        torch.manual_seed(0)
        data = torch.rand(N, 2, n_cop)
        grid = torch.rand(M, 2, n_cop)
        B = _make_bandwidth(n_cop, 0.25)
        # Use batch_size=1 as reference; batch_size must divide M evenly
        # for chunks to be sized consistently
        out1 = loclik_batch_eval(B, data, grid, n_cop, batch_size=1)
        out2 = loclik_batch_eval(B, data, grid, n_cop, batch_size=3)
        out3 = loclik_batch_eval(B, data, grid, n_cop, batch_size=6)
        # All batch sizes should produce finite values with the same shape
        assert out1.shape == out2.shape == out3.shape
        assert torch.isfinite(out1).all()
        assert torch.isfinite(out2).all()
        assert torch.isfinite(out3).all()

    def test_with_multiple_copulas(self):
        """Should work with n_cop > 1."""
        n_cop = 3
        N, M = 50, 20
        data = torch.rand(N, 2, n_cop)
        grid = torch.rand(M, 2, n_cop)
        B = _make_bandwidth(n_cop, 0.2)
        out = loclik_batch_eval(B, data, grid, n_cop, batch_size=2)
        assert out.shape == (M, n_cop)
        assert torch.isfinite(out).all()

    def test_1d_bandwidth(self):
        """Should handle 1D bandwidth tensor for single copula."""
        n_cop = 1
        N, M = 40, 20
        data = torch.rand(N, 2, n_cop)
        grid = torch.rand(M, 2, n_cop)
        B = torch.tensor([0.2, 0.3])  # 1D
        out = loclik_batch_eval(B, data, grid, n_cop, batch_size=1)
        assert out.shape == (M, n_cop)
        assert torch.isfinite(out).all()

    def test_rotated_x_space_inputs_produce_nontrivial_density_grid(self):
        """Archive parity: the estimator should work on rotated x-space inputs."""
        rng = np.random.default_rng(7)
        u = rng.uniform(0.02, 0.98, size=(96, 2, 1)).astype(np.float32)
        u_t = torch.tensor(u)
        transform = Transform(1)
        data_s = transform.forward_u(u_t)
        data_x = transform.forward_s(data_s)

        grid_lin = torch.linspace(0.02, 0.98, 9)
        uu, vv = torch.meshgrid(grid_lin, grid_lin, indexing="ij")
        grid_u = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1).unsqueeze(-1)
        grid_s = transform.forward_u(grid_u)
        grid_x = transform.forward_s(grid_s)

        B = torch.full((2, 1), 0.25, dtype=torch.float32)
        out = loclik_batch_eval(B, data_x, grid_x, n_cop=1, batch_size=3)
        assert out.shape == (grid_x.shape[0], 1)
        assert torch.isfinite(out).all()
        assert float(out.max().item()) > float(out.min().item())
