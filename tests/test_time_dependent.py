"""
Tests for time-dependent vine copula modules.

Tests:
- TimeBandwidthFlow: forward shape, positivity
- MLPEdgeFlow: forward shape, positivity
- TimeDependentVineCopula: loss computation
- TimeSeriesVineDataset: dataset iteration
- Synthetic time series generation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from dvc_package.time.flows import TimeBandwidthFlow, MLPEdgeFlow
from dvc_package.time.data import (
    TimeSeriesVineDataset,
    generate_synthetic_time_series,
    create_data_loader,
    preprocess_real_data,
    compute_time_varying_correlations,
)


# ---------------------------------------------------------------------------
# TimeBandwidthFlow tests
# ---------------------------------------------------------------------------

class TestTimeBandwidthFlow:
    """Test TimeBandwidthFlow network."""

    def test_output_shape(self):
        """Output should be (batch, output_dim)."""
        flow = TimeBandwidthFlow(hidden_dim=32, output_dim=2, n_layers=2)
        t = torch.linspace(0, 1, 10).unsqueeze(-1)
        out = flow(t)
        assert out.shape == (10, 2)

    def test_output_positive(self):
        """Bandwidth outputs must be strictly positive."""
        flow = TimeBandwidthFlow(hidden_dim=32, output_dim=4, n_layers=2)
        t = torch.linspace(0, 1, 20).unsqueeze(-1)
        out = flow(t)
        assert (out > 0).all()

    def test_1d_input(self):
        """Should accept 1D time input and unsqueeze automatically."""
        flow = TimeBandwidthFlow(hidden_dim=16, output_dim=2, n_layers=2,
                                  use_batch_norm=False)
        t = torch.linspace(0, 1, 5)
        out = flow(t)
        assert out.shape == (5, 2)

    def test_gradient_flow(self):
        """Gradients should flow through the network."""
        flow = TimeBandwidthFlow(hidden_dim=16, output_dim=2, n_layers=2,
                                  use_batch_norm=False)
        t = torch.linspace(0, 1, 10).unsqueeze(-1)
        out = flow(t)
        loss = out.sum()
        loss.backward()
        # Check at least the first layer has gradients
        for p in flow.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() > 0
                break

    def test_different_activations(self):
        """Network should work with different activation functions."""
        for act in ["relu", "elu", "leaky_relu"]:
            flow = TimeBandwidthFlow(hidden_dim=16, output_dim=2, n_layers=2,
                                      activation=act, use_batch_norm=False)
            t = torch.linspace(0, 1, 5).unsqueeze(-1)
            out = flow(t)
            assert out.shape == (5, 2)
            assert (out > 0).all()


# ---------------------------------------------------------------------------
# MLPEdgeFlow tests
# ---------------------------------------------------------------------------

class TestMLPEdgeFlow:
    """Test MLPEdgeFlow network."""

    def test_output_shape(self):
        """Output should be (batch, out_dim)."""
        flow = MLPEdgeFlow(hidden_dim=32, out_dim=2)
        t = torch.linspace(0, 1, 10).unsqueeze(-1)
        out = flow(t)
        assert out.shape == (10, 2)

    def test_output_positive(self):
        """Bandwidth outputs must be strictly positive."""
        flow = MLPEdgeFlow(hidden_dim=32, out_dim=2)
        t = torch.linspace(0, 1, 20).unsqueeze(-1)
        out = flow(t)
        assert (out > 0).all()

    def test_1d_input(self):
        """Should accept 1D time input."""
        flow = MLPEdgeFlow(hidden_dim=16, out_dim=3)
        t = torch.linspace(0, 1, 8)
        out = flow(t)
        assert out.shape == (8, 3)

    def test_is_nn_module(self):
        """Should be a proper nn.Module."""
        flow = MLPEdgeFlow()
        assert isinstance(flow, nn.Module)
        params = list(flow.parameters())
        assert len(params) > 0


# ---------------------------------------------------------------------------
# Synthetic time series generation
# ---------------------------------------------------------------------------

class TestSyntheticTimeSeries:
    """Test synthetic time series generation."""

    @pytest.mark.parametrize("evolution", ["sinusoidal", "linear", "piecewise", "constant"])
    def test_output_shape(self, evolution):
        """Generated data should have the correct shape."""
        data, time_idx = generate_synthetic_time_series(
            n_time_steps=10, n_samples=50, n_variables=3,
            correlation_evolution=evolution, seed=42
        )
        assert data.shape == (10, 50, 3)
        assert time_idx.shape == (10,)

    def test_reproducible_with_seed(self):
        """Same seed should produce same data."""
        d1, t1 = generate_synthetic_time_series(5, 20, 2, seed=123)
        d2, t2 = generate_synthetic_time_series(5, 20, 2, seed=123)
        np.testing.assert_array_equal(d1, d2)

    def test_time_indices_monotone(self):
        """Time indices should be monotonically increasing."""
        _, time_idx = generate_synthetic_time_series(20, 10, 2, seed=0)
        assert np.all(np.diff(time_idx) > 0)


# ---------------------------------------------------------------------------
# TimeSeriesVineDataset tests
# ---------------------------------------------------------------------------

class TestTimeSeriesVineDataset:
    """Test TimeSeriesVineDataset."""

    def test_length(self):
        """Dataset length should match n_time_steps."""
        data, time_idx = generate_synthetic_time_series(10, 50, 3, seed=42)
        ds = TimeSeriesVineDataset(data, time_idx)
        assert len(ds) == 10

    def test_getitem_returns_tuple(self):
        """__getitem__ should return (data, time) tuple."""
        data, time_idx = generate_synthetic_time_series(10, 50, 3, seed=42)
        ds = TimeSeriesVineDataset(data, time_idx)
        d, t = ds[0]
        assert isinstance(d, torch.Tensor)
        assert isinstance(t, torch.Tensor)

    def test_normalized_time(self):
        """When normalize_time=True, time should be in [0,1]."""
        data, time_idx = generate_synthetic_time_series(10, 50, 3, seed=42)
        ds = TimeSeriesVineDataset(data, time_idx, normalize_time=True)
        for i in range(len(ds)):
            _, t = ds[i]
            assert 0 <= t.item() <= 1

    def test_window_size(self):
        """With window_size, dataset should return windows."""
        data, time_idx = generate_synthetic_time_series(20, 50, 3, seed=42)
        ws = 5
        ds = TimeSeriesVineDataset(data, time_idx, window_size=ws)
        assert len(ds) == 20 - ws + 1
        d, t = ds[0]
        assert d.shape[0] == ws


# ---------------------------------------------------------------------------
# DataLoader creation
# ---------------------------------------------------------------------------

class TestCreateDataLoader:
    """Test DataLoader creation."""

    def test_creates_dataloader(self):
        """create_data_loader should return a working DataLoader."""
        data, time_idx = generate_synthetic_time_series(10, 50, 3, seed=42)
        loader = create_data_loader(data, time_idx, batch_size=4)
        batch = next(iter(loader))
        assert len(batch) == 2  # (data, time)


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessRealData:
    """Test real data preprocessing."""

    def test_standardize(self):
        """Standardized data should have mean ≈ 0 and std ≈ 1."""
        rng = np.random.default_rng(42)
        data = rng.normal(5, 3, (5, 100, 2))
        processed, t = preprocess_real_data(data, standardize=True, remove_outliers=False)
        # Each time-step/variable should be roughly standardized
        for ti in range(5):
            for vi in range(2):
                assert abs(np.mean(processed[ti, :, vi])) < 0.5
                assert abs(np.std(processed[ti, :, vi]) - 1.0) < 0.5

    def test_2d_input(self):
        """2D input should be reshaped to 3D."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 3))
        processed, t = preprocess_real_data(data, standardize=False, remove_outliers=False)
        assert processed.shape == (1, 100, 3)

    def test_outlier_removal(self):
        """Outliers beyond threshold should be replaced."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((1, 200, 2))
        data[0, 0, 0] = 100.0  # Extreme outlier
        processed, _ = preprocess_real_data(data, standardize=False, remove_outliers=True, outlier_threshold=3.0)
        assert processed[0, 0, 0] < 10.0  # Should be clipped


# ---------------------------------------------------------------------------
# Time-varying correlations
# ---------------------------------------------------------------------------

class TestTimeVaryingCorrelations:
    """Test time-varying correlation computation."""

    def test_output_shape(self):
        """Output should be (n_time_steps, n_vars, n_vars)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 100, 3))
        corrs = compute_time_varying_correlations(data)
        assert corrs.shape == (5, 3, 3)

    def test_diagonal_is_one(self):
        """Diagonal of correlation matrix should be 1."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((3, 200, 2))
        corrs = compute_time_varying_correlations(data)
        for t in range(3):
            np.testing.assert_allclose(np.diag(corrs[t]), 1.0, atol=1e-10)

    def test_symmetric(self):
        """Correlation matrices should be symmetric."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((3, 200, 4))
        corrs = compute_time_varying_correlations(data)
        for t in range(3):
            np.testing.assert_allclose(corrs[t], corrs[t].T, atol=1e-10)
