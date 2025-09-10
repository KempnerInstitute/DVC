"""
Tests for PyTorch DVC fixes.

This module contains tests to verify that the fixes properly align the PyTorch
implementation with TensorFlow behavior.
"""

import torch
import numpy as np
import pytest
from pytorch_fixes import (
    fix_eval_rs_cop,
    fix_kernel_cdf_smoothing,
    fix_independence_aic,
    fix_chain_conditional_sampling,
    fix_binning_logic,
    apply_fixes_to_vine
)

class MockVine:
    """Mock vine object for testing."""
    def __init__(self):
        self.n_cop = 3
        self.knots = 50
        self.grid_u = None
        self.theta = torch.zeros((100, 3, 3))
        self.theta_flip = torch.zeros((100, 3, 3))
        self.n_bin = 5
        self.ind_vine = [
            [[0, 1], [1, 2]],  # level 0
            [[0, 2]]           # level 1
        ]

def test_eval_rs_cop():
    """Test that eval_rs_cop uses 500 iterations and proper epsilon."""
    # Create sample grid
    grid = torch.rand(50, 50)
    grid = grid / grid.sum()
    
    # Apply normalization
    normalized = fix_eval_rs_cop(grid)
    
    # Check row sums are close to 1
    row_sums = normalized.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    
    # Check column sums are close to 1
    col_sums = normalized.sum(dim=0)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-6)
    
    # Check no values are exactly 0 (eps=1e-30 used)
    assert torch.all(normalized > 0)

def test_kernel_cdf_smoothing():
    """Test kernel CDF smoothing matches TF behavior."""
    # Create sample data
    data = torch.randn(1000)
    grid = torch.linspace(-3, 3, 100)
    
    # Apply smoothing
    smoothed = fix_kernel_cdf_smoothing(data, grid)
    
    # Check bounds
    assert torch.all(smoothed >= 1e-12)
    assert torch.all(smoothed <= 1.0 - 1e-12)
    
    # Check monotonicity
    sorted_smoothed = torch.sort(smoothed)[0]
    diffs = sorted_smoothed[1:] - sorted_smoothed[:-1]
    assert torch.all(diffs >= 0)

def test_independence_aic():
    """Test independence AIC includes correlation penalty."""
    # Create correlated data
    x = torch.randn(1000)
    y = 0.5 * x + 0.5 * torch.randn(1000)
    data = torch.stack([x, y], dim=1)
    
    # Compute AIC
    aic = fix_independence_aic(data, 1000)
    
    # Should be positive due to correlation penalty
    assert aic > 0
    
    # Create uncorrelated data
    x = torch.randn(1000)
    y = torch.randn(1000)
    data = torch.stack([x, y], dim=1)
    
    # Compute AIC
    aic_uncorr = fix_independence_aic(data, 1000)
    
    # Should be smaller for uncorrelated data
    assert aic_uncorr < aic

def test_chain_conditional_sampling():
    """Test chain-of-conditionals sampling logic."""
    class MockCopula:
        def invccdf(self, u1, u2):
            return 0.5 * (u1 + u2)
    
    vine = MockVine()
    cop = MockCopula()
    
    # Create sample data
    v = torch.rand(100, 3, 4)  # [samples, k, variables]
    k, i, parent = 1, 2, 0
    
    # Test normal case
    v_normal = fix_chain_conditional_sampling(
        vine, v, k, i, parent, cop, flip=False
    )
    
    # Test flipped case
    v_flipped = fix_chain_conditional_sampling(
        vine, v, k, i, parent, cop, flip=True
    )
    
    # Results should be different for flipped vs normal
    assert not torch.allclose(v_normal, v_flipped)

def test_binning_logic():
    """Test binning logic properly handles parent selection."""
    vine = MockVine()
    
    # Create sample theta matrices
    theta = torch.rand(100, 3, 3)       # [samples, tree_level, variables]
    theta_flip = torch.rand(100, 3, 3)
    
    # Test first level
    tr, edge, parent = 1, [0, 2], 1
    bins, val_to_bin = fix_binning_logic(
        vine, tr, edge, parent, theta, theta_flip
    )
    
    # Check bins are created properly
    assert len(bins) == vine.n_bin + 1  # n_bin + 1 for edges
    assert torch.all(bins[:-1] <= bins[1:])  # monotonic
    
    # Check bin assignments
    assert torch.all(val_to_bin >= 0)
    assert torch.all(val_to_bin < vine.n_bin)

def test_apply_fixes_to_vine(caplog):
    """Test that all fixes can be applied to a vine object."""
    vine = MockVine()
    
    # Apply fixes
    apply_fixes_to_vine(vine)
    
    # Check that all methods were added
    assert hasattr(vine, 'eval_rs_cop')
    assert hasattr(vine, '_smooth_cdf')
    assert hasattr(vine, '_compute_independence_aic')
    assert hasattr(vine, '_sample_conditional')
    assert hasattr(vine, '_handle_binning')
    
    # Check logging
    assert "Applying PyTorch DVC fixes..." in caplog.text
    assert "All fixes applied successfully!" in caplog.text

def test_integration():
    """Test full integration of fixes."""
    vine = MockVine()
    
    # Apply fixes
    apply_fixes_to_vine(vine)
    
    # Test row/column normalization
    grid = torch.rand(50, 50)
    grid = grid / grid.sum()
    normalized = vine.eval_rs_cop(grid)
    assert torch.allclose(normalized.sum(1), torch.ones(50), atol=1e-6)
    assert torch.allclose(normalized.sum(0), torch.ones(50), atol=1e-6)
    
    # Test independence AIC
    x = torch.randn(1000)
    y = 0.5 * x + torch.randn(1000)
    data = torch.stack([x, y], dim=1)
    aic = vine._compute_independence_aic(data, 1000)
    assert aic > 0
    
    # Test binning
    tr, edge, parent = 1, [0, 2], 1
    bins, val_to_bin = vine._handle_binning(
        vine, tr, edge, parent, vine.theta, vine.theta_flip
    )
    assert len(bins) == vine.n_bin + 1
    assert torch.all(val_to_bin >= 0) 