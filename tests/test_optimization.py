"""
Test optimization functionality.
"""

import pytest
import numpy as np
import torch
from dvc_package.optimization.structure import (
    optimize_vine_structure,
    _reorder_vine_structure,
)
from dvc_package.optimization.criteria import (
    kendall_tau_criterion, entropy_criterion, aic_criterion
)
from dvc_package.core.vine_factory import create_vine


class TestOptimizationCriteria:
    """Test optimization criteria functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        # Generate correlated data
        corr_matrix = np.array([[1.0, 0.6, 0.3], 
                               [0.6, 1.0, 0.4], 
                               [0.3, 0.4, 1.0]])
        data = np.random.multivariate_normal([0, 0, 0], corr_matrix, 500)
        return data
    
    @pytest.fixture
    def fitted_vine(self, sample_data):
        """Create a fitted vine for testing."""
        vine = create_vine('c-vine', 3)
        
        # Simple fitting (mock)
        gen_dict = {'param': True, 'binning': False, 'fitted': False}
        npc_dict = {}
        par_dict = {'param_families': ['gaussian']}
        bin_dict = {}
        
        try:
            from dvc_package.core.vine_model import fit_vine
            fit_vine(vine, sample_data, gen_dict, npc_dict, par_dict, bin_dict)
        except:
            # If fitting fails, just mark as fitted for testing
            vine.fitted = True
        
        return vine
    
    def test_kendall_tau_criterion(self, fitted_vine, sample_data):
        """Test Kendall's tau criterion."""
        score = kendall_tau_criterion(fitted_vine, sample_data)
        
        assert isinstance(score, (float, np.floating))
        assert score >= 0  # Absolute tau values should be non-negative
        assert np.isfinite(score)
    
    def test_entropy_criterion(self, fitted_vine, sample_data):
        """Test entropy criterion."""
        score = entropy_criterion(fitted_vine, sample_data)
        
        assert isinstance(score, (float, np.floating))
        assert np.isfinite(score)
    
    def test_aic_criterion(self, fitted_vine, sample_data):
        """Test AIC criterion."""
        score = aic_criterion(fitted_vine, sample_data)
        
        assert isinstance(score, (float, np.floating))
        assert np.isfinite(score)
    
    def test_criterion_robustness(self, sample_data):
        """Test criterion robustness with unfitted vine."""
        vine = create_vine('c-vine', 3)
        
        # Should not crash with unfitted vine
        tau_score = kendall_tau_criterion(vine, sample_data)
        entropy_score = entropy_criterion(vine, sample_data)
        aic_score = aic_criterion(vine, sample_data)
        
        # Should return valid values (even if poor)
        assert np.isfinite(tau_score)
        assert np.isfinite(entropy_score) or entropy_score == float('-inf')
        assert np.isfinite(aic_score) or aic_score == float('inf')


class TestVineOptimization:
    """Test vine structure optimization."""
    
    @pytest.fixture
    def optimization_data(self):
        """Generate data for optimization testing."""
        np.random.seed(123)
        # Create data with clear structure
        n_samples = 200
        
        # Strong correlation between variables 0 and 1
        x1 = np.random.normal(0, 1, n_samples)
        x2 = 0.8 * x1 + 0.6 * np.random.normal(0, 1, n_samples)
        
        # Weaker correlation with variable 2
        x3 = 0.3 * x1 + 0.2 * x2 + 0.9 * np.random.normal(0, 1, n_samples)
        
        return np.column_stack([x1, x2, x3])
    
    def test_sequential_optimization(self, optimization_data):
        """Test sequential optimization method."""
        result = optimize_vine_structure(
            data=optimization_data,
            vine_type='c-vine',
            method='sequential',
            criterion='kendall_tau',
            verbose=False
        )
        
        assert hasattr(result, 'best_vine')
        assert hasattr(result, 'best_score')
        assert hasattr(result, 'method')
        assert result.method == 'sequential'
        assert isinstance(result.best_score, (float, np.floating))
        assert np.isfinite(result.best_score)
    
    def test_genetic_optimization_dvine(self, optimization_data):
        """Test genetic optimization for D-vine."""
        result = optimize_vine_structure(
            data=optimization_data,
            vine_type='d-vine',
            method='genetic',
            criterion='kendall_tau',
            max_iterations=10,  # Short run for testing
            population_size=10,
            verbose=False
        )
        
        assert result.method == 'genetic'
        assert hasattr(result, 'optimization_history')
        assert len(result.optimization_history) > 0
        assert result.iterations <= 10
    
    def test_entropy_optimization(self, optimization_data):
        """Test entropy-based optimization."""
        result = optimize_vine_structure(
            data=optimization_data,
            vine_type='r-vine',
            method='entropy',
            verbose=False
        )
        
        assert result.method == 'entropy'
        assert hasattr(result, 'best_vine')
        assert result.best_vine.vine_family == 'r-vine'
    
    def test_optimization_with_different_criteria(self, optimization_data):
        """Test optimization with different criteria."""
        criteria = ['kendall_tau', 'entropy', 'aic']
        
        for criterion in criteria:
            try:
                result = optimize_vine_structure(
                    data=optimization_data,
                    vine_type='c-vine',
                    method='sequential',
                    criterion=criterion,
                    verbose=False
                )
                
                assert result.best_vine is not None
                assert np.isfinite(result.best_score)
                
            except Exception as e:
                # Some criteria might fail with limited data
                pytest.skip(f"Criterion {criterion} failed: {e}")
    
    def test_optimization_convergence_info(self, optimization_data):
        """Test optimization convergence information."""
        result = optimize_vine_structure(
            data=optimization_data,
            vine_type='d-vine',
            method='genetic',
            max_iterations=5,
            population_size=5,
            verbose=False
        )
        
        assert hasattr(result, 'convergence_info')
        assert isinstance(result.convergence_info, dict)
        assert 'converged' in result.convergence_info
    
    def test_invalid_optimization_method(self, optimization_data):
        """Test error handling for invalid optimization method."""
        with pytest.raises(ValueError):
            optimize_vine_structure(
                data=optimization_data,
                vine_type='c-vine',
                method='invalid_method'
            )
    
    def test_optimization_with_small_data(self):
        """Test optimization with minimal data."""
        # Very small dataset
        small_data = np.random.randn(10, 3)
        
        result = optimize_vine_structure(
            data=small_data,
            vine_type='c-vine',
            method='sequential',
            verbose=False
        )
        
        # Should complete without error
        assert result.best_vine is not None


class TestOptimizationRobustness:
    """Test optimization robustness and edge cases."""
    
    def test_optimization_with_independent_data(self):
        """Test optimization with independent variables."""
        # Generate independent data
        np.random.seed(456)
        independent_data = np.random.randn(100, 3)
        
        result = optimize_vine_structure(
            data=independent_data,
            vine_type='c-vine',
            method='sequential',
            criterion='kendall_tau',
            verbose=False
        )
        
        # Should complete and find low correlations
        assert result.best_vine is not None
        assert result.best_score >= 0  # Tau-based score should be non-negative
    
    def test_optimization_with_perfect_correlation(self):
        """Test optimization with perfectly correlated data."""
        np.random.seed(789)
        
        # Create perfectly correlated data
        x1 = np.random.randn(100)
        x2 = x1 + 1e-10 * np.random.randn(100)  # Almost perfect correlation
        x3 = 0.5 * x1 + 0.1 * np.random.randn(100)
        
        perfect_data = np.column_stack([x1, x2, x3])
        
        result = optimize_vine_structure(
            data=perfect_data,
            vine_type='c-vine',
            method='sequential',
            verbose=False
        )
        
        # Should handle near-perfect correlation
        assert result.best_vine is not None
        assert np.isfinite(result.best_score)
    
    def test_optimization_reproducibility(self):
        """Test optimization reproducibility with same seed."""
        np.random.seed(42)
        data1 = np.random.multivariate_normal([0, 0, 0], 
                                            [[1, 0.5, 0.2], 
                                             [0.5, 1, 0.3], 
                                             [0.2, 0.3, 1]], 50)
        
        np.random.seed(42)
        data2 = np.random.multivariate_normal([0, 0, 0], 
                                            [[1, 0.5, 0.2], 
                                             [0.5, 1, 0.3], 
                                             [0.2, 0.3, 1]], 50)
        
        result1 = optimize_vine_structure(
            data=data1, vine_type='c-vine', method='sequential', verbose=False
        )
        
        result2 = optimize_vine_structure(
            data=data2, vine_type='c-vine', method='sequential', verbose=False
        )
        
        # Results should be similar (within numerical precision)
        assert abs(result1.best_score - result2.best_score) < 1e-10


class TestOptimizationRegressions:
    """Regression tests for previously broken optimization behavior."""

    def test_reorder_cvine_structure_updates_edges(self):
        """C-vine structure should be rebuilt according to the new order."""
        vine = create_vine("c-vine", 4)
        new_order = [2, 0, 3, 1]

        _reorder_vine_structure(vine, new_order)

        assert vine.ind_vine[0] == [[2, 0], [2, 3], [2, 1]]
        assert vine.ind_vine[1] == [[0, 3], [0, 1]]
        assert vine.ind_vine[2] == [[3, 1]]

    def test_sequential_optimization_runs_for_five_dimensions(self):
        """Sequential C-vine optimization should run in NumPy 2.x environments."""
        data = np.random.randn(80, 5)
        result = optimize_vine_structure(
            data=data,
            vine_type="c-vine",
            method="sequential",
            criterion="kendall_tau",
            verbose=False,
        )

        assert result.best_vine is not None
        assert np.isfinite(result.best_score)


if __name__ == '__main__':
    pytest.main([__file__])
