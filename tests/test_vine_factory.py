"""
Test vine factory functionality.
"""

import pytest
import numpy as np
from dvc_package.core.vine_factory import create_vine, VineType, CVine, DVine, RVine
from dvc_package.core.objects import margin_obj


class TestVineFactory:
    """Test vine factory functions."""
    
    def test_create_cvine(self):
        """Test C-vine creation."""
        vine = create_vine('c-vine', vine_depth=3)
        
        assert vine.vine_family == 'c-vine'
        assert vine.n_cop == 3
        assert len(vine.margin) == 3
        assert len(vine.ind_vine) == 2  # 3D vine has 2 tree levels
        
        # Check C-vine structure (star topology)
        level_0_edges = vine.ind_vine[0]
        expected_edges_0 = [[0, 1], [0, 2]]  # Center node 0
        assert level_0_edges == expected_edges_0
    
    def test_create_dvine(self):
        """Test D-vine creation."""
        vine = create_vine('d-vine', vine_depth=4)
        
        assert vine.vine_family == 'd-vine'
        assert vine.n_cop == 4
        assert len(vine.ind_vine) == 3  # 4D vine has 3 tree levels
        
        # D-vine should have path structure
        level_0_edges = vine.ind_vine[0]
        assert len(level_0_edges) == 3  # 4 variables -> 3 edges in first level
    
    def test_create_rvine(self):
        """Test R-vine creation."""
        vine = create_vine('r-vine', vine_depth=3)
        
        assert vine.vine_family == 'r-vine'
        assert vine.n_cop == 3
        assert hasattr(vine, 'r_matrix')
        assert vine.r_matrix.shape == (3, 3)
        
        # Check diagonal elements
        for i in range(3):
            assert vine.r_matrix[i, i] == i + 1
    
    def test_create_with_custom_margins(self):
        """Test vine creation with custom margins."""
        vine = create_vine(
            'c-vine', 
            vine_depth=2,
            margin_types=['norm', 'uniform']
        )
        
        assert len(vine.margin) == 2
        assert vine.margin[0].dist == 'norm'
        assert vine.margin[1].dist == 'uniform'
    
    def test_create_with_custom_families(self):
        """Test vine creation with custom copula families."""
        families = ['gaussian', 'clayton']
        vine = create_vine('c-vine', vine_depth=3, families=families)
        
        assert vine.families == families
    
    def test_enum_vine_type(self):
        """Test using VineType enum."""
        vine = create_vine(VineType.C_VINE, vine_depth=3)
        assert vine.vine_family == 'c-vine'
        
        vine = create_vine(VineType.D_VINE, vine_depth=3)
        assert vine.vine_family == 'd-vine'
        
        vine = create_vine(VineType.R_VINE, vine_depth=3)
        assert vine.vine_family == 'r-vine'
    
    def test_invalid_vine_type(self):
        """Test error handling for invalid vine type."""
        with pytest.raises(ValueError):
            create_vine('invalid-vine', vine_depth=3)
    
    def test_cvine_class(self):
        """Test CVine class directly."""
        margins = [margin_obj('norm', (0.0, 1.0)) for _ in range(3)]
        vine = CVine(['gaussian'], 3, margins, 30)
        
        assert isinstance(vine, CVine)
        assert vine.vine_family == 'c-vine'
        
        # Check structure was built
        assert len(vine.ind_vine) == 2
    
    def test_dvine_class(self):
        """Test DVine class directly."""
        margins = [margin_obj('norm', (0.0, 1.0)) for _ in range(4)]
        vine = DVine(['gaussian'], 4, margins, 30, variable_order=[0, 2, 1, 3])
        
        assert isinstance(vine, DVine)
        assert vine.vine_family == 'd-vine'
        assert vine.variable_order == [0, 2, 1, 3]
    
    def test_rvine_class(self):
        """Test RVine class directly."""
        margins = [margin_obj('norm', (0.0, 1.0)) for _ in range(3)]
        
        # Test with custom R-matrix
        r_matrix = np.array([[1, 1, 1], [0, 2, 2], [0, 0, 3]])
        vine = RVine(['gaussian'], 3, margins, 30, r_matrix=r_matrix)
        
        assert isinstance(vine, RVine)
        assert vine.vine_family == 'r-vine'
        np.testing.assert_array_equal(vine.r_matrix, r_matrix)
    
    def test_rvine_random_matrix(self):
        """Test R-vine with random matrix generation."""
        margins = [margin_obj('norm', (0.0, 1.0)) for _ in range(4)]
        vine = RVine(['gaussian'], 4, margins, 30)
        
        # Check that a valid R-matrix was generated
        assert vine.r_matrix.shape == (4, 4)
        
        # Check diagonal
        for i in range(4):
            assert vine.r_matrix[i, i] == i + 1
    
    def test_dvine_variable_order(self):
        """Test D-vine with specific variable ordering."""
        vine = create_vine('d-vine', vine_depth=4, variable_order=[3, 1, 0, 2])
        
        assert hasattr(vine, 'variable_order')
        assert vine.variable_order == [3, 1, 0, 2]


class TestVineStructures:
    """Test vine structure generation."""
    
    def test_cvine_structure_3d(self):
        """Test 3D C-vine structure."""
        vine = create_vine('c-vine', 3)
        
        # Level 0: (0,1), (0,2) - center is 0
        # Level 1: (1,2|0) - remaining pair conditioned on 0
        
        assert len(vine.ind_vine) == 2
        assert vine.ind_vine[0] == [[0, 1], [0, 2]]
        assert vine.ind_vine[1] == [[1, 2]]
    
    def test_dvine_structure_4d(self):
        """Test 4D D-vine structure."""
        vine = create_vine('d-vine', 4, variable_order=[0, 1, 2, 3])
        
        # Level 0: (0,1), (1,2), (2,3) - path structure
        # Level 1: (0,2|1), (1,3|2) - adjacent pairs
        # Level 2: (0,3|1,2) - final pair
        
        assert len(vine.ind_vine) == 3
        
        # Check first level has path structure
        level_0 = vine.ind_vine[0]
        assert len(level_0) == 3
    
    def test_rvine_structure_consistency(self):
        """Test R-vine structure consistency."""
        vine = create_vine('r-vine', 4)
        
        # Check that structure was generated from R-matrix
        assert len(vine.ind_vine) == 3  # 4D vine has 3 levels
        
        # Each level should have decreasing number of edges
        edge_counts = [len(level) for level in vine.ind_vine]
        assert edge_counts == [3, 2, 1]  # 4D: 3, 2, 1 edges per level


if __name__ == '__main__':
    pytest.main([__file__])
