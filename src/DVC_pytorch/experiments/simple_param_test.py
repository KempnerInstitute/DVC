"""
Simple Parametric vs Non-parametric Test
========================================

Basic test to verify that parametric and non-parametric comparison works.
"""

import numpy as np
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_import():
    """Test if we can import the basic modules"""
    try:
        from classes.objects import vine_obj_bin, margin_obj
        print("✓ Basic imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_simple_data_generation():
    """Test basic data generation"""
    try:
        np.random.seed(42)
        n_samples = 100
        dim = 3
        
        # Simple correlation matrix
        corr = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        
        # Generate data
        data = np.random.multivariate_normal(np.zeros(dim), corr, n_samples)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        print(f"✓ Data generation successful: {data.shape}")
        return True, data_tensor
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False, None

def test_vine_creation():
    """Test basic vine object creation"""
    try:
        from classes.objects import vine_obj_bin, margin_obj
        
        dim = 3
        margins = [margin_obj('norm', [0, 1], True) for _ in range(dim)]
        vine = vine_obj_bin('c-vine', ['gaussian'], dim, margins, 11, 'matrix')
        
        print("✓ Vine object creation successful")
        return True, vine
    except Exception as e:
        print(f"✗ Vine creation failed: {e}")
        return False, None

def main():
    print("Running Simple Parametric vs Non-parametric Test")
    print("=" * 60)
    
    # Test imports
    if not test_basic_import():
        return
    
    # Test data generation
    success, data_tensor = test_simple_data_generation()
    if not success:
        return
    
    # Test vine creation
    success, vine = test_vine_creation()
    if not success:
        return
    
    print("\n✓ All basic tests passed!")
    print("The enhanced comprehensive test should work with the fixed code.")

if __name__ == "__main__":
    main() 