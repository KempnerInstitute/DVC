#!/usr/bin/env python3
"""
Simple test script to verify the multivariate Gaussian vine analysis works
"""

import sys
import os

# Add the DVC_tensorflow directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
dvc_tensorflow_dir = os.path.join(project_root, 'src', 'DVC_tensorflow')
sys.path.append(dvc_tensorflow_dir)

try:
    from multivariate_gaussian_vine_analysis import Multivariate_Gaussian_Vine_Analysis
    print("✓ Script imports successfully")
    
    # Test basic initialization
    analyzer = Multivariate_Gaussian_Vine_Analysis(dim=3, n_samples=100, vine_type='c-vine')
    print("✓ Class initialization works")
    
    # Test data generation
    data = analyzer.simulate_multivariate_gaussian()
    print(f"✓ Data generation works, shape: {data.shape}")
    
    print("✓ All basic tests passed!")
    print("The main script should work correctly.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc() 