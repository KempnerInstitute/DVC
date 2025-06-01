#!/usr/bin/env python3
"""
Lightweight test version of the multivariate Gaussian vine analysis
- Smaller dimensions and sample sizes for HPC compatibility
- Reduced computational complexity
"""

import os
import sys

# Add the DVC_tensorflow directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
dvc_tensorflow_dir = os.path.join(project_root, 'src', 'DVC_tensorflow')
sys.path.append(dvc_tensorflow_dir)

from multivariate_gaussian_vine_analysis import Multivariate_Gaussian_Vine_Analysis

def main():
    """Lightweight test of the analysis"""
    print("Running LIGHTWEIGHT version of the analysis...")
    print("=" * 50)
    
    # Lightweight configuration
    dimensions = 3           # Reduced from 6
    n_samples = 500         # Reduced from 3000
    vine_type = 'c-vine'    # C-vine is most reliable
    
    # Create analyzer
    analyzer = Multivariate_Gaussian_Vine_Analysis(
        dim=dimensions, 
        n_samples=n_samples, 
        vine_type=vine_type
    )
    
    # Run analysis
    try:
        print("Starting lightweight analysis...")
        results = analyzer.run_full_analysis()
        
        print("\n" + "=" * 50)
        print("✓ LIGHTWEIGHT ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        # Print quick summary
        print(f"Processed {dimensions}D data with {n_samples} samples")
        print(f"Correlation MAE (Vine): {results['correlation_errors']['vine_mae']:.4f}")
        print(f"Entropy Error: {results['entropy']['entropy_error']:.4f} bits")
        
        return results
        
    except Exception as e:
        print(f"\nError during lightweight analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 