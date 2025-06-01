#!/usr/bin/env python3
"""
Entropy-Based R-vine Optimization Demonstration

This script demonstrates the new entropy-based optimization methods
integrated into the comprehensive vine analysis framework.

Run this to see:
1. Traditional Kendall's tau optimization (Classical)
2. Modern entropy-based optimization (Information-theoretic)
3. Sequential greedy optimization (Advanced)
4. Random baseline optimization (Exploration)
5. Fixed structures (C-vine, D-vine)

Author: DVC Analysis Team
Date: 2025
"""

import os
import sys

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the comprehensive analysis framework
from comprehensive_vine_analysis_HS import Comprehensive_Vine_Analyzer

def main():
    """Run focused entropy optimization comparison"""
    
    print("="*80)
    print("ENTROPY-BASED R-VINE OPTIMIZATION DEMONSTRATION")
    print("="*80)
    print("This analysis compares classical correlation-based methods")
    print("with modern information-theoretic approaches for vine structure optimization.")
    print()
    print("METHODS TO BE TESTED:")
    print("1. Classical: Kendall's tau + Prim's MST (Traditional)")
    print("2. Modern: Copula entropy maximization (Innovative)")
    print("3. Advanced: Sequential greedy optimization (Sophisticated)")
    print("4. Baseline: Random structure exploration (Control)")
    print("5. Fixed: C-vine and D-vine structures (Baseline)")
    print("="*80)
    
    # Conservative settings for reliable demonstration
    analyzer = Comprehensive_Vine_Analyzer(
        dim=4,              # 4D for manageable complexity
        n_samples=800,      # Sufficient samples for stable estimation
        timeout_minutes=15  # Reasonable timeout for demonstration
    )
    
    try:
        print("\nStarting comprehensive entropy optimization analysis...")
        print("This will test all optimization methods on multiple data types")
        print("with controlled higher-order interactions.\n")
        
        # Run the complete analysis
        all_results = analyzer.run_comprehensive_analysis()
        
        if all_results:
            print(f"\n🎉 ENTROPY OPTIMIZATION ANALYSIS COMPLETED!")
            print(f"\nResults demonstrate the comparison between:")
            print(f"• Classical tau-based optimization (established method)")
            print(f"• Modern entropy-based optimization (new method)")
            print(f"• Advanced sequential optimization (sophisticated method)")
            print(f"• Random baseline exploration (control method)")
            print(f"• Fixed vine structures (traditional baselines)")
            print(f"\nCheck the results directory for detailed visualizations:")
            print(f"• comprehensive_vine_analysis.png - Method comparison")
            print(f"• entropy_decomposition_analysis.png - Entropy breakdown")
            print(f"• comprehensive_results.json - Detailed numerical results")
            
        else:
            print(f"\n⚠️ Analysis completed but no successful configurations found.")
            print(f"This might indicate parameter tuning or data generation issues.")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Analysis interrupted by user.")
        
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n{'='*80}")
        print("ENTROPY OPTIMIZATION DEMONSTRATION COMPLETE")
        print("="*80)

if __name__ == "__main__":
    main() 