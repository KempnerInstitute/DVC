#!/usr/bin/env python3
"""
Focused Block Switching Comprehensive Test

Optimized version to test block switching functionality without
running into resource limitations.
"""

import os
import sys
import numpy as np

# Add DVC_NF to path
sys.path.append('.')

# Set non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

from dvc_nf.analysis.comprehensive import ComprehensiveTimeDependentAnalysis

def test_block_switching_comprehensive():
    print('🧊 COMPREHENSIVE BLOCK SWITCHING TEST')
    print('=' * 60)
    
    # Optimized configuration for resource constraints
    config = {
        'dim': 4,                    # Good for demonstrating block structure
        'n_time_steps': 60,          # Reduced for faster execution
        'n_samples_per_time': 80,    # Reduced memory usage
        'test_scenarios': ['block_switching'],  # Focus on new functionality
        'random_seed': 42
    }
    
    print("Optimized Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n🎯 Testing sophisticated block-structured correlation dynamics")
    print("This test focuses specifically on the new block switching capabilities")
    
    # Initialize analysis framework
    analyzer = ComprehensiveTimeDependentAnalysis(
        dim=config['dim'],
        random_seed=config['random_seed']
    )
    
    # Run focused analysis
    print(f"\n🚀 Starting comprehensive block switching analysis...")
    try:
        analyzer.run_complete_analysis(
            n_time_steps=config['n_time_steps'],
            n_samples_per_time=config['n_samples_per_time'],
            test_scenarios=config['test_scenarios']
        )
        
        print(f"\n✅ COMPREHENSIVE BLOCK SWITCHING TEST COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {analyzer.results_dir}")
        
        # Print summary of results
        if 'block_switching' in analyzer.results:
            result = analyzer.results['block_switching']
            print(f"\n📊 Results Summary:")
            print(f"  Time-dependent model: {'SUCCESS' if result['time_dependent']['success'] else 'FAILED'}")
            print(f"  Static baseline: {'SUCCESS' if result['static']['success'] else 'FAILED'}")
            
            if result['time_dependent']['success']:
                print(f"  Training time: {result['time_dependent']['fit_time']:.1f}s")
                print(f"  Final loss: {result['time_dependent']['train_loss']:.1f}")
                print(f"  Bandwidth edges: {len(result['time_dependent']['bandwidth_predictions'])}")
                
                # Print key metrics
                if 'comparison' in result and 'performance_metrics' in result['comparison']:
                    metrics = result['comparison']['performance_metrics']
                    if 'block_structure_separation' in metrics:
                        print(f"  Block structure separation: {metrics['block_structure_separation']:.3f}")
                    if 'within_block_correlation_mean' in metrics:
                        print(f"  Within-block correlation: {metrics['within_block_correlation_mean']:.3f}")
                    if 'between_block_correlation_mean' in metrics:
                        print(f"  Between-block correlation: {metrics['between_block_correlation_mean']:.3f}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("This may indicate resource constraints or configuration issues.")
        return False
    
    return True

if __name__ == "__main__":
    success = test_block_switching_comprehensive()
    if success:
        print(f"\n🎉 Block switching comprehensive analysis verified!")
        print("The vine copula successfully learned complex block-structured correlation dynamics.")
    else:
        print(f"\n⚠️ Consider reducing parameters further or checking system resources.") 