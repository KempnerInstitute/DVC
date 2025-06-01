#!/usr/bin/env python3
"""
Test script for advanced simulation scenarios

This script validates all four advanced scenarios:
1. Ising-like model with time-varying couplings
2. Hidden Markov regime switching  
3. Log-linear synergy model with triple interactions
4. Spatiotemporal image blocks

Tests both data generation and visualization capabilities.
"""

import os
import sys
import numpy as np
import time

# Add DVC_NF to path
sys.path.append('.')

# Set non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

from dvc_nf.data.generators import TimeDependentDataGenerator

def test_advanced_scenarios():
    print('🔬 TESTING ADVANCED SIMULATION SCENARIOS')
    print('=' * 60)
    
    # Test configuration
    config = {
        'dim': 4,
        'n_time_steps': 25,      # Reduced for testing
        'n_samples_per_time': 80, # Reduced for testing
        'random_seed': 42
    }
    
    print(f"Test configuration: {config}")
    
    # Initialize generator
    generator = TimeDependentDataGenerator(dim=config['dim'], random_seed=config['random_seed'])
    
    test_results = {}
    
    # 1. TEST ISING-LIKE MODEL
    print(f"\n🧲 Testing Ising-like model...")
    start_time = time.time()
    
    try:
        ising_data, ising_times, ising_metadata = generator.generate_ising_time_series(
            n_time_steps=config['n_time_steps'],
            n_samples_per_time=config['n_samples_per_time'],
            mcmc_sweeps=30  # Reduced for testing
        )
        
        ising_time = time.time() - start_time
        
        # Validate Ising data
        assert ising_data.shape == (config['n_time_steps'], config['n_samples_per_time'], config['dim'])
        assert np.all(np.isin(ising_data, [-1, 1])), "Ising data should be in {-1, +1}"
        assert ising_metadata['type'] == 'ising_time_series'
        assert 'coupling_stats' in ising_metadata
        
        # Test magnetization
        magnetization = np.mean(ising_data, axis=(1, 2))
        magnetization_range = np.max(magnetization) - np.min(magnetization)
        
        test_results['ising'] = {
            'success': True,
            'time': ising_time,
            'data_shape': ising_data.shape,
            'magnetization_range': magnetization_range,
            'coupling_stats': ising_metadata['coupling_stats']
        }
        
        print(f"✅ Ising test passed!")
        print(f"  Generation time: {ising_time:.2f}s")
        print(f"  Magnetization range: {magnetization_range:.3f}")
        print(f"  Coupling statistics: {ising_metadata['coupling_stats']}")
        
        # Test visualization
        generator.visualize_generated_data('ising', save_plots=True)
        print(f"✅ Ising visualization test passed!")
        
    except Exception as e:
        test_results['ising'] = {'success': False, 'error': str(e)}
        print(f"❌ Ising test failed: {e}")
    
    # 2. TEST HIDDEN MARKOV MODEL
    print(f"\n🔄 Testing Hidden Markov Model...")
    start_time = time.time()
    
    try:
        hmm_data, hmm_times, hmm_metadata = generator.generate_hmm_regimes(
            n_time_steps=config['n_time_steps'],
            n_samples_per_time=config['n_samples_per_time'],
            n_regimes=3,
            regime_transition=0.2
        )
        
        hmm_time = time.time() - start_time
        
        # Validate HMM data
        assert hmm_data.shape == (config['n_time_steps'], config['n_samples_per_time'], config['dim'])
        assert hmm_metadata['type'] == 'hmm_regimes'
        assert 'regime_sequence' in hmm_metadata
        assert 'regime_stats' in hmm_metadata
        
        # Test regime dynamics
        regime_sequence = hmm_metadata['regime_sequence']
        unique_regimes = len(np.unique(regime_sequence))
        regime_switches = np.sum(np.diff(regime_sequence) != 0)
        
        test_results['hmm'] = {
            'success': True,
            'time': hmm_time,
            'data_shape': hmm_data.shape,
            'unique_regimes': unique_regimes,
            'regime_switches': regime_switches,
            'regime_stats': hmm_metadata['regime_stats']
        }
        
        print(f"✅ HMM test passed!")
        print(f"  Generation time: {hmm_time:.2f}s")
        print(f"  Unique regimes: {unique_regimes}")
        print(f"  Regime switches: {regime_switches}")
        
        # Test visualization
        generator.visualize_generated_data('hmm', save_plots=True)
        print(f"✅ HMM visualization test passed!")
        
    except Exception as e:
        test_results['hmm'] = {'success': False, 'error': str(e)}
        print(f"❌ HMM test failed: {e}")
    
    # 3. TEST LOG-LINEAR SYNERGY MODEL
    print(f"\n🧠 Testing Log-linear synergy model...")
    start_time = time.time()
    
    try:
        loglinear_data, loglinear_times, loglinear_metadata = generator.generate_loglinear_synergy(
            n_time_steps=config['n_time_steps'],
            n_samples_per_time=config['n_samples_per_time'],
            triple_synergy=True,
            gibbs_sweeps=30  # Reduced for testing
        )
        
        loglinear_time = time.time() - start_time
        
        # Validate log-linear data
        assert loglinear_data.shape == (config['n_time_steps'], config['n_samples_per_time'], config['dim'])
        assert np.all(np.isin(loglinear_data, [0, 1])), "Log-linear data should be in {0, 1}"
        assert loglinear_metadata['type'] == 'loglinear_synergy'
        assert 'synergy_stats' in loglinear_metadata
        
        # Test activity levels
        activity_levels = np.mean(loglinear_data, axis=(1, 2))
        activity_range = np.max(activity_levels) - np.min(activity_levels)
        
        test_results['loglinear'] = {
            'success': True,
            'time': loglinear_time,
            'data_shape': loglinear_data.shape,
            'activity_range': activity_range,
            'synergy_stats': loglinear_metadata['synergy_stats']
        }
        
        print(f"✅ Log-linear test passed!")
        print(f"  Generation time: {loglinear_time:.2f}s")
        print(f"  Activity range: {activity_range:.3f}")
        print(f"  Synergy statistics: {loglinear_metadata['synergy_stats']}")
        
        # Test visualization
        generator.visualize_generated_data('loglinear', save_plots=True)
        print(f"✅ Log-linear visualization test passed!")
        
    except Exception as e:
        test_results['loglinear'] = {'success': False, 'error': str(e)}
        print(f"❌ Log-linear test failed: {e}")
    
    # 4. TEST SPATIOTEMPORAL IMAGE BLOCKS
    print(f"\n🌊 Testing Spatiotemporal image blocks...")
    start_time = time.time()
    
    try:
        spatiotemporal_data, spatiotemporal_times, spatiotemporal_metadata = generator.generate_spatiotemporal_image_blocks(
            height=8, width=8,
            n_time_steps=config['n_time_steps'],
            block_rows=2, block_cols=2,
            n_frames_per_time=config['n_samples_per_time']
        )
        
        spatiotemporal_time = time.time() - start_time
        
        # Validate spatiotemporal data
        expected_blocks = 4  # 2x2 grid
        assert spatiotemporal_data.shape == (config['n_time_steps'], config['n_samples_per_time'], expected_blocks)
        assert spatiotemporal_metadata['type'] == 'spatiotemporal_image_blocks'
        assert 'spatial_stats' in spatiotemporal_metadata
        
        # Test spatial correlations
        spatial_stats = spatiotemporal_metadata['spatial_stats']
        block_correlations = spatial_stats['block_correlations']
        correlation_range = np.max(block_correlations) - np.min(block_correlations)
        
        test_results['spatiotemporal'] = {
            'success': True,
            'time': spatiotemporal_time,
            'data_shape': spatiotemporal_data.shape,
            'n_blocks': spatiotemporal_metadata['n_blocks'],
            'correlation_range': correlation_range
        }
        
        print(f"✅ Spatiotemporal test passed!")
        print(f"  Generation time: {spatiotemporal_time:.2f}s")
        print(f"  Number of blocks: {spatiotemporal_metadata['n_blocks']}")
        print(f"  Correlation range: {correlation_range:.3f}")
        
        # Test visualization
        generator.visualize_generated_data('spatiotemporal', save_plots=True)
        print(f"✅ Spatiotemporal visualization test passed!")
        
    except Exception as e:
        test_results['spatiotemporal'] = {'success': False, 'error': str(e)}
        print(f"❌ Spatiotemporal test failed: {e}")
    
    # SUMMARY
    print(f"\n📊 TEST SUMMARY")
    print('=' * 60)
    
    total_tests = len(test_results)
    successful_tests = sum(1 for result in test_results.values() if result['success'])
    
    print(f"Total tests: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    for scenario, result in test_results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {scenario.upper()}: {status}")
        if result['success']:
            print(f"    • Generation time: {result['time']:.2f}s")
            print(f"    • Data shape: {result['data_shape']}")
    
    # Performance analysis
    if successful_tests > 0:
        print(f"\n⚡ PERFORMANCE ANALYSIS")
        print('-' * 30)
        
        successful_results = {k: v for k, v in test_results.items() if v['success']}
        times = [result['time'] for result in successful_results.values()]
        scenarios = list(successful_results.keys())
        
        fastest_idx = np.argmin(times)
        slowest_idx = np.argmax(times)
        
        print(f"Fastest scenario: {scenarios[fastest_idx].upper()} ({times[fastest_idx]:.2f}s)")
        print(f"Slowest scenario: {scenarios[slowest_idx].upper()} ({times[slowest_idx]:.2f}s)")
        print(f"Average generation time: {np.mean(times):.2f}s")
        print(f"Total generation time: {np.sum(times):.2f}s")
    
    print(f"\n🎉 Advanced scenarios testing completed!")
    
    return test_results

if __name__ == "__main__":
    test_results = test_advanced_scenarios() 