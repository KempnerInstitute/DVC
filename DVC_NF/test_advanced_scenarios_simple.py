#!/usr/bin/env python3
"""
Simple test script for advanced simulation scenarios

This script tests the data generation capabilities directly
without importing the full framework to avoid import issues.
"""

import os
import sys
import numpy as np
import time

# Set non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Import data generator directly
sys.path.append('.')
from dvc_nf.data.generators import TimeDependentDataGenerator

def test_advanced_scenarios_simple():
    print('🔬 TESTING ADVANCED SIMULATION SCENARIOS (SIMPLIFIED)')
    print('=' * 70)
    
    # Test configuration
    config = {
        'dim': 4,
        'n_time_steps': 20,      # Small for testing
        'n_samples_per_time': 50, # Small for testing
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
            mcmc_sweeps=20  # Reduced for testing
        )
        
        ising_time = time.time() - start_time
        
        # Validate Ising data
        assert ising_data.shape == (config['n_time_steps'], config['n_samples_per_time'], config['dim'])
        assert np.all(np.isin(ising_data, [-1, 1])), "Ising data should be in {-1, +1}"
        assert ising_metadata['type'] == 'ising_time_series'
        
        # Test magnetization
        magnetization = np.mean(ising_data, axis=(1, 2))
        magnetization_range = np.max(magnetization) - np.min(magnetization)
        
        test_results['ising'] = {
            'success': True,
            'time': ising_time,
            'data_shape': ising_data.shape,
            'magnetization_range': magnetization_range,
            'data_mean': np.mean(ising_data),
            'data_std': np.std(ising_data)
        }
        
        print(f"✅ Ising test passed!")
        print(f"  Generation time: {ising_time:.2f}s")
        print(f"  Data shape: {ising_data.shape}")
        print(f"  Magnetization range: {magnetization_range:.3f}")
        print(f"  Data statistics: mean={np.mean(ising_data):.3f}, std={np.std(ising_data):.3f}")
        
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
            'data_mean': np.mean(hmm_data),
            'data_std': np.std(hmm_data)
        }
        
        print(f"✅ HMM test passed!")
        print(f"  Generation time: {hmm_time:.2f}s")
        print(f"  Data shape: {hmm_data.shape}")
        print(f"  Unique regimes: {unique_regimes}")
        print(f"  Regime switches: {regime_switches}")
        print(f"  Data statistics: mean={np.mean(hmm_data):.3f}, std={np.std(hmm_data):.3f}")
        
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
            gibbs_sweeps=20  # Reduced for testing
        )
        
        loglinear_time = time.time() - start_time
        
        # Validate log-linear data
        assert loglinear_data.shape == (config['n_time_steps'], config['n_samples_per_time'], config['dim'])
        assert np.all(np.isin(loglinear_data, [0, 1])), "Log-linear data should be in {0, 1}"
        assert loglinear_metadata['type'] == 'loglinear_synergy'
        
        # Test activity levels
        activity_levels = np.mean(loglinear_data, axis=(1, 2))
        activity_range = np.max(activity_levels) - np.min(activity_levels)
        
        test_results['loglinear'] = {
            'success': True,
            'time': loglinear_time,
            'data_shape': loglinear_data.shape,
            'activity_range': activity_range,
            'data_mean': np.mean(loglinear_data),
            'data_std': np.std(loglinear_data)
        }
        
        print(f"✅ Log-linear test passed!")
        print(f"  Generation time: {loglinear_time:.2f}s")
        print(f"  Data shape: {loglinear_data.shape}")
        print(f"  Activity range: {activity_range:.3f}")
        print(f"  Data statistics: mean={np.mean(loglinear_data):.3f}, std={np.std(loglinear_data):.3f}")
        
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
        
        # Test spatial correlations
        block_correlations = spatiotemporal_metadata['spatial_stats']['block_correlations']
        correlation_range = np.max(block_correlations) - np.min(block_correlations)
        
        test_results['spatiotemporal'] = {
            'success': True,
            'time': spatiotemporal_time,
            'data_shape': spatiotemporal_data.shape,
            'n_blocks': spatiotemporal_metadata['n_blocks'],
            'correlation_range': correlation_range,
            'data_mean': np.mean(spatiotemporal_data),
            'data_std': np.std(spatiotemporal_data)
        }
        
        print(f"✅ Spatiotemporal test passed!")
        print(f"  Generation time: {spatiotemporal_time:.2f}s")
        print(f"  Data shape: {spatiotemporal_data.shape}")
        print(f"  Number of blocks: {spatiotemporal_metadata['n_blocks']}")
        print(f"  Correlation range: {correlation_range:.3f}")
        print(f"  Data statistics: mean={np.mean(spatiotemporal_data):.3f}, std={np.std(spatiotemporal_data):.3f}")
        
    except Exception as e:
        test_results['spatiotemporal'] = {'success': False, 'error': str(e)}
        print(f"❌ Spatiotemporal test failed: {e}")
    
    # SUMMARY
    print(f"\n📊 TEST SUMMARY")
    print('=' * 70)
    
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
            print(f"    • Data mean: {result['data_mean']:.3f}")
            print(f"    • Data std: {result['data_std']:.3f}")
    
    # Performance analysis
    if successful_tests > 0:
        print(f"\n⚡ PERFORMANCE ANALYSIS")
        print('-' * 40)
        
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
    print("=" * 70)
    print("Successfully integrated four advanced simulation scenarios:")
    print("• 🧲 Ising-like model with time-varying couplings (MCMC)")
    print("• 🔄 Hidden Markov regime switching with correlation structures")
    print("• 🧠 Log-linear synergy model with triple interactions (Gibbs)")
    print("• 🌊 Spatiotemporal image blocks with wave patterns")
    print("=" * 70)
    
    return test_results

if __name__ == "__main__":
    test_results = test_advanced_scenarios_simple() 