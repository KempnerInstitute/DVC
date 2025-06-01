#!/usr/bin/env python3
"""
Quick test of block switching functionality
"""

import os
import sys
import numpy as np

# Add DVC_NF to path
sys.path.append('.')

# Set non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

from dvc_nf.data.generators import TimeDependentDataGenerator

def test_block_switching():
    print('🧊 BLOCK SWITCHING FUNCTIONALITY TEST')
    print('=' * 50)
    
    # Create generator
    generator = TimeDependentDataGenerator(dim=4, random_seed=42)
    
    # Generate block switching data
    print("📊 Generating block-structured switching data...")
    data, times, metadata = generator.generate_block_switching_correlation_data(
        n_time_steps=30,
        n_samples_per_time=80,
        n_regimes=3,
        switch_probability=0.1
    )
    
    print(f"✅ Generated data shape: {data.shape}")
    print(f"✅ Block structure: {metadata['block_sizes']}")
    print(f"✅ Block indices: {metadata['block_indices']}")
    print(f"✅ Number of regimes: {metadata['n_regimes']}")
    print(f"✅ Regime switches: {np.sum(np.diff(metadata['regime_sequence']) != 0)}")
    print(f"✅ Entropy range: [{np.nanmin(metadata['entropy_evolution']):.3f}, {np.nanmax(metadata['entropy_evolution']):.3f}] bits")
    
    # Test visualization
    print("\n📈 Testing visualization...")
    try:
        generator.visualize_generated_data('block_switching', save_plots=True)
        print("✅ Visualization created successfully!")
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
    
    # Show regime information
    print(f"\n📋 Regime Analysis:")
    unique_regimes, counts = np.unique(metadata['regime_sequence'], return_counts=True)
    for regime, count in zip(unique_regimes, counts):
        print(f"  Regime {regime}: {count} time steps ({count/len(metadata['regime_sequence'])*100:.1f}%)")
    
    # Show entropy statistics by regime
    print(f"\n📊 Entropy by Regime:")
    for regime in unique_regimes:
        regime_mask = metadata['regime_sequence'] == regime
        regime_entropies = np.array(metadata['entropy_evolution'])[regime_mask]
        avg_entropy = np.nanmean(regime_entropies)
        print(f"  Regime {regime}: {avg_entropy:.3f} ± {np.nanstd(regime_entropies):.3f} bits")
    
    print(f"\n🎉 Block switching functionality test completed successfully!")
    return True

if __name__ == "__main__":
    test_block_switching() 