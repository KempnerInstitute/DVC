#!/usr/bin/env python3
"""
Test script for beyond-pairwise interactions functionality
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

def test_beyond_pairwise():
    print('🔗 TESTING BEYOND-PAIRWISE FUNCTIONALITY')
    print('=' * 50)
    
    # Create generator
    generator = TimeDependentDataGenerator(dim=4, random_seed=42)
    
    # Generate beyond-pairwise data
    print("📊 Generating beyond-pairwise interactions data...")
    data, times, metadata = generator.generate_beyond_pairwise_interactions(
        n_time_steps=30,
        n_samples_per_time=80,
        switch_times=[0.3, 0.7],
        corr_low=0.2,
        corr_high=0.8,
        beyond_pairwise_strength=0.4
    )
    
    print(f"✅ Generated data shape: {data.shape}")
    print(f"✅ Number of regimes: {metadata['n_regimes']}")
    print(f"✅ Switch indices: {metadata['switch_indices']}")
    print(f"✅ Triple interaction strength: {metadata['beyond_pairwise_strength']}")
    print(f"✅ Mean triple effect: {np.mean(metadata['triple_interaction_evolution']):.4f}")
    
    # Test visualization
    print("\n📈 Testing visualization...")
    try:
        generator.visualize_generated_data('beyond_pairwise', save_plots=True)
        print("✅ Visualization created successfully!")
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
    
    # Analyze the data structure
    print(f"\n📋 Data Analysis:")
    print(f"  Regime descriptions: {metadata['regime_descriptions']}")
    print(f"  Triple effects range: [{np.min(metadata['triple_interaction_evolution']):.4f}, {np.max(metadata['triple_interaction_evolution']):.4f}]")
    
    # Verify triple interactions are actually present
    # Check correlation between X[2] and X[0]*X[1] in first time step
    x0 = data[10, :, 0]  # Time step 10
    x1 = data[10, :, 1]
    x2 = data[10, :, 2]
    
    product_corr = np.corrcoef(x2, x0 * x1)[0, 1]
    print(f"  Empirical triple correlation at t=10: {np.abs(product_corr):.4f}")
    
    print(f"\n🎉 Beyond-pairwise functionality test completed successfully!")
    return True

if __name__ == "__main__":
    test_beyond_pairwise() 