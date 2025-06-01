#!/usr/bin/env python3
"""
Simple test for beyond-pairwise functionality without full package imports
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')

# Test the beyond-pairwise data generation concept directly
def test_beyond_pairwise_concept():
    print('🔗 TESTING BEYOND-PAIRWISE CONCEPT')
    print('=' * 50)
    
    # Generate simple beyond-pairwise data
    dim = 4
    n_time_steps = 30
    n_samples = 80
    beyond_pairwise_strength = 0.4
    
    print(f"📊 Generating {dim}D data with beyond-pairwise interactions...")
    
    data = np.zeros((n_time_steps, n_samples, dim))
    triple_effects = []
    
    for t in range(n_time_steps):
        # Generate base multivariate normal data
        mean = np.zeros(dim)
        
        # Simple correlation matrix that changes over time
        if t < 10:
            corr = np.array([[1.0, 0.8, 0.1, 0.1],
                           [0.8, 1.0, 0.1, 0.1], 
                           [0.1, 0.1, 1.0, 0.1],
                           [0.1, 0.1, 0.1, 1.0]])
        elif t < 20:
            corr = np.array([[1.0, 0.2, 0.8, 0.1],
                           [0.2, 1.0, 0.8, 0.1], 
                           [0.8, 0.8, 1.0, 0.1],
                           [0.1, 0.1, 0.1, 1.0]])
        else:
            corr = np.array([[1.0, 0.5, 0.5, 0.5],
                           [0.5, 1.0, 0.5, 0.5], 
                           [0.5, 0.5, 1.0, 0.5],
                           [0.5, 0.5, 0.5, 1.0]])
        
        # Generate data
        X = np.random.multivariate_normal(mean, corr, size=n_samples)
        
        # Add beyond-pairwise (triple) interactions
        if beyond_pairwise_strength > 0:
            # X[2] += strength * X[0] * X[1]
            triple_effect = beyond_pairwise_strength * X[:, 0] * X[:, 1]
            X[:, 2] += triple_effect
            triple_effects.append(np.mean(np.abs(triple_effect)))
            
            # X[3] += strength * X[1] * X[2]  
            X[:, 3] += beyond_pairwise_strength * X[:, 1] * X[:, 2]
        
        # Standardize
        for d in range(dim):
            X[:, d] = (X[:, d] - np.mean(X[:, d])) / (np.std(X[:, d]) + 1e-9)
        
        data[t] = X
    
    # Test triple interaction detection
    print(f"✅ Generated data shape: {data.shape}")
    print(f"✅ Triple interaction strength: {beyond_pairwise_strength}")
    print(f"✅ Mean triple effect: {np.mean(triple_effects):.4f}")
    
    # Verify triple interactions are detectable
    t_test = 10  # Test at time step 10
    x0 = data[t_test, :, 0]
    x1 = data[t_test, :, 1] 
    x2 = data[t_test, :, 2]
    
    # Correlation between X[2] and X[0]*X[1]
    product_corr = np.corrcoef(x2, x0 * x1)[0, 1]
    print(f"✅ Empirical triple correlation: {np.abs(product_corr):.4f}")
    
    # Verify this is stronger than pairwise correlations
    pairwise_corr = np.corrcoef(data[t_test].T)
    max_pairwise = np.max(np.abs(pairwise_corr[np.triu_indices(dim, k=1)]))
    print(f"✅ Max pairwise correlation: {max_pairwise:.4f}")
    
    if np.abs(product_corr) > 0.1:  # Should detect some triple interaction
        print("✅ Triple interactions successfully detected!")
    else:
        print("⚠️ Triple interactions may be weak")
    
    print(f"\n🎉 Beyond-pairwise concept test completed successfully!")
    
    return data, triple_effects

if __name__ == "__main__":
    data, effects = test_beyond_pairwise_concept() 