"""
Quick test to verify basic DVC PyTorch functionality
"""

import torch
import numpy as np
from scipy import stats

# Import core modules
from classes.objects import vine_obj_bin, margin_obj
from grid.grid_op import create_grids

print("DVC PyTorch Quick Test")
print("=" * 50)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Generate simple test data
n_samples = 500
n_dims = 3

# Create correlated data
mean = np.zeros(n_dims)
cov = np.eye(n_dims)
cov[0, 1] = cov[1, 0] = 0.7
cov[1, 2] = cov[2, 1] = 0.5
data = np.random.multivariate_normal(mean, cov, n_samples)

# Convert to uniform margins
data_uniform = np.zeros_like(data)
for i in range(n_dims):
    data_uniform[:, i] = stats.rankdata(data[:, i]) / (n_samples + 1)

# Convert to PyTorch
data_torch = torch.tensor(data_uniform, dtype=torch.float32, device=device)
print(f"\nData shape: {data_torch.shape}")

# Create margins
margins = [margin_obj('empirical', None, True) for _ in range(n_dims)]

# Create vine copula
print("\nCreating C-vine copula...")
vine = vine_obj_bin(
    vine_family='c-vine',
    families=['gaussian'],
    vine_depth=n_dims - 1,
    margin=margins,
    knots=16,
    method=None
)

# Create grids
vine.grid_u, vine.grid_s, vine.grid_x = create_grids(vine.knots, device=device)

# Set up parameters
gen_dict = {
    'binning': False,
    'parallel': False,
    'param': True,
    'vine_depth': n_dims - 1
}

par_dict = {
    'param_families': ['gaussian', 'clayton', 'ind']
}

bin_dict = {
    'n_bin': 1
}

# Fit vine
print("Fitting vine copula...")
try:
    vine.fit(data_torch, gen_dict, {}, par_dict, bin_dict)
    print("✓ Fitting successful!")
    
    # Test evaluation
    test_data = data_torch[:10]
    print(f"\nEvaluating on test data with shape: {test_data.shape}")
    print(f"Training data shape was: {data_torch.shape}")
    p, p_cop, logp = vine.evaluation(test_data)
    print(f"✓ Evaluation successful! Mean log-likelihood: {logp.mean().item():.3f}")
    
    # Check structure
    print(f"\nVine structure:")
    print(f"- Family: {vine.vine_family}")
    print(f"- R-matrix shape: {vine.r_matrix.shape}")
    print(f"- Number of trees: {len(vine.copulas)}")
    
    # Check selected families
    if vine.copulas:
        print(f"\nSelected copula families:")
        for i, tree in enumerate(vine.copulas):
            families = [cop.family for cop in tree]
            print(f"  Tree {i}: {families}")
    
    print("\n" + "=" * 50)
    print("Quick test passed! ✓")
    print("=" * 50)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc() 