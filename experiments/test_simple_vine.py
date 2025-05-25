"""
Simple test for D-vine with 2 dimensions to debug NaN issue
"""

import numpy as np
import sys
sys.path.append('src')

from DVC import vine_obj_bin, margin_obj, fit_vine

# Set random seed
np.random.seed(42)

# Generate simple 2D correlated data
n_samples = 200
rho = 0.5
cov_matrix = np.array([[1.0, rho], [rho, 1.0]])
data = np.random.multivariate_normal([0, 0], cov_matrix, n_samples).astype(np.float32)

print(f"Data shape: {data.shape}")
print(f"Data correlation: {np.corrcoef(data.T)[0, 1]:.3f}")

# Create a simple D-vine
vine = vine_obj_bin(
    vine_family='d-vine',
    families=['gaussian', 'ind'],
    vine_depth=2,
    margin=[],
    knots=50
)

# Set margins
for i in range(2):
    vine.margin.append(margin_obj('norm', [0, 1], True))

# Configuration
gen_dict = {"parallel": False, "param": True, "binning": False}
par_dict = {"param_families": ["gaussian", "ind"]}
npc_dict = {"method": "local", "n_iter": 100}
bin_dict = {"n_bin": 1}

# Fit the vine
print("\nFitting vine...")
fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
print("Vine fitted!")

# Check what copulas were fitted
print("\nFitted copulas:")
for level_idx, level_copulas in enumerate(vine.copulas):
    print(f"  Level {level_idx}:")
    for edge_idx, cop in enumerate(level_copulas):
        edge = vine.ind_vine[level_idx][edge_idx] if edge_idx < len(vine.ind_vine[level_idx]) else None
        if edge and hasattr(cop, 'family'):  # Parametric
            print(f"    Edge {edge}: family={cop.family}, theta={cop.theta}")

# Test sampling step by step
print("\nDebugging D-vine sampling...")
print("D-vine structure should sample U1, then U2|U1")

# Try manual sampling
import torch
from DVC.vine_model import _hfunc_param

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nsamples = 5

# Step 1: Sample U1
U1 = torch.rand(nsamples, device=device)
print(f"\nStep 1 - U1 (uniform): {U1.cpu().numpy()}")

# Step 2: Sample U2|U1
V2 = torch.rand(nsamples, device=device)
print(f"Step 2 - V2 (uniform): {V2.cpu().numpy()}")

# Get the copula for edge (0,1)
cop = vine.copulas[0][0]
print(f"Copula for edge (0,1): family={cop.family}, theta={cop.theta}")

# Apply h-function
U2 = _hfunc_param(U1, V2, cop, inverse=True)
print(f"Step 2 - U2|U1 (after h-inverse): {U2.cpu().numpy()}")

# Check for NaN
if torch.any(torch.isnan(U2)):
    print("WARNING: U2 contains NaN!")
else:
    print("U2 looks good (no NaN)")

# Now try the full sampling
print("\nTrying full vine sampling...")
try:
    samples = vine.sample(10)
    print(f"Generated samples shape: {samples.shape}")
    
    # Check for NaN
    if np.any(np.isnan(samples)):
        print("WARNING: Samples contain NaN values!")
        print(f"NaN locations: {np.argwhere(np.isnan(samples))}")
        print(f"Sample values:\n{samples}")
    else:
        print("Samples look good (no NaN)")
        print(f"Sample mean: {samples.mean(axis=0)}")
        print(f"Sample std: {samples.std(axis=0)}")
        print(f"Sample correlation: {np.corrcoef(samples.T)[0, 1]:.3f}")
        
except Exception as e:
    print(f"Error generating samples: {e}")
    import traceback
    traceback.print_exc() 