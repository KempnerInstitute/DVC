"""
Test for 5D vine to debug where NaN appears
"""

import numpy as np
import sys
sys.path.append('src')

from DVC_pyolder import vine_obj_bin, margin_obj, fit_vine

# Set random seed
np.random.seed(42)

# Generate 5D correlated data (same as comprehensive test)
dim = 5
A = np.random.randn(dim, dim)
cov_matrix = np.dot(A.T, A)  # Ensure positive definite
# Normalize to correlation matrix with unit variance
D = np.sqrt(np.diag(cov_matrix))
cov_matrix = cov_matrix / np.outer(D, D)

n_samples = 500
data = np.random.multivariate_normal(np.zeros(dim), cov_matrix, n_samples).astype(np.float32)

print(f"Data shape: {data.shape}")
print(f"Condition number of covariance: {np.linalg.cond(cov_matrix):.2f}")

# Create a D-vine
vine = vine_obj_bin(
    vine_family='d-vine',
    families=['gaussian', 'ind'],
    vine_depth=dim,
    margin=[],
    knots=50
)

# Set margins
for i in range(dim):
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
print("\nDebugging D-vine sampling step by step...")

import torch
from DVC_pyolder.vine_model import _hfunc_param

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nsamples = 5

# Manually simulate the sampling process
U = torch.empty(nsamples, dim, dtype=torch.float32, device=device)
U[:, 0] = torch.rand(nsamples, device=device)
print(f"U[:,0]: {U[:, 0].cpu().numpy()}")

for i in range(1, dim):
    print(f"\nSampling U[:,{i}]...")
    this_Ui = torch.rand(nsamples, device=device)
    print(f"  Initial random: {this_Ui.cpu().numpy()}")
    
    for level in range(i):
        edges_level = vine.ind_vine[level]
        j = i - (level+1)
        if j < 0:
            break
        
        # find the edge
        edge_idx = None
        for idx_e, e in enumerate(edges_level):
            if (e[0] == j and e[1] == i) or (e[0] == i and e[1] == j):
                edge_idx = idx_e
                break
        
        if edge_idx is None:
            print(f"  Level {level}: No edge found between {j} and {i}")
            continue
        
        cop = vine.copulas[level][edge_idx]
        print(f"  Level {level}: Edge ({j},{i}), copula={cop.family if hasattr(cop, 'family') else 'nonparam'}")
        
        # Apply h-function
        this_Ui_before = this_Ui.clone()
        this_Ui = _hfunc_param(U[:, j], this_Ui, cop, inverse=True)
        print(f"    Before h-inverse: {this_Ui_before.cpu().numpy()}")
        print(f"    After h-inverse: {this_Ui.cpu().numpy()}")
        
        # Check for NaN
        if torch.any(torch.isnan(this_Ui)):
            print(f"    WARNING: NaN detected after h-inverse!")
            print(f"    U[:,{j}]: {U[:, j].cpu().numpy()}")
            if hasattr(cop, 'theta'):
                print(f"    Copula theta: {cop.theta}")
    
    U[:, i] = this_Ui
    print(f"Final U[:,{i}]: {U[:, i].cpu().numpy()}")

# Now try the full sampling
print("\n\nTrying full vine sampling...")
try:
    samples = vine.sample(10)
    print(f"Generated samples shape: {samples.shape}")
    
    # Check for NaN
    if np.any(np.isnan(samples)):
        print("WARNING: Samples contain NaN values!")
        nan_locs = np.argwhere(np.isnan(samples))
        print(f"NaN locations (row, col): {nan_locs}")
        print(f"First few samples:\n{samples[:5]}")
    else:
        print("Samples look good (no NaN)")
        print(f"Sample correlation:\n{np.corrcoef(samples.T).round(3)}")
        
except Exception as e:
    print(f"Error generating samples: {e}")
    import traceback
    traceback.print_exc() 