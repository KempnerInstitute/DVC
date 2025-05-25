"""
Simple debug script to test vine sampling
"""

import numpy as np
import sys
sys.path.append('src')

from DVC import vine_obj_bin, margin_obj, fit_vine

# Set random seed
np.random.seed(42)

# Generate simple data
n_samples = 100
dim = 3
data = np.random.randn(n_samples, dim).astype(np.float32)

# Create a simple vine
vine = vine_obj_bin(
    vine_family='d-vine',
    families=['gaussian', 'ind'],
    vine_depth=dim,
    margin=[],
    knots=50
)
# Set parametric mode
vine.param = True
vine.binning = False

# Set margins
for i in range(dim):
    vine.margin.append(margin_obj('norm', [0, 1], True))

# Set parametric mode
vine.param = True
vine.binning = False

# Configuration
gen_dict = {"parallel": False, "param": True, "binning": False}
par_dict = {"param_families": ["gaussian", "ind"]}
npc_dict = {"method": "local", "n_iter": 100}
bin_dict = {"n_bin": 1}

# Fit the vine
print("Fitting vine...")
fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
print("Vine fitted!")

# Check what copulas were fitted
print("\nFitted copulas:")
for level_idx, level_copulas in enumerate(vine.copulas):
    print(f"  Level {level_idx}:")
    for edge_idx, cop in enumerate(level_copulas):
        edge = vine.ind_vine[level_idx][edge_idx]
        if hasattr(cop, 'family'):  # Parametric
            print(f"    Edge {edge}: family={cop.family}, theta={cop.theta}")
        else:  # Non-parametric
            print(f"    Edge {edge}: non-parametric, opt_bw={cop.opt_bw}")

# Try to generate samples
print("\nGenerating samples...")
try:
    samples = vine.sample(10)
    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample values:\n{samples}")
    
    # Check for NaN
    if np.any(np.isnan(samples)):
        print("WARNING: Samples contain NaN values!")
        print(f"NaN locations: {np.argwhere(np.isnan(samples))}")
    else:
        print("Samples look good (no NaN)")
        
except Exception as e:
    print(f"Error generating samples: {e}")
    import traceback
    traceback.print_exc() 