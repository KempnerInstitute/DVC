"""
Debug script to identify the list index out of range error
"""

import numpy as np
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classes.objects import vine_obj_bin, margin_obj
from pre_proc.preparation import prep_cop

def debug_simple_fit():
    """Test a very simple case to see where the error occurs"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate very simple 3D data
    n_samples = 100
    dim = 3
    
    # Simple independent normal data first
    data = np.random.normal(0, 1, (n_samples, dim))
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # Create margin objects
    margins = [margin_obj('norm', [0, 1], True) for _ in range(dim)]
    
    # Try C-vine first
    print("Testing C-vine...")
    try:
        vine = vine_obj_bin('c-vine', ['gaussian'], dim, margins, 11, 'matrix')
        
        # Prepare data
        data_prep = prep_cop(data_tensor, vine, 'no_sort')
        data_prep = torch.tensor(data_prep, dtype=torch.float32)
        
        # Simple config
        gen_dict = {
            'param': True,
            'binning': False,
            'fitted': False,
            'parallel': True,
            'vine_depth': dim
        }
        
        par_dict = {'param_families': ['gaussian']}
        npc_dict = {'opt_method': 'trust-exact', 'batch_paral': 10}
        bin_dict = {'n_bin': 1}
        
        # This is where the error should occur
        vine.fit(data_prep, gen_dict, npc_dict, par_dict, bin_dict)
        print("C-vine fit successful!")
        
    except Exception as e:
        print(f"C-vine failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_simple_fit() 