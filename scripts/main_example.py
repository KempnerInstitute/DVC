###############################################
# scripts/main_example.py  --->version 2
###############################################
import sys
import os
    
import numpy as np
import torch

from DVC_pyolder.objects import vine_obj_bin, margin_obj
from DVC_pyolder.preparation import prep_cop
from DVC_pyolder.vine_model import fit_vine, evaluate_vine, sample_vine
from DVC_pyolder.info_estimation import vine_entropy

this_file_dir = os.path.dirname(os.path.abspath(__file__))        
project_root = os.path.dirname(this_file_dir)                     
src_folder = os.path.join(project_root, "src")                    
if src_folder not in sys.path:
    sys.path.insert(0, src_folder)
    

def main():
    # 1) Generate synthetic data
    np.random.seed(42)
    n_samples = 3000
    dim = 3
    x = np.random.rand(n_samples, dim)  # shape [3000,3] in [0,1]

    # 2) Build margin objects, e.g. standard normal placeholders
    margin_vine = []
    for i in range(dim):
        margin_vine.append(margin_obj(dist='norm', theta=[0.,1.], is_cont=True))

    # 3) Create a vine_obj_bin
    vine = vine_obj_bin(
        vine_family='r-vine',     # or 'c-vine','d-vine'
        families='kercop',        # 'kercop' for nonparam approach
        vine_depth=dim,           # dimension
        margin=margin_vine,
        knots=50,
        method='matrix',
        r_matrix=None             # we pass None => default matrix approach
    )

    # 4) Dictionary config for fitting
    gen_dict = {
        'parallel': True,
        'binning': False,
        'param': False,     # if True => param approach
        'vine_depth': dim,
        'fitted': False
    }
    npc_dict = {
        'opt_method': 'LL1',
        'batch_paral': 3
    }
    par_dict = {
        'param_families': ["ind","gaussian","student","clayton","claytonrot90"]
    }
    bin_dict = {
        'n_bin': 3
    }

    # 5) Fit the vine
    vine.fit(x, gen_dict, npc_dict, par_dict, bin_dict)

    # 6) Evaluate the vine PDF at random test points in [0,1]
    test_pts = torch.rand((1000,dim), dtype=torch.float32)
    p, p_cop, logmarg = vine.evaluation(test_pts)
    print("Evaluated p shape:", p.shape)  # should be [1000]

    # 7) Sample from the vine
    samples_vine = vine.sample(500)
    print("vine samples shape:", samples_vine.shape)

    # 8) Approximate the vine's entropy
    info_dict = {
        'alpha': 0.05,
        'cases': 500,
        'iterations': 5
    }
    H_est = vine_entropy(vine, info_dict)
    print("Approx. entropy from vine:", H_est)

if __name__=="__main__":
    main()