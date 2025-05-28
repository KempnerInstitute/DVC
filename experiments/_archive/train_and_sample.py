##################################################
# train_and_sample.py (example usage)
##################################################
import numpy as np
import torch
from DVC_pyolder.objects import vine_obj_bin, margin_obj
from DVC_pyolder.vine_model import fit_vine
from DVC_pyolder.d_vine_fix import apply_d_vine_fix

def main():
    # Generate some sample data in 3D
    N = 2000
    x = np.random.randn(N, 3)
    # Suppose we want to model it with a D-vine
    margins = [margin_obj('norm', (0.0, 1.0)),  # just placeholders
               margin_obj('norm', (0.0, 1.0)),
               margin_obj('norm', (0.0, 1.0))]

    # Create vine object
    vine = vine_obj_bin(vine_family='d-vine', families=['gaussian']*3, 
                        vine_depth=3, margin=margins, knots=50)

    # Fit
    vine = fit_vine(vine, x, param=True)

    # Patch the sampling method to do correct chain-of-conditional logic
    apply_d_vine_fix(vine)

    # Now sample
    samples = vine.sample(1000)
    print("Sample shape:", samples.shape)

    # Check correlations
    corr_empirical = np.corrcoef(samples.T)
    print("Empirical correlation matrix:\n", corr_empirical)

if __name__ == "__main__":
    main() 