"""
Example usage of the PyTorch DVC (Deep Vine Copula) implementation
"""

import torch
import numpy as np
from classes.objects import vine_obj_bin, margin_obj
from pre_proc.preparation import prep_cop

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate some example data
    np.random.seed(42)
    n_samples = 1000
    n_dims = 5
    
    # Generate correlated data using a covariance matrix
    mean = np.zeros(n_dims)
    cov = np.eye(n_dims)
    for i in range(n_dims):
        for j in range(n_dims):
            if i != j:
                cov[i, j] = 0.5 * np.exp(-abs(i - j))
                cov[j, i] = cov[i, j]
    
    data = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Convert to uniform margins using empirical CDF
    from scipy import stats
    data_uniform = np.zeros_like(data)
    for i in range(n_dims):
        data_uniform[:, i] = stats.rankdata(data[:, i]) / (n_samples + 1)
    
    # Convert to PyTorch tensor
    data_tensor = torch.tensor(data_uniform, dtype=torch.float32, device=device)
    
    # Create margin objects
    margins = []
    for i in range(n_dims):
        margin = margin_obj(dist='empirical', theta=None, is_cont=True)
        margins.append(margin)
    
    # Create vine copula object
    print("\nCreating R-vine copula...")
    vine = vine_obj_bin(
        vine_family='r-vine',
        families=['gaussian', 'clayton', 'student'],  # Available copula families
        vine_depth=n_dims - 1,  # Full vine
        margin=margins,
        knots=32,  # Grid size
        method='optimal'  # Use optimal tree structure
    )
    
    # Set up fitting parameters
    gen_dict = {
        'binning': False,       # No binning
        'parallel': False,      # Sequential fitting
        'param': True,          # Use parametric copulas
        'fitted': False,
        'vine_depth': n_dims - 1  # Full vine
    }
    
    npc_dict = {
        'opt_method': 'MISE',   # For non-parametric (not used with param=True)
        'batch_paral': 1
    }
    
    par_dict = {
        'param_families': ['gaussian', 'clayton', 'student', 'ind']  # Families to try
    }
    
    bin_dict = {
        'n_bin': 1  # Not used when binning=False
    }
    
    # Fit the vine copula
    print("\nFitting vine copula...")
    print("This implements parametric copula fitting with family selection by AIC.")
    
    try:
        vine.fit(data_tensor, gen_dict, npc_dict, par_dict, bin_dict)
        
        print("\nVine copula fitted successfully!")
        print(f"Number of trees: {len(vine.copulas)}")
        print(f"Correlations in first tree: {vine.correlations[0] if vine.correlations else 'N/A'}")
        
        # Show selected copula families for first tree
        if vine.copulas:
            print("\nSelected copula families in first tree:")
            for i, cop in enumerate(vine.copulas[0]):
                print(f"  Edge {i}: {cop.family}")
        
        # Example of evaluating log-likelihood on test data
        print("\nEvaluating on test data...")
        test_data = torch.randn(100, n_dims, device=device)
        # log_lik = vine.evaluation(test_data)
        print("Note: Full evaluation method is not yet implemented")
        
    except NotImplementedError as e:
        print(f"\nImplementation note: {e}")
        print("The parametric fitting is partially implemented.")
        print("Non-parametric fitting requires additional modules (bandwidth selection, local likelihood).")
    
    print("\n" + "="*60)
    print("Summary of PyTorch DVC Implementation Status:")
    print("="*60)
    print("✅ Core utilities (tensor ops, interpolation, probability functions)")
    print("✅ Data preprocessing and transformations")
    print("✅ Grid operations")
    print("✅ Parametric copula PDFs and fitting (Gaussian, Student-t, Clayton)")
    print("✅ Vine tree structure operations")
    print("✅ Basic vine copula fitting framework")
    print("⚠️  Parametric fitting with AIC selection (functional)")
    print("❌ Non-parametric copula fitting (requires bandwidth selection)")
    print("❌ Full vine evaluation/likelihood computation")
    print("❌ Sampling from fitted vine copulas")
    print("❌ Prediction and information measures")
    print("\nThe core mathematical operations have been successfully converted to PyTorch!")

if __name__ == "__main__":
    main() 