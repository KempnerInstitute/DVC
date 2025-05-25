"""
Comprehensive example demonstrating all DVC PyTorch functionality
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Import all DVC modules
from classes.objects import vine_obj_bin, margin_obj
from sampling.vine_sample import vine_copula_sample, vine_cop_par_sample
from pred.prediction import predict_vine
from info.info_estimation import vine_entropy
from plot.plot_vine import plot_vine, plot_copula_contour, plot_vine_matrix
from grid.grid_op import create_grids

def generate_correlated_data(n_samples=1000, n_dims=5, correlation_type='mixed'):
    """Generate correlated data with known structure"""
    
    if correlation_type == 'mixed':
        # Create a mix of different dependencies
        # First two: strong Gaussian correlation
        mean = np.zeros(2)
        cov = [[1, 0.8], [0.8, 1]]
        data_12 = np.random.multivariate_normal(mean, cov, n_samples)
        
        # Next two: Clayton copula
        theta = 2.0
        u1 = np.random.uniform(0, 1, n_samples)
        v = np.random.uniform(0, 1, n_samples)
        u2 = (1 + u1**(-theta) * (v**(-theta/(1+theta)) - 1))**(-1/theta)
        data_34 = np.column_stack([stats.norm.ppf(u1), stats.norm.ppf(u2)])
        
        # Last one: independent
        data_5 = np.random.normal(0, 1, (n_samples, 1))
        
        # Combine all
        data = np.hstack([data_12, data_34, data_5])
        
        # Add some cross-dependencies
        data[:, 4] += 0.3 * data[:, 0]  # Make dim 5 slightly dependent on dim 1
        
    return data

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Generate test data
    print("\n1. Generating test data...")
    n_samples = 2000
    n_dims = 5
    data = generate_correlated_data(n_samples, n_dims)
    
    # Convert to uniform margins for copula fitting
    data_uniform = np.zeros_like(data)
    for i in range(n_dims):
        data_uniform[:, i] = stats.rankdata(data[:, i]) / (n_samples + 1)
    
    # Convert to PyTorch tensor
    data_torch = torch.tensor(data_uniform, dtype=torch.float32, device=device)
    
    # 2. Create margins
    print("\n2. Creating margin objects...")
    margins = []
    for i in range(n_dims):
        margin = margin_obj(dist='empirical', theta=None, is_cont=True)
        margins.append(margin)
    
    # 3. Fit parametric vine copula
    print("\n3. Fitting parametric vine copula...")
    vine_param = vine_obj_bin(
        vine_family='r-vine',
        families=['gaussian', 'clayton', 'student'],
        vine_depth=n_dims - 1,
        margin=margins,
        knots=32,
        method='optimal'
    )
    
    # Create grids
    vine_param.grid_u, vine_param.grid_s, vine_param.grid_x = create_grids(
        vine_param.knots, device=device
    )
    
    # Set up fitting parameters
    gen_dict_param = {
        'binning': False,
        'parallel': False,
        'param': True,
        'vine_depth': n_dims - 1
    }
    
    npc_dict = {}
    
    par_dict = {
        'param_families': ['gaussian', 'clayton', 'student', 'ind']
    }
    
    bin_dict = {
        'n_bin': 1
    }
    
    # Fit parametric model
    vine_param.fit(data_torch, gen_dict_param, npc_dict, par_dict, bin_dict)
    
    print("\nParametric vine structure:")
    print(f"Vine family: {vine_param.vine_family}")
    print(f"R-matrix:\n{vine_param.r_matrix}")
    
    # Report selected copulas
    print("\nSelected copula families by tree:")
    for i, tree in enumerate(vine_param.copulas):
        families = [cop.family for cop in tree]
        print(f"  Tree {i}: {families}")
    
    # 4. Fit non-parametric vine copula
    print("\n4. Fitting non-parametric vine copula...")
    vine_nonparam = vine_obj_bin(
        vine_family='c-vine',
        families=['gaussian'],
        vine_depth=2,  # Limit depth for speed
        margin=margins,
        knots=16,
        method=None
    )
    
    # Create grids
    vine_nonparam.grid_u, vine_nonparam.grid_s, vine_nonparam.grid_x = create_grids(
        vine_nonparam.knots, device=device
    )
    
    gen_dict_nonparam = {
        'binning': False,
        'parallel': False,
        'param': False,
        'vine_depth': 2
    }
    
    npc_dict = {
        'opt_method': 'LL1',
        'batch_paral': 1
    }
    
    # Fit non-parametric model
    vine_nonparam.fit(data_torch, gen_dict_nonparam, npc_dict, {}, bin_dict)
    
    print("\nNon-parametric vine fitted successfully!")
    
    # 5. Model evaluation
    print("\n5. Evaluating models on test data...")
    test_size = 100
    test_indices = torch.randperm(n_samples)[:test_size]
    test_data = data_torch[test_indices]
    
    # Evaluate parametric model
    p_param, p_cop_param, logp_param = vine_param.evaluation(test_data)
    print(f"Parametric model - Mean log-likelihood: {logp_param.mean().item():.3f}")
    
    # Evaluate non-parametric model
    p_nonparam, p_cop_nonparam, logp_nonparam = vine_nonparam.evaluation(test_data)
    print(f"Non-parametric model - Mean log-likelihood: {logp_nonparam.mean().item():.3f}")
    
    # 6. Sampling
    print("\n6. Sampling from fitted models...")
    n_samples_gen = 500
    
    # Sample from parametric model
    if vine_param.param:
        samples_param, u_param, _, _ = vine_cop_par_sample(vine_param, n_samples_gen)
        print(f"Generated {n_samples_gen} samples from parametric model")
    
    # Sample from non-parametric model
    samples_nonparam, u_nonparam, _, _ = vine_copula_sample(vine_nonparam, n_samples_gen)
    print(f"Generated {n_samples_gen} samples from non-parametric model")
    
    # 7. Prediction
    print("\n7. Demonstrating prediction...")
    # Use first 4 dimensions to predict the 5th
    pred_dim = 4
    exp_dim = 50
    
    # Select subset for prediction
    pred_indices = torch.randperm(n_samples)[:20]
    pred_data = data_torch[pred_indices, :4]  # Use first 4 dims
    
    # Add a dummy column for the dimension to predict
    pred_data_full = torch.cat([pred_data, torch.zeros(20, 1, device=device)], dim=1)
    
    # Predict using parametric model
    p_pred, y_ml, y_em = predict_vine(pred_data_full, vine_param, pred_dim, exp_dim)
    print(f"Prediction completed - ML estimates shape: {y_ml.shape}")
    
    # 8. Information measures
    print("\n8. Computing information measures...")
    info_dict = {
        'alpha': 0.05,
        'cases': 1000,
        'iterations': 5
    }
    
    # Estimate entropy (simplified version)
    entropy_est = vine_entropy(vine_param, info_dict)
    print(f"Estimated entropy: {entropy_est:.3f}")
    
    # 9. Visualization
    print("\n9. Creating visualizations...")
    
    # Plot vine structure
    fig1 = plot_vine_matrix(vine_param)
    plt.savefig('vine_matrix_structure.png', dpi=150, bbox_inches='tight')
    print("Saved: vine_matrix_structure.png")
    
    # Plot copula PDFs
    fig2 = plot_vine('pdf', vine_param)
    plt.savefig('vine_copula_pdfs.png', dpi=150, bbox_inches='tight')
    print("Saved: vine_copula_pdfs.png")
    
    # Plot specific copula contour
    if len(vine_param.copulas) > 0 and len(vine_param.copulas[0]) > 0:
        fig3 = plot_copula_contour(vine_param, tree_level=0, copula_index=0)
        plt.savefig('copula_contour_tree0.png', dpi=150, bbox_inches='tight')
        print("Saved: copula_contour_tree0.png")
    
    # 10. Binning example
    print("\n10. Demonstrating binning support...")
    vine_binned = vine_obj_bin(
        vine_family='d-vine',
        families=['gaussian'],
        vine_depth=n_dims - 1,
        margin=margins,
        knots=32,
        method=None
    )
    
    # Create grids
    vine_binned.grid_u, vine_binned.grid_s, vine_binned.grid_x = create_grids(
        vine_binned.knots, device=device
    )
    
    gen_dict_binned = {
        'binning': True,
        'parallel': False,
        'param': True,
        'vine_depth': n_dims - 1
    }
    
    bin_dict_binned = {
        'n_bin': 5  # Use 5 bins
    }
    
    # Fit with binning
    vine_binned.fit(data_torch, gen_dict_binned, {}, par_dict, bin_dict_binned)
    print(f"Fitted D-vine with {bin_dict_binned['n_bin']} bins")
    
    # Compare correlations
    print("\n11. Correlation analysis...")
    print("True data correlations (Kendall's tau):")
    for i in range(n_dims - 1):
        tau, _ = stats.kendalltau(data[:, i], data[:, i + 1])
        print(f"  Variables {i+1}-{i+2}: {tau:.3f}")
    
    print("\nFitted model correlations (first tree):")
    if hasattr(vine_param, 'correlations') and len(vine_param.correlations) > 0:
        for i, tau in enumerate(vine_param.correlations[0]):
            print(f"  Edge {i+1}: {tau:.3f}")
    
    print("\n" + "="*60)
    print("Comprehensive DVC PyTorch example completed successfully!")
    print("="*60)
    
    # Close all plots
    plt.close('all')
    
    return {
        'vine_param': vine_param,
        'vine_nonparam': vine_nonparam,
        'vine_binned': vine_binned,
        'samples_param': samples_param if 'samples_param' in locals() else None,
        'samples_nonparam': samples_nonparam,
        'entropy': entropy_est
    }

if __name__ == "__main__":
    results = main() 