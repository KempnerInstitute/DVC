import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add the TensorFlow implementation to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "DVC_tensorflow"))

# Import from our implementation
from DVC.objects import vine_obj_bin, margin_obj

# Try to import from the TensorFlow implementation
try:
    from src.classes.objects import vine_obj_bin as tf_vine_obj_bin
    from src.classes.objects import margin_obj as tf_margin_obj
    from src.pre_proc.preparation import prep_cop
    from src.pre_proc.define_copulas import define_copulas
    tf_available = True
except ImportError:
    print("TensorFlow implementation not available for direct comparison")
    tf_available = False

def generate_gaussian_data(n_samples, dim, rho=0.6, seed=42):
    """Generate samples from a multivariate Gaussian with uniform correlation rho"""
    np.random.seed(seed)
    
    # Create correlation matrix with uniform correlation
    cov = np.full((dim, dim), rho)
    np.fill_diagonal(cov, 1.0)
    
    # Generate samples
    mean = np.zeros(dim)
    data = np.random.multivariate_normal(mean, cov, size=n_samples)
    
    return data, cov

def compare_correlation_preservation(dims=[3, 4, 5], n_train=5000, n_samples=2000, rho=0.6):
    """
    Compare how well each implementation preserves correlations in D-vines
    """
    results = {}
    
    for dim in dims:
        print(f"\n{'-'*80}")
        print(f"Comparing implementations for dimension: {dim}")
        print(f"{'-'*80}")
        
        # Generate data
        data, true_corr = generate_gaussian_data(n_train, dim, rho)
        print(f"Generated {n_train} samples from {dim}D Gaussian with rho={rho}")
        
        # Print true correlation matrix
        print("\nTrue correlation matrix:")
        print(np.round(true_corr, 3))
        
        # Create storage for results
        results[dim] = {
            'true_corr': true_corr,
            'our_impl': {},
            'tf_impl': {}
        }
        
        # Fit with our implementation
        print("\nFitting our implementation (D-vine)...")
        margins = []
        for i in range(dim):
            loc, scale = np.mean(data[:, i]), np.std(data[:, i])
            margin = margin_obj('norm', [loc, scale], True)
            margins.append(margin)
        
        # Create and fit D-vine using our implementation
        our_vine = vine_obj_bin('d-vine', ['gaussian'], dim, margins, 40, 'optimal')
        
        # Prepare dictionaries for vine.fit()
        gen_dict = {'param': True, 'binning': False, 'fitted': False}
        npc_dict = {}
        par_dict = {'param_families': ['gaussian']}
        bin_dict = {'n_bin': 1}
        
        # Fit the vine with our implementation
        our_vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, None)
        
        # Generate samples with our implementation
        our_samples = our_vine.sample(n_samples)
        our_corr = np.corrcoef(our_samples, rowvar=False)
        our_corr_error = np.mean(np.abs(our_corr - true_corr))
        
        print(f"Our implementation correlation matrix error: {our_corr_error:.4f}")
        
        # Fit with TensorFlow implementation if available
        if tf_available:
            print("\nFitting TensorFlow implementation (D-vine)...")
            try:
                # Convert data to torch tensor
                data_t = torch.tensor(data, dtype=torch.float32)
                
                # Define margins
                tf_margins = [tf_margin_obj('norm', [0.0, 1.0], True) for _ in range(dim)]
                
                # Define the vine structure
                r_matrix, cop_vine, ind_vine, nodes, matrix_edges, _ = define_copulas('d-vine', 'optimal', False, 1, dim)
                
                # Create vine object
                tf_vine = tf_vine_obj_bin('d-vine', "kercop", dim, tf_margins, 50, 'optimal', r_matrix)
                
                # Preprocess data
                data_prep = prep_cop(data_t, tf_vine, 'rand')
                
                # Prepare fit dictionaries
                tf_gen_dict = {'parallel': True, 'binning': False, 'param': True, 'vine_depth': dim, 'fitted': False}
                tf_par_dict = {'param_families': ["ind", "gaussian"]}
                tf_npc_dict = {'opt_method': 'LL1', 'batch_paral': 3}
                tf_bin_dict = {'n_bin': 1}
                
                # Fit the vine
                tf_vine.fit(data_prep, tf_gen_dict, tf_npc_dict, tf_par_dict, tf_bin_dict)
                
                # Generate samples
                tf_samples = tf_vine.sample(n_samples).numpy()
                tf_corr = np.corrcoef(tf_samples, rowvar=False)
                tf_corr_error = np.mean(np.abs(tf_corr - true_corr))
                
                print(f"TensorFlow implementation correlation matrix error: {tf_corr_error:.4f}")
                
                # Store the TF results
                results[dim]['tf_impl'] = {
                    'corr': tf_corr,
                    'error': tf_corr_error,
                    'samples': tf_samples
                }
            
            except Exception as e:
                print(f"Error with TensorFlow implementation: {str(e)}")
        
        # Store our results
        results[dim]['our_impl'] = {
            'corr': our_corr,
            'error': our_corr_error,
            'samples': our_samples
        }
        
        # Compare the non-adjacent variable correlations (most important test)
        print("\nNon-adjacent variable correlation comparison:")
        print(f"{'Variable Pair':<15} | {'True':<10} | {'Our Impl':<10} | {'TF Impl':<10} | {'Our Error':<10} | {'TF Error':<10}")
        print("-" * 75)
        
        for i in range(dim):
            for j in range(i+2, dim):  # Only non-adjacent pairs (distance >= 2)
                true_val = true_corr[i, j]
                our_val = our_corr[i, j]
                our_err = abs(our_val - true_val)
                
                if tf_available and 'corr' in results[dim]['tf_impl']:
                    tf_val = results[dim]['tf_impl']['corr'][i, j]
                    tf_err = abs(tf_val - true_val)
                    print(f"{f'({i},{j})':<15} | {true_val:<10.4f} | {our_val:<10.4f} | {tf_val:<10.4f} | {our_err:<10.4f} | {tf_err:<10.4f}")
                else:
                    print(f"{f'({i},{j})':<15} | {true_val:<10.4f} | {our_val:<10.4f} | {'N/A':<10} | {our_err:<10.4f} | {'N/A':<10}")
    
    return results

def plot_comparison_results(results, output_dir="comparison_results"):
    """Plot the comparison results"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot error comparison across dimensions
    dims = sorted(results.keys())
    our_errors = [results[dim]['our_impl']['error'] for dim in dims]
    tf_errors = [results[dim]['tf_impl']['error'] if 'tf_impl' in results[dim] and 'error' in results[dim]['tf_impl'] else np.nan for dim in dims]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dims, our_errors, 'o-', label="Our Implementation")
    if tf_available:
        plt.plot(dims, tf_errors, 's-', label="TensorFlow Implementation")
    plt.xlabel("Dimension")
    plt.ylabel("Mean Absolute Error (Correlation)")
    plt.title("Correlation Preservation Error Comparison")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "error_comparison.png"))
    plt.close()
    
    # Plot non-adjacent variable correlation errors 
    for dim in dims:
        if dim <= 3:  # Skip dims with too few non-adjacent pairs
            continue
            
        true_corr = results[dim]['true_corr']
        our_corr = results[dim]['our_impl']['corr']
        
        # Get non-adjacent pairs
        pairs = []
        non_adj_true = []
        non_adj_our = []
        non_adj_tf = []
        
        for i in range(dim):
            for j in range(i+2, dim):  # Only non-adjacent pairs
                pairs.append(f"({i},{j})")
                non_adj_true.append(true_corr[i, j])
                non_adj_our.append(our_corr[i, j])
                
                if tf_available and 'tf_impl' in results[dim] and 'corr' in results[dim]['tf_impl']:
                    tf_corr = results[dim]['tf_impl']['corr']
                    non_adj_tf.append(tf_corr[i, j])
        
        # Create a bar plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(pairs))
        width = 0.2
        
        plt.bar(x - width, non_adj_true, width, label="True")
        plt.bar(x, non_adj_our, width, label="Our Implementation")
        if tf_available and non_adj_tf:
            plt.bar(x + width, non_adj_tf, width, label="TensorFlow Implementation")
        
        plt.xlabel("Variable Pair")
        plt.ylabel("Correlation")
        plt.title(f"{dim}D Gaussian - Non-Adjacent Variable Correlations")
        plt.xticks(x, pairs)
        plt.legend()
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, f"non_adj_corr_{dim}d.png"))
        plt.close()

if __name__ == "__main__":
    # Run the comparison
    results = compare_correlation_preservation(dims=[3, 4, 5], n_train=5000, n_samples=2000)
    
    # Plot the results
    plot_comparison_results(results)
    
    print("\nComparison completed. Results saved in the 'comparison_results' directory.") 