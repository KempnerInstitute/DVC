import numpy as np
import torch
import tensorflow as tf
from scipy import stats
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import sys
import os

# Add the correct paths
DVC_ROOT = "/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/DVC"
sys.path.append(DVC_ROOT)

# Import after adding path
from src.DVC_pyolder.objects import vine_obj_bin as PyTorchVine
from src.DVC_tensorflow.classes.objects import vine_obj_bin as TFVine
from src.DVC_pyolder.sampling import vine_copula_sample as pt_vine_sample
from src.DVC_tensorflow.sampling.vine_sample import vine_copula_sample as tf_vine_sample

def generate_ground_truth_data(n_samples: int, correlation_matrix: np.ndarray) -> np.ndarray:
    """Generate ground truth data with known correlations."""
    dim = correlation_matrix.shape[0]
    # Use multivariate normal for ground truth
    return np.random.multivariate_normal(
        mean=np.zeros(dim),
        cov=correlation_matrix,
        size=n_samples
    )

def compute_correlations(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Pearson and Kendall correlations."""
    dim = data.shape[1]
    pearson = np.zeros((dim, dim))
    kendall = np.zeros((dim, dim))
    
    for i in range(dim):
        for j in range(dim):
            pearson[i,j] = stats.pearsonr(data[:,i], data[:,j])[0]
            kendall[i,j] = stats.kendalltau(data[:,i], data[:,j])[0]
    
    return pearson, kendall

def fit_and_sample_pytorch(data: np.ndarray, n_samples: int, config: Dict) -> np.ndarray:
    """Fit PyTorch vine and generate samples."""
    vine = PyTorchVine()
    
    # Convert data to torch
    data_torch = torch.from_numpy(data).float()
    
    # Fit vine
    vine.fit(
        data_torch,
        gen_dict=config['gen'],
        npc_dict=config['npc'],
        par_dict=config['par'],
        bin_dict=config['bin']
    )
    
    # Sample
    samples, _, _, _ = pt_vine_sample(vine, n_samples)
    return samples.cpu().numpy()

def fit_and_sample_tensorflow(data: np.ndarray, n_samples: int, config: Dict) -> np.ndarray:
    """Fit TensorFlow vine and generate samples."""
    vine = TFVine()
    
    # Convert data to TF
    data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
    
    # Fit vine
    vine.fit(
        data_tf,
        gen_dict=config['gen'],
        npc_dict=config['npc'],
        par_dict=config['par'],
        bin_dict=config['bin']
    )
    
    # Sample
    samples, _, _, _ = tf_vine_sample(vine, n_samples)
    return samples.numpy()

def plot_correlation_comparison(ground_truth: np.ndarray, 
                              pytorch_samples: np.ndarray,
                              tensorflow_samples: np.ndarray,
                              save_path: str):
    """Plot correlation comparison between implementations."""
    # Compute correlations
    gt_pearson, gt_kendall = compute_correlations(ground_truth)
    pt_pearson, pt_kendall = compute_correlations(pytorch_samples)
    tf_pearson, tf_kendall = compute_correlations(tensorflow_samples)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Pearson correlation comparison
    axes[0,0].scatter(gt_pearson.flatten(), pt_pearson.flatten(), alpha=0.5, label='PyTorch')
    axes[0,0].scatter(gt_pearson.flatten(), tf_pearson.flatten(), alpha=0.5, label='TensorFlow')
    axes[0,0].plot([-1,1], [-1,1], 'k--')
    axes[0,0].set_title('Pearson Correlation Comparison')
    axes[0,0].set_xlabel('Ground Truth')
    axes[0,0].set_ylabel('Estimated')
    axes[0,0].legend()
    
    # Kendall correlation comparison
    axes[0,1].scatter(gt_kendall.flatten(), pt_kendall.flatten(), alpha=0.5, label='PyTorch')
    axes[0,1].scatter(gt_kendall.flatten(), tf_kendall.flatten(), alpha=0.5, label='TensorFlow')
    axes[0,1].plot([-1,1], [-1,1], 'k--')
    axes[0,1].set_title('Kendall Correlation Comparison')
    axes[0,1].set_xlabel('Ground Truth')
    axes[0,1].set_ylabel('Estimated')
    axes[0,1].legend()
    
    # PyTorch vs TensorFlow comparison
    axes[1,0].scatter(pt_pearson.flatten(), tf_pearson.flatten(), alpha=0.5)
    axes[1,0].plot([-1,1], [-1,1], 'k--')
    axes[1,0].set_title('PyTorch vs TensorFlow (Pearson)')
    axes[1,0].set_xlabel('PyTorch')
    axes[1,0].set_ylabel('TensorFlow')
    
    axes[1,1].scatter(pt_kendall.flatten(), tf_kendall.flatten(), alpha=0.5)
    axes[1,1].plot([-1,1], [-1,1], 'k--')
    axes[1,1].set_title('PyTorch vs TensorFlow (Kendall)')
    axes[1,1].set_xlabel('PyTorch')
    axes[1,1].set_ylabel('TensorFlow')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_error_metrics(ground_truth: np.ndarray, 
                         pytorch_samples: np.ndarray,
                         tensorflow_samples: np.ndarray) -> Dict:
    """Compute error metrics for both implementations."""
    gt_pearson, gt_kendall = compute_correlations(ground_truth)
    pt_pearson, pt_kendall = compute_correlations(pytorch_samples)
    tf_pearson, tf_kendall = compute_correlations(tensorflow_samples)
    
    metrics = {
        'pytorch': {
            'pearson_mse': np.mean((gt_pearson - pt_pearson)**2),
            'kendall_mse': np.mean((gt_kendall - pt_kendall)**2),
            'pearson_mae': np.mean(np.abs(gt_pearson - pt_pearson)),
            'kendall_mae': np.mean(np.abs(gt_kendall - pt_kendall))
        },
        'tensorflow': {
            'pearson_mse': np.mean((gt_pearson - tf_pearson)**2),
            'kendall_mse': np.mean((gt_kendall - tf_kendall)**2),
            'pearson_mae': np.mean(np.abs(gt_pearson - tf_pearson)),
            'kendall_mae': np.mean(np.abs(gt_kendall - tf_kendall))
        }
    }
    return metrics

def run_comprehensive_comparison(dims: List[int], n_samples: int = 10000):
    """Run comprehensive comparison tests for different dimensions."""
    results = {}
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(DVC_ROOT, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    for dim in dims:
        print(f"\nTesting {dim}-dimensional vine...")
        
        # Generate ground truth correlation matrix
        correlation_matrix = np.eye(dim)
        # Add some non-trivial correlations
        for i in range(dim):
            for j in range(i+1, dim):
                correlation_matrix[i,j] = correlation_matrix[j,i] = 0.5 / (abs(i-j))
        
        # Generate ground truth data
        ground_truth = generate_ground_truth_data(n_samples, correlation_matrix)
        
        # Configuration for both implementations
        config = {
            'gen': {'method': 'optimal', 'family': 'r-vine'},
            'npc': {'bw': 'scott', 'kernel': 'gaussian'},
            'par': {'param_families': ['gaussian', 'student']},
            'bin': {'n_bin': 10}
        }
        
        # Fit and sample from both implementations
        try:
            pytorch_samples = fit_and_sample_pytorch(ground_truth, n_samples, config)
            tensorflow_samples = fit_and_sample_tensorflow(ground_truth, n_samples, config)
            
            # Plot comparisons
            plot_correlation_comparison(
                ground_truth, 
                pytorch_samples, 
                tensorflow_samples,
                os.path.join(results_dir, f'correlation_comparison_dim{dim}.png')
            )
            
            # Compute metrics
            metrics = compute_error_metrics(ground_truth, pytorch_samples, tensorflow_samples)
            results[dim] = metrics
            
            print(f"\nResults for {dim} dimensions:")
            print("\nPyTorch metrics:")
            for k, v in metrics['pytorch'].items():
                print(f"{k}: {v:.6f}")
            print("\nTensorFlow metrics:")
            for k, v in metrics['tensorflow'].items():
                print(f"{k}: {v:.6f}")
            
        except Exception as e:
            print(f"Error in {dim} dimensions: {str(e)}")
            results[dim] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    # Test dimensions
    dims = [2, 3, 4, 5]
    
    # Run comparison
    results = run_comprehensive_comparison(dims)
    
    # Save results
    results_dir = os.path.join(DVC_ROOT, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'implementation_comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=4) 