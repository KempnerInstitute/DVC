import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from typing import Dict, List

# Add the correct paths
DVC_ROOT = "/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/DVC"
sys.path.append(DVC_ROOT)

# Import after adding path
from test_implementation_comparison import run_comprehensive_comparison

def generate_correlation_patterns(dim: int) -> Dict[str, np.ndarray]:
    """Generate different correlation patterns for testing."""
    patterns = {}
    
    # Pattern 1: Decreasing correlations with distance
    decreasing = np.eye(dim)
    for i in range(dim):
        for j in range(i+1, dim):
            decreasing[i,j] = decreasing[j,i] = 0.5 / (abs(i-j))
    patterns['decreasing'] = decreasing
    
    # Pattern 2: Block diagonal
    block = np.eye(dim)
    block_size = 2
    for i in range(0, dim-1, block_size):
        for j in range(block_size):
            for k in range(block_size):
                if j != k and i+j < dim and i+k < dim:
                    block[i+j,i+k] = block[i+k,i+j] = 0.7
    patterns['block'] = block
    
    # Pattern 3: Alternating positive/negative
    alternating = np.eye(dim)
    for i in range(dim):
        for j in range(i+1, dim):
            alternating[i,j] = alternating[j,i] = 0.3 * (-1)**(i+j)
    patterns['alternating'] = alternating
    
    # Pattern 4: Constant correlation
    constant = np.ones((dim, dim)) * 0.5
    np.fill_diagonal(constant, 1.0)
    patterns['constant'] = constant
    
    return patterns

def plot_correlation_matrices(patterns: Dict[str, np.ndarray], save_path: str):
    """Plot all correlation patterns."""
    n_patterns = len(patterns)
    fig, axes = plt.subplots(1, n_patterns, figsize=(5*n_patterns, 4))
    
    for i, (name, matrix) in enumerate(patterns.items()):
        im = axes[i].imshow(matrix, vmin=-1, vmax=1, cmap='RdBu')
        axes[i].set_title(name.capitalize())
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_pattern_tests(dims: List[int], n_samples: int = 10000):
    """Run tests for each correlation pattern."""
    all_results = {}
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(DVC_ROOT, "test_results", "correlation_patterns")
    os.makedirs(results_dir, exist_ok=True)
    
    for dim in dims:
        print(f"\nTesting {dim}-dimensional patterns...")
        patterns = generate_correlation_patterns(dim)
        
        # Plot correlation patterns
        plot_correlation_matrices(
            patterns, 
            os.path.join(results_dir, f'correlation_patterns_dim{dim}.png')
        )
        
        # Test each pattern
        pattern_results = {}
        for pattern_name, correlation_matrix in patterns.items():
            print(f"\nTesting pattern: {pattern_name}")
            
            # Generate ground truth data
            ground_truth = np.random.multivariate_normal(
                mean=np.zeros(dim),
                cov=correlation_matrix,
                size=n_samples
            )
            
            # Configuration
            config = {
                'gen': {'method': 'optimal', 'family': 'r-vine'},
                'npc': {'bw': 'scott', 'kernel': 'gaussian'},
                'par': {'param_families': ['gaussian', 'student']},
                'bin': {'n_bin': 10}
            }
            
            try:
                # Run comparison
                results = run_comprehensive_comparison([dim], n_samples)
                pattern_results[pattern_name] = results[dim]
                
            except Exception as e:
                print(f"Error testing pattern {pattern_name}: {str(e)}")
                pattern_results[pattern_name] = {'error': str(e)}
        
        all_results[dim] = pattern_results
    
    return all_results

def analyze_results(results: Dict):
    """Analyze and print summary of results."""
    print("\nSummary of Results:")
    print("=" * 50)
    
    for dim, pattern_results in results.items():
        print(f"\nDimension: {dim}")
        print("-" * 30)
        
        for pattern, metrics in pattern_results.items():
            if 'error' in metrics:
                print(f"\nPattern {pattern}: Error - {metrics['error']}")
                continue
                
            print(f"\nPattern: {pattern}")
            print("PyTorch vs TensorFlow comparison:")
            
            pt_metrics = metrics['pytorch']
            tf_metrics = metrics['tensorflow']
            
            for metric in ['pearson_mse', 'kendall_mse', 'pearson_mae', 'kendall_mae']:
                pt_val = pt_metrics[metric]
                tf_val = tf_metrics[metric]
                diff = abs(pt_val - tf_val)
                
                print(f"{metric}:")
                print(f"  PyTorch:    {pt_val:.6f}")
                print(f"  TensorFlow: {tf_val:.6f}")
                print(f"  Difference: {diff:.6f}")

if __name__ == "__main__":
    # Test dimensions
    dims = [2, 3, 4, 5]
    
    # Run tests
    results = run_pattern_tests(dims)
    
    # Analyze results
    analyze_results(results)
    
    # Save results
    results_dir = os.path.join(DVC_ROOT, "test_results", "correlation_patterns")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'correlation_pattern_results.json'), 'w') as f:
        json.dump(results, f, indent=4) 