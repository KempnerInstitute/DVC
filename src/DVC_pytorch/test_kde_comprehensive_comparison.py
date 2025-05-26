import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import stats
from utils.prob_op import kde, kernel_pdf2, kde_wrapper
from utils.kde_simple import kde_gaussian, silverman_bandwidth, scott_bandwidth

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def generate_test_datasets():
    """Generate various test datasets for comprehensive comparison"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    datasets = {
        'Normal': {
            'data': torch.randn(1000),
            'true_pdf': lambda x: stats.norm.pdf(x, 0, 1),
            'description': 'Standard Normal Distribution'
        },
        'Bimodal': {
            'data': torch.cat([torch.randn(500) - 2.5, torch.randn(500) + 2.5]),
            'true_pdf': lambda x: 0.5 * stats.norm.pdf(x, -2.5, 1) + 0.5 * stats.norm.pdf(x, 2.5, 1),
            'description': 'Bimodal Normal Distribution'
        },
        'Uniform': {
            'data': torch.rand(1000) * 4 - 2,
            'true_pdf': lambda x: np.where((x >= -2) & (x <= 2), 0.25, 0),
            'description': 'Uniform Distribution [-2, 2]'
        },
        'Exponential': {
            'data': torch.distributions.Exponential(1.0).sample((1000,)),
            'true_pdf': lambda x: stats.expon.pdf(x, scale=1.0),
            'description': 'Exponential Distribution (λ=1)'
        },
        'Mixed': {
            'data': torch.cat([
                torch.randn(300) * 0.5 - 2,  # Narrow normal at -2
                torch.rand(400) * 2,          # Uniform [0, 2]
                torch.randn(300) * 0.3 + 3    # Very narrow normal at 3
            ]),
            'true_pdf': lambda x: (0.3 * stats.norm.pdf(x, -2, 0.5) + 
                                  0.4 * np.where((x >= 0) & (x <= 2), 0.5, 0) + 
                                  0.3 * stats.norm.pdf(x, 3, 0.3)),
            'description': 'Mixed Distribution (Normal + Uniform + Normal)'
        },
        'Heavy-tailed': {
            'data': torch.tensor(np.random.standard_t(df=3, size=1000), dtype=torch.float32),
            'true_pdf': lambda x: stats.t.pdf(x, df=3),
            'description': 'Student-t Distribution (df=3)'
        }
    }
    
    return datasets

def evaluate_kde_methods(data, true_pdf, x_eval):
    """Evaluate all KDE methods on given data"""
    methods = {
        'Original DCT': lambda d: kde(d, n=len(x_eval)),
        'kernel_pdf2': lambda d: kernel_pdf2(d),
        'FFT (Simple)': lambda d: kde_gaussian(d, n=len(x_eval), method='fft'),
        'cdist': lambda d: kde_gaussian(d, n=len(x_eval), method='cdist'),
        'cdist_chunked': lambda d: kde_gaussian(d, n=len(x_eval), method='cdist_chunked'),
        'FFT (Silverman)': lambda d: kde_gaussian(d, n=len(x_eval), method='fft', 
                                                 bandwidth=silverman_bandwidth(d)),
        'FFT (Scott)': lambda d: kde_gaussian(d, n=len(x_eval), method='fft', 
                                             bandwidth=scott_bandwidth(d))
    }
    
    results = {}
    
    for name, method in methods.items():
        start_time = time.time()
        try:
            density, mesh = method(data)
            elapsed = time.time() - start_time
            
            # Interpolate to common grid for comparison
            if len(mesh) != len(x_eval):
                from scipy.interpolate import interp1d
                f = interp1d(mesh.numpy(), density.numpy(), 
                           bounds_error=False, fill_value=0)
                density_eval = torch.tensor(f(x_eval.numpy()))
            else:
                density_eval = density
            
            # Calculate metrics
            dx = x_eval[1] - x_eval[0]
            integral = torch.sum(density_eval) * dx
            
            # MSE against true PDF
            true_values = torch.tensor([true_pdf(x) for x in x_eval.numpy()])
            mse = torch.mean((density_eval - true_values) ** 2)
            
            # KL divergence approximation (avoiding log(0))
            eps = 1e-10
            kl_div = torch.sum(true_values * torch.log((true_values + eps) / (density_eval + eps))) * dx
            
            results[name] = {
                'density': density_eval,
                'mesh': x_eval,
                'time': elapsed,
                'integral': integral.item(),
                'mse': mse.item(),
                'kl_divergence': kl_div.item()
            }
            
        except Exception as e:
            print(f"Error in {name}: {e}")
            results[name] = None
    
    return results

def plot_comparison_results(datasets_results, datasets):
    """Create comprehensive visualization of results"""
    
    # Create figure with subplots
    n_datasets = len(datasets)
    fig = plt.figure(figsize=(20, 4 * n_datasets))
    
    # Color palette for methods
    method_colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    for idx, (dataset_name, dataset_info) in enumerate(datasets.items()):
        if dataset_name not in datasets_results:
            continue
            
        results = datasets_results[dataset_name]
        data = dataset_info['data']
        true_pdf = dataset_info['true_pdf']
        
        # Subplot 1: KDE comparison
        ax1 = plt.subplot(n_datasets, 3, idx * 3 + 1)
        
        # Plot histogram
        ax1.hist(data.numpy(), bins=50, density=True, alpha=0.3, 
                color='gray', label='Histogram')
        
        # Plot true PDF
        x_range = results['x_eval'].numpy()
        true_values = [true_pdf(x) for x in x_range]
        ax1.plot(x_range, true_values, 'k--', linewidth=2, label='True PDF')
        
        # Plot KDE results
        for i, (method_name, method_result) in enumerate(results['methods'].items()):
            if method_result is not None:
                ax1.plot(x_range, method_result['density'].numpy(), 
                        color=method_colors[i], linewidth=1.5, 
                        label=f"{method_name} (∫={method_result['integral']:.3f})",
                        alpha=0.8)
        
        ax1.set_title(f'{dataset_info["description"]}')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Error metrics
        ax2 = plt.subplot(n_datasets, 3, idx * 3 + 2)
        
        method_names = []
        mse_values = []
        kl_values = []
        
        for method_name, method_result in results['methods'].items():
            if method_result is not None:
                method_names.append(method_name)
                mse_values.append(method_result['mse'])
                kl_values.append(abs(method_result['kl_divergence']))
        
        x_pos = np.arange(len(method_names))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, mse_values, width, label='MSE', alpha=0.8)
        
        # Use secondary y-axis for KL divergence
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x_pos + width/2, kl_values, width, 
                            label='|KL Divergence|', alpha=0.8, color='orange')
        
        ax2.set_xlabel('Method')
        ax2.set_ylabel('MSE', color='C0')
        ax2_twin.set_ylabel('|KL Divergence|', color='orange')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(method_names, rotation=45, ha='right')
        ax2.tick_params(axis='y', labelcolor='C0')
        ax2_twin.tick_params(axis='y', labelcolor='orange')
        
        # Add legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax2.set_title('Error Metrics')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Timing comparison
        ax3 = plt.subplot(n_datasets, 3, idx * 3 + 3)
        
        times = [method_result['time'] * 1000 for method_name, method_result 
                in results['methods'].items() if method_result is not None]
        
        bars = ax3.bar(x_pos, times, alpha=0.8)
        
        # Color bars by method
        for i, bar in enumerate(bars):
            bar.set_color(method_colors[i])
        
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Time (ms)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(method_names, rotation=45, ha='right')
        ax3.set_title('Computation Time')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, times)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('kde_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive comparison plot as 'kde_comprehensive_comparison.png'")
    
    return fig

def create_performance_summary_table(datasets_results):
    """Create a summary table of performance metrics"""
    
    # Prepare data for table
    summary_data = []
    
    for dataset_name, results in datasets_results.items():
        for method_name, method_result in results['methods'].items():
            if method_result is not None:
                summary_data.append({
                    'Dataset': dataset_name,
                    'Method': method_name,
                    'Time (ms)': method_result['time'] * 1000,
                    'Integral': method_result['integral'],
                    'MSE': method_result['mse'],
                    '|KL Div|': abs(method_result['kl_divergence'])
                })
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Convert to table format
    import pandas as pd
    df = pd.DataFrame(summary_data)
    
    # Create pivot tables for better visualization
    pivot_time = df.pivot(index='Method', columns='Dataset', values='Time (ms)')
    pivot_integral = df.pivot(index='Method', columns='Dataset', values='Integral')
    pivot_mse = df.pivot(index='Method', columns='Dataset', values='MSE')
    
    # Plot tables
    table_data = []
    for method in pivot_time.index:
        row = [method]
        for dataset in pivot_time.columns:
            time_val = pivot_time.loc[method, dataset]
            integral_val = pivot_integral.loc[method, dataset]
            mse_val = pivot_mse.loc[method, dataset]
            
            cell_text = f"T: {time_val:.1f}ms\nI: {integral_val:.3f}\nMSE: {mse_val:.4f}"
            row.append(cell_text)
        table_data.append(row)
    
    table = ax.table(cellText=table_data,
                    colLabels=['Method'] + list(pivot_time.columns),
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(len(pivot_time.columns) + 1):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 0:  # Method column
                    cell.set_facecolor('#d4d4d4')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#f0f0f0')
    
    plt.title('KDE Methods Performance Summary\n(T: Time, I: Integral, MSE: Mean Squared Error)', 
              fontsize=14, pad=20)
    plt.savefig('kde_performance_summary.png', dpi=300, bbox_inches='tight')
    print("Saved performance summary table as 'kde_performance_summary.png'")
    
    return fig

def test_scalability():
    """Test scalability with different dataset sizes"""
    sizes = [100, 500, 1000, 5000, 10000, 50000]
    methods = {
        'Original DCT': lambda d: kde(d, n=128),
        'FFT (Simple)': lambda d: kde_gaussian(d, n=128, method='fft'),
        'cdist': lambda d: kde_gaussian(d, n=128, method='cdist'),
        'cdist_chunked': lambda d: kde_gaussian(d, n=128, method='cdist_chunked')
    }
    
    results = {method: [] for method in methods}
    
    print("\nScalability Test:")
    for size in sizes:
        print(f"\nTesting with {size} samples...")
        data = torch.randn(size)
        
        for method_name, method_func in methods.items():
            if size > 20000 and 'cdist' in method_name and 'chunked' not in method_name:
                # Skip non-chunked cdist for large datasets
                results[method_name].append(np.nan)
                continue
                
            start_time = time.time()
            try:
                _, _ = method_func(data)
                elapsed = time.time() - start_time
                results[method_name].append(elapsed * 1000)  # Convert to ms
                print(f"  {method_name}: {elapsed*1000:.2f} ms")
            except Exception as e:
                print(f"  {method_name}: Failed - {e}")
                results[method_name].append(np.nan)
    
    # Plot scalability results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_name, times in results.items():
        valid_mask = ~np.isnan(times)
        ax.plot(np.array(sizes)[valid_mask], np.array(times)[valid_mask], 
               marker='o', linewidth=2, markersize=8, label=method_name)
    
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title('KDE Methods Scalability Comparison')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('kde_scalability_comparison.png', dpi=300)
    print("\nSaved scalability plot as 'kde_scalability_comparison.png'")
    
    return results

def main():
    """Run comprehensive KDE comparison tests"""
    print("=== Comprehensive KDE Methods Comparison ===\n")
    
    # Generate test datasets
    datasets = generate_test_datasets()
    
    # Evaluate all methods on all datasets
    datasets_results = {}
    
    for dataset_name, dataset_info in datasets.items():
        print(f"\nProcessing {dataset_name} dataset...")
        
        # Create evaluation grid
        data = dataset_info['data']
        x_min = data.min().item() - 1
        x_max = data.max().item() + 1
        x_eval = torch.linspace(x_min, x_max, 200)
        
        # Evaluate methods
        method_results = evaluate_kde_methods(data, dataset_info['true_pdf'], x_eval)
        
        datasets_results[dataset_name] = {
            'methods': method_results,
            'x_eval': x_eval
        }
        
        # Print summary
        print(f"\n{dataset_name} Results Summary:")
        for method_name, result in method_results.items():
            if result is not None:
                print(f"  {method_name:20s}: Time={result['time']*1000:6.2f}ms, "
                      f"Integral={result['integral']:.4f}, MSE={result['mse']:.6f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_comparison_results(datasets_results, datasets)
    create_performance_summary_table(datasets_results)
    
    # Test scalability
    print("\nTesting scalability...")
    scalability_results = test_scalability()
    
    print("\n=== Comparison Complete ===")
    print("Generated files:")
    print("  - kde_comprehensive_comparison.png")
    print("  - kde_performance_summary.png")
    print("  - kde_scalability_comparison.png")

if __name__ == "__main__":
    main() 