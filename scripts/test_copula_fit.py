import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from DVC_pyolder.param_copula import fit_gaussian, copulapdf, copulainvccdf
from DVC_pyolder.objects import cop_par_obj

# Generate bivariate Gaussian data with known correlation
def generate_test_data(n_samples=5000, rho=0.6):
    """Generate bivariate Gaussian test data with correlation rho"""
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    data = np.random.multivariate_normal(mean, cov, size=n_samples)
    return data

def transform_to_copula(data):
    """Transform data to uniform margins (empirical CDF)"""
    n = data.shape[0]
    u_data = np.zeros_like(data)
    for j in range(data.shape[1]):
        u_data[:, j] = np.argsort(np.argsort(data[:, j])) / n
    return u_data

def print_debug_stats(data, u_data, rho_hat, rho_true):
    """Print various statistics for debugging"""
    # Original correlation
    corr_data = np.corrcoef(data, rowvar=False)[0,1]
    
    # Rank correlation and Kendall's tau
    rank_corr = np.corrcoef(np.argsort(data[:,0]), np.argsort(data[:,1]))[0,1]
    tau, _ = kendalltau(data[:,0], data[:,1])
    
    # Uniform margins correlation
    corr_u = np.corrcoef(u_data, rowvar=False)[0,1]
    
    print("\nDiagnostic Statistics:")
    print(f"True rho: {rho_true}")
    print(f"Estimated rho: {rho_hat}")
    print(f"Pearson corr in original data: {corr_data:.4f}")
    print(f"Pearson corr in uniform margins: {corr_u:.4f}")
    print(f"Rank correlation: {rank_corr:.4f}")
    print(f"Kendall's tau: {tau:.4f}")
    print(f"Theoretical relation tau = (2/π)arcsin(ρ) gives approx. rho = {np.sin(np.pi*tau/2):.4f}")

# Test different correlation values
correlations = [0.3, 0.6, 0.9, -0.5]
plt.figure(figsize=(12, 10))

for i, rho in enumerate(correlations):
    # Generate data
    data = generate_test_data(n_samples=5000, rho=rho)
    u_data = transform_to_copula(data)
    
    # Convert to torch tensor for the fit_gaussian function
    u_tensor = torch.tensor(u_data, dtype=torch.float32)
    
    # Fit Gaussian copula
    rho_hat, loglik, aic = fit_gaussian(u_tensor)
    
    # Print diagnostics
    print(f"\n===== Test for rho = {rho} =====")
    print_debug_stats(data, u_data, rho_hat, rho)
    
    # Create copula object
    cop = cop_par_obj("gaussian", rho_hat)
    
    # Generate samples from the fitted copula
    n_samples = 1000
    samples = np.zeros((n_samples, 2))
    samples[:, 0] = np.random.rand(n_samples)  # First margin is U[0,1]
    
    # Generate conditional samples
    for i in range(n_samples):
        uv = torch.tensor([[samples[i, 0], 0.5]], dtype=torch.float32)  # Dummy v value
        samples[i, 1] = copulainvccdf(cop, uv).item()
    
    # Plot results
    plt.subplot(2, 2, i+1)
    
    # Original uniform data
    plt.scatter(u_data[:500, 0], u_data[:500, 1], alpha=0.3, label='Original')
    
    # Generated samples
    plt.scatter(samples[:500, 0], samples[:500, 1], alpha=0.3, label='Generated')
    
    plt.title(f"True ρ = {rho}, Estimated ρ = {rho_hat:.4f}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("U₁")
    plt.ylabel("U₂")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig('copula_fit_test.png')
print("\nSaved test plot to 'copula_fit_test.png'")

# Test with original data values (should be near true value)
print("\n\n======= DIRECT CORRELATION ESTIMATE (SHOULD MATCH TRUE VALUE) =======")
for rho in [0.3, 0.6, 0.9, -0.5]:
    # Generate Gaussian data
    data = generate_test_data(n_samples=5000, rho=rho)
    
    # Directly compute correlation (z-transform)
    z = np.column_stack([
        (data[:, 0] - np.mean(data[:, 0])) / np.std(data[:, 0]),
        (data[:, 1] - np.mean(data[:, 1])) / np.std(data[:, 1])
    ])
    direct_rho = np.corrcoef(z, rowvar=False)[0, 1]
    
    print(f"True rho: {rho}, Direct normal score estimate: {direct_rho:.4f}, Diff: {direct_rho-rho:.4f}")

plt.show() 