"""
Debug log-likelihood calculation issues in DVC PyTorch
"""

import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Import PyTorch version
from classes.objects import vine_obj_bin, margin_obj
from grid.grid_op import create_grids
from param.cond_copula import copulapdf

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

def test_copula_pdf():
    """Test if copula PDF functions are working correctly"""
    print("\n" + "="*60)
    print("Testing Copula PDF Functions")
    print("="*60)
    
    # Test data in uniform space
    u_test = np.array([[0.3, 0.7], [0.5, 0.5], [0.1, 0.9], [0.8, 0.2]]).reshape(-1, 2, 1)
    
    # Test Gaussian copula
    from classes.objects import cop_par_obj
    
    # Independence copula (rho = 0)
    cop_ind = cop_par_obj('ind', [])
    pdf_ind = copulapdf(cop_ind, u_test)
    print(f"\nIndependence copula PDF: {pdf_ind.flatten()}")
    print(f"Expected: all 1.0")
    
    # Gaussian copula with correlation
    cop_gauss = cop_par_obj('gaussian', [0.7])
    pdf_gauss = copulapdf(cop_gauss, u_test)
    print(f"\nGaussian copula (rho=0.7) PDF: {pdf_gauss.flatten()}")
    print(f"Min: {pdf_gauss.min():.4f}, Max: {pdf_gauss.max():.4f}")
    
    # Clayton copula
    cop_clayton = cop_par_obj('clayton', [2.0])
    pdf_clayton = copulapdf(cop_clayton, u_test)
    print(f"\nClayton copula (theta=2.0) PDF: {pdf_clayton.flatten()}")
    print(f"Min: {pdf_clayton.min():.4f}, Max: {pdf_clayton.max():.4f}")
    
    # Check for NaN or negative values
    for name, pdf in [('Independence', pdf_ind), ('Gaussian', pdf_gauss), ('Clayton', pdf_clayton)]:
        pdf_np = pdf.cpu().numpy() if torch.is_tensor(pdf) else pdf
        if np.any(np.isnan(pdf_np)):
            print(f"WARNING: {name} copula PDF contains NaN!")
        if np.any(pdf_np <= 0):
            print(f"WARNING: {name} copula PDF contains non-positive values!")

def test_simple_vine():
    """Test a simple 2D vine copula"""
    print("\n" + "="*60)
    print("Testing Simple 2D Vine Copula")
    print("="*60)
    
    # Generate simple 2D data with known correlation
    n_samples = 500
    rho = 0.7
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    data = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Convert to uniform margins
    data_uniform = np.zeros_like(data)
    for i in range(2):
        data_uniform[:, i] = stats.rankdata(data[:, i]) / (n_samples + 1)
    
    # Convert to PyTorch
    data_torch = torch.tensor(data_uniform, dtype=torch.float32, device=device)
    
    # True Kendall's tau
    true_tau, _ = stats.kendalltau(data[:, 0], data[:, 1])
    print(f"\nTrue Kendall's tau: {true_tau:.4f}")
    print(f"True Pearson correlation: {rho}")
    
    # Create margins
    margins = [margin_obj('empirical', None, True) for _ in range(2)]
    
    # Create vine copula
    vine = vine_obj_bin(
        vine_family='c-vine',
        families=['gaussian'],
        vine_depth=1,
        margin=margins,
        knots=32,
        method=None
    )
    
    # Create grids
    vine.grid_u, vine.grid_s, vine.grid_x = create_grids(vine.knots, device=device)
    
    # Fit
    gen_dict = {
        'binning': False,
        'parallel': False,
        'param': True,
        'vine_depth': 1
    }
    par_dict = {
        'param_families': ['gaussian', 'student', 'ind']
    }
    bin_dict = {'n_bin': 1}
    
    print("\nFitting vine copula...")
    vine.fit(data_torch, gen_dict, {}, par_dict, bin_dict)
    
    # Check fitted parameters
    if vine.copulas and len(vine.copulas) > 0:
        cop = vine.copulas[0][0]
        print(f"\nSelected copula family: {cop.family}")
        print(f"Fitted parameter: {cop.theta}")
        
        if cop.family == 'gaussian':
            fitted_rho = cop.theta if np.isscalar(cop.theta) else cop.theta[0]
            print(f"Fitted correlation: {fitted_rho:.4f}")
            print(f"Error: {abs(fitted_rho - rho):.4f}")
    
    # Evaluate on test data
    test_indices = torch.randperm(n_samples)[:10]
    test_data = data_torch[test_indices]
    
    print("\nEvaluating on test data...")
    print(f"Test data shape: {test_data.shape}")
    
    # Debug evaluation step by step
    p, p_cop, log_p = vine.evaluation(test_data)
    
    print(f"\nProbability densities:")
    print(f"Total p: {p[:5]}")
    print(f"Copula p: {p_cop[:5]}")
    print(f"Log p: {log_p[:5]}")
    
    # Check marginal log-likelihoods
    if hasattr(vine, 'logf'):
        marginal_loglik = torch.sum(vine.logf[:, :, 0], dim=1)
        print(f"\nMarginal log-likelihood: {marginal_loglik[:5]}")
        
        if vine.logf.shape[2] > 1:
            copula_loglik = torch.sum(torch.sum(vine.logf[:, :, 1:], dim=2), dim=1)
            print(f"Copula log-likelihood: {copula_loglik[:5]}")
    
    # Compute expected log-likelihood for independence
    expected_loglik_ind = 0.0  # log(1) = 0 for independence copula
    print(f"\nExpected log-likelihood for independence: {expected_loglik_ind}")
    
    # For Gaussian copula, compute analytical log-likelihood
    if vine.copulas[0][0].family == 'gaussian':
        rho_fit = vine.copulas[0][0].theta if np.isscalar(vine.copulas[0][0].theta) else vine.copulas[0][0].theta[0]
        # Transform to normal space
        z1 = stats.norm.ppf(test_data[:, 0].cpu().numpy())
        z2 = stats.norm.ppf(test_data[:, 1].cpu().numpy())
        
        # Gaussian copula log-density
        log_det = -0.5 * np.log(1 - rho_fit**2)
        quad_form = -0.5 * rho_fit**2 / (1 - rho_fit**2) * (z1**2 + z2**2 - 2*rho_fit*z1*z2)
        expected_copula_loglik = log_det + quad_form
        
        print(f"\nAnalytical Gaussian copula log-likelihood: {expected_copula_loglik[:5]}")

def test_marginal_densities():
    """Test marginal density estimation"""
    print("\n" + "="*60)
    print("Testing Marginal Density Estimation")
    print("="*60)
    
    # Generate data from known distribution
    n_samples = 1000
    data = np.random.normal(0, 1, n_samples)
    
    # Compute kernel density estimate
    from utils.prob_op import kernel_pdf2
    den, mden = kernel_pdf2(torch.tensor(data, dtype=torch.float32))
    
    # Evaluate at test points
    test_points = np.linspace(-3, 3, 10)
    
    # Use interpolation
    from utils.interpolation import interp1d_np
    test_tensor = torch.tensor(test_points, dtype=torch.float32)
    kde_values = interp1d_np(test_tensor, mden, den)
    
    # True density
    true_density = stats.norm.pdf(test_points)
    
    print(f"\nTest points: {test_points}")
    print(f"KDE values: {kde_values}")
    print(f"True density: {true_density}")
    print(f"Mean absolute error: {np.mean(np.abs(kde_values.numpy() - true_density)):.4f}")

if __name__ == "__main__":
    # Run tests
    test_copula_pdf()
    test_marginal_densities()
    test_simple_vine()
    
    print("\n" + "="*60)
    print("Debug completed!")
    print("="*60) 