"""
Complete Example of PyTorch DVC Implementation
Demonstrates all major functionality including fitting, evaluation, 
entropy estimation, mutual information, and sampling.
"""

import numpy as np
import sys
sys.path.append('src')

from DVC_pyolder import (
    vine_obj_bin, margin_obj, prep_cop, 
    fit_vine, vine_entropy, mutual_information,
    predict_vine, predict_conditional
)

def generate_correlated_data(n_samples=1000, dim=5):
    """Generate synthetic data with known correlation structure"""
    # Create correlation matrix
    rho = 0.6
    corr_matrix = np.eye(dim)
    for i in range(dim):
        for j in range(dim):
            if i != j:
                corr_matrix[i, j] = rho ** abs(i - j)
    
    # Generate multivariate normal data
    mean = np.zeros(dim)
    data = np.random.multivariate_normal(mean, corr_matrix, size=n_samples)
    
    return data.astype(np.float32)

def main():
    print("=" * 60)
    print("PyTorch DVC Complete Example")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    np.random.seed(42)
    data = generate_correlated_data(n_samples=500, dim=4)
    print(f"   Data shape: {data.shape}")
    print(f"   Data range: [{data.min():.2f}, {data.max():.2f}]")
    
    # 2. Create and fit vine copula
    print("\n2. Creating and fitting vine copula...")
    
    # Create vine object
    vine = vine_obj_bin(
        vine_family='d-vine',    # or 'c-vine', 'r-vine'
        families=['gaussian', 'ind', 'clayton'],  # Available copula families
        vine_depth=data.shape[1],                 # Number of dimensions
        margin=[],                                # Will be set below
        knots=50                                  # Grid size
    )
    
    # Set additional properties
    vine.param = True        # Use parametric copulas
    vine.binning = False     # No binning
    
    # Set margins (assuming normal margins)
    for i in range(data.shape[1]):
        vine.margin.append(margin_obj('norm', [0, 1], True))
    
    # Configuration dictionaries
    gen_dict = {
        "parallel": True,
        "param": True,      # Use parametric copulas
        "binning": False    # No binning
    }
    npc_dict = {"method": "local", "n_iter": 100}
    par_dict = {"param_families": ["gaussian", "ind", "clayton"]}
    bin_dict = {"n_bin": 5}
    
    # Fit the vine
    fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
    print("   ✓ Vine fitting complete")
    print(f"   ✓ Vine structure: {vine.vine_family}")
    print(f"   ✓ Number of copulas fitted: {sum(len(level) for level in vine.copulas)}")
    
    # Show fitted copula families
    print("\n   Fitted copula families by tree level:")
    for level_idx, level_copulas in enumerate(vine.copulas):
        families = [cop.family for cop in level_copulas]
        print(f"     Tree {level_idx}: {families}")
    
    # 3. Evaluate the vine on test points
    print("\n3. Evaluating vine on test points...")
    test_data = generate_correlated_data(n_samples=100, dim=4)
    p, p_copula, log_marg = vine.evaluation(test_data)
    print(f"   ✓ Mean density: {p.mean().item():.4f}")
    print(f"   ✓ Mean copula density: {p_copula.mean().item():.4f}")
    
    # 4. Estimate entropy
    print("\n4. Estimating vine entropy...")
    info_dict = {
        'alpha': 0.05,
        'cases': 1000,
        'iterations': 10
    }
    entropy = vine_entropy(vine, info_dict)
    print(f"   ✓ Estimated entropy: {entropy:.4f}")
    
    # 5. Estimate mutual information
    print("\n5. Estimating mutual information...")
    # MI between first two variables and last two variables
    mi = mutual_information(vine, [0, 1], [2, 3], info_dict)
    print(f"   ✓ MI between variables [0,1] and [2,3]: {mi:.4f}")
    
    # 6. Generate samples from the vine
    print("\n6. Generating samples from fitted vine...")
    n_samples_gen = 1000
    samples = vine.sample(n_samples_gen)
    print(f"   ✓ Generated {n_samples_gen} samples")
    print(f"   ✓ Sample shape: {samples.shape}")
    print(f"   ✓ Sample mean: {samples.mean(axis=0)}")
    print(f"   ✓ Sample std: {samples.std(axis=0)}")
    
    # 7. Conditional prediction
    print("\n7. Conditional prediction...")
    # Predict variables 2,3 given values for variables 0,1
    given_data = np.array([[0.5, -0.5], [1.0, 0.0], [-0.5, 0.5]], dtype=np.float32)
    predicted = predict_conditional(vine, given_data, [0, 1], [2, 3])
    print(f"   ✓ Conditional prediction implemented!")
    print(f"   Given variables [0,1]:")
    for i, obs in enumerate(given_data):
        print(f"     Observation {i+1}: {obs}")
        print(f"     Predicted [2,3]: {predicted[i]}")
    
    # Also get quantiles
    from DVC_pyolder import predict_conditional_quantiles
    quantiles = predict_conditional_quantiles(vine, given_data[:1], [0, 1], [2, 3], 
                                            quantiles=[0.1, 0.5, 0.9], n_samples=2000)
    print(f"\n   Prediction quantiles for first observation:")
    for q, values in quantiles.items():
        print(f"     {q*100:.0f}% quantile: {values[0]}")
    
    # 8. Compare empirical and fitted correlations
    print("\n8. Correlation comparison...")
    print("   Empirical correlation matrix:")
    emp_corr = np.corrcoef(data.T)
    print(emp_corr.round(3))
    
    print("\n   Fitted copula correlations (Kendall's tau):")
    # For Gaussian copulas, rho ≈ sin(π/2 * τ)
    for level_idx, level_copulas in enumerate(vine.copulas):
        for edge_idx, cop in enumerate(level_copulas):
            if hasattr(cop, 'family') and cop.family == 'gaussian':
                edge = vine.ind_vine[level_idx][edge_idx]
                rho = cop.theta[0] if isinstance(cop.theta, list) else cop.theta
                tau = 2/np.pi * np.arcsin(rho)
                print(f"     Edge {edge}: ρ={rho:.3f}, τ={tau:.3f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main() 