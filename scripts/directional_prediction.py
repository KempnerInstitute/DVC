import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time

from DVC.config import load_config
from DVC.objects import vine_obj_bin, margin_obj
from scipy.stats import multivariate_normal

# ------------------------------------------------------------
# Load configuration and data
# ------------------------------------------------------------
CFG_PATH = Path(__file__).parent.parent / "configs" / "gauss_nd.yaml"
cfg = load_config(CFG_PATH if CFG_PATH.exists() else None)

# Data generation parameters
n_samples = cfg['data']['n_samples']
dim = cfg['data']['dim'] 
rho = cfg['data']['rho']

print(f"Generating {n_samples} samples from {dim}D Gaussian with rho={rho}")

# Synthetic Gaussian data
cov_true = np.full((dim, dim), rho)
np.fill_diagonal(cov_true, 1.0)
data = np.random.multivariate_normal(np.zeros(dim), cov_true, size=n_samples)

# ------------------------------------------------------------
# Build and fit vine model
# ------------------------------------------------------------
margins = [margin_obj('norm', [0.0, 1.0], True) for _ in range(dim)]

vine = vine_obj_bin(
    cfg['vine']['family'],
    ['gaussian'],
    dim,
    margins,
    knots=cfg['vine']['knots'],
    method=cfg['vine']['method']
)

# Fit dictionaries
gen_dict = {
    'param': cfg['general']['param'],
    'binning': cfg['general']['binning'],
    'fitted': False
}

npc_dict = cfg.get('npc', {})
par_dict = {'param_families': ['gaussian']}
bin_dict = {'n_bin': 1}

print("Fitting vine model...")
vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, cfg)

# ------------------------------------------------------------
# Directional Prediction (following the tree order)
# ------------------------------------------------------------
normal = torch.distributions.Normal(0., 1.)

def predict_next_variable(vine, previous_values, target_idx):
    """
    Predict the next variable in the vine sequence given previously observed values.
    
    This function implements maximum likelihood estimation along the vine structure
    by following the vine's natural conditioning path.
    
    Parameters:
    -----------
    vine : vine_obj_bin
        The fitted vine object
    previous_values : list
        Values of variables that have already been observed/conditioned on
    target_idx : int
        Index of the variable to predict
    
    Returns:
    --------
    float
        The predicted value for the target variable
    """
    # For a C-vine, the ordering is naturally defined - variable 0 is 
    # the root, then 1, 2, etc. We'll check if the structure matches this expectation.
    if vine.vine_family == 'c-vine':
        conditioning_indices = list(range(len(previous_values)))
    else:
        # Get the natural ordering from the vine structure
        conditioning_indices = []
        for lvl in range(len(vine.ind_vine)):
            for edge in vine.ind_vine[lvl]:
                for idx in edge:
                    if idx not in conditioning_indices and idx != target_idx:
                        conditioning_indices.append(idx)
        
        # Ensure we only use previously observed variables
        conditioning_indices = conditioning_indices[:len(previous_values)]
    
    # Create test points with different possible values for target_idx
    search_range = np.linspace(-3, 3, 100)
    log_probs = []
    
    # Evaluate likelihood of each potential value
    for val in search_range:
        x = np.zeros(dim)
        for i, prev_idx in enumerate(conditioning_indices):
            x[prev_idx] = previous_values[i]
        x[target_idx] = val
        
        # Convert to tensor for vine.logpdf
        x_tensor = torch.tensor([x], dtype=torch.float32)
        log_prob = vine.logpdf(x_tensor).item()
        log_probs.append(log_prob)
    
    # Find maximum likelihood value
    ml_idx = np.nanargmax(log_probs)
    return search_range[ml_idx]

# Conditional predictions for the Gaussian case (ground truth)
def gaussian_conditional_mean(x_given, indices_given, target_idx, cov):
    """Compute conditional mean of a multivariate Gaussian."""
    sigma_11 = cov[target_idx, target_idx]
    sigma_12 = cov[target_idx, indices_given]
    sigma_22 = cov[np.ix_(indices_given, indices_given)]
    
    # Compute conditional mean: mu_1|2 = Sigma_12 * Sigma_22^(-1) * x_2
    # (means are zero in our case)
    sigma_22_inv = np.linalg.inv(sigma_22)
    conditional_mean = sigma_12.dot(sigma_22_inv).dot(x_given)
    
    return conditional_mean

# ---------------------------------------------------------------
# Experiment 1: Prediction along the vine path
# ---------------------------------------------------------------
print("\n--- Experiment 1: Sequential prediction along vine path ---")

# Generate test data
test_size = 100
test_data = np.random.multivariate_normal(np.zeros(dim), cov_true, size=test_size)

# Prepare arrays for true and predicted values
true_values = []
pred_values_vine = []
pred_values_gauss = []

# Measure timing
vine_time = 0
gauss_time = 0

# For each test point, predict one variable at a time in sequence
for test_idx in range(test_size):
    observed_values = []
    observed_indices = []
    
    # Predict each variable in sequence
    if vine.vine_family == 'c-vine':
        # For C-vine, the root is first (typically 0), 
        # then each variable conditioned on the root, etc.
        prediction_order = list(range(dim))
    else:
        # For other vines, get order from structure or use default
        prediction_order = []
        for lvl in range(len(vine.ind_vine)):
            for edge in vine.ind_vine[lvl]:
                for idx in edge:
                    if idx not in prediction_order:
                        prediction_order.append(idx)
        
        # Ensure all variables are included
        for i in range(dim):
            if i not in prediction_order:
                prediction_order.append(i)
    
    for i, var_idx in enumerate(prediction_order):
        if i < 1:  # Skip first variable (nothing to predict from)
            observed_values.append(test_data[test_idx, var_idx])
            observed_indices.append(var_idx)
            continue
        
        # Record true value
        true_val = test_data[test_idx, var_idx]
        true_values.append(true_val)
        
        # Vine prediction
        start_time = time.time()
        vine_pred = predict_next_variable(vine, 
                                          [test_data[test_idx, idx] for idx in observed_indices], 
                                          var_idx)
        vine_time += time.time() - start_time
        pred_values_vine.append(vine_pred)
        
        # Gaussian prediction (ground truth)
        start_time = time.time()
        gauss_pred = gaussian_conditional_mean(
            test_data[test_idx, observed_indices],
            observed_indices,
            var_idx,
            cov_true
        )
        gauss_time += time.time() - start_time
        pred_values_gauss.append(gauss_pred)
        
        # Update observed values for next prediction
        observed_values.append(test_data[test_idx, var_idx])
        observed_indices.append(var_idx)

# Convert to numpy arrays
true_values = np.array(true_values)
pred_values_vine = np.array(pred_values_vine)
pred_values_gauss = np.array(pred_values_gauss)

# Calculate MSE
mse_vine = np.mean((pred_values_vine - true_values)**2)
mse_gauss = np.mean((pred_values_gauss - true_values)**2)

print(f"Number of predictions: {len(true_values)}")
print(f"Vine model MSE: {mse_vine:.4f} (time: {vine_time:.2f}s)")
print(f"True Gaussian MSE: {mse_gauss:.4f} (time: {gauss_time:.2f}s)")
print(f"Ratio vine/true MSE: {mse_vine/mse_gauss:.2f}x")

# Plot prediction results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(true_values, pred_values_vine, alpha=0.5, label='Vine prediction')
plt.scatter(true_values, pred_values_gauss, alpha=0.5, label='Gaussian prediction')
plt.plot([-3, 3], [-3, 3], 'k--', label='Perfect prediction')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Prediction Comparison')
plt.legend()
plt.grid(True)

# Plot error distribution
plt.subplot(1, 2, 2)
plt.hist(pred_values_vine - true_values, bins=20, alpha=0.5, label='Vine error')
plt.hist(pred_values_gauss - true_values, bins=20, alpha=0.5, label='Gaussian error')
plt.xlabel('Prediction error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.legend()
plt.tight_layout()
plt.savefig('prediction_paths.png')
print("Saved prediction path plots to 'prediction_paths.png'")

# ---------------------------------------------------------------
# Experiment 2: Varying prediction difficulty with dimension
# ---------------------------------------------------------------
print("\n--- Experiment 2: Prediction difficulty with dimension ---")

# Select range of dimensions to test
dim_range = range(1, min(5, dim))
mse_by_dim_vine = []
mse_by_dim_gauss = []

# For each prediction dimension (how many variables we condition on)
for pred_dim in dim_range:
    true_vals = []
    vine_preds = []
    gauss_preds = []
    
    # For each test point
    for test_idx in range(test_size):
        # We'll always predict the last dimension, conditioning on pred_dim previous ones
        conditioning_vars = list(range(pred_dim))
        target_var = pred_dim
        
        # True value
        true_val = test_data[test_idx, target_var]
        true_vals.append(true_val)
        
        # Vine prediction
        vine_pred = predict_next_variable(vine, 
                                         [test_data[test_idx, idx] for idx in conditioning_vars],
                                         target_var)
        vine_preds.append(vine_pred)
        
        # Gaussian prediction
        gauss_pred = gaussian_conditional_mean(
            test_data[test_idx, conditioning_vars],
            conditioning_vars,
            target_var,
            cov_true
        )
        gauss_preds.append(gauss_pred)
    
    # Calculate MSE for this dimension
    mse_vine = np.mean(np.array([(p - t)**2 for p, t in zip(vine_preds, true_vals)]))
    mse_gauss = np.mean(np.array([(p - t)**2 for p, t in zip(gauss_preds, true_vals)]))
    
    mse_by_dim_vine.append(mse_vine)
    mse_by_dim_gauss.append(mse_gauss)
    
    print(f"Dimension {pred_dim}: Vine MSE={mse_vine:.4f}, Gauss MSE={mse_gauss:.4f}, Ratio={mse_vine/mse_gauss:.2f}x")

# Plot MSE vs dimension
plt.figure(figsize=(10, 6))
plt.plot(list(dim_range), mse_by_dim_vine, 'o-', label='Vine model')
plt.plot(list(dim_range), mse_by_dim_gauss, 'o-', label='Gaussian model')
plt.xlabel('Number of conditioning variables')
plt.ylabel('MSE')
plt.title('Prediction Error vs. Dimension')
plt.legend()
plt.grid(True)
plt.savefig('prediction_by_dimension.png')
print("Saved dimension scaling plot to 'prediction_by_dimension.png'")

plt.show() 