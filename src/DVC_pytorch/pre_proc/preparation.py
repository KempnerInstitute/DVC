import torch
import numpy as np
from scipy import stats
from utils.tensor_op import uniquetol

def prep_cop(x, vine1, sort_n):
    """Prepare copula data with optional sorting"""
    d = x.shape[1]
    device = x.device
    dtype = x.dtype
    
    if sort_n == 'sort':
        # Calculate correlations
        corr = torch.zeros(d, d, dtype=dtype, device=device)
        for i in range(d):
            for j in range(d):
                # Convert to numpy for kendalltau
                x_i = x[:, i].cpu().numpy()
                x_j = x[:, j].cpu().numpy()
                tau, _ = stats.kendalltau(x_i, x_j)
                corr[i, j] = abs(tau)
        
        # Order columns by correlation
        ord1 = [0]
        for i in range(1, d):
            # Find remaining indices
            remaining = list(set(range(d)) - set(ord1))
            if remaining:
                # Get correlations with last selected column
                corr_vals = [corr[ord1[-1], j] for j in remaining]
                # Find index with maximum correlation
                max_idx = remaining[np.argmax(corr_vals)]
                ord1.append(max_idx)
        
        ord1 = torch.tensor(ord1, dtype=torch.long, device=device)
    else:
        ord1 = torch.arange(d, dtype=torch.long, device=device)
    
    # Reorder columns
    x = x[:, ord1]
    
    # Prepare margins
    e = torch.empty_like(x)
    for i in range(d):
        x_col = x[:, i]
        e[:, i] = prep_copula(x_col, 0)
        vine1.margin[i].ker = e[:, i].cpu().numpy()
    
    return e

def prep_copula(X_new, tim):
    """Prepare single copula margin"""
    device = X_new.device
    dtype = X_new.dtype
    
    # Generate uniform samples
    u_dist = torch.distributions.Uniform(
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(1.0, dtype=dtype, device=device)
    )
    samples = u_dist.sample((X_new.shape[0],))
    
    margin1 = torch.zeros_like(X_new)
    
    # Check if all values are the same
    unique_vals = uniquetol(X_new, 1e-5)
    if len(unique_vals) == 1:
        margin1 = X_new + samples * 1e-10
    else:
        # Get unique sorted values
        ad1, inverse_indices = torch.unique(X_new, sorted=True, return_inverse=True)
        
        # Calculate differences
        ad = torch.zeros_like(ad1)
        ad[:-1] = ad1[1:] - ad1[:-1]
        ad[-1] = ad[-2] if len(ad) > 1 else 1e-10
        
        # Select batch size based on data size
        data_size = X_new.shape[0]
        if data_size < 500:
            batch_size = 1
        elif data_size < 1000:
            batch_size = 2
        elif data_size < 4000:
            batch_size = 10
        elif data_size < 10000:
            batch_size = 20
        elif data_size < 20000:
            batch_size = 50
        elif data_size < 100000:
            batch_size = 100
        else:
            batch_size = 200
        
        # Process in batches
        Di2 = torch.zeros_like(X_new)
        batch_len = data_size // batch_size
        
        for j in range(batch_size):
            start_idx = batch_len * j
            if j == batch_size - 1:
                # Last batch takes remaining elements
                end_idx = data_size
            else:
                end_idx = batch_len * (j + 1)
            
            # Get batch
            batch_indices = inverse_indices[start_idx:end_idx]
            
            # Map to differences
            Di2[start_idx:end_idx] = ad[batch_indices]
        
        # Add uniform noise proportional to NN distance
        if tim == 0:
            margin1 = X_new + Di2 * samples * 1e-10
        elif tim == 1:
            margin1 = X_new + Di2 * samples
    
    return margin1 