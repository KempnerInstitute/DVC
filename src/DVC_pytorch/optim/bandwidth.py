import torch

############################ BANDWIDTH RULE THUMB ######################

def bandwidth_mul(data, deg, n_cop):
    """
    Rule of thumb for computing the bandwidth for kernel density estimation
    
    Args:
        data: Data tensor of shape (n_samples, 2, n_cop)
        deg: Degree parameter (typically 2 for bivariate)
        n_cop: Number of copulas
        
    Returns:
        bw: Bandwidth tensor of shape (2, n_cop)
    """
    device = data.device
    dtype = data.dtype
    n = data.shape[0]
    
    # Center the data
    xc = data - torch.mean(data, dim=0)
    
    # Handle case where data is 2D
    if xc.dim() == 2:
        xc = xc.unsqueeze(-1)
    
    # Compute Cholesky decomposition for each copula
    chol_list = []
    for jj in range(n_cop):
        # Compute covariance matrix
        c1 = torch.matmul(xc[:, :, jj].t(), xc[:, :, jj]) / (n - 1)
        # Compute Cholesky decomposition
        try:
            chol1 = torch.linalg.cholesky(c1).t()
        except:
            # If Cholesky fails, add small diagonal term for numerical stability
            c1 = c1 + 1e-6 * torch.eye(c1.shape[0], device=device, dtype=dtype)
            chol1 = torch.linalg.cholesky(c1).t()
        chol_list.append(chol1)
    
    # Stack choleskys
    chol = torch.stack(chol_list)
    chol = chol.permute(1, 2, 0)
    
    # Silverman's rule of thumb for bandwidth
    bw = 5 * n**(-1 / (4 * deg + 2)) * chol
    
    # Extract diagonal elements
    bw1 = bw[0, 0, :]
    bw2 = bw[1, 1, :]
    bw = torch.stack([bw1, bw2])
    
    # Scale factor
    bw = bw / 10
    
    return bw 