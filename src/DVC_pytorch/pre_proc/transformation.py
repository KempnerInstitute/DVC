import torch
from utils.bijector import NormalCDF
from utils.tensor_op import check_bound3

class Transform(object):
    """Transform object for copula transformations"""
    
    def __init__(self, n_cop):
        """Create a Transform object.
        Args:
            n_cop: Number of copulas
        """
        self.n_cop = n_cop
        self.mu = None
        self.coeff = None
    
    def forward_u(self, obj_u):
        """Transform from uniform (u,v) space to standard normal (s,r) space"""
        loc = torch.tensor(0.0, dtype=obj_u.dtype, device=obj_u.device)
        scale = torch.tensor(1.0, dtype=obj_u.dtype, device=obj_u.device)
        
        normal_cdf = NormalCDF(loc, scale)
        obj_s = normal_cdf.forward(obj_u)
        
        # Bound check
        obj_s = check_bound3(obj_s, 
                           torch.tensor(3.2, dtype=obj_u.dtype, device=obj_u.device),
                           torch.tensor(-3.2, dtype=obj_u.dtype, device=obj_u.device))
        return obj_s
    
    def forward_s(self, obj_s):
        """Transform from (s,r) space to rotated (x,y) space using PCA"""
        device = obj_s.device
        dtype = obj_s.dtype
        
        # Compute PCA coefficients if not already computed
        if self.coeff is None:
            coeff_list = []
            
            for i in range(self.n_cop):
                # SVD decomposition
                u, s, v = torch.linalg.svd(obj_s[:, :, i])
                coeff_list.append(v)
            
            coeff = torch.stack(coeff_list, dim=2)
            
            # Enforce positive maximum
            coeff1_list = []
            for i in range(self.n_cop):
                # Find index of maximum absolute value
                ind_p = torch.argmax(torch.abs(coeff[:, :, i]))
                # Get the row index
                row_idx = ind_p // coeff.shape[1]
                
                # Get the sign of the maximum value in that row
                max_val = coeff[row_idx, :, i]
                sign_val = torch.sign(torch.max(torch.abs(max_val)))
                
                # Apply sign to ensure positive maximum
                coeff2 = sign_val * coeff[:, :, i]
                coeff1_list.append(coeff2)
            
            self.coeff = torch.stack(coeff1_list, dim=2)
        
        # Compute mean if not already computed
        if self.mu is None:
            self.mu = torch.mean(obj_s, dim=0)
        
        # Handle 2D case
        if obj_s.dim() == 2:
            obj_s = obj_s.unsqueeze(-1)
            obj_s = obj_s.repeat(1, 1, self.n_cop)
        
        # Center the data
        mu1 = self.mu.unsqueeze(0).repeat(obj_s.shape[0], 1, 1)
        
        # Apply rotation
        obj_x_list = []
        for i in range(self.n_cop):
            # Matrix multiplication for rotation
            data_x1 = torch.matmul(obj_s[:, :, i] - mu1[:, :, i], self.coeff[:, :, i])
            obj_x_list.append(data_x1)
        
        obj_x = torch.stack(obj_x_list, dim=2)
        
        return obj_x 