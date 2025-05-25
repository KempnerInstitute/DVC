import torch
import numpy as np

class NormalCDF:
    """Normal CDF bijector"""
    def __init__(self, loc, scale):
        # Ensure loc and scale are tensors
        if not torch.is_tensor(loc):
            loc = torch.tensor(float(loc), dtype=torch.float32)
        if not torch.is_tensor(scale):
            scale = torch.tensor(float(scale), dtype=torch.float32)
        self.loc = loc
        self.scale = scale
        self.normal_dist = torch.distributions.Normal(loc, scale)
        self.name = "NormalCDF"
    
    def forward(self, y):
        """Compute Phi^{-1}(y) = InverseCDF(y) - forward transforms from uniform to normal"""
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)
        return self.normal_dist.icdf(y)
    
    def inverse(self, x):
        """Compute Phi(x) = CDF(x) - inverse transforms from normal to uniform"""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.normal_dist.cdf(x)
    
    def inverse_log_det_jacobian(self, x):
        """Log PDF of the normal distribution"""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.normal_dist.log_prob(x)
    
    def __call__(self, x, inverse=False):
        """Allow bijector to be called directly"""
        if inverse:
            return self.inverse(x)
        else:
            return self.forward(x)


class GammaCDF:
    """Gamma CDF bijector"""
    def __init__(self, alpha, beta):
        # Ensure alpha and beta are tensors
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(float(alpha), dtype=torch.float32)
        if not torch.is_tensor(beta):
            beta = torch.tensor(float(beta), dtype=torch.float32)
        self.alpha = alpha
        self.beta = beta
        self.gamma_dist = torch.distributions.Gamma(alpha, beta)
        self.name = "GammaCDF"
    
    def forward(self, y):
        """Compute InverseCDF(y) - forward transforms from uniform to gamma"""
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)
        return self.gamma_dist.icdf(y)
    
    def inverse(self, x):
        """Compute CDF(x) - inverse transforms from gamma to uniform"""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.gamma_dist.cdf(x)
    
    def inverse_log_det_jacobian(self, x):
        """Log PDF of the gamma distribution"""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.gamma_dist.log_prob(x)
    
    def __call__(self, x, inverse=False):
        """Allow bijector to be called directly"""
        if inverse:
            return self.inverse(x)
        else:
            return self.forward(x) 