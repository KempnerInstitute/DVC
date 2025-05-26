import torch
import numpy as np
from typing import Optional, Tuple, Union
from scipy.special import gamma, digamma
from scipy.optimize import minimize_scalar


class ParametricCopula:
    """Base class for parametric copula families"""
    
    def __init__(self, family: str, theta: Union[float, torch.Tensor], 
                 bounds: Optional[Tuple[float, float]] = None):
        self.family = family
        self.theta = theta
        self.bounds = bounds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def pdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Copula density c(u,v)"""
        raise NotImplementedError
        
    def cdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Copula CDF C(u,v)"""
        raise NotImplementedError
        
    def h_function(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """h-function h(v|u) = ∂C(u,v)/∂u"""
        raise NotImplementedError
        
    def h_inverse(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Inverse h-function"""
        raise NotImplementedError
        
    def log_pdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Log copula density"""
        return torch.log(self.pdf(u, v) + 1e-20)
        
    def tau_to_theta(self, tau: float) -> float:
        """Convert Kendall's tau to copula parameter"""
        raise NotImplementedError
        
    def theta_to_tau(self) -> float:
        """Convert copula parameter to Kendall's tau"""
        raise NotImplementedError


class GaussianCopula(ParametricCopula):
    """Gaussian (normal) copula"""
    
    def __init__(self, rho: float):
        super().__init__('gaussian', rho, bounds=(-0.999, 0.999))
        self.rho = torch.tensor(rho, device=self.device)
        
    def pdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Gaussian copula density"""
        # Clamp to avoid numerical issues
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        # Ensure rho is on the same device as input
        device = u.device
        self.rho = self.rho.to(device)
        
        # Inverse normal CDF
        normal = torch.distributions.Normal(0, 1)
        x = normal.icdf(u)
        y = normal.icdf(v)
        
        # Copula density
        rho2 = self.rho * self.rho
        factor = 1 / torch.sqrt(1 - rho2)
        exponent = -(rho2 * (x**2 + y**2) - 2*self.rho*x*y) / (2*(1-rho2))
        
        return factor * torch.exp(exponent)
        
    def cdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Gaussian copula CDF - requires numerical integration"""
        # Simplified: use independence approximation for now
        # Full implementation would use bivariate normal CDF
        return u * v  # TODO: Implement proper bivariate normal CDF
        
    def h_function(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """h-function for Gaussian copula"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        normal = torch.distributions.Normal(0, 1)
        z1 = normal.icdf(u)
        z2 = normal.icdf(v)
        
        # h(v|u) = Φ((z2 - ρz1) / sqrt(1 - ρ²))
        h = normal.cdf((z2 - self.rho * z1) / torch.sqrt(1 - self.rho**2))
        return torch.clamp(h, 1e-6, 1-1e-6)
        
    def h_inverse(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Inverse h-function for Gaussian copula"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        h = torch.clamp(h, 1e-6, 1-1e-6)
        
        normal = torch.distributions.Normal(0, 1)
        z1 = normal.icdf(u)
        z_h = normal.icdf(h)
        
        # v = Φ(ρz1 + sqrt(1-ρ²) * Φ^(-1)(h))
        z2 = self.rho * z1 + torch.sqrt(1 - self.rho**2) * z_h
        v = normal.cdf(z2)
        
        return torch.clamp(v, 1e-6, 1-1e-6)
        
    def tau_to_theta(self, tau: float) -> float:
        """Convert Kendall's tau to correlation parameter"""
        return np.sin(np.pi * tau / 2)
        
    def theta_to_tau(self) -> float:
        """Convert correlation to Kendall's tau"""
        return 2 * np.arcsin(self.rho.item()) / np.pi


class StudentCopula(ParametricCopula):
    """Student-t copula"""
    
    def __init__(self, rho: float, nu: float):
        super().__init__('student', torch.tensor([rho, nu]), 
                        bounds=[(-0.999, 0.999), (2.0, 30.0)])
        self.rho = torch.tensor(rho, device=self.device)
        self.nu = torch.tensor(nu, device=self.device)
        
    def pdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Student-t copula density"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        # Ensure tensors are on the same device
        device = u.device
        self.rho = self.rho.to(device)
        self.nu = self.nu.to(device)
        
        # Inverse Student-t CDF using scipy
        from scipy import stats
        u_np = u.cpu().numpy()
        v_np = v.cpu().numpy()
        nu_val = self.nu.cpu().item()
        
        t1_np = stats.t.ppf(u_np, nu_val)
        t2_np = stats.t.ppf(v_np, nu_val)
        
        t1 = torch.tensor(t1_np, device=device, dtype=u.dtype)
        t2 = torch.tensor(t2_np, device=device, dtype=v.dtype)
        
        # Copula density
        rho2 = self.rho * self.rho
        factor1 = 1 / torch.sqrt(1 - rho2)
        
        # Bivariate Student-t density / product of marginals
        s = (t1**2 + t2**2 - 2*self.rho*t1*t2) / (self.nu * (1 - rho2))
        
        factor2 = torch.lgamma((self.nu + 2)/2) - torch.lgamma(self.nu/2)
        factor2 = torch.exp(factor2)
        factor2 *= (1 + s)**(-(self.nu + 2)/2)
        
        # Remove marginal densities
        marg1 = (1 + t1**2/self.nu)**(-(self.nu + 1)/2)
        marg2 = (1 + t2**2/self.nu)**(-(self.nu + 1)/2)
        
        return factor1 * factor2 / (marg1 * marg2)
        
    def cdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Student-t copula CDF"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        # Ensure tensors are on the same device
        device = u.device
        self.rho = self.rho.to(device)
        self.nu = self.nu.to(device)
        
        # Inverse Student-t CDF using scipy
        from scipy import stats
        u_np = u.cpu().numpy()
        v_np = v.cpu().numpy()
        nu_val = self.nu.cpu().item()
        
        t1_np = stats.t.ppf(u_np, nu_val)
        t2_np = stats.t.ppf(v_np, nu_val)
        
        t1 = torch.tensor(t1_np, device=device, dtype=u.dtype)
        t2 = torch.tensor(t2_np, device=device, dtype=v.dtype)
        
        # Copula CDF
        normal = torch.distributions.Normal(0, 1)
        z1 = normal.icdf(u)
        z2 = normal.icdf(v)
        
        # Bivariate Student-t CDF
        t_dist = torch.distributions.StudentT(self.nu)
        C = t_dist.cdf((t1**2 + t2**2 - 2*self.rho*t1*t2) / torch.sqrt(1 - self.rho**2))
        
        return C
        
    def h_function(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """h-function for Student-t copula"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        # Ensure tensors are on the same device
        device = u.device
        self.rho = self.rho.to(device)
        self.nu = self.nu.to(device)
        
        # Use scipy for inverse CDF
        from scipy import stats
        u_np = u.cpu().numpy()
        v_np = v.cpu().numpy()
        nu_val = self.nu.cpu().item()
        
        t1_np = stats.t.ppf(u_np, nu_val)
        t2_np = stats.t.ppf(v_np, nu_val)
        
        t1 = torch.tensor(t1_np, device=device, dtype=u.dtype)
        t2 = torch.tensor(t2_np, device=device, dtype=v.dtype)
        
        # h(v|u) = T_{ν+1}((t2 - ρt1) / sqrt((ν + t1²)(1 - ρ²) / (ν + 1)))
        numerator = t2 - self.rho * t1
        denominator = torch.sqrt((self.nu + t1**2) * (1 - self.rho**2) / (self.nu + 1))
        
        # Use scipy for CDF
        arg_np = (numerator / denominator).cpu().numpy()
        h_np = stats.t.cdf(arg_np, nu_val + 1)
        h = torch.tensor(h_np, device=device, dtype=u.dtype)
        
        return torch.clamp(h, 1e-6, 1-1e-6)
        
    def h_inverse(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Inverse h-function for Student-t copula"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        h = torch.clamp(h, 1e-6, 1-1e-6)
        
        # Ensure tensors are on the same device
        device = u.device
        self.rho = self.rho.to(device)
        self.nu = self.nu.to(device)
        
        # Use scipy for inverse CDF
        from scipy import stats
        u_np = u.cpu().numpy()
        h_np = stats.t.ppf(h.cpu().numpy(), self.nu.cpu().item())
        h = torch.tensor(h_np, device=device, dtype=u.dtype)
        
        # v = Φ(ρz1 + sqrt(1-ρ²) * Φ^(-1)(h))
        normal = torch.distributions.Normal(0, 1)
        z1 = normal.icdf(u)
        z_h = normal.icdf(h)
        
        z2 = self.rho * z1 + torch.sqrt(1 - self.rho**2) * z_h
        v = normal.cdf(z2)
        
        return torch.clamp(v, 1e-6, 1-1e-6)
        
    def tau_to_theta(self, tau: float) -> float:
        """Convert Kendall's tau to Student-t parameter"""
        return 2 * np.arcsin(self.rho.item()) / np.pi
        
    def theta_to_tau(self) -> float:
        """Kendall's tau for Student-t copula"""
        return 2 * np.arcsin(self.rho.item()) / np.pi


class ClaytonCopula(ParametricCopula):
    """Clayton copula"""
    
    def __init__(self, theta: float):
        super().__init__('clayton', theta, bounds=(0.0, 20.0))
        self.theta = torch.tensor(theta, device=self.device)
        
    def pdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Clayton copula density"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        if self.theta < 1e-6:
            # Independence
            return torch.ones_like(u)
            
        # c(u,v) = (1+θ)(uv)^(-θ-1)(u^(-θ) + v^(-θ) - 1)^(-2-1/θ)
        term1 = (1 + self.theta)
        term2 = (u * v)**(-self.theta - 1)
        term3 = (u**(-self.theta) + v**(-self.theta) - 1)**(-2 - 1/self.theta)
        
        return term1 * term2 * term3
        
    def cdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Clayton copula CDF"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        if self.theta < 1e-6:
            return u * v
            
        # C(u,v) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)
        return torch.pow(u**(-self.theta) + v**(-self.theta) - 1, -1/self.theta)
        
    def h_function(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """h-function for Clayton copula"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        if self.theta < 1e-6:
            return v
            
        # h(v|u) = u^(-θ-1) * (u^(-θ) + v^(-θ) - 1)^(-1/θ - 1)
        term1 = u**(-self.theta - 1)
        term2 = (u**(-self.theta) + v**(-self.theta) - 1)**(-1/self.theta - 1)
        
        return torch.clamp(term1 * term2, 1e-6, 1-1e-6)
        
    def tau_to_theta(self, tau: float) -> float:
        """Convert Kendall's tau to Clayton parameter"""
        return 2 * tau / (1 - tau)
        
    def theta_to_tau(self) -> float:
        """Kendall's tau for Clayton copula"""
        return self.theta.item() / (self.theta.item() + 2)


class GumbelCopula(ParametricCopula):
    """Gumbel copula"""
    
    def __init__(self, theta: float):
        super().__init__('gumbel', theta, bounds=(1.0, 20.0))
        self.theta = torch.tensor(theta, device=self.device)
        
    def pdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Gumbel copula density"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        if abs(self.theta - 1) < 1e-6:
            return torch.ones_like(u)
            
        # Transform to log scale
        log_u = -torch.log(u)
        log_v = -torch.log(v)
        
        # A = (log_u^θ + log_v^θ)^(1/θ)
        A = torch.pow(log_u**self.theta + log_v**self.theta, 1/self.theta)
        
        # Copula CDF
        C = torch.exp(-A)
        
        # Derivatives
        dA_du = log_u**(self.theta-1) / (u * A**(self.theta-1))
        dA_dv = log_v**(self.theta-1) / (v * A**(self.theta-1))
        d2A_dudv = -(self.theta-1) * log_u**(self.theta-1) * log_v**(self.theta-1) / \
                   (u * v * A**(2*self.theta-1))
        
        # Copula density
        c = C * (dA_du * dA_dv + d2A_dudv)
        
        return c
        
    def cdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Gumbel copula CDF"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        if abs(self.theta - 1) < 1e-6:
            return u * v
            
        # C(u,v) = exp(-((-log u)^θ + (-log v)^θ)^(1/θ))
        log_u = -torch.log(u)
        log_v = -torch.log(v)
        A = torch.pow(log_u**self.theta + log_v**self.theta, 1/self.theta)
        
        return torch.exp(-A)
        
    def h_function(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """h-function for Gumbel copula"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        if abs(self.theta - 1) < 1e-6:
            return v
            
        C = self.cdf(u, v)
        log_u = -torch.log(u)
        log_v = -torch.log(v)
        
        A = torch.pow(log_u**self.theta + log_v**self.theta, 1/self.theta)
        
        # h(v|u) = ∂C/∂u = C * (log_u)^(θ-1) / (u * A^(θ-1))
        h = C * log_u**(self.theta-1) / (u * A**(self.theta-1))
        
        return torch.clamp(h, 1e-6, 1-1e-6)
        
    def tau_to_theta(self, tau: float) -> float:
        """Convert Kendall's tau to Gumbel parameter"""
        return 1 / (1 - tau)
        
    def theta_to_tau(self) -> float:
        """Kendall's tau for Gumbel copula"""
        return 1 - 1/self.theta.item()


class FrankCopula(ParametricCopula):
    """Frank copula"""
    
    def __init__(self, theta: float):
        super().__init__('frank', theta, bounds=(-30.0, 30.0))
        self.theta = torch.tensor(theta, device=self.device)
        
    def pdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Frank copula density"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        if abs(self.theta) < 1e-6:
            return torch.ones_like(u)
            
        # c(u,v) = θ(1-e^(-θ))e^(-θ(u+v)) / ((1-e^(-θ)) - (1-e^(-θu))(1-e^(-θv)))²
        exp_theta = torch.exp(-self.theta)
        exp_theta_u = torch.exp(-self.theta * u)
        exp_theta_v = torch.exp(-self.theta * v)
        
        numerator = self.theta * (1 - exp_theta) * torch.exp(-self.theta * (u + v))
        denominator = ((1 - exp_theta) - (1 - exp_theta_u) * (1 - exp_theta_v))**2
        
        return numerator / denominator
        
    def cdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Frank copula CDF"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        if abs(self.theta) < 1e-6:
            return u * v
            
        # C(u,v) = -1/θ * log(1 + (e^(-θu)-1)(e^(-θv)-1)/(e^(-θ)-1))
        exp_theta = torch.exp(-self.theta)
        exp_theta_u = torch.exp(-self.theta * u)
        exp_theta_v = torch.exp(-self.theta * v)
        
        term = 1 + (exp_theta_u - 1) * (exp_theta_v - 1) / (exp_theta - 1)
        
        return -torch.log(term) / self.theta
        
    def h_function(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """h-function for Frank copula"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        if abs(self.theta) < 1e-6:
            return v
            
        exp_theta = torch.exp(-self.theta)
        exp_theta_u = torch.exp(-self.theta * u)
        exp_theta_v = torch.exp(-self.theta * v)
        
        # h(v|u) = e^(-θu)(e^(-θv)-1) / ((e^(-θ)-1) + (e^(-θu)-1)(e^(-θv)-1))
        numerator = exp_theta_u * (exp_theta_v - 1)
        denominator = (exp_theta - 1) + (exp_theta_u - 1) * (exp_theta_v - 1)
        
        return torch.clamp(numerator / denominator, 1e-6, 1-1e-6)
        
    def tau_to_theta(self, tau: float) -> float:
        """Convert Kendall's tau to Frank parameter (numerical)"""
        if abs(tau) < 1e-6:
            return 0.0
            
        # Use numerical optimization
        def objective(theta):
            if abs(theta) < 1e-6:
                return tau**2
            debye = (1/theta) * (1 - 1/theta + 1/theta**2 * (1 - np.exp(-theta)))
            return (1 - 4/theta * debye - tau)**2
            
        result = minimize_scalar(objective, bounds=(-30, 30), method='bounded')
        return result.x
        
    def theta_to_tau(self) -> float:
        """Kendall's tau for Frank copula"""
        theta = self.theta.item()
        if abs(theta) < 1e-6:
            return 0.0
            
        # Debye function D₁(θ) = 1/θ ∫₀^θ t/(e^t-1) dt
        # Approximation: D₁(θ) ≈ 1 - θ/4 + θ²/36 - θ⁴/3600
        if abs(theta) < 1:
            debye = 1 - theta/4 + theta**2/36 - theta**4/3600
        else:
            # Integral approximation
            debye = (1/theta) * (1 - 1/theta + 1/theta**2 * (1 - np.exp(-theta)))
            
        return 1 - 4/theta * debye


class JoeCopula(ParametricCopula):
    """Joe copula"""
    
    def __init__(self, theta: float):
        super().__init__('joe', theta, bounds=(1.0, 30.0))
        self.theta = torch.tensor(theta, device=self.device)
        
    def pdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Joe copula density"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        if abs(self.theta - 1) < 1e-6:
            return torch.ones_like(u)
            
        # Transform
        u_bar = 1 - u
        v_bar = 1 - v
        
        # A = (u_bar^θ + v_bar^θ - u_bar^θ * v_bar^θ)
        A = u_bar**self.theta + v_bar**self.theta - u_bar**self.theta * v_bar**self.theta
        
        # Copula CDF
        C = 1 - A**(1/self.theta)
        
        # Density (complex expression)
        term1 = A**(-2 + 1/self.theta)
        term2 = u_bar**(self.theta-1) * v_bar**(self.theta-1)
        term3 = (self.theta - 1) * A + u_bar**self.theta * v_bar**self.theta
        
        return term1 * term2 * term3
        
    def cdf(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Joe copula CDF"""
        u = torch.clamp(u, 1e-6, 1-1e-6)
        v = torch.clamp(v, 1e-6, 1-1e-6)
        
        if abs(self.theta - 1) < 1e-6:
            return u * v
            
        # C(u,v) = 1 - ((1-u)^θ + (1-v)^θ - (1-u)^θ(1-v)^θ)^(1/θ)
        u_bar = 1 - u
        v_bar = 1 - v
        
        A = u_bar**self.theta + v_bar**self.theta - u_bar**self.theta * v_bar**self.theta
        
        return 1 - A**(1/self.theta)
        
    def tau_to_theta(self, tau: float) -> float:
        """Convert Kendall's tau to Joe parameter (numerical)"""
        if abs(tau) < 1e-6:
            return 1.0
            
        # Use approximation for Joe copula
        # τ ≈ 1 - 4/(θ(θ+2)) for large θ
        # Solving: θ² + 2θ - 4/(1-τ) = 0
        a = 1
        b = 2
        c = -4/(1-tau)
        
        theta = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        return max(1.0, theta)
        
    def theta_to_tau(self) -> float:
        """Kendall's tau for Joe copula"""
        theta = self.theta.item()
        if abs(theta - 1) < 1e-6:
            return 0.0
            
        # Numerical integration or approximation
        # For Joe copula: τ = 1 - 4 ∑_{k=1}^∞ 1/(k(θk+2)(θ(k-1)+2))
        # Approximation for large θ: τ ≈ 1 - 4/(θ(θ+2))
        
        if theta > 5:
            return 1 - 4/(theta * (theta + 2))
        else:
            # Use series approximation
            tau = 0
            for k in range(1, 100):
                term = 1 / (k * (theta*k + 2) * (theta*(k-1) + 2))
                tau += term
                if term < 1e-10:
                    break
            return 1 - 4 * tau


def create_copula(family: str, theta: Union[float, list, torch.Tensor]) -> ParametricCopula:
    """
    Factory function to create copula objects
    
    Args:
        family: Copula family name
        theta: Parameter(s) for the copula
        
    Returns:
        ParametricCopula instance
    """
    family = family.lower()
    
    if family in ['gaussian', 'normal']:
        return GaussianCopula(float(theta))
    elif family in ['student', 't']:
        if isinstance(theta, (list, tuple, torch.Tensor)):
            return StudentCopula(float(theta[0]), float(theta[1]))
        else:
            raise ValueError("Student copula requires [rho, nu] parameters")
    elif family == 'clayton':
        return ClaytonCopula(float(theta))
    elif family == 'gumbel':
        return GumbelCopula(float(theta))
    elif family == 'frank':
        return FrankCopula(float(theta))
    elif family == 'joe':
        return JoeCopula(float(theta))
    else:
        raise ValueError(f"Unknown copula family: {family}")


def fit_copula_mle(family: str, u: torch.Tensor, v: torch.Tensor, 
                   init_tau: Optional[float] = None) -> ParametricCopula:
    """
    Fit copula parameters using maximum likelihood estimation
    
    Args:
        family: Copula family name
        u, v: Uniform marginal data
        init_tau: Initial Kendall's tau for starting values
        
    Returns:
        Fitted copula object
    """
    from scipy.optimize import minimize
    
    # Get initial parameter from Kendall's tau if not provided
    if init_tau is None:
        # Estimate empirical Kendall's tau
        n = len(u)
        tau = 0
        for i in range(n):
            for j in range(i+1, n):
                tau += torch.sign((u[i] - u[j]) * (v[i] - v[j]))
        init_tau = 2 * tau.item() / (n * (n-1))
    
    # Get initial parameter value
    temp_copula = create_copula(family, 0.5 if family == 'gaussian' else 2.0)
    init_theta = temp_copula.tau_to_theta(init_tau)
    
    # Define negative log-likelihood
    def neg_loglik(theta):
        try:
            if family in ['student', 't']:
                cop = create_copula(family, [theta[0], theta[1]])
            else:
                cop = create_copula(family, float(theta))
            
            log_pdf = cop.log_pdf(u, v)
            return -torch.sum(log_pdf).item()
        except:
            return 1e10
    
    # Set bounds based on family
    if family in ['gaussian', 'normal']:
        bounds = [(-0.99, 0.99)]
    elif family in ['student', 't']:
        bounds = [(-0.99, 0.99), (2.0, 30.0)]
        init_theta = [init_theta, 5.0]  # Default nu = 5
    elif family == 'clayton':
        bounds = [(0.01, 20.0)]
    elif family == 'gumbel':
        bounds = [(1.01, 20.0)]
    elif family == 'frank':
        bounds = [(-30.0, 30.0)]
    elif family == 'joe':
        bounds = [(1.01, 30.0)]
    else:
        bounds = None
    
    # Optimize
    result = minimize(neg_loglik, init_theta, method='L-BFGS-B', bounds=bounds)
    
    # Create fitted copula
    if family in ['student', 't']:
        return create_copula(family, result.x)
    else:
        return create_copula(family, float(result.x)) 