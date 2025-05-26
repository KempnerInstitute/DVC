import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.prob_op import dct1d, idct1d
import math

# Generate test data
np.random.seed(42)
n_samples = 500
data_np = np.random.randn(n_samples)
data = torch.tensor(data_np, dtype=torch.float32)

# Setup
n = 128
MIN = data.min() - 0.5
MAX = data.max() + 0.5
R = MAX - MIN

# Create histogram
counts = torch.histc(data, bins=n, min=MIN.item(), max=MAX.item())
initial_data = counts / counts.sum()

# DCT
a = dct1d(initial_data)
I = torch.arange(1, n, dtype=torch.float32) ** 2
a2 = (a[1:] / 2) ** 2

# Get actual sample size
N = data.shape[0]

print(f"N (unique values): {N}")
print(f"a2 stats: min={a2.min():.6f}, max={a2.max():.6f}, mean={a2.mean():.6f}")

# Define the fixed-point equation
def fixed_point_kde(t):
    l = 7
    pi = torch.tensor(math.pi, dtype=torch.float32)
    N_tensor = torch.tensor(N, dtype=torch.float32)
    
    # Initial f calculation with numerical stability
    exp_term = -I * pi**2 * t
    exp_term = torch.clamp(exp_term, min=-50.0)  # Prevent underflow
    f = 2 * torch.pow(pi, 2*l) * torch.sum(torch.pow(I, l) * a2 * torch.exp(exp_term))
    
    if f <= 0:
        print(f"  Warning: f <= 0 at t={t:.6f}, f={f:.6e}")
        return torch.tensor(float('inf'))
    
    for s in range(l-1, 1, -1):
        s_tensor = torch.tensor(s, dtype=torch.float32)
        K0 = torch.exp(torch.lgamma(s_tensor + 1) - torch.lgamma(s_tensor/2 + 1) - 0.5 * torch.log(2 * pi))
        const = (1 + torch.pow(torch.tensor(0.5), s_tensor + 0.5)) / 3
        
        time_arg = 2 * const * K0 / N_tensor / f
        if time_arg <= 0:
            print(f"  Warning: time_arg <= 0 at s={s}, time_arg={time_arg:.6e}")
            return torch.tensor(float('inf'))
            
        time = torch.pow(time_arg, 2 / (3 + 2*s_tensor))
        
        exp_term = -I * pi**2 * time
        exp_term = torch.clamp(exp_term, min=-50.0)
        f = 2 * torch.pow(pi, 2*s_tensor) * torch.sum(torch.pow(I, s_tensor) * a2 * torch.exp(exp_term))
        
        if f <= 0:
            print(f"  Warning: f <= 0 at s={s}, f={f:.6e}")
            return torch.tensor(float('inf'))
    
    final_arg = 2 * N_tensor * torch.sqrt(pi) * f
    if final_arg <= 0:
        print(f"  Warning: final_arg <= 0, final_arg={final_arg:.6e}")
        return torch.tensor(float('inf'))
        
    out = t - torch.pow(final_arg, -2/5)
    return out

# Test the fixed point function
print("\nTesting fixed point function:")
t_values = np.logspace(-6, 0, 50)
fp_values = []

for t in t_values:
    fp_val = fixed_point_kde(torch.tensor(t, dtype=torch.float32))
    fp_values.append(fp_val.item())
    if abs(fp_val) < 0.001:
        print(f"  t={t:.6f}, f(t)={fp_val:.6f} <-- Near zero!")

fp_values = np.array(fp_values)

# Plot the fixed point function
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.semilogx(t_values, fp_values)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Fixed point function')
plt.grid(True)

# Zoom in on the region near zero
plt.subplot(2, 2, 2)
mask = np.abs(fp_values) < 1.0
if np.any(mask):
    plt.plot(t_values[mask], fp_values[mask])
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.title('Fixed point function (zoomed)')
    plt.grid(True)

# Test different bandwidth values on the KDE
plt.subplot(2, 2, 3)
xmesh = MIN + R * torch.arange(n, dtype=torch.float32) / n
t_test_values = [0.0001, 0.001, 0.01, 0.1]
colors = ['red', 'green', 'blue', 'orange']

for t, color in zip(t_test_values, colors):
    a_t = a * torch.exp(-torch.arange(n, dtype=torch.float32)**2 * np.pi**2 * t / 2)
    density = idct1d(a_t) / R
    plt.plot(xmesh.numpy(), density.numpy(), color=color, label=f't={t}')

plt.legend()
plt.title('KDE with different t values')
plt.xlabel('x')
plt.ylabel('Density')

# Find the root using bisection
from utils.prob_op import find_root_bisection
t_star = find_root_bisection(fixed_point_kde, low=0.0, high=1.0, dtype=torch.float32)
print(f"\nOptimal t_star from bisection: {t_star:.6f}")

# Plot the optimal KDE
plt.subplot(2, 2, 4)
a_t_opt = a * torch.exp(-torch.arange(n, dtype=torch.float32)**2 * np.pi**2 * t_star / 2)
density_opt = idct1d(a_t_opt) / R

plt.hist(data_np, bins=30, density=True, alpha=0.5, label='Histogram')
plt.plot(xmesh.numpy(), density_opt.numpy(), 'r-', label=f'KDE (t*={t_star:.4f})', linewidth=2)

# True distribution
from scipy.stats import norm
x_true = np.linspace(-4, 4, 100)
y_true = norm.pdf(x_true, loc=0, scale=1)
plt.plot(x_true, y_true, 'k--', label='True N(0,1)')
plt.legend()
plt.title('Final KDE result')

plt.tight_layout()
plt.savefig('bandwidth_debug.png')
print("\nPlot saved as bandwidth_debug.png")

# Check the area under the curve
dx = xmesh[1] - xmesh[0]
area = (density_opt * dx).sum()
print(f"\nArea under KDE curve: {area:.6f}") 