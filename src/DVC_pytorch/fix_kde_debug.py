"""Debug KDE NaN issue"""
import torch
from utils.prob_op import kde, dct1d, idct1d, fixed_point, find_root_secant
from utils.tensor_op import replace_negative
import math

# Generate test data
torch.manual_seed(42)
data = torch.randn(100)

# Run KDE step by step
device = data.device
dtype = data.dtype

MIN = torch.min(data)
MAX = torch.max(data)
N = 128

print(f"Data range: [{MIN:.3f}, {MAX:.3f}]")

pi = torch.tensor(math.pi, dtype=dtype, device=device)
R = MAX - MIN
print(f"R = {R:.3f}")

nbins = N
xmesh = torch.linspace(0, R, 128, dtype=dtype, device=device) + MIN

# Get unique values
data_unique, _ = torch.unique(data, sorted=True, return_inverse=True)
N_samples = torch.ceil(torch.tensor((data.shape[0] - 1) / 2, dtype=dtype, device=device)) * 2
N_samples = N_samples.to(torch.int32)
print(f"N_samples: {N_samples}")

# Compute histogram
counts = torch.histc(data, bins=128, min=MIN, max=MAX)
print(f"Histogram counts - min: {counts.min()}, max: {counts.max()}, sum: {counts.sum()}")
print(f"Number of non-zero bins: {(counts > 0).sum()}")

init_data = counts / N_samples.float()
print(f"After first norm - sum: {init_data.sum()}")

init_data = init_data / torch.sum(init_data)
print(f"After second norm - sum: {init_data.sum()}")
print(f"init_data has NaN: {torch.any(torch.isnan(init_data))}")

# Apply DCT
a = dct1d(init_data)
print(f"DCT result - has NaN: {torch.any(torch.isnan(a))}")
print(f"DCT first few values: {a[:5]}")

I = torch.square(torch.arange(1, 128, 1, dtype=dtype, device=device))
a2 = torch.square(a[1:] / 2)
print(f"a2 - min: {a2.min()}, max: {a2.max()}")
print(f"a2 has NaN: {torch.any(torch.isnan(a2))}")
print(f"a2 has zero: {torch.any(a2 == 0)}")

N_float = N_samples.float()

# Find optimal bandwidth using fixed point iteration
tol = 1e-12 + 0.01 * (N_float - 50) / 1000

# Test fixed point function
test_t = torch.tensor(0.1, dtype=dtype, device=device)
fp_result = fixed_point(test_t, N_float, I, a2)
print(f"Fixed point test: {fp_result}")
print(f"Fixed point has NaN: {torch.isnan(fp_result)}")

# Simple secant method implementation for finding root
t_star = find_root_secant(lambda t: fixed_point(t, N_float, I, a2), tol, device=device, dtype=dtype)
print(f"t_star: {t_star}")
print(f"t_star has NaN: {torch.isnan(t_star)}")

# Apply bandwidth
a_1 = torch.exp(-(torch.arange(0, nbins, dtype=dtype, device=device)**2 * (pi**2) * t_star) / 2)
print(f"a_1 - min: {a_1.min()}, max: {a_1.max()}")
print(f"a_1 has NaN: {torch.any(torch.isnan(a_1))}")

a_t = a * a_1
print(f"a_t has NaN: {torch.any(torch.isnan(a_t))}")

# Apply inverse DCT
density = idct1d(a_t) / R
print(f"density has NaN: {torch.any(torch.isnan(density))}")
print(f"density - min: {density.min()}, max: {density.max()}")

# Ensure non-negative
if torch.any(density < 0):
    eps = torch.finfo(dtype).eps
    density = replace_negative(density, eps)
    print(f"After replace_negative - has NaN: {torch.any(torch.isnan(density))}") 