# KDE Fix Summary for DVC PyTorch Implementation

## Issue Identified
The kernel density estimation (KDE) was producing flat densities (nearly zero variance) due to incorrect bandwidth selection. The primary issue was:

```python
# WRONG: Using number of unique values
N = len(torch.unique(data))

# CORRECT: Using actual sample size
N = data.shape[0]
```

## Root Cause
When using `len(torch.unique(data))` instead of `data.shape[0]`:
- For data with many repeated values (e.g., integer data), N would be very small
- This caused the bandwidth selection algorithm to think it had a tiny dataset
- The algorithm would then select an extremely large bandwidth
- This resulted in over-smoothing, producing essentially flat densities

## Fixes Applied

### 1. Fixed N Calculation in `kde()` function
**File**: `utils/prob_op.py`, line 293
```python
# Changed from:
N = len(torch.unique(data))
# To:
N = data.shape[0]
```

### 2. Updated DCT Implementation
The DCT implementation was updated to match TensorFlow's FFT-based approach:
```python
def dct1d(data):
    """Discrete cosine transform using FFT - matching TensorFlow implementation"""
    n = data.shape[0]
    if n == 1:
        return data
    
    # Extend the data by mirroring
    extended = torch.cat([data, torch.flip(data[1:n-1], dims=[0])], dim=0)
    
    # Perform FFT
    result = torch.fft.fft(extended.to(complex_dtype))
    result = torch.real(result)
    
    return result[:n]
```

### 3. Improved Root Finding Algorithm
Enhanced the `find_root_bisection()` function with:
- Better handling of numerical edge cases
- Checking for invalid function values (inf/nan)
- Adaptive interval selection when no sign change is found
- More robust evaluation at boundary points

## Test Results

### Before Fix:
- Density variance: ~1e-14 to 1e-17 (essentially flat)
- Bandwidth selection: Often hit upper bound (1.0)
- Fixed point function: Returned `-inf` for many values

### After Fix:
- Density variance: ~1e-3 to 1e-7 (proper variation)
- Bandwidth selection: Reasonable values (e.g., 0.000036)
- Fixed point function: Well-behaved across entire range

## Verification
The fix was verified with:
1. **Repeated values test**: Data with only 3 unique values but 900 samples
2. **Normal distribution test**: Standard Gaussian data
3. **Integer data test**: Random integers 0-9

All tests now produce proper density estimates with appropriate variance.

## Additional Notes
- The TensorFlow implementation had the same bug (using `tf.size(tf.unique(data)[0])`)
- This is a common mistake in KDE implementations where the distinction between sample size and number of unique values is important
- The fix ensures that the bandwidth selection algorithm sees the true sample size, leading to appropriate smoothing 