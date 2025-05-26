# Final KDE and Marginal PDF Fix Summary

## Issues Identified and Fixed

### 1. Primary Issue: Incorrect Sample Size in KDE
**Problem**: The kernel density estimation was using the number of unique values instead of the actual sample size for bandwidth selection.

**Root Cause**:
```python
# WRONG: Only counts distinct values
N = len(torch.unique(data))  

# CORRECT: Uses actual sample size
N = data.shape[0]
```

**Impact**: When data had many repeated values (e.g., integer data), N would be very small, causing the bandwidth selection algorithm to oversmooth, resulting in flat densities.

### 2. DCT Implementation Issue
**Problem**: The discrete cosine transform implementation didn't match TensorFlow's FFT-based approach.

**Fix**: Updated to use FFT-based DCT:
```python
def dct1d(data):
    """Discrete cosine transform using FFT"""
    n = data.shape[0]
    if n == 1:
        return data
    
    # Extend by mirroring: [data, reverse(data[1:n-1])]
    extended = torch.cat([data, torch.flip(data[1:n-1], dims=[0])], dim=0)
    
    # Perform FFT
    result = torch.fft.fft(extended.to(complex_dtype))
    result = torch.real(result)
    
    return result[:n]
```

### 3. Numerical Stability in Root Finding
**Problem**: The bisection method for finding optimal bandwidth was not handling edge cases well.

**Fixes Applied**:
- Added checks for function evaluation failures at bounds
- Implemented adaptive interval selection when no sign change is found
- Added handling for inf/nan values during iteration
- Improved initial interval selection with multiple test points

### 4. Division by Zero in kernel_pdf2
**Problem**: For integer data, the bimodal distribution handling code would divide by zero when all values in one group were identical.

**Fix**: Added check for identical values:
```python
if torch.abs(max_pow1 - min_pow1) < 1e-16:
    # Use a small range around the value
    center = (max_pow1 + min_pow1) / 2
    R = 1e-6
    min_pow1 = center - R/2
    max_pow1 = center + R/2
else:
    R = max_pow1 - min_pow1
```

## Test Results

### Before Fixes:
- Density variance: ~1e-14 to 1e-17 (essentially flat)
- Bandwidth selection: Often hit upper bound (1.0)
- Integer data: Produced NaN values
- Fixed point function: Returned `-inf` for many values

### After Fixes:
- Normal distribution: variance ~2e-2, proper bell curve shape
- Uniform distribution: variance ~7e-3, appropriate flat shape
- Integer data: variance properly computed (no NaN)
- Bimodal distribution: variance ~4e-3, shows two peaks
- Bandwidth selection: Reasonable values (e.g., 0.000036)
- Fixed point function: Well-behaved across entire range

## Files Modified

1. **utils/prob_op.py**:
   - Fixed N calculation in `kde()` function (line 293)
   - Updated `dct1d()` and `idct1d()` to use FFT-based approach
   - Improved `find_root_bisection()` with better numerical stability
   - Fixed division by zero in `kernel_pdf2()`

2. **Test files created**:
   - `test_kde_fix.py`: Tests KDE with various data types
   - `debug_kde_detailed.py`: Detailed debugging of bandwidth selection
   - `test_marginal_fix.py`: Tests marginal PDF estimation
   - `debug_integer_kde.py`: Specific test for integer data issues

## Verification
All marginal PDF estimations now produce proper variance values and correct density shapes for:
- Normal distributions
- Uniform distributions  
- Integer/discrete data
- Bimodal distributions
- Data with many repeated values

The PyTorch implementation now correctly matches the expected behavior for kernel density estimation and marginal PDF calculation. 