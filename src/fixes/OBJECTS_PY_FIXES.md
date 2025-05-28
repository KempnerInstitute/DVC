# Fixes Applied to objects.py

This document outlines the critical fixes applied to the PyTorch implementation of `vine_obj_bin` in `objects.py` based on comparison with the TensorFlow version.

## 1. vine_depth Handling (CRITICAL FIX)

### Issue
The PyTorch code was overwriting `self.n_cop` with the data dimension `d`, losing the intended vine depth limitation.

### Fix
```python
# Before (incorrect):
self.n_cop = d  # This overwrites the vine_depth setting!

# After (correct):
self.vine_depth = gen_dict['vine_depth'] - 1  # Match TensorFlow
# Removed the line that sets self.n_cop = d
```

### Impact
- Now correctly respects the `vine_depth` parameter
- Allows fitting partial vines (e.g., only first 2 trees of a 5D vine)
- Trees beyond `vine_depth` are properly set to independence copulas

## 2. fitted Flag Handling (CRITICAL FIX)

### Issue
The PyTorch code was hardcoding `self.fitted = True`, ignoring the `gen_dict['fitted']` parameter.

### Fix
```python
# Before (incorrect):
self.fitted = True  # Always true!

# After (correct):
self.fitted = gen_dict['fitted']  # Respect the parameter
```

### Additional Logic
When `fitted == True`:
- The code switches to `vine_family = 'r-vine'` and `method = 'matrix'`
- Skips the actual copula fitting
- Only updates correlation tracking arrays

## 3. Margin.ker Handling

### Issue
The PyTorch code wasn't using `margin[i].ker` when available, unlike TensorFlow.

### Fix
```python
# Now checks for margin.ker before using data
if hasattr(self.margin[i], 'ker') and self.margin[i].ker is not None:
    ccc = torch.tensor(self.margin[i].ker, device=device, dtype=dtype)
    interp_cdf, mar_s1, mar_p1 = kernel_cdf(ccc, ccc, self.grid_u.ex)
else:
    interp_cdf, mar_s1, mar_p1 = kernel_cdf(q[:, i], q[:, i], self.grid_u.ex)
```

## 4. Binning Logic Structure

### Issue
The PyTorch code had incomplete binning logic compared to TensorFlow's extensive conditional blocks.

### Partial Fix
Added proper branching structure:
```python
if tr == 0 or self.binning == False:
    # No binning logic
else:  # self.binning == True and tr > 0
    # Binning logic for higher trees
    # TODO: Implement full binning logic
```

Note: Full binning implementation is marked as TODO and requires additional work.

## 5. Independence Copulas for Trees Beyond vine_depth

### Issue
The PyTorch code wasn't properly handling trees beyond the specified vine depth.

### Fix
Added explicit handling:
```python
if tr > self.vine_depth:
    # Use independence copulas
    if self.parallel:
        families = ["ind"]
        # Fit independence copulas
    else:
        # Non-parallel independence copulas
```

## 6. Proper Initialization Order

### Issue
Data transformation and theta initialization were happening in the wrong order.

### Fix
- Initialize theta arrays early
- Only transform data when needed for specific tree fitting
- Separate initialization from fitting logic based on `fitted` flag

## Remaining Issues (TODO)

1. **Full Binning Implementation**: The complex per-bin fitting logic from TensorFlow needs to be fully ported
2. **sort_n Parameter**: Currently forced to 'no_sort', should be configurable
3. **Parallel Non-parametric Fitting**: Not yet implemented
4. **Non-parametric H-functions**: Computation needs to be completed

## Testing

A test script `test_object_fixes.py` verifies:
- vine_depth is correctly used (not overwritten)
- fitted flag is respected
- margin.ker is used when available
- Basic binning flags are set correctly

## Summary

These fixes bring the PyTorch `objects.py` much closer to the TensorFlow behavior, particularly for the critical issues of vine depth handling and the fitted flag. The implementation now correctly:

1. Respects the intended vine depth instead of always using full dimension
2. Allows skipping fitting when `fitted=True`
3. Uses margin kernel data when provided
4. Has the structure for binning (though not fully implemented)
5. Properly handles independence copulas for trees beyond vine_depth 