# Index Remapping Fix for DVC PyTorch Sampling

## Problem Description

The PyTorch implementation of DVC vine sampling was producing nearly independent samples despite fitting the copulas correctly. The correlation preservation was severely compromised, with correlation errors as high as 0.67 (essentially random).

## Root Cause

The issue was a missing index remapping step that exists in the TensorFlow implementation. When accessing copulas during sampling, the code needs to map the "raw" column index to the actual position where that edge's copula is stored.

### TensorFlow Code (Correct)
```python
# Calculate raw column index
col = i - k - 1

# Remap to actual copula position
ind_array = np.array(vine.ind_edge_rel[tr])
ind_col = np.where(ind_array == col)
col = ind_col[0][0]

# Now use remapped col to access copula
cdf_grid = vine.copulas[tr].cdf[:, :, col]
```

### PyTorch Code (Before Fix - Incorrect)
```python
# Calculate raw column index
col = i - k - 1

# Directly use col without remapping - WRONG!
cdf_grid = self.vine.copulas[tr].cdf[:, :, col]
```

## The Fix

Added a `_remap_col_index()` method to the `VineSampler` class:

```python
def _remap_col_index(self, tr: int, col: int) -> int:
    """
    Remap column index to match the actual edge ordering in copulas array.
    This is the critical missing step from TensorFlow implementation.
    """
    if hasattr(self.vine, 'ind_edge_rel') and tr < len(self.vine.ind_edge_rel):
        ind_array = self.vine.ind_edge_rel[tr]
        # Find where col matches in the array
        if isinstance(ind_array, np.ndarray):
            idx_matches = np.where(ind_array == col)[0]
            if len(idx_matches) > 0:
                return idx_matches[0]
        else:
            # If it's a list or other iterable
            idx_matches = [idx for idx, val in enumerate(ind_array) if val == col]
            if len(idx_matches) > 0:
                return idx_matches[0]
    
    # If no remapping found, return original col
    return col
```

Then updated all places where copulas are accessed:

```python
# In sampling loops
col_remapped = self._remap_col_index(tr, col)
v[:, k, i] = copulainvccdf_torch(self.vine.copulas[tr][col_remapped], vv)

# In h-function computation
col_remapped = self._remap_col_index(tr, col)
cop_obj = self.vine.copulas[tr][col_remapped]
```

## Impact

The fix dramatically improved sampling accuracy:

### Before Fix
- Max correlation error: 0.6667
- Mean correlation error: 0.3883
- Samples were nearly independent despite fitted copulas

### After Fix
- Max correlation error: 0.0676 (90% reduction!)
- Mean correlation error: 0.0343 (91% reduction!)
- Proper dependency structure preservation
- 100% marginal distribution test pass rate

## Why This Matters

The `ind_edge_rel` array stores the actual ordering of edges in each tree level, which may differ from the natural ordering implied by `col = i - k - 1`. Without this remapping:

1. The wrong copula is used for each edge
2. Dependencies between variables are scrambled
3. The resulting samples don't reflect the fitted model

## Lessons Learned

1. **Index Mapping is Critical**: When porting algorithms, pay special attention to any index transformations or mappings
2. **Test Correlation Preservation**: Always verify that sampling preserves the expected correlation structure
3. **Compare Implementations Carefully**: Small differences in indexing can have dramatic effects on results

## Related Files

- `sampling/vine_sampler.py`: Contains the fix
- `test_sampling_accuracy.py`: Comprehensive test showing improvements
- `test_accuracy_focused.py`: Focused test with key configurations
- `IMPLEMENTATION_STATUS.md`: Overall status documentation 