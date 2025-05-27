# Random R-Matrix Generation Fix

This document describes the fix applied to align the `random_r_matrix_gen` function return signature between PyTorch and TensorFlow versions.

## Issue

The PyTorch and TensorFlow versions of `random_r_matrix_gen` had different return signatures:

**PyTorch (before fix)**:
```python
def random_r_matrix_gen(dim):
    # ...
    return rr, E, nodes, matrix_edges
```

**TensorFlow**:
```python
def random_r_matrix_gen(dim):
    # ...
    return r_matrix, ind_vine, nodes, E
```

## Solution

Updated the PyTorch version to match TensorFlow's return signature:

```python
def random_r_matrix_gen(dim):
    """Generate random R-matrix"""
    ind_vine = []
    for tr in range(0, dim-1, 1):
        if tr == 0:
            edges, weights = random_tree(dim, ind_vine, tr)
        else:
            edges, weights = random_tree(dim, ind_vine, tr)
        ind_vine.append(edges)
    
    # Get r_matrix from prepare_optimal
    r_matrix, E, nodes = prepare_optimal(dim, ind_vine)
    # Update with prepare_regular to get final ind_vine
    E, updated_ind_vine, nodes, matrix_edges = prepare_regular(r_matrix)
    
    # Return in TensorFlow order: (r_matrix, ind_vine, nodes, E)
    return r_matrix, updated_ind_vine, nodes, E
```

## Changes Made

1. **Variable naming**: Renamed `rr` to `r_matrix` for consistency
2. **Return order**: Changed from `(rr, E, nodes, matrix_edges)` to `(r_matrix, updated_ind_vine, nodes, E)`
3. **Return content**: Now returns `updated_ind_vine` instead of the initial `ind_vine`, matching TensorFlow behavior

## Impact

- **No breaking changes**: All existing callers use `_` placeholders for unused return values
- **Improved consistency**: PyTorch and TensorFlow implementations now have identical signatures
- **Better maintainability**: Reduces confusion when switching between implementations

## Verification

Checked all usages of `random_r_matrix_gen`:
- `classes/objects.py`: Uses only first return value (r_matrix)
- `experiments/test_comprehensive_vines.py`: Uses only first return value

All callers are already compatible with the new return signature. 