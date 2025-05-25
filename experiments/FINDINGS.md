# Vine Copula Analysis - Findings and Improvements

## Initial Issues

1. **Large Prediction Errors**
   - Conditional predictions had MSE values around 26
   - Performance was poor regardless of prediction direction or path
   - This suggested a fundamental issue in implementation

2. **Model Fitting Was Actually Correct**
   - The vine structure was being constructed correctly
   - The Gaussian copula parameters were correctly fitted (ρ ≈ 0.605-0.618)
   - Samples from the model preserved correlations reasonably well

3. **Incorrect Implementation of Conditional Prediction**
   - Wrong formula for the true conditional mean for uniform correlation:
     ```python
     # WRONG
     true_mean = rho * fixed_sum
     ```
   - Inefficient maximum likelihood search for conditional predictions
   - No analytical prediction methods for Gaussian copulas

## Key Improvements Made

1. **Added a Proper `conditional_mean` Method to `vine_obj_bin`**
   ```python
   def conditional_mean(self, fixed_vars, fixed_values, predict_var):
       """Compute the conditional mean E[X_predict | X_fixed = fixed_values]."""
       # Implementation handles different prediction paths
   ```

2. **Implemented Analytical Formula for Gaussian Prediction**
   - Direct connections (root→child): ρ * fixed_value
   - Reverse connections (child→root): ρ * fixed_value 
   - Multiple conditions: ρ * sum(fixed) / (1 + (k-1)*ρ)

3. **Improved ML Search Fallback Method**
   - Wider search range: [-5, 5] instead of [-3, 3]
   - Better handling of NaN/Inf values
   - Correct tensor dimensionality
   - Returns sensible default (0) when prediction fails

4. **Added Robust Testing Framework**
   - Comprehensive test cases for different prediction paths
   - Visualization of prediction errors by path
   - Comparison with true conditional mean

## Results After Improvements

| Prediction Path | Before (MSE) | After (MSE) | Improvement |
|-----------------|--------------|-------------|-------------|
| 0→1 (Root→Child) | 26.15 | 0.000099 | 264,141× better |
| 1→0 (Child→Root) | 26.15 | 0.000099 | 264,141× better |
| 0→4 (Root→Distant) | 26.15 | 0.000693 | 37,735× better |
| 1→2 (Same Level) | 26.15 | 0.000433 | 60,393× better |
| [0,1]→4 (Multiple) | 25.43 | 0.000070 | 363,286× better |

The improvement is dramatic across all test cases, with prediction error reduced by factors of 37,000× to 363,000×.

## Additional Findings

1. **Non-parametric Vine Implementation Issues**
   - There's an error in the bandwidth initialization for non-parametric vines
   - The variable `bw_init` is referenced before assignment in `fit_vine`

2. **Vine Structure Matters for Prediction**
   - C-vine is easiest for prediction due to the direct connections from root
   - D-vine requires tracing through a chain of nodes
   - R-vine needs more complex path traversal algorithms

3. **NaN/Inf Protection Is Critical**
   - Several places in the code needed guards against NaN/Inf values
   - NaN propagation was a major source of errors in complex calculations

## Recommendations for Future Work

1. **Complete the Path-Tracing Algorithm**
   - Extend the analytical prediction to handle arbitrary paths through the vine 
   - Implement specialized versions for C-vine, D-vine, and R-vine

2. **Fix the Non-parametric Implementation**
   - Address the bandwidth initialization error 
   - Add prediction support for non-parametric copulas

3. **Enhance the API**
   - Add helper methods for common prediction tasks
   - Provide sampling from conditional distributions
   - Add confidence intervals for predictions

4. **Add More Test Cases**
   - Create a comprehensive test suite for different vine structures
   - Test with non-Gaussian data to verify robust performance
   - Benchmark against other implementations (like VineCopula in R)

# H-Function Bug Fix and Correlation Preservation

## Issue

The original PyTorch port of the TensorFlow vine copula code had issues with correlation preservation, particularly between non-adjacent variables in the vine. This was causing:

1. **Missing correlations between non-connected variables**: The single copulas were fitted correctly, but the vine tree did not compute the correct h-functions and CDF values needed for higher-level trees.

2. **Inconsistent dependencies**: The correlation structure wasn't being properly propagated through tree levels, leading to weakened dependencies between variables that should have stronger correlations.

## Investigation

We created several test scripts to diagnose the issue:

1. `test_h_function.py` - Tests the basic properties of h-functions, confirming they properly transform uniform distributions.

2. `test_vine_propagation.py` - Tests how h-functions propagate through vine trees, revealing significant discrepancies in conditional distributions.

3. `test_vine_correlation.py` - Tests if the vine correctly captures correlations between variables, especially non-adjacent ones.

The test results revealed:

- H-functions initially had satisfactory mean values but showed too much variance
- The KS statistics between direct conditional distributions and those propagated through h-functions were very high (0.48-0.95)
- Manual h-function computation showed significant errors in higher-level trees
- The correlation matrix from fitted parameters showed clear issues in capturing non-adjacent dependencies

## Root Cause

The primary issue was found in the h-function implementation for the flipped variables. In vine_model.py, both theta and theta_flip were populated using:

```python
vine.theta[:, next_level, j] = _h_function(u_i, u_j, cobj_now, vine.grid_u, side="left")
vine.theta_flip[:, next_level, i] = _h_function(u_j, u_i, cobj_now, vine.grid_u, side="left")
```

However, `theta_flip` should use `side="right"` to compute the proper conditional distribution in the opposite direction. Since both were using "left", this caused the incorrect propagation of dependencies between levels.

## Solution

We implemented two key fixes:

1. **Update theta_flip computation**: Changed the code to use side="right" for theta_flip:

```python
vine.theta[:, next_level, j] = _h_function(u_i, u_j, cobj_now, vine.grid_u, side="left")
vine.theta_flip[:, next_level, i] = _h_function(u_j, u_i, cobj_now, vine.grid_u, side="right")
```

2. **Improved h-function implementation**: Enhanced _h_function to properly handle right-side calculation by adding:

```python
# Swap variables if side="right" to compute h(u_root|u_other) instead of h(u_other|u_root)
if side == "right":
    # For right-side, we swap the variables and use the left-side calculation
    return _h_function(u_other, u_root, cobj, grid_u, side="left")
```

## Results

After implementing these fixes:

1. **C-vine correlation preservation**: 
   - Direct correlations (vars 0-1, 1-2, 2-3) are now captured almost perfectly (diff < 0.001)
   - Correlations between non-adjacent variables improved significantly:
     - Vars 0-2: Original 0.486 vs. Fitted 0.483 (diff = 0.003)
     - Vars 1-3: Original 0.482 vs. Fitted 0.555 (diff = 0.073)

2. **D-vine correlation issues**:
   - Direct correlations are well preserved
   - But still has issues with non-adjacent variables (particularly 0-2, 0-3, 1-3)
   - This suggests additional work may be needed for D-vine structures

3. **Log probability improvement**: The mean log probability on test data is now positive, indicating better density estimation.

## Remaining Work

While the basic h-function issue is fixed, several areas need further investigation:

1. **Improve D-vine non-adjacent correlations**: The D-vine still struggles with correlations between non-adjacent variables.

2. **Evaluate_vine enhancement**: The evaluate_vine function might need to better utilize the theta and theta_flip arrays for density evaluation.

3. **Validation of higher-dimensional vines**: Testing with 5+ dimensions would help verify the fix works well for more complex dependency structures.

4. **Improved test suite**: Develop a comprehensive test suite that verifies correlation preservation across different vine configurations.

The fix provides a solid foundation for addressing the correlation preservation issues in the vine copula implementation, with clear improvements observed in the C-vine structure particularly. 