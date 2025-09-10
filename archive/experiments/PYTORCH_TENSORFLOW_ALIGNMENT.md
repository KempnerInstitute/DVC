# PyTorch vs TensorFlow DVC Implementation Alignment

## Key Differences Identified

### 1. **Parametric Copula Fitting Strategy**

#### TensorFlow Implementation:
- Uses gradient-based optimization (Nadam) with careful initialization
- Optimizes the negative log-likelihood directly with proper bounds checking
- For Gaussian copula: starts with `pos_trace = 0.5`, learning rate `lr = 0.005`
- Convergence tolerance: `1e-3`, max iterations: 100-200
- **Key**: The cost function computes the actual copula PDF, not just correlation

#### PyTorch Implementation Issues:
- Uses simple correlation estimation without proper MLE
- No gradient-based optimization for parameter estimation
- Relies too heavily on Kendall's tau conversion

### 2. **Non-Parametric Optimization (MISE)**

#### TensorFlow Implementation:
- Two-phase optimization: first without normalization, then with normalization
- Uses cross-validation with 5-fold split
- Bandwidth optimization via Nadam with specific learning rates
- Phase 1: `max_iter=70, lr=0.1, conv_tol=1e-5`
- Phase 2: `max_iter=100, lr=0.03, conv_tol=5e-5`
- Uses `eval_rs_cop` for proper copula normalization with 500 iterations

#### PyTorch Implementation Issues:
- Simplified optimization without cross-validation
- Different normalization approach
- Missing the sophisticated two-phase strategy

### 3. **Theta Propagation (Critical for Correlation Preservation)**

#### TensorFlow Implementation:
```python
# After fitting copula at level tr, update theta for next level
ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s[:,:,i], grid_s.min, grid_s.max, cdf1[:,:,i], axis=-2)
interp_cdf, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_data, grid_u.ex)

if flip_flag[i] == False:
    theta[:,tr+1,ind_edge_rel[i]] = interp_cdf
else:
    theta_flip[:,tr+1,ind_edge_rel[i]] = interp_cdf
```

#### PyTorch Implementation Issues:
- Different interpolation approach
- Missing the kernel CDF step for ensuring uniform margins
- Potentially incorrect h-function implementation

### 4. **Copula Normalization**

#### TensorFlow: 
- Uses iterative row-column normalization (50-500 iterations)
- Projects to U-V space, normalizes, then projects back
- Ensures proper copula properties

#### PyTorch:
- Simpler normalization approach
- May not preserve copula properties as well

### 5. **Grid Operations and Transformations**

#### TensorFlow:
- Consistent use of `tfp.math.batch_interp_regular_nd_grid`
- Proper handling of grid boundaries
- Careful dtype management

#### PyTorch:
- Different interpolation methods
- Potential numerical instability at boundaries

## Implementation Fixes Needed

1. **Reimplement Parametric Fitting**
   - Add gradient-based optimization for all copula families
   - Use proper MLE with copula PDFs
   - Match TensorFlow's initialization and convergence criteria

2. **Fix Non-Parametric Optimization**
   - Implement two-phase MISE optimization
   - Add cross-validation split
   - Match the normalization iterations

3. **Correct Theta Propagation**
   - Ensure proper kernel CDF transformation
   - Fix h-function to match TensorFlow's approach
   - Verify flip_flag handling

4. **Improve Numerical Stability**
   - Add boundary checks matching TensorFlow
   - Use consistent epsilon values (1e-15 for PDF floor)
   - Implement proper NaN/Inf handling

5. **Performance Optimizations**
   - Consider TensorFlow's batching strategies
   - Implement parallel edge fitting where applicable 