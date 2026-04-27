# Nonparametric h-function diagnosis: residual multivariate bias

## TL;DR

Before the h-axis fix, a 4D AR(1) Gaussian D-vine fit with the NP path gave
held-out TC ≈ −1.6 nats (truth +0.43). Per-edge mean log-likelihoods at
level 1 were +0.44 and +0.53 even though the truth at every higher AR(1)
tree level is exact conditional independence (truth: +0.0).

Empirically the marginal uniformity of propagated h-values is fine
(KS ≈ 0.001), but the **joint** correlation between propagated h-values at
level 1 is +0.77 vs truth 0. So the h-function is leaving the conditioning
in.

```
Level 0  edge0=(0,1):  pearson(u_in1, u_in2) = +0.49   (truth ~ +0.50)  ✓
Level 0  edge1=(1,2):  pearson(u_in1, u_in2) = +0.47   (truth ~ +0.50)  ✓
Level 0  edge2=(2,3):  pearson(u_in1, u_in2) = +0.47   (truth ~ +0.50)  ✓
Level 1  edge0:        pearson(u_in1, u_in2) = +0.77   (truth = 0)      ✗
Level 1  edge1:        pearson(u_in1, u_in2) = +0.77   (truth = 0)      ✗
Level 2  edge0:        pearson(u_in1, u_in2) = -0.63   (truth = 0)      ✗
```

Reproducer: [scripts/diagnose_np.py](../scripts/diagnose_np.py) on a 4D
Gaussian AR(1) with rho=0.5 and 5,000 samples.

After the fix, the same reproducer gives level-1 propagated-input
correlations of approximately +0.015 and +0.007, and a held-out 4D AR(1)
TC check gives NP D-vine TC ≈ +0.453 versus truth +0.432.

## Updated root cause: h-grid and interpolation axis conventions

The original symptom was correct, but the culprit was one level lower than
the oriented-copula wrapper.  `cdf_grid_fun` was documented and consumed like
the parametric `copulaccdf`, i.e. as

```python
h(v | u) = d C(u, v) / d u,
```

but the implementation was integrating the first grid axis, which gives
`h(u | v)`. In addition, the regular-grid interpolation path passed `(u, v)`
directly to `torch.grid_sample`; PyTorch expects `(width, height)`, so this
read h-grids with the two axes swapped.

The transpose+swap oriented-copula path is not itself mathematically wrong:
if the base grid stores `h(v | u)`, then transposing the fitted pair density
and evaluating on swapped inputs is a valid way to produce `h(u | v)`.
The problem was that the base grid did not actually store `h(v | u)`.

## Why the failure mainly appears in higher trees

At level 0 the fitted density itself is evaluated on the ranked margin
pseudo-observations, so pairwise log-density can look reasonable. The problem
appears when those pairwise fits are converted into h-values for the next tree:
the wrong h-axis leaves conditioning information in the propagated variables.

C-vines with a convenient hub can look better because their propagation uses a
more regular orientation pattern, but this was not a principled C-vine/D-vine
difference. The underlying h-grid convention needed to be fixed for all vine
families.

## Fixes considered

### (A) Force exchangeable KDE for the level-0 pair densities.

Symmetrize the fitted pdf:
`pdf_sym = 0.5 * (pdf_base + pdf_base.T)` and use `pdf_sym` for both
directions. Costs a small fit-quality hit on inherently asymmetric copulas
(Clayton/Gumbel/Joe), but the level-0 ones in our scenarios are largely
elliptical, so this should recover most of the bias in AR(1)-style cases
without rearchitecting the h-function.

This is no longer recommended as a scientific fix. It can hide the Gaussian
diagnostic failure, but it is a model restriction and would damage the
nonparametric path exactly where asymmetric tail copulas matter.

### (B) Correct the h-function axis convention.

The implemented fix is:

- `cdf_grid_fun(..., axis=1)` now returns `h(v | u)` by default.
- `cdf_grid_fun(..., axis=0)` is available for direct `h(u | v)` grids.
- `interp_regular_nd_grid` preserves natural point order by swapping
  coordinates only at the `torch.grid_sample` boundary.
- The existing transpose+swap oriented path can be kept, because it is valid
  once the base h-grid convention is correct.

An equivalent future cleanup would be to build reversed h-copulas directly
with `axis=0` and remove read-side swaps for flipped edges. That would be a
refactor, not a different scientific model.

### (C) [bandwidth] Larger bandwidths at higher tree levels.

Independent of the orientation issue, the kernel bandwidth chosen at level
0 from raw rank-data may be too narrow once we propagate into level 1, where
the inputs are already quite uniform-looking. A small upscaling (e.g.,
multiply bandwidth by 1.2 at level ≥ 1) helps mask the bias. This is a
cosmetic fix, not a correctness fix.

## Recommended next steps

1. Keep the axis-convention fix and do not symmetrize by default.
2. Add/keep regression tests for h-grid direction and regular-grid
   interpolation axis order.
3. Re-run the showcase NP benchmark and compare to the
   parametric DVC numbers. Even with correct higher-tree h-functions, the
   NP path will still pay a kernel-approximation cost vs the parametric
   path, so it should be presented as a supplementary direction rather than
   a main-result method (matching your prior assessment).
