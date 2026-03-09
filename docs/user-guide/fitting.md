# Fitting Vine Copulas

This page shows the supported in-code workflow for fitting a vine model with parametric pair copulas.
The parametric fitter supports `C-vine`, `D-vine`, and `R-vine`. The static
nonparametric fitter now also supports `C-vine`, `D-vine`, and `R-vine` for
fit/evaluation/sampling through the legacy edge-index bookkeeping.

## 1. Create a Vine Structure

```python
from dvc_package.core.vine_factory import create_vine

vine = create_vine(
    vine_type="c-vine",   # "c-vine" | "d-vine" | "r-vine"
    vine_depth=4,
    families=["ind", "gaussian", "clayton", "frank", "gumbel"],
)
```

## 2. Fit Pair-Copulas

```python
from dvc_package.core.vine_model import fit_vine

gen_dict = {"param": True, "binning": False, "fitted": False}
npc_dict = {}  # nonparametric options (unused in this parametric example)
par_dict = {"param_families": ["ind", "gaussian", "clayton", "frank", "gumbel"]}
bin_dict = {}

fit_vine(vine, X, gen_dict, npc_dict, par_dict, bin_dict)
```

`X` must be a NumPy array of shape `(n_samples, n_variables)`.

## 3. Optional Structure Optimization

```python
from dvc_package.optimization.structure import optimize_vine_structure

result = optimize_vine_structure(
    data=X,
    vine_type="c-vine",
    method="sequential",   # sequential | genetic | entropy | hybrid
    criterion="kendall_tau",
    verbose=False,
)
vine = result.best_vine
```

Then call `fit_vine(...)` on `vine`.

## 4. Optional Vine-Type Selection

If you want to compare `C-vine`, `D-vine`, and `R-vine` and keep the best one
under `AIC`, `BIC`, or raw log-likelihood:

```python
from dvc_package.core.vine_factory import optimize_vine_type

best_vine = optimize_vine_type(
    data=X,
    selection_criterion="aic",   # "aic" | "bic" | "loglik"
    optimize_structure=True,
    optimization_method="sequential",
    optimization_criterion="kendall_tau",
    par_dict={"param_families": ["ind", "gaussian", "clayton", "frank"]},
)
```

## Notes

- Family aliases `ind` and `independence` are both accepted.
- `criterion="aic"`/`"bic"` is treated as a minimization objective in optimization.
- For reproducibility in experiments, set NumPy and PyTorch random seeds before fitting.
- Static nonparametric fitting via `gen_dict={"param": False}` supports
  `C-vine`, `D-vine`, and `R-vine` for fit/evaluation/sampling in the
  unbinned path.
- The fitted nonparametric edge objects store explicit conditional-CDF
  (`ccdf_grid`) metadata used consistently for higher-tree propagation and
  inverse-`h` sampling.
- Binning remains unimplemented in the current PyTorch nonparametric path.
- For `R-vine`, prefer an explicit `r_matrix` or `optimize_vine_structure(...)`
  over the default random initialization when running scientific experiments.
