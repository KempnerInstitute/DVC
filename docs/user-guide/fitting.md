# Fitting Vine Copulas

This page shows the supported in-code workflow for fitting a vine model with parametric pair copulas.

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

## Notes

- Family aliases `ind` and `independence` are both accepted.
- `criterion="aic"`/`"bic"` is treated as a minimization objective in optimization.
- For reproducibility in experiments, set NumPy and PyTorch random seeds before fitting.

