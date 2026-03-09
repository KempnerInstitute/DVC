# Core API Reference

## Data Conventions

- `X`: NumPy array with shape `(n_samples, n_variables)`
- `points`: tensor or array-like with shape `(n_points, n_variables)`
- `uv`: tensor with shape `(n_points, 2)` for pair-copula utilities

## Primary Objects

### `vine_obj_bin`

Container returned by `create_vine(...)` and used by the fitting, evaluation, and sampling functions.

Common fields used by the public API:
- `vine_family`
- `n_cop`
- `ind_vine`
- `copulas`
- `param`
- `fitted`

### `cop_par_obj`

Parametric pair-copula container with:
- `family`
- `theta`

## Structure Construction

### `create_vine`

Import:

```python
from dvc_package.core.vine_factory import create_vine
```

Signature:

```python
create_vine(vine_type, vine_depth, families=None, margin=None, **kwargs)
```

Use this to allocate a `C-vine`, `D-vine`, or `R-vine` object before fitting.

Key arguments:
- `vine_type`: `"c-vine"`, `"d-vine"`, `"r-vine"`, or `VineType`
- `vine_depth`: number of variables
- `families`: candidate pair-copula families
- `margin`: optional marginal objects

Example:

```python
vine = create_vine(
    "c-vine",
    vine_depth=4,
    families=["ind", "gaussian", "clayton", "frank"],
)
```

### `optimize_vine_type`

Import:

```python
from dvc_package.core.vine_factory import optimize_vine_type
```

Signature:

```python
optimize_vine_type(
    data,
    selection_criterion="aic",
    optimize_structure=False,
    optimization_method="sequential",
    optimization_criterion="kendall_tau",
    families=None,
    par_dict=None,
    vine_kwargs=None,
)
```

Fits candidate `C-vine`, `D-vine`, and `R-vine` models and returns the best fitted vine.

Example:

```python
best_vine = optimize_vine_type(
    X,
    selection_criterion="bic",
    optimize_structure=True,
    par_dict={"param_families": ["ind", "gaussian", "student", "clayton"]},
)
```

### `optimize_vine_structure`

Import:

```python
from dvc_package.optimization.structure import optimize_vine_structure
```

Signature:

```python
optimize_vine_structure(
    data,
    vine_type="c-vine",
    method="sequential",
    criterion="kendall_tau",
    families=None,
    max_iterations=100,
    random_seed=42,
    verbose=True,
)
```

Returns an optimization result object with `best_vine`, `best_score`, and method metadata.

## Static Vine Workflow

### `fit_vine`

Import:

```python
from dvc_package.core.vine_model import fit_vine
```

Signature:

```python
fit_vine(vine, x, gen_dict, npc_dict, par_dict, bin_dict, cfg=None)
```

Behavior:
- Uses the parametric path when `gen_dict["param"]` is true
- Dispatches to the nonparametric path when `gen_dict["param"]` is false

Common parameter dictionaries:

```python
gen_dict = {"param": True, "binning": False, "fitted": False}
npc_dict = {}
par_dict = {"param_families": ["ind", "gaussian", "clayton", "frank"]}
bin_dict = {}
```

Example:

```python
fit_vine(vine, X, gen_dict, npc_dict, par_dict, bin_dict)
```

### `evaluate_vine`

Import:

```python
from dvc_package.core.vine_model import evaluate_vine
```

Signature:

```python
evaluate_vine(vine, points)
```

Returns the fitted joint density evaluated at `points`. The function accepts tensors or array-like input.

Example:

```python
points = torch.tensor(X[:32], dtype=torch.float32)
joint_pdf, copula_pdf, log_marginals = evaluate_vine(vine, points)
```

### `sample_vine`

Import:

```python
from dvc_package.core.vine_model import sample_vine
```

Signature:

```python
sample_vine(vine, nsamples, cfg=None)
```

Returns samples from a fitted vine as a tensor or array consistent with the fitted path.

Example:

```python
samples = sample_vine(vine, 1000)
```

## Static Nonparametric Workflow

### `fit_nonparametric_vine`

Import:

```python
from dvc_package.core.nonparametric_vine import fit_nonparametric_vine
```

Signature:

```python
fit_nonparametric_vine(vine, x, gen_dict, npc_dict, par_dict, bin_dict, cfg=None)
```

Use this directly when you want explicit access to the nonparametric fitter. The unbinned path supports `C-vine`, `D-vine`, and `R-vine`.

Typical nonparametric options:

```python
npc_dict = {
    "opt_method": "LL1",
    "batch_size": 256,
    "validation_fraction": 0.15,
    "final_normalization_iters": 25,
}
```

### `evaluate_nonparametric_vine`

Import:

```python
from dvc_package.core.nonparametric_vine import evaluate_nonparametric_vine
```

Signature:

```python
evaluate_nonparametric_vine(vine, points)
```

Returns the nonparametric joint density evaluated at `points`.

## Pair-Copula Utilities

### `parametric_fit`

Import:

```python
from dvc_package.core.param_copula import parametric_fit
```

Signature:

```python
parametric_fit(u, families, n_cop)
```

Inputs:
- `u`: NumPy array with shape `(n_samples, 2, n_cop)`
- `families`: list of candidate families
- `n_cop`: number of edges in the batch

Returns:
- `aic2`
- `thetas_list`
- `logp_list`

### `copulapdf`

Import:

```python
from dvc_package.core.param_copula import copulapdf
```

Signature:

```python
copulapdf(cop_p, uv)
```

Evaluates a pair-copula density on `uv`.

### `copulaccdf`

Import:

```python
from dvc_package.core.param_copula import copulaccdf
```

Signature:

```python
copulaccdf(cop_p, uv)
```

Evaluates the pair-copula conditional CDF used for higher-tree propagation.

### `copulainvccdf`

Import:

```python
from dvc_package.core.param_copula import copulainvccdf
```

Signature:

```python
copulainvccdf(cop_p, uv)
```

Evaluates the inverse conditional CDF used by the sampling path.

## Information Estimation

### `vine_entropy`

Import:

```python
from dvc_package.core.info_estimation import vine_entropy
```

Signature:

```python
vine_entropy(vine, info_dict)
```

Required `info_dict` keys:
- `alpha`
- `cases`
- `iterations`

Optional keys:
- `units`: `"bits"` or `"nats"`
- `seed`

### `cond_vine_entropy`

Import:

```python
from dvc_package.core.info_estimation import cond_vine_entropy
```

Signature:

```python
cond_vine_entropy(vine, vine_f2, info_dict)
```

Returns:
- entropy of the conditioning vine
- conditional entropy
- iteration-level information values

### `mutual_information`

Import:

```python
from dvc_package.core.info_estimation import mutual_information
```

Signature:

```python
mutual_information(vine, X_indices, Y_indices, info_dict)
```

Use integer index lists for `X_indices` and `Y_indices`.

## Minimal End-to-End Example

```python
import numpy as np
import torch

from dvc_package.core.vine_factory import create_vine
from dvc_package.core.vine_model import fit_vine, evaluate_vine, sample_vine

X = np.random.randn(500, 4).astype(np.float32)

vine = create_vine("c-vine", vine_depth=4, families=["ind", "gaussian", "frank"])
fit_vine(
    vine,
    X,
    {"param": True, "binning": False, "fitted": False},
    {},
    {"param_families": ["ind", "gaussian", "frank"]},
    {},
)

pdf, copula_pdf, log_marg = evaluate_vine(vine, torch.tensor(X[:16]))
samples = sample_vine(vine, 256)
```
