# Evaluation and Information Measures

## Density Evaluation

```python
import torch
from dvc_package.core.vine_model import evaluate_vine

points = torch.tensor(X_eval, dtype=torch.float32)
joint_pdf, copula_pdf, log_marginals = evaluate_vine(vine, points)
```

You can also call:

```python
joint_pdf, copula_pdf, log_marginals = vine.evaluation(X_eval)
```

## Sampling

```python
samples = vine.sample(nsamples=1000)
```

## Entropy

```python
from dvc_package.core.info_estimation import vine_entropy

info = {"alpha": 0.05, "cases": 1000, "iterations": 10}
H_bits = vine_entropy(vine, info)
```

## Mutual Information

```python
from dvc_package.core.info_estimation import mutual_information

I_xy = mutual_information(
    vine,
    X_indices=[0, 1],   # variable indices for first set
    Y_indices=[2],      # variable indices for second set
    info_dict={"alpha": 0.05, "cases": 1000, "iterations": 10},
)
```

## Practical Notes

- Use larger `cases` for lower Monte Carlo variance in entropy/MI estimates.
- Some sampling tests remain marked as known issues in `tests/test_vine_pipeline.py`.
- For fair method comparisons, keep `info` settings fixed across models.
