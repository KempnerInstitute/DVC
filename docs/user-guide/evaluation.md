# Evaluation

- Density: use `vine.evaluation(X)` to obtain densities and log-densities.
- Entropy and MI: use `src/DVC/info_estimation.py` APIs.

Example (entropy):

```python
from dvc.core.info_estimation import vine_entropy
H_bits = vine_entropy(vine, {'alpha': 0.05, 'cases': 1000, 'iterations': 10})
```

Example (mutual information):

```python
from dvc.core.info_estimation import mutual_information
mi_bits = mutual_information(vine, X_indices=[0,1], Y_indices=[2], info={'alpha': 0.05, 'cases': 1000, 'iterations': 10})
```
