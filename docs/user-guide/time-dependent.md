# Time-dependent modeling

`DVC_time` maps time to bandwidth parameters for each pair copula. The model learns b_ij(t) with small neural networks, allowing nonparametric copulas to adapt over time.

Workflow:
- Fit a base vine with DVC_pytorch.
- Initialize `TimeDependentVineCopula` with the fitted vine and the number of time steps.
- Optimize the negative log-likelihood over batches of (X_t, t).

Example:

```python
import torch
import dvc.backends.pytorch as DVC
from dvc.time import TimeDependentVineCopula

# Fit base vine (see fitting.md)
# vine = ...

model = TimeDependentVineCopula(vine, n_time_steps=T, hidden_dim=32)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = model.negative_log_likelihood(batch_x, batch_t)
loss.backward()
opt.step()
```
