# Time-Dependent Modeling

The repository currently has two time-dependent layers:

- `dvc_package.time.flows`: standalone neural bandwidth-flow modules (`TimeBandwidthFlow`, `MLPEdgeFlow`)
- `dvc_package.time.models`: a higher-level wrapper (`TimeDependentVine`) for coupling a base vine with a time-conditioned bandwidth network

## Flow Modules

```python
import torch
from dvc_package.time.flows import TimeBandwidthFlow

flow = TimeBandwidthFlow(hidden_dim=64, output_dim=2, n_layers=3)
t = torch.linspace(0, 1, 32)
bw = flow(t)  # shape: (32, 2), strictly positive
```

## Wrapping a Base Vine

```python
import torch
from dvc_package.core.vine_factory import create_vine
from dvc_package.time.models import create_time_dependent_vine

base_vine = create_vine("c-vine", vine_depth=3)
time_vine = create_time_dependent_vine(base_vine, hidden_dims=[64, 32], device="cpu")

x = torch.randn(16, 3)
t = torch.linspace(0, 1, 16)
nll = time_vine(x, t)  # per-sample negative log-likelihood proxy
```

## Data Utilities for Temporal Experiments

```python
from dvc_package.time.data import (
    generate_synthetic_time_series,
    compute_time_varying_correlations,
)

data, time_idx = generate_synthetic_time_series(
    n_time_steps=50,
    n_samples=100,
    n_variables=3,
    correlation_evolution="sinusoidal",
    seed=42,
)
corrs = compute_time_varying_correlations(data)
```

## Notes

- The time-dependent components are research-stage and best used with controlled synthetic benchmarks first.
- For paper-grade claims, prefer reporting metrics from reproducible scripts/configs in `configs/` and `scripts/run_experiment.py`.

