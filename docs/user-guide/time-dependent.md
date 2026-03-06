# Time-Dependent Modeling

The repository currently has two time-dependent layers:

- `dvc_package.time.flows`: standalone neural bandwidth-flow modules (`TimeBandwidthFlow`, `MLPEdgeFlow`)
- `dvc_package.time.models`: a higher-level wrapper (`TimeDependentVine`) for coupling a base vine with a time-conditioned bandwidth network

## Implemented Behavior (Current)

- `TimeDependentVine.fit_bandwidth_flow(...)` trains a time-to-bandwidth network using held-out local-likelihood copula NLL.
- `TimeDependentVine.forward(x, t)` evaluates per-sample NLL using:
  - rank-normalized reference pairs per time slice,
  - level-0 vine edges,
  - learned time-conditioned bandwidths.
- The NeurIPS simulation runner uses this path as the `KDE-flow (time BW)` baseline.
- Dynamic Gaussian comparators are implemented in `dvc_package.baselines`:
  - `tvgl.py` (`tvgl_frobenius`)
  - `gaussian_state_space.py` (`gaussian_copula_state_space_nll_fit_eval`)
- The config-driven `TimeDependentExperiment` path uses the canonical
  `dvc_package.time.models.TimeDependentVine` API.

## Not Yet Fully Implemented

- Time-conditioned sampling from a full dynamic vine is not implemented yet. Sampling currently follows the fitted base vine, while time-conditioning is used in likelihood evaluation.
- Dynamic updates of higher-tree conditional copulas are not implemented in `time.models`; the flow path is level-0 focused.
- For paper claims, treat this module as a controlled benchmark component, not a fully general dynamic-vine inference engine.

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

# Set per-time reference data once.
time_vine.set_reference_data(data_by_time, time_points=t_points)

x = torch.randn(16, 3)
t = torch.linspace(0, 1, 16)
nll = time_vine(x, t)  # per-sample negative log-likelihood
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
- For paper-grade claims, prefer reporting metrics from reproducible paper assets in `drafts/configs/`, `drafts/scripts/`, and `scripts/run_experiment.py`.
