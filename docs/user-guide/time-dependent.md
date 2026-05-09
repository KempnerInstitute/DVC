# Time-Dependent Modeling

The repository currently has two time-dependent layers:

- `dvc_package.time.flows`: standalone neural bandwidth-flow modules (`TimeBandwidthFlow`, `MLPEdgeFlow`)
- `dvc_package.time.models`: a higher-level wrapper (`TimeDependentVine`) for coupling a base vine with a time-conditioned bandwidth network
- `dvc_package.time.joint_dynamic_cvine` / `latent_state_dynamic_cvine`: jointly fitted parametric dynamic C-vines that fit temporal dynamics and vine dependence together over the full sequence
- `dvc_package.time.nonparametric_dynamic_cvine`: windowed and jointly fitted nonparametric dynamic vines with time-dependent bandwidth trajectories over fixed `C`/`D`/`R` structures

Related static nonparametric support exists in `dvc_package.core.nonparametric_vine`. The unbinned path fits kernel/local-likelihood pair-copulas over the full vine, supports `C-vine`, `D-vine`, and `R-vine`, and uses explicit conditional-CDF grids for higher-tree propagation and inverse-`h` sampling. That static implementation is the base for the dynamic nonparametric extensions below.

## Implemented Behavior (Current)

- `JointDynamicCVine.fit(...)` fits a fixed-order C-vine jointly across all time points by parameterizing each edge's dependence trajectory over time and selecting each edge family globally.
- `LatentStateDynamicCVine.fit(...)` adds a low-rank shared latent state so multiple edges can borrow temporal strength through a common dynamic trajectory.
- `TimeDependentVine.fit_bandwidth_flow(...)` trains a time-to-bandwidth network using held-out local-likelihood copula NLL.
- `TimeDependentVine.forward(x, t)` evaluates per-sample NLL using:
  - rank-normalized reference pairs per time slice,
  - level-0 vine edges,
  - learned time-conditioned bandwidths.
- The simulation benchmark suite uses this path as the `KDE-flow (time BW)` baseline.
- Dynamic Gaussian comparators are implemented in `dvc_package.baselines`:
  - `tvgl.py` (`tvgl_frobenius`)
  - `gaussian_state_space.py` (`gaussian_copula_state_space_nll_fit_eval`)
- The config-driven `TimeDependentExperiment` path uses the canonical
  `dvc_package.time.models.TimeDependentVine` API.

## Not Yet Fully Implemented

- Time-conditioned sampling from a full dynamic vine is not implemented yet. Sampling currently follows the fitted base vine, while time-conditioning is used in likelihood evaluation.
- Dynamic updates of higher-tree conditional copulas are not implemented in `time.models`; the flow path is level-0 focused.
- Dynamic nonparametric vines are implemented as research-stage models in
  `dvc_package.time.nonparametric_dynamic_cvine`, but they are less mature than
  the static unbinned nonparametric path and the parametric joint/latent models.
- Treat this module as a controlled benchmark component, not a fully general dynamic-vine inference engine.

## Flow Modules

```python
import torch
from dvc_package.time.flows import TimeBandwidthFlow

flow = TimeBandwidthFlow(hidden_dim=64, output_dim=2, n_layers=3)
t = torch.linspace(0, 1, 32)
bw = flow(t)  # shape: (32, 2), strictly positive
```

## Joint Dynamic C-Vines

```python
from dvc_package.time import JointDynamicCVine, LatentStateDynamicCVine

joint = JointDynamicCVine(
    families=["gaussian", "student", "independence"],
    n_basis=3,
    smoothness_penalty=0.5,
)
joint_result = joint.fit(train_windows)
joint_nll = joint.evaluate(test_windows)

latent = LatentStateDynamicCVine(
    families=["gaussian", "student", "independence"],
    order=joint_result.order,
    latent_dim=1,
    n_epochs=40,
)
latent_result = latent.fit(train_windows)
latent_nll = latent.evaluate(test_windows)
```

## Dynamic Nonparametric Vines

```python
from dvc_package.time.nonparametric_dynamic_cvine import (
    WindowedDynamicNonparametricVine,
    JointDynamicNonparametricVine,
)

windowed = WindowedDynamicNonparametricVine(
    vine_type="auto",           # "c-vine" | "d-vine" | "r-vine" | "auto"
    selection_criterion="aic",
    optimize_structure=False,
    knots=7,
)
windowed_result = windowed.fit(train_windows)

joint_np = JointDynamicNonparametricVine(
    vine_type=windowed_result.config["selected_vine_type"],
    knots=7,
    trajectory_type="basis",
    trajectory_kwargs={"n_basis": 2},
)
joint_np_result = joint_np.fit(train_windows)
joint_np_nll = joint_np.evaluate(test_windows)
```

Implementation note:
- `JointDynamicNonparametricVine` now reuses the same internal edge recursion as the validated static nonparametric fitter instead of maintaining a separate higher-tree bookkeeping path.
- Dynamic nonparametric models support fixed `C-vine`, `D-vine`, and `R-vine` structures, and they can select the best vine family on pooled data via `vine_type="auto"`.
- The joint optimizer warm-starts each edge trajectory from per-window static nonparametric bandwidth fits, then smooths those targets through a shared temporal trajectory model.
- If a particular edge trajectory becomes numerically unstable, that edge falls back to the finite per-window target bandwidths instead of emitting `NaN` bandwidths.

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

- The jointly fitted dynamic C-vine classes are the main path when you want to fit temporal dynamics and dependence simultaneously rather than refitting a separate copula per time window.
- The time-dependent bandwidth-flow components remain research-stage and are best used with controlled synthetic benchmarks first.
- Pin exact configs, seeds, and result manifests when reporting comparative numbers.
