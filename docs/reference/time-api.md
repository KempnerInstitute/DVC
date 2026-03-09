# Time API Reference

## Input Conventions

Dynamic models accept either:
- a sequence of arrays, one array per time step, each with shape `(n_samples_t, n_variables)`
- a 3D array with shape `(n_time_steps, n_samples, n_variables)`

When a model also accepts `time_points`, it expects a one-dimensional array-like object with length `n_time_steps`.

## Time-Conditioned Bandwidth Model

### `TimeDependentVine`

Import:

```python
from dvc_package.time.models import TimeDependentVine
```

Constructor:

```python
TimeDependentVine(base_vine, bandwidth_flow=None, device="cpu")
```

Main methods:
- `set_reference_data(data_by_time, time_points=None)`
- `forward(x, time)`
- `fit_bandwidth_flow(train_data_by_time, time_points=None, ...)`
- `sample(n_samples, time_points)`

Use this class when the time-dependent component is a bandwidth network attached to a fitted base vine. The current likelihood path uses level-0 edges.

Example:

```python
from dvc_package.core.vine_factory import create_vine
from dvc_package.time.models import create_time_dependent_vine

base_vine = create_vine("c-vine", vine_depth=3)
time_vine = create_time_dependent_vine(base_vine, hidden_dims=[64, 32], device="cpu")
time_vine.set_reference_data(train_windows, time_points=t_grid)
time_vine.fit_bandwidth_flow(train_windows, time_points=t_grid, n_epochs=50)
```

## Joint Dynamic Parametric Models

### `JointDynamicCVine`

Import:

```python
from dvc_package.time import JointDynamicCVine
```

Constructor:

```python
JointDynamicCVine(
    families=None,
    n_basis=4,
    smoothness_penalty=5.0,
    ridge_penalty=1e-3,
    order=None,
    student_df_grid=(4.0, 8.0, 16.0),
    maxiter=80,
)
```

Main methods:
- `fit(data_by_time, time_points=None)`
- `evaluate(data_by_time)`

Use this class when edge parameters should follow a jointly fitted smooth trajectory over time.

Example:

```python
model = JointDynamicCVine(
    families=["gaussian", "student", "clayton", "frank"],
    n_basis=4,
    smoothness_penalty=2.0,
)
result = model.fit(train_windows, time_points=t_grid)
nll = model.evaluate(test_windows)
```

### `LatentStateDynamicCVine`

Import:

```python
from dvc_package.time import LatentStateDynamicCVine
```

Constructor:

```python
LatentStateDynamicCVine(
    families=None,
    order=None,
    selection_n_basis=4,
    selection_smoothness_penalty=5.0,
    latent_dim=2,
    transition_penalty=1e-2,
    n_epochs=250,
    lr=2e-2,
)
```

Main methods:
- `fit(data_by_time, time_points=None)`
- `evaluate(data_by_time)`

Use this class when multiple edge trajectories should share a low-rank latent temporal state.

### `RegularizedDynamicCVine`

Import:

```python
from dvc_package.time import RegularizedDynamicCVine
```

Constructor:

```python
RegularizedDynamicCVine(
    families=None,
    root_switch_penalty=0.0,
    family_switch_penalty=0.0,
    parameter_drift_penalty=0.0,
    parameter_smoothing=0.0,
    root_score_method="aic",
)
```

Main methods:
- `fit(data_by_time, time_points=None)`
- `evaluate(data_by_time)`

Use this class when each window should still be fit separately but the selected roots, families, and parameters should be regularized over time.

## Dynamic Nonparametric Models

### `WindowedDynamicNonparametricVine`

Import:

```python
from dvc_package.time import WindowedDynamicNonparametricVine
```

Constructor:

```python
WindowedDynamicNonparametricVine(
    vine_type="auto",
    order=None,
    knots=30,
    optimize_structure=False,
    selection_criterion="aic",
    optimization_method="sequential",
    optimization_criterion="kendall_tau",
    vine_kwargs=None,
    npc_dict=None,
)
```

Main methods:
- `fit(data_by_time, time_points=None)`
- `evaluate(data_by_time)`

Use `vine_type="auto"` to select among `C-vine`, `D-vine`, and `R-vine` on pooled data before fitting each time step.

### `JointDynamicNonparametricVine`

Import:

```python
from dvc_package.time import JointDynamicNonparametricVine
```

Constructor:

```python
JointDynamicNonparametricVine(
    vine_type="auto",
    order=None,
    knots=30,
    trajectory_type="basis",
    trajectory_kwargs=None,
    optimize_structure=False,
    selection_criterion="aic",
    optimization_method="sequential",
    optimization_criterion="kendall_tau",
    batch_size=256,
    n_epochs=120,
    warm_start_epochs=80,
    lr=1e-2,
    smoothness_penalty=1e-2,
    gradient_clip=5.0,
    normalization_iters=12,
    final_normalization_iters=24,
    vine_kwargs=None,
)
```

Main methods:
- `fit(data_by_time, time_points=None)`
- `evaluate(data_by_time)`

Use this class when nonparametric bandwidths should be fitted as a joint time trajectory rather than refit independently at each time step.

## Trajectory Models

### `BasisTrajectory`

Import:

```python
from dvc_package.time import BasisTrajectory
```

Constructor:

```python
BasisTrajectory(
    output_dim,
    n_basis=4,
    width_scale=1.5,
    constraint="identity",
    min_value=0.0,
    max_value=1.0,
)
```

Use for smooth radial-basis time trajectories.

### `MLPTrajectory`

Import:

```python
from dvc_package.time import MLPTrajectory
```

Constructor:

```python
MLPTrajectory(
    output_dim,
    hidden_dims=None,
    activation="relu",
    dropout_rate=0.1,
    constraint="identity",
    min_value=0.0,
    max_value=1.0,
)
```

Use for unconstrained neural trajectories.

### `StateSpaceTrajectory`

Import:

```python
from dvc_package.time import StateSpaceTrajectory
```

Constructor:

```python
StateSpaceTrajectory(
    output_dim,
    latent_dim=3,
    n_steps=8,
    transition_penalty=1e-2,
    constraint="identity",
    min_value=0.0,
    max_value=1.0,
)
```

Use for latent state-space trajectories with a shared linear readout.

## Minimal Dynamic Example

```python
from dvc_package.time import JointDynamicCVine, JointDynamicNonparametricVine

joint_par = JointDynamicCVine(
    families=["gaussian", "student", "frank"],
    n_basis=3,
    smoothness_penalty=1.0,
)
par_result = joint_par.fit(train_windows, time_points=t_grid)
par_nll = joint_par.evaluate(test_windows)

joint_np = JointDynamicNonparametricVine(
    vine_type="auto",
    knots=21,
    trajectory_type="basis",
    trajectory_kwargs={"n_basis": 3},
)
np_result = joint_np.fit(train_windows, time_points=t_grid)
np_nll = joint_np.evaluate(test_windows)
```
