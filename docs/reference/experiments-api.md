# Experiment API Reference

## `ExperimentConfig`

Import:

```python
from dvc_package.experiments.runner import ExperimentConfig
```

Purpose:
- stores the configuration passed to `ExperimentRunner`

Required fields:
- `name`
- `description`
- `output_dir`
- `data_config`
- `vine_types`
- `copula_families`
- `dimensions`
- `optimization_methods`

Common optional fields:
- `optimization_enabled`
- `time_dependent`
- `time_config`
- `evaluation_metrics`
- `n_monte_carlo_samples`
- `n_bootstrap_runs`
- `n_parallel_jobs`
- `random_seed`
- `device`

Example:

```python
cfg = ExperimentConfig(
    name="basic_benchmark",
    description="Static benchmark run",
    output_dir="results/basic_benchmark",
    data_config={"type": "benchmark", "scenarios": ["gaussian"], "n_samples": 500},
    vine_types=["c-vine", "d-vine"],
    copula_families=["ind", "gaussian", "frank"],
    dimensions=[3, 5],
    optimization_methods=["sequential"],
)
```

## `ExperimentRunner`

Import:

```python
from dvc_package.experiments.runner import ExperimentRunner
```

Constructor:

```python
ExperimentRunner(config)
```

Main method:

```python
runner.run()
```

Behavior:
- prepares datasets from `config.data_config`
- generates the experiment grid
- runs the configured fits and evaluations
- writes outputs under `config.output_dir`

Example:

```python
runner = ExperimentRunner(cfg)
results = runner.run()
```

## `run_experiment`

Import:

```python
from dvc_package.experiments.runner import run_experiment
```

Signature:

```python
run_experiment(config, output_dir=None, n_runs=1, seed=42, verbose=False)
```

Use this helper when the configuration is already available as a dictionary or an `ExperimentConfig` instance.

Example:

```python
results = run_experiment(
    {
        "name": "quick_run",
        "description": "Single benchmark run",
        "output_dir": "results/quick_run",
        "data_config": {"type": "synthetic", "n_samples": 400},
        "vine_types": ["c-vine"],
        "copula_families": ["ind", "gaussian", "clayton"],
        "dimensions": [4],
        "optimization_methods": ["sequential"],
    },
    n_runs=2,
    seed=7,
)
```

## Config File Runner

The repository also provides a script interface for YAML configs:

```bash
python scripts/run_experiment.py drafts/configs/simulation_benchmarks.yaml
```

Use the script path when the source of truth is a YAML file instead of a Python config object.
