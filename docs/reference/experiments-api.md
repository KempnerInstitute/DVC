# Experiment API Reference

The package provides two complementary runner layers:

| Layer | Module | Purpose | Driven by |
| --- | --- | --- | --- |
| `ExperimentRunner` | `dvc_package.experiments.experiment_framework` | YAML-configured dispatcher across the bundled experiment types | `scripts/run_experiment.py` |
| `BenchmarkRunner` | `dvc_package.experiments.runner` | Programmatic Cartesian sweep over scenarios, dimensions, vine types, families, and optimization methods | `dvc-experiment` CLI; direct Python use |

Both classes are re-exported from `dvc_package.experiments`.

## YAML Dispatcher: `ExperimentRunner`

```python
from dvc_package.experiments import ExperimentRunner, ExperimentConfig

runner = ExperimentRunner()
results = runner.run_from_config("configs/probability_analysis.yaml")
```

Supported `analysis_config.experiment_type` values:

- `probability_analysis`
- `entropy_analysis`
- `time_dependent`

The paper-specific `simulation_benchmarks` suite (multi-panel diagnostic
figures and the manuscript scenario set) lives outside the public package
under `drafts/projects/paper_benchmarks/run_suite.py` because the figure
layouts are paper-bound. The reusable scenario generators and metric helpers
remain importable from `dvc_package.experiments.simulation_benchmarks`.

Each config is a YAML file with at minimum:

```yaml
name: my_experiment
description: Brief description.
output_dir: results/my_experiment
seed: 0
analysis_config:
  experiment_type: probability_analysis
```

`scripts/run_experiment.py --create-examples` materializes one starter config
for each of the first three experiment types under `configs/`.

## Programmatic Benchmark Runner: `BenchmarkRunner`

`BenchmarkRunner` consumes a `BenchmarkConfig` dataclass and runs every
combination of (scenario × dimension × vine type × family × optimization
method).

```python
from dvc_package.experiments import BenchmarkConfig, BenchmarkRunner

cfg = BenchmarkConfig(
    name="basic_benchmark",
    description="Static benchmark run",
    output_dir="results/basic_benchmark",
    data_config={"type": "benchmark", "scenarios": ["gaussian"], "n_samples": 500},
    vine_types=["c-vine", "d-vine"],
    copula_families=["ind", "gaussian", "frank"],
    dimensions=[3, 5],
    optimization_methods=["sequential"],
)

runner = BenchmarkRunner(cfg)
results = runner.run()
```

`BenchmarkConfig` fields:

- Required: `name`, `description`, `output_dir`, `data_config`, `vine_types`,
  `copula_families`, `dimensions`, `optimization_methods`.
- Optional: `optimization_enabled`, `time_dependent`, `time_config`,
  `evaluation_metrics`, `n_monte_carlo_samples`, `n_bootstrap_runs`,
  `n_parallel_jobs`, `random_seed`, `device`, `save_models`,
  `save_detailed_results`, `create_plots`.

### `run_experiment` Helper

For one-shot benchmark runs from a Python dict:

```python
from dvc_package.experiments import run_experiment

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

## CLI Entry Point

The `dvc-experiment` CLI wraps `run_experiment` (the function) and accepts a
`--config` YAML pointing at a `BenchmarkConfig`-shaped document:

```bash
dvc-experiment --config configs/your_benchmark.yaml --output-dir results/
```

For YAML configs that target the higher-level `ExperimentRunner` dispatcher,
use the script runner instead:

```bash
python scripts/run_experiment.py configs/probability_analysis.yaml
```
