# Experiment Runner How-To

There are two experiment entry points in `scripts/`:

- `run_experiment.py` consumes generic YAML configs whose
  `analysis_config.experiment_type` is one of `probability_analysis`,
  `entropy_analysis`, or `time_dependent`.
- `run_finance_crisis_benchmark.py` is a dedicated runner for the real-world
  multi-asset crisis benchmark.

The paper-specific simulation benchmark orchestrator lives outside the public
package, under `drafts/projects/paper_benchmarks/run_suite.py`.

## Materialize and Run an Example

```bash
python scripts/run_experiment.py --create-examples
python scripts/run_experiment.py configs/probability_analysis.yaml
```

## Run the Finance Crisis Benchmark

```bash
python scripts/run_finance_crisis_benchmark.py --config configs/finance_crisis_benchmarks.yaml
```

## Change Logging Verbosity

```bash
python scripts/run_experiment.py configs/probability_analysis.yaml --log-level DEBUG
```

## Override the Output Directory

```bash
python scripts/run_experiment.py configs/probability_analysis.yaml --output-dir results/custom_run
```

## List Available Configs

```bash
python scripts/run_experiment.py --list-examples
```

## Common Failure Checks

- Confirm the package is importable with `pip install -e .`.
- Verify the config path exists and has an `analysis_config.experiment_type`
  the runner supports.
- Start with smaller sample sizes when debugging long runs.
- Keep the exact YAML config next to any result directory you plan to report.
