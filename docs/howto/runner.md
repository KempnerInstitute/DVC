# Experiment Runner How-To

## Run One Experiment

```bash
python scripts/run_experiment.py configs/finance_crisis_benchmarks.yaml
```

## Change Logging Verbosity

```bash
python scripts/run_experiment.py configs/finance_crisis_benchmarks.yaml --log-level DEBUG
```

## Override Output Directory

```bash
python scripts/run_experiment.py configs/finance_crisis_benchmarks.yaml --output-dir results/custom_run
```

## Generate and Inspect Example Configs

```bash
python scripts/run_experiment.py --create-examples
python scripts/run_experiment.py --list-examples
```

## Common Failure Checks

- Confirm package import path with `pip install -e .`.
- Verify the config path exists.
- Start with smaller sample sizes when debugging long runs.
- Keep the exact YAML config next to any result directory you plan to report.
