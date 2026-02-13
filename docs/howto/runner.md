# Experiment Runner How-To

## Run One Experiment

```bash
python scripts/run_experiment.py configs/probability_analysis.yaml
```

## Change Logging Verbosity

```bash
python scripts/run_experiment.py configs/entropy_analysis.yaml --log-level DEBUG
```

## Override Output Directory

```bash
python scripts/run_experiment.py configs/time_dependent.yaml --output-dir results/custom_run
```

## Generate and Inspect Example Configs

```bash
python scripts/run_experiment.py --create-examples
python scripts/run_experiment.py --list-examples
```

## Common Failure Checks

- Confirm package import path (`pip install -e .`).
- Verify config path exists.
- Start with smaller sample sizes when debugging long runs.

