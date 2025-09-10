# Time-dependent runner

Use `scripts/time_runner.py` to train and evaluate a time-dependent vine copula on synthetic scenarios and log metrics.

Example:

```bash
python scripts/time_runner.py --outdir runs/exp1
```

Config (JSON file):

```json
{
  "dim": 4,
  "scenario": "regime", 
  "n_time_steps": 20,
  "n_per_time": 120,
  "flow": "spline",
  "vine_type": "c-vine",
  "epochs": 12,
  "lr": 0.001,
  "early_patience": 4
}
```

Run with config:

```bash
python scripts/time_runner.py --config cfg.json --outdir runs/exp2
```
