# Running Experiments

Experiments are configured with YAML files in `configs/` and executed with:

```bash
python scripts/run_experiment.py <config.yaml>
```

## Useful Commands

```bash
# list available configs
python scripts/run_experiment.py --list-examples

# generate baseline config files in configs/
python scripts/run_experiment.py --create-examples

# run a provided config
python scripts/run_experiment.py configs/probability_analysis.yaml
python scripts/run_experiment.py configs/entropy_analysis.yaml
python scripts/run_experiment.py configs/time_dependent.yaml
```

## Config Structure

Typical top-level keys:

- `name`
- `description`
- `output_dir`
- `seed`
- `data_config`
- `vine_config`
- `analysis_config`
- `plot_config`

Use the existing config files as templates for custom studies.

## Outputs

Runs typically write:

- result summaries (`.json`/`.pkl`)
- logs
- plots (if enabled)

under the configured `output_dir`.

## Paper Table Generation

To run the standard benchmark configs and export paper-ready tables:

```bash
python scripts/generate_neurips_tables.py --run
```

This creates CSV and LaTeX tables in `results/neurips_tables/`.

## Reproducibility Tips

- Fix `seed` in config and avoid changing it between ablations.
- Keep sample counts and Monte Carlo settings fixed when comparing methods.
- Store the exact config file next to each result directory.
