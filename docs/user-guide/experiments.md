# Running Experiments

DVC experiments are configured with YAML files. Use `configs/` for reusable
public examples and project-specific config directories for local studies.

```bash
python scripts/run_experiment.py <config.yaml>
```

## Useful Commands

```bash
# Materialize bundled example configs into configs/.
python scripts/run_experiment.py --create-examples

# List the YAML configs currently in configs/.
python scripts/run_experiment.py --list-examples

# Run one of the example configs.
python scripts/run_experiment.py configs/probability_analysis.yaml

# Run the real-world finance crisis benchmark (separate runner).
python scripts/run_finance_crisis_benchmark.py --config configs/finance_crisis_benchmarks.yaml

# Run a standalone dynamic C-vine example.
python scripts/run_dynamic_cvine_example.py --output-dir results/dynamic_cvine_example
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

Time-dependent configs can additionally set `time_config` fields such as
windowing, model variant, smoothness penalties, and held-out scoring options.
Store the exact config file next to each result directory for reproducibility.

## Outputs

Runs typically write result summaries, logs, and optional plots under the
configured `output_dir`. Generated results should stay out of version control
unless they are small, intentional fixtures.

## Paper Reproduction

Paper table/figure generation is staged under `drafts/projects/` and will be
released as a separate reproduction bundle. It is not part of the public package
API.

## Reproducibility Tips

- Fix `seed` and data splits before comparing methods.
- Keep sample counts and Monte Carlo settings fixed across ablations.
- Record package versions and hardware for runtime-sensitive experiments.
- Use held-out copula likelihoods consistently when comparing dependence
  models.
