# Final Showcase Benchmark

This folder packages the final standalone showcase benchmark.

This folder adds:

- `run_showcase_benchmark.py`: standalone final benchmark entrypoint
- `showcase_analysis_utils.py`: stable shared utilities used by the final benchmark
- `generate_fig7_showcase.py`: final figure-generation entrypoint
- `run_final_showcase_benchmark.py`: convenience wrapper with the recommended defaults
- `run_final_showcase.sh`: small bash launcher
- `SHOWCASE_CHANGES.md`: short note on what changed from the original setup and why

## Recommended Modes

You can also call the standalone benchmark entrypoint directly:

```bash
conda run -n dvc python scripts_ale_final/run_showcase_benchmark.py \
  --preset contrast_harder \
  --out results/showcase_ale_final/contrast_harder_parametric
```

### Main benchmark

Use the repaired **parametric DVC** benchmark as the main comparator:

```bash
conda run -n dvc python scripts_ale_final/run_final_showcase_benchmark.py --mode parametric
```

This writes by default to:

- `results/showcase_ale_final/contrast_harder_parametric`

The resulting `summary.json` now includes per-window oracle targets:
`truth_tc_total`, `truth_tc_pair_oracle`, `truth_tc_higher_oracle`,
`truth_pair_mi01`, `truth_pair_mi56`, and `truth_tail_lambda_lower`.  The
phasewise summary also reports absolute errors against these truth values.

### Supplementary benchmark with repaired NP DVC

Add the repaired **nonparametric DVC** as a supplementary comparator:

```bash
conda run -n dvc python scripts_ale_final/run_final_showcase_benchmark.py --mode with_np
```

This writes by default to:

- `results/showcase_ale_final/contrast_harder_with_np`

The `with_np` mode uses the repaired nonparametric defaults that were found to be the most scientifically credible in the audit:

- `d-vine`
- `knots = 7`
- `higher_tree_validation_margin = 0.05`
- `np_temporal_smoothing = 0.12`, which couples the nonparametric density grids
  across nearby time windows while preserving validated independence edges

## Bash Launcher

You can run the same modes through the bash wrapper:

```bash
./scripts_ale_final/run_final_showcase.sh parametric
./scripts_ale_final/run_final_showcase.sh with_np
```

Extra flags are forwarded to the Python wrapper. For example:

```bash
./scripts_ale_final/run_final_showcase.sh with_np --skip-mine --skip-nf
./scripts_ale_final/run_final_showcase.sh parametric --include-regularized-dvc
```

## Useful Options

Examples:

```bash
conda run -n dvc python scripts_ale_final/run_final_showcase_benchmark.py \
  --mode parametric \
  --preset contrast_harder \
  --include-regularized-dvc
```

```bash
conda run -n dvc python scripts_ale_final/run_final_showcase_benchmark.py \
  --mode with_np \
  --skip-mine \
  --skip-nf
```

Available wrapper options:

- `--mode {parametric,with_np}`
- `--preset {fixed,contrast_harder}`
- `--n-seeds`
- `--base-seed`
- `--n-per-time`
- `--multiplicative-noise-std`
- `--skip-mine`
- `--skip-nf`
- `--include-regularized-dvc`
- `--out`

## Recommendation

- Use `contrast_harder` + `parametric` as the main final benchmark.
- Use `with_np` only as a supplementary robustness comparison.
- Do not promote the nonparametric DVC line into the main Figure 7 benchmark by default.
- Report model quality against the oracle fields, not only against the
  DVC-minus-baseline gaps.

The default `contrast_harder` multiplicative noise is `0.10`, matching the
hard stress-test setting used for the final comparison. This creates a
near-deterministic triplet phase: DVC detects and decomposes the signal better
than Gaussian baselines, but all finite-sample fitted models underestimate the
large oracle TC. Use `--multiplicative-noise-std` for calibration sweeps.

## Final Figure Script

The final figure entrypoint is:

```bash
conda run -n dvc python scripts_ale_final/generate_fig7_showcase.py
```

Figure 7 overlays these oracle targets on the total-correlation,
pair-vs-higher-order, and representative pairwise-MI panels.

To point it at a different results folder:

```bash
DVC_SHOWCASE_RESULTS_DIR=results/showcase_ale_final/contrast_harder_with_np \
conda run -n dvc python scripts_ale_final/generate_fig7_showcase.py
```
