# High-Order Dependency Analysis Protocol

This protocol is for using DVC to study high-order dependence in:

- neural population recordings,
- and multi-agent interaction trajectories.

## 1. Data Setup

- Input matrix per window: `X in R^{n_samples x d}`.
- For temporal analysis: split into windows or use `(time, sample, variable)` arrays.
- Standardize each variable per window when scales differ strongly.

## 2. Baselines

Run at least:

1. Pairwise correlation baseline (Pearson/Spearman matrix).
2. Multivariate Gaussian baseline (covariance + analytic entropy where applicable).
3. DVC vine models (`c-vine`, `d-vine`, `r-vine`) with shared data splits.

## 3. Core Experiments

For each dataset/window:

1. Fit C-vine, D-vine, and R-vine with identical family sets.
2. If structure is unknown, run `optimize_vine_structure(...)`.
3. Record:
   - fitted edge families,
   - criterion scores (Kendall tau / AIC / entropy proxy),
   - entropy and MI estimates.

## 4. Temporal Experiments

Two options:

1. Windowed static vines:
   - fit one vine per time window,
   - track edge changes, entropy/MI trajectories.
2. Time-flow models:
   - use `dvc_package.time` flows for continuous-time bandwidth modeling,
   - compare against the windowed baseline.

## 5. Metrics to Report

- `H(X)` (vine entropy estimate)
- `I(X_A; X_B)` for meaningful subsets
- fraction of non-Gaussian selected edges
- stability of edge set across windows (e.g., Jaccard overlap)
- predictive log-likelihood on held-out windows (if available)

## 6. Statistical Reliability

- Repeat each condition across multiple seeds.
- Bootstrap windows/samples where possible.
- Report mean +/- std (or confidence intervals) for key metrics.

## 7. Recommended Figures

1. Pairwise correlation matrix vs vine edge graph.
2. Edge-family composition bar chart.
3. Entropy/MI trajectory over time.
4. Stability curve of edge overlap across consecutive windows.

## 8. Paper-Ready Table Template

- Dataset
- `d`, `n_samples`, number of windows
- Best vine type by selection criterion
- Entropy estimate (vine vs Gaussian)
- % non-Gaussian edges
- Temporal stability score

For paper reproduction, keep the exact run configs, seeds, result manifests,
and figure-generation scripts in the dedicated reproduction bundle rather than
mixing them into the public package API.
