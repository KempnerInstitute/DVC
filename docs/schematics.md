# Schematics

This page summarizes the model and repository boundaries used by DVC.

## Model Flow

```text
raw observations
      |
      v
rank pseudo-observations
      |
      v
fixed vine factorization
      |
      +--> static C-/D-/R-vine fitting
      |
      +--> temporal C-vine variants
             |
             +--> smooth edge trajectories
             +--> switching family/parameter paths
             +--> regularized windowed controls
      |
      v
held-out copula scores, sampling, entropy, MI, diagnostics
```

## Release Boundary

```text
public repository
  src/dvc_package/      reusable library
  examples/             runnable minimal examples
  scripts/              general entry points
  configs/              reusable configs
  docs/                 public documentation
  tests/                public regression tests

project workspace
  drafts/projects/      paper-only scripts and runbooks
  drafts/figures/       generated paper figures
  drafts/tables/        generated paper tables
  drafts/logs/          local run logs
```

The public package should remain useful without the paper workspace. The paper
workspace can depend on the public package, but the package should not depend on
paper-only scripts.
