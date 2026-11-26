# Schematics

This page summarizes the static and temporal models and their use in a research project.

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

## From observations to a figure

```text
simulated observations or public recordings
       |
       v
study-specific preprocessing and data splits
       |
       v
dvc_package.modulation       fit conditional trajectories or binary contrasts
       |
       v
held-out predictions, estimated effects and simulation controls
       |
       v
projects/vine_modulation/figures/       source tables and figures
```

The [project guide](user-guide/vine-modulation.md) gives the commands for both worked examples and all seven figures.
