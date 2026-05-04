# Repository Structure

The repository is organized around a clean boundary between reusable package
code and project-specific research workflows.

```text
DVC/
├── src/dvc_package/      # reusable library code
├── examples/             # small runnable examples
├── scripts/              # public command-line entry points
├── configs/              # reusable experiment configs
├── tests/                # public regression and unit tests
├── docs/                 # user, API, schematic, and release docs
├── drafts/               # ignored paper/project workspace
└── archive/              # legacy reference code
```

## Public Package

`src/dvc_package/` is the importable package. The main subpackages are:

- `core`: vine objects, pair-copula fitting, density evaluation, sampling,
  structure creation, and information estimation.
- `time`: temporal C-vine estimators, time-series helpers, trajectory models,
  and normalizing-flow utilities.
- `optimization`: vine-structure optimization criteria and search routines.
- `experiments`: YAML-configured experiment runner and reusable benchmark
  helpers.
- `baselines`: Gaussian, state-space, TVGL-style, MINE, NF-copula, and optional
  backend comparators.
- `real_data`: public dataset loaders/preprocessors that can be used outside
  the paper workflow.

## Public Scripts

Root-level `scripts/` should contain only reusable entry points:

- package and installation checks,
- generic experiment execution,
- reusable public-data download helpers,
- small dynamic-vine examples.

Paper-only figure generation, Slurm jobs, local diagnostics, and one-off
benchmark orchestration should live under `drafts/projects/` rather than
`scripts/`.

## Project Workspace

`drafts/` is ignored by the public package repository. It is used for:

- manuscripts and venue-specific build files,
- paper figures and tables,
- paper-specific configs,
- project-only scripts,
- generated logs, results, and temporary artifacts.

Keeping this directory out of the public package avoids leaking local paths,
development notes, generated data, and venue-specific paper workflows into the
general-purpose release.

## Release Boundary

Before a public release, verify that:

- no machine-specific paths or local notes appear outside `drafts/` or `archive/`,
- public docs only reference runnable public commands,
- `scripts/` contains reusable entry points only,
- tests pass from a clean checkout,
- generated data/results are not tracked,
- paper reproduction assets are packaged separately with a manifest.
