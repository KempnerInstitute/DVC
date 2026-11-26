# Repository structure

| Directory | Contents |
| --- | --- |
| `src/dvc_package/core/` | Vine structures, copula families, density evaluation, sampling and information estimates |
| `src/dvc_package/time/` | Smooth and switching dynamic C-vines and temporal data helpers |
| `src/dvc_package/modulation/` | Conditional pair trajectories, population fits and fixed-margin spike contrasts |
| `src/dvc_package/optimization/` | Structure criteria and ordering algorithms |
| `src/dvc_package/baselines/` | Reference models for dependence and predictive comparisons |
| `src/dvc_package/experiments/` | Configured experiments and reusable benchmarks |
| `src/dvc_package/real_data/` | Public dataset loaders |
| `src/dvc_package/cli/` | Library command-line entry points |
| `examples/`, `scripts/`, `configs/` | Runnable examples, study-independent tools and configuration files |
| `projects/` | Research workflows that use the library |
| `tests/` | Regression and API tests for the installed package |

The [vine modulation project](user-guide/vine-modulation.md) combines simulations, hippocampal analyses, local tests and figure recipes. It imports the same `dvc_package.modulation` implementation available to other users.
