# Dynamic Vine Copulas

Dynamic Vine Copulas (DVC) is a Python library for fitting multivariate dependence models. A vine represents a joint distribution through conditional bivariate copulas. DVC supports static C-, D- and R-vines, time-dependent C-vines, and models in which other variables modulate a pair's association.

**Studying changing interactions?** The vine modulation project includes a continuous example in which pairwise distributions stay fixed while a three-variable association reverses. A second example uses three binary spike indicators and holds all pair margins fixed. Both run on a CPU with simulated data. [Study guide](projects/vine_modulation/README.md)

## Installation

From a checkout of this repository, use Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/test_installation.py
```

The study guide gives a separate, version-pinned Python 3.10 environment for reproducing its figures. Conda users can follow the [setup guide](docs/setup.md).

## Fit a static vine

The example fits a three-variable C-vine and evaluates its copula density on ten held-out observations. Each row is an observation and each column is a variable.

```python
import numpy as np

from dvc_package import create_vine, evaluate_vine, fit_vine

rng = np.random.default_rng(0)
covariance = [[1.0, 0.6, 0.2], [0.6, 1.0, 0.4], [0.2, 0.4, 1.0]]
samples = rng.multivariate_normal(np.zeros(3), covariance, size=510).astype("float32")
train, held_out = samples[:500], samples[500:]
families = ["independence", "gaussian", "student", "clayton", "gumbel"]
vine = create_vine("c-vine", vine_depth=train.shape[1], families=families)
fit_vine(
    vine, train,
    gen_dict={"param": True, "binning": False, "fitted": True},
    npc_dict={}, par_dict={"param_families": families}, bin_dict={},
)
joint_density, copula_density, log_marginal_density = evaluate_vine(vine, held_out)
print(copula_density)
```

## Choose an analysis

| Question | Where to start |
| --- | --- |
| Which copula families and vine structure fit the observations? | [Static fitting](docs/user-guide/fitting.md) |
| How do density, sampling, entropy and mutual information work with a fitted vine? | [Evaluation](docs/user-guide/evaluation.md) |
| How does a pair's association or tail shape evolve over time? | [Time-dependent models](docs/user-guide/time-dependent.md) |
| Does a third variable modulate a pair, and does that modulation change? | [Estimators, examples and study reproduction](projects/vine_modulation/README.md) |
| How do I run a configured benchmark? | [Experiment runner](docs/howto/runner.md) |

The reusable package lives in `src/dvc_package/`. Runnable examples are in `examples/`, study workflows in `projects/`, and command-line tools in `scripts/`. [Documentation](docs/index.md) · [Contributing](CONTRIBUTING.md) · [MIT license](LICENSE)
