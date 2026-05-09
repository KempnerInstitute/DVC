# Dynamic Vine Copulas

Dynamic Vine Copulas (DVC) is a Python package for vine-copula modeling,
dependence estimation, and time-varying copula analysis. It provides reusable
building blocks for static C-, D-, and R-vines, parametric and nonparametric
pair-copula fitting, information-theoretic readouts, and dynamic C-vine
estimators used for temporal dependence diagnostics.

This repository is being prepared for public release. The public package is
kept separate from paper-only orchestration: reusable library code lives under
`src/dvc_package`, runnable examples live under `examples` and `scripts`, and
paper-specific figure/table workflows are staged under `drafts/projects`.

## What You Can Do

- Fit static C-, D-, and R-vines with Gaussian, Student-t, Clayton, Frank,
  Gumbel, Joe, and independence pair-copula families.
- Fit nonparametric local-likelihood vines for exploratory dependence
  modeling.
- Evaluate copula log likelihoods, sample fitted vines, and estimate entropy
  and mutual information from fitted models.
- Compare vine structures and optimize vine type/ordering with Kendall-tau,
  AIC-style, entropy-guided, and hybrid criteria.
- Fit temporal C-vine variants for time-indexed dependence:
  `JointDynamicCVine`, `SwitchingDynamicCVine`, `RegularizedDynamicCVine`,
  `LatentStateDynamicCVine`, and experimental dynamic nonparametric C-vines.
- Run reproducible YAML-configured experiments through the public experiment
  runner.

## Installation

With conda:

```bash
conda env create -f environment.yml
conda activate dvc-env
pip install -e .
```

With venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Validate the install:

```bash
python scripts/test_installation.py
python -m pytest -q
```

## Quick Start

```python
import numpy as np

from dvc_package.core.vine_factory import create_vine
from dvc_package.core.vine_model import evaluate_vine, fit_vine

rng = np.random.default_rng(0)
x = rng.multivariate_normal(
    mean=np.zeros(3),
    cov=np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.4], [0.2, 0.4, 1.0]]),
    size=1000,
).astype("float32")

families = ["independence", "gaussian", "student", "clayton", "gumbel"]
vine = create_vine("c-vine", vine_depth=x.shape[1], families=families)

fit_vine(
    vine,
    x,
    gen_dict={"param": True, "binning": False, "fitted": True},
    npc_dict={},
    par_dict={"param_families": families},
    bin_dict={},
)

log_density = evaluate_vine(vine, x[:10])
print(log_density)
```

Try the included examples:

```bash
python examples/basic_vine_example.py
python examples/entropy_analysis_example.py
python examples/time_dependent_example.py
python scripts/run_dynamic_cvine_example.py --output-dir results/dynamic_cvine_example
```

## Configured Experiments

The general experiment runner consumes YAML configs whose
`analysis_config.experiment_type` selects the experiment class
(`probability_analysis`, `entropy_analysis`, `time_dependent`):

```bash
python scripts/run_experiment.py --create-examples
python scripts/run_experiment.py --list-examples
python scripts/run_experiment.py configs/probability_analysis.yaml
```

The real-world finance crisis benchmark uses a dedicated entry point because it
fetches data from public sources and orchestrates its own scenario set:

```bash
python scripts/run_finance_crisis_benchmark.py --config configs/finance_crisis_benchmarks.yaml
```

Paper-reproduction configs, figure scripts, and result manifests are staged
under `drafts/projects` and are not part of the public package release.

## Documentation

- Docs index: `docs/index.md`
- Setup guide: `docs/setup.md`
- Repository structure: `docs/structure.md`
- Static fitting: `docs/user-guide/fitting.md`
- Evaluation: `docs/user-guide/evaluation.md`
- Time-dependent models: `docs/user-guide/time-dependent.md`
- Schematics: `docs/schematics.md`
- Release plan: `docs/release-plan.md`
- Data release plan: `docs/data-release.md`

Build HTML docs locally with MkDocs:

```bash
pip install mkdocs
mkdocs build
```

## Repository Layout

```text
DVC/
├── src/dvc_package/      # reusable package code
├── examples/             # small runnable examples
├── scripts/              # public command-line entry points
├── configs/              # reusable experiment configs
├── tests/                # public test suite
├── docs/                 # user, API, schematic, and release docs
├── drafts/               # ignored paper/project workspace
└── archive/              # local-only legacy material, ignored by release
```

## Public Release Boundary

The release package should include general-purpose code, examples, tests, docs,
and small configs. It should not include local logs, generated results, raw
datasets, paper-only figure scripts, machine-specific paths, or exploratory notes. The
paper reproduction bundle will be released separately with pinned configs,
result manifests, and dataset acquisition instructions.

See `docs/release-plan.md` and `docs/data-release.md` for the staged release
checklist.
