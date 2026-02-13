# DVC: Dynamic Vine Copula Library

PyTorch-based tooling for vine copula modeling, information-theoretic analysis, and early-stage time-dependent dependency modeling.

## What This Repository Contains

- Parametric pair-copula fitting (`gaussian`, `student`, `clayton`, `frank`, `gumbel`, `ind`/`independence`)
- Vine structure construction for C-vine, D-vine, and R-vine
- Vine-level density evaluation, sampling, entropy, and mutual information utilities
- Structure optimization routines (sequential, genetic, entropy-guided, hybrid)
- Time-series helpers and neural flow modules for time-conditioned bandwidth modeling
- YAML-based experiment runner and reference configurations
- Archived TensorFlow baseline in `archive/`

## Installation

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate dvc-env
pip install -e .
```

### Option 2: venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Validation

```bash
# Full unit test suite
pytest -q

# Optional environment verification
python scripts/test_installation.py
```

## Minimal Usage

```python
import numpy as np
from dvc_package.core.vine_factory import create_vine
from dvc_package.core.vine_model import fit_vine
from dvc_package.core.info_estimation import vine_entropy

# Synthetic 3D data
data = np.random.multivariate_normal(
    mean=[0.0, 0.0, 0.0],
    cov=[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]],
    size=1000,
)

vine = create_vine("c-vine", vine_depth=3, families=["ind", "gaussian", "clayton"])

gen_dict = {"param": True, "binning": False, "fitted": False}
npc_dict = {}
par_dict = {"param_families": ["ind", "gaussian", "clayton"]}
bin_dict = {}

fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)

entropy_bits = vine_entropy(vine, {"alpha": 0.05, "cases": 1000, "iterations": 10})
print(entropy_bits)
```

## Running Examples

```bash
python examples/basic_vine_example.py
python examples/entropy_analysis_example.py
python examples/time_dependent_example.py
```

## Running Configured Experiments

```bash
# List/create configs
python scripts/run_experiment.py --list-examples
python scripts/run_experiment.py --create-examples

# Run one config
python scripts/run_experiment.py configs/probability_analysis.yaml
```

## Generate NeurIPS Tables

Run benchmark configs and generate CSV/LaTeX tables for the paper:

```bash
python scripts/generate_neurips_tables.py --run
```

Outputs are written under `results/neurips_tables/`:
- `master_summary.csv` / `.tex`
- `probability_vine_detail.csv` / `.tex`
- `entropy_method_detail.csv` / `.tex`
- `time_pair_detail.csv` / `.tex`

## Prepare Standalone Draft Assets

To generate benchmark artifacts and vendor all paper assets into `drafts/`
(tables, figures, and result JSON summaries), run:

```bash
python scripts/prepare_draft_assets.py --run --compile
```

This writes to:
- `drafts/tables/neurips_tables/`
- `drafts/figures/neurips_results/`
- `drafts/artifacts/results/`
- `drafts/assets_manifest.json`

## Current Status and Known Gaps

- Core unit tests pass under NumPy 2.x and current PyTorch.
- Sampling-path regressions from `tests/test_vine_pipeline.py` are now covered by passing tests.
- Time-dependent modeling APIs are available, but parts are still in active refinement and should be treated as research code.

## Repository Layout

```text
DVC/
├── src/dvc_package/      # library code
├── tests/                # unit tests
├── examples/             # runnable examples
├── configs/              # YAML experiment configs
├── docs/                 # docs and research notes
├── drafts/               # paper drafts (NeurIPS 2026 draft included)
└── archive/              # TensorFlow legacy baseline
```
