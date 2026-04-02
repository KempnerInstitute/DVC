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
python -m pytest -q

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
python scripts/run_finance_crisis_benchmark.py --config configs/finance_crisis_benchmarks.yaml
```

## Generate Benchmark Tables

Run benchmark configs and generate CSV/LaTeX tables for the paper:

```bash
python scripts/generate_benchmark_tables.py --run
```

Outputs are written under `results/benchmark_tables/`:
- `master_summary.csv` / `.tex`
- `probability_vine_detail.csv` / `.tex`
- `entropy_method_detail.csv` / `.tex`
- `time_pair_detail.csv` / `.tex`
- `simulation_benchmark_detail.csv` / `.tex` (from `configs/simulation_benchmarks.yaml`)
- `finance_crisis_detail.csv` / `.tex` (when finance crisis results are included)

## Dalgleish Latent-State Analysis

The repository also contains a real-data analysis pipeline for the Dalgleish et al. photostimulation dataset under `dataset_stimulation/`.

Environment used locally:

```bash
conda activate dvc
```

Run the validated latent-state benchmark:

```bash
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp \
python scripts/stimulation_exp_benchmark/build_dalgleish_dvc_dataset.py \
  --data_root dataset_stimulation \
  --out_root dvc_ready \
  --d 10 \
  --seed 0 \
  --selection_mode topk_responsive
```

Run the maintained latent analysis script:

```bash
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp \
python scripts/stimulation_exp_benchmark/run_dalgleish_latent_publication_analysis.py \
  --data_root dataset_stimulation \
  --out_root dvc_ready \
  --results_root results/stimulation_exp_benchmark \
  --family_variant stable \
  --seed 0 \
  --n_repeats 2
```

Redraw the publication-facing figures from the validated latent publication summaries:

```bash
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp \
python scripts/stimulation_exp_benchmark/refresh_dalgleish_latent_publication_figures.py \
  --results_root results/stimulation_exp_benchmark \
  --out_root dvc_ready
```

Compact run book:

```bash
bash scripts/stimulation_exp_benchmark/run_stimulation_exp_benchmark.sh
```

The maintained workflow now uses:

- `scripts/stimulation_exp_benchmark/build_dalgleish_dvc_dataset.py`
- `scripts/stimulation_exp_benchmark/run_dalgleish_latent_publication_analysis.py`
- `scripts/stimulation_exp_benchmark/refresh_dalgleish_latent_publication_figures.py`

Older exploratory or intermediate Dalgleish scripts are archived under:

- `scripts/debug_stimulation_exp/`

Main outputs are written to:

- `results/stimulation_exp_benchmark/data/`
- `results/stimulation_exp_benchmark/plots/`
- mirrored key CSVs in `dvc_ready/`

Key latent-state outputs:

- `latent_state_source_space_summary.csv`
  Source-space comparison for targeted, mixed, and non-targeted latent spaces.
- `latent_state_dose_summary.csv`
  Dose-conditioned summaries for the chosen latent formulation.
- `latent_state_dynamic_summary.csv`
  Exploratory within-session dynamic summaries.
- `latent_state_interpretability.csv`
  PCA variance, stability, temporal balance, and target-related summaries.
- `latent_followup_static_summary.csv`
  Session-level and pooled static source-space summaries for the publication follow-up.
- `latent_followup_dose_summary.csv`
  Session-level and pooled by-dose summaries with bootstrap-ready aggregates.
- `latent_followup_dynamic_summary.csv`
  Rolling-window dynamic summaries for both within-window PCA and common-basis sensitivity views.
- `latent_followup_family_summary.csv`
  Raw stable-family and grouped family-usage summaries.
- `latent_followup_pc_summary.csv`
  PC profile summaries and interpretability associations.
- `latent_followup_stats_summary.csv`
  Session-level inference summaries for the main comparisons.
- `latent_publication_static_summary.csv`
  Final publication-pass static summary, including source-space sessions, baseline-vs-full comparisons, and Gaussian-to-1-trunc vs 1-trunc-to-full decomposition.
- `latent_publication_control_summary.csv`
  Reduced-rank catch/control feasibility screen and any control-session summaries if catch is clean enough.
- `latent_publication_family_summary.csv`
  Publication-facing family usage summaries, including raw stable families and grouped dependence classes.
- `latent_publication_dynamic_summary.csv`
  Early/middle/late dynamic summaries for both blockwise-basis and common-basis latent views.
- `latent_publication_pc_summary.csv`
  Final PCA variance, stability, temporal balance, and target-related interpretability summaries.
- `latent_publication_stats_summary.csv`
  Session-level bootstrap/sign-flip summaries for Panels A-D.
- `latent_publication_baseline_feasibility.csv`
  Explicit record of which repository baselines ran cleanly in the latent-state publication pass and which did not.
- `latent_publication_metadata.json`
  Final figure-panel choice, control feasibility, and paper-decision metadata.

Main figure files:

- `fig_latent_publication_final.png`
  Final publication-style figure with Panel A baseline comparisons, Panel B pairwise-versus-higher-order decomposition, Panel C source-space biology, and Panel D grouped dependence type.
- `fig_latent_publication_dose_supplement.png`
  Supplement figure showing dose robustness for the main non-targeted latent variant.
- `fig_latent_publication_family_supplement.png`
  Supplement figure with grouped dependence classes by dose and raw stable-family usage by source space.
- `fig_latent_publication_dynamic_supplement.png`
  Earlier exploratory dynamic/history figure retained for reference, but not part of the refreshed main figure set.

Overview markdown:

- `scripts/stimulation_exp_benchmark/DALGLEISH_LATENT_ANALYSIS_OVERVIEW.md`
  Reader-facing explanation of the dataset, preprocessing, latent-state formulation, hypotheses, main findings, limits, and rerun commands.

Figure/source mapping:

- `latent_publication_figure_panel_map.json`
  Small manifest that records which source tables feed each refreshed publication figure panel.

## Prepare Standalone Draft Assets

To generate benchmark artifacts and vendor all paper assets into `drafts/`
(tables, figures, and result JSON summaries), run:

```bash
python scripts/prepare_draft_assets.py --run --compile
```

This writes to:
- `drafts/tables/benchmark_tables/`
- `drafts/figures/benchmark_results/`
- `drafts/artifacts/results/`
- `drafts/assets_manifest.json`

## Current Status and Known Gaps

- Core unit tests pass under NumPy 2.x and current PyTorch.
- Sampling-path regressions from `tests/test_vine_pipeline.py` are now covered by passing tests.
- Time-dependent modeling APIs are available, but parts are still in active refinement and should be treated as research code.

## Documentation Pointers

- Docs index: `docs/index.md`
- Time-dependent implementation status: `docs/user-guide/time-dependent.md`
- Comparable methods and benchmark extensions: `docs/research/comparable_methods.md`

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
