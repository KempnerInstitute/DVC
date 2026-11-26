# Setup Guide

## Prerequisites

- Python 3.10 or 3.11
- `pip` (or Conda/Mamba)

## Option A: Conda

```bash
conda env create -f environment.yml
conda activate dvc-env
pip install -e .
```

## Option B: venv

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Verify Installation

```bash
python scripts/test_installation.py
python -m pip install pytest
python -m pytest -q tests
```

## Quick Experiment Smoke Test

```bash
python scripts/run_experiment.py --create-examples
python scripts/run_experiment.py configs/probability_analysis.yaml
```

For the real-world finance benchmark (separate runner):

```bash
python scripts/run_finance_crisis_benchmark.py --config configs/finance_crisis_benchmarks.yaml
```

## GPU Notes

- DVC uses PyTorch device selection at runtime.
- Install a CUDA-enabled PyTorch build if you want GPU acceleration.
- Validate with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

For the paper figures, use the pinned scientific Python environment in the [vine modulation guide](user-guide/vine-modulation.md).
