# Setup Guide

## Prerequisites

- Python 3.8+
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
pip install -r requirements.txt
pip install -e .
```

## Verify Installation

```bash
pytest -q
python scripts/test_installation.py
```

## Quick Experiment Smoke Test

```bash
python scripts/run_experiment.py --list-examples
python scripts/run_experiment.py configs/probability_analysis.yaml
```

## GPU Notes

- DVC uses PyTorch device selection at runtime.
- Install a CUDA-enabled PyTorch build if you want GPU acceleration.
- Validate with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

