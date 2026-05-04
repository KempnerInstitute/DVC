#!/bin/bash
# Activate the local Dynamic Vine Copulas development environment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Activating DVC environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dvc-env

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

echo "DVC environment activated"
echo "Repository: ${ROOT_DIR}"
echo "Python: $(which python)"
echo
echo "Useful commands:"
echo "  python scripts/test_installation.py"
echo "  python -m pytest -q"
echo "  python examples/basic_vine_example.py"
echo "  python scripts/run_experiment.py --list-examples"
