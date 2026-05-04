#!/bin/bash
# Create a local conda environment for Dynamic Vine Copulas.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${DVC_ENV_NAME:-dvc-env}"

echo "Setting up Dynamic Vine Copulas"
echo "Repository: ${ROOT_DIR}"
echo "Environment: ${ENV_NAME}"
echo

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available. Install Miniconda or Anaconda first."
  exit 1
fi

cd "${ROOT_DIR}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment ${ENV_NAME} already exists."
  echo "Remove it manually with 'conda env remove -n ${ENV_NAME}' if you want a fresh rebuild."
else
  conda env create -f environment.yml -n "${ENV_NAME}"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
pip install -e .

python scripts/test_installation.py

echo
echo "Setup complete."
echo "Activate later with:"
echo "  source activate_dvc.sh"
echo
echo "Suggested smoke tests:"
echo "  python examples/basic_vine_example.py"
echo "  python scripts/run_dynamic_cvine_example.py --output-dir results/dynamic_cvine_example"
