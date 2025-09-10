#!/bin/bash
# Activation script for DVC environment

echo "Activating DVC environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dvc-env

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "DVC environment activated!"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Display usage information
echo ""
echo "Ready to run experiments!"
echo ""
echo "Usage examples:"
echo "  python scripts/run_experiment.py --create-examples"
echo "  python scripts/run_experiment.py configs/probability_analysis.yaml"
echo "  python scripts/run_experiment.py configs/entropy_analysis.yaml"
echo "  python scripts/run_experiment.py configs/time_dependent.yaml"
echo ""
echo "For help: python scripts/run_experiment.py --help"
