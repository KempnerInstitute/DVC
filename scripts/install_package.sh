#!/bin/bash
# Installation script for DVC package

set -e  # Exit on any error

echo "Installing Dynamic Vine Copulas"
echo "=================================================="

# Check if we're in the correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10+ required, found Python $python_version"
    exit 1
fi

echo "Python version: $python_version"

# Check if poetry is available
if command -v poetry &> /dev/null; then
    echo "Poetry found, using Poetry for installation"
    
    # Install with poetry
    echo "Installing dependencies with Poetry..."
    poetry install
    
    echo "Running tests..."
    poetry run pytest tests/ -v
    
    echo "Installation completed successfully!"
    echo ""
    echo "To activate the environment:"
    echo "  poetry shell"
    echo ""
    echo "To run CLI commands:"
    echo "  poetry run dvc-fit --help"
    echo "  poetry run dvc-entropy --help"
    echo "  poetry run dvc-time --help"
    echo "  poetry run dvc-experiment --help"
    
elif command -v pip &> /dev/null; then
    echo "Poetry not found, using pip for installation"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    # Install package in development mode
    echo "Installing package in development mode..."
    pip install -e .
    
    # Install development dependencies
    echo "Installing development dependencies..."
    pip install pytest pytest-cov black isort flake8 mypy
    
    echo "Running tests..."
    python -m pytest tests/ -v
    
    echo "Installation completed successfully!"
    echo ""
    echo "To activate the environment:"
    echo "  source venv/bin/activate"
    echo ""
    echo "CLI commands are available as:"
    echo "  dvc-fit --help"
    echo "  dvc-entropy --help"
    echo "  dvc-time --help"
    echo "  dvc-experiment --help"
    
else
    echo "Error: Neither Poetry nor pip found. Please install Poetry or ensure pip is available."
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Browse the runnable examples in examples/"
echo "2. Read docs/setup.md and docs/user-guide/ for end-to-end usage"
echo "3. Materialize the bundled YAML configs:"
echo "     python scripts/run_experiment.py --create-examples"
echo "4. Run an example experiment:"
echo "     python scripts/run_experiment.py configs/probability_analysis.yaml"
echo ""
echo "Real-world finance benchmark uses a separate runner:"
echo "  python scripts/run_finance_crisis_benchmark.py --config configs/finance_crisis_benchmarks.yaml"
