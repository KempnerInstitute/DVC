#!/bin/bash
# Installation script for DVC package

set -e  # Exit on any error

echo "Installing DVC (Distributed Vine Copula) Package"
echo "=================================================="

# Check if we're in the correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ required, found Python $python_version"
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
echo "1. Check out the examples in the examples/ directory"
echo "2. Try the configuration files in configs/"
echo "3. Read the documentation in docs/"
echo "4. Run experiments with the provided YAML configs"
echo ""
echo "Example usage:"
echo "  # Fit a vine copula to data"
echo "  dvc-fit --data examples/sample_data.csv --vine-type r-vine --optimize --output model.pkl"
echo ""
echo "  # Run a benchmark experiment"
echo "  dvc-experiment --config configs/experiments/basic_vine_comparison.yaml --output-dir results/"
echo ""
echo "Happy modeling!"
