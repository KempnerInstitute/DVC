#!/bin/bash
#SBATCH --job-name=dvc_benchmarks
#SBATCH --partition=kempner
#SBATCH --account=kempner_dev
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/n/holylabs/kempner_dev/Users/hsafaai/Code/DVC/logs/dvc_benchmarks_%j.out
#SBATCH --error=/n/holylabs/kempner_dev/Users/hsafaai/Code/DVC/logs/dvc_benchmarks_%j.err

set -euo pipefail

echo "=========================================="
echo "DVC Benchmark Suite"
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

PROJECT_DIR=/n/holylabs/kempner_dev/Users/hsafaai/Code/DVC
PYTHON="${PROJECT_DIR}/.venv/bin/python"

cd "${PROJECT_DIR}"
export PYTHONPATH=src:${PYTHONPATH:-}

# Verify python environment
echo "Python: ${PYTHON}"
${PYTHON} --version

# ---- Step 1: Run generate_benchmark_tables.py --run (executes all 4 configs) ----
echo ""
echo "=========================================="
echo "[Step 1/2] Running generate_benchmark_tables.py --run"
echo "Started at $(date)"
echo "=========================================="

${PYTHON} scripts/generate_benchmark_tables.py --run

echo "[Step 1/2] Completed at $(date)"

# ---- Step 2: Run prepare_draft_assets.py ----
echo ""
echo "=========================================="
echo "[Step 2/2] Running prepare_draft_assets.py"
echo "Started at $(date)"
echo "=========================================="

# Note: do NOT pass --run here since tables were already generated above.
# This step copies tables, figures, and result JSONs into drafts/.
${PYTHON} scripts/prepare_draft_assets.py

echo "[Step 2/2] Completed at $(date)"

echo ""
echo "=========================================="
echo "DVC Benchmark Suite finished at $(date)"
echo "=========================================="
