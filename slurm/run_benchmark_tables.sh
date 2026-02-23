#!/bin/bash
#SBATCH --job-name=dvc_benchmark_tables
#SBATCH --partition=kempner
#SBATCH --account=kempner_dev
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/n/holylabs/kempner_dev/Users/hsafaai/Code/DVC/logs/dvc_benchmark_tables_%j.out
#SBATCH --error=/n/holylabs/kempner_dev/Users/hsafaai/Code/DVC/logs/dvc_benchmark_tables_%j.err

set -euo pipefail

echo "=========================================="
echo "DVC Benchmark Table Generation"
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

# List of configs to run
CONFIGS=(
    "configs/probability_analysis.yaml"
    "configs/entropy_analysis.yaml"
    "configs/time_dependent.yaml"
    "configs/simulation_benchmarks.yaml"
)

TOTAL=${#CONFIGS[@]}
FAILED=0
SUCCEEDED=0

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    STEP=$(( i + 1 ))

    echo ""
    echo "=========================================="
    echo "[${STEP}/${TOTAL}] Running config: ${CONFIG}"
    echo "Started at $(date)"
    echo "=========================================="

    if ${PYTHON} scripts/generate_benchmark_tables.py --run --configs "${CONFIG}"; then
        echo "[${STEP}/${TOTAL}] SUCCEEDED: ${CONFIG}"
        SUCCEEDED=$(( SUCCEEDED + 1 ))
    else
        echo "[${STEP}/${TOTAL}] FAILED: ${CONFIG}"
        FAILED=$(( FAILED + 1 ))
    fi

    echo "[${STEP}/${TOTAL}] Completed at $(date)"
done

echo ""
echo "=========================================="
echo "Summary: ${SUCCEEDED}/${TOTAL} configs succeeded, ${FAILED}/${TOTAL} failed"
echo "Job finished at $(date)"
echo "=========================================="

if [ "${FAILED}" -gt 0 ]; then
    echo "WARNING: ${FAILED} config(s) failed. Check logs for details."
    exit 1
fi
