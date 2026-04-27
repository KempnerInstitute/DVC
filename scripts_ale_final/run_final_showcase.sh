#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-parametric}"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

conda run -n dvc python scripts_ale_final/run_final_showcase_benchmark.py \
  --mode "$MODE" \
  "$@"
