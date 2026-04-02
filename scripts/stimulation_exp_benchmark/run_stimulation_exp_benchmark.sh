#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-dataset_stimulation}"
OUT_ROOT="${OUT_ROOT:-dvc_ready}"
RESULTS_ROOT="${RESULTS_ROOT:-results/stimulation_exp_benchmark}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"

print_header() {
  echo "Dalgleish stimulation benchmark run book"
  echo "root: $ROOT_DIR"
  echo "python: $PYTHON_BIN"
  echo "data_root: $DATA_ROOT"
  echo "out_root: $OUT_ROOT"
  echo "results_root: $RESULTS_ROOT"
  echo
}

print_notes() {
  cat <<'EOF'
Notes:
- Activate the intended environment first, for example: `conda activate dvc`
- Step 1 requires the raw Dalgleish session folders under `dataset_stimulation/`
- If the raw dataset is missing, dataset generation and analysis will fail
- The maintained workflow now has:
  - 1 dataset-generation script
  - 1 analysis script
  - 1 figure-refresh script
- Older exploratory and debugging scripts are archived in `scripts/debug_stimulation_exp/`
EOF
  echo
}

print_commands() {
  cat <<EOF
1. Dataset generation
MPLCONFIGDIR=$MPLCONFIGDIR XDG_CACHE_HOME=$XDG_CACHE_HOME \\
$PYTHON_BIN scripts/stimulation_exp_benchmark/build_dalgleish_dvc_dataset.py \\
  --data_root "$DATA_ROOT" \\
  --out_root "$OUT_ROOT" \\
  --d 10 \\
  --seed 0 \\
  --selection_mode topk_responsive

2. Analysis
MPLCONFIGDIR=$MPLCONFIGDIR XDG_CACHE_HOME=$XDG_CACHE_HOME \\
$PYTHON_BIN scripts/stimulation_exp_benchmark/run_dalgleish_latent_publication_analysis.py \\
  --data_root "$DATA_ROOT" \\
  --out_root "$OUT_ROOT" \\
  --results_root "$RESULTS_ROOT" \\
  --family_variant stable \\
  --seed 0 \\
  --n_repeats 2

3. Figure refresh
MPLCONFIGDIR=$MPLCONFIGDIR XDG_CACHE_HOME=$XDG_CACHE_HOME \\
$PYTHON_BIN scripts/stimulation_exp_benchmark/refresh_dalgleish_latent_publication_figures.py \\
  --results_root "$RESULTS_ROOT" \\
  --out_root "$OUT_ROOT"
EOF
  echo
}

run_dataset() {
  MPLCONFIGDIR="$MPLCONFIGDIR" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    "$PYTHON_BIN" scripts/stimulation_exp_benchmark/build_dalgleish_dvc_dataset.py \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --d 10 \
    --seed 0 \
    --selection_mode topk_responsive
}

run_analysis() {
  MPLCONFIGDIR="$MPLCONFIGDIR" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    "$PYTHON_BIN" scripts/stimulation_exp_benchmark/run_dalgleish_latent_publication_analysis.py \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --results_root "$RESULTS_ROOT" \
    --family_variant stable \
    --seed 0 \
    --n_repeats 2
}

run_figures() {
  MPLCONFIGDIR="$MPLCONFIGDIR" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    "$PYTHON_BIN" scripts/stimulation_exp_benchmark/refresh_dalgleish_latent_publication_figures.py \
    --results_root "$RESULTS_ROOT" \
    --out_root "$OUT_ROOT"
}

MODE="${1:-print}"

print_header
print_notes

case "$MODE" in
  print|help|--help|-h)
    print_commands
    ;;
  dataset)
    run_dataset
    ;;
  analysis)
    run_analysis
    ;;
  figures)
    run_figures
    ;;
  all)
    run_dataset
    run_analysis
    run_figures
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Use one of: print, dataset, analysis, figures, all" >&2
    exit 1
    ;;
esac
