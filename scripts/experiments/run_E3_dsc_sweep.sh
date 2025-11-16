#!/usr/bin/env bash
set -euo pipefail
# Run experiment E3: sweep Dual Semantic Chunker hyperparameters.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -d ".venv311" ]]; then
  source ".venv311/bin/activate"
elif [[ -d ".venv" ]]; then
  source ".venv/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
HOST="${HOST:-http://localhost}"
DSC_PORT="${DSC_PORT:-8002}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/E3_dsc_sweep}"
SKIP_FLAG="${SKIP_EXISTING:-1}"

DOCS=(
  "datasets/archive/test_data/text/3m_marine_oem_sop.txt"
  "datasets/archive/test_data/text/DOA_Food_Man_Proc_Stor.txt"
  "datasets/archive/test_data/text/op_firesafety_guideline.txt"
)

mkdir -p "${OUTPUT_ROOT}"

CMD=(
  "${PYTHON_BIN}" "scripts/experiments/dsc_sweep.py"
  --host "${HOST}"
  --port "${DSC_PORT}"
  --documents "${DOCS[@]}"
  --min-parent-values 5 10 15
  --max-parent-values 80 120 160
  --delta-window-values 15 25 35
  --threshold-values 0.8 1.0 1.2
  --heading-options true false
  --output-root "${OUTPUT_ROOT}"
)

if [[ "${SKIP_FLAG}" == "1" ]]; then
  CMD+=(--skip-existing)
fi

echo "[E3] Running DSC sweep ..."
"${CMD[@]}"

echo "[E3] Sweep complete. Summary: ${OUTPUT_ROOT}/dsc_sweep_summary.csv"
