#!/usr/bin/env bash
set -euo pipefail
# Run experiment E1: sweep CHUNK_MAX_CHARS (and optional strides) for the fixed chunker.

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
FIXED_PORT="${FIXED_PORT:-8000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/E1_fixed_sweep}"
SKIP_FLAG="${SKIP_EXISTING:-1}"

DOCS=(
  "datasets/archive/test_data/text/3m_marine_oem_sop.txt"
  "datasets/archive/test_data/text/DOA_Food_Man_Proc_Stor.txt"
  "datasets/archive/test_data/text/op_firesafety_guideline.txt"
)

mkdir -p "${OUTPUT_ROOT}"

CMD=(
  "${PYTHON_BIN}" "scripts/experiments/fixed_sweep.py"
  --host "${HOST}"
  --port "${FIXED_PORT}"
  --documents "${DOCS[@]}"
  --max-chars 1000 1500 2000 3000
  --stride-values 0 250 500
  --output-root "${OUTPUT_ROOT}"
)

if [[ "${SKIP_FLAG}" == "1" ]]; then
  CMD+=(--skip-existing)
fi

echo "[E1] Running fixed chunk-size sweep ..."
"${CMD[@]}"

echo "[E1] Sweep complete. Summary: ${OUTPUT_ROOT}/fixed_sweep_summary.csv"
