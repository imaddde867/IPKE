#!/usr/bin/env bash
set -euo pipefail
# Run experiment E0: fixed chunking baseline with CHUNK_MAX_CHARS=2000 and prompt P0.

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
OUTPUT_ROOT="${OUTPUT_ROOT:-results/E0_baseline_fixed_p0}"
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
  --max-chars 2000
  --output-root "${OUTPUT_ROOT}"
)

if [[ "${SKIP_FLAG}" == "1" ]]; then
  CMD+=(--skip-existing)
fi

echo "[E0] Running fixed_sweep baseline ..."
"${CMD[@]}"

SOURCE_SUMMARY="${OUTPUT_ROOT}/fixed_sweep_summary.csv"
TARGET_SUMMARY="${OUTPUT_ROOT}/E0_summary.csv"
if [[ -f "${SOURCE_SUMMARY}" && ! -f "${TARGET_SUMMARY}" ]]; then
  cp "${SOURCE_SUMMARY}" "${TARGET_SUMMARY}"
  echo "[E0] Copied ${TARGET_SUMMARY}"
else
  echo "[E0] Summary already exists or source missing; skip copy."
fi

echo "[E0] Baseline complete. Results in ${OUTPUT_ROOT}" 
