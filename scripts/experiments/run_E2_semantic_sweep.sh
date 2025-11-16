#!/usr/bin/env bash
set -euo pipefail
# Run experiment E2: sweep semantic chunker hyperparameters (EBBC-style).

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
SEMANTIC_PORT="${SEMANTIC_PORT:-8001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/E2_semantic_sweep}"
SKIP_FLAG="${SKIP_EXISTING:-1}"

DOCS=(
  "datasets/archive/test_data/text/3m_marine_oem_sop.txt"
  "datasets/archive/test_data/text/DOA_Food_Man_Proc_Stor.txt"
  "datasets/archive/test_data/text/op_firesafety_guideline.txt"
)

mkdir -p "${OUTPUT_ROOT}"

CMD=(
  "${PYTHON_BIN}" "scripts/experiments/semantic_sweep.py"
  --host "${HOST}"
  --port "${SEMANTIC_PORT}"
  --documents "${DOCS[@]}"
  --chunk-max-chars 2000
  --lambda-values 0.05 0.15 0.25
  --window-values 20 30 40
  --min-sent-values 2
  --max-sent-values 40
  --output-root "${OUTPUT_ROOT}"
)

if [[ "${SKIP_FLAG}" == "1" ]]; then
  CMD+=(--skip-existing)
fi

echo "[E2] Running semantic chunker sweep ..."
"${CMD[@]}"

echo "[E2] Sweep complete. Summary: ${OUTPUT_ROOT}/semantic_sweep_summary.csv"
