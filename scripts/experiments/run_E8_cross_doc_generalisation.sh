#!/usr/bin/env bash
set -euo pipefail
# Run experiment E8: evaluate SEM_BEST + P3 on each document independently.

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
RESULTS_ROOT="${RESULTS_ROOT:-results/E8_cross_doc}"
SKIP_FLAG="${SKIP_EXISTING:-1}"

# Semantic best placeholders (override from sweep results).
SEM_BEST_LAMBDA="${SEM_BEST_LAMBDA:-TODO_SET_SEM_LAMBDA}"
SEM_BEST_WINDOW="${SEM_BEST_WINDOW:-TODO_SET_SEM_WINDOW}"
SEM_BEST_MIN_SENT="${SEM_BEST_MIN_SENT:-2}"
SEM_BEST_MAX_SENT="${SEM_BEST_MAX_SENT:-40}"

ensure_value() {
  local name="$1"
  local value="$2"
  if [[ "${value}" == TODO_* ]]; then
    echo "[E8] Please set ${name} before running (edit script or export env var)." >&2
    exit 1
  fi
}

ensure_value "SEM_BEST_LAMBDA" "${SEM_BEST_LAMBDA}"
ensure_value "SEM_BEST_WINDOW" "${SEM_BEST_WINDOW}"

DOC_KEYS=("doc_3m" "doc_doa_food" "doc_firesafety")
DOC_PATHS=(
  "datasets/archive/test_data/text/3m_marine_oem_sop.txt"
  "datasets/archive/test_data/text/DOA_Food_Man_Proc_Stor.txt"
  "datasets/archive/test_data/text/op_firesafety_guideline.txt"
)

mkdir -p "${RESULTS_ROOT}"
RUN_DIRS=()

for idx in "${!DOC_KEYS[@]}"; do
  key="${DOC_KEYS[$idx]}"
  doc_path="${DOC_PATHS[$idx]}"
  run_dir="${RESULTS_ROOT}/${key}"
  mkdir -p "${run_dir}"

  cmd=(
    "${PYTHON_BIN}" "scripts/experiments/run_chunker_eval.py"
    --config-id "${key}"
    --method "breakpoint_semantic"
    --host "${HOST}"
    --port "${SEMANTIC_PORT}"
    --service-name "ipke-semantic"
    --container-name "ipke-semantic-chunking"
    --run-dir "${run_dir}"
    --prompt-mode "P3"
    --summary-param "experiment=E8"
    --summary-param "config=${key}"
    --summary-param "chunker=semantic"
    --summary-param "prompt=P3"
    --summary-param "doc_key=${key}"
    --env "CHUNK_MAX_CHARS=2000"
    --env "SEM_LAMBDA=${SEM_BEST_LAMBDA}"
    --env "SEM_WINDOW_W=${SEM_BEST_WINDOW}"
    --env "SEM_MIN_SENTENCES_PER_CHUNK=${SEM_BEST_MIN_SENT}"
    --env "SEM_MAX_SENTENCES_PER_CHUNK=${SEM_BEST_MAX_SENT}"
  )

  cmd+=(--documents "${doc_path}")

  if [[ "${SKIP_FLAG}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi

  echo "[E8] Running SEM_BEST on ${key} (${doc_path}) ..."
  "${cmd[@]}"
  RUN_DIRS+=("${run_dir}")
done

SUMMARY_PATH="${RESULTS_ROOT}/E8_cross_doc_summary.csv"
echo "[E8] Aggregating summary rows -> ${SUMMARY_PATH}"
"${PYTHON_BIN}" "scripts/experiments/build_summary_csv.py" \
  --run-dirs "${RUN_DIRS[@]}" \
  --output "${SUMMARY_PATH}" \
  --extra-column "experiment=E8"

echo "[E8] Cross-document evaluation finished. Results in ${RESULTS_ROOT}"
