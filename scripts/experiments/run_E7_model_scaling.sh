#!/usr/bin/env bash
set -euo pipefail
# Run experiment E7: compare small vs large LLM backends on SEM_BEST + P3.

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
RESULTS_ROOT="${RESULTS_ROOT:-results/E7_model_scaling}"
SKIP_FLAG="${SKIP_EXISTING:-1}"

# Semantic best chunker placeholders (override after E2/E3 review).
SEM_BEST_LAMBDA="${SEM_BEST_LAMBDA:-TODO_SET_SEM_LAMBDA}"
SEM_BEST_WINDOW="${SEM_BEST_WINDOW:-TODO_SET_SEM_WINDOW}"
SEM_BEST_MIN_SENT="${SEM_BEST_MIN_SENT:-2}"
SEM_BEST_MAX_SENT="${SEM_BEST_MAX_SENT:-40}"

# Model endpoints/names (override via env vars per deployment).
SMALL_MODEL_NAME="${SMALL_MODEL_NAME:-mistral-7b}"
SMALL_MODEL_ENDPOINT="${SMALL_MODEL_ENDPOINT:-TODO_SET_SMALL_ENDPOINT}"
LARGE_MODEL_NAME="${LARGE_MODEL_NAME:-TODO_SET_LARGE_MODEL}"
LARGE_MODEL_ENDPOINT="${LARGE_MODEL_ENDPOINT:-TODO_SET_LARGE_ENDPOINT}"

ensure_value() {
  local name="$1"
  local value="$2"
  if [[ "${value}" == TODO_* ]]; then
    echo "[E7] Please set ${name} before running (edit script or export env var)." >&2
    exit 1
  fi
}

ensure_value "SEM_BEST_LAMBDA" "${SEM_BEST_LAMBDA}"
ensure_value "SEM_BEST_WINDOW" "${SEM_BEST_WINDOW}"
ensure_value "SMALL_MODEL_ENDPOINT" "${SMALL_MODEL_ENDPOINT}"
ensure_value "LARGE_MODEL_NAME" "${LARGE_MODEL_NAME}"
ensure_value "LARGE_MODEL_ENDPOINT" "${LARGE_MODEL_ENDPOINT}"

DOCS=(
  "datasets/archive/test_data/text/3m_marine_oem_sop.txt"
  "datasets/archive/test_data/text/DOA_Food_Man_Proc_Stor.txt"
  "datasets/archive/test_data/text/op_firesafety_guideline.txt"
)

mkdir -p "${RESULTS_ROOT}"
RUN_DIRS=()

run_model_case() {
  local label="$1"
  local model_name="$2"
  local backend_url="$3"
  local run_dir="${RESULTS_ROOT}/${label}"
  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" "scripts/experiments/run_chunker_eval.py"
    --config-id "${label}"
    --method "breakpoint_semantic"
    --host "${HOST}"
    --port "${SEMANTIC_PORT}"
    --service-name "ipke-semantic"
    --container-name "ipke-semantic-chunking"
    --run-dir "${run_dir}"
    --prompt-mode "P3"
    --summary-param "experiment=E7"
    --summary-param "config=${label}"
    --summary-param "prompt=P3"
    --summary-param "chunker=semantic"
    --summary-param "model=${label}"
    --env "CHUNK_MAX_CHARS=2000"
    --env "SEM_LAMBDA=${SEM_BEST_LAMBDA}"
    --env "SEM_WINDOW_W=${SEM_BEST_WINDOW}"
    --env "SEM_MIN_SENTENCES_PER_CHUNK=${SEM_BEST_MIN_SENT}"
    --env "SEM_MAX_SENTENCES_PER_CHUNK=${SEM_BEST_MAX_SENT}"
    --env "LLM_MODEL_NAME=${model_name}"
    --env "LLM_BACKEND_URL=${backend_url}"
  )

  cmd+=(--documents)
  for doc in "${DOCS[@]}"; do
    cmd+=("${doc}")
  done

  if [[ "${SKIP_FLAG}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi

  echo "[E7] Running ${label} (${model_name}) ..."
  "${cmd[@]}"
  RUN_DIRS+=("${run_dir}")
}

run_model_case "small_model" "${SMALL_MODEL_NAME}" "${SMALL_MODEL_ENDPOINT}"
run_model_case "large_model" "${LARGE_MODEL_NAME}" "${LARGE_MODEL_ENDPOINT}"

SUMMARY_PATH="${RESULTS_ROOT}/E7_model_scaling_summary.csv"
echo "[E7] Aggregating summary rows -> ${SUMMARY_PATH}"
"${PYTHON_BIN}" "scripts/experiments/build_summary_csv.py" \
  --run-dirs "${RUN_DIRS[@]}" \
  --output "${SUMMARY_PATH}" \
  --extra-column "experiment=E7"

echo "[E7] Model scaling comparison complete. Results in ${RESULTS_ROOT}"
