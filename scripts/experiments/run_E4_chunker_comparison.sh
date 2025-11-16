#!/usr/bin/env bash
set -euo pipefail
# Run experiment E4: compare best Fixed, Semantic, and DSC chunkers under prompt P0.

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
SEMANTIC_PORT="${SEMANTIC_PORT:-8001}"
DSC_PORT="${DSC_PORT:-8002}"
RESULTS_ROOT="${RESULTS_ROOT:-results/E4_chunker_comparison}"
SKIP_FLAG="${SKIP_EXISTING:-1}"

# TODO: fill in best hyperparameters after inspecting E1-E3 summaries (overridable via env vars).
FIXED_BEST_MAX_CHARS="${FIXED_BEST_MAX_CHARS:-TODO_SET_FIXED_MAX}"
FIXED_BEST_STRIDE="${FIXED_BEST_STRIDE:-0}"

SEM_BEST_LAMBDA="${SEM_BEST_LAMBDA:-TODO_SET_SEM_LAMBDA}"
SEM_BEST_WINDOW="${SEM_BEST_WINDOW:-TODO_SET_SEM_WINDOW}"
SEM_BEST_MIN_SENT="${SEM_BEST_MIN_SENT:-2}"
SEM_BEST_MAX_SENT="${SEM_BEST_MAX_SENT:-40}"

DSC_BEST_PARENT_MIN="${DSC_BEST_PARENT_MIN:-TODO_SET_DSC_PARENT_MIN}"
DSC_BEST_PARENT_MAX="${DSC_BEST_PARENT_MAX:-TODO_SET_DSC_PARENT_MAX}"
DSC_BEST_DELTA_WINDOW="${DSC_BEST_DELTA_WINDOW:-TODO_SET_DSC_DELTA_WINDOW}"
DSC_BEST_THRESHOLD_K="${DSC_BEST_THRESHOLD_K:-TODO_SET_DSC_THRESHOLD}"
DSC_BEST_USE_HEADINGS="${DSC_BEST_USE_HEADINGS:-TODO_SET_TRUE_FALSE}"

DOCS=(
  "datasets/archive/test_data/text/3m_marine_oem_sop.txt"
  "datasets/archive/test_data/text/DOA_Food_Man_Proc_Stor.txt"
  "datasets/archive/test_data/text/op_firesafety_guideline.txt"
)

ensure_value() {
  local name="$1"
  local value="$2"
  if [[ "${value}" == TODO_* ]]; then
    echo "[E4] Please set ${name} before running (edit script or export env var)." >&2
    exit 1
  fi
}

ensure_value "FIXED_BEST_MAX_CHARS" "${FIXED_BEST_MAX_CHARS}"
ensure_value "SEM_BEST_LAMBDA" "${SEM_BEST_LAMBDA}"
ensure_value "SEM_BEST_WINDOW" "${SEM_BEST_WINDOW}"
ensure_value "DSC_BEST_PARENT_MIN" "${DSC_BEST_PARENT_MIN}"
ensure_value "DSC_BEST_PARENT_MAX" "${DSC_BEST_PARENT_MAX}"
ensure_value "DSC_BEST_DELTA_WINDOW" "${DSC_BEST_DELTA_WINDOW}"
ensure_value "DSC_BEST_THRESHOLD_K" "${DSC_BEST_THRESHOLD_K}"
ensure_value "DSC_BEST_USE_HEADINGS" "${DSC_BEST_USE_HEADINGS}"

mkdir -p "${RESULTS_ROOT}"

RUN_DIRS=()

run_eval() {
  local config_id="$1"
  local method="$2"
  local service="$3"
  local container="$4"
  local port="$5"
  local run_subdir="$6"
  shift 6
  local run_dir="${RESULTS_ROOT}/${run_subdir}"
  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" "scripts/experiments/run_chunker_eval.py"
    --config-id "${config_id}"
    --method "${method}"
    --host "${HOST}"
    --port "${port}"
    --service-name "${service}"
    --container-name "${container}"
    --run-dir "${run_dir}"
    --summary-param "experiment=E4"
    --summary-param "config=${config_id}"
  )

  cmd+=(--documents)
  for doc in "${DOCS[@]}"; do
    cmd+=("${doc}")
  done

  if [[ "${SKIP_FLAG}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi

  cmd+=("$@")

  echo "[E4] Running ${config_id} ..."
  "${cmd[@]}"
  RUN_DIRS+=("${run_dir}")
}

# Fixed best @ P0
FIXED_ARGS=(--env "CHUNK_MAX_CHARS=${FIXED_BEST_MAX_CHARS}" --summary-param "chunker=fixed" --summary-param "prompt=P0")
if [[ -n "${FIXED_BEST_STRIDE}" && "${FIXED_BEST_STRIDE}" != "0" ]]; then
  FIXED_ARGS+=(--env "CHUNK_STRIDE_CHARS=${FIXED_BEST_STRIDE}")
fi
run_eval \
  "fixed_best_p0" "fixed" "ipke-fixed" "ipke-fixed-chunking" "${FIXED_PORT}" "fixed_best_p0" \
  --prompt-mode "P0" \
  "${FIXED_ARGS[@]}"

# Semantic best @ P0
run_eval \
  "semantic_best_p0" "breakpoint_semantic" "ipke-semantic" "ipke-semantic-chunking" "${SEMANTIC_PORT}" "semantic_best_p0" \
  --prompt-mode "P0" \
  --env "CHUNK_MAX_CHARS=2000" \
  --env "SEM_LAMBDA=${SEM_BEST_LAMBDA}" \
  --env "SEM_WINDOW_W=${SEM_BEST_WINDOW}" \
  --env "SEM_MIN_SENTENCES_PER_CHUNK=${SEM_BEST_MIN_SENT}" \
  --env "SEM_MAX_SENTENCES_PER_CHUNK=${SEM_BEST_MAX_SENT}" \
  --summary-param "chunker=semantic" \
  --summary-param "prompt=P0"

# DSC best @ P0
run_eval \
  "dsc_best_p0" "dsc" "ipke-dsc" "ipke-dsc-chunking" "${DSC_PORT}" "dsc_best_p0" \
  --prompt-mode "P0" \
  --env "DSC_PARENT_MIN_SENTENCES=${DSC_BEST_PARENT_MIN}" \
  --env "DSC_PARENT_MAX_SENTENCES=${DSC_BEST_PARENT_MAX}" \
  --env "DSC_DELTA_WINDOW=${DSC_BEST_DELTA_WINDOW}" \
  --env "DSC_THRESHOLD_K=${DSC_BEST_THRESHOLD_K}" \
  --env "DSC_USE_HEADINGS=${DSC_BEST_USE_HEADINGS}" \
  --summary-param "chunker=dsc" \
  --summary-param "prompt=P0"

SUMMARY_PATH="${RESULTS_ROOT}/E4_chunker_comparison_summary.csv"
echo "[E4] Aggregating summary rows -> ${SUMMARY_PATH}"
"${PYTHON_BIN}" "scripts/experiments/build_summary_csv.py" \
  --run-dirs "${RUN_DIRS[@]}" \
  --output "${SUMMARY_PATH}" \
  --extra-column "experiment=E4"

echo "[E4] Done. Results stored under ${RESULTS_ROOT}"
