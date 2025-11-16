#!/usr/bin/env bash
set -euo pipefail
# Run experiment E9: produce paired outputs for human evaluation (Fixed-BEST P0 vs SEM_BEST P3).

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
RESULTS_ROOT="${RESULTS_ROOT:-results/E9_human_eval}"
SKIP_FLAG="${SKIP_EXISTING:-1}"
GOLD_DIR="${GOLD_DIR:-datasets/archive/gold_human}"

# Placeholder hyperparameters (override via env var or edit).
FIXED_BEST_MAX_CHARS="${FIXED_BEST_MAX_CHARS:-TODO_SET_FIXED_MAX}"
FIXED_BEST_STRIDE="${FIXED_BEST_STRIDE:-0}"
SEM_BEST_LAMBDA="${SEM_BEST_LAMBDA:-TODO_SET_SEM_LAMBDA}"
SEM_BEST_WINDOW="${SEM_BEST_WINDOW:-TODO_SET_SEM_WINDOW}"
SEM_BEST_MIN_SENT="${SEM_BEST_MIN_SENT:-2}"
SEM_BEST_MAX_SENT="${SEM_BEST_MAX_SENT:-40}"

ensure_value() {
  local name="$1"
  local value="$2"
  if [[ "${value}" == TODO_* ]]; then
    echo "[E9] Please set ${name} before running (edit script or export env var)." >&2
    exit 1
  fi
}

ensure_value "FIXED_BEST_MAX_CHARS" "${FIXED_BEST_MAX_CHARS}"
ensure_value "SEM_BEST_LAMBDA" "${SEM_BEST_LAMBDA}"
ensure_value "SEM_BEST_WINDOW" "${SEM_BEST_WINDOW}"

DOCS=(
  "datasets/archive/test_data/text/3m_marine_oem_sop.txt"
  "datasets/archive/test_data/text/DOA_Food_Man_Proc_Stor.txt"
  "datasets/archive/test_data/text/op_firesafety_guideline.txt"
)

mkdir -p "${RESULTS_ROOT}"
RUN_DIRS=()

run_case() {
  local label="$1"
  local method="$2"
  local service="$3"
  local container="$4"
  local port="$5"
  local prompt="$6"
  shift 6
  local run_dir="${RESULTS_ROOT}/${label}"
  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" "scripts/experiments/run_chunker_eval.py"
    --config-id "${label}"
    --method "${method}"
    --host "${HOST}"
    --port "${port}"
    --service-name "${service}"
    --container-name "${container}"
    --run-dir "${run_dir}"
    --prompt-mode "${prompt}"
    --summary-param "experiment=E9"
    --summary-param "config=${label}"
    --summary-param "prompt=${prompt}"
    --summary-param "chunker=${label%%_*}"
  )

  cmd+=(--documents)
  for doc in "${DOCS[@]}"; do
    cmd+=("${doc}")
  done

  if [[ "${SKIP_FLAG}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi

  cmd+=("$@")

  echo "[E9] Running ${label} ..."
  "${cmd[@]}"
  RUN_DIRS+=("${run_dir}")
}

FIXED_ARGS=(--env "CHUNK_MAX_CHARS=${FIXED_BEST_MAX_CHARS}")
if [[ -n "${FIXED_BEST_STRIDE}" && "${FIXED_BEST_STRIDE}" != "0" ]]; then
  FIXED_ARGS+=(--env "CHUNK_STRIDE_CHARS=${FIXED_BEST_STRIDE}")
fi
run_case \
  "fixed_best_p0" "fixed" "ipke-fixed" "ipke-fixed-chunking" "${FIXED_PORT}" "P0" \
  --summary-param "pair_role=baseline" \
  "${FIXED_ARGS[@]}"

run_case \
  "sembest_p3" "breakpoint_semantic" "ipke-semantic" "ipke-semantic-chunking" "${SEMANTIC_PORT}" "P3" \
  --env "CHUNK_MAX_CHARS=2000" \
  --env "SEM_LAMBDA=${SEM_BEST_LAMBDA}" \
  --env "SEM_WINDOW_W=${SEM_BEST_WINDOW}" \
  --env "SEM_MIN_SENTENCES_PER_CHUNK=${SEM_BEST_MIN_SENT}" \
  --env "SEM_MAX_SENTENCES_PER_CHUNK=${SEM_BEST_MAX_SENT}" \
  --summary-param "pair_role=final"

SUMMARY_PATH="${RESULTS_ROOT}/E9_human_eval_summary.csv"
echo "[E9] Aggregating summary rows -> ${SUMMARY_PATH}"
"${PYTHON_BIN}" "scripts/experiments/build_summary_csv.py" \
  --run-dirs "${RUN_DIRS[@]}" \
  --output "${SUMMARY_PATH}" \
  --extra-column "experiment=E9"

SAMPLES_PATH="${RESULTS_ROOT}/E9_human_eval_samples.json"
echo "[E9] Bundling predictions for human review -> ${SAMPLES_PATH}"
"${PYTHON_BIN}" "scripts/experiments/prepare_human_eval_samples.py" \
  --system "fixed_best_p0=${RESULTS_ROOT}/fixed_best_p0/predictions" \
  --system "sembest_p3=${RESULTS_ROOT}/sembest_p3/predictions" \
  --gold-dir "${GOLD_DIR}" \
  --output "${SAMPLES_PATH}"

echo "[E9] Human evaluation package ready under ${RESULTS_ROOT}"
