#!/usr/bin/env bash
set -euo pipefail
# Run experiment E6: interaction between chunking (Fixed vs Semantic best) and prompting (P0 vs P3).

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
RESULTS_ROOT="${RESULTS_ROOT:-results/E6_interaction}"
SKIP_FLAG="${SKIP_EXISTING:-1}"

# Shared best-config placeholders (override via env vars after E1/E2 analysis).
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
    echo "[E6] Please set ${name} before running (edit script or export env var)." >&2
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

run_combo() {
  local chunker="$1"  # fixed_best or sembest
  local prompt="$2"
  local run_dir="${RESULTS_ROOT}/${chunker}_${prompt,,}"
  local config_id="${chunker}_${prompt}"
  local method service container port
  local env_args=()

  case "${chunker}" in
    fixed_best)
      method="fixed"
      service="ipke-fixed"
      container="ipke-fixed-chunking"
      port="${FIXED_PORT}"
      env_args+=(--env "CHUNK_MAX_CHARS=${FIXED_BEST_MAX_CHARS}")
      if [[ -n "${FIXED_BEST_STRIDE}" && "${FIXED_BEST_STRIDE}" != "0" ]]; then
        env_args+=(--env "CHUNK_STRIDE_CHARS=${FIXED_BEST_STRIDE}")
      fi
      ;;
    sembest)
      method="breakpoint_semantic"
      service="ipke-semantic"
      container="ipke-semantic-chunking"
      port="${SEMANTIC_PORT}"
      env_args+=(--env "CHUNK_MAX_CHARS=2000")
      env_args+=(--env "SEM_LAMBDA=${SEM_BEST_LAMBDA}")
      env_args+=(--env "SEM_WINDOW_W=${SEM_BEST_WINDOW}")
      env_args+=(--env "SEM_MIN_SENTENCES_PER_CHUNK=${SEM_BEST_MIN_SENT}")
      env_args+=(--env "SEM_MAX_SENTENCES_PER_CHUNK=${SEM_BEST_MAX_SENT}")
      ;;
    *)
      echo "[E6] Unknown chunker key: ${chunker}" >&2
      exit 1
      ;;
  esac

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
    --prompt-mode "${prompt}"
    --summary-param "experiment=E6"
    --summary-param "config=${config_id}"
    --summary-param "chunker=${chunker}"
    --summary-param "prompt=${prompt}"
  )

  cmd+=(--documents)
  for doc in "${DOCS[@]}"; do
    cmd+=("${doc}")
  done

  if [[ "${SKIP_FLAG}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi

  cmd+=("${env_args[@]}")

  echo "[E6] Running ${chunker} with ${prompt} ..."
  "${cmd[@]}"
  RUN_DIRS+=("${run_dir}")
}

run_combo fixed_best P0
run_combo fixed_best P3
run_combo sembest P0
run_combo sembest P3

SUMMARY_PATH="${RESULTS_ROOT}/E6_interaction_summary.csv"
echo "[E6] Aggregating summary rows -> ${SUMMARY_PATH}"
"${PYTHON_BIN}" "scripts/experiments/build_summary_csv.py" \
  --run-dirs "${RUN_DIRS[@]}" \
  --output "${SUMMARY_PATH}" \
  --extra-column "experiment=E6"

echo "[E6] Interaction study completed. Results in ${RESULTS_ROOT}"
