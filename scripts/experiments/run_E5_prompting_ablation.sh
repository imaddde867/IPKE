#!/usr/bin/env bash
set -euo pipefail
# Run experiment E5: prompt ablation (P0-P3) on the best semantic chunker configuration.

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
RESULTS_ROOT="${RESULTS_ROOT:-results/E5_prompting}"
SKIP_FLAG="${SKIP_EXISTING:-1}"

# TODO: update SEM_BEST_* values once the preferred semantic config is known.
SEM_BEST_LAMBDA="${SEM_BEST_LAMBDA:-TODO_SET_SEM_LAMBDA}"
SEM_BEST_WINDOW="${SEM_BEST_WINDOW:-TODO_SET_SEM_WINDOW}"
SEM_BEST_MIN_SENT="${SEM_BEST_MIN_SENT:-2}"
SEM_BEST_MAX_SENT="${SEM_BEST_MAX_SENT:-40}"

ensure_value() {
  local name="$1"
  local value="$2"
  if [[ "${value}" == TODO_* ]]; then
    echo "[E5] Please set ${name} before running (edit script or export env var)." >&2
    exit 1
  fi
}

ensure_value "SEM_BEST_LAMBDA" "${SEM_BEST_LAMBDA}"
ensure_value "SEM_BEST_WINDOW" "${SEM_BEST_WINDOW}"

DOCS=(
  "datasets/archive/test_data/text/3m_marine_oem_sop.txt"
  "datasets/archive/test_data/text/DOA_Food_Man_Proc_Stor.txt"
  "datasets/archive/test_data/text/op_firesafety_guideline.txt"
)

mkdir -p "${RESULTS_ROOT}"
RUN_DIRS=()

run_prompt_eval() {
  local prompt_mode="$1"
  local run_dir="${RESULTS_ROOT}/sembest_${prompt_mode,,}"
  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" "scripts/experiments/run_chunker_eval.py"
    --config-id "sembest_${prompt_mode}"
    --method "breakpoint_semantic"
    --host "${HOST}"
    --port "${SEMANTIC_PORT}"
    --service-name "ipke-semantic"
    --container-name "ipke-semantic-chunking"
    --run-dir "${run_dir}"
    --prompt-mode "${prompt_mode}"
    --summary-param "experiment=E5"
    --summary-param "config=sembest_${prompt_mode}"
    --summary-param "prompt=${prompt_mode}"
    --summary-param "chunker=semantic"
    --env "CHUNK_MAX_CHARS=2000"
    --env "SEM_LAMBDA=${SEM_BEST_LAMBDA}"
    --env "SEM_WINDOW_W=${SEM_BEST_WINDOW}"
    --env "SEM_MIN_SENTENCES_PER_CHUNK=${SEM_BEST_MIN_SENT}"
    --env "SEM_MAX_SENTENCES_PER_CHUNK=${SEM_BEST_MAX_SENT}"
  )

  cmd+=(--documents)
  for doc in "${DOCS[@]}"; do
    cmd+=("${doc}")
  done

  if [[ "${SKIP_FLAG}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi

  echo "[E5] Running SEM_BEST with prompt ${prompt_mode} ..."
  "${cmd[@]}"
  RUN_DIRS+=("${run_dir}")
}

for prompt in P0 P1 P2 P3; do
  run_prompt_eval "${prompt}"
done

SUMMARY_PATH="${RESULTS_ROOT}/E5_prompting_summary.csv"
echo "[E5] Aggregating summary rows -> ${SUMMARY_PATH}"
"${PYTHON_BIN}" "scripts/experiments/build_summary_csv.py" \
  --run-dirs "${RUN_DIRS[@]}" \
  --output "${SUMMARY_PATH}" \
  --extra-column "experiment=E5"

echo "[E5] Prompting ablation finished. Results in ${RESULTS_ROOT}"
