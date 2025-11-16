#!/usr/bin/env bash
set -euo pipefail
# Run the full E0-E9 experiment suite (configurable via SKIP_EXX flags).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -d ".venv311" ]]; then
  source ".venv311/bin/activate"
elif [[ -d ".venv" ]]; then
  source ".venv/bin/activate"
fi

# Allow skipping individual experiments via SKIP_EXX=1 (E7 defaults to skip).
SKIP_E0="${SKIP_E0:-0}"
SKIP_E1="${SKIP_E1:-0}"
SKIP_E2="${SKIP_E2:-0}"
SKIP_E3="${SKIP_E3:-0}"
SKIP_E4="${SKIP_E4:-0}"
SKIP_E5="${SKIP_E5:-0}"
SKIP_E6="${SKIP_E6:-0}"
SKIP_E7="${SKIP_E7:-1}"
SKIP_E8="${SKIP_E8:-0}"
SKIP_E9="${SKIP_E9:-0}"

run_step() {
  local label="$1"
  local script_path="$2"
  local skip="$3"
  if [[ "${skip}" == "1" ]]; then
    echo "[ALL] Skipping ${label} (${script_path})"
    return
  fi
  echo "[ALL] Running ${label} via ${script_path}"
  bash "${script_path}"
}

run_step "E0 Baseline" "scripts/experiments/run_E0_baseline.sh" "${SKIP_E0}"
run_step "E1 Fixed sweep" "scripts/experiments/run_E1_fixed_sweep.sh" "${SKIP_E1}"
run_step "E2 Semantic sweep" "scripts/experiments/run_E2_semantic_sweep.sh" "${SKIP_E2}"
run_step "E3 DSC sweep" "scripts/experiments/run_E3_dsc_sweep.sh" "${SKIP_E3}"
run_step "E4 Chunker comparison" "scripts/experiments/run_E4_chunker_comparison.sh" "${SKIP_E4}"
run_step "E5 Prompting ablation" "scripts/experiments/run_E5_prompting_ablation.sh" "${SKIP_E5}"
run_step "E6 Interaction" "scripts/experiments/run_E6_chunking_prompt_interaction.sh" "${SKIP_E6}"
run_step "E7 Model scaling" "scripts/experiments/run_E7_model_scaling.sh" "${SKIP_E7}"
run_step "E8 Cross-doc generalisation" "scripts/experiments/run_E8_cross_doc_generalisation.sh" "${SKIP_E8}"
run_step "E9 Human eval outputs" "scripts/experiments/run_E9_human_eval_outputs.sh" "${SKIP_E9}"

echo "[ALL] Experiment pipeline finished."
