#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
GOLD_TIERB="${1:-$ROOT_DIR/datasets/archive/gold_human_tierb}"

if [[ ! -d "$ROOT_DIR/logs" ]]; then
  echo "[tierb] logs directory not found under $ROOT_DIR" >&2
  exit 1
fi

find "$ROOT_DIR/logs" -type d -name 'run_*' | while read -r run_dir; do
  tierb_dir="$run_dir/tierb"
  if [[ ! -d "$tierb_dir" ]]; then
    echo "[tierb] Skipping $run_dir (no tierb predictions)"
    continue
  fi
  out_file="$run_dir/evaluation_tierb.json"
  echo "[tierb] Evaluating $run_dir -> $out_file"
  python "$ROOT_DIR/tools/evaluate.py" \
    --gold_dir "$GOLD_TIERB" \
    --pred_dir "$tierb_dir" \
    --tier B \
    --out_file "$out_file"
done
