#!/usr/bin/env bash

# Minimal Tier A extraction + evaluation pipeline.
# Assumes the virtualenv is activated and the llama.cpp model path is configured.

set -euo pipefail

OUT_DIR="logs/extractions/tierA_latest"

# 1) Extract structured outputs for the baseline tier-A documents.
python - <<PY
import asyncio
from pathlib import Path

from src.pipelines.baseline import TIER_A_TEST_DOCS, extract_documents
from src.processors.streamlined_processor import StreamlinedDocumentProcessor

OUT_DIR = Path("${OUT_DIR}")


async def main():
    processor = StreamlinedDocumentProcessor()

    def status(doc_id, payload):
        print(f"{doc_id}: {len(payload['steps'])} steps / {len(payload['entities'])} entities")

    await extract_documents(processor, doc_sources=TIER_A_TEST_DOCS, run_dir=OUT_DIR, status_callback=status)


asyncio.run(main())
PY

# 2) Evaluate Tier A metrics against the manual gold references.
python evaluate.py \
  --gold_dir datasets/archive/gold_human \
  --pred_dir "${OUT_DIR}" \
  --tier A \
  --embedding_model models/embeddings/all-mpnet-base-v2 \
  --out_file "${OUT_DIR}/tierA_report.json"

# Optional: display the JSON report
cat "${OUT_DIR}/tierA_report.json"
