#!/usr/bin/env bash

# Minimal Tier A extraction + evaluation pipeline.
# Assumes the virtualenv is activated and the llama.cpp model path is configured.

set -euo pipefail

OUT_DIR="logs/extractions/tierA_latest"

# 1) Extract structured outputs for the baseline tier-A documents.
python - <<'PY'
import asyncio, json
from dataclasses import asdict
from pathlib import Path
from src.processors.streamlined_processor import StreamlinedDocumentProcessor

docs = {
    "3M_OEM_SOP": "datasets/archive/manual_sources/text/3m_marine_oem_sop.txt",
    "DOA_Food_Man_Proc_Stor": "datasets/archive/manual_sources/text/DOA_Food_Man_Proc_Stor.txt",
    "op_firesafety_guideline": "datasets/archive/manual_sources/text/op_firesafety_guideline.txt",
}
out_dir = Path("logs/extractions/tierA_latest")
out_dir.mkdir(parents=True, exist_ok=True)

async def main():
    processor = StreamlinedDocumentProcessor()
    for doc_id, path in docs.items():
        result = await processor.process_document(file_path=path, document_id=doc_id)
        payload = {
            "document_id": doc_id,
            "document_type": result.document_type,
            "steps": result.extraction_result.steps,
            "constraints": result.extraction_result.constraints,
            "entities": [asdict(ent) for ent in result.extraction_result.entities],
            "confidence_score": result.extraction_result.confidence_score,
            "processing_time": result.processing_time,
            "strategy_used": result.extraction_result.strategy_used,
            "metadata": result.metadata,
        }
        (out_dir / f"{doc_id}.json").write_text(json.dumps(payload, indent=2))
        print(f"{doc_id}: {len(payload['steps'])} steps / {len(payload['entities'])} entities")

asyncio.run(main())
PY

# 2) Evaluate Tier A metrics against the manual gold references.
python evaluate.py \
  --gold_dir datasets/archive/manual_gold \
  --pred_dir "${OUT_DIR}" \
  --tier A \
  --embedding_model models/embeddings/all-mpnet-base-v2 \
  --out_file "${OUT_DIR}/tierA_report.json"

# Optional: display the JSON report
cat "${OUT_DIR}/tierA_report.json"
