# Explainium — Procedural Knowledge Extraction (Starting Point)

Explainium ingests technical, safety, compliance, and operational documents and extracts granular facts and procedural items using a local LLM. The current system runs fully offline with a llama.cpp-backed model and exposes both an API and a Streamlit UI. This README describes the current starting-point state: what exists today, how extraction works, runtime setup, and the current evaluation method.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-00a86b.svg)](https://fastapi.tiangolo.com)
[![Offline](https://img.shields.io/badge/processing-offline-success.svg)](https://github.com)

## Overview

- LLM-only extractor: local Mistral-7B Instruct via `llama-cpp-python` produces JSON entities from text.
- Multi-format ingestion: PDF, DOCX, TXT, PPT(X), images via OCR, spreadsheets, and audio via Whisper.
- Configurable chunking/generation: `chunk_size`, `n_ctx`, `max_tokens`, temperature, GPU backend.
- Confidence tracking: entity-level confidences averaged into `confidence_score` per document.
- API and UI: FastAPI endpoints and a Streamlit front-end for hands-on runs.

## Architecture

- API: `src/api/app.py:1` (FastAPI app, endpoints `/`, `/health`, `/config`, `/extract`, `/stats`, `/clear-cache`).
- Processing: `src/processors/streamlined_processor.py:22` (ingestion from files + routing to the LLM engine).
- LLM Engine: `src/ai/knowledge_engine.py:121` (llama.cpp model invocation, prompt, parsing, caching, scoring).
- Config: `src/core/unified_config.py:35` (env + defaults, GPU auto-detect, LLM/runtime parameters).
- Logging/Middleware: `src/logging_config.py:69`, `src/middleware.py:29` (structured logs, error handling, request IDs).
- Graph (for future structured outputs): `src/graph/models.py:39`, `src/graph/schema.json:1`.

### How Extraction Works

1. Ingest document
   - PDFs: PyMuPDF (`fitz`) text extraction.
   - DOC/DOCX: `python-docx` paragraph text.
   - PPT/PPTX: `python-pptx` slide text.
   - Images: `easyocr` text OCR (if installed).
   - Spreadsheets: `pandas` to string view of data.
   - Audio: Whisper transcription (if installed).
   - Code: `src/processors/streamlined_processor.py:92` → `_extract_*` functions.

2. Chunking and prompting
   - Text is split into chunks of `chunk_size` (default 2000 chars), up to `llm_max_chunks` (default 10), with a heuristic to break on sentence boundaries near the end of a chunk.
   - Each chunk is sent to the LLM with an extraction prompt requesting a JSON array of items.
   - Code: `src/ai/knowledge_engine.py:186` → `_iter_chunks`, `src/ai/knowledge_engine.py:204` → `_process_chunk_with_llm`.

3. Local LLM inference
   - Backend: `llama_cpp.Llama` with `llm_n_ctx`, `llm_max_tokens`, `llm_temperature`, `llm_top_p`.
   - GPU: auto-detected `metal` (Apple Silicon) / `cuda` (NVIDIA) / `cpu` fallback with `LLM_GPU_LAYERS` control.
   - Code: `src/ai/knowledge_engine.py:47` → `_initialize`, `_configure_gpu`.

4. JSON parsing and entity model
   - The engine extracts the first JSON array found in the LLM output and maps items into `ExtractedEntity(content, entity_type, category, confidence, context)`.
   - If parsing fails, that chunk contributes no entities (no repair step is currently applied).
   - Code: `src/ai/knowledge_engine.py:229` → `_parse_llm_response`.

5. Confidence and caching
   - Entity-level `confidence` is taken from the model output; if missing, it defaults to `CONFIDENCE_THRESHOLD`.
   - Document-level `confidence_score` is the average of entity confidences.
   - A simple content-hash cache avoids repeated recomputation for unchanged inputs.
   - Code: `src/ai/knowledge_engine.py:147` (defaults), `src/ai/knowledge_engine.py:168` (confidence_score), `src/ai/knowledge_engine.py:303` (cache).

### What the LLM Extracts

Prompt asks for granular items across categories, for example:
- `procedure_step`, `process_sequence`
- `safety_requirement`, `warning_caution`
- `equipment_tool`, `material_product`
- `specification`, `measurement_value`, `time_duration`, `quality_standard`
- `requirement_condition`, `maintenance_instruction`, `contact_info`

The exact list is in the prompt in `src/ai/knowledge_engine.py:208`.

## Setup

### Install
```bash
pip install -r requirements.txt
```

Optional (only if you use these features):
- OCR: `pip install easyocr`
- Audio transcription: `pip install openai-whisper`

### Local Model (Required)

- Place a llama.cpp GGUF model at:
  - `models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf`
- The system does not auto-download models. Update `llm_model_path` in `src/core/unified_config.py:62` or via env if using a different path.

### Environment and GPU

Set via environment variables (examples):
```bash
export EXPLAINIUM_ENV=development
export MAX_FILE_SIZE_MB=50
export CONFIDENCE_THRESHOLD=0.8
export QUALITY_THRESHOLD=0.7

export ENABLE_GPU=true
export GPU_BACKEND=auto      # auto | metal | cuda | cpu
export LLM_GPU_LAYERS=-1     # -1=all layers on GPU, 0=CPU
export GPU_MEMORY_FRACTION=0.8
```

`n_ctx`, `chunk_size`, `llm_max_tokens`, and other parameters are configurable; see `src/core/unified_config.py:210` → `get_llm_config`.

## Run

### API Server
```bash
python main.py
# or
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Endpoints
- `/` basic info
- `/health` health check
- `/config` current LLM/runtime config
- `/extract` multipart file upload → JSON entities
- `/stats` processing stats
- `/clear-cache` clear in-memory caches

Example
```bash
curl -X POST http://localhost:8000/extract \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@documents_samples/datasets/Good/extracted/3m_marine_oem_sop.txt'
```

Response shape
```json
{
  "document_id": "...",
  "document_type": "text",
  "entities": [
    {"content": "...", "entity_type": "specification", "category": "distance_requirement", "confidence": 0.95, "context": "..."}
  ],
  "confidence_score": 0.87,
  "processing_time": 2.34,
  "strategy_used": "llm_default",
  "metadata": {"file_format": ".txt", "file_name": "...", "entities_extracted": 42}
}
```

### Streamlit UI
```bash
streamlit run streamlit_app.py
```

- Tune `chunk_size`, `n_ctx`, `llm_max_tokens`, temperature, GPU settings.
- Upload any supported file and inspect extracted entities in a table.

## Current Evaluation Method

Baseline evaluation focuses on consistency, extract coverage, and safety via light-weight metrics and human review:

- Document-level metrics
  - Entity count and `confidence_score` (reported by the engine).
  - Processing stats (chunks processed, average time) via `/stats`.

- Manual review on 3 seed documents
  - `documents_samples/datasets/Good/extracted/3m_marine_oem_sop.txt`
  - `documents_samples/datasets/Good/extracted/DOA_Food_Man_Proc_Stor.txt`
  - `documents_samples/datasets/Good/extracted/op_firesafety_guideline.txt`
  - Check for presence of key procedural steps, obvious misses/spurious items, and any safety‑critical mistakes.

- What “confidence” means now
  - It is model-supplied per entity; when absent, defaults to `CONFIDENCE_THRESHOLD` (configurable).
  - The system logs a warning if the averaged `confidence_score` drops below `QUALITY_THRESHOLD`; results are not filtered.

Structured evaluation for knowledge graphs is scaffolded (for future use):
- Script: `tests/evaluation/evaluate_structured.py:193` compares predicted vs. gold graphs using macro entity F1, relation F1, and ordering F1 (NEXT adjacency) against the schema in `src/graph/schema.json`.
- Use when the extractor outputs the procedural graph format (planned iteration).

## Roadmap (Evaluation & Extraction)

- Text-phase metrics (planned)
  - Step presence F1 via soft-matching and order correlation (Kendall/Spearman) on matched steps.
  - Optional ROUGE-L on concatenated steps for paraphrase adequacy.

- Graph-phase metrics (when emitting `Graph` JSON)
  - Triple-level F1 (typed relations), role/slot F1 per step, NEXT-edge F1 + order correlation, constraint checks.

- Prompting/model ablations
  - Compare baseline prompts vs. few-shot; adjust `chunk_size`, `n_ctx`, `llm_max_tokens`; test different local models.

## Known Limitations (Current State)

- No auto-download of models; a local GGUF file is required.
- JSON parsing is strict; if the model emits trailing text or malformed arrays, that chunk yields zero entities.
- Confidence is model-reported; no separate calibration step is performed.
- No embeddings/vector DB are used today (reserved for future).

## Development

### Tests
```bash
pytest -q
```

Useful checks
- Engine import: `python -c "from src.ai.knowledge_engine import UnifiedKnowledgeEngine; print('OK')"`
- Evaluate structured graph (future): `python tests/evaluation/evaluate_structured.py --ground-truth gt.json --predictions pred.json --pretty`

## System Requirements

- Python 3.12+
- 8GB+ RAM (more recommended for larger context windows)
- Optional GPU for acceleration: Apple Silicon (Metal) or NVIDIA (CUDA)

## Tips

- If `n_ctx < (approx_input_tokens + llm_max_tokens)`, reduce `chunk_size` or `llm_max_tokens`.
- Lower temperature (e.g., 0.0–0.1) improves determinism and JSON compliance.
- If you encounter frequent JSON parse failures, try reducing `chunk_size` and `llm_max_tokens` for tighter prompts.
