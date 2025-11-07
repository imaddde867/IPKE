# Industrial Procedural Knowledge Extraction (IPKE)

Streamlined system to extract structured procedural knowledge (steps, constraints, entities) from unstructured technical documents. Powered by a local LLM (Mistral‑7B Instruct via llama.cpp), with a FastAPI service, a Streamlit UI, and an evaluation pipeline.

## Highlights
- LLM‑driven extraction with chunking and normalization
- Local inference via llama.cpp (CUDA/Metal/CPU auto‑detect)
- Unified config with sane defaults and .env overrides
- FastAPI endpoints + Streamlit UI for quick iteration
- JSON logging with correlation IDs (request tracing)
- Baseline pipeline + evaluator with reproducible metrics
- Broad format support (PDF/DOCX/TXT/CSV/PPTX/Images/Audio)

## Repository Layout
- `main.py` — starts the FastAPI server with model checks
- `app.py` — Streamlit UI for interactive uploads
- `evaluate.py` — evaluator for structured predictions (Tier A & B)
- `run_baseline_loops.py` — batch extract + evaluate over test set
- `scripts/` — preflight checks and metric plots
- `src/`
  - `api/app.py` — FastAPI app, routes, models, middleware wiring
  - `ai/knowledge_engine.py` — llama.cpp integration + JSON parsing/normalization
  - `processors/streamlined_processor.py` — format‑aware loaders + knowledge engine orchestration
  - `core/unified_config.py` — single source of truth for all settings
  - `graph/` — pydantic graph models and JSON schema
  - `middleware.py`, `exceptions.py`, `logging_config.py` — infra and ergonomics
- `tests/` — unit + integration tests (API, processor, config, evaluation)
- `models/` — expected local model artifacts (LLM/embeddings)

## Quickstart
1) Environment
- Python 3.10+ recommended (Ran my tests 3.11 and runs well)
- macOS (Metal), Linux/Windows (CUDA) or CPU‑only

2) Install
```
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

3) Download the LLM (one‑time)
- Default path: `models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf`
```
huggingface-cli login
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  --include "mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
  --local-dir models/llm --local-dir-use-symlinks False
```

4) Configure (optional but i recommend it)
- Copy `.env.example` to `.env` and adjust. Key knobs below.

5) Run the API
```
python main.py
# or
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
- Open docs: `http://localhost:8000/docs`

6) Run the UI
```
streamlit run app.py
```

## API Overview
- `GET /` and `GET /health` — health status
- `GET /config` — active model/runtime settings
- `POST /extract` — multipart file upload → structured extraction
- `GET /stats` — processor + engine stats
- `POST /clear-cache` — clear in‑memory extraction cache

Example:
```
curl -X POST http://localhost:8000/extract \
  -F "file=@/path/to/document.pdf"
```
Response (shape):
```
{
  "document_id": "...",
  "document_type": "manual|document|text|...",
  "entities": [
    {"content": "...", "entity_type": "...", "category": "...", "confidence": 0.9, "context": "..."}
  ],
  "steps": [{"id": "S1", "text": "...", "order": 1, "confidence": 0.9}],
  "constraints": [{"id": "C1", "text": "...", "steps": ["S1"], "confidence": 0.85}],
  "confidence_score": 0.83,
  "processing_time": 1.23,
  "strategy_used": "llm_default",
  "metadata": {...}
}
```

## How It Works
- `StreamlinedDocumentProcessor`
  - Detects format → extracts text (PDF via PyMuPDF, DOCX via python‑docx, PPTX via python‑pptx, CSV/XLSX via pandas, images via easyocr, audio via Whisper). Falls back to plain text.
  - Chunks input and calls the `UnifiedKnowledgeEngine`.
  - Tracks stats and returns consistent `ProcessingResult`.
- `UnifiedKnowledgeEngine`
  - Uses llama.cpp to run Mistral‑7B with GPU auto‑detection (Metal/CUDA/CPU fallback).
  - Prompts the model to return a single JSON object; robustly parses variants.
  - Normalizes steps/constraints/entities; caches results; collects performance metrics.
- Infra
  - JSON logging with correlation IDs; central error handling middleware.
  - Unified configuration w/ environment profiles (dev/test/prod) and helper getters.

## Configuration (.env)
Most used variables (see `.env.example` and `src/core/unified_config.py`):
- `EXPLAINIUM_ENV` — `development|testing|production` (default: development)
- `ENABLE_GPU` — `true|false` (default: true)
- `GPU_BACKEND` — `auto|metal|cuda|cpu` (default: auto)
- `LLM_GPU_LAYERS` — `-1` for all layers, or cap (e.g., `32`) (default: -1)
- `GPU_MEMORY_FRACTION` — fraction of total GPU memory (default: 0.8)
- `CHUNK_SIZE` — characters per chunk (default: 2000)
- `LLM_N_CTX` — context window (default: 8192)
- `LLM_TEMPERATURE` — decoding temperature (default: 0.1)
- `LLM_MAX_TOKENS` — max generation tokens (default: 1536)
- `CONFIDENCE_THRESHOLD` / `QUALITY_THRESHOLD` — scoring gates
- `MAX_WORKERS` — threadpool workers
- `API_HOST` — bind host for the API (default: 127.0.0.1)

## Evaluation & Baselines
- Preflight checks (assets/deps):
```
python scripts/baseline_preflight.py          # add --json for machine output
```
- Run extraction + evaluation loops (saves per‑run predictions + reports):
```
python run_baseline_loops.py --runs 3 --out logs/baseline_runs
```
- Plot trends and aggregates:
```
python scripts/plot_baseline_metrics.py
```
Notes:
- Defaults expect a local gold set and embedding model. Adjust `src/pipelines/baseline.py` paths or pass explicit args to `evaluate.py`.
- Evaluator (`evaluate.py`) computes headline metrics (StepF1, AdjacencyF1, Kendall, ConstraintCoverage, ConstraintAttachmentF1, A_score, GraphF1, NEXT_EdgeF1, Logic_EdgeF1, B_score) and writes a JSON report.

## Supported Formats
- Text: `.pdf`, `.doc/.docx`, `.txt`, `.rtf`
- Spreadsheets: `.csv`, `.xls/.xlsx`
- Presentations: `.ppt/.pptx`
- Images: `.jpg/.jpeg/.png/.gif/.bmp/.tiff` (OCR via easyocr)
- Audio: `.mp3/.wav/.flac/.aac` (transcribe via Whisper)

Install optional extras from `requirements.txt` for OCR/Audio/PDF/Office support. Without them, extraction logs a warning and returns empty content for that modality.

## Logging & Diagnostics
- Logs to console and `logs/app.log` in JSON with correlation IDs.
- `GET /stats` returns processor and engine counters (documents processed, avg times, cache hits, etc.).
- Streamlit UI exposes key tuning parameters and shows basic diagnostics.

## License
MIT — see `LICENSE`.
