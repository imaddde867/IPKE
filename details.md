This file provides guidance to any dev working with code in this repository.

Project: Industrial Procedural Knowledge Extraction (IPKE)

Common commands
- Create venv + install deps
  - macOS/Linux:
    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
- Install spaCy English model (required by preflight/tests)
  ```bash
  python -m spacy download en_core_web_sm
  ```
- Run the API (FastAPI via uvicorn, includes startup checks)
  ```bash
  python main.py
  # docs at http://localhost:8000/docs
  ```
  Alternative (without the model preflight in main.py):
  ```bash
  uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --log-level info
  ```
- Run the Streamlit UI
  ```bash
  streamlit run streamlit_app.py
  ```
- Run tests (pytest)
  ```bash
  pytest -q
  pytest tests/test_processor.py -q                          # single file
  pytest tests/test_processor.py::TestDocumentProcessor::test_process_plain_text -q  # single test
  pytest -m integration -q                                  # only tests marked "integration"
  ```
  Notes: tests will skip LLM-dependent paths if backends are unavailable. To force CPU-safe config during tests, you can set `EXPLAINIUM_ENV=testing` or `GPU_BACKEND=cpu`.
- Baseline evaluation scripts
  ```bash
  # Preflight checks for assets/backends (returns non-zero if something is missing)
  python scripts/baseline_preflight.py          # add --json for machine output

  # Run extraction + evaluation loops (writes predictions to logs/ and a JSON report)
  python scripts/run_baseline_loops.py --runs 3 --out logs/baseline_runs

  # Generate plots from existing run outputs
  python scripts/plot_baseline_metrics.py
  ```
  Important: `src/pipelines/baseline.evaluate_predictions` calls a CLI `tools/evaluate.py` that is not present in this repo. Until that evaluator is added, `scripts/run_baseline_loops.py` will fail at the evaluation step. Workarounds:
  - Use `scripts/baseline_preflight.py` to verify setup, then adapt your workflow to only run extraction (e.g., import and call `extract_documents` from `src.pipelines.baseline` in a short script), or
  - Use the lightweight evaluator at `tests/evaluation/evaluate_structured.py` for ad‑hoc comparisons of a single prediction vs. ground truth.
- Model download (Metal/GGUF, optional helper)
  ```bash
  # Will prompt for Hugging Face token, then download the GGUF model into models/llm
  python scripts/download_llm.py
  ```
  Or via huggingface-cli as in README:
  ```bash
  huggingface-cli login
  huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
    --include "mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
    --local-dir models/llm --local-dir-use-symlinks False
  ```

High-level architecture
- Configuration (src/core/unified_config.py)
  - Single source of truth for env, runtime, and model settings.
  - Environments: development (default), testing, production. Testing disables GPU and uses CPU-safe defaults.
  - GPU backend detection: `auto` resolves to `metal` on Apple Silicon, `cuda` when `nvidia-smi` is available, else `cpu`.
  - LLM settings expose both llama.cpp (GGUF) and transformers (HF) parameters; note that the GGUF model path is currently a code default (`models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf`) and not overridden via env.
- Logging and middleware (src/logging_config.py, src/middleware.py)
  - JSON logging with correlation IDs; middleware attaches `X-Correlation-ID` to requests/responses and centralizes error handling.
- API layer (src/api/app.py)
  - FastAPI app with endpoints: `/`, `/health`, `/config`, `/extract`, `/stats`, `/clear-cache`.
  - Validates uploads (size, type), saves temp file, delegates to the processor, returns normalized entities/steps/constraints with timing/strategy metadata.
- Document processing (src/processors/streamlined_processor.py)
  - Detects file type and extracts text via optional libs (PyMuPDF, python-docx, easyocr, whisper, pandas/pptx).
  - Calls the knowledge engine for extraction; returns a compact `ProcessingResult` and tracks per-format stats.
- Knowledge engine (src/ai/knowledge_engine.py)
  - Strategy pattern with two backends:
    - `LlamaCppStrategy` for Metal/GGUF via `llama-cpp-python`.
    - `TransformersStrategy` for CUDA/CPU via HF `transformers` (+ optional 4/8‑bit using `bitsandbytes`).
  - Chunks input, prompts for a single JSON object, robustly parses, normalizes steps/constraints, and aggregates metrics. Simple in‑memory caching and performance counters included.
- Graph domain + adapter (src/graph/models.py, src/graph/adapter.py)
  - Pydantic models and JSON schema for a procedural graph (Steps, Conditions, Equipment, Parameters; typed Edges).
  - `flat_to_tierb` converts baseline flat outputs into Tier‑B nodes/edges expected by the evaluator.
- Baseline pipeline (src/pipelines/baseline.py)
  - Orchestrates batch extraction for canonical test docs under `datasets/archive/test_data/text/` and writes both flat and Tier‑B JSONs under a run dir.
  - Expects a CLI evaluator at `tools/evaluate.py` to produce metrics (report JSON) and provides helpers to accumulate/summarize metrics across runs.

Repo‑specific notes and gotchas
- Model placement for Metal/llama.cpp
  - `main.py` checks `models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf` at startup; place the GGUF there (see download helper above). There is no current `.env` override for this path.
- .env template
  - See `.env.example` for commonly tuned runtime knobs (backend selection, chunk sizes, thresholds). Typical variables: `EXPLAINIUM_ENV`, `GPU_BACKEND`, `ENABLE_GPU`, `LLM_GPU_LAYERS`, `GPU_MEMORY_FRACTION`, `CHUNK_SIZE`, `QUALITY_THRESHOLD`.
- Datasets and embeddings
  - Gold annotations live under `datasets/archive/gold_human/` and an embedding model is expected under `models/embeddings/all-mpnet-base-v2` (present in repo). Preflight checks verify these.
- Lint/format
  - No linter/formatter configuration is present in the repo; no commands are defined for linting.

Key references from README.md (kept minimal)
- Dual‑backend inference (Metal via llama.cpp GGUF, CUDA via transformers + bitsandbytes).
- FastAPI endpoints and Streamlit UI for iteration; JSON logging with correlation IDs.
- Evaluation scripts expect local gold data and an evaluator CLI; plotting helpers generate trend/aggregate visuals from run outputs.
