# Industrial Procedural Knowledge Extraction (IPKE)

Streamlined system to extract structured procedural knowledge (steps, constraints, entities) from unstructured technical documents. Powered by a local LLM (Mistral‑7B Instruct) with a dual-backend architecture for Metal and CUDA GPUs.

## Highlights
- **Dual-Backend LLM Inference**:
  - **`llama.cpp` on Metal**: Optimized for high-performance inference on Apple Silicon GPUs.
  - **`transformers` on CUDA**: Leverages PyTorch and `bitsandbytes` for efficient inference on NVIDIA GPUs.
- LLM‑driven extraction with chunking and normalization
- Unified config with sane defaults and `.env` overrides
- FastAPI endpoints + Streamlit UI for quick iteration
- JSON logging with correlation IDs (request tracing)
- Baseline pipeline + evaluator with reproducible metrics
- Broad format support (PDF/DOCX/TXT/CSV/PPTX/Images/Audio)

## Repository Layout
- `main.py` — starts the FastAPI server with model checks
- `streamlit_app.py` — Streamlit UI for interactive uploads
- `tools/evaluate.py` — evaluator for structured predictions (Tier A & B)
- `scripts/run_baseline_loops.py` — batch extract + evaluate over test set
- `scripts/` — preflight checks and metric plots
- `src/`
  - `api/app.py` — FastAPI app, routes, models, middleware wiring
  - `ai/knowledge_engine.py` — Dual-backend LLM integration (`llama.cpp` and `transformers`)
  - `processors/streamlined_processor.py` — format‑aware loaders + knowledge engine orchestration
  - `core/unified_config.py` — single source of truth for all settings
  - `graph/` — pydantic graph models and JSON schema
- `tests/` — unit + integration tests

## Quickstart

1) **Environment**
- Python 3.10+
- **For Metal**: macOS with Apple Silicon.
- **For CUDA**: Linux/Windows with an NVIDIA GPU.

2) **Install Dependencies**
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2.1) Install spaCy English model (required)
```bash
python -m spacy download en_core_web_sm
```
- If your environment is offline, pre-download the wheel and install it:
```bash
pip install en_core_web_sm-*.whl
```
- The preflight script checks this model.

3) **Set up Your Model**

The setup depends on your `GPU_BACKEND` choice (`metal` or `cuda`).

**For `metal` backend (Apple Silicon):**
You must download the GGUF model manually.
```bash
huggingface-cli login
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  --include "mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
  --local-dir models/llm --local-dir-use-symlinks False
```
Ensure your `.env` file points to this model via `LLM_MODEL_PATH`.

**For `cuda` backend (NVIDIA GPU):**
No download is needed. The `transformers` library will automatically download the model specified by `LLM_MODEL_ID` on the first run.

4) **Configure Your Environment**
- Copy `.env.example` to `.env`.
- Set `GPU_BACKEND` to `metal` or `cuda`.
- Adjust other settings as needed (see Configuration section).

5) **Run the API**
```bash
python main.py
```
- Open docs: `http://localhost:8000/docs`

6) **Run the UI**
```bash
streamlit run streamlit_app.py
```

## How It Works
- `StreamlinedDocumentProcessor`
  - Detects format and extracts text from various file types.
  - Chunks input and calls the `UnifiedKnowledgeEngine`.
- `UnifiedKnowledgeEngine`
  - **Detects the GPU backend** (`metal`, `cuda`, or `cpu`) from the configuration.
  - **Initializes the appropriate strategy**:
    - `LlamaCppStrategy`: For `metal`, uses `llama-cpp-python` to run a GGUF model.
    - `TransformersStrategy`: For `cuda`, uses Hugging Face `transformers` to run a PyTorch model with `bitsandbytes` quantization.
  - Prompts the model to return a single JSON object and robustly parses the output.
- **Infra**
  - JSON logging with correlation IDs and central error handling.
  - Unified configuration with environment profiles.

## Configuration (.env)
Set the `GPU_BACKEND` to `metal`, `cuda`, or `auto`.

**Common Settings:**
- `EXPLAINIUM_ENV`: `development|testing|production`
- `ENABLE_GPU`: `true|false`
- `CHUNK_SIZE`, `CHUNK_MAX_CHARS`, `CHUNKING_METHOD`, `LLM_N_CTX`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`
- Semantic chunking knobs: `SEM_LAMBDA`, `SEM_WINDOW_W`, `SEM_MIN_SENTENCES_PER_CHUNK`, `SEM_MAX_SENTENCES_PER_CHUNK`, `SEM_SIMILARITY`
- Dual Semantic Chunker knobs: `DSC_PARENT_MIN_SENTENCES`, `DSC_PARENT_MAX_SENTENCES`, `DSC_DELTA_WINDOW`, `DSC_THRESHOLD_K`, `DSC_USE_HEADINGS`
- Diagnostics: `DEBUG_CHUNKING=true` emits per-chunk spans/metadata in the API response

**`metal` Backend Settings (`llama.cpp`):**
- `LLM_MODEL_PATH`: Path to your local `.gguf` model file.
- `LLM_N_GPU_LAYERS`: Number of layers to offload to the GPU (`-1` for all).

**`cuda` Backend Settings (`transformers`):**
- `LLM_MODEL_ID`: Hugging Face model ID (e.g., `mistralai/Mistral-7B-Instruct-v0.2`).
- `LLM_QUANTIZATION`: `4bit` or `8bit` for quantization on CUDA.

## Chunking Methods
Select the chunker via `CHUNKING_METHOD` (default `fixed`). Every chunk is capped by `CHUNK_MAX_CHARS` to keep LLM prompts safe.

1. **`fixed`** – legacy behavior: greedily splits every ~`CHUNK_MAX_CHARS` characters on whitespace. Zero dependencies, ideal for smoke tests or extremely small machines.
2. **`breakpoint_semantic`** – sentence-level dynamic programming. Steps:
   - SpaCy (`en_core_web_sm`) splits sentences.
   - Sentence embeddings (default `models/embeddings/all-mpnet-base-v2`) provide cosine similarities between consecutive sentences.
   - Objective: maximise average cohesion minus a per-break penalty `SEM_LAMBDA`, constrained by window `SEM_WINDOW_W` and min/max sentence counts.
3. **`dsc`** (Dual Semantic Chunker) – two stage:
   - Parent blocks via distance deltas `d_t = 1 - cos(e_t, e_{t+1})` and adaptive threshold `θ = μ + k·σ` (`DSC_THRESHOLD_K`) with rolling window `DSC_DELTA_WINDOW`. `DSC_PARENT_MIN_SENTENCES` / `DSC_PARENT_MAX_SENTENCES` bound block size, heading heuristics (`DSC_USE_HEADINGS=true`) bias toward enumerated sections.
   - Each parent block is refined by the breakpoint semantic chunker to produce child chunks, inheriting parent metadata.

Example configurations:

```bash
export CHUNKING_METHOD=breakpoint_semantic
export EMBEDDING_MODEL_PATH=models/embeddings/all-mpnet-base-v2
export SEM_LAMBDA=0.15
export SEM_WINDOW_W=30
export SEM_MIN_SENTENCES_PER_CHUNK=2
export SEM_MAX_SENTENCES_PER_CHUNK=40
export CHUNK_MAX_CHARS=2000
```

```bash
export CHUNKING_METHOD=dsc
export DSC_PARENT_MIN_SENTENCES=10
export DSC_PARENT_MAX_SENTENCES=120
export DSC_DELTA_WINDOW=25
export DSC_THRESHOLD_K=1.0
export DSC_USE_HEADINGS=true
```

> **Embedding asset**: semantic chunkers expect `models/embeddings/all-mpnet-base-v2` (SentenceTransformers format). Download once with  
> `huggingface-cli download sentence-transformers/all-mpnet-base-v2 --local-dir models/embeddings/all-mpnet-base-v2 --local-dir-use-symlinks False`.

Turn on `DEBUG_CHUNKING=true` to record chunk spans, cohesion, and parent metadata in every extraction response/log line—handy when tuning `SEM_LAMBDA`/`θ` for EBBC or DSC chunkers.

## Evaluation & Baselines
- Preflight checks (assets/deps):
```
python scripts/baseline_preflight.py          # add --json for machine output
```
- Run extraction + evaluation loops (saves per‑run predictions + reports):
```
python scripts/run_baseline_loops.py --runs 3 --out logs/baseline_runs
```
- Plot trends and aggregates:
```
python scripts/plot_baseline_metrics.py
```
Notes:
- Defaults expect a local gold set and embedding model. Adjust `src/pipelines/baseline.py` paths or pass explicit args to `tools/evaluate.py`.
- Evaluator (`tools/evaluate.py`) computes headline metrics (StepF1, AdjacencyF1, Kendall, ConstraintCoverage, ConstraintAttachmentF1, A_score, GraphF1, NEXT_EdgeF1, Logic_EdgeF1, B_score) and writes a JSON report.
- 2025-11-09: Added a tiny Tier-B adapter at `src/graph/adapter.py` to convert flat predictions (steps/constraints/entities) into `nodes[]` + `edges[]` with lowercase relation types (e.g., "next", "condition_on") for `evaluate.py`. Baseline extractor remains flat; the adapter is optional.
- Next session: Evaluate Tier A + Tier B with `evaluate.py`; only after that, touch semantic chunking and validators.

## Extraction & Evaluation

Run batch extraction and evaluation on test documents:

```bash
# Single extraction run with evaluation
python scripts/run_baseline_loops.py

# Multiple runs for statistical analysis
python scripts/run_baseline_loops.py --runs 5

# Include visualizations
python scripts/run_baseline_loops.py --visualize
```

**Configuration:** Unlimited chunks (`max_chunks=0`), full GPU acceleration

**Output:** `logs/extraction/` with predictions, metrics, and optional plots

### Tier-B conversion for gold and evaluation
- Convert human gold (flat) to Tier‑B nodes/edges:
```
python -m tools.convert_flat_to_tierb \
  --in datasets/archive/gold_human \
  --out datasets/archive/gold_human_tierb
```
- Evaluate Tier A (flat):
```
python tools/evaluate.py \
  --gold_dir datasets/archive/gold_human \
  --pred_dir logs/baseline_runs/run_1 \
  --tier A \
  --out_file logs/baseline_runs/run_1/eval_tierA.json
```
- Evaluate Tier B (derived nodes/edges):
```
python tools/evaluate.py \
  --gold_dir datasets/archive/gold_human_tierb \
  --pred_dir logs/baseline_runs/run_1/tierb \
  --tier B \
  --out_file logs/baseline_runs/run_1/eval_tierB.json
```

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
