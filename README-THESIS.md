# Thesis Experiment Pipeline (Docker + llama.cpp on Apple Silicon)

This document walks through preparing the models, running all three chunking strategies inside Docker (Linux) on an Apple Silicon machine, and collecting thesis-ready results for the three benchmark documents. Docker isolates the workload from macOS mutex issues while `llama.cpp` with the `GPU_BACKEND=metal` setting drives M4 performance.

## 1. Download and Stage Models

1. Create the model folders that are mounted into the containers:
   ```bash
   mkdir -p models/llm models/embeddings models/audio
   ```
2. Download checkpoints:
   - **LLM**: `mistral-7b-instruct-v0.2.Q4_K_M.gguf` → `models/llm/`
   - **Embeddings**: `all-mpnet-base-v2` (or your chosen encoder) → `models/embeddings/`
   - **Audio / Whisper (optional)**: place extra models under `models/audio/`
3. Keep these large assets out of Git. They are mounted read-only by `docker-compose`.

## 2. Build Containers with Metal-Friendly llama.cpp

1. Copy your thesis PDFs into `data/` (defaults: `3M_OEM_SOP.pdf`, `DOA_Food_Proc.pdf`, `op_firesafety_guideline.pdf`).
2. Build the multi-stage image (Python 3.10, spaCy pinned, llama-cpp-python included):
   ```bash
   docker compose build
   ```
3. Launch all three services with Metal acceleration enabled through environment variables:
   ```bash
   docker compose up -d ipke-fixed ipke-semantic ipke-dsc
   docker compose ps
   ```
   Each service listens on a unique host port:
   - `ipke-fixed`: `http://localhost:8000`
   - `ipke-semantic`: `http://localhost:8001`
   - `ipke-dsc`: `http://localhost:8002`

The compose file mounts:

| Host Folder | Container Path | Purpose |
|-------------|----------------|---------|
| `./models`  | `/app/models`  | LLM + embedding checkpoints (read-only) |
| `./data`    | `/app/data`    | Input PDFs for experiments |
| `./results` | `/app/results` | Saved JSON outputs and summaries |

`GPU_BACKEND=metal`, `ENABLE_GPU=true`, and `LLM_GPU_LAYERS=-1` ensure the llama.cpp backend is selected while still allowing CPU fallback if Metal is unavailable.

## 3. Run All Thesis Experiments

Use the automation script to send every document to every chunking service:

```bash
python scripts/run_thesis_experiments.py \
  --host http://localhost \
  --output-dir results
```

The script streams each PDF to the correct FastAPI container, saves the full response to `results/<document>_<method>.json`, and writes `results/thesis_summary.json` with aggregated metrics (confidence, chunk count, average chunk size/cohesion, processing time, etc.).

Flags:
- `--skip-existing` to avoid overwriting completed runs
- `--timeout` to tune request limits (default 15 minutes)
- `--fixed-port/--semantic-port/--dsc-port` if you remap ports

## 4. Analyze the Outputs

Each JSON file contains:
- `steps`, `constraints`, and `entities` from the LLM
- `quality_metrics` including `chunk_count`, `avg_chunk_size`, `avg_chunk_cohesion`, and `avg_sentences_per_chunk`
- `confidence_score` plus API-side `processing_time`

Recommended workflow:
1. Inspect `results/thesis_summary.json` for a quick comparison.
2. Load individual JSON files into your analysis notebooks or visualization tools.
3. Feed the outputs into `tools/evaluate.py` if you need quantitative scoring.

## 5. Comparison Table (fill after running)

| Document | Method | Result JSON | Confidence | Chunk Count | Processing Time (s) |
|----------|--------|-------------|------------|-------------|---------------------|
| 3M_OEM_SOP.pdf | fixed | `results/3M_OEM_SOP_fixed.json` | _(fill from JSON)_ | _(fill)_ | _(fill)_ |
| 3M_OEM_SOP.pdf | breakpoint_semantic | `results/3M_OEM_SOP_breakpoint_semantic.json` | _(fill)_ | _(fill)_ | _(fill)_ |
| 3M_OEM_SOP.pdf | dsc | `results/3M_OEM_SOP_dsc.json` | _(fill)_ | _(fill)_ | _(fill)_ |
| DOA_Food_Proc.pdf | fixed | `results/DOA_Food_Proc_fixed.json` | _(fill)_ | _(fill)_ | _(fill)_ |
| DOA_Food_Proc.pdf | breakpoint_semantic | `results/DOA_Food_Proc_breakpoint_semantic.json` | _(fill)_ | _(fill)_ | _(fill)_ |
| DOA_Food_Proc.pdf | dsc | `results/DOA_Food_Proc_dsc.json` | _(fill)_ | _(fill)_ | _(fill)_ |
| op_firesafety_guideline.pdf | fixed | `results/op_firesafety_guideline_fixed.json` | _(fill)_ | _(fill)_ | _(fill)_ |
| op_firesafety_guideline.pdf | breakpoint_semantic | `results/op_firesafety_guideline_breakpoint_semantic.json` | _(fill)_ | _(fill)_ | _(fill)_ |
| op_firesafety_guideline.pdf | dsc | `results/op_firesafety_guideline_dsc.json` | _(fill)_ | _(fill)_ | _(fill)_ |

Update the table once the experiments finish to capture the metrics that will appear in the final thesis chapter.
