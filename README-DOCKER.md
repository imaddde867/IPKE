# Docker & Compose Guide

This guide covers how to prepare model assets, mount persistent volumes, and run each chunking strategy with the provided `Dockerfile` and `docker-compose.yml`. Follow it before collecting metrics for thesis experiments.

## 1. Download & Stage Models

1. Create the expected directory skeleton locally (mirrors `/app/models` in the container):
   ```bash
   mkdir -p models/llm models/embeddings
   ```
2. Download the required weights into those folders:
   - LLM: place `mistral-7b-instruct-v0.2.Q4_K_M.gguf` under `models/llm/`.
   - Embeddings: place `all-mpnet-base-v2` (or your chosen encoder) under `models/embeddings/`.
   - Whisper/audio extras can live under `models/audio/` if needed.
3. Keep these assets out of Git—only mount them at runtime.
4. If you are running chunking-only workflows (no LLM), set `SKIP_MODEL_CHECK=true` so the server boots without the GGUF file.

## 2. Volume Mount Strategy

`docker-compose.yml` mounts three host directories into every service:

| Host Path  | Container Path | Purpose                                  |
|------------|----------------|------------------------------------------|
| `./models` | `/app/models`  | LLM, embedding, and Whisper checkpoints  |
| `./data`   | `/app/data`    | Uploaded documents / experiment batches  |
| `./results`| `/app/results` | Extraction outputs and evaluation traces |

This layout keeps large assets and generated artifacts persistent across container rebuilds while the image itself remains lightweight. Ensure all three paths exist locally before running compose.

## 3. Run Each Chunking Method

Build the image once, then spin up the specific service you need:

```bash
docker compose build

# Fixed window chunking (SKIP_MODEL_CHECK=true by default)
docker compose up ipke-fixed

# Breakpoint semantic chunking
docker compose up ipke-semantic

# DSC chunking
docker compose up ipke-dsc
```

Each service exposes a different host port (8000/8001/8002) but serves the FastAPI app internally on 8000. Health checks hit `/health`, so `docker compose ps` shows status quickly. Override or extend environment variables in `docker-compose.yml` if you need custom chunk sizes or embedding paths.

## 4. Thesis Experiment Workflow

1. **Preflight**: After downloading models, run `python scripts/baseline_preflight.py` locally (optional but recommended) to confirm asset availability before building images.
2. **Build & Launch**: Use the compose commands above to start the desired chunking mode. Confirm health via `curl localhost:PORT/health`.
3. **Document Ingestion**: Drop anonymized evaluation documents into `data/`. The containers see them under `/app/data`, which is what the ingestion processors expect.
4. **Extraction & Logging**: Results written by the API land in `results/` (host) so you can version, compare, or feed them into `tools/evaluate.py`.
5. **Batch Experiments**: Automate runs with `scripts/run_baseline_loops.py --runs N`, pointing the script at the specific service endpoint (e.g., `http://localhost:8001` for semantic chunking). Capture metrics under `results/` for thesis reporting.

By keeping models mounted and chunking configs isolated per service, you can iterate on extraction strategies without rebuilding large images or copying GB-scale checkpoints into the container layers.
