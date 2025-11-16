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

## 5. Running on Google Cloud GPU VMs

High-memory NVIDIA GPUs on Google Compute Engine dramatically cut extraction latency compared to local CPU-only runs. The repo now ships with a Compose overlay that wires in CUDA defaults and requests GPUs from Docker:

1. Provision a Compute Engine VM with an NVIDIA GPU (e.g., L4, T4, or A100) and install the [NVIDIA drivers + Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Confirm `nvidia-smi` works before continuing.
2. Copy your `models/`, `data/`, and `results/` folders to the VM (or mount a GCS bucket) exactly as described earlier.
3. Launch the GPU-tuned stack with the overlay file:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gcp.yml up -d ipke-fixed
   ```
   Swap `ipke-fixed` for `ipke-semantic`/`ipke-dsc` as needed. The overlay injects `EXPLAINIUM_ENV=cloud`, enables CUDA in the FastAPI app, and issues a GPU device request so Docker binds the accelerator into each container.
4. (Optional) Override specific knobs—`LLM_MODEL_ID`, `LLM_MAX_TOKENS`, chunking bounds, etc.—via `ENV=... docker compose ...` to fine-tune for your VM size. The new `cloud` configuration preset automatically picks sane CUDA defaults whenever `GOOGLE_CLOUD_PROJECT` is present or `EXPLAINIUM_ENV=cloud`.

With GPUs attached you can keep `LLM_CPU_MAX_TOKENS_CAP` disabled (set to `0`) for full-quality generations while still reusing the same experiment scripts described above.
