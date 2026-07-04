# IPKE — Industrial Procedural Knowledge Extraction

**Target:** ECIR 2027 Resource Paper — *IPKE-Bench: A Constraint-Aware Benchmark for Procedural Knowledge Extraction from Safety-Critical Industrial Documents*

A local, privacy-preserving pipeline and benchmark for extracting structured Procedural Knowledge Graphs (PKGs) from industrial safety procedures. IPKE makes constraint attachment a first-class evaluation task — prior benchmarks measure step coverage and graph structure but leave constraint-to-step linking unmeasured. The accompanying dataset (IPKE-Bench) is the primary contribution; IPKE provides the reproducible local baseline.

## Primary Contributions (ECIR Resource Track)

1. **IPKE-Bench dataset** — 12–15 publicly licensed industrial/safety-procedure documents with human-reviewed step, constraint, and attachment annotations (κ ≥ 0.61 for all IAA pairs).
2. **Constraint Attachment Evaluation** — fuzzy semantic alignment metric (cosine ≥ 0.75) and strict exact-match for constraint-to-step edges. Not measured in PAGED, CAMB, or KEO.
3. **Local Baselines** — P0 (zero-shot fixed), DSC+P3 (two-stage structured), reproducible across 5 seeds with 95% CIs and paired bootstrap.
4. **Constraint-Aware Retrieval Task** — queries targeting specific constraint types (guard, parameter, precondition). Text-chunk RAG vs PKG-backed retrieval comparison.
5. **Reproducibility Package** — dataset datasheet, annotation guidelines, one-command evaluator, loader, and re-run harness.

## Method Kernel

- **Dual Semantic Chunker (DSC)** — global DP objective over heading-aligned embeddings: `J(B) = Σ H(b) − λ|B|` with heading bonus β·𝟙[j is heading].
- **P3 Two-Stage Decomposition** — decouples step extraction (Stage 1) from constraint attachment (Stage 2, with mandatory step-ID back-reference). Reduces schema drift in mid-size models.

## Reproducible Commands

```bash
uv sync
make test
make smoke-extract
make eval
```

Experiment artifacts are written under `runs/` and are intentionally ignored by git.

## Direction

- **[Research Vision — the North Star](docs/research-vision.md)** — the one-sentence thesis, what "elite" means concretely, the venue ladder toward a top-tier publication, and the principles to carry. Read this first.
- [Execution direction](docs/paper/2026-07-04-execution-direction.md) — current issue board and work order.
- [Resource-track PRD](docs/paper/ipke-bench-resource-prd.md) — requirements.

## Research Reproducibility

- [Reproducibility guide](docs/reproducibility.md)
- [Annotation methodology](docs/methods/annotation-pipeline.md)
- [Implemented DSC method note](docs/methods/dsc-implementation.md)
- [Paper dataset workspace](datasets/paper/README.md)

## Run IPKE

Use extras as needed:

```bash
uv sync --extra llm
uv sync --extra app
uv sync --extra extras
uv sync --extra neo4j
```

```ini
# .env
GPU_BACKEND=metal
CHUNKING_METHOD=dual_semantic
PROMPTING_STRATEGY=P3
# Deduplicate overlapping content and enforce clean constraints/steps
ENABLE_CHUNK_DEDUP=true
```

```bash
# Reproduce chunking experiments
uv run python scripts/experiments/run_all_chunking_experiments.py \
  --documents datasets/archive/test_data/text/*.txt

# API surface (requires the `app` extra)
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000  # http://localhost:8000/docs
```

Research distribution for academic and regulated industrial settings. See `LICENSE`.

---

Turku University of Applied Sciences · 2025

## Local LLM (Mistral 7B, GGUF)

- Download weights (requires a Hugging Face token):  
  `python - <<'PY'\nfrom huggingface_hub import hf_hub_download\nhf_hub_download(\n  repo_id=\"TheBloke/Mistral-7B-Instruct-v0.2-GGUF\",\n  filename=\"mistral-7b-instruct-v0.2.Q4_K_M.gguf\",\n  local_dir=\"models/llm\",\n  local_dir_use_symlinks=False,\n)\nPY`

- Metal (Apple silicon, fastest locally):  
  `uv sync --extra llm --index-url https://abetlen.github.io/llama-cpp-python/whl/metal`  
  Test: `bash scripts/test_mistral_metal.sh`

- CUDA (Linux x86_64, NVIDIA):  
  `uv sync --extra llm` installs the pinned CUDA 12.4 `llama-cpp-python` wheel from `pyproject.toml`.  
  Test: `bash scripts/test_mistral_cuda.sh`

The app will pick up the GGUF at `models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf`; set `LLM_N_GPU_LAYERS=-1` to offload all layers to Metal/CUDA.

## Hardware Compatibility

IPKE auto-detects your hardware and gracefully falls back to CPU if GPU acceleration is unavailable:

| Hardware | Configuration | Notes |
|----------|--------------|-------|
| **NVIDIA GPU** | `GPU_BACKEND=cuda` | Auto-detected if CUDA available |
| **Apple Silicon** | `GPU_BACKEND=metal` | Auto-detected on macOS with MPS |
| **CPU only** | `GPU_BACKEND=cpu` | Default fallback, no GPU required |

The system uses `torch.cuda.is_available()` and `torch.backends.mps.is_available()` with try/except guards to ensure safe operation on any platform.

See [`docs/notes/hardware-validation-rtx5060-cuda132.md`](docs/notes/hardware-validation-rtx5060-cuda132.md)
for the 2026-05-27 Blackwell/CUDA 13.2 validation note and its evaluation caveats.
