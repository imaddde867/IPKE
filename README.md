# IPKE - Industrial Procedural Knowledge Extraction

**Research direction:** method paper on skeleton-conditioned, source-grounded constraint
attachment for procedural graph extraction with local language models. See
[ADR-0005](docs/adr/0005-ipke-method-paper-primary.md) and the
[approved method design](docs/superpowers/specs/2026-07-10-ipke-method-paper-design.md).

IPKE extracts Procedural Knowledge Graphs from safety-critical technical documents. Its
central method separates the procedural step skeleton from typed constraint extraction,
then grounds and attaches each constraint to the step or steps it governs. The corpus,
taxonomy, validators, and metrics are supporting evaluation infrastructure, not the
paper's primary contribution.

## Research contributions under test

1. **Skeleton-conditioned constraint attachment** - compare step-conditioned generation
   against schema-, parser-, call-, and budget-matched alternatives.
2. **Constraint-preserving segmentation** - test whether hierarchy-aware segmentation
   keeps constraint-step pairs in context and improves attachment. This remains secondary
   until the controlled result exists.
3. **Local quality-cost analysis** - report attachment quality together with calls,
   tokens, latency, and memory across local model families.
4. **Evaluation infrastructure** - manually verified procedural annotations, explicit
   relation metrics, provenance-complete runs, and reproducible component metrics.

## Method Kernel

- **Dual Semantic Chunker (DSC)** — global DP objective over heading-aligned embeddings: `J(B) = Σ H(b) − λ|B|` with heading bonus β·𝟙[j is heading].
- **P3 Two-Stage Decomposition** — decouples step extraction (Stage 1) from constraint attachment (Stage 2, with mandatory step-ID back-reference). Reduces schema drift in mid-size models.

## Evidence status

The workspace retains 8 legacy candidates with 256 steps and 231 constraints; the
provisional manifest selects 5 for development. It has 0 production annotations. The
candidate schema is aligned, and the paper gate now rejects marker-only sign-off by
requiring exact item anchors, canonical artifact paths, verified hashes, complete item
decisions, and frozen evidence packages. The first agent-prepared EPA packet removes
most transcription but still leaves eight decisions to a primary human. Human primary
passes, a frozen manifest, blind coverage, and exact-span experiment inputs remain open. See
[BENCHMARK.md](BENCHMARK.md) for the supporting-data audit.

## Reproducible Commands

```bash
uv sync
make test              # unit tests
make gold-pipeline     # strict gold validation + D1 blindness regeneration (fresh-clone gate)
make repro-blindness   # asserts the pinned D1 numbers (32 vs 231, 7.22x)
make smoke-extract
make eval              # validator + blindness + multiseed dry-run plan
```

Experiment artifacts are written under `runs/` and are intentionally ignored by git.

## Direction

- **[Research vision](docs/research-vision.md)** - method thesis and publication bar.
- [Approved method design](docs/superpowers/specs/2026-07-10-ipke-method-paper-design.md) - causal protocol and evidence gates.
- [Execution direction](docs/paper/2026-07-04-execution-direction.md) - current issue board and work order.
- [Superseded resource PRD](docs/paper/ipke-bench-resource-prd.md) - retained as historical and supporting-infrastructure context.

## Research Reproducibility

- [Reproducibility guide](REPRODUCIBILITY.md)
- [Annotation methodology](docs/methods/annotation-pipeline.md)
- [Annotation guidelines](docs/annotation/guidelines.md) · [constraint taxonomy](docs/annotation/constraint-types.md)
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

Turku University of Applied Sciences · 2025–2026

## Local LLM (Mistral 7B, GGUF)

- Download weights (requires a Hugging Face token):

  ```bash
  python - <<'PY'
  from huggingface_hub import hf_hub_download
  hf_hub_download(
      repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
      filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
      local_dir="models/llm",
  )
  PY
  ```

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
