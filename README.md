# IPKE — Industrial Procedural Knowledge Extraction

IPKE extracts **Procedural Knowledge Graphs (PKGs)** from safety-critical industrial and
regulatory documents using local language models. A PKG represents a procedure as ordered
steps plus typed constraints — preconditions, guards, parameters, warnings — explicitly
attached to the steps they govern.

Everything runs locally on Metal or CUDA; no cloud APIs. IPKE is built for
privacy-preserving use in academic and regulated industrial settings.

## Method

- **Dual Semantic Chunker (DSC)** — global dynamic-programming segmentation over
  heading-aligned embeddings: `J(B) = Σ H(b) − λ|B|`, with a heading bonus
  `β·𝟙[j is heading]`. See
  [docs/methods/dsc-implementation.md](docs/methods/dsc-implementation.md) and
  [docs/adr/0001-implement-dsc-global-dp.md](docs/adr/0001-implement-dsc-global-dp.md).
- **P3 Two-Stage Prompting** — step extraction (Stage 1) is decoupled from constraint
  attachment (Stage 2), where every constraint must carry a step-ID back-reference.
  The decomposition reduces schema drift in mid-size models.
- **Procedural Fidelity Score (Φ)** — composite metric:
  `0.5·ConstraintCoverage + 0.3·StepF1 + 0.2·Kendall τ`, reported alongside AdjacencyF1.

## Evaluation Corpus

Eight rights-cleared procedural documents across five source families (US EPA, NASA,
NIOSH, USGS, and the open-hardware Open Lab Starter Kit): US federal public-domain works
plus CC BY-SA material. Each document carries source-grounded annotations with exact
offsets for steps, typed constraints, and attachments under a locked taxonomy of six
constraint types × three enforcement levels (`must` / `should` / `may`).

- Datasheet: [docs/dataset/datasheet.md](docs/dataset/datasheet.md) · Provenance:
  `datasets/paper/public_sources_manifest.csv` (per-document license + SHA-256)
- Annotation guidelines:
  [docs/annotation/guidelines.md](docs/annotation/guidelines.md) · Taxonomy:
  [docs/annotation/constraint-types.md](docs/annotation/constraint-types.md)

### Motivating result

Across the seed corpus, an LLM-drafted annotation pass produced **7.22× fewer
constraints** than the reviewed gold (32 vs 231); at the Tier-A semantic matcher
(SBERT cosine ≥ 0.75) it recovered only **6.1%** of reviewed constraints (**37.7%** at
cosine ≥ 0.50). Draft and gold come from different annotation regimes, so this is
annotation-economics evidence about naive LLM extraction, not an extractor-quality
claim. Reproduce with `make eval-blindness`; see [BENCHMARK.md](BENCHMARK.md).

## Quickstart

```bash
uv sync --extra llm        # local LLM backend (other extras: app, extras, neo4j)
make test                  # unit tests
make smoke-extract         # end-to-end smoke run

# Extract a PKG from a procedural document
uv run python scripts/run_pkg_extraction.py \
  --input-path datasets/paper/text/usgs_groundwater_technical_procedures_tm1_a1.txt \
  --chunking-method dsc \
  --prompting-strategy P3
```

```ini
# .env
GPU_BACKEND=metal            # cuda | metal | cpu (auto-detected fallback)
CHUNKING_METHOD=dual_semantic
PROMPTING_STRATEGY=P3
ENABLE_CHUNK_DEDUP=true
```

## Repository Layout

```text
src/
  ai/                   # chunker -> prompting strategy -> graph orchestration, LLM backends
  processors/chunkers/  # DSC plus fixed-size and semantic-breakpoint ablations
  evaluation/           # Phi, StepF1, AdjacencyF1, Kendall, ConstraintCoverage, Smatch
  graph/                # Pydantic PKG models, builder, optional Neo4j persistence
  validation/           # schema and constraint validators
datasets/
  paper/                # evaluation corpus: sources, annotations, provenance manifest
  archive/              # seed-corpus gold annotations and source texts
schemas/                # JSON schemas for annotations and evidence packages
scripts/                # experiment runners, validators, reporting
tests/                  # pytest suite
```

## Documentation

| Document | Contents |
|---|---|
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Full reproduction guide |
| [BENCHMARK.md](BENCHMARK.md) | Corpus, taxonomy, quality gates |
| [docs/methods/annotation-pipeline.md](docs/methods/annotation-pipeline.md) | Annotation production pipeline |
| [docs/annotation/independent-annotator-workflow.md](docs/annotation/independent-annotator-workflow.md) | Blind second-pass annotator workflow |
| [docs/paper/related-work.md](docs/paper/related-work.md) | Positioning against prior benchmarks |
| [docs/notes/hardware-validation-rtx5060-cuda132.md](docs/notes/hardware-validation-rtx5060-cuda132.md) | Blackwell / CUDA 13.2 validation note |

## Local LLM (Mistral 7B, GGUF)

Download weights (requires a Hugging Face token):

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
- CUDA (Linux x86_64, NVIDIA):
  `uv sync --extra llm` installs the pinned CUDA 12.4 `llama-cpp-python` wheel from
  `pyproject.toml`.

The app picks up the GGUF at `models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf`; set
`LLM_N_GPU_LAYERS=-1` to offload all layers to the GPU.

## Hardware Compatibility

| Hardware | Configuration | Notes |
|----------|--------------|-------|
| **NVIDIA GPU** | `GPU_BACKEND=cuda` | Auto-detected if CUDA available |
| **Apple Silicon** | `GPU_BACKEND=metal` | Auto-detected on macOS with MPS |
| **CPU only** | `GPU_BACKEND=cpu` | Default fallback, no GPU required |

## API

With the `app` extra:

```bash
uv sync --extra app
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000   # http://localhost:8000/docs
```

## License & Citation

See [LICENSE](LICENSE). Source documents retain their original licenses; per-document
terms are tracked in `datasets/paper/public_sources_manifest.csv`. Pre-publication,
please cite the repository:

```bibtex
@misc{elmouss2026ipke,
  author = {Elmouss, Imad Eddine},
  title  = {IPKE: Industrial Procedural Knowledge Extraction},
  year   = {2026},
  url    = {https://github.com/imaddde867/IPKE}
}
```

---

Turku University of Applied Sciences · 2025–2026
