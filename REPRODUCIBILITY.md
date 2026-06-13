# Reproducibility

This file documents how to regenerate every table in the paper from scratch on a fresh
clone. It is a reviewer expectation at ECIR/ESWC and should be updated whenever the
experimental setup changes.

---

## Hardware requirements

| Component | Minimum | Used in paper |
|---|---|---|
| RAM | 16 GB | 16 GB |
| VRAM (GPU path) | 8 GB (Q4 model) | Metal / CUDA |
| CPU (CPU-only path) | 8 cores | Apple M-series or x86-64 |
| Disk | 10 GB (models + data) | ~15 GB with FAA alternates |

The default backend is `llama_cpp` with Metal (Apple Silicon) or CUDA (NVIDIA).
All metric computation and dataset tooling runs CPU-only.

---

## Software environment

Python 3.12+. Package manager: `uv` only. Do not use pip or raw venv.

```bash
git clone https://github.com/imaddde867/IPKE.git
cd IPKE
uv sync --extra extras
uv run python -m spacy download en_core_web_sm
```

Install the llama-cpp backend for your hardware:

```bash
# Apple Metal
uv pip install llama-cpp-python \
  --index-url https://abetlen.github.io/llama-cpp-python/whl/metal \
  --force-reinstall

# NVIDIA CUDA 12.1
uv pip install llama-cpp-python \
  --index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
  --force-reinstall

# CPU-only (slow, for testing)
uv pip install llama-cpp-python --force-reinstall
```

---

## Model

| Setting | Value |
|---|---|
| Model | mistralai/Mistral-7B-Instruct-v0.2 |
| Quantization | Q4_K_M (GGUF) |
| Backend | llama_cpp |
| Context length | 4096 tokens |
| Temperature | 0.1 |
| Default seed | 42 |
| Seeds used for paper | 0, 1, 2, 3, 4 (N=5) |
| Chunking method | dual_semantic (DSC) |
| Prompting strategy | P3 (two-stage) |

Download the GGUF model:

```bash
# Place at: models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
# Source: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
```

Set the model path:

```bash
export LLM_MODEL_PATH=models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

---

## Estimated runtimes (per document, Apple M2 Pro)

| Configuration | Runtime per doc |
|---|---|
| DSC + P3 (Metal, Q4_K_M) | ~2-4 min |
| DSC + P3 (CPU-only, Q4_K_M) | ~15-30 min |
| Metric evaluation (CPU) | < 5 sec per doc |
| Full multi-seed sweep (5 seeds x 8 docs) | ~3-4 hours (Metal) |

---

## Step-by-step: regenerate all tables

### 1. Run default test suite

```bash
uv run pytest
```

Expected: all non-integration tests pass. No GPU or model required.

### 2. Validate gold dataset

```bash
uv run python -c "
import json, jsonschema, glob
schema = json.load(open('schemas/ipke_annotation.schema.json'))
for f in glob.glob('datasets/paper/gold/*.json'):
    jsonschema.validate(json.load(open(f)), schema)
    print('PASS', f.split('/')[-1])
"
```

### 3. Reproduce IAA report (Table: annotator agreement)

```bash
uv run python scripts/compute_iaa.py \
  --gold-dir datasets/paper/gold \
  --second-dir datasets/paper/second_pass \
  --out results/iaa_reproduced.json

# Compare with committed report:
diff <(python3 -c "import json; print(json.dumps(json.load(open('datasets/paper/reports/issue_53_iaa_report.json')), sort_keys=True, indent=2))") \
     <(python3 -c "import json; print(json.dumps(json.load(open('results/iaa_reproduced.json')), sort_keys=True, indent=2))")
```

Expected: identical output to `datasets/paper/reports/issue_53_iaa_report.json`.

### 4. Run multi-seed extraction sweep (main results table)

Requires model file. Set `LLM_MODEL_PATH` before running.

```bash
# Paper configuration: DSC + P3, 5 seeds, all 8 paper docs
uv run python scripts/eval_multiseed.py \
  --gold-dir datasets/paper/gold \
  --text-dir datasets/paper/text \
  --seeds 5 \
  --phi-weights 0.5:0.3:0.2 \
  --phi-weights 0.4:0.4:0.2 \
  --phi-weights 0.6:0.2:0.2 \
  --out-dir results/paper_run/

# Ablation: fixed chunker + P3
uv run python scripts/eval_multiseed.py \
  --gold-dir datasets/paper/gold \
  --text-dir datasets/paper/text \
  --seeds 5 \
  --chunker fixed \
  --prompter P3 \
  --out-dir results/ablation_fixed_p3/

# Ablation: DSC + zero-shot (P0)
uv run python scripts/eval_multiseed.py \
  --gold-dir datasets/paper/gold \
  --text-dir datasets/paper/text \
  --seeds 5 \
  --chunker dsc \
  --prompter P0 \
  --out-dir results/ablation_dsc_p0/

# Ablation: fixed + zero-shot (baseline)
uv run python scripts/eval_multiseed.py \
  --gold-dir datasets/paper/gold \
  --text-dir datasets/paper/text \
  --seeds 5 \
  --chunker fixed \
  --prompter P0 \
  --out-dir results/ablation_baseline/
```

### 5. Compute bootstrap significance

After running steps 4 ablations, compare full system vs baseline:

```bash
uv run python scripts/eval_multiseed.py \
  --gold-dir datasets/paper/gold \
  --text-dir datasets/paper/text \
  --seeds 5 \
  --out-dir results/paper_run_compare/ \
  --compare-against results/ablation_baseline/results_detail_<timestamp>.csv \
  --bootstrap-n 10000
```

The `results_summary_<ts>.csv` file contains `Phi_pvalue` — this is the p-value for
the paired bootstrap test (H0: mean Phi difference == 0). Report this value in the
paper next to the main DSC+P3 vs baseline comparison. p < 0.05 is the minimum;
p < 0.01 is preferred.

### 6. Also evaluate on thesis gold (for continuity with thesis numbers)

```bash
uv run python scripts/eval_multiseed.py \
  --gold-dir datasets/archive/gold_human \
  --text-dir datasets/archive/test_data/text \
  --seeds 5 \
  --out-dir results/thesis_gold_run/
```

---

## Phi weight sensitivity

The paper reports Phi under three weighting schemes to verify ranking stability:

| Scheme | Coverage | StepF1 | Kendall |
|---|---|---|---|
| Default (paper primary) | 0.5 | 0.3 | 0.2 |
| Step-heavy | 0.4 | 0.4 | 0.2 |
| Coverage-heavy | 0.6 | 0.2 | 0.2 |

These are computed automatically when `--phi-weights` is passed to `eval_multiseed.py`.
The `results_summary_*.csv` contains one column per scheme. If the DSC+P3 ranking is
stable across all three, report that. If not, it is a metric validity problem to
disclose.

---

## Single-document annotation draft

To draft a new gold file for a text document (requires configured model):

```bash
uv run python tools/annotate_gold.py datasets/paper/text/<doc>.txt \
  --doc-id <doc_id> \
  --domain <domain>
# Output: datasets/paper/gold_drafts/<doc_id>.json
# Validate only (no extraction):
uv run python tools/annotate_gold.py datasets/paper/gold/<doc>.json --skip-model
```

---

## IAA check for a document pair

```bash
uv run python tools/iaa_check.py \
  datasets/paper/gold/<doc>.json \
  datasets/paper/second_pass/<doc>.json
# Exit 0 = PASS (token kappa >= 0.7), Exit 1 = FAIL
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_PATH` | (required for GGUF) | Path to .gguf model file |
| `LLM_RANDOM_SEED` | 42 | RNG seed for inference |
| `LLM_TEMPERATURE` | 0.1 | Inference temperature |
| `CHUNKING_METHOD` | dual_semantic | Chunking strategy |
| `PROMPTING_STRATEGY` | P3 | Prompting strategy (P0/P1/P2/P3) |
| `GPU_BACKEND` | auto | metal / cuda / cpu |
| `EXPLAINIUM_ENV` | development | testing = lightweight mocks |

---

## Checklist before paper submission

### Benchmark / Dataset (resource track requirement)
- [ ] All 8 existing gold files human-reviewed (`quality.review_status == 'reviewed'`)
- [ ] 4 additional documents annotated to reach 12 total (ECIR resource minimum)
- [ ] IAA on ≥ 30% of docs (≥ 4 files), all κ ≥ 0.61 (substantial, Landis & Koch 1977)
- [ ] OLSK re-annotated (κ = 0.531 in current draft, below threshold)
- [ ] Dataset datasheet (metadata, license, collection process, limitations) committed
- [ ] Annotation guidelines committed as `docs/annotation/guidelines.md`
- [ ] JSON-LD / schema.org export example included for IR community

### Experiments
- [ ] `make eval` dry-run passes on fresh clone ✓
- [ ] Multi-seed sweep completed (N=5 seeds, n_docs=12)
- [ ] Ablation table: 4 configurations × 5 seeds
- [ ] Bootstrap p-value computed for main comparison (`--compare-against`)
- [ ] Phi sensitivity table (3 weight schemes, auto-generated by `make eval-full`)
- [ ] Constraint-type breakdown table (guard / parameter / precondition / postcondition)
- [ ] Text-RAG vs PKG-backed retrieval comparison (constraint-type query recall)

### Metadata
- [ ] Model, quantization, temperature, seed, hardware listed in §4
- [ ] Dataset released under CC-BY (or embargoed with rationale for private-source docs)
- [ ] Artifact availability badge confirmed (ECIR resource track: Available + Functional)
