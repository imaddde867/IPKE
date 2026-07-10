# Reproducibility

This file separates historical/development diagnostics from confirmatory IPKE
method-paper evidence. `make eval` and `make gold-pipeline` reproduce the legacy D1
diagnostics and dry-run the superseded D2 sweep plan; neither establishes paper
eligibility. Before a confirmatory sweep begins, the open P0 gates are an explicit
inclusion manifest, 12 eligible human-verified procedures, eligible independent
second-pass annotations, corrected and frozen causal controls, and completed runs of
the eventual canonical experiment design. The human-verification criterion is exposed
as `make eval-paper-gate`.

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
| Historical development seeds | 0, 1, 2, 3, 4 (N=5) |
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
| Historical development sweep (5 seeds x 8 candidate files) | ~3-4 hours (Metal) |

---

## Development diagnostics and paper-evidence gate

The current eight-file directory is a historical/development candidate corpus, not a
confirmatory test set. It includes artifacts that the July 11 manual audits exclude
pending issue #112 or a source-faithful rebuild. The commands below remain useful for
structural checks and legacy diagnostics, but their outputs are not current paper
evidence.

### 1. Run default test suite

```bash
uv run pytest
```

Expected: all non-integration tests pass. No GPU or model required.

### 2. Validate gold dataset

- `make eval-validate`: structural and annotation-contract validation.
- `make eval-paper-gate`: structural validation plus explicit human verification; this
  intentionally fails until the human sign-off issue is complete.

Run structural validation during development:

```bash
make eval-validate
```

Before using the corpus as paper evidence, run the release gate:

```bash
make eval-paper-gate
```

`make eval-validate` can pass while the corpus is unsigned. The paper-evidence
gate currently exits non-zero and names unsigned files in the current gold
directory; that is expected until the confirmatory inclusion manifest and
`PAPER_GOLD` agree and a human verifies every included gold. Do not sign an
excluded artifact merely to make this directory-wide gate pass.

### 3. Check IAA eligibility

The current `datasets/paper/second_pass` files are draft placeholders and are not
eligible for reported IAA. The command below should fail closed until independent,
reviewed second-pass annotations replace them.

```bash
uv run python scripts/compute_iaa.py \
  --gold-dir datasets/paper/gold \
  --second-dir datasets/paper/second_pass \
  --out results/iaa_reproduced.json
```

Expected today: non-zero exit with `IAA eligibility failed`.

### 4. Historical 5x8 development sweep template (not confirmatory)

Requires a model file. Set `LLM_MODEL_PATH` before running. These legacy commands
process the entire current candidate directory, including excluded artifacts, and do
not implement the corrected causal controls. `make eval-full` now depends on
`eval-paper-gate`, so it cannot begin while the directory is unsigned. Even after
human sign-off, this 5x8 configuration does not become confirmatory: issue #112 must
define the inclusion manifest and the causal-control design must be corrected and
frozen first.

```bash
# Development configuration: DSC + P3, 5 seeds, current gold directory
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

### 5. Compute bootstrap significance for historical development outputs

After running the step 4 development ablations, compare the legacy full-system and
baseline outputs:

```bash
uv run python scripts/eval_multiseed.py \
  --gold-dir datasets/paper/gold \
  --text-dir datasets/paper/text \
  --seeds 5 \
  --out-dir results/paper_run_compare/ \
  --compare-against results/ablation_baseline/results_detail_<timestamp>.csv \
  --bootstrap-n 10000
```

The `results_summary_<ts>.csv` file contains `Phi_pvalue` - the paired-bootstrap
p-value for H0: mean Phi difference == 0. Treat it as a development diagnostic only;
do not report it as confirmatory evidence from the current method-paper design.

### 6. Also evaluate on thesis gold (for continuity with thesis numbers)

```bash
uv run python scripts/eval_multiseed.py \
  --gold-dir datasets/archive/gold_human \
  --text-dir datasets/archive/test_data/text \
  --seeds 5 \
  --out-dir results/thesis_gold_run/
```

---

## Historical Phi weight sensitivity

The legacy runner computes Phi under three weighting schemes to inspect ranking
stability during development:

| Scheme | Coverage | StepF1 | Kendall |
|---|---|---|---|
| Default (paper primary) | 0.5 | 0.3 | 0.2 |
| Step-heavy | 0.4 | 0.4 | 0.2 |
| Coverage-heavy | 0.6 | 0.2 | 0.2 |

These are computed automatically when `--phi-weights` is passed to `eval_multiseed.py`.
The `results_summary_*.csv` contains one column per scheme. Stability across these
weights is not a substitute for the corrected confirmatory causal-control design.

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

### Confirmatory procedure corpus (method-paper requirement)
- [ ] Explicit confirmatory inclusion manifest committed and used by experiment commands
- [ ] Every gold in the inclusion manifest explicitly human-verified; `make eval-paper-gate` exits 0 against the matching active gold set
- [ ] 12 eligible procedures included; the NASA NPR 8715.3D requirements stress test does not count
- [ ] IAA on ≥ 30% of docs (≥ 4 files), all κ ≥ 0.61 (substantial, Landis & Koch 1977)
- [ ] OLSK re-annotated (κ = 0.531 in current draft, below threshold)
- [ ] Dataset datasheet (metadata, license, collection process, limitations) committed
- [ ] Annotation guidelines committed as `docs/annotation/guidelines.md`
- [ ] JSON-LD / schema.org export example included for IR community

### Experiments
- [ ] Corrected causal controls frozen before any headline sweep
- [ ] Canonical confirmatory runner consumes the inclusion manifest
- [ ] Historical `make eval` dry-run passes on fresh clone ✓
- [ ] Multi-seed sweep completed (N=5 seeds, n_docs=12)
- [ ] Ablation table: 4 configurations × 5 seeds
- [ ] Bootstrap p-value computed for main comparison (`--compare-against`)
- [ ] Phi sensitivity plan updated for the canonical confirmatory runner
- [ ] Constraint-type breakdown table (guard / parameter / precondition / postcondition)
- [ ] Text-RAG vs PKG-backed retrieval comparison (constraint-type query recall)

### Metadata
- [ ] Model, quantization, temperature, seed, hardware listed in §4
- [ ] Dataset released under CC-BY (or embargoed with rationale for private-source docs)
- [ ] Artifact availability requirements confirmed for the target method-paper venue
