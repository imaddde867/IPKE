# CLAUDE.md — IPKE

Read this before doing anything in this repo.

## Who and what

**Imad Eddine Elmouss** — Research Engineer at CoRe (Cognitive Technologies Research Group),
Turku University of Applied Sciences, Finland. This repo is the codebase for his bachelor's
thesis *Structured Procedural Knowledge Extraction from Industrial Documentation Using LLMs*
(Dec 2025, Turku UAS).

**Current goal**: convert the thesis into a 6-page short paper for **ECIR 2027**
(Short Paper Track, deadline **12 October 2026**). The code is not the product — publishable
results are the product. Every task in this repo should map to either running experiments,
improving the evaluation framework, or producing paper-ready outputs.

---

## What the thesis proved (your baseline)

Pipeline: **Dual Semantic Chunking (DSC)** + **P3 Two-Stage Prompting** + Procedural Fidelity
score **Phi**.

Headline result: Mistral-7B-Instruct Q4_K_M (local, llama-cpp) + DSC + P3 reaches
Phi=0.699 and 75% constraint coverage on the 3M Marine SOP. Llama-3.1-70B with naive
zero-shot: Phi=0.439, 50% coverage. A 10x smaller model beats the big one because the
pipeline does the heavy lifting, not scale.

Evaluated on 3 documents only. Single-seed. Constraint-Attachment F1 collapses to 0 due to
strict string ID matching (known metric artefact). No human validation. No comparison against
contemporary baselines (PAGED, etc.). These are the paper's open wounds.

---

## Repo layout

```
src/
  ai/
    prompting/          # Prompt strategies: zero_shot, few_shot, chain_of_thought, two_stage (P3)
    llm_backends.py     # LLM backend abstraction (llama-cpp today; extend for OpenAI-compatible)
    knowledge_engine.py # Orchestrates chunker + strategy + graph builder
  evaluation/
    metrics.py          # Phi, StepF1, AdjacencyF1, Kendall, ConstraintCoverage, ConstraintAttachmentF1
                        # Tier-A (steps+constraints) and Tier-B (full graph, smatch-based)
    smatch.py           # Smatch graph alignment for Tier-B
  processors/
    chunkers/
      dual_semantic.py  # DSC: heading-aware parent segmentation + embedding cohesion sub-chunks
      fixed.py          # Fixed-size chunker (ablation baseline)
      breakpoint.py     # Semantic breakpoint chunker (ablation baseline)
  graph/
    builder.py          # Assembles per-chunk extractions into a full PKG
    models.py           # Pydantic models for PKG nodes/edges
    neo4j_connector.py  # Neo4j persistence (optional, not needed for paper experiments)
  validation/
    schema_validator.py # JSON schema validation on LLM output
scripts/
  experiments/
    run_all_chunking_experiments.py   # Main experiment runner (extend this for paper runs)
    run_chunker_eval.py               # Chunker-only evaluation
    experiment_utils.py               # Shared helpers
    prepare_human_eval_samples.py     # Samples for human expert rating study
  reevaluate_metrics.py               # Re-score cached predictions with updated metrics
  run_pkg_extraction.py               # Run extraction on a single document
datasets/
  archive/
    test_data/text/                   # 3 source documents (plain text)
    test_data/gold/                   # Gold annotations (Tier-A: steps + constraints)
    gold_human/                       # Human-revised gold (Tier-A)
    gold_human_tierb/                 # Human-revised gold (Tier-B: full graph)
  Samples/                            # Raw PDFs for new document intake
```

---

## Dev commands

```bash
# Setup (uv only — do not use pip or venv directly)
uv venv && source .venv/bin/activate
uv sync --extra extras

# For Metal (Apple Silicon M4):
uv sync --extra extras --index-url https://abetlen.github.io/llama-cpp-python/whl/metal

# For CUDA (RTX 3090 / 5090):
uv sync --extra extras --index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Run without downloading the model (code/eval work only):
SKIP_MODEL_CHECK=true uv run python main.py

# Unit tests (fast, no model required):
uv run pytest

# Integration tests (require model + hardware):
uv run pytest -m ""

# Reproduce thesis chunking experiments:
uv run python scripts/experiments/run_all_chunking_experiments.py \
  --documents datasets/archive/test_data/text/*.txt

# Re-score saved predictions after changing metrics.py:
uv run python scripts/reevaluate_metrics.py --run-dir <run_dir>

# Single document extraction:
uv run python scripts/run_pkg_extraction.py --document datasets/archive/test_data/text/3m_marine_oem_sop.txt
```

---

## Hardware map

| Machine | Access | VRAM | Use for |
|---|---|---|---|
| Mac M4 (local) | native | 16 GB unified | dev, fast iteration, Metal llama-cpp |
| RTX 3090 | `ssh core` | 24 GB | primary model inference; 4B-14B comfortable; 4-bit Mistral/Qwen |
| RTX 5090-01 | `ssh 172.24.50.2` | ~32 GB | large model runs, parallel seeds |
| RTX 5090-02 | `ssh 172.24.50.227` | ~32 GB | parallel multi-seed runs |
| CSC Puhti | `ssh puhti.csc.fi` | batch | long batch jobs; submit via SLURM; not always-on |

**Model runtime preference**: LM Studio over Ollama (clearer quant/context visibility). Use
LM Studio's OpenAI-compatible local endpoint (`http://localhost:1234/v1`) so the same
inference code works across machines. Avoid cloud APIs — this is a privacy-preserving pipeline
by design.

**Quantization defaults**: 4-bit (Q4_K_M) for text LLMs on 3090. fp16 acceptable on 5090s
for quality-sensitive runs. Always record exact model version, quantization, and seed in
experiment logs.

---

## Paper experiment gap — what's missing for ECIR

This is the core of all current work. Tasks are prioritized P0 (non-negotiable) to P2
(opportunistic).

### P0 — paper is not credible without these

**P0.1 — Expand dataset to 8-15 documents**
- Current: 3 documents (3M Marine, DOA Food, Fire Safety), 47 pages total.
- Target: 8-15 documents across distinct industrial domains.
- Safe public sources (no partner clearance needed):
  - OSHA Process Safety Management factsheets (osha.gov, freely citeable)
  - NIST SP 800-series operational/maintenance guides (public domain)
  - John Deere / Caterpillar public service manuals (OEM-published)
  - HSE UK guidance documents (hse.gov.uk)
- Intake pipeline: PDF -> text (PyMuPDF already in deps) -> gold annotation (JSON, same schema
  as `datasets/archive/test_data/gold/`) -> peer audit (30% sample, Cohen's kappa > 0.7 target)
- Do NOT build a UI for annotation. Use the existing JSON schema in a text editor.
- Files go in: `datasets/paper/text/` and `datasets/paper/gold/`

**P0.2 — Multi-seed runs with 95% confidence intervals**
- Current: single seed per document per strategy.
- Target: >= 3 seeds, paired bootstrap CI on per-chunk Phi across documents.
- Implementation: add `--seeds` flag to experiment runner; fix random state via LLM temperature
  and a seed parameter passed to the backend; use `scipy.stats.bootstrap` or manual paired
  bootstrap across documents for CIs.
- Key file to extend: `scripts/experiments/run_all_chunking_experiments.py` and
  `scripts/experiments/experiment_utils.py`.

**P0.3 — Fuzzy constraint-attachment metric**
- Current: ConstraintAttachmentF1 collapses to 0 due to strict string ID matching. This is
  flagged as a known artefact in the thesis. The paper cannot ship with a metric that always
  reports 0.
- Fix: semantic attachment metric. A predicted (constraint, step) pair is correct if:
  cosine_similarity(pred_constraint_embedding, gold_constraint_embedding) >= 0.75 AND
  cosine_similarity(pred_step_embedding, gold_step_embedding) >= 0.75.
  Use `EmbeddingCache` (already in `src/evaluation/metrics.py`) — it is already available.
- Report both strict (current) and fuzzy versions for transparency.
- Key file: `src/evaluation/metrics.py` — add a `fuzzy_constraint_attachment_f1` function,
  then expose it in `evaluate_tier_a_document` alongside the existing strict version.

**P0.4 — At least one additional model family**
- Current: Mistral-7B vs Llama-3.1-70B (both via llama-cpp GGUF).
- Add: Qwen2.5-7B-Instruct Q4_K_M — it's a stronger 7B baseline as of 2026 and will likely
  produce an interesting data point on the efficiency frontier.
- Optional additions (do not block on): OpenEuroLLM-Finnish-7B (Finnish SOP angle, strong
  European venue differentiator), Phi-4-mini (~3.8B, smaller efficiency point).
- These are inference re-runs on existing documents. Wire them through the same backend
  abstraction in `src/ai/llm_backends.py`. If using LM Studio's OpenAI-compatible endpoint,
  you likely need to add an `OpenAICompatibleBackend` class.

### P1 — lifts paper from ok to elite

**P1.1 — Small expert human study (start ASAP — it has the longest lead time)**
- 2-3 domain experts (CoRe supervisors, TeoAly partner contacts with industrial process
  backgrounds) rate 40-60 sampled extractions on: step comprehensiveness (1-5), sequence
  correctness (1-5), constraint clarity (1-5), trust to use in production (1-5).
- Script for sampling already exists: `scripts/experiments/prepare_human_eval_samples.py`
- Correlate expert ratings with Phi via Spearman. Target rho > 0.5.
- This is the only task with a social/coordination blocker. Ask David or Mikko this week.

**P1.2 — Constraint-type breakdown**
- Break constraints into: guards (IF/THEN), parameter thresholds (value > X), role assignments,
  sequencing preconditions.
- Show per-type recall under each prompting strategy. This requires adding a `constraint_type`
  field to gold annotations and to the P3 extraction schema.
- Expected finding: "P3 lifts guard recall from ~12% to ~70% and parameter recall from ~9%
  to ~58%." That kind of table earns reviewer trust.

**P1.3 — PAGED metric comparison**
- Run PAGED's evaluation metric on your extractions (or run your metric on PAGED's gold) so
  reviewers can locate IPKE on the existing benchmark map.
- One paragraph in the paper, one row added to Table 1.

### P2 — opportunistic

**P2.1 — Finnish SOP extension**: only if a CoRe partner provides Finnish docs AND P0 is done.
**P2.2 — Phi weight sensitivity ablation**: recompute Phi for w_c in {0.3, 0.5, 0.7}; show
  ranking of methods is stable. Pre-empts "arbitrary weights" reviewer concern. Small effort.
**P2.3 — Two-turn CoT fix**: test the P2 CoT failure fix ("generate reasoning, then second
  call with schema-only instructions"). If it works, another data point on the prompting axis.

---

## LLM backend notes

Current backend: `llama-cpp-python` loading GGUF files locally. Works for Mistral on Mac
(Metal) and 3090/5090 (CUDA). For paper experiments with multiple models, adding an
`OpenAICompatibleBackend` in `src/ai/llm_backends.py` that hits LM Studio's local endpoint
is the cleanest path — avoids re-downloading GGUF for every model and works across machines.

LM Studio is preferred over Ollama (clearer quantization/context visibility, faster model
switching for benchmarking). Use LM Studio for dev; GGUF direct via llama-cpp for final runs
where reproducibility matters (no LM Studio version dependency).

Model path expected at: `models/llm/<model_name>.gguf`. Set `LLM_N_GPU_LAYERS=-1` to offload
all layers to GPU. `SKIP_MODEL_CHECK=true` skips model presence check for code-only work.

---

## Git conventions

All commits must be authored as **Imad <imad.e.elmouss@turkuamk.fi>**. Never use a different
name or email. Never add "Co-Authored-By" trailers. The repo is sole-authored research work
belonging to Imad Eddine Elmouss.

```bash
git config user.name "Imad"
git config user.email "imad.e.elmouss@turkuamk.fi"
```

---

## Coding conventions

- Python 3.12+. `uv` for env management. `ruff` for lint/format. Strict types where they add
  clarity; Pydantic v2 at all structured output boundaries.
- Hardware-agnostic: use try/except guards on `torch.cuda.is_available()` and
  `torch.backends.mps.is_available()`. All model paths and hardware selection via env vars.
- Centralized JSON logging via `src/logging_config.py`. Use `get_logger(__name__)`, not
  `print()` or bare `logging.info()`.
- Experiment outputs go in a timestamped `runs/<experiment_name>/<timestamp>/` directory,
  not committed to git. Keep raw predictions as JSON; re-score with `reevaluate_metrics.py`
  when the metric changes.
- No Neo4j required for paper experiments. The graph builder works in-memory; Neo4j is for
  the deployed application, not evaluation.
- No cloud API calls. Privacy-preserving by design.

---

## What not to do

- Do not build an annotation UI. JSON in a text editor, David/Mikko audit 30%.
- Do not replicate PAGED end-to-end. Run their metric on your data.
- Do not add Finnish SOP extension before P0 is done.
- Do not rename Phi. "Procedural Fidelity" is established.
- Do not let tests require a live LLM or Neo4j unless marked `@pytest.mark.integration`.
- Do not add model files (`.gguf`) or raw partner SOPs to git.
- Do not touch the FastAPI / Streamlit / Neo4j stack for paper work. It's the demo
  application layer; the paper needs the experiment pipeline.

---

## Paper timeline checkpoints

| Date | Milestone |
|---|---|
| Now - June 2 | Decisions: confirm ECIR 2027, confirm co-authors, identify which public SOPs to use |
| June 3 - July 7 | P0.1 dataset expansion + P0.2 multi-seed engineering |
| July 8 - Aug 11 | P0.3 fuzzy metric, P0.4 new model, P1.1 expert study (start recruitment NOW) |
| Aug 12 - Sep 8 | First full draft |
| Sep 9 - Sep 22 | Co-author review passes, LaTeX port to LNCS template |
| Sep 22 - Oct 12 | Polish, reproducibility check, submit |

Submission deadline: **12 October 2026**. Non-negotiable.

---

## Reproducibility checklist (ECIR requires this)

- [ ] Public repo cited in paper (deanonymize after review)
- [ ] Gold annotations released under CC-BY where partners allow (confirm per-doc)
- [ ] Exact model name, quantization, temperature, seed, hardware in §4
- [ ] One-command rerun: `make eval` (wire this up before submission)
- [ ] Fresh-clone test: verify setup works from scratch by end of June

---

## Status (update at end of every session touching experiments)

- 2025-12 — Thesis submitted. 3 documents, single seed, Mistral-7B + DSC + P3 = Phi=0.699.
- 2026-05-27 — Returning to repo for paper push. Hardware upgraded: M4 + RTX 3090 + 2x
  RTX 5090 + CSC Puhti. Paper target: ECIR 2027 short paper, deadline Oct 12 2026.
  P0 experiments not yet started. Expert recruitment not started. Dataset still 3 docs.
