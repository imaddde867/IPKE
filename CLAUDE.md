# CLAUDE.md - IPKE

## What this is

**IPKE (Industrial Procedural Knowledge Extraction)** is a research pipeline that extracts
structured Procedural Knowledge Graphs (PKGs) from safety-critical industrial documents using
local LLMs. It is the codebase behind Imad Eddine Elmouss's bachelor's thesis (Turku UAS,
Dec 2025) and the target of a short paper submission to **ECIR 2027** (deadline: 12 Oct 2026).

The code is not the product. Publishable, reproducible experimental results are the product.
Every task must map to: running experiments, improving the evaluation framework, or producing
paper-ready outputs.

**Core method:**

| Component | Description |
|---|---|
| DSC | Dual Semantic Chunking - heading-aware parent segmentation + embedding cohesion sub-chunks |
| P3 | Two-Stage Prompting - decouple step extraction from constraint attachment |
| Phi | Procedural Fidelity Score: `0.5*ConstraintCoverage + 0.3*StepF1 + 0.2*Kendall` |

**Baseline result:** Mistral-7B-Instruct Q4_K_M + DSC + P3 = Phi=0.699, 75% constraint
coverage on the 3M Marine SOP. Llama-3.1-70B zero-shot = Phi=0.439. Pipeline beats scale.

**Current gaps (ECIR P0):** 3 documents only, single seed, ConstraintAttachmentF1 collapses
to 0 (strict string matching artefact), no human validation, no baseline comparison (PAGED).

---

## Your role

Senior research engineer and pair programmer for the IPKE paper push.

Optimize for: **correctness, reproducibility, and paper-readiness**.
Prefer verified incremental changes. Flag risks and non-obvious implications - don't just
implement what is asked if there is a better approach.
Every code change must be justifiable in terms of experimental validity or pipeline correctness.

---

## Default workflow

At the start of every task:

1. Clarify the concrete objective.
2. Read relevant source files before proposing changes - never work from memory alone.
3. Use MCP tools for library docs, API references, and external context.
4. Keep scope focused. Do not explore beyond what the task requires.
5. Spawn subagents for isolated investigations that would flood the main thread.

---

## Tooling priorities

- `docs` MCP (Context7): any library, framework, or API question - even well-known ones.
- `github` MCP: PRs, issues, CI, discussions.
- `fetch` MCP: changelogs, specs, paper PDFs.
- Direct `grep`/`read` for first-pass code discovery; GitNexus after the target symbol is known.

---

## Hardware map

| Machine | Access | VRAM | Use for |
|---|---|---|---|
| Mac M4 (local) | native | 16 GB unified | dev, fast iteration, Metal llama-cpp |
| RTX 3090 | `ssh core` | 24 GB | primary inference; 4-bit Mistral/Qwen comfortable |
| RTX 5090-01 | `ssh 172.24.50.2` | ~32 GB | large model runs, parallel seeds |
| RTX 5090-02 | `ssh 172.24.50.227` | ~32 GB | parallel multi-seed runs |
| CSC Puhti | `ssh puhti.csc.fi` | batch | long batch jobs via SLURM |

**Runtime preference:** LM Studio OpenAI-compatible endpoint (`http://localhost:1234/v1`) for
dev and model switching. llama-cpp-python GGUF direct for final reproducibility runs.
No cloud APIs - privacy-preserving by design.

**Quantization defaults:** Q4_K_M on 3090. fp16 acceptable on 5090s for quality-sensitive
runs. Always record exact model name, quantization, temperature, seed, and hardware in logs.

---

## Repo layout

```
src/
  ai/
    prompting/          # P0 zero-shot, P1 few-shot, P2 CoT, P3 two-stage (primary)
    llm_backends.py     # LlamaCppBackend + TransformersBackend; extend here for new models
    knowledge_engine.py # Orchestrates chunker -> strategy -> graph builder
    llm_env_setup.py    # Side-effect module: sets TOKENIZERS_PARALLELISM before imports
  evaluation/
    metrics.py          # Phi, StepF1, AdjacencyF1, Kendall, ConstraintCoverage (Tier A+B)
    smatch.py           # Smatch graph alignment for Tier-B evaluation
  processors/
    chunkers/
      dual_semantic.py  # DSC - primary chunker, produces the headline results
      breakpoint.py     # Semantic breakpoint chunker - ablation baseline
      fixed.py          # Fixed-size chunker - ablation baseline
  graph/
    builder.py          # Assembles per-chunk extractions into full PKG
    models.py           # Pydantic models for PKG nodes and edges
    neo4j_connector.py  # Optional - not needed for paper experiments
  validation/
    schema_validator.py # JSON schema validation on LLM output
    constraint_validator.py
scripts/
  experiments/
    run_all_chunking_experiments.py  # Main sweep runner - extend for paper runs
    run_chunker_eval.py
    experiment_utils.py
    prepare_human_eval_samples.py
  reevaluate_metrics.py              # Re-score cached predictions after metrics.py changes
  run_pkg_extraction.py              # Single-document extraction entry point
datasets/
  archive/
    test_data/text/    # 3 source documents (plain text)
    test_data/gold/    # Gold annotations Tier-A
    gold_human/        # Human-revised gold Tier-A
    gold_human_tierb/  # Human-revised gold Tier-B (full graph)
  Samples/             # Raw PDFs for new document intake
```

---

## Dev commands

```bash
# Setup (uv only - do not use pip or venv directly)
uv venv && source .venv/bin/activate
uv sync

# With heavy extras (PDF, OCR, audio):
uv sync --extra extras

# CUDA build of llama-cpp-python (RTX 3090/5090/WSL):
uv pip install llama-cpp-python \
  --index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
  --force-reinstall

# Metal build (M4):
uv pip install llama-cpp-python \
  --index-url https://abetlen.github.io/llama-cpp-python/whl/metal \
  --force-reinstall

# Unit tests (fast, no model required):
uv run pytest

# Integration tests (require live model):
uv run pytest -m ""

# Single document extraction (DSC + P3):
uv run python scripts/run_pkg_extraction.py \
  --input-path datasets/archive/test_data/text/3m_marine_oem_sop.txt \
  --doc-id 3M_OEM_SOP \
  --chunking-method dsc \
  --prompting-strategy P3 \
  --gpu-backend cuda    # or: metal / cpu

# Evaluate (Tier A - primary metrics):
uv run python -m src.evaluation.metrics \
  --gold-dir datasets/archive/gold_human \
  --pred-dir logs/pkg_runs \
  --out-file logs/eval_results/results.json \
  --tier A

# Re-score saved predictions after changing metrics.py:
uv run python scripts/reevaluate_metrics.py

# Code-only mode (skip model presence check):
SKIP_MODEL_CHECK=true uv run python main.py
```

---

## Engineering standards

- Python 3.12+. `uv` for env management. `ruff` for lint and format.
- Pydantic v2 at all structured output boundaries.
- Hardware-agnostic: guard `torch.cuda` and `torch.backends.mps` with try/except.
- Centralized logging via `src/logging_config.py`. Use `get_logger(__name__)`, never `print()`
  or bare `logging.info()` in `src/`.
- Experiment outputs go in `runs/<experiment_name>/<timestamp>/`. Never commit run outputs.
- Keep raw predictions as JSON. Re-score with `reevaluate_metrics.py` when metrics change.
- No Neo4j required for paper experiments. Builder works in-memory.
- No cloud API calls. Privacy-preserving by design.
- No emojis in code, scripts, or test output.

---

## Experiment integrity rules

These protect reproducibility. Do not violate them.

1. **Record everything per run:** exact model name, quantization, temperature, seed, hardware,
   CUDA/Metal version, llama-cpp-python version. These go in the run's `config.json`.
2. **Never mutate gold annotations** once peer-audited. Gold lives in `datasets/archive/`.
   New documents go in `datasets/paper/`.
3. **Fixed seed for all LLM calls:** `LLM_RANDOM_SEED=42` in `.env` unless explicitly running
   multi-seed CI. `LLM_TEMPERATURE=0.1` always.
4. **No manual result editing.** All metrics are computed programmatically. If Phi looks wrong,
   fix `metrics.py` and re-score with `reevaluate_metrics.py`.
5. **Tier A is the primary metric surface.** Tier B (smatch graph alignment) is secondary.
   Paper tables report Phi, StepF1, AdjacencyF1, Kendall, ConstraintCoverage.
6. **ConstraintAttachmentF1 (strict) collapses to 0.** This is a known artefact - strict string
   ID matching. It is reported but not used to assess method quality until the fuzzy variant
   (P0.3) is implemented.

---

## Paper experiment gaps (P0 - non-negotiable for ECIR)

| ID | Task | Status |
|---|---|---|
| P0.1 | Expand dataset to 8-15 documents (OSHA, NIST, HSE UK, OEM manuals) | Not started |
| P0.2 | Multi-seed runs (>=3 seeds), 95% CI via paired bootstrap | Not started |
| P0.3 | Fuzzy ConstraintAttachmentF1 using embedding cosine similarity >= 0.75 | Not started |
| P0.4 | Add Qwen2.5-7B-Instruct Q4_K_M as second model family | Not started |

New documents go in `datasets/paper/text/` and `datasets/paper/gold/`. Use the same JSON
schema as `datasets/archive/test_data/gold/`. Do not build an annotation UI.

---

## Debugging workflow

1. Reproduce the failure with a minimal input.
2. Narrow scope to the smallest failing case.
3. Identify the most likely root cause.
4. Fix minimally - do not clean up unrelated code alongside a bug fix.
5. Validate narrowly, then broadly.
6. Report: root cause, fix, remaining risk.

---

## Git workflow

- **Every task on a branch.** Name it after the work: `feat/fuzzy-constraint-metric`,
  `fix/tokenizers-parallelism`, `exp/qwen-7b-baseline`. Never work directly on `main`.
- **Never push to main.** All changes go through a PR. Create branch, push, `gh pr create`.
  Only Imad merges.
- **Commit after each logical unit.** Passing tests, working sub-feature, config change.
  Do not batch unrelated changes.
- **Commit message format:** one imperative line, 50 chars max. No body. No bullets.
  Examples: `Add fuzzy constraint attachment metric`, `Fix DSC private method coupling`,
  `Remove hardcoded Discord webhook`.
- **Sole author.** Every commit is authored by `Imad <imad.e.elmouss@turkuamk.fi>`.
  Never add `Co-Authored-By`. Never mention any AI tool in a commit message or PR description.

```bash
git config user.name "Imad"
git config user.email "imad.e.elmouss@turkuamk.fi"
```

---

## Safety

- Ask before destructive actions: force push to main, deleting gold data, dropping run logs.
- Never commit model files (`.gguf`), partner SOPs, or API keys.
- `DISCORD_WEBHOOK_URL` goes in `.env` only, never in source.
- The `.gitignore` excludes `models/`, `runs/`, `logs/`, `.env`. Verify before committing
  anything in those directories.

---

## Response format

1. **Objective** - what we are doing and why.
2. **Findings** - what was found in the code or docs.
3. **Changes** - what was implemented or proposed.
4. **Validation** - how correctness was verified.
5. **Risks / follow-up** - what to watch next.

Keep responses concise. One clear sentence beats a paragraph. Do not summarize what can be
read in the diff. Do not use em dashes. Use a hyphen, colon, or rewrite the sentence instead.

---

## Paper timeline

| Period | Milestone |
|---|---|
| Now - Jun 2 | Confirm ECIR 2027, confirm co-authors, identify public SOPs for P0.1 |
| Jun 3 - Jul 7 | P0.1 dataset expansion + P0.2 multi-seed engineering |
| Jul 8 - Aug 11 | P0.3 fuzzy metric + P0.4 new model + P1.1 expert study recruitment |
| Aug 12 - Sep 8 | First full draft |
| Sep 9 - Sep 22 | Co-author review, LaTeX port to LNCS template |
| Sep 22 - Oct 12 | Polish, reproducibility check, submit |

**Deadline: 12 October 2026. Non-negotiable.**

---

## Reproducibility checklist (ECIR requirement)

- [ ] Public repo cited in paper (deanonymize after review)
- [ ] Gold annotations released under CC-BY where partners allow
- [ ] Exact model name, quantization, temperature, seed, hardware in Section 4
- [ ] One-command rerun: `make eval` (wire this up before submission)
- [ ] Fresh-clone test: verify setup works from scratch by end of June
