# ADR-0004: Make ECIR Resource Paper the Primary Target

Status: accepted

Date: 2026-06-13

## Context

The original paper plan framed IPKE as a 6-page ECIR short paper extending the thesis
method result. That path leaned on the "pipeline over parameters" story: a local
quantized model plus Dual Semantic Chunking and Two-Stage Decomposition outperforming
larger models under weaker prompting.

Deep review of the current landscape changes the risk profile:

- Procedural graph extraction already has benchmarks such as PAGED.
- Procedural Knowledge Graph extraction with human evaluation already exists.
- Industrial KG construction and maintenance-domain LLM benchmarks are active areas.
- Recent RAG/chunking work already supports the broad claim that structure-aware
  segmentation matters.

IPKE's strongest defensible gap is narrower: safety-critical industrial procedures need
Constraint Attachment to be measured directly. Existing resources under-measure whether
guards, warnings, parameter thresholds, and preconditions attach to the exact Procedural
Step they govern.

ECIR 2027 has a Resource Paper Track that explicitly evaluates novelty, availability,
reliability, utility, impact, licensing, documentation, and maintenance plans for
datasets, tools, benchmark protocols, and reusable resources.

## Decision

The primary target is now an ECIR 2027 Resource Paper centered on **IPKE-Bench**.

IPKE-Bench is a rights-screened, constraint-aware benchmark for Procedural Knowledge
Extraction from safety-critical industrial documents. The method pipeline becomes the
reference local/private baseline on that benchmark.

The old 6-page short-paper method framing is demoted to fallback status.

## Consequences

Positive:

- The paper can claim a reusable community resource rather than only a thesis-method
  improvement.
- The dataset, annotation schema, IAA, loaders, metrics, and reproducibility package
  become first-class contributions.
- The CoRe value is stronger: a citable local-first industrial AI benchmark that can
  support future SOP, report, speech, and operator-support projects.
- The resource track allows 12 pages and single-blind review, which fits artifact-heavy
  work better than a compressed short paper.

Negative:

- The resource bar is higher: reviewers can inspect the repository and will expect
  documentation, licensing clarity, examples, maintenance plans, and working loaders.
- The dataset must be human-reviewed before any paper claim.
- IAA coverage and annotation quality are no longer optional polish.
- The paper needs a retrieval/resource-use story, not only extraction metrics.

## Required Follow-up

- Update `AGENTS.md`, `CONTEXT.md`, `REPRODUCIBILITY.md`, dataset notes, and vault notes
  to reflect the resource-paper path.
- Treat `docs/paper/ipke-bench-resource-prd.md` as the active PRD.
- Update GitHub issues so open work tracks IPKE-Bench gates.
- Add or document a text-RAG vs PKG-backed retrieval/query task.
- Do not run headline model sweeps until reviewed gold and IAA gates are closed.
