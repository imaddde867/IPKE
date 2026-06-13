# IPKE-Bench Resource Paper PRD

## Objective

Build IPKE into an elite ECIR 2027 Resource Paper by releasing **IPKE-Bench**: a
constraint-aware benchmark and evaluation package for Procedural Knowledge Extraction
from safety-critical industrial documents.

The benchmark is the contribution. IPKE is the reference local/private baseline.

## Venue

Primary target: ECIR 2027 Resource Paper Track.

Reasons:

- The track explicitly welcomes novel datasets, labelled annotations, software tools,
  benchmarking protocols, reproducibility infrastructure, and resource documentation.
- Resource papers can be up to 12 pages plus references.
- Review is single-blind, so reviewers can inspect the actual repository and dataset.
- ECIR explicitly asks for availability, licensing, reliability, utility, limitations,
  maintenance plans, and reuse potential.

Fallback:

- ECIR 2027 Short Paper only if the resource is not mature enough.
- ESWC/ISWC-style semantic-web venue if the work grows into RDF/SHACL/SPARQL and
  ontology-centric contributions.

## Problem

Industrial procedures contain safety-critical constraints: preconditions, guards,
warnings, prohibitions, acceptance criteria, parameter thresholds, and exceptions.
Existing procedural graph extraction and procedural text understanding benchmarks
evaluate steps, order, graph edges, or entity state changes, but they do not make
**Constraint Attachment** the primary object of evaluation.

A constraint without an explicit link to the governed Procedural Step is not
operationally useful. This is the gap IPKE-Bench should own.

## Intended Users

- Researchers evaluating LLM-based procedural graph extraction.
- Industrial AI teams building local/private document-to-knowledge pipelines.
- CoRe projects that need shared schemas for SOPs, field reports, maintenance notes,
  meeting-derived procedures, and operator-support systems.
- Reviewers who need a reproducible, rights-screened dataset instead of thesis-only
  anecdotal results.

## Resource Surface

IPKE-Bench must include:

- Public-source manifest with stable URLs, hashes, usage notes, licenses, and risk
  screen.
- Plain-text excerpts for all selected documents.
- Human-reviewed Tier-A annotation JSON files.
- Second-pass annotations for IAA.
- Annotation schema and guidelines.
- Loader and validation scripts.
- Tier-A and Tier-B evaluation scripts.
- Baseline outputs or commands to regenerate them.
- Datasheet-style documentation.
- Reproducibility commands for tests, IAA, metrics, and multi-seed runs.

## Primary Claims

Claim 1:
IPKE-Bench introduces a rights-screened industrial procedure benchmark where
Constraint Attachment is a first-class evaluation target.

Claim 2:
Local, quantized LLM pipelines can produce useful Procedural Knowledge Graphs when
task decomposition and structure-aware chunking are evaluated against constraint-aware
metrics.

Claim 3:
Text-chunk retrieval and procedural graph retrieval fail differently: chunk RAG is
strong for local fact lookup, while PKG-backed retrieval is better for questions that
ask which constraints govern a step or which steps are blocked by a guard.

The old "pipeline over parameters" result can support the paper, but it must not be
the primary claim unless regenerated on reviewed gold with multi-seed confidence
intervals.

## Required Evaluation

P0:

- Human-review all 8 current Tier-A files.
- Reach at least 30 percent second-pass annotation coverage.
- Ensure each IAA document reaches κ >= 0.61, preferably κ >= 0.70.
- Run schema validation over all annotations.
- Run IAA report regeneration from committed scripts.
- Run IPKE baseline sweeps with 5 seeds after reviewed gold is ready.
- Report StepF1, AdjacencyF1, Kendall, ConstraintCoverage,
  ConstraintAttachmentF1, and Procedural Fidelity Score.
- Report strict and semantic/fuzzy attachment where possible.
- Report 95 percent confidence intervals and paired bootstrap comparisons.

P1:

- Expand to 12-15 public documents if schedule allows.
- Add constraint-type breakdown: guard, warning, parameter, precondition,
  postcondition, exception, prohibition.
- Add text-RAG vs PKG-backed retrieval/query evaluation.
- Compare against at least one external procedural graph or KG construction baseline
  where technically feasible.

## Acceptance Criteria

Resource quality:

- All included data has source, license/usage note, checksum, and document ID.
- All paper-claimed gold files have `quality.review_status = "reviewed"`.
- Dataset notes disclose AI-assisted drafting and human correction.
- Annotation guidelines are sufficient for a new annotator to reproduce the task.
- The resource can be loaded and validated with one documented command.

Evaluation quality:

- No paper number is manually edited.
- All metrics can be regenerated from committed scripts.
- IAA is reported per document and aggregate.
- Failure modes are disclosed, especially policy-heavy documents and small n.

Paper quality:

- The introduction frames IPKE-Bench as enabling industrial information access and
  safety-aware procedural retrieval.
- Related work compares against PAGED, procedural KG extraction with human evaluation,
  EDC-style KG construction, industrial maintenance KG work, and recent RAG/chunking
  studies.
- Limitations discuss English-only scope, public-document bias, AI-assisted drafting,
  licensing limits, and lack of private partner SOPs unless clearance is obtained.

## Non-Goals

- Do not build an annotation UI for this paper.
- Do not fine-tune models unless it becomes a separate follow-up paper.
- Do not make multimodal diagrams/P&IDs part of the current paper.
- Do not publish partner-private SOPs.
- Do not keep expanding model families before the reviewed dataset and IAA gates are
  closed.

## Current Blockers

- Current Tier-A files are still AI-drafted until human review is recorded.
- IAA coverage is below the preferred resource-paper bar.
- OLSK second-pass κ is below the paper threshold.
- Text-RAG vs PKG-backed retrieval is not yet implemented.

## Source Pointers

- ECIR 2027 Resource Track:
  https://www.ecir2027.co.uk/call-for-resource-papers
- ECIR 2027 dates:
  https://www.ecir2027.co.uk/
- Dataset notes:
  `docs/paper/dataset_notes.md`
- Reproducibility:
  `REPRODUCIBILITY.md`
- Domain glossary:
  `CONTEXT.md`
