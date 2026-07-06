# IPKE-Bench

Top-level entry point for **IPKE-Bench**, the constraint-aware benchmark for procedural knowledge extraction from safety-critical industrial documents. This file distinguishes the benchmark from `IPKE` (the local extraction pipeline) and routes you to the right docs depending on what you want to do.

## TL;DR

**IPKE-Bench** is an evaluation-only dataset, a locked annotation taxonomy, and a paper-grade validator. Use it to compare procedural-extraction systems on the *constraint-attachment* axis — which existing benchmarks do not measure.

**IPKE** is the reference local/private extraction pipeline that demonstrates the benchmark can be cleared. The pipeline is a strong baseline but **not** the contribution of the IPKE-Bench paper.

| You want to … | Read this |
|---|---|
| Cite the benchmark | `docs/paper/ipke-bench-resource-prd.md` (PRD) and the (forthcoming) ECIR 2027 Resource Paper. |
| Run an extractor and report on IPKE-Bench | `REPRODUCIBILITY.md`, then `make eval-validate` and `make eval-full` |
| Annotate a new document for the corpus | `docs/annotation/guidelines.md` + `docs/annotation/constraint-types.md` |
| Be an independent second-pass annotator | `docs/annotation/independent-annotator-workflow.md` |
| Understand the corpus composition | `docs/dataset/datasheet.md` (Gebru format) |
| Reproduce the §1 motivating result | `make eval-blindness` — runs `scripts/constraint_blindness_report.py` |
| Build / extend the IPKE pipeline | `README.md` (the pipeline-side entry doc) |
| Understand the domain vocabulary | `CONTEXT.md` |
| Follow the current execution direction | `docs/paper/2026-07-04-execution-direction.md` |

## What makes IPKE-Bench different

Existing procedural-knowledge benchmarks (PAGED, KEO, CAMB, Carriero & Celino 2024) measure step coverage, ordering, graph topology, or entity state. They do not treat *constraint attachment* — the explicit edge that binds a safety guard, parameter, precondition, or role assignment to the step it governs — as a primary evaluation target.

IPKE-Bench fills that gap. Annotations carry typed constraints (`precondition`, `postcondition`, `guard`, `parameter`, `role_assignment`, `reference`) with `enforcement` mapped from the source modal verb (`must` / `should` / `may`), each explicitly attached to one or more steps via `attached_to` (step-embedded) or `applies_to` (procedure-level).

## §1 motivating result (framing decided 2026-07-06 — see docs/paper/D1_SCOPE_DECISION.md)

**Corpus depth (headline).** The thin bounded-excerpt seed pass held 43 steps / 117 constraints across the 8 documents. Re-annotating the *same* documents end-to-end under the locked full-subprocedure scope rule, with source-verbatim constraint grounding, yields **256 steps / 231 reviewed constraints** — the constraint scaffolding that excerpt-scale annotation leaves unmeasured.

**Cross-regime illustration (labelled).** The fixed thin-era LLM draft holds 32 constraints vs the 231 reviewed (**7.22× expansion**); at the Tier-A matcher (SBERT cos ≥ 0.75) it recovers 6.1%, at cos ≥ 0.50 it recovers 37.7%. Draft and gold come from different annotation regimes — annotation-economics evidence, not an extractor-quality claim. The apples-to-apples number is the D2 P0 zero-shot baseline's ConstraintCoverage on the signed-off benchmark.

Regenerate (informational) with:

```bash
make eval-blindness      # regenerates + prints, no assertions
make repro-blindness     # asserts the pinned numbers (32 vs 231, 7.22x)
```

Reports land in `datasets/paper/reports/constraint_blindness_v2_sbert{050,075}.json`.

## Corpus

| Released | Target |
|---|---|
| 8 (seed corpus, 2026-06-13) | 12-15 (ECIR submission) |

Per-document statistics: see `docs/dataset/datasheet.md` §2.3.

Target additions for genre diversity (tracked in PRD):

- FAA AC 43.13-1B (aviation maintenance)
- FDA Food Code (food safety)
- NIST SP 800-61 Rev. 2 (computer security incident handling)
- Open-license OEM service manual

## Quality gates

A gold annotation is paper-grade if and only if `uv run python scripts/validate_paper_gold.py --strict` returns no errors. Gates:

- `quality.review_status == "reviewed"`
- `quality.annotator` and `quality.review_date` set
- Every constraint has `type` ∈ the locked 6-type vocabulary
- Every constraint has `enforcement` ∈ {must, should, may}
- Every constraint has `attached_to` (step-embedded) or `applies_to` (procedure-level) referencing a valid step ID
- Every constraint has non-empty `text`

IAA gate (open, requires recruited annotators):

- ≥ 4 documents (≥ 30% of the 12-doc target) have independent second-pass annotation by a human blind to gold.
- Every IAA pair has Cohen's κ ≥ 0.61 (substantial, Landis & Koch).

## Licensing

The annotation layer (JSON files, taxonomy, guidelines, evaluation harness) is released under CC-BY 4.0. Per-document source licensing is tracked in `datasets/paper/public_sources_manifest.csv` — all 8 seed documents are US federal works (public domain) or open-licensed.

## Citing

Pre-publication, cite the PRD and the GitHub repository. A formal BibTeX entry will land in this file at ECIR 2027 paper acceptance.

## Status

| Component | State |
|---|---|
| Locked constraint taxonomy | ✅ shipped (PR #85) |
| Annotation guidelines | ✅ shipped (PR #85) |
| Paper-grade validator + 12 unit tests | ✅ shipped (PR #85) |
| Seed corpus (8 docs) reviewed | ✅ shipped (PR #85) |
| §1 motivating result (D1) | ✅ shipped (PR #85) |
| Datasheet (Gebru format) | ✅ shipped (PR #85) |
| Independent annotator workflow | ✅ shipped (PR #85) |
| Recruited independent annotators | 🟡 outreach pending |
| Corpus expansion to 12 docs | 🟡 candidate sources identified |
| D2 baseline sweep (4 configs × 5 seeds) | ⏳ open; first milestone is one real-model non-empty metrics row |
| D3 constraint-aware retrieval task | ⏳ optional, post-D2 |

See `docs/plans/2026-06-13-ipke-bench-taxonomy-and-review.md` for the sprint history that produced this state.
See `docs/paper/2026-07-04-execution-direction.md` for the current issue order and active board state.
