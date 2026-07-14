# IPKE Supporting Evaluation Corpus

This file describes the corpus, taxonomy, and validators used to evaluate IPKE. ADR-0005
makes the IPKE method the paper's primary contribution. The evaluation corpus remains
scientifically necessary, but it is not positioned as a standalone benchmark paper.

## TL;DR

The active artifact contains source texts, procedural annotations, a locked constraint
taxonomy, and validation tooling. It supports controlled comparisons of
skeleton-conditioned attachment, filtering, segmentation, and local inference cost.

**IPKE is the contribution under test.** Existing work already represents and evaluates
constraint-flow structure, so this repository must not claim that constraint attachment
itself is absent from all prior benchmarks.

| You want to … | Read this |
|---|---|
| Understand the method paper | `docs/adr/0005-ipke-method-paper-primary.md` and the approved method design |
| Run a controlled IPKE experiment | `REPRODUCIBILITY.md`; paper runs also require the explicit evidence gate |
| Annotate a new document for the corpus | `docs/annotation/guidelines.md` + `docs/annotation/constraint-types.md` |
| Be an independent second-pass annotator | `docs/annotation/independent-annotator-workflow.md` |
| Understand the corpus composition | `docs/dataset/datasheet.md` (Gebru format) |
| Reproduce the §1 motivating result | `make eval-blindness` — runs `scripts/constraint_blindness_report.py` |
| Build / extend the IPKE pipeline | `README.md` (the pipeline-side entry doc) |
| Understand the domain vocabulary | `CONTEXT.md` |
| Follow the current execution direction | `docs/paper/2026-07-04-execution-direction.md` |

## Evaluation role

Existing procedural-knowledge benchmarks (PAGED, KEO, CAMB, Carriero & Celino 2024) measure step coverage, ordering, graph topology, or entity state. They do not treat *constraint attachment* — the explicit edge that binds a safety guard, parameter, precondition, or role assignment to the step it governs — as a primary evaluation target.

The annotations provide fine-grained types (`precondition`, `postcondition`, `guard`,
`parameter`, `role_assignment`, `reference`), enforcement labels (`must`, `should`,
`may`), and explicit attachment targets. This supports the IPKE causal protocol and
failure analysis. It does not establish novelty by itself.

## Historical annotation-process result

The thin bounded-excerpt seed pass held 43 steps and 117 constraints across the eight
documents. The later full-subprocedure, agent-reviewed pass contains 256 steps and 231
constraints. These counts describe different annotation regimes. They are not an
extractor-quality result and are not the method-paper headline.

**Cross-regime illustration (labelled).** The fixed thin-era LLM draft holds 32 constraints vs the 231 reviewed (**7.22× expansion**); at the Tier-A matcher (SBERT cos ≥ 0.75) it recovers 6.1%, at cos ≥ 0.50 it recovers 37.7%. Draft and gold come from different annotation regimes — annotation-economics evidence, not an extractor-quality claim. The apples-to-apples number will be the controlled P0 baseline's ConstraintCoverage on the production-eligible supporting corpus.

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

A custom-validator pass is necessary but not sufficient for paper evidence. Gates include:

- JSON Schema and structural validation pass
- frozen manifest membership and a final artifact under `datasets/paper/production/`
- complete primary-human source pass recorded in a frozen evidence sidecar
- exact source, bounded-span, and final-annotation byte hashes
- exact Unicode source offsets for every accepted step and constraint
- no unresolved primary decisions or pending-human-sign-off marker
- source grounding and exact-span experiment-input checks pass
- Every constraint has `type` ∈ the locked 6-type vocabulary
- Every constraint has `enforcement` ∈ {must, should, may}
- Every constraint has `attached_to` (step-embedded) or `applies_to` (procedure-level) referencing a valid step ID
- Every constraint has non-empty `text`

IAA gate (open, requires recruited annotators):

- At least 25% of experiment-eligible procedures are selected before model results and
  receive a frozen source-only blind pass.
- Every preregistered raw pair is preserved and reported before adjudication.
- A third human who created neither pass adjudicates disagreements.
- Attachment-edge F1 ≥ 0.70 is the G0 protocol gate; kappa remains diagnostic and low
  pairs are not discarded.

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
| Legacy candidates (8 docs) | 🟡 0 production eligible; human evidence open in #108 |
| Exact-anchor evidence boundary | ✅ schema and fail-closed runtime gate implemented |
| First agent-prepared EPA review candidate | ✅ 14 steps, 15 constraints, exact anchors; 8 human decisions open |
| Historical D1 annotation comparison | ⚪ supporting context only |
| Datasheet (Gebru format) | ✅ shipped (PR #85) |
| Independent annotator workflow | ✅ shipped (PR #85) |
| Recruited independent annotators | 🟡 outreach pending |
| Corpus expansion to 12 docs | 🟡 candidate sources identified |
| Controlled C0-C4 pilot | ⏳ blocked by production evidence and exact-span inputs |
| Explicit relation evaluation | ⏳ open in #109 |
| Full method sweep | ⏳ blocked in #55 |
| Constraint-aware retrieval | ⚪ optional follow-up |

See `docs/plans/2026-06-13-ipke-bench-taxonomy-and-review.md` for the sprint history that produced this state.
See `docs/paper/2026-07-04-execution-direction.md` for the current issue order and active
board state.
