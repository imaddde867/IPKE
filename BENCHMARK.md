# IPKE Supporting Evaluation Corpus

This file describes the corpus, taxonomy, and validators used to evaluate IPKE. The IPKE
extraction method is the primary research contribution; this evaluation corpus is
supporting infrastructure.

## TL;DR

The active artifact contains source texts, procedural annotations, a locked constraint
taxonomy, and validation tooling. It supports controlled comparisons of
skeleton-conditioned attachment, filtering, segmentation, and local inference cost.

**IPKE is the contribution under test.** Existing work already represents and evaluates
constraint-flow structure, so this repository must not claim that constraint attachment
itself is absent from all prior benchmarks.

| You want to … | Read this |
|---|---|
| Understand the method | `README.md` and `docs/methods/dsc-implementation.md` |
| Run a controlled IPKE experiment | `REPRODUCIBILITY.md` |
| Annotate a new document for the corpus | `docs/annotation/guidelines.md` + `docs/annotation/constraint-types.md` |
| Be an independent second-pass annotator | `docs/annotation/independent-annotator-workflow.md` |
| Understand the corpus composition | `docs/dataset/datasheet.md` (Gebru format) |
| Reproduce the §1 motivating result | `make eval-blindness` — runs `scripts/constraint_blindness_report.py` |
| Build / extend the IPKE pipeline | `README.md` (the pipeline-side entry doc) |

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

**Cross-regime illustration (labelled).** The fixed thin-era LLM draft holds 32 constraints vs the 231 reviewed (**7.22× expansion**); at the Tier-A matcher (SBERT cos ≥ 0.75) it recovers 6.1%, at cos ≥ 0.50 it recovers 37.7%. Draft and gold come from different annotation regimes — annotation-economics evidence, not an extractor-quality claim. Like-for-like extractor comparisons on the fully verified corpus are planned.

Regenerate (informational) with:

```bash
make eval-blindness      # regenerates + prints, no assertions
make repro-blindness     # asserts the pinned numbers (32 vs 231, 7.22x)
```

Reports land in `datasets/paper/reports/constraint_blindness_v2_sbert{050,075}.json`.

## Corpus

| Released | Target |
|---|---|
| 8 (seed corpus, 2026-06-13) | 12-15 (planned expansion) |

Per-document statistics: see `docs/dataset/datasheet.md` §2.3.

Target additions for genre diversity:

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

Pre-publication, cite the GitHub repository. A formal BibTeX entry will land in this file at paper acceptance.

## Status

| Component | State |
|---|---|
| Locked constraint taxonomy (6 types × 3 enforcement levels) | ✅ shipped |
| Annotation guidelines + independent annotator workflow | ✅ shipped |
| Paper-grade validator + unit tests | ✅ shipped |
| Datasheet (Gebru format) | ✅ shipped |
| Seed-corpus source texts, annotations, provenance manifest | ✅ released |
| Human verification of seed annotations | 🟡 in progress (`docs/annotation/manual-review/`) |
| Corpus expansion beyond 8 documents | 🟡 candidate sources identified |
