# IPKE Evaluation Corpus

Supporting evaluation data for IPKE: rights-cleared procedural source documents with
source-grounded step, constraint, and constraint-attachment annotations under a locked
six-type × three-enforcement taxonomy. See [BENCHMARK.md](../../BENCHMARK.md) for the
corpus role and [docs/dataset/datasheet.md](../../docs/dataset/datasheet.md) for the
full datasheet.

## Layout

- `text/` — plain-text extracted source documents.
- `gold/` — seed-corpus annotations (model-assisted drafting, agent audit, human manual
  review; review records in `docs/annotation/manual-review/`). Schema:
  `schemas/ipke_annotation.schema.json`.
- `segments/` — heading-aware segmentation of each source document.
- `second_pass/` — source-only blind second-pass annotations with their source texts.
- `adjudication_decisions/` — item-level adjudication records for annotation disagreements.
- `review_candidates/`, `review_packets/` — model-assisted review candidates and the
  reviewer packets prepared for them.
- `gold_v1_bounded_excerpt_archive/` — archived first-pass bounded-excerpt annotations.
- `reports/` — inter-annotator agreement reports and annotation statistics.
- `corpus_manifest.json` — typed corpus membership manifest.
- `public_sources_manifest.csv` — provenance for every source document: URL, title,
  license, SHA-256 hash, size, access date, and conversion command.
- `primary_pass/`, `production/`, `evidence/` — reserved layout for future independent
  human annotation rounds and frozen evidence packages
  (`schemas/ipke_annotation_evidence.schema.json`).

## Document Selection Criteria

Priority: publicly licensable, stable URL, citable, varied procedure types and domains.

1. Safety/regulatory procedures: EPA guidance, OSHA PSM examples, HSE UK
2. Equipment / maintenance: OLSK CNC, USGS field sampling, public maintenance SOPs
3. Quality / process: ISO-aligned public process guides

All eight released sources are US federal works (public domain) or open-licensed
(CC BY-SA). Private or partner-restricted SOPs must never be committed here.

## Rules

- Do not mutate `datasets/archive/` gold annotations.
- Keep document IDs stable once annotation starts.
- Record source URL, access date, license, and conversion command in
  `public_sources_manifest.csv`.
- Preserve model-assisted candidates and raw human passes as separate artifacts.

## Validation

Every annotation must:

1. Parse as valid JSON against `schemas/ipke_annotation.schema.json`.
2. Contain stable step, constraint, attachment, and relation identifiers.
3. Resolve every accepted step and constraint to exact source offsets.
4. Use constraint types from the locked vocabulary and enforcement labels in
   {must, should, may}.
5. Attach every constraint to a valid step (`attached_to`) or to the procedure level
   (`applies_to`).

Structural check over all released annotations:

```bash
make eval-validate
```
