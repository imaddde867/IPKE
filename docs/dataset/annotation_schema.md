# IPKE Annotation Schema

IPKE annotations are procedure-level JSON files aligned to the existing
`datasets/archive/test_data/gold/` shape. They are not standalone span annotations.

`schemas/ipke_annotation.schema.json` is the permissive, candidate-compatible structural
schema. The current `datasets/paper/gold/` files are immutable legacy candidates.
Primary outputs belong in `datasets/paper/primary_pass/`, final annotations in
`datasets/paper/production/`, and frozen evidence packages in
`datasets/paper/evidence/` under `schemas/ipke_annotation_evidence.schema.json`.

Production eligibility additionally requires every step and constraint to carry exact
end-exclusive Unicode `provenance.char_start` and `provenance.char_end` offsets inside
`procedure.source`. The runtime evidence validator checks those offsets against the
committed UTF-8 source. It also binds the filename, annotation ID, sidecar ID, canonical
artifact paths, and exact-byte hashes for the candidate, primary pass, optional blind
pass, agreement report, adjudication output, and final annotation. The sidecar also
records source URL, retrieval date, version, page range, section identity, and
redistribution status, plus complete step, constraint, and relation decision ledgers.
Primary and blind outputs must use the exact frozen procedure document ID and character
window recorded by that sidecar.
Constraints require stable IDs. ID-less relations use
`REL:<source>:<type>:<target>` as their canonical ledger identity.

Blind-pair reports bind raw attachment TP/FP/FN and adjudication review-item IDs. The
validator recomputes the pair counts from frozen artifacts, aggregates them across the
blind subset for the 0.70 G0 gate, and requires typed adjudication decisions to cover the
report items exactly.
Disagreement item IDs are deterministically derived from the frozen primary/blind
step, constraint, explicit-relation, and attachment projections; report authors cannot
choose or omit them.

## Required Top-Level Fields

- `procedure` - document-level metadata.
- `steps` - ordered extracted procedure steps.
- `constraints` - recommended for new paper Tier-A annotations when document-level constraints exist.

Existing archive files also include `relations`, `resources_catalog`, `canon`, and `quality`.
The paper schema allows these fields and keeps top-level `constraints` optional for backward
compatibility with archive-shape gold files.

## Procedure

Required fields:

- `doc_id` - stable ID matching `datasets/paper/public_sources_manifest.csv`.
- `title` - human-readable document or procedure title.

Recommended fields:

- `version`
- `domain`
- `source`

`source` should include page, section, character offsets, sentence ID, or URL metadata when
available. Keep provenance traceable enough to re-check the source text.

## Steps

Each step should describe one procedural action or check. Recommended fields follow the archive
gold files:

- `id` - stable step ID such as `S1`.
- `label` - concise step text.
- `action_verb` - normalized action.
- `action_object` - canonical object and source surface form.
- `arguments` - instruments, targets, locations, or other action arguments.
- `resources` - referenced tools, materials, PPE, documents, or equipment IDs.
- `parameters` - named quantities, values, units, ranges, tolerances, times, or settings.
- `constraints` - step-local preconditions, guards, warnings, acceptance criteria, or postconditions.
- `flags` - optional, parallel, safety-critical, or similar booleans.
- `provenance` - document, page, character, sentence, and section evidence.

Step IDs must remain stable after review because evaluation uses them for ordering, adjacency,
and constraint attachment checks.

## Constraints

Use top-level `constraints` for document-level constraints that do not attach cleanly to one step.
Use step-local `constraints` for preconditions, guards, warnings, acceptance criteria, and
postconditions tied to a specific step.

Recommended constraint fields:

- `id`
- `type`
- `text`
- `applies_to`
- `provenance`

## Resources

Archive gold stores resources under `resources_catalog` with lists such as `tools`, `materials`,
`ppe`, and `documents`. Paper annotations may also include a top-level `resources` field when a
simpler document-local resource list is useful. Resource references inside steps should use stable
resource IDs.

## Label Taxonomy Guidance

The source-selection plan uses this label vocabulary:

- `ACTION`
- `OBJECT`
- `TOOL`
- `MATERIAL`
- `ACTOR`
- `PARAMETER`
- `VALUE`
- `UNIT`
- `TIME`
- `CONDITION`
- `PRECONDITION`
- `POSTCONDITION`
- `WARNING`
- `PROHIBITION`
- `CHECK`
- `EXCEPTION`
- `OUTCOME`
- `REFERENCE`

IPKE stores these mainly as procedure fields rather than BRAT-style spans:

- `ACTION` maps to `steps[].action_verb` and `steps[].label`.
- `OBJECT` maps to `steps[].action_object` and arguments.
- `TOOL`, `MATERIAL`, and other resources map to `resources_catalog` and step `resources`.
- `PARAMETER`, `VALUE`, `UNIT`, and `TIME` map to `steps[].parameters`.
- `CONDITION`, `PRECONDITION`, `POSTCONDITION`, `WARNING`, `PROHIBITION`, `CHECK`, and `EXCEPTION` map to step-local or top-level `constraints` and flags.
- `OUTCOME` maps to postconditions, acceptance criteria, or terminal step labels.
- `REFERENCE` maps to provenance, source pointers, documents, figures, tables, or relation fields.

This structure keeps the gold set directly usable by the existing IPKE evaluation pipeline while
still preserving enough semantic detail for later span or graph evaluation.
