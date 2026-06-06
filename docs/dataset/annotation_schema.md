# IPKE Annotation Schema

IPKE gold annotations are procedure-level JSON files aligned to the existing
`datasets/archive/test_data/gold/` shape. They are not standalone span annotations.

New paper gold files should be stored in `datasets/paper/gold/` and validated with
`schemas/ipke_annotation.schema.json`.

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
