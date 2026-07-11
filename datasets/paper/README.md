# IPKE Evaluation Corpus Workspace

Supporting evaluation data for the IPKE method paper. The corpus is not the primary
contribution and the current directory is not a released benchmark or a confirmatory
test set.

## Target

Twelve rights-cleared, source-family-diverse procedures with manually corrected,
human-verified step, constraint, and **constraint attachment** annotations. A
`quality.review_status = "reviewed"` value alone is not enough for paper evidence.

## Layout

- `text/` — plain-text extracted source documents, including candidates and alternates.
- `gold/` — legacy candidate annotations (step, constraint, attachment). Directory
  presence does not make a file confirmatory gold. Schema:
  `schemas/ipke_annotation.schema.json`.
- `second_pass/` — independent second annotations for IAA computation.
- `reports/` — IAA reports and annotation statistics.

The eight JSON files in `gold/` are legacy candidates, not the active confirmatory
corpus. As of 2026-07-11, NASA is excluded as a requirements stress test and current
OLSK/NIOSH golds are excluded pending manual rebuild. Issue #112 tracks the explicit
confirmatory inclusion manifest. No result may infer membership from directory presence
or `selected_for_gold=true` alone.

## Document Selection Criteria

Priority: publicly licensable, stable URL, citable, varied procedure types and domains.

1. Safety/regulatory procedures: EPA guidance, OSHA PSM examples, HSE UK
2. Equipment / maintenance: OLSK CNC, USGS field sampling, public maintenance SOPs
3. Quality / process: ISO-aligned public process guides

Avoid: partner-private SOPs (index separately under `datasets/private/` with access gating).

## Rules

- Do not mutate `datasets/archive/` gold annotations.
- Do not commit partner-private SOPs to this directory.
- Keep document IDs stable once annotation starts.
- Record source URL, access date, license, and conversion command in the gold file's `metadata.source` field.

## Validation

Every gold file must:

1. Parse as valid JSON against `schemas/ipke_annotation.schema.json`.
2. Contain `steps`, `constraints`, stable step IDs, stable constraint IDs.
3. Carry an explicit non-pending `+ human-verified:<handle>` marker for paper inclusion.
4. Belong to the frozen confirmatory split.
5. Pass schema, structural, grounding, attachment, and evidence-eligibility checks.

Development structural check:

`make eval-validate`

Paper-evidence check, intentionally failing until sign-off and manifest alignment:

`make eval-paper-gate`
