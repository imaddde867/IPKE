# IPKE Evaluation Corpus Workspace

Supporting evaluation data for the IPKE method paper. The corpus is not the primary
contribution and the current directory is not a released benchmark or a confirmatory
test set.

## Target

Twelve rights-cleared, source-family-diverse procedures with manually corrected,
source-grounded step, constraint, and **constraint attachment** production annotations.
Each procedure needs a complete independent primary-human source pass, exact item-level
anchors, and an annotation log. At least 25% also needs a source-only blind pass and
independent adjudication. A status or signature marker alone is not paper evidence.
Corpus expansion follows, rather than precedes, a successful annotation-protocol and
two-document method pilot.

## Layout

- `text/` — plain-text extracted source documents, including candidates and alternates.
- `gold/` — immutable legacy model-assisted, agent-audited candidates. Directory
  presence does not make a file production gold. Schema:
  `schemas/ipke_annotation.schema.json`.
- `corpus_manifest.json` — typed evaluation membership contract. Its current
  `provisional` status permits development only.
- `second_pass/` — current placeholders and future source-only blind annotations.
- `reports/` — IAA reports and annotation statistics.

The required logical artifact roles are candidate, primary human pass, blind pass,
annotation log, adjudication record, and final production annotation. Their durable
physical layout and schemas must be implemented before production review begins. Do not
collapse these roles by editing a candidate in place.

The eight JSON files in `gold/` are legacy candidates, not the active confirmatory
corpus. As of 2026-07-11, NASA is excluded as a requirements stress test and current
OLSK/NIOSH golds are excluded pending manual rebuild. The provisional manifest selects
the other five candidates for development; none is human verified. No result may infer
membership from directory presence or `selected_for_gold=true` alone.

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
- Preserve model or agent candidates and raw human passes as separate artifacts.
- Never establish evidence by changing only `review_status` or an annotator marker.

## Validation

Every production annotation must:

1. Parse as valid JSON against `schemas/ipke_annotation.schema.json`.
2. Contain stable step, constraint, attachment, and relation identifiers.
3. Resolve every accepted step and constraint to exact source offsets.
4. Carry a complete independent primary-human pass and time/edit log.
5. Carry a frozen blind pass, raw agreement report, and independent adjudication record
   when selected for the preregistered 25% subset.
6. Preserve any limited principal-investigator escalation and evidence.
7. Belong to the frozen confirmatory split.
8. Pass declared-schema, structural, grounding, attachment, relation, provenance,
   agreement, adjudication, and evidence-eligibility checks.

Development structural check of all eight retained candidates:

`make eval-validate`

Paper-evidence check, intentionally failing today. The current implementation checks
manifest status and signature markers but does not yet establish the complete protocol:

`make eval-paper-gate`
