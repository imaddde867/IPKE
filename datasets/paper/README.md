# IPKE-Bench Dataset Workspace

ECIR 2027 Resource Paper benchmark dataset. Primary contribution of the paper.

## Target

12–15 publicly licensed industrial / safety-procedure documents with human-reviewed step, constraint, and **constraint attachment** annotations. All gold files must reach `quality.review_status = "reviewed"` before inclusion in any reported result.

## Layout

- `text/` — plain-text extracted source documents. This can include downloaded alternates that are not part of the active reviewed release.
- `gold/` — Tier-A gold annotations (step, constraint, attachment). Schema: `schemas/ipke_annotation.schema.json`.
- `second_pass/` — independent second annotations for IAA computation.
- `reports/` — IAA reports and annotation statistics.

Active corpus counts come from `gold/` plus `selected_for_gold=true` in
`public_sources_manifest.csv`, not from every file present in `text/`.
Manifest rows with `review_status=deferred_candidate` are downloaded
alternates only; they require a reviewed gold file before inclusion in any
paper result.

## Document Selection Criteria

Priority: publicly licensable, stable URL, citable, varied procedure types and domains.

1. Safety/regulatory: NASA NPR, EPA guidance, OSHA PSM examples, HSE UK
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
3. Have `quality.review_status = "reviewed"` for paper inclusion.
4. Have `quality.token_label_kappa ≥ 0.61` if a second-pass pair exists.

Run: `uv run python scripts/validate_gold.py datasets/paper/gold/`
