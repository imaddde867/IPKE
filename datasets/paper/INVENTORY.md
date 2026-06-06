# Paper Dataset Inventory

This document is the human-readable map of the public paper corpus.
It is generated and reviewed by hand. For machine-readable metadata,
see `public_sources_manifest.csv` (every row with `selected_for_gold=true`).

## Tier-A Documents (8)

The 8 documents that anchor Tier A of the paper, one per pilot source family,
plus the closest possible mixed-regulatory/standards sources for diversity:

1. `usgs_nfm_collection_water_samples_a4` - USGS, environmental water sampling.
2. `usgs_groundwater_technical_procedures_tm1_a1` - USGS, groundwater techniques.
3. `epa_guidance_preparing_sops_qag6` - EPA, regulatory SOP guidance.
4. `epa_field_sampling_measurement_procedure_validation` - EPA, field sampling SOP. (double-annotated)
5. `epa_field_operations_manual_filter_sampling_sop` - EPA, filter sampling SOP.
6. `niosh_nmam_5th_edition_ebook` - CDC_NIOSH, analytical methods manual. (double-annotated)
7. `nasa_npr_8715_3d_general_safety` - NASA, safety program requirements.
8. `olsk_small_cnc_v1_workbook` - OLSK, community hardware SOP. (double-annotated)

The Tier-A set is the bounded set the paper reports Phi on. It does not
include the FAA alternates.

## Source Families (6)

| family | tier-a count | alternate count | notes |
|---|---|---|---|
| USGS | 2 | 0 | Water and groundwater field procedures. |
| FAA | 0 | 2 | Aviation maintenance handbooks; alternates, not in Tier A. |
| EPA | 3 | 0 | SOP guidance, field sampling, filter sampling. |
| CDC_NIOSH | 1 | 0 | NMAM 5th edition general considerations. |
| NASA | 1 | 0 | NPR 8715.3D general safety program. |
| OLSK | 1 | 0 | Small CNC V1 workbook. |

The 2 FAA handbooks are downloaded and textified for completeness
(`datasets/paper/text/faa_amt_general_handbook_2023.txt`,
`faa_amt_airframe_handbook.txt`) but are not part of the bounded
Tier-A set. They are flagged in the manifest as `selected_for_gold=false`.

## Double-Annotated Documents (3)

Three documents have a second-pass annotation in `second_pass/`:

- `epa_field_sampling_measurement_procedure_validation.pilot_gold.json` <-> `epa_field_sampling_measurement_procedure_validation.json`
- `niosh_nmam_5th_edition_ebook.pilot_gold.json` <-> `niosh_nmam_5th_edition_ebook.json`
- `olsk_small_cnc_v1_workbook.pilot_gold.json` <-> `olsk_small_cnc_v1_workbook.json`

IAA is computed across all three pairs and aggregated in
`reports/issue_53_iaa_report.json`.

## Extracted Text Directory

`datasets/paper/text/` contains 10 files: 8 Tier-A documents plus 2
FAA handbooks. The directory is reproduced by:

```bash
uv run python scripts/extract_public_documents.py \
  --manifest datasets/paper/public_sources_manifest.csv \
  --raw-dir <raw-download-dir> \
  --out-dir datasets/paper/text \
  --selection download
```

The default mode is `--selection gold`, which produces only the 8
Tier-A files. Both behaviors are tested in
`tests/test_extractor_selection_modes.py`.

## Review Batches

- `annotation_batches/batch_pilot_self_review.json` - the self-review batch
  covering all 8 Tier-A documents.
- `annotation_batches/batch_pilot_second_pass.json` - the second-pass
  batch covering the 3 double-annotated documents.
- `annotation_batches/manifest_pilot_status.json` - per-document status
  records (gold_status, annotation_status, annotation_scope, annotator_count).

## Schema

All pilot gold files conform to `schemas/ipke_paper_tiera.schema.json`
(strict). The looser `schemas/ipke_annotation.schema.json` is retained
for backward compatibility with prior IAA reports.

## Licenses and Attribution

- `LICENSES.md` (root) - redistribution policy per source family.
- `ATTRIBUTION.md` (root) - canonical attribution list.
- `LICENSES.md` excludes `MyFixit` and `iFixit` content from the corpus.

## Linked Artifacts

- `public_sources_manifest.csv` - machine-readable manifest (22 columns).
- `annotation_batches/manifest_pilot_status.json` - per-document status records.
- `reports/issue_53_iaa_report.json` - IAA report across 3 documents.
- `REVISION_NOTES.md` - audit trail for the public-gold revision.
- `annotation_protocol.md` - annotation procedure.

## Updating This File

When adding a new Tier-A document, append a row to the Tier-A list,
update the source-family table, and open a PR.
