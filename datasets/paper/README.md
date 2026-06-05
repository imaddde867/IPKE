# Paper Dataset Workspace

This directory is for ECIR 2027 paper documents added after the thesis archive.

## Layout

- `text/` - plain-text public SOP/manual documents used for experiments.
- `gold/` - Tier A gold annotations using the same schema as `datasets/archive/test_data/gold/`.
- `public_sources_manifest.csv` - rights-screened source manifest for the public paper corpus.

## Rules

- Do not mutate `datasets/archive/` gold annotations.
- Do not commit partner-private SOPs.
- Prefer public documents with stable URLs and licenses.
- Record source URL, access date, license, conversion command, and any extraction caveat in the experiment run metadata.
- Keep document IDs stable once gold annotation starts.
- Do not commit raw PDFs, spreadsheets, screenshots, or downloaded source files.
- Store final extracted plain text in `datasets/paper/text/`.
- Store final reviewed gold JSON in `datasets/paper/gold/`.
- Keep `document_id` values aligned with `public_sources_manifest.csv`.

## Public Source Manifest

`public_sources_manifest.csv` is the source of truth for first-wave public documents for
issue 53. It records the source family, stable URL, downloaded local filename, SHA-256,
byte size, license or usage note, and selection flags.

Selection fields:

- `selected_for_download` means the source was included in the first local raw-document harvest.
- `selected_for_gold` means the source is part of the initial Tier-A annotation target set.
- `risk_level` is a practical rights and extraction-risk screen, not a legal conclusion.

Raw downloaded files live outside this repository. Do not copy them into `datasets/paper/` or
any committed path.

## Annotation Schema

Gold annotations use a procedure-level JSON shape, not BRAT-style standalone spans. The schema
for new paper annotations is documented in `docs/dataset/annotation_schema.md` and machine
checked by `schemas/ipke_annotation.schema.json`.

## Validation

Before running paper experiments, each new gold file must parse as JSON and include:

- `steps`
- `constraints`
- stable step IDs
- stable constraint IDs
