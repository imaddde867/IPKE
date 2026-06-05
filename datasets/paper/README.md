# Paper Dataset Workspace

This directory is for ECIR 2027 paper documents added after the thesis archive.

## Layout

- `text/` - plain-text public SOP/manual documents used for experiments.
- `gold/` - Tier A gold annotations using the same schema as `datasets/archive/test_data/gold/`.

## Rules

- Do not mutate `datasets/archive/` gold annotations.
- Do not commit partner-private SOPs.
- Prefer public documents with stable URLs and licenses.
- Record source URL, access date, license, conversion command, and any extraction caveat in the experiment run metadata.
- Keep document IDs stable once gold annotation starts.

## Validation

Before running paper experiments, each new gold file must parse as JSON and include:

- `steps`
- `constraints`
- stable step IDs
- stable constraint IDs
