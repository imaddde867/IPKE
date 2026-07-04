# Reproducibility

## Fresh Clone

```bash
uv sync
make test
```

The default uv `dev` dependency group installs pytest for local validation.

## Single-Document Smoke Extraction

```bash
make smoke-extract
```

Outputs are written to:

```text
runs/pkg_extraction/3m_marine_oem_sop/
```

The run directory contains:

- `3M_OEM_SOP_extracted.json`
- `3M_OEM_SOP_pkg.json`
- `config.json`

## Evaluation

```bash
make eval
```

This runs a dry-run multi-seed sweep against `datasets/paper/gold` (paper Tier-A documents).
No model is required. At the current 8-document seed stage, `make eval` runs strict
gold validation, regenerates the D1 constraint-blindness reports, and dry-runs the
D2 sweep plan. Final ECIR table regeneration still requires 12 reviewed documents,
eligible independent second-pass annotations, and completed D2 model runs. For the
full sweep see `make eval-full` and `REPRODUCIBILITY.md`.

## Required Metadata

Each final experiment run must record:

- git SHA
- document ID
- input SHA256
- model name or path
- quantization
- temperature
- random seed
- backend
- hardware backend
- Python version
- platform

## Known Limits

- Thesis archive results cover three documents.
- ECIR paper experiments require `datasets/paper/` expansion.
- ConstraintAttachmentF1 remains strict unless the fuzzy metric task is implemented.
