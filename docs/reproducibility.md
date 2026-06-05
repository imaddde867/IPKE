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

This evaluates predictions against `datasets/archive/gold_human`.

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
