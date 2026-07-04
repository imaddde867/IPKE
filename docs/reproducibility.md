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

## Gold Annotation Pipeline

The gold annotations are produced by a model-assisted-draft + human-adjudication
pipeline documented in full at `docs/methods/annotation-pipeline.md`. One-command
targets:

```bash
make gold-pipeline    # deterministic gate: strict-validate the 8 golds + regenerate D1 numbers (no model)
make gold-draft DOC=<doc_id> SEG=<segments.json> CAND=<candidate_id>   # draft one procedure (needs model backend)
make gold-adjudicate DOC=<doc_id>                                     # replay a persisted adjudication log -> reviewed gold
make iaa-setup        # select >=30% subset + emit blank (anchoring-safe) second-pass scaffolds
make iaa              # score step/constraint/relation F1 + Cohen's kappa over completed second passes
```

The committed golds under `datasets/paper/gold/` are the source of truth;
`make gold-pipeline` is the target a reviewer runs on a fresh clone to confirm
the release numbers regenerate.

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
