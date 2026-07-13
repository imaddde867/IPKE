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

This runs a dry-run multi-seed sweep against the five candidates selected by
`datasets/paper/corpus_manifest.json`. No model is required. The command structurally
validates all retained candidates, regenerates the historical D1 diagnostics, and
dry-runs the legacy sweep. It is not a confirmatory method-paper command. See
`REPRODUCIBILITY.md` for the human-evidence and C0-C4 requirements.

## Gold Annotation Pipeline

Model tools and agents produce annotation candidates only. Independent humans create
production annotations under `docs/methods/annotation-pipeline.md`. Current development
targets are:

```bash
make gold-pipeline    # validate retained candidates and regenerate historical D1 diagnostics
make repro-blindness  # assert the pinned D1 cross-regime numbers (32 vs 231, 7.22x); non-gate
make gold-draft DOC=<doc_id> SEG=<segments.json> CAND=<candidate_id>   # create a candidate
make gold-adjudicate DOC=<doc_id>                                     # replay historical candidate decisions
make iaa-setup        # prepare the frozen >=25% blind subset after protocol gates pass
make iaa              # preserve pre-adjudication agreement for every selected pair
```

The committed files under `datasets/paper/gold/` are legacy candidates, not production
gold. `make gold-pipeline` confirms development diagnostics only.

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
- Confirmatory method-paper experiments require frozen exact-span C0-C4 controls and the complete human-evidence protocol.
- Constraint matching reports both the fuzzy Tier-A protocol matcher (SBERT cos ≥ 0.75, `src/evaluation/alignment.py`) and strict exact-match; per-type recall at small n is brittle and is always reported threshold-attached.
- Gold `relations` (NEXT / ALTERNATIVE_TO) are annotated but Tier-B currently synthesizes its NEXT chain from steps-array order; wiring the scorers to gold relations is an open design choice.
