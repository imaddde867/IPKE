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
No model is required. At the current five-candidate development stage, `make eval` runs strict
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
make repro-blindness  # assert the pinned D1 cross-regime numbers (32 vs 231, 7.22x); non-gate
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
- ECIR paper experiments require `datasets/paper/` expansion to 12 documents plus completed D2 model runs.
- Constraint matching reports both the fuzzy Tier-A protocol matcher (SBERT cos ≥ 0.75, `src/evaluation/alignment.py`) and strict exact-match; per-type recall at small n is brittle and is always reported threshold-attached.
- Gold `relations` (NEXT / ALTERNATIVE_TO) are annotated but Tier-B currently synthesizes its NEXT chain from steps-array order; wiring the scorers to gold relations is an open design choice.
