# Annotation evidence packages

This directory holds one frozen sidecar per production annotation:

```text
datasets/paper/evidence/<doc_id>.json
```

The contract is `schemas/ipke_annotation_evidence.schema.json`. A sidecar binds the
source URL, retrieval date, version, page range, section, redistribution status,
authoritative UTF-8 bytes, bounded Unicode character span, primary pass, optional
blind/adjudication artifacts, and final annotation bytes by SHA-256.

Do not create placeholder sidecars. In particular, do not copy historical records from
`datasets/paper/adjudication_decisions/`: those are agent decision records and cannot
serve as human evidence.

Participant handles in committed sidecars must be stable pseudonyms such as `P-001`.
Names, email addresses, affiliations, consent records, and the pseudonym-to-person key
remain outside the public repository under the study coordinator's control.

Production paths are separate from the retained candidates:

```text
datasets/paper/gold/<doc_id>.json          # immutable legacy candidate
datasets/paper/review_candidates/<doc_id>.json # candidate offered to primary reviewer
datasets/paper/primary_pass/<doc_id>.json  # frozen primary-human output
datasets/paper/second_pass/<doc_id>.json   # frozen source-only blind output, if selected
datasets/paper/reports/<doc_id>_agreement.json # frozen pre-adjudication agreement
datasets/paper/production/<doc_id>.json    # final experiment annotation
datasets/paper/evidence/<doc_id>.json      # evidence package
```

The validator enforces these logical paths, loads every referenced artifact, and hashes
its exact on-disk bytes. Offsets are end-exclusive indices into the UTF-8-decoded Python
string, so they count Unicode code points rather than encoded bytes. Every production
step and constraint must carry its own `provenance.char_start` and
`provenance.char_end` inside `procedure.source`.

The primary decision ledger covers steps, constraints, and relations. Relation decisions
include order edges, branches, alternatives, and other explicit graph relations; a file
cannot become production evidence while those decisions are absent. A relation without an
explicit `id` uses `REL:<source>:<type>:<target>` as its stable ledger identity;
constraints require explicit, unique IDs.

For blind-selected files, the validator recomputes attachment-edge TP/FP/FN from the
hashed primary and blind annotations and checks the frozen report. G0 is applied to the
counts aggregated across the complete preregistered blind subset, not to individual
pairs. Typed adjudication decisions must exactly cover the report's disagreement and
seeded-audit IDs and name the adjudicator as decision maker.

Run the candidate-compatible structural check with `make eval-validate`. Run the
fail-closed production boundary with `make eval-paper-gate`.
