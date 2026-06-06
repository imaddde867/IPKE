# Annotation Protocol (Pilot v0.1)

This protocol applies to `datasets/paper/gold/` files committed under the
`pilot_gold` status. It is a draft protocol, not a finalized annotation
guideline. Future revisions will introduce independent annotators, a
calibration pass, and adjudicated disagreements.

## Scope

- Each gold file annotates one bounded source excerpt, not a full document.
- An excerpt is a contiguous block of procedure text with a known
  `section` or page range recorded in `procedure.source`.
- Annotators must produce, at minimum:
  - `steps` (ordered, with stable `id` and `label`),
  - step-local `constraints` (preconditions, warnings, orderings,
    postconditions),
  - `procedure.audit` (reclassification + reviewer metadata).

## Source Order

- Annotators read the source text first, then may consult a model draft.
- The model draft (if any) is recorded in
  `datasets/paper/annotation_batches/<batch_id>.json` under
  `model_draft_provenance`.
- After draft review, the annotator edits the JSON directly. They do
  not paste raw model output.

## Step and Constraint Discipline

- Step boundaries follow the source: one step per procedural action or
  check. Do not split one source action into two steps.
- Constraint text must be a substring (after whitespace normalization)
  of the source excerpt, or a near-paraphrase with `provenance.reason`
  explaining the deviation.
- Constraint attachment uses `attached_to` (preferred), `applies_to`, or
  `targets` and must reference a step `id` that exists in the file.

## Audit Block

- `gold_status` is `pilot_gold` for all current paper files. Future
  research-grade gold will use `gold`.
- `annotation_status` is `draft`, `reviewed`, `double_annotated`, or
  `adjudicated`.
- `annotator_count` records the number of distinct annotators who
  contributed to the file.
- `reviewed_by` is a human or batch id, not a model name.

## Adjudication

- Pilot annotations are not adjudicated. Disagreements are recorded in
  the IAA report (`datasets/paper/reports/issue_53_iaa_report.json`)
  and flagged as pilot workflow output, not paper-grade reliability.

## Limitations

- This protocol is a scaffold. Do not cite `pilot_gold` files as
  research-grade gold in the paper.
