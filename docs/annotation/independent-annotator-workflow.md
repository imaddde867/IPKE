# Source-Only Blind Annotation Workflow

Status: paused until the manifest, IAA subset, and coordinator assignments are frozen

This workflow produces the independent blind pass required by the Human Evidence
Recovery Design. It is not candidate correction and it does not begin from
`datasets/paper/gold/`.

## Assignment

- Expect about three active hours per bounded procedure. Record actual active minutes.
- Work only on documents assigned through the frozen manifest and IAA subset.
- At least 25% of experiment-eligible procedures receive a blind pass.
- Annotation count alone does not guarantee authorship. Credit follows institutional and
  venue policy.
- Use GitHub issue #90 for assignments, compensation, consent, and scheduling.

## Non-negotiable blindness

Do not view any of the following for your assigned procedure before committing and
freezing your pass:

- `datasets/paper/gold/<doc_id>.json`;
- model or agent candidates;
- source-to-candidate audit packets;
- primary human annotations;
- adjudication records;
- another annotator's work on that procedure.

Accidental exposure invalidates the assignment. Close the material, report the exposure,
and have the coordinator reassign the document to another annotator. Continuing after
exposure cannot produce a source-only blind pass.

## Before annotation

Read:

1. `CONTEXT.md` for active terminology and evidence rules;
2. `docs/annotation/guidelines.md` for annotation decisions;
3. `docs/annotation/constraint-types.md` for the locked vocabulary;
4. `docs/dataset/datasheet.md` for corpus limitations.

Do not begin a production assignment until:

- `datasets/paper/corpus_manifest.json` is frozen;
- `datasets/paper/iaa_subset.json` contains only eligible assigned procedures;
- the coordinator supplies a frozen source span and blank scaffold.

The assignment is resolved from the frozen files. This document intentionally contains
no hard-coded document table.

## Provided artifacts

For each assigned procedure the coordinator supplies:

- an authoritative source-only text file containing the exact bounded procedure;
- source URL, version, checksum, page range, section, and original-text offsets;
- a blank scaffold with `review_status = "unreviewed"` and no first-pass items;
- an assignment ID and your role;
- the locked annotation schema and validation command.

The source text, not a candidate annotation, is the authority. The original PDF may be
consulted for table layout and typography, but annotate only content represented by the
authoritative committed text unless the coordinator records a source correction.

## Per-document procedure

### 1. Start the effort record

Record the assignment ID, annotator handle, source checksum, start time, and active
minutes. Exclude breaks and unrelated setup time.

### 2. Read the full source span

Read the entire bounded procedure once before finalizing labels. Identify procedural
actions, constraints, explanatory material, examples, definitions, and references.

### 3. Annotate from source only

For every step:

- use one source-faithful atomic operation;
- record stable local ID, label, verb, object, order, and relations;
- record exact end-exclusive Unicode `char_start` and `char_end`;
- do not infer an operation absent from the text.

For every constraint:

- record verbatim or minimally normalized text;
- select one locked type and one evidence-supported enforcement value;
- attach it to every governed step and no unknown step;
- record exact end-exclusive Unicode `char_start` and `char_end`;
- flag uncertainty instead of defaulting an ambiguous modal to `must`.

Definitions and nonbinding examples are not constraints. If an example appears binding in
context, flag that decision for post-freeze adjudication.

### 4. Inspect negative regions

Re-read source paragraphs containing no annotation. Check modal verbs, prohibitions,
warnings, emergency headings, quantities, timing, role language, and cross-sentence
dependencies. This reduces omissions that two annotators might otherwise share.

### 5. Validate

Run the assignment's declared-schema, structural, grounding, identifier, attachment,
relation, and provenance checks. A blank or partial scaffold remains `unreviewed`.

Set `review_status = "reviewed"` only after the complete source pass, exact anchors,
uncertainty record, effort log, and all deterministic checks are complete.

### 6. Freeze and commit

Record the final annotation checksum and active minutes. Commit the annotation and its
log on a branch named `annotation/blind-<doc_id>-<handle>` with a concise project-format
commit message. Link issue #90 and declare that no candidate or other pass was viewed.

Do not amend semantic labels after reveal. Any later changes belong to a separate
adjudication artifact.

## Reveal and agreement

After both passes are frozen, the coordinator:

1. verifies the source, schema, annotation, and log hashes;
2. computes step, constraint, attachment-edge, relation, type, enforcement,
   evidence-span, and token-label agreement;
3. stores the full pre-adjudication report for every preregistered pair;
4. reports low and high agreement without selecting pairs by outcome;
5. assigns a third human adjudicator who did not create either pass.

Attachment-edge F1 of at least 0.70 is the current G0 gate. A pair below the gate remains
part of the reported aggregate and triggers protocol investigation. Cohen's kappa is a
diagnostic, not an admission rule.

Omission and candidate-survival metrics are computed only after reveal. The blind
annotator must not receive candidate-derived counts during annotation.

## Independent adjudication

The adjudicator reviews every disagreement against the source and records the chosen
label, evidence, and rationale. They also audit rare classes, prohibitions, emergency
actions, implicit attachments, a seeded sample of agreements, and a seeded sample of
unannotated source regions.

Routine disagreements are adjudicated without the principal investigator. Escalate only
unresolved taxonomy, implicit-evidence, scope, or safety-critical decisions. Preserve the
two original passes, raw agreement, adjudication record, and final production annotation
separately.

## When uncertain

- Step boundary: choose the most source-faithful atomic boundary and flag uncertainty.
- Constraint type or enforcement: choose the best supported value and flag uncertainty;
  never default automatically to `must`.
- Zero-constraint step: retain it if it is genuinely unconstrained and record why.
- Example or definition: exclude it unless the surrounding text makes it binding, then
  flag it.
- Source extraction defect: stop that item and report the exact source location rather
  than silently correcting the authoritative text.

## Completion criteria

- The assignment remained source-only and blind.
- The complete bounded procedure was reviewed.
- Every item has exact source offsets.
- The annotation and effort log are complete and frozen.
- Deterministic validation passes.
- The pass and raw agreement remain preserved regardless of score.
- A different human adjudicates after reveal.

See also:

- `docs/superpowers/specs/2026-07-13-human-evidence-recovery-design.md`
- `docs/annotation/SIGN_OFF_ISSUE.md`
- `docs/methods/annotation-pipeline.md`
- GitHub issue #90
