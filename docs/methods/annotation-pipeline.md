# Annotation Evidence Pipeline

Status: active protocol from 2026-07-13

This document operationalizes the Human Evidence Recovery Design for the supporting
IPKE evaluation corpus. The method-paper design remains the authority for experimental
gates and claims.

The current `datasets/paper/gold/` files are model-assisted, agent-audited candidates.
They are not production gold and cannot become paper evidence through a status change or
signature marker alone.

## Normative pipeline

```text
frozen source and bounded procedure
  -> immutable model or agent candidate
  -> agent source-to-candidate audit and decision packet
  -> independent human complete source pass
  -> exact item anchors and annotation log
  -> source-only blind second pass for at least 25%
  -> frozen pre-adjudication agreement
  -> independent human adjudication
  -> PI escalation only for unresolved scientific decisions
  -> production annotation and paper-evidence gate
```

The roles are deliberately separate. Models and agents reduce mechanical work. Humans
make production annotation decisions. A person who annotated a procedure cannot
adjudicate that procedure.

## 1. Freeze source and scope

Before drafting or reviewing, record:

- stable source URL and retrieval date;
- source version or revision;
- SHA-256 of the authoritative committed text;
- redistribution status and any third-party-content exception;
- page range and section identifier;
- exact Unicode `char_start` and end-exclusive `char_end` for one complete procedure.

The normal scope is one coherent procedure with roughly 15 to 40 atomic steps. A shorter
complete procedure may be accepted through a logged human scope decision. Do not expand
the boundary merely to meet a target count.

## 2. Produce and preserve a candidate

`annotate_assisted.py` and other model tools may propose steps, constraints, attachments,
relations, and metadata. Their outputs remain candidates with complete model, prompt,
seed, temperature, and source provenance.

An agent audit may:

- compare every proposed item with the frozen source;
- identify omissions, duplicates, unsupported actions, type errors, and bad attachments;
- calculate candidate and source hashes;
- propose exact evidence offsets;
- create an uncertainty ledger and compact human decision packet.

An agent audit may not sign, adjudicate as a human, or promote the candidate. Candidate
JSON and agent decision history remain immutable evidence of how assistance behaved.
`adjudicate.py replay` reproduces a historical candidate transformation only. It is not a
route to production gold.

## 3. Complete the primary human pass

One named human reviews the entire frozen source span. Candidate assistance is allowed,
but the reviewer must inspect source regions with and without suggestions and add omitted
content. Accepting or rejecting a prefilled list without a complete source pass is not
eligible.

The reviewer decides:

- procedure scope;
- step identity, granularity, order, and relations;
- constraint identity, text, type, and enforcement;
- every constraint-to-step attachment;
- exact evidence offsets for every accepted step and constraint;
- whether uncertain or implicit content needs escalation.

The primary pass is separately attributable to that human. A provenance marker may
record completion, but the marker is not evidence without the primary-pass record and
annotation log.

## 4. Record the annotation log

Every primary pass records:

- candidate SHA-256 and frozen-source SHA-256;
- reviewer handle and role;
- active review minutes, excluding breaks;
- candidate steps accepted, edited, rejected, and added;
- candidate constraints accepted, edited, rejected, and added;
- unresolved decisions and evidence locations;
- final annotation SHA-256.

Use separate step and constraint counts. For each item class:

```text
edit_rate = (edited + rejected) / candidate_items
omission_rate = added / final_items
```

These measurements describe the assistance workflow. They are not a human-effort or
automation-bias claim until the study design is frozen and sufficiently powered.

The exact sidecar schema and validator are part of the evidence-contract work. Until
they are implemented, a candidate cannot pass the paper-evidence gate.

## 5. Collect the blind second pass

At least 25% of experiment-eligible procedures are selected before model results are
inspected. Selection is source-family aware and stored in the frozen IAA subset.

The second annotator receives only:

- the frozen source span;
- the locked taxonomy and annotation guidelines;
- a blank, `unreviewed` scaffold with no first-pass items.

The second annotator must not view the candidate, primary pass, audit packet, or another
annotator's work. Accidental exposure invalidates the assignment and requires a different
annotator. Both passes are frozen before reveal.

## 6. Measure before adjudication

Compute step, constraint, attachment-edge, relation, type, enforcement, evidence-span,
and token-label agreement on every preregistered pair. Preserve the raw annotations,
hashes, and complete pre-adjudication report.

Attachment-edge F1 of at least 0.70 is the current G0 protocol gate. Cohen's kappa and
the other measures are diagnostics. A low-agreement pair remains in the aggregate and
triggers investigation; it is never discarded for failing a threshold.

## 7. Adjudicate independently

A third named human who did not annotate the procedure resolves every disagreement
against the source. The adjudicator also reviews:

- prohibitions and emergency actions;
- rare type and enforcement classes;
- implicit and cross-sentence attachments;
- a seeded sample of agreements;
- a seeded sample of source regions with no annotation.

Routine disagreement belongs to the adjudicator. Escalate only unresolved taxonomy,
implicit-evidence, scope, or safety-critical decisions to the principal investigator.
Record the evidence, alternatives, adjudicator recommendation, final decision, and
decision maker.

## 8. Establish experiment eligibility

A production annotation is experiment eligible only when all applicable evidence is
present and passes:

1. frozen source, rights, scope, and split membership;
2. declared JSON Schema and structural validation;
3. exact grounding for every step and constraint;
4. complete primary-human pass and annotation log;
5. blind-pass, raw-agreement, and independent-adjudication records for selected files;
6. logged principal-investigator escalations, if any;
7. frozen manifest and experiment configuration.

`review_status = "reviewed"`, a non-placeholder annotator, or
`+ human-verified:<handle>` is insufficient without these records.

## Current candidate snapshot

As of 2026-07-13, the retained candidate directory contains eight JSON files with 256
steps and 231 constraints. The provisional manifest selects five development candidates:
three EPA and two USGS. NASA is excluded as the wrong genre; OLSK and NIOSH are excluded
pending rebuild. No file has completed the primary human protocol or an eligible blind
second pass.

The older 43-step, 117-constraint bounded-excerpt annotations are preserved under
`datasets/paper/gold_v1_bounded_excerpt_archive/`. Historical counts and agent decision
records are diagnostic provenance, not semantic-correctness evidence.

## Development commands

```bash
# Structural checks over all retained legacy candidates. This is not paper eligibility.
make eval-validate

# Manifest, human-evidence, and release gate. Expected to fail today.
make eval-paper-gate

# Historical candidate drafting. Output remains a candidate.
make gold-draft DOC=<doc_id> SEG=<segments.json> CAND=<candidate_id>

# Historical deterministic candidate replay. It cannot produce human evidence.
make gold-adjudicate DOC=<doc_id>

# Blind subset preparation and agreement reporting after the subset is frozen.
make iaa-setup
make iaa
```

See also:

- `docs/superpowers/specs/2026-07-13-human-evidence-recovery-design.md`
- `docs/annotation/SIGN_OFF_ISSUE.md`
- `docs/annotation/independent-annotator-workflow.md`
- `docs/annotation/guidelines.md`
- `docs/annotation/constraint-types.md`
