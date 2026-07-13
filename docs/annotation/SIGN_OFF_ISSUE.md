# Complete independent production annotations

**Type:** gold quality and evidence gate

**Blocks:** confirmatory method-paper evaluation

**Owners:** assigned primary reviewers, blind annotators, and adjudicators
**Estimate:** about three active hours for a full primary pass, measured rather than assumed

This path is retained because existing documentation and issues link to it. It replaces
the former instruction to verify an agent candidate briefly and append a human marker.
A marker-only sign-off is not paper evidence.

## Current state

The directory `datasets/paper/gold/` contains eight model-assisted, agent-audited
candidates with 256 steps and 231 constraints. None has completed the production-human
protocol.

The provisional manifest selects these five development candidates:

- `epa_field_operations_manual_filter_sampling_sop`
- `epa_field_sampling_measurement_procedure_validation`
- `epa_guidance_preparing_sops_qag6`
- `usgs_groundwater_technical_procedures_tm1_a1`
- `usgs_nfm_collection_water_samples_a4`

It excludes:

- NASA NPR 8715.3D as a requirements stress test, not an executable procedure;
- OLSK Small CNC pending a source-faithful rebuild;
- NIOSH Method 2005 pending a source-faithful rebuild.

Inclusion in the provisional manifest does not make a candidate human verified or
experiment eligible.

## Required roles

For each production procedure assign:

1. one primary human reviewer who completes the source pass;
2. one source-only blind annotator if the procedure is in the preregistered blind subset;
3. one adjudicator who did not annotate that procedure;
4. the principal investigator only for unresolved taxonomy, implicit-evidence, scope, or
   safety-critical decisions.

Reviewers may be recruited or paid. The principal investigator is not the default
primary reviewer and does not perform routine transcription.

## Per-procedure work

### 1. Freeze the assignment

Before review, record the source URL, retrieval date, version, checksum, page range,
section, exact Unicode procedure offsets, candidate checksum, reviewer, and role.

Do not begin production review while the declared schema cannot represent item-level
evidence offsets or the annotation-log contract. Candidate audits may continue in
`docs/annotation/manual-review/` while that boundary is repaired.

### 2. Audit the candidate mechanically

An agent may prepare a source-to-candidate audit containing:

- proposed high-confidence corrections;
- suspected omissions, duplicates, and unsupported items;
- exact evidence offsets;
- provenance defects;
- a compact list of human decisions.

This audit is advisory. Preserve the candidate and audit history unchanged.

### 3. Complete the primary human pass

The named reviewer reads the entire frozen source span, including regions with no model
suggestion. Candidate assistance is allowed, but the reviewer personally decides every
step, constraint, type, enforcement value, attachment, relation, and evidence span.

The production pass must be separately attributable to the human. Directly changing
`review_status` or appending `+ human-verified:<handle>` to an agent candidate does not
establish this pass.

### 4. Record exact anchors and effort

Every accepted step and constraint needs end-exclusive Unicode `char_start` and
`char_end` values into the authoritative committed text, plus page and section where
available.

The annotation log records:

- source, candidate, and final annotation hashes;
- reviewer and role;
- active minutes excluding breaks;
- accepted, edited, rejected, and added steps;
- accepted, edited, rejected, and added constraints;
- unresolved decisions and their evidence.

### 5. Freeze and validate the primary pass

Run declared-schema, structural, grounding, provenance, attachment, and relation checks.
Freeze the pass and its log before any comparison with a blind annotation.

### 6. Complete the blind subset

At least 25% of experiment-eligible procedures are selected before results are inspected.
The blind annotator works from the same source span and a blank `unreviewed` scaffold.
They must not see the candidate, primary pass, audit packet, or another pass.

Accidental exposure invalidates that assignment and requires reassignment.

### 7. Preserve raw agreement

Score and preserve every preregistered pair before adjudication. Report low-agreement
pairs rather than excluding them. Attachment-edge F1 of at least 0.70 is the current G0
gate; Cohen's kappa and other agreement metrics remain diagnostics.

### 8. Adjudicate independently

A third human resolves every disagreement against source evidence and audits rare,
implicit, prohibited, emergency, and negative-region cases. Escalate only the remaining
scientific decisions to the principal investigator. Record every escalation and outcome.

## First review order

1. EPA MFC calibration: use
   `docs/annotation/manual-review/2026-07-13-epa-mfc-calibration.md` after it is committed.
2. EPA procedure validation: resolve the four decisions already listed in
   `docs/annotation/manual-review/2026-07-10-epa-validation.md`.
3. NIOSH Method 2005: rebuild the bounded 542-word procedure from the July 11 audit.
4. Remaining manifest-selected candidates: audit one bounded source at a time.

This order creates one small end-to-end reviewer packet before scaling recruitment.

## Definition of done

- [ ] The manifest is frozen and assigns each experiment-eligible procedure.
- [ ] Every included procedure has a complete primary human source pass.
- [ ] Every accepted step and constraint has exact source offsets.
- [ ] Every primary pass has a complete time and edit log.
- [ ] At least 25% has a source-only blind second pass selected in advance.
- [ ] Every selected raw pair and pre-adjudication report is preserved.
- [ ] A different human adjudicated every disagreement.
- [ ] Principal-investigator escalations are limited and logged.
- [ ] Candidate, primary, blind, adjudication, and final artifacts remain distinguishable.
- [ ] Declared schema, grounding, evidence, manifest, and experiment gates pass.

## Explicitly insufficient

- Running `scripts/sign_off_gold.py` only to append a marker.
- Treating `review_status = "reviewed"` as semantic verification.
- Treating agent audit as a human first pass.
- Correcting a blind pass after viewing the primary pass and then reporting agreement.
- Discarding a selected pair because its agreement is low.
- Expanding the corpus before the two-document protocol and method pilot work.
