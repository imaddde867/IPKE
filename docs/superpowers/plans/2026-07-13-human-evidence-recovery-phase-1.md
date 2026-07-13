# Human Evidence Recovery Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stamp-only sign-off with the approved tiered human-evidence protocol and deliver the first source-to-candidate audit plus a separate agent-prepared review artifact without mutating legacy annotation gold.

**Architecture:** The method-paper design remains the research authority. A separate human-evidence design defines roles, evidence states, and gates; active context and annotation workflow documents consume that design. Manual-review records and namespaced packets contain source-grounded proposals. Agent-prepared candidates live outside immutable legacy gold and cannot enter evaluation.

**Tech Stack:** Markdown, JSON source inspection, SHA-256, JSON Schema, existing IPKE validators, GitHub Issues.

## Global Constraints

- Models and agents produce candidates only and never human verification.
- One independent human completes a full source pass for every production procedure.
- At least 25% of experiment-eligible procedures receive a source-only blind second pass.
- A person who did not annotate the procedure adjudicates disagreements.
- The principal investigator resolves only remaining taxonomy, implicit-evidence, or safety-critical disputes.
- Every accepted step and constraint must resolve to exact committed-source character offsets.
- Preserve candidate JSON, blind passes, raw agreement, annotation logs, and adjudication decisions as separate artifacts.
- Do not edit `datasets/archive/` or stamp any current candidate as human verified.
- Do not run a confirmatory sweep while the manifest is provisional or human-evidence gates fail.

---

### Task 1: Record the controlling protocol

**Files:**
- Create: `docs/superpowers/specs/2026-07-13-human-evidence-recovery-design.md`
- Create: `docs/superpowers/plans/2026-07-13-human-evidence-recovery-phase-1.md`
- Modify: `CONTEXT.md:86-108`

**Interfaces:**
- Consumes: ADR-0005 and `docs/superpowers/specs/2026-07-10-ipke-method-paper-design.md`.
- Produces: one normative role and evidence-state contract referenced by active project context.

- [x] **Step 1: Add the approved design**

Record the decision, role separation, evidence states, per-procedure workflow,
annotation-efficiency measurements, corpus strategy, first work packet, gates, and
non-goals. State explicitly that routine transcription is not a principal-investigator
responsibility and agent work cannot establish human evidence.

- [x] **Step 2: Update active context**

Replace marker-only Gold Annotation and threshold-only IAA definitions with these
requirements:

- immutable agent candidate;
- independent complete primary source pass;
- exact item-level evidence anchors;
- primary-pass timing and edit log;
- at least 25% source-only blind second annotation;
- frozen raw agreement for every selected pair;
- independent adjudication;
- limited and logged principal-investigator escalation;
- frozen-manifest membership and all deterministic gates.

- [x] **Step 3: Check terminology**

Run:

```bash
rg -n "stamp-only|>=30%|≥30%|all κ|Imad performs|human-verified:<handle>" \
  CONTEXT.md docs/superpowers/specs/2026-07-13-human-evidence-recovery-design.md
```

Expected: no active-context text treats a marker or agreement threshold as sufficient
evidence.

- [x] **Step 4: Commit**

```bash
git add -f docs/superpowers/specs/2026-07-13-human-evidence-recovery-design.md \
  docs/superpowers/plans/2026-07-13-human-evidence-recovery-phase-1.md
git add CONTEXT.md
git commit -m "Define human evidence recovery protocol"
```

### Task 2: Replace the active annotation workflow

**Files:**
- Modify: `docs/methods/annotation-pipeline.md`
- Modify: `docs/annotation/SIGN_OFF_ISSUE.md`
- Modify: `docs/annotation/independent-annotator-workflow.md`
- Modify: `datasets/paper/README.md`
- Modify: `REPRODUCIBILITY.md`
- Modify: `docs/reproducibility.md`
- Modify: `Makefile` comments for candidate and blind-pass terminology
- Modify: `docs/annotation/guidelines.md`
- Modify: `docs/annotation/constraint-types.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: Human Evidence Recovery Design roles and gates.
- Produces: instructions usable by candidate auditors, primary reviewers, blind annotators, adjudicators, and the principal investigator.

- [x] **Step 1: Replace the pipeline diagram**

Use this normative sequence:

```text
frozen source and bounded procedure
  -> immutable agent draft and audit candidate
  -> independent human complete source pass
  -> exact anchors and annotation log
  -> source-only blind second pass for at least 25%
  -> frozen pre-adjudication agreement
  -> independent adjudication
  -> principal-investigator escalation only for unresolved scientific decisions
  -> production annotation and paper-evidence gate
```

Retitle the 256-step and 231-constraint inventory as a dated legacy-candidate snapshot.
Do not call `adjudicate.py replay` a route to production gold.

- [x] **Step 2: Replace stamp-only sign-off instructions**

Keep `docs/annotation/SIGN_OFF_ISSUE.md` at the same path for incoming links, but change
its purpose to independent production annotation. Remove the instruction that
`sign_off_gold.py` can establish evidence by appending a marker. Define complete-pass,
anchor, logging, blind-subset, agreement-freeze, adjudication, and escalation completion
criteria.

- [x] **Step 3: Correct blind-annotation instructions**

Make accidental candidate exposure invalidate and reassign that document. Resolve the
assignment list from the frozen manifest and IAA subset instead of a hard-coded table.
Start scaffolds as `unreviewed`, require item-level offsets, remove automatic `must`,
record active minutes, preserve every preregistered raw pair, and adjudicate after scoring.

- [x] **Step 4: Align corpus and reproduction documentation**

Distinguish the artifact roles without inventing directories: agent candidate, primary
human pass, blind pass, adjudication record, and final production annotation. State that
the current five manifest-selected files are development candidates and that validation
must eventually check primary-pass provenance, anchors, logs, blind coverage, raw
agreement, adjudication, and frozen membership.

- [x] **Step 5: Check contradictions**

Run:

```bash
rg -n ">=30%|≥30%|Imad performs|sign_off_gold.py|all κ|κ >=|κ ≥|draft gold" \
  docs/methods/annotation-pipeline.md \
  docs/annotation/SIGN_OFF_ISSUE.md \
  docs/annotation/independent-annotator-workflow.md \
  docs/annotation/guidelines.md docs/annotation/constraint-types.md \
  datasets/paper/README.md REPRODUCIBILITY.md docs/reproducibility.md Makefile CLAUDE.md
```

Expected: any match is explicitly historical or diagnostic, never a production
eligibility rule.

- [x] **Step 6: Commit**

```bash
git add docs/methods/annotation-pipeline.md \
  docs/annotation/SIGN_OFF_ISSUE.md \
  docs/annotation/independent-annotator-workflow.md \
  datasets/paper/README.md REPRODUCIBILITY.md
git commit -m "Align annotation workflow with human review"
```

### Task 3: Deliver the EPA MFC reviewer packet

**Files:**
- Create: `docs/annotation/manual-review/2026-07-13-epa-mfc-calibration.md`
- Create: `datasets/paper/review_candidates/epa_field_operations_manual_filter_sampling_sop.json`
- Create: `datasets/paper/review_packets/epa_field_operations_manual_filter_sampling_sop.json`
- Create: `schemas/ipke_review_packet.schema.json`
- Create: `tests/test_epa_review_candidate.py`
- Inspect only: `datasets/paper/gold/epa_field_operations_manual_filter_sampling_sop.json`
- Inspect only: `datasets/paper/text/epa_field_operations_manual_filter_sampling_sop.txt`
- Inspect only: `datasets/paper/second_pass/_source/epa_field_operations_manual_filter_sampling_sop.txt`

**Interfaces:**
- Consumes: exact source range `609373:611246` and the current candidate annotation.
- Produces: a source-grounded proposed correction set and a compact human decision packet. It does not produce gold.

- [x] **Step 1: Verify the frozen span**

Confirm that the Unicode substring and blind-source file are identical and record SHA-256
`6c14cfbdf5c380067a83528013f68182204ec0418db8ccb6de13e04204a5bf70`.

- [x] **Step 2: Record the source action spine**

Record source items 6.1.1 through 6.1.10 and the proposed 14-step source-faithful spine.
Propose merging candidate S8-S11, deleting unsupported S12, preserving S13-S18 subject
to the terminal-action decision, and rebuilding the NEXT chain.

- [x] **Step 3: Record constraint corrections**

Separate action restatements from genuine parameters, guards, preconditions, and
postconditions. Record high-confidence proposals and isolate these human decisions:

- drop or retain illustrative C2;
- narrow the 6.1.7 parameter clauses;
- attach the form-computation postcondition;
- keep or merge S16-S18;
- treat Figure 4 as provenance or a reference constraint;
- accept the 14-step short-procedure exception;
- finalize IDs only after semantic decisions.

- [x] **Step 4: Record provenance defects**

Record actual version `Revision No. 10 (February 2025)`, source pages 4 to 6 of 6,
missing item-level offsets, stale adjudication metadata, stale zero-constraint notes, and
legacy null page fields now tolerated only by the candidate-compatible schema.

- [x] **Step 5: Validate without editing gold**

Run declared JSON Schema validation, the custom strict validator, exact-grounding and
packet-reconciliation tests, and the human-evidence gate. Expected: candidate validation
passes; the human-evidence gate fails because no primary-human output exists. Confirm
`git diff -- datasets/paper/gold` is empty.

- [x] **Step 6: Commit**

```bash
git add docs/annotation/manual-review/2026-07-13-epa-mfc-calibration.md
git commit -m "Audit EPA MFC calibration candidate"
```

### Task 4: Enforce the exact-anchor evidence boundary

**Files:**
- Create: `schemas/ipke_annotation_evidence.schema.json`
- Create: `src/evaluation/evidence.py` production assessor
- Create: `datasets/paper/evidence/README.md`
- Modify: `schemas/ipke_annotation.schema.json`
- Modify: `scripts/validate_paper_gold.py`
- Modify: `scripts/eval_multiseed.py`
- Modify: `Makefile` and active evidence documentation

- [x] **Step 1: Repair candidate-schema compatibility**

Allow `page: null`, apply the constraint definition to embedded constraint arrays, and
prove all eight retained candidates validate without making exact item offsets optional
for production.

- [x] **Step 2: Add the evidence sidecar schema**

Bind source, bounded span, candidate use, primary pass, decisions, optional blind and
adjudication records, and final annotation by SHA-256. Keep participant identifiers
pseudonymous and require selected blind assignments to include agreement and independent
adjudication records.

- [x] **Step 3: Add the production assessor**

Leave the high-blast-radius marker assessor unchanged. Add a separate boundary that
checks raw-byte hashes, Unicode code-point offsets, procedure containment, item decision
coverage against the loaded candidate, duplicate identifiers, link integrity, decision
spans, timestamps, canonical artifact paths, every referenced artifact hash, unresolved
decisions, and actor separation.

- [x] **Step 4: Wire both execution gates**

Make the paper validator and direct multiseed runner require matching sidecars unless an
explicit development-only override is used. Keep candidate structural validation and
dry-run behavior backward-compatible.

- [x] **Step 5: Verify and commit**

Run targeted tests, the full non-integration suite, Ruff, the expected failing paper
gate, GitNexus change detection, and confirm no candidate or archive annotation changed.

Verified on 2026-07-13: 328 tests passed with 13 integration tests deselected; Ruff and
candidate validation passed; the paper gate failed closed on the five absent production
files; GitNexus reported medium scope across the two expected evidence-validation flows;
legacy gold, archives, and blind-pass files had no diff.

## Self-review

- Spec coverage: role separation, evidence states, 25% blind coverage, independent
  adjudication, limited principal-investigator escalation, exact anchors, structured
  annotation logs, corpus strategy, and first audit each have an implementation task.
- Placeholder scan: no TODO, TBD, generic error-handling request, or unspecified code
  step remains.
- Type consistency: candidate, primary pass, blind pass, adjudication, and production
  annotation names match the controlling design.
