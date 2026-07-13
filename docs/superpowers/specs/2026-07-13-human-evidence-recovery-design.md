# Human Evidence Recovery Design

Status: approved for implementation on 2026-07-13

## Decision

Remove routine annotation transcription from the principal investigator's role without
weakening the evidence standard. Use a tiered Draft-Blind-Adjudicate workflow for the
supporting evaluation corpus:

1. models and agents may draft, audit, repair, and evidence-anchor annotation candidates;
2. one independent human performs a complete source pass for every production procedure;
3. at least 25% of experiment-eligible procedures receive a source-only blind second pass;
4. a person who did not annotate that procedure adjudicates disagreements;
5. the principal investigator resolves only remaining taxonomy, implicit-constraint, or
   safety-critical disputes;
6. no model or agent may mark its own work as human verified or experiment eligible.

The workflow supports the method-first paper defined by ADR-0005. It does not restore the
superseded benchmark-first framing.

## Why this is necessary

The current candidate corpus is useful but not paper evidence. It contains model-assisted,
agent-adjudicated files with no named human verification and no eligible independent blind
second pass. Asking the principal investigator to transcribe or re-annotate every procedure
has stopped progress and spends scarce expert time on mechanical work.

Replacing human judgment with model-generated gold would make the evaluation circular.
The chosen workflow instead moves drafting, consistency checking, evidence lookup, and
decision preparation away from the principal investigator while preserving independent
human judgment at the evidence boundary.

## Roles

### Models and agents

- Produce candidate steps, constraints, attachments, relations, and evidence anchors.
- Audit every candidate against the frozen source span.
- Apply only high-confidence candidate corrections with explicit provenance.
- Flag uncertain, implicit, cross-sentence, rare-type, and safety-critical cases.
- Produce a compact decision packet for human review.
- Run deterministic validation and preserve raw candidate history.
- Never sign, adjudicate as a human, or promote a candidate to gold.

### Primary human reviewer

- Read the complete bounded source procedure, not only the suggested annotations.
- Add omissions and reject unsupported candidate content.
- Verify step identity and granularity, constraint identity, type, enforcement,
  attachment, order, relation type, and evidence span.
- Record review time and candidate accept, edit, reject, and add decisions.
- Sign only after personally checking the final annotation.

The reviewer may be recruited or paid. The principal investigator does not need to be the
primary reviewer.

### Blind second annotator

- Work from the same frozen source span without seeing the candidate, first pass, or
  adjudication record.
- Complete the annotation independently before reveal.
- Cover at least 25% of the final experiment-eligible procedures, selected before results
  are inspected.

### Adjudicator

- Must not have annotated the same procedure.
- Resolve every first-pass versus blind-pass disagreement against source evidence.
- Audit rare classes, prohibitions, emergency actions, implicit attachments, and a seeded
  sample of agreements and negative source regions.
- Escalate unresolved scientific decisions to the principal investigator.

### Principal investigator

- Approve the frozen protocol, corpus membership, and experiment split.
- Resolve escalated taxonomy, scope, implicit-constraint, and safety-critical decisions.
- Review go or stop gates and paper claims.
- Does not perform routine transcription.

## Evidence states

The existing states remain separate:

1. **Candidate drafted**: model or agent output with complete provenance.
2. **Structurally valid**: declared schema, identifiers, enumerations, attachments, and
   relations pass deterministic checks.
3. **Source grounded**: every accepted item resolves to a frozen source evidence span.
4. **Manually reviewed**: one human completed the full source pass.
5. **Human verified**: that human signed the final annotation.
6. **Agreement eligible**: a blind second pass and independent adjudication are complete.
7. **Experiment eligible**: all required gates pass and the procedure belongs to the
   frozen split.

No state is inferred from directory placement or `review_status = "reviewed"` alone.

## Per-procedure workflow

1. Freeze the source URL, retrieval date, version, checksum, page range, section identity,
   exact Unicode character offsets, and redistribution status.
2. Freeze one complete bounded procedure, normally 15 to 40 procedural steps.
3. Preserve the raw model-assisted candidate.
4. Run an agent source-to-candidate audit and produce:
   - high-confidence proposed corrections;
   - exact evidence anchors;
   - an uncertainty ledger;
   - a compact human decision packet.
5. Have the primary human review the complete source span and final candidate.
6. Run declared-schema, structure, grounding, attachment, relation, provenance, and
   evidence-eligibility validation.
7. For the preregistered blind subset, collect an independent source-only annotation and
   adjudicate all disagreements.
8. Audit a seeded sample of agreements and source regions with no annotation.
9. Preserve raw passes, corrections, timing, disagreements, adjudication decisions, and
   the final signed file as distinct artifacts.

## Annotation-efficiency measurements

Capture these fields without making them a headline claim until the study design is
frozen:

- reviewer minutes per procedure and per adjudicated surviving item;
- candidate acceptance, modification, deletion, and human-added counts;
- unsupported-candidate rate;
- human-added omission rate;
- evidence-span validity;
- agreement by constraint type and enforcement level;
- performance on rare, implicit, and cross-sentence constraints.

If staffing permits, counterbalance assisted and blind conditions across annotators and
documents. Assistance condition is then analysed as a fixed effect with annotator and
document treated as grouping variables. Document-level resampling is required.

This study is secondary to the C3 versus C2 method contrast unless it is separately
preregistered and sufficiently powered.

## Corpus strategy

- Prefer a smaller fully eligible confirmatory corpus over a larger unsigned corpus.
- Keep agent-generated expansion candidates in a separate development or silver pool.
- Do not add another large manual merely to increase document count.
- Select complete, bounded, rights-screened procedures with source-family and domain
  diversity.
- Rebuild the excluded NIOSH Method 2005 bounded procedure before considering it for
  inclusion.
- Keep the NIOSH surface-sampling artifact development-only because it was used for DSC
  tuning.
- Screen short public candidates from FDA, NPS or USGS SOP collections, CISA playbooks,
  and HSE before acquisition. Genre and reuse eligibility must pass before annotation.

## First work packet

1. Finish and protect the existing manifest-scoped runner change.
2. Record this operating model in the active annotation documentation.
3. Audit the 338-word EPA MFC calibration candidate and create a human decision packet.
4. Repair the declared schema and validation boundary without inventing missing source
   metadata.
5. Rebuild the bounded 542-word NIOSH Method 2005 candidate.
6. Recruit four human contributors so reviewer and adjudicator roles can rotate.
7. Build exact-span C0-C4 execution and run a two-document development pilot.

## Gates

No confirmatory model sweep may begin until the method design's G0 and G1 gates pass.
Specifically:

- every test annotation is human verified;
- at least 25% is independently double-annotated blind;
- attachment-edge agreement F1 is at least 0.70;
- disagreements are independently adjudicated;
- every accepted annotation item is source grounded;
- the declared schema and all deterministic validators agree;
- the experiment split and method configuration are frozen.

## Non-goals

- Calling agent output human gold.
- Asking the principal investigator to transcribe the corpus.
- Treating structural validation as semantic correctness.
- Expanding to twelve documents before the annotation protocol passes a pilot.
- Weakening evidence gates to meet a venue deadline.
- Running the historical P3 versus P0 sweep as confirmatory evidence.
