# IPKE Method Context for Agents

This tracked file keeps its historical filename so existing links continue to resolve.
It is the committed source of truth for agents because the root `AGENTS.md` and
`CLAUDE.md` are personal, gitignored files.

## What this repository delivers

**Primary research contribution:** IPKE, a method for skeleton-conditioned,
source-grounded constraint attachment in procedural graph extraction with local language
models.

**Supporting infrastructure:** the active corpus, annotation taxonomy, schemas,
validators, metrics, and reproducibility tooling.

ADR-0005 supersedes the benchmark-first ECIR Resource Paper direction. Read
`docs/superpowers/specs/2026-07-10-ipke-method-paper-design.md` before changing research
behavior.

## Active causal claim

Test whether step-skeleton-conditioned raw extraction beats call- and budget-matched
self-refinement on document-macro ConstraintAttachmentF1. Filtering is a separate effect.
Segmentation is a separate secondary experiment based on constraint-step co-location and
downstream attachment.

Do not claim novelty for generic two-stage prompting, constraint representation,
constraint-flow edges, or Dual Semantic Chunking.

## Current evidence state

- 8 retained legacy candidates, 256 steps, and 231 constraints; 5 are selected for
  provisional development.
- 0 production annotations and 0 frozen human evidence packages.
- Candidate schema and custom validation pass. Production eligibility separately
  requires exact item anchors, canonical paths, verified artifact-chain hashes, and a
  frozen evidence sidecar.
- The EPA MFC review candidate now has 14 proposed steps, 15 constraints, and exact
  anchors; eight scientific decisions remain for a primary human reviewer.
- Full-document extraction is mismatched with mostly bounded gold spans.
- Explicit gold relations are ignored by parts of the evaluator.
- Existing model-result headlines must not be reused until the controlled protocol is
  rerun.

## Mandatory workflow guards

Before editing any function, class, or method:

1. Run upstream GitNexus impact analysis.
2. Report the blast radius.
3. Warn explicitly before proceeding on HIGH or CRITICAL impact.

Before committing:

1. Run GitNexus change detection.
2. Run the narrowest relevant tests.
3. Run the full non-integration suite before claiming the branch is complete.
4. Run structural and grounding validation after any gold change.

## Gold rules

- Production corrections are human source-to-gold judgments. Agents may prepare
  explicitly ineligible review candidates and namespaced packets.
- Automation may verify schema, spans, hashes, identifiers, provenance, paths, roles,
  splits, and placeholders; it may not assert human decisions.
- Agent review cannot apply `+ human-verified:<handle>`.
- Never align an independent second pass to gold before agreement is computed.
- Never mutate audited files under `datasets/archive/`.
- Do not change the locked constraint type or enforcement vocabularies ad hoc.

## Active critical path

1. #107 method-first repository alignment.
2. #108 primary-human evidence collection and production artifact migration.
3. Frozen blind-subset assignments and coordinator-controlled actor separation.
4. #109 explicit relation evaluation.
5. Two-document exact-span C0-C4 pilot.
6. #87 source-family-diverse corpus expansion and frozen split.
7. #55 full controlled method sweep.

Optional retrieval, JSON-LD, UI, and Finnish extensions follow the causal and evidence
gates.

## See also

- `docs/adr/0005-ipke-method-paper-primary.md`
- `docs/superpowers/specs/2026-07-10-ipke-method-paper-design.md`
- `docs/research-vision.md`
- `docs/paper/2026-07-04-execution-direction.md`
- `CONTEXT.md`
- `REPRODUCIBILITY.md`
- `docs/annotation/guidelines.md`
- `docs/agents/issue-tracker.md`
