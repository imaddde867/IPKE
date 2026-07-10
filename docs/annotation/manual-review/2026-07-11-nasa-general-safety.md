# NASA General Safety Manual Review

Document: `nasa_npr_8715_3d_general_safety`

Gold: `datasets/paper/gold/nasa_npr_8715_3d_general_safety.json`

Source: `datasets/paper/text/nasa_npr_8715_3d_general_safety.txt`

Status: manual source-to-gold audit complete; no gold changes applied; human sign-off
pending.

## Review method

The source was read directly from lines 870-936 alongside all 39 steps, 26 embedded
constraints, 38 relations, metadata, and the declared source span. Commands were used
only to locate and display source and annotation text. No script generated, classified,
attached, or transformed annotations.

## Scope decision

Section 2.5.2 is a role-partitioned requirements block, not an executable procedure. It
contains 18 lettered requirements under three responsible roles. The current gold splits
coordinated actions, qualifiers, purposes, and object lists into 39 apparent steps, then
turns document order into a 38-edge `NEXT` chain that the source does not assert.

This document must be excluded from the confirmatory procedural-extraction evaluation.
It may remain as a separately reported policy/requirements stress test after correction
and human verification. It must not contribute to the primary procedural skeleton or
constraint-attachment claims.

The declared source span is accurate: character 50391 starts section 2.5.2 at line 870,
and exclusive end 54798 is the first character of section 2.5.3 at line 937.

## High-confidence corrections for a future requirements annotation

| Items | Source lines | Required correction |
|---|---:|---|
| All relations S1-S39 | 882-936 | Delete the 38-edge linear `NEXT` chain. Document order is not execution order. S13 to S14 is the only possible later sequence candidate. |
| S1-S3, C5, C6, C31 | 883-885 | Consolidate PMC approval and both concurrences as one Category I requirement. Use `ensure`; the project manager is not the approving actor. |
| S4-S5, C7 | 886-887 | Consolidate as resource-availability assurance. The out-of-house qualifier applies only to the prime-contractor component. |
| S6-S7, C8, C9 | 888-889 | Consolidate approval and concurrence as one Category I change-control requirement. Use `ensure`. |
| S8, C10 | 890-892 | Keep the conditional requirement. Narrow C10 to the applicability condition; the remaining text is the action. |
| S9, C11, C32 | 893-895 | Keep S9 and C32. Narrow C11 to the coordination role so it does not duplicate the assignment action. |
| S10-S11, C12-C13 | 896-898 | Merge S11 into S10. PRA application is a Category I/II component of required expertise, not a later verification step. Retype C13 as applicability. |
| S12, C14 | 899-900 | Relabel as ensuring funding from permitted sources. Narrow C14 to the funding-source restriction. |
| S13, C15 | 902-903 | Keep S13. Narrow C15 to the formulation-phase phrase and retype it as `parameter`. |
| S14, C16 | 902-903 | Keep S14. Narrow C16 to the lifecycle duration and retype it as `parameter`. |
| S15, C17 | 904-905 | Retype the Tables 2.1 and 2.2 citation as `reference`. |
| S16-S18, C3 | 906-908 | Consolidate as one SSTP-content requirement. The coordinated content objects are not ordered actions. Retype C3 as `postcondition`. |
| S19-S22, C19 | 918-920 | Consolidate the lettered item. Keep its deliverables as arguments and narrow C19 to the project-manager consultation role. |
| S23-S24, C23 | 921-922 | Consolidate or represent as parallel conjuncts, never `NEXT`. Narrow C23 to the SSTP reference. |
| S25, C24 | 923-924 | Keep the integration requirement, but remove C24 if outcome text may not duplicate its step. |
| S26, C25 | 925 | Keep the residual-risk determination requirement and remove C25, which repeats the action. |
| S27-S29, C26-C27 | 926-928 | Consolidate as one compound requirement or two unordered conjuncts. S29 is document scope, not a later step. |
| S30-S33, C28 | 929-931 | Consolidate as one participation requirement with four activity objects. Narrow C28 to level and lifecycle parameters. |
| S34-S36, C29 | 932-935 | Consolidate as one reporting-channel requirement. Status and problem-area coverage are arguments or a channel outcome, not later reporting steps. |
| S37-S39, C30 | 936 | Consolidate as one support requirement with three objects. Remove the duplicate action constraint unless a specific external reference is recoverable. |
| C31, C32, C22 | 882, 893, 901 | Preserve role-assignment type and `must` enforcement, then update attachments after consolidation. |
| Provenance | 866, 914 | Set procedure pages to 22-23 and step pages to 22 for S1-S18 and 23 for S19-S39 if the requirements annotation is retained. |

All 26 `must` enforcement values are grounded by governing `shall` clauses. No complete
normative clause is omitted. The failures are structural and semantic, not missing source
coverage.

## Human adjudication required

1. Decide whether compound requirements are one step or atomic conjuncts linked by a new
   `AND` or parallel relation. The current relation vocabulary cannot express conjunction.
2. Decide whether S13 and S14 remain separate lifecycle actions and, if so, whether they
   justify the sole `NEXT` edge.
3. Standardize Category I/II and `When ...` applicability as `guard` or categorical
   `parameter`.
4. Decide whether a requirement outcome may intentionally duplicate a step as a
   `postcondition`.
5. Add an explicit `requirements_block` scope or genre if this document remains in a
   secondary stress-test collection.
6. Correct the review date only after checking the provenance of the documented review
   pass. Do not add a human-verification marker until a human signs off.

## Release criterion

- Do not include this file in confirmatory method evaluation, independent of annotation
  quality.
- Keep `make eval-paper-gate` failing until human verification is complete.
- If retained as a stress test, manually rebuild the annotation under a requirements
  representation before reporting any metric.
