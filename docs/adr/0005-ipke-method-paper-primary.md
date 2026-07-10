# ADR-0005: Make the IPKE Method the Primary Contribution

Status: accepted

Date: 2026-07-10

Supersedes: ADR-0004

## Context

ADR-0004 made IPKE-Bench and an ECIR Resource Paper the primary direction. A subsequent
manual source-to-gold audit, repository audit, and updated literature review found that
the resource framing would hide the more valuable research question while relying on
evidence that is not yet publication ready.

The active gold files are structurally consistent under the custom validator but remain
unsigned, fail the declared JSON Schema, and cover only bounded portions of source
documents that the current runner processes in full. The P3 versus P0 comparison also
bundles step conditioning with prompt detail, calls, filtering, and parser behavior.

Prior work already represents action-constraint links and uses multi-stage procedural
graph extraction. The defensible method question is narrower: whether explicit
step-skeleton conditioning improves fine-grained, source-grounded constraint attachment
under local-model and inference-cost constraints.

## Decision

IPKE itself is the primary contribution. The paper will lead with skeleton-conditioned,
source-grounded constraint attachment for procedural graph extraction with local language
models.

The corpus, taxonomy, annotation guidelines, schemas, validators, and evaluation harness
remain required supporting infrastructure. They are not positioned as the paper's primary
contribution.

Hierarchy-aware segmentation is secondary and claimable only if a separate controlled
evaluation demonstrates improved constraint-step co-location and downstream attachment.

The controlling design is
`docs/superpowers/specs/2026-07-10-ipke-method-paper-design.md`.

## Consequences

- The ECIR Resource Paper plan is no longer active.
- Existing benchmark-first documentation and issues must be superseded or rewritten.
- No headline sweep runs until gold eligibility and causal controls are corrected.
- ConstraintAttachmentF1 and grounded constraint recall become primary outcomes.
- Phi becomes secondary unless redesigned and validated.
- Full-document segmentation and exact-span extraction are evaluated separately.
- Optional application surfaces are kept outside the research kernel.
- Human sign-off remains human-only; agent review cannot satisfy it.

## Required follow-up

1. Align the README, research vision, execution direction, AGENTS guidance, and 2ndBrain.
2. Create a human-owned gold sign-off issue and block paper experiments on it.
3. Implement explicit evidence-eligibility states.
4. Manually repair gold defects and source coverage one document at a time.
5. Replace conflicting experiment runners with one provenance-complete experiment module.
6. Run a two-document controlled pilot before any full compute matrix.
