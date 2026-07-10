# IPKE Execution Direction

Updated: 2026-07-10

This file originally described the ECIR Resource Paper push. ADR-0005 superseded that
direction. It is retained at the existing path because issues and repository entry points
already link here.

## Framing

IPKE itself is the primary paper contribution. The method under test is
skeleton-conditioned, source-grounded constraint attachment for procedural graph
extraction with local language models.

The corpus, taxonomy, annotation workflow, validators, and metrics are supporting
evaluation infrastructure. They must be strong enough to support the method claim but
are not framed as a benchmark-paper contribution.

Controlling design:
`docs/superpowers/specs/2026-07-10-ipke-method-paper-design.md`.

## Current ground truth

- The active corpus has 8 documents, 256 steps, and 231 constraints.
- No active gold is human verified. Every annotation still carries pending sign-off.
- The custom strict validator passes, but all active files currently fail the declared
  JSON Schema because null page values are disallowed.
- Most annotations cover a bounded subprocedure while the current multi-seed runner reads
  complete documents. Existing full-document results are not valid method evidence.
- P3 versus P0 bundles conditioning, prompt detail, calls, filtering, and parser effects.
- Explicit gold relations are not yet authoritative in Tier-A and Tier-B evaluation.
- The unit suite is healthy; experiment validity is the critical path.

## Active critical path

| Issue | Owner | Purpose |
|---|---|---|
| #107 | agent | Align durable repository direction with ADR-0005. |
| #108 | human | Complete manual human sign-off for active gold. |
| #109 | agent | Make explicit gold relations authoritative in evaluation. |
| #110 | human | Decide redistribution rights for historical Git blobs. |
| #111 | agent | Align JSON Schema and the strict gold-validation contract. |
| #55 | blocked agent | Run the controlled IPKE method sweep after evidence gates. |
| #87 | human | Expand the supporting evaluation corpus with rights-cleared diversity. |
| #90 | human | Recruit independent blind annotators. |

Optional retrieval, JSON-LD, and Finnish extensions remain below the causal method,
signed-data, and evaluation-validity work.

## Immediate work order

1. Finish #107 and keep every active entry point consistent with ADR-0005.
2. Add an explicit evidence-eligibility gate so `review_status="reviewed"` cannot enter a
   paper run without human verification.
3. Fix #111 so schema-valid and strict-validator-valid mean the same thing.
4. Correct active gold manually, one source at a time. Do not apply a human marker.
5. Have the human owner complete #108 after personally reviewing the corrections.
6. Implement #109 before reporting graph-structure metrics.
7. Build one canonical experiment module and prove C0-C4 on two corrected documents.
8. Freeze development and source-family-held-out splits before confirmatory evaluation.
9. Expand compute only after the pilot, parser, truncation, provenance, and cached
   re-scoring gates pass.

## Manual gold order

1. EPA procedure validation: repair high-confidence step, type, enforcement, attachment,
   and branching defects; record scope ambiguities for human adjudication.
2. OLSK: exclude from method evidence until pipeline-visible text contains the assembly
   instructions represented in gold.
3. NASA: represent normative requirements without manufacturing temporal order.
4. NIOSH: include the omitted SPECIAL PRECAUTIONS safety content and calculation formula.
5. USGS groundwater and surface-water procedures: restore omitted assumptions,
   applicability guards, calculations, and branches.
6. Remaining EPA documents: remove illustrative or descriptive content promoted to
   mandatory actions or constraints.

Annotation decisions are manual. Scripts may check schema, spans, identifiers, hashes,
provenance, splits, and placeholders only.

## Experiment sequence

### Attachment first

Run C0 Joint/raw, C1 Joint/filtered, C2 Self-refine/raw, C3 Skeleton/raw, and C4 Full
IPKE on the exact annotated spans. The primary contrast is C3 minus C2. Report filter
effects separately through C1 minus C0 and C4 minus C3.

### Segmentation second

Compare full context, fixed windows, heading-only, semantic breakpoints, and
hierarchy-aware segmentation using manually supported boundaries and constraint-step
co-location. Generic cohesion scores are diagnostics, not paper evidence.

### Full system last

Only after component gates pass, run the full model and source-family matrix. Preserve
raw predictions and complete run metadata. Treat documents or source families as the
independent units and seeds as nested repeated measurements.

## Git and release hygiene

- Work on branches and merge through PRs. Only Imad merges.
- Do not push directly to `main`.
- Do not rewrite Git history until #110 has a recorded rights decision and explicit
  approval.
- Do not delete active or archived gold to simplify the corpus.
- Do not edit generated result JSON manually.
- Do not stamp human verification on behalf of the human reviewer.

## Superseded work

- ADR-0004 and the ECIR Resource Paper direction are historical.
- Issue #105 is closed as superseded by #107.
- D1 cross-regime counts are historical annotation-process evidence, not an extractor
  result and not the method-paper headline.
- D3 retrieval, JSON-LD, and cross-lingual extensions are optional follow-up work.
