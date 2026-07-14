# IPKE Execution Direction

Updated: 2026-07-13

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

## Current evidence state

- The retained legacy-candidate inventory has 8 documents, 256 steps, and 231
  constraints. The provisional manifest selects five procedure candidates.
- No production annotation or frozen human-evidence package exists yet.
- All eight retained candidates pass the candidate-compatible JSON Schema and strict
  structural validator. That does not make them production gold.
- The production gate now requires exact source anchors, canonical artifact paths,
  verified hashes across the candidate/primary/blind/adjudication chain, and a complete
  primary-human decision log. A status or marker cannot pass it.
- Most annotations cover a bounded subprocedure while the current multi-seed runner reads
  complete documents. Existing full-document results are not valid method evidence.
- P3 versus P0 bundles conditioning, prompt detail, calls, filtering, and parser effects.
- Explicit gold relations are not yet authoritative in Tier-A and Tier-B evaluation.
- The unit suite is healthy; experiment validity is the critical path.

## Active critical path

| Issue | Owner | Branch state / purpose |
|---|---|---|
| #107 | agent | Direction alignment implemented in draft PR #113. |
| #108 | human | Open: complete independent production-human annotation evidence. |
| #109 | agent | Open: make explicit gold relations authoritative in evaluation. |
| #110 | human | Open: decide redistribution rights for historical Git blobs. |
| #111 | agent | Schema and production-evidence boundary implemented in draft PR #113. |
| #55 | blocked agent | Run the controlled IPKE method sweep after evidence gates. |
| #87 | human | Expand the supporting evaluation corpus with rights-cleared diversity. |
| #90 | human | Recruit independent blind annotators. |

Optional retrieval, JSON-LD, and Finnish extensions remain below the causal method,
signed-data, and evaluation-validity work.

## Immediate work order

1. Complete the EPA MFC primary-human source pass from the agent-prepared candidate and
   namespaced review packet. Eight scientific decisions remain human-owned.
2. Freeze the primary output and evidence package, then prove the production gate on
   that one document.
3. Prepare the next manifest-selected source packet without mutating legacy candidates.
4. Recruit and assign the source-only blind subset and independent adjudicators.
5. Implement #109 before reporting graph-structure metrics.
6. Build one canonical experiment module and prove C0-C4 on two eligible documents.
7. Freeze development and source-family-held-out splits before confirmatory evaluation.
8. Expand compute only after the pilot, parser, truncation, provenance, and cached
   re-scoring gates pass.

## Manual gold order

1. EPA MFC calibration: complete the primary-human pass from
   `datasets/paper/review_candidates/epa_field_operations_manual_filter_sampling_sop.json`.
2. EPA procedure validation: repair high-confidence step, type, enforcement, attachment,
   and branching defects; record scope ambiguities for human adjudication.
3. OLSK: exclude from method evidence until pipeline-visible text contains the assembly
   instructions represented in gold.
4. NASA: retain as a separately labelled requirements stress test without manufacturing
   temporal order.
5. NIOSH: include the omitted SPECIAL PRECAUTIONS safety content and calculation formula.
6. USGS groundwater and surface-water procedures: restore omitted assumptions,
   applicability guards, calculations, and branches.
7. Remaining EPA documents: remove illustrative or descriptive content promoted to
   mandatory actions or constraints.

Agents may prepare source-grounded candidates and decision packets. Production decisions
remain human. Validators check schema, spans, identifiers, hashes, provenance, roles,
splits, and artifact identity.

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
