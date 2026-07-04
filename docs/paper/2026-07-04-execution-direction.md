# IPKE Execution Direction - 2026-07-04

This document is the active working direction for the ECIR 2027 IPKE-Bench paper
push. It supersedes older roadmap language that still treats gold review as open
or suggests creating a separate artifact repo now.

## Framing

The paper direction is **IPKE-Bench as the primary contribution**: a
constraint-aware benchmark, annotation taxonomy, validator, and evaluation
protocol for procedural knowledge extraction from safety-critical industrial
documents.

IPKE remains the reference local/private extraction pipeline and the strongest
baseline. It is important, but it is not the paper's primary contribution.

The CoRe link should be stated narrowly and defensibly. IPKE-Bench supports the
family of TeoAly/ADINO pilots that turn unstructured procedural, operator, or
document content into structured representations, or use a procedure as a
reference for checking observed work. It does not claim to underlie unrelated
vision or time-series pilots.

## Ground Truth

- 8 seed gold files are reviewed and strict-validator clean.
- D1 constraint-blindness is the motivating result and must stay reproducible.
- The tracker is now reduced to granular active work; stale umbrella/duplicate
  issues are closed with comments pointing to replacements.
- Do not create a new repo now. Keep hardening this repo and defer packaging
  decisions until the P0 evaluation and IAA gates are under control.

## Active GitHub Issues

### P0 - Do First

| Issue | Owner mode | Why it matters |
|---|---|---|
| #89 Pin D1 constraint-blindness reproduction | ready-for-agent | Locks the paper's Section 1 motivating result. |
| #55 Run multi-seed and multi-model evaluation sweep | ready-for-agent | Produces real paper numbers with CIs. |
| #90 Recruit independent annotators for IAA gate | ready-for-human | Required for any defensible kappa claim. |

### P1 - Paper-Grade Artifact and Analysis

| Issue | Owner mode | Notes |
|---|---|---|
| #87 Expand corpus to 12 documents | ready-for-human | Needs genre diversity, not just count. |
| #93 Track unmatched paper text status | ready-for-human | Resolve `niosh_nmam_surface_sampling_guidance` ambiguity. |
| #98 Finalize dataset datasheet | ready-for-human | Existing datasheet needs final license/status polish after #87. |
| #66 Beta sensitivity for DSC | ready-for-agent | Run after DSC/DP setup is stable. |
| #67 Phi weight sensitivity | ready-for-agent | Run from stored sweep outputs after #55. |
| #96 PAGED metric comparison | ready-for-agent | One benchmark-map comparison row. |
| #97 JSON-LD export | ready-for-agent | Reusability and ESWC/Semantic-Web angle. |

### P2 - Optional Differentiators

| Issue | Owner mode | Notes |
|---|---|---|
| #95 Optional D3 constraint-aware retrieval | ready-for-agent | Do only if D2 lands early; cut before D2 under time pressure. |
| #99 Expert human study | ready-for-human | Nice-to-have, not a Resource Track blocker. |
| #100 Finnish-language extension | ready-for-human | Only if suitable Finnish SOPs are cleared. |

## What Is Not Active

- Old umbrella issue #60 is closed. It contained stale gold-review blockers.
- Duplicate annotator issue #86 is closed in favor of #90.
- Separate artifact-repo planning #54 is closed. Package later, after P0.
- Pipeline cleanup #78 is closed/deferred. It is not in the paper critical path.
- #91, #92, and #94 are closed/deferred to keep the board focused on the paper
  direction.
- `create_ipke_issues.sh` is no longer part of the workflow. It can be deleted.

## Immediate Work Order

1. Start #89 and keep it small: make `make eval-blindness` reproduce the pinned
   8-document D1 result and fail if counts drift.
2. Start #55 only after #89 is pinned enough that the result source is stable.
   The first milestone is one real-model, non-empty metrics row. Do not start
   with the full grid.
3. In parallel, push #90 with humans: confirmed annotators, assignment list,
   and blind second-pass process.
4. Only then expand to #87 and #93. Corpus growth without IAA discipline weakens
   the paper.

## Local Hygiene

Work on a branch. Do not work directly on `main`. Do not rewrite reviewed gold
or report JSONs by hand. If a result is wrong, fix the script and regenerate it.
