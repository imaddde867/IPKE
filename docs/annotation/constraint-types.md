# IPKE Constraint Type Taxonomy

Locked vocabulary for `constraint.type` and `constraint.enforcement` in
`datasets/paper/gold/*.json` and `datasets/paper/production/*.json`. All annotators must
use these values; any deviation invalidates the annotation for paper-grade evaluation.
The retained `gold/` files are agent-audited candidates, not human-verified production
gold.

## Design rationale

The taxonomy is intentionally tight. Six types cover every constraint observed across the
eight-file legacy candidate corpus. This is empirical coverage, not proof that the
taxonomy is closed. The split into a type axis and an orthogonal enforcement axis
captures the requirement, recommendation, and permission distinctions expressed through
modal verbs.

If an item does not fit one of the six types, flag it rather than forcing a label. The
independent adjudicator decides whether it is not a constraint or exposes a taxonomy gap.
Only an unresolved gap is escalated to the principal investigator before any vocabulary
change. Silent drift is forbidden.

## Type axis

| Type | Definition | Examples (source text excerpts) |
|---|---|---|
| `precondition` | A condition that must hold, or an action that must already have occurred, before the step can begin. | "After all systems are calibrated and all equipment is operating, the telephone and modems connections are tested." (EPA CASTNET, S3) — calibration must precede communications test.<br>"Attach brackets first to the vertical profiles, then to the top and bottom" (OLSK, S4) — brackets-first is an ordering precondition. |
| `postcondition` | A condition that must hold, or an artifact that must exist, after the step completes. | "Record the graduation value (the HOLD) in the Hold column of the water-level measurement field form." (USGS GWPD 1, S4) — produces a field-form entry. |
| `guard` | A condition that must hold *during* the step, or a hazard that must be actively avoided. Includes warnings, prohibitions, and active checks. Negative form ("do not …") is a guard with enforcement=must. | "Wear gloves, lab coat, and safety glasses while handling chemicals." (NIOSH 5022, procedure-level)<br>"Pay attention to top and bottom of the panel." (OLSK, S2)<br>"Do not use plastics other than fluorocarbon polymers" (USGS NFM, S2) |
| `parameter` | A quantitative value, range, tolerance, categorical choice, or rate that the step must satisfy. | "Sample at an accurately known flow rate between 1 and 3 L/min for a total sample volume of 50 to 1000 L." (NIOSH 5022, S2)<br>"SOPs should be systematically reviewed on a periodic basis, e.g. every 1-2 years." (EPA QA/G-6, S3)<br>"Locate the first sampling vertical at a distance of one-half of the selected increment width from the edge of the water." (USGS NFM, S5) |
| `role_assignment` | Specifies who performs, authorises, signs off, or is responsible for the step. | "Mission Directorate Associate Administrators shall ensure that program and project Safety and Mission Assurance (SMA) Plans satisfy …" (NASA NPR 8715.3D, S1)<br>"The QA Manager (or designee) is generally the individual responsible for maintaining a file listing all current quality-related SOPs." (EPA QA/G-6, S4) |
| `reference` | Pointer to an external standard, document, or section that the annotator believes the executor must consult to complete the step correctly. Not a definition or motivating commentary. | "Microbiological analyses: Collect samples for microbiological analyses using equipment and techniques described in NFM 7." (USGS NFM, S2) — explicit external-doc consultation required. |

## Enforcement axis

| Enforcement | Source-text signal | Examples |
|---|---|---|
| `must` | "shall", "must", "is required", "will", imperative form without softener, prohibition ("do not"). | "Project managers **shall** ensure that hazards … are controlled" — `role_assignment` / must. |
| `should` | "should", "recommended", "preferred", "best practice", non-mandatory imperative. | "SOPs **should be** written with sufficient detail …" — `parameter` / should. |
| `may` | "may", "can", "is acceptable", "is permitted", optional clause. | "use of electronic signatures … is **an acceptable substitution** for paper" — `guard` / may. |

When the source verb is ambiguous, choose the best supported enforcement value and record
the evidence, uncertainty, and alternatives. Do not default automatically to `must`.

## Mapping from the seed-corpus ad-hoc types

This is the historical migration table for the eight legacy candidates. It does not
replace source review in a production annotation.

| Ad-hoc type used in draft/review | Locked type | Locked enforcement | Notes |
|---|---|---|---|
| `precondition` | `precondition` | `must` | identity |
| `postcondition` | `postcondition` | `must` | identity |
| `warning` | `guard` | `should` | cautionary |
| `prohibition` | `guard` | `must` | text already contains "do not" / "must not" |
| `approval` | `precondition` | `must` | sign-off must precede next step |
| `requirement` | MANUAL | `must` | most common; must read each to classify into precondition / postcondition / guard / parameter / role_assignment |
| `documentation` | `postcondition` | `must` | produces a record |
| `role_assignment` | `role_assignment` | `must` | identity |
| `tolerance` | `parameter` | `should` | quantitative latitude |
| `parameter` | `parameter` | `must` | identity (if used) |
| `selection_rule` | `parameter` | `must` | categorical choice |
| `location_rule` | `parameter` | `must` | geometric/positional |
| `storage` | `parameter` | `must` | physical condition (temp, container) |
| `review_cycle` | `parameter` | `must` | temporal cadence |
| `order` | `precondition` | `must` | X-before-Y ordering |
| `recommendation` | `guard` | `should` | non-mandatory imperative |
| `guideline` | `guard` | `should` | non-mandatory advice |
| `permission` | `guard` | `may` | allowed alternative |
| `reference` | `reference` | `must` | identity |
| `definition` | — DROP — | — | not a constraint; meta-commentary |
| `purpose` | — DROP — | — | not a constraint; explanatory annotation |

## Quality bar

- Every constraint MUST have `type` ∈ the six locked types.
- Every constraint MUST have `enforcement` ∈ {must, should, may}.
- Every constraint MUST have `attached_to` listing at least one step ID (or `applies_to` for procedure-level constraints).
- Every constraint text MUST be drawn from the source verbatim or near-verbatim (annotator may correct OCR artifacts, normalise whitespace, expand contractions; may NOT paraphrase or summarise).
- Production annotations MUST carry exact end-exclusive Unicode source offsets for every
  accepted step and constraint. The retained candidates predate this requirement and
  carry section-level provenance only; see the datasheet.

## Relationship to PAGED and Carriero & Celino

**PAGED does encode constraint-to-action edges** — this must be stated
accurately, because the differentiation is not "we have constraint edges and
they don't." PAGED (Du et al., ACL 2024; 3,394 documents) represents two
constraint element types — `DataConstraint` and `ActionConstraint` — and links
them to actions via `Constraint Flow` edges (5,807 of them), using fixed
relation templates such as *"For {Action}, pay attention to that
{ActionConstraint}"* and *"{Action} require access to {DataConstraint}"*.

The IPKE evaluation design differs on four axes that a reviewer can verify,
none of which is "presence of constraint edges":

1. **Semantic typing depth.** PAGED collapses all constraints into
   data-vs-action (2 types). The IPKE taxonomy distinguishes six —
   precondition, postcondition, guard, parameter, role_assignment,
   reference — which separates, e.g., a *guard* ("do not proceed until
   pressure < 5 bar") from a *parameter* ("torque to 12 N·m") that PAGED's
   `ActionConstraint` would merge.

2. **The enforcement (deontic) axis — PAGED has none.** IPKE scores an
   orthogonal must / should / may grade drawn from the modal verbs of the
   source ("shall" / "should" / "may"). In a regulated SOP this is the line
   between a violation and a suggestion; no procedural-graph benchmark
   currently measures it. It is a differentiating axis of the evaluation
   design, not a novelty claim by itself.

3. **Attachment scored by exact step-id F1, not text overlap.** PAGED evaluates
   its textual elements (actor / action / constraint) with a BLEU-based
   surface-form score. IPKE scores the *attachment edge itself* —
   whether the constraint is bound to the correct step id — as strict + fuzzy
   precision/recall. We measure the binding; PAGED measures the string.

4. **Gold provenance and domain.** PAGED states expert annotation at its scale
   is too costly and derives its gold from a pre-existing BPMN business-process
   model collection (Dumas et al., 2018) plus WikiHow-trained segmentation and
   hand-written templates — i.e. template-derived silver over business-process /
   WikiHow-style text. The IPKE corpus targets real regulated
   safety-critical SOPs (EPA / NASA / NIOSH / USGS), but its current annotations
   are agent-audited candidates and human verification is pending.
   Small-and-verified is a deliberate counterpart to large-and-templated only
   after that verification lands; until then it is small-and-source-grounded
   (see the datasheet).

A finding *inside* PAGED supports the IPKE direction directly: PAGED reports that
LLMs reach state-of-the-art on extracting element *text* (actor/action/
constraint) but stay below ~0.5 F1 on gateway/flow *structure* prediction.
Extracting constraint text is easy; getting the logical binding right is where
models fail — which is precisely the edge the IPKE evaluation isolates.

Carriero & Celino 2024 extracts steps, actions, objects, equipment, and temporal information but does not represent guards, role assignments, or parameter constraints. The IPKE taxonomy is a strict superset of their schema with respect to constraint coverage.

The taxonomy is additive to the existing benchmark map and is supporting
infrastructure for the IPKE method study, not the paper's primary contribution.

### Capability comparison (drop-in for Related Work)

| Axis | PAGED (ACL 2024) | Carriero & Celino 2024 | **IPKE corpus (candidate)** |
|---|---|---|---|
| Constraint types | 2 (data / action) | 0 (no constraint schema) | **6 typed** |
| Enforcement (must/should/may) | ✗ | ✗ | **✓** |
| Attachment metric | BLEU text overlap | — | **exact step-id F1 (strict + fuzzy)** |
| Gold provenance | template-derived silver (BPMN + WikiHow) | expert (small) | **agent-audited candidates; human verification pending** |
| Domain | business process / WikiHow | recipes / how-to | **regulated safety SOPs** |
| Scale | 3,394 docs | small | **8 candidate docs (5 manifest-included)** |

## Change log

- 2026-06-13 — initial lock based on observed types across 8 agent-reviewed seed candidates (PR #85).
- 2026-10-23 — correct verification and contribution claims after the method-first reset.
