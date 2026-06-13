# IPKE-Bench Constraint Type Taxonomy

Locked vocabulary for `constraint.type` and `constraint.enforcement` in `datasets/paper/gold/*.json`. All annotators must use these values; any deviation invalidates the annotation for paper-grade evaluation.

## Design rationale

The taxonomy is intentionally tight. Six types cover every constraint observed across the IPKE-Bench seed corpus (8 documents, 117 constraints after the 2026-06-13 standards-review drop of two meta-commentary items, spanning safety regulations, lab SOPs, assembly workbooks, and field-sampling protocols). The 6-type closure is comparable to procedural-knowledge conventions in PAGED (sequential / conditional / parallel / optional) and Carriero & Celino 2024 (steps / actions / objects / equipment / temporal), but specialised to *constraint-on-step* edges rather than step-to-step or step-to-entity edges. The split into a type axis and an orthogonal `enforcement` axis (must / should / may) absorbs the requirement/recommendation/permission distinction that natural-language SOPs make explicitly via modal verbs ("shall" / "should" / "may").

If an annotator encounters a constraint that does not fit one of the six types, the constraint is either (a) not actually a constraint and should be dropped, or (b) flags a genuine gap in the taxonomy that must be raised in a PR before being added — drift is forbidden.

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

Default when source verb is ambiguous: `must`. Annotators must justify any `should` or `may` classification in `review_notes`.

## Mapping from the seed-corpus ad-hoc types

This is the canonical migration table for the 8 reviewed gold files. Apply mechanically except where marked `MANUAL`.

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

## Relationship to PAGED and Carriero & Celino

PAGED operates on step-to-step edges (sequential / conditional / parallel / optional). IPKE-Bench's `precondition` overlaps PAGED's `sequential`; the rest of the taxonomy is orthogonal because PAGED does not encode constraint-on-step edges.

Carriero & Celino 2024 extracts steps, actions, objects, equipment, and temporal information but does not represent guards, role assignments, or parameter constraints. IPKE-Bench's taxonomy is a strict superset of their schema with respect to constraint coverage.

The taxonomy is therefore *additive* to the existing benchmark map and is the methodological contribution of the IPKE-Bench Resource Paper, alongside the dataset itself.

## Change log

- 2026-06-13 — initial lock based on observed types across 8 reviewed seed gold files (PR #85).
