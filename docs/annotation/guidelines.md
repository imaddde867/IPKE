# IPKE Evaluation Annotation Guidelines

Authoritative decision procedure for production annotations. Primary reviewers, blind
annotators, and adjudicators must follow it. Role and evidence requirements live in
`../methods/annotation-pipeline.md`.

## Read first

- [`constraint-types.md`](constraint-types.md) — locked taxonomy. Memorise the six types and three enforcement levels before annotating.
- [`../../CONTEXT.md`](../../CONTEXT.md) — repo vocabulary (PKG, DSC, P3, Φ).
- [`../../schemas/ipke_annotation.schema.json`](../../schemas/ipke_annotation.schema.json) — JSON schema. Run validation after every save.

## Annotation environment

Open the source `.txt` file and the target `.json` file side by side. The extracted text
is the annotation input because it is what the extractor sees. Consult the authoritative
PDF only to resolve page provenance, tables, symbols, or OCR ambiguity; record any such
decision in `review_notes`.

Primary human reviewers may use a frozen candidate for assistance, but must inspect the
complete source span and add omissions. Source-only blind annotators must start from a
blank `unreviewed` scaffold and may not view any candidate or another pass. Adjudicators
work only after both passes and raw agreement are frozen.

## Annotation scope (LOCKED 2026-07-04 — supersedes the seed-corpus bounded_excerpt rule)

> **The scope rule.** Annotate **exactly one coherent, complete procedure** — the smallest self-contained unit of work a practitioner would execute in one sitting, from its first actionable step to its terminal step — **end to end, with every constraint in that unit**. Target **15–40 steps**. Do not truncate a procedure to hit a step count, and do not stitch two unrelated procedures together to reach one.

`quality.annotation_scope = "full_subprocedure"` for all golds produced under this rule. `procedure.source.section` MUST name the exact sub-procedure boundary (e.g. `"6.3 Filter Sample Collection (6.3.1–6.3.11)"`), so the unit is reproducible from the source text alone.

### Why this replaces the old rule

The seed corpus used `bounded_excerpt` with a floor of "at least 4 steps and 6 constraints." That floor optimised for annotation cost and produced thin 4–9-step fragments (mean 5.4 steps; see `audit_summary.md`). Thin fragments cannot demonstrate the paper's headline contribution — **constraint attachment at procedure scale** — and their arbitrary excerpt boundaries make IAA scope-dependent and hard to defend. A single interactive strong-model pass on the *same* filter-sampling document produced 30 steps / 64 constraints when scoped to one complete SOP sub-procedure, versus the 6-step gold. That 5× depth at the same validator contract is the target.

### How to pick the unit

1. **Find the procedural spine.** Scan the document's table of contents / section headings for the finest-grained heading under which the text is a *continuous ordered sequence of actions* (numbered steps, imperative sentences). In a manual this is typically one third-level SOP section (e.g. `6.3.x`), one NMAM method, or one named field procedure — never the whole manual, never the whole chapter.
2. **Prefer the unit that is richest in constraints**, not merely longest. The benchmark is about constraints-on-steps; choose the sub-procedure whose steps carry the most guards, parameters, preconditions, and role assignments.
3. **Take the whole unit.** Once chosen, annotate every step and every constraint in that unit. Do NOT stop at 40 — if the natural unit is 45 steps, annotate all 45 and note it. Do NOT pad to 15 by absorbing the next unrelated section.
4. **One gold file = one procedure.** If a document contains several equally good candidate procedures, pick the single most representative one for the seed gold; additional procedures from the same document become separate corpus-expansion golds, not additions to this file.

### Edge cases

- **Requirements / policy documents.** A normative requirements block is not an
  executable procedure. Exclude it from confirmatory procedural evaluation rather than
  converting clause order into steps or `NEXT` edges. It may be retained in a separately
  labeled `requirements_stress_test` collection only after its representation and
  evaluation protocol are defined.
- **Very short standalone procedures (< 15 steps but genuinely complete).** Acceptable only if the procedure is truly self-contained end-to-end (e.g. a complete 12-step calibration checklist). Keep the whole thing, set `annotation_scope = "full_subprocedure"`, and note in `review_notes` that the natural unit is < 15 steps. Never split a longer procedure down to this size.

## Step identification

A **procedural step** is one ordered, actionable operation that an agent must perform. Heuristics:

- The step is something an executor *does*. Information-only paragraphs are not steps.
- A numbered list item ("1. Open the well.") is almost always a step.
- An imperative verb beginning a sentence ("Calibrate each personal sampling pump …") is usually a step.
- Declarative sentences embedded in step descriptions ("The data logger is checked and all operations verified prior to departure of the Station Initiation Team") translate to a step IF they describe an action; otherwise treat them as constraints on a neighbouring step.

### When to split a step

Split when source text introduces a logically separate phase of executable work, even if
the source bullets actions under one heading. Do not split coordinated objects,
qualifiers, purposes, or parts-list nouns into actions.

### When NOT to split

Sub-bullets that enumerate objects or components of one action belong in `arguments` or
structured parameters of one parent step, not separate steps.

## Constraint identification

A **constraint** is a condition, guard, parameter, role assignment, or external reference that governs one or more steps. Constraints are the safety-and-correctness scaffolding around the procedural backbone and the primary IPKE method-evaluation target.

### Source signals that indicate a constraint

- Modal verbs: "shall", "must", "should", "may", "will".
- Imperative clauses inside or adjacent to a step description: "Pay attention to …", "Do not …", "Remove protective foils.", "Wear gloves …".
- Quantitative or categorical specifications: "between 1 and 3 L/min", "every 1-2 years", "with blue carpenter's chalk".
- Role/actor naming: "Project managers shall ensure …", "The QA Manager is generally responsible …".
- "If … then" clauses that bind to a step: "If the SOP describes a process that is no longer followed, it should be withdrawn from the current file and archived."

### Choosing the type

Use the decision sequence in order; stop at the first match.

1. Does the constraint say "BEFORE this step, X must hold or have happened"? → `precondition`.
2. Does the constraint say "AFTER this step completes, X must hold or be produced"? → `postcondition`.
3. Does the constraint specify WHO performs, signs off, or is responsible? → `role_assignment`.
4. Does the constraint give a quantitative range, tolerance, categorical choice, or rate? → `parameter`.
5. Does the constraint point to an external standard/document that must be consulted? → `reference`.
6. Otherwise (cautionary, prohibition, active check during the step): → `guard`.

### Choosing the enforcement level

Read the modal verb in the source.

- "shall" / "must" / "will" / "is required" / unmodified imperative ("Do not …") → `must`.
- "should" / "recommended" / "preferred" → `should`.
- "may" / "can" / "is acceptable" / "is permitted" → `may`.

When the source is ambiguous, choose the best evidence-supported value and flag the exact
source span and alternatives. Do not default automatically to `must`. Routine
disagreement is resolved after both passes are frozen; only unresolved scientific cases
are escalated to the principal investigator.

### Attachment

Every constraint MUST be attached to at least one step via:

- `constraint.attached_to: [step_id, …]` when the constraint lives inside a single step's `constraints` array, OR
- top-level `constraint.applies_to: [step_id, …]` when the constraint applies to multiple steps (e.g. procedure-level safety constraints like "wear gloves throughout").

A constraint with zero attachment is invalid and must be either dropped or attached. Detached constraints are the single biggest source of low Constraint Attachment F1 in baseline extractors — annotators must not introduce them.

## Verbatim wording rule

Constraint text MUST be drawn from the source verbatim or near-verbatim. The annotator MAY:

- correct OCR artifacts (mojibake, broken hyphens, missing whitespace),
- normalise capitalisation when the source mid-sentence breaks it,
- expand cross-document references inline (e.g. "(NPR 8705.4)" stays inline; do not summarise).

The annotator MAY NOT:

- paraphrase to "make the sentence cleaner",
- summarise multi-sentence constraints into one,
- introduce wording not in the source.

The reason: the IPKE-Bench evaluation uses both exact-match and fuzzy semantic alignment (cos ≥ 0.75) for constraint comparison. Paraphrased gold widens the gap between gold and any extractor that returns the source-faithful text — including human annotators in the second pass. This was the root cause of the κ = 0.531 → 0.430 drop on OLSK in the seed corpus.

## Step-constraint cardinality

A real procedural step usually has 1–6 attached constraints. A step with 0 constraints is suspicious — re-read the source to confirm that nothing was missed. A step with > 10 constraints is suspicious — consider whether it should be split into multiple steps.

The pre-reannotation seed corpus averaged 2.7 constraints per step (117 / 43) under the old thin-excerpt scope; this figure is a lower bound and will be recomputed after the full-subprocedure re-annotation. Cardinality is a per-step sanity check, not a corpus-level target — do not drop real constraints to hit an average.

## What to drop

The following are NOT constraints, even when they look like one:

- **Definitions**: "Validation is the confirmation by examination and the provision of objective evidence …" — informative meta, not a binding condition. Capture in commentary if useful; do NOT add to the constraint set.
- **Purpose statements**: "A wetted chalk mark will identify that part of the tape that was submerged." — explains WHY a step exists; not a separate binding condition.
- **Examples**: "Examples of risk reduction strategies include …" — illustrative, not enforceable.
- **Historical notes**: "Per the Government Paperwork Elimination Act of 1998 …" — permits something but the constraint is the permission itself; cite the act only inside the constraint text.

Exception: an annotator may keep a definition or purpose as a `reference` constraint if the source explicitly cites it as required reading for executing the step.

## Independence rule (CRITICAL for IAA)

A blind annotator must not view a candidate, primary pass, audit packet, adjudication
record, or another annotator's pass for the same procedure until their own annotation is
complete, frozen, and committed. Accidental exposure invalidates the assignment and
requires reassignment.

If you need clarification on annotation procedure, ask via these guidelines or post a question on the relevant issue. Do NOT email another annotator to ask "how did you do this section."

## Saving and validating

After every edit:

```bash
uv run python3 -c "
import json, jsonschema
schema = json.load(open('schemas/ipke_annotation.schema.json'))
d = json.load(open('datasets/paper/gold/<file>.json'))
jsonschema.validate(d, schema)
print('PASS')
"
```

Set the following fields in `quality` before considering the file done:

- `review_status: "reviewed"`
- `annotator: "<your-name>"`
- `review_date: "<YYYY-MM-DD>"`
- `review_notes: "<1-3 sentence summary of changes from draft, plus any ambiguity flags>"` — to adjudicate a strict-validator warning, embed the exact token `step:{id} {kind} adjudicated` (e.g. `step:S1 zero_constraints adjudicated`) in this field; one token per warning suppressed.

These fields record that a review pass occurred; they do not establish paper eligibility.
Only a human who personally checks the final annotation may append
`+ human-verified:<handle>`, and the file must also belong to the frozen confirmatory
split.

Every accepted step and constraint also requires exact end-exclusive Unicode
`char_start` and `char_end` offsets into the authoritative committed source. The primary
pass needs source, candidate, and final hashes; active minutes; and separate accepted,
edited, rejected, and added counts for steps and constraints. A marker without these
records is not evidence.

## Worked examples

### Constraint typing and structure

No current candidate is a canonical worked example until the full production-human
protocol is complete. Use the declared JSON Schema, taxonomy, and examples in this
guideline as the decision contract. The July 2026 source audits under `manual-review/`
show why structural validity and agent review are insufficient.

### Scope selection (the locked rule in practice)

**Document:** `epa_field_operations_manual_filter_sampling_sop.txt` (~108k words — a whole field-operations manual containing dozens of SOPs, 6.1.1 → 6.6.8).

- **WRONG (old bounded_excerpt rule):** annotate sections 6.3.3–6.4.1 as a 6-step / 19-constraint excerpt. This straddles the tail of one SOP and the head of the next, hits the "4-step floor," and stops arbitrarily. Result: a thin fragment with boundaries no second annotator would independently reproduce.
- **RIGHT (locked rule):** identify the finest heading that is one continuous ordered action sequence — SOP **6.3 "Filter Sample Collection"** — and annotate it end-to-end: `procedure.source.section = "6.3 Filter Sample Collection (6.3.1–6.3.11)"`, every collection step (mount holder → load filter → set flow rate → run → record → recover), every guard ("do not touch the filter face"), every parameter ("flow rate 1–3 L/min", "total volume 50–1000 L"), every postcondition ("record the sample ID and volume on the chain-of-custody form"). Target 15–40 steps; take the whole unit even if it lands at 27.

The boundary is **semantic and reproducible from the source alone**: any annotator asked for "SOP 6.3, collection procedure" delimits the same span. That is what makes the IAA defensible.

**Scope selection worksheet** (record in `review_notes` for each new gold): (1) candidate sub-procedure heading + line range; (2) why it is the richest coherent unit; (3) final step count and whether it fell outside 15–40 (with justification).

## Migration provenance

When the seed corpus was migrated from the original 20 ad-hoc constraint types to the locked 6-type vocabulary (2026-06-13 sprint), each ambiguous `requirement` entry was manually classified. The classification map is preserved in `docs/annotation/requirement-classifications.json` and the template script that generated it is `docs/annotation/requirement-classifications.template.json`. These two files document the per-constraint decisions; future annotators should consult them when an existing seed-corpus constraint's type is questioned. The migration script that applied them is `scripts/migrate_constraint_types.py`.

## Change log

- 2026-06-13 — initial guidelines drafted from the seed corpus annotation pass (PR #85).
- 2026-07-04 - **scope rule locked.** Replaced the seed-corpus `bounded_excerpt` floor ("≥ 4 steps / 6 constraints") with the full-subprocedure rule (one coherent complete procedure, 15–40 steps, `annotation_scope = "full_subprocedure"`). The model-assisted harness and agent decision replay produced eight historical candidates, not human-adjudicated gold.
- 2026-07-13 - Added the complete primary-human pass, source-only blind subset, independent adjudication, exact-anchor, annotation-log, and limited PI-escalation requirements.
