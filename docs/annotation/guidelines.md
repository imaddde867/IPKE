# IPKE-Bench Annotation Guidelines

Authoritative procedure for producing a paper-grade gold file. Independent second-pass annotators MUST follow this document. Any deviation invalidates the inter-annotator agreement (IAA) computation for that file.

## Read first

- [`constraint-types.md`](constraint-types.md) — locked taxonomy. Memorise the six types and three enforcement levels before annotating.
- [`../../CONTEXT.md`](../../CONTEXT.md) — repo vocabulary (PKG, DSC, P3, Φ).
- [`../../schemas/ipke_annotation.schema.json`](../../schemas/ipke_annotation.schema.json) — JSON schema. Run validation after every save.

## Annotation environment

Open the source `.txt` file and the target `.json` file side by side. Do NOT use the rendered PDF — use the extracted text that the pipeline operates on. The whole point of the benchmark is that the extractor sees the same text the annotator does.

For PR reviewers: do NOT rely on the LLM-drafted gold (`unreviewed` status) as a starting point. Either start from a blank schema-valid skeleton or use a different annotator's pass — never anchor to the draft text.

## Annotation scope

Every gold file currently uses `quality.annotation_scope = "bounded_excerpt"`. This means the annotation covers a clearly delimited section of the source (named in `procedure.source.section`), not the whole document. The bounded scope is intentional for the seed corpus: it keeps annotation cost tractable and lets the benchmark target specific procedural complexity (multi-step + multi-constraint sections) without dragging in unrelated front-matter.

When adding a new gold file to expand the corpus, prefer bounded_excerpt scope of 1-3 pages with at least 4 steps and 6 constraints in the chosen section. If a full procedure is small enough to annotate end-to-end (e.g. a 2-page maintenance checklist), use `annotation_scope = "full_procedure"` and document why.

## Step identification

A **procedural step** is one ordered, actionable operation that an agent must perform. Heuristics:

- The step is something an executor *does*. Information-only paragraphs are not steps.
- A numbered list item ("1. Open the well.") is almost always a step.
- An imperative verb beginning a sentence ("Calibrate each personal sampling pump …") is usually a step.
- Declarative sentences embedded in step descriptions ("The data logger is checked and all operations verified prior to departure of the Station Initiation Team") translate to a step IF they describe an action; otherwise treat them as constraints on a neighbouring step.

### When to split a step

Split when source text introduces a logically separate phase of work, even if the source bullets them under one heading. NASA NPR 8715.3D §1.5.1 is one source paragraph but the SMA-plan-coverage activity and the milestone-review-topics activity are separate procedural steps with different deliverables.

### When NOT to split

Sub-bullets that enumerate parts of a single action (eliminate / reduce-likelihood / reduce-severity / improve-state-of-knowledge in NASA §1.7.1.1) belong in `arguments` of one parent step, not separate steps.

## Constraint identification

A **constraint** is a condition, guard, parameter, role assignment, or external reference that governs one or more steps. Constraints are the safety-and-correctness scaffolding around the procedural backbone — they are what IPKE-Bench is centrally about.

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

When the source is ambiguous, default to `must` and add a note to `quality.review_notes` flagging the ambiguity. Independent second annotators will diverge most on `should` vs `must` calls — be conservative.

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

The seed corpus averages 2.7 constraints per step (119 / 43).

## What to drop

The following are NOT constraints, even when they look like one:

- **Definitions**: "Validation is the confirmation by examination and the provision of objective evidence …" — informative meta, not a binding condition. Capture in commentary if useful; do NOT add to the constraint set.
- **Purpose statements**: "A wetted chalk mark will identify that part of the tape that was submerged." — explains WHY a step exists; not a separate binding condition.
- **Examples**: "Examples of risk reduction strategies include …" — illustrative, not enforceable.
- **Historical notes**: "Per the Government Paperwork Elimination Act of 1998 …" — permits something but the constraint is the permission itself; cite the act only inside the constraint text.

Exception: an annotator may keep a definition or purpose as a `reference` constraint if the source explicitly cites it as required reading for executing the step.

## Independence rule (CRITICAL for IAA)

A second-pass annotator MUST NOT look at any other annotator's gold file for the same document until their own annotation is complete and committed. Looking at the gold breaks IAA — κ becomes meaningless.

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
- `review_notes: "<1-3 sentence summary of changes from draft, plus any ambiguity flags>"`

## Worked example

The `datasets/paper/gold/nasa_npr_8715_3d_general_safety.json` and `datasets/paper/gold/epa_guidance_preparing_sops_qag6.json` files are the canonical worked examples for these guidelines. Read them end-to-end before annotating your first file.

## Migration provenance

When the seed corpus was migrated from the original 20 ad-hoc constraint types to the locked 6-type vocabulary (2026-06-13 sprint), each ambiguous `requirement` entry was manually classified. The classification map is preserved in `docs/annotation/requirement-classifications.json` and the template script that generated it is `docs/annotation/requirement-classifications.template.json`. These two files document the per-constraint decisions; future annotators should consult them when an existing seed-corpus constraint's type is questioned. The migration script that applied them is `scripts/migrate_constraint_types.py`.

## Change log

- 2026-06-13 — initial guidelines drafted from the seed corpus annotation pass (PR #85).
