# Gold Annotation Methodology

> **Method-first update, 2026-07-11.** This pipeline produces model-assisted candidates
> and review provenance for IPKE evaluation. It does not produce human-verified gold by
> itself. The old ECIR resource-paper language and eight-file outcome below are
> historical. Current exclusions and release criteria live in
> `docs/annotation/SIGN_OFF_ISSUE.md` and `docs/annotation/manual-review/`.

*Model-assisted drafting with explicit provenance and later human verification.* This is
a historical methods draft, not a paper-ready section. Rewrite it after the
confirmatory manifest, manual correction, human verification, and independent
annotation gates are complete.

## Motivation

The IPKE supporting corpus annotates industrial procedures with **constraint
attachment**: each
procedural step carries typed constraints (precondition, postcondition, guard,
parameter, role_assignment, reference), each with an enforcement level (must,
should, may) and an explicit attachment to the step(s) it governs. Producing
this by hand across long technical documents is expensive — a single bounded
procedure of 15–40 steps with attached constraints is roughly three hours of
focused annotator time. Annotating the full corpus by hand, twice (for
agreement), was not feasible within the project timeline.

The historical pipeline used **model-assisted drafting + agent adjudication**. A
separate named-human verification gate remains incomplete; agent review is not gold
authorship.

## Pipeline overview

```
 source .txt
     │
     ▼
 (1) segment_procedures.py ──► candidate procedure spans (char ranges, step estimates)
     │        human selects ONE bounded 15–40 step procedure per document
     ▼
 (2) annotate_assisted.py ──► draft gold  (review_status="unreviewed")
     │        two-stage prompt, structural self-validation + repair loop
     ▼
 (3) adjudicate.py ──► agent-adjudicated candidate (accept / edit / reject, logged)
     │        independent source-grounded pass; decisions persisted for replay
     ▼
 (4) validate_paper_gold.py --strict ──► locked-taxonomy + IAA-metadata gate
     │
     ▼
 (5) setup_iaa_subset.py + compute_iaa.py ──► ≥30% double annotation, F1 + κ
```

Each stage is a committed script under `scripts/`; the committed golds under
`datasets/paper/gold/` are the source of truth, and `make gold-pipeline`
re-runs the deterministic validation + motivating-numbers gate on a fresh clone.

### (1) Segmentation

`segment_procedures.py` scans a document for procedure boundaries (numbered
headings, method markers, step lists) and emits candidate spans with a
char-range and an *upper-bound* step estimate. The estimate is an upper bound
because spans bleed into adjacent narrative — one candidate labelled "Leak Test"
absorbed an entire appendix of stacked SOPs (~80 apparent steps). **Final
procedure boundary selection is a human judgment call**, anchored on where a
genuine, coherent, end-to-end procedure lives and capped to a window
(~9,000–11,000 chars) so the drafting model sees exactly one procedure. The
selected section per document is recorded in the gold's `procedure.source`.

### (2) Model-assisted drafting

`annotate_assisted.py` runs a two-stage prompt (P3: structure extraction, then
constraint attachment) over the selected span. The model backend is injectable —
`host.llm` in-session, or any OpenAI-compatible endpoint via
`IPKE_LLM_BASE_URL` / `IPKE_LLM_MODEL` / `IPKE_LLM_API_KEY` — so the pipeline is
not tied to one provider. The harness performs:

- **hyphenation repair** on born-digital text extraction artifacts;
- **structural self-validation** against the locked taxonomy (types,
  enforcement levels, attachment resolvability) with a bounded repair loop
  (`--max-retries`, default 3);
- emission of a draft with `review_status="unreviewed"` and
  `annotator="model-assisted:<model>"`, plus `quality._draft_diagnostics`.

A draft is explicitly **not** gold. An adjudication pass may set
`review_status="reviewed"`, but that status is not human verification and cannot make
the file paper eligible.

### (3) Adjudication

`adjudicate.py` turns a draft into an agent-adjudicated candidate through an explicit
accept / edit / reject decision on every element. Two modes matter:

- **`review`** — interactive, element-by-element human sign-off.
- **`replay`** — deterministic re-application of a persisted decision log
  (`datasets/paper/adjudication_decisions/<doc>.json`), so a gold can be
  regenerated from its draft + decisions with no interactive step.

For the historical candidate set we ran a **source-grounded agent critic pass**: a
reviewer (distinct from the drafting step) read the source span and returned
accept/edit/reject decisions — typically 2–4 edits and 0–2 rejects per document
(type reclassifications, enforcement corrections, and rejection of the rare
hallucinated or duplicated constraint). Rejects drop; edits overwrite named
fields; surviving constraints are re-embedded under the first surviving attached
step (orphans move to top-level and are logged). The engine strips
`_draft_diagnostics`, sets `review_status="reviewed"`, and writes the full
decision log plus a tally into `quality.review_notes`.

The resulting `annotator` field is stamped honestly:
`model-assisted:<model> + agent-adjudicated (pending human sign-off)`. This is
not claimed as independent human authorship. The genuine human–human agreement
figure comes from the second-pass workflow (§5), and Imad performs a final
human sign-off before submission.

### (4) Validation gate

`validate_paper_gold.py --strict` enforces the contract: `review_status ==
"reviewed"`, annotator and review_date set; every constraint has a locked type,
a valid enforcement level, non-empty text, and at least one attachment that
resolves to a real step id. Strict mode also flags steps with zero or >10
constraints; genuine low-density steps are acknowledged with an honest
per-step adjudication token in `review_notes` rather than by inventing
constraints. All eight legacy candidates pass structural `--strict` validation, which
does not imply semantic correctness or human verification.

### (5) Inter-annotator agreement

Historically, `setup_iaa_subset.py` selected a stratified **≥30% subset** (3 of 8
documents, one per distinct domain family). That subset is paused because it contains
excluded NASA. After issue #112 replaces and freezes it, the tool emits **blank** second-pass scaffolds
carrying only `doc_id`, procedure title, and the source char span — deliberately
**no first-pass steps or constraints**. This is the anchoring-bias control: the
second annotator authors from the same source window *without* reading the
first-pass gold, which is what gives the reported κ statistical meaning. It also
writes the exact source span to `second_pass/_source/<doc>.txt` so both passes
cover identical scope. `compute_iaa.py` then scores step F1, constraint F1,
relation F1, and token-label Cohen's κ; `setup_iaa_subset.py report` wraps it
and scores only completed subset docs so stale files cannot pollute the run.

## Corpus statistics (current release)

Eight bounded procedures under the locked `full_subprocedure` scope
(15–40 steps each):

| Document | Domain | Steps | Constraints |
|---|---|---:|---:|
| epa_field_operations_manual_filter_sampling_sop | field_operations | 18 | 9 |
| epa_field_sampling_measurement_procedure_validation | quality_assurance | 35 | 41 |
| epa_guidance_preparing_sops_qag6 | quality_assurance | 36 | 34 |
| nasa_npr_8715_3d_general_safety | safety_requirements | 39 | 29 |
| niosh_nmam_5th_edition_ebook | analytical_chemistry | 34 | 21 |
| usgs_groundwater_technical_procedures_tm1_a1 | hydrology | 29 | 12 |
| usgs_nfm_collection_water_samples_a4 | hydrology | 36 | 45 |
| olsk_small_cnc_v1_workbook | mechanical_assembly | 24 | 8 |
| **Total** | **8 domains** | **251** | **199** |

**Constraint type distribution:** postcondition 60, parameter 42, guard 38,
precondition 29, reference 16, role_assignment 14.
**Enforcement distribution:** must 172, should 42, may 17 (after the 2026-07-06
verbatim-grounding pass realigned enforcement to the guidelines' modal-verb
mapping).

Constraint density is genuinely source-dependent, not a defect: calibration and
assembly procedures (MFC calibration 12, OLSK electronic-box 9) carry far
fewer constraints than QA-governance and field-sampling procedures (EPA
validation 44, USGS NFM 63). This variance is recorded in `review_notes`, not
smoothed away.

For continuity: the pre-release thin golds covered the same 8 documents at 43
steps / 117 constraints total; the re-annotation under the locked scope, after
the 2026-07-06 source-verbatim grounding and completion pass (mis-located spans
corrected, missed constraints added, restatements dropped — per-file log in
each gold's `review_notes`), reaches **256 steps / 231 constraints** — a 6.0×
increase in step depth — while the old bounded-excerpt golds are preserved
under `datasets/paper/gold_v1_bounded_excerpt_archive/` and the before/after
counts in `datasets/paper/gold_depth_comparison.json`.

## Edge cases and how they were handled

- **NASA NPR 8715.3D (requirements/policy doc)** — not temporally ordered like
  an SOP; each *shall/should* requirement is treated as a step. Noted in
  `review_notes` so the structure is not mistaken for a defect.
- **EPA procedure-validation (short standalone, <2k words)** — annotated
  end-to-end as a single procedure rather than a sub-section.
- **OLSK CNC workbook (mechanical assembly)** — intrinsically constraint-light;
  low density is expected and documented, not corrected upward.
- **9th document (niosh_nmam_surface_sampling_guidance)** — deliberately **out
  of scope**: it is guidance/background with no bounded 15–40 step procedure.
  The deliverable is 8 documents.

## Threats to validity

- **Model-authorship of golds.** Mitigated by the unreviewed-until-adjudicated
  gate, the honest `annotator` stamp, independent human sign-off, and the
  independent second pass for agreement. The κ figure is the check that the
  golds are reproducible rather than one process's opinion.
- **Anchoring bias in the second pass.** Mitigated by blank scaffolds and the
  source-span-only working set; the workflow forbids opening the gold before
  commit (`docs/annotation/independent-annotator-workflow.md`).
- **Scope-selection subjectivity.** The bounded-procedure boundary is a human
  choice; it is made explicit and reproducible via the recorded
  `procedure.source` char range and section label.

## Reproducing this

```bash
# Deterministic gate (no model needed) — validate the 8 golds + regenerate D1 numbers:
make gold-pipeline

# Re-draft one procedure (needs a model backend):
make gold-draft DOC=<doc_id> SEG=<segments.json> CAND=<candidate_id>

# Deterministically re-apply a persisted adjudication log to a draft:
make gold-adjudicate DOC=<doc_id>

# Prepare the ≥30% IAA subset + blank scaffolds, then score once second passes exist:
make iaa-setup
make iaa
```

See also: `docs/annotation/guidelines.md` (annotation decision procedure),
`docs/annotation/constraint-types.md` (locked vocabulary),
`docs/annotation/independent-annotator-workflow.md` (second-pass protocol),
`docs/reproducibility.md` (environment + evaluation).
