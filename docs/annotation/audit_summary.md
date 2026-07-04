# IPKE-Bench Gold Audit — Pre-Reannotation Baseline

**Date:** 2026-07-04
**Purpose:** Establish the exact validator contract and the measured state of the 8 seed golds before the model-assisted re-annotation pass. This is the frozen "before" record for the methodology section.

## 1. The validator contract (what a gold MUST satisfy)

Source of truth: `scripts/validate_paper_gold.py` + `src/benchmark/taxonomy.py`. A paper-grade gold passes iff:

- `quality.review_status == "reviewed"`, and `quality.annotator` and `quality.review_date` are set.
- Every constraint (both step-embedded `steps[].constraints[]` and top-level `constraints[]`) has:
  - `type` ∈ **{precondition, postcondition, guard, parameter, role_assignment, reference}** (locked 6-type vocab)
  - `enforcement` ∈ **{must, should, may}**
  - non-empty `text`
  - at least one `attached_to` **or** `applies_to` ref, and every ref resolves to a real step `id`.
- **Warnings** (fail only under `--strict`, suppressible via a `review_notes` token `step:{id} {kind} adjudicated`):
  - a step with 0 attached constraints (`zero_constraints`)
  - a step with >10 attached constraints (`too_many_constraints`)

Note the validator imports `src.benchmark.taxonomy`, so it must be run with the repo root on `PYTHONPATH` (`uv run` does this; a bare `python3 scripts/...` needs `PYTHONPATH=.`).

### Required JSON shape (from canonical example `nasa_npr_8715_3d_general_safety.json`)

- Top keys: `procedure`, `steps`, `constraints`, `relations`, `quality`.
- `procedure`: `{doc_id, title, version, domain, source:{doc_id, page, section}}`.
- `steps[]`: `{id, label, action_verb, action_object, arguments[], parameters[], constraints[], flags{}, provenance{}}`.
- constraint: `{id, type, text, attached_to:[step_id...], enforcement}`.
- `quality`: `{annotation_scope, review_status, annotator, review_date, review_notes}`.

## 2. Measured state of the 8 golds (the "thin fragment" problem)

**All 8 currently PASS the validator, including `--strict`.** They are *valid*, not *broken*. The problem is **scope**, not correctness.

| doc | steps | constraints | c/step | 0-constraint steps | src words | scope | section |
|---|---|---|---|---|---|---|---|
| usgs_groundwater_technical_procedures_tm1_a1 | 9 | 14 | 1.56 | 2 | 61,859 | bounded_excerpt | GWPD 1 Instr 1– |
| epa_field_operations_manual_filter_sampling_sop | 6 | 19 | 3.17 | 0 | 107,912 | bounded_excerpt | 6.3.3–6.4.1 |
| niosh_nmam_5th_edition_ebook | 6 | 12 | 2.00 | 1 | 368,683 | bounded_excerpt | SAMPLING & SAMPLE PREP |
| epa_field_sampling_measurement_procedure_validation | 5 | 15 | 3.00 | 0 | 1,710 | bounded_excerpt | 3.2.1–5.1 |
| usgs_nfm_collection_water_samples_a4 | 5 | 24 | 4.80 | 0 | 72,753 | bounded_excerpt | EWI sampling steps |
| epa_guidance_preparing_sops_qag6 | 4 | 17 | 4.25 | 0 | 14,837 | bounded_excerpt | 2.0 SOP Process |
| nasa_npr_8715_3d_general_safety | 4 | 9 | 2.25 | 0 | 13,796 | bounded_excerpt | 1.5.1–1.7.1.1 |
| olsk_small_cnc_v1_workbook | 4 | 7 | 1.75 | 0 | 2,201 | bounded_excerpt | 01.1–01.4 |
| **TOTAL / MEAN** | **43** | **117** | **2.72** | **3** | — | all bounded_excerpt | — |

Constraint **type** distribution (n=117): postcondition 34, parameter 33, guard 26, precondition 13, role_assignment 9, reference 2.
Constraint **enforcement** distribution: must 84, should 33, may 0.

## 3. Root cause (confirmed in writing)

The thin golds were **not** annotator error. `docs/annotation/guidelines.md` (§ Annotation scope) contained the written rule:

> *"prefer bounded_excerpt scope of 1-3 pages with at least 4 steps and 6 constraints in the chosen section."*

Every gold obeyed that floor precisely (min observed: 4 steps). The rule optimised for *tractable annotation cost*, which was correct for a first pass, but it is in direct tension with the paper's headline contribution — **constraint attachment at procedure scale**. A 4-step excerpt cannot demonstrate rich constraint-attachment structure, and the arbitrary excerpt boundaries make IAA scope-dependent and hard to defend to reviewers.

**Calibration reference:** an interactive Claude Opus pass on `epa_field_operations_manual_filter_sampling_sop` (same doc as the 6-step gold) produced **30 steps / 64 constraints** when scoped to one *complete* SOP sub-procedure — 5× the depth at the same validator contract. That is the target density.

## 4. Corpus gaps flagged for the re-annotation step

- **9 source texts, 8 golds.** `niosh_nmam_surface_sampling_guidance.txt` has **no gold**. Decision needed in Step 6: annotate it (→ 9-doc corpus) or exclude it from the seed set with a documented reason.
- **`may` enforcement is entirely absent** (0/117). Either the corpus genuinely has no permissions, or `may`-constraints were dropped. The harness prompt must explicitly probe for permitted/optional clauses so this is a measured absence, not an artifact.
- **NASA NPR 8715.3D** is a policy/requirements document; it may not contain a single 15–40-step *executable* procedure. Flagged for a scope-shape decision during re-annotation (candidate for `full_procedure` on a bounded requirements block, or exclusion).

## 5. Scope decision to encode in guidelines.md (Step 2)

Replace the "4 steps / 6 constraints minimum" rule with: **annotate one coherent, complete procedure or SOP sub-procedure end-to-end (target 15–40 steps).** New golds use `annotation_scope = "full_subprocedure"`, with `procedure.source.section` naming the exact sub-procedure boundary. The unit is *semantic* (one self-contained procedure a practitioner would execute in one sitting), not a page count.
