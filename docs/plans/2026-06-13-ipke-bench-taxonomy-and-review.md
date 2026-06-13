# 2026-06-13 — IPKE-Bench gold review + taxonomy lock + standards review

## Context

Entering this sprint:

- The ECIR 2027 Resource Paper direction was committed in PR #84 (merged earlier in the day).
- 8 paper-gold files (`datasets/paper/gold/*.json`) existed but all had `quality.review_status = "unreviewed"` — they were LLM-drafted skeletons.
- Issue #60 tracked the gold-review blocker.
- The advisor flagged that the prior session's claimed "complete" state was not actually complete; vault notes, ADRs, and the GitHub issue had not been updated.

## Goal

Take the IPKE-Bench seed corpus from LLM-drafted skeletons to paper-grade artifact with a defensible §1 motivating result, a locked annotation methodology, and a paper-grade validator. The benchmark must hold up to ECIR Resource Track artifact review.

## Decisions

### 1. Lock the constraint taxonomy

The first review pass used 20 ad-hoc constraint types (`requirement`, `tolerance`, `selection_rule`, `purpose`, `permission`, …). Advisor flagged: this drift invalidates κ-grade IAA across annotators and weakens the comparison-to-PAGED framing. The taxonomy was locked to **6 types × 3 enforcement levels**:

- Types: `precondition`, `postcondition`, `guard`, `parameter`, `role_assignment`, `reference`.
- Enforcement: `must`, `should`, `may` (mapped from source-text modal verbs).

Captured in `docs/annotation/constraint-types.md` with a mechanical migration table from the 20 ad-hoc types to the locked vocabulary, and worked examples from the seed corpus.

### 2. Annotation guidelines

The locked taxonomy is only enforceable with a written procedure for human annotators. `docs/annotation/guidelines.md` captures:

- Decision sequence for assigning type and enforcement.
- Verbatim-wording rule (no paraphrasing of source).
- IAA independence rule (no looking at gold; no drift-correcting second_pass to match gold).
- Step / constraint cardinality heuristics.
- What to drop (definitions, historical notes, examples).

### 3. IAA methodology correction

Existing `datasets/paper/second_pass/*.json` files appear LLM-drafted, not independent. They were marked `quality.review_status = "llm_draft"` and explicitly excluded from paper IAA computation. The κ ≥ 0.61 acceptance gate is open pending recruitment of independent human annotators. A recruitment memo was drafted in `~/Documents/2ndBrain/Projects/IPKE Paper - Thesis to Congress/07-annotator-recruitment-memo.md`.

### 4. PRD reframe

The previous 5-contribution list (dataset + protocol + baselines + retrieval + reproducibility) was wide for a Resource Paper. Reframed to a singular IPKE-Bench artifact contribution, with three ranked demonstration experiments:

- **D1** constraint-blindness baseline — REQUIRED, done.
- **D2** local baseline sweep — EXPECTED before submission.
- **D3** constraint-aware retrieval — NICE TO HAVE.

Cut order under time pressure: D3 → D2 → never D1.

### 5. Constraint-blindness as §1 result

Cheap finding produced during this sprint: comparing the original LLM-drafted gold (at `origin/main`) against the human-reviewed gold using the Tier-A protocol matcher (SBERT cos ≥ 0.75) gives **3.66× under-production of constraints** (32 vs 117) and 20.5% macro recall. The expansion ratio is matcher-independent and is the durable §1 claim.

A token-Jaccard prototype gave 9.3% recall but biased the result (didn't recognise source-faithful vs paraphrased text as equivalent). Switched to SBERT to match the Tier-A protocol; advisor verified.

## Implementation

### Sprint A — gold review

1. Reviewed all 8 gold files end-to-end against source `.txt` (NASA NPR 8715.3D §1.5.1-1.7.1.1; EPA QA/G-6 §2.0; OLSK CNC §01.1-01.4; EPA CASTNET §6.3.3-6.4.1; EPA LSASD §3.2.1-5.1; NIOSH NMAM Method 5022 SAMPLING+PREP; USGS GWPD 1 Instructions 1-9; USGS NFM EWI §Step 1-2).
2. Expanded constraint coverage 3.66×.
3. Set `review_status="reviewed"`, `annotator="imad"`, `review_date="2026-06-13"`, `review_notes` with delta from draft.

### Sprint B — taxonomy lock

1. `docs/annotation/constraint-types.md` — locked vocab + mapping table.
2. `docs/annotation/guidelines.md` — annotation procedure.
3. `scripts/migrate_constraint_types.py` — reproducible migration.
4. `docs/annotation/requirement-classifications.json` — manual classification of the 55 ambiguous `requirement` entries.
5. Migration applied: 117 constraints retyped, 2 dropped (`definition`, `purpose` — meta).

### Sprint C — validator and §1 baseline

1. `schemas/ipke_annotation.schema.json` — added `enforcement` enum.
2. `scripts/validate_paper_gold.py` — paper-grade validator stricter than schema.
3. `tests/test_validate_paper_gold.py` — 12 unit tests covering all reject paths.
4. `scripts/constraint_blindness_report.py` — D1 baseline using SBERT cos ≥ 0.75 (Tier-A protocol).
5. `datasets/paper/reports/constraint_blindness_v2_sbert{050,075}.json` — committed reports.

### Sprint D — standards review fixes

Two-axis review (Standards + Spec) caught:

- **HARD violations** (4): OLSK `review_notes` proposed drift-correcting second_pass against gold (forbidden); 5 enforcement values contradicted source modal verbs (4 in EPA QA/G-6, 1 in OLSK); EPA QA/G-6 C9 was a historical note that the guidelines explicitly say to drop; OLSK `review_notes` said "warning → tolerance" but file had "parameter".
- **JUDGEMENT calls** (4): USGS NFM C24 single-step procedure-level should embed or extend; `llm_draft` review_status missing from CONTEXT.md vocabulary; dead `constraint_paper_grade` schema $def; no GitNexus impact analysis recorded.
- **Spec violations** (2 HIGH): PRD P0 gate said "12 paper-claimed gold files" but only 8 exist; datasheet was listed in P0 but did not exist.
- **Spec partials** (4): `make eval` not wired to D1; manifest missing `conversion_command` + `review_status` columns; `requirement-classifications.json` not referenced in guidelines; verbatim-wording rule needs spot-check.

All hard + high addressed in commit chain; judgement + partial addressed where straightforward.

### Sprint E — artifact maturity

1. `docs/dataset/datasheet.md` — Gebru-format datasheet covering motivation, composition, collection, preprocessing, uses, distribution, maintenance.
2. `Makefile` — added `eval-validate`, `eval-blindness`, `eval-iaa`; `make eval` now runs validator + D1 before the dry-run sweep.
3. `datasets/paper/public_sources_manifest.csv` — added `conversion_command` + `review_status` columns.
4. `CONTEXT.md` — added locked `review_status`, constraint type, and enforcement vocabularies plus IAA independence rule.

## Resulting state

- 8 gold files reviewed (43 steps, 117 constraints after standards-review drops).
- Locked taxonomy + guidelines committed; migration is auditable and reproducible.
- Paper-grade validator passing (12/12 unit tests).
- §1 motivating result regenerable in one command (`make eval-blindness`).
- Datasheet committed.
- PRD honest about 8-of-12 corpus gap.
- Recruitment memo drafted (vault).
- 223 tests passing.

## Open

Critical-path (gates ECIR submission):

1. **Recruit 4 independent annotators** for blind second-pass IAA.
2. **Corpus expansion** to 12 docs with genre diversity (FAA AC 43.13-1B, FDA Food Code, NIST SP 800-61, OEM service manual).
3. **D2 baseline sweep** once corpus is finalised.

Lower priority:

4. **D3 constraint-aware retrieval task**.
5. **PAGED metric comparison row**.
6. **JSON-LD export script**.
7. **Spot-check verbatim-wording rule** against source `.txt` for all 8 reviewed files.

## Decision: don't do these for this paper

- Expert human study (Spearman ρ vs Φ). Optional.
- Finnish-language extension. Gated on partner SOPs.
- Multimodal P&IDs. Out of scope.
- Fine-tuning. Out of scope.
- Annotation UI. Not needed.

## Pointers

- Resource PRD: `docs/paper/ipke-bench-resource-prd.md`
- ADRs: `docs/adr/000{1,2,3,4}-*.md`
- Datasheet: `docs/dataset/datasheet.md`
- Constraint taxonomy: `docs/annotation/constraint-types.md`
- Guidelines: `docs/annotation/guidelines.md`
- Validator: `scripts/validate_paper_gold.py`
- D1 reporter: `scripts/constraint_blindness_report.py`
- Recruitment memo (vault): `~/Documents/2ndBrain/Projects/IPKE Paper - Thesis to Congress/07-annotator-recruitment-memo.md`
- PR #85: the squashed delivery of this sprint.
