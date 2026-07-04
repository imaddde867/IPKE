# Human sign-off: verify and stamp the 8 IPKE-Bench golds

**Type:** gold-quality / release-gate
**Blocks:** ECIR 2027 Resource-track submission (dataset must be human-verified, not just model-assisted)
**Assignee:** @imaddde867
**Estimate:** 3–5 h focused review (≈25–40 min/doc)

---

## Why this exists

All 8 golds currently carry the annotator string
`model-assisted:claude-opus-4-8 + agent-adjudicated (pending human sign-off)`.
They pass the strict validator, but **no human has verified the labels yet.**
For a resource paper, the annotator-of-record must be a person. This issue is
the single gate that converts the corpus from *model-assisted draft* to
*human-verified gold*. Do not submit, and do not cite IAA numbers, until this
is closed.

The provenance is deliberately additive: sign-off **appends**
`+ human-verified:imad` rather than erasing the model-assisted lineage, so the
datasheet still tells the truth about how each label was produced.

---

## What you're signing off on

251 steps / 199 constraints across 8 documents (locked `full_subprocedure`
scope, 6-type taxonomy: precondition, postcondition, guard, parameter,
role_assignment, reference; enforcement ∈ must/should/may).

| Document | Section signed off | Steps | Constraints |
|---|---|--:|--:|
| epa_field_operations_manual_filter_sampling_sop | 5.14 Repair | 18 | 9 |
| epa_field_sampling_measurement_procedure_validation | Procedure Development, Validation & Approval | 35 | 41 |
| epa_guidance_preparing_sops_qag6 | 2.0 SOP Process | 36 | 34 |
| nasa_npr_8715_3d_general_safety | 2.5.2 System Safety Technical Plan | 39 | 29 |
| niosh_nmam_5th_edition_ebook | Method 2005 (Nitroaromatic compounds) | 34 | 21 |
| olsk_small_cnc_v1_workbook | 01 Electronic Box assembly | 24 | 8 |
| usgs_groundwater_technical_procedures_tm1_a1 | GWPD 1 (water-level measurement) | 29 | 12 |
| usgs_nfm_collection_water_samples_a4 | EWI sampling steps | 36 | 45 |
| **TOTAL** | | **251** | **199** |

---

## Do this, in order

### 1. Review each gold against its source text (the actual work)

For each doc, open the gold beside its source section and read for **label
correctness**, not existence — the validator already guarantees structure.
Ask, per document:

- **Step boundaries** — is each step one atomic action? No merged/split steps?
- **Constraint TYPE** — is each constraint the right one of the 6? The common
  confusions to catch: precondition (before) vs guard (conditional/if-then) vs
  postcondition (verify-after); parameter (a value/setting) vs reference
  (points to another doc/section).
- **Enforcement** — `must` (mandatory), `should` (recommended), `may`
  (optional). Skim for MUST/SHALL vs SHOULD vs MAY in the source.
- **Attachment** — does each constraint's `attached_to`/`applies_to` point at
  the step it actually governs? Mis-attachment is the headline metric of the
  paper; this is the highest-value thing to check.
- **Coverage** — did we miss an obvious constraint, or invent one not in the text?

Open a gold + source side by side:
```bash
# gold
$EDITOR datasets/paper/gold/olsk_small_cnc_v1_workbook.json
# source span that was annotated
$EDITOR datasets/paper/second_pass/_source/<doc>.txt   # for the 3 IAA docs
# or the full source
$EDITOR datasets/paper/text/<doc>.txt
```

Fix anything wrong **directly in the gold JSON**. If you change type/enforcement/
attachment, keep the edit minimal and re-run the validator (step 3) after.

> Watch-items flagged during adjudication (spend extra time here):
> - **nasa** is a requirements document, not a linear procedure — steps are
>   requirement clauses; confirm that reads correctly as "procedure."
> - **olsk** is constraint-light (8 across 24 steps) by nature — confirm we
>   didn't under-annotate, but don't manufacture constraints.
> - **niosh** carries numeric flow-rate/volume parameters — check the numbers.

### 2. Stamp your sign-off (mechanical — one command)

Dry-run first (writes nothing, shows before/after):
```bash
python3 scripts/sign_off_gold.py --annotator imad
```
Then apply to all 8 and auto-validate:
```bash
python3 scripts/sign_off_gold.py --annotator imad --apply
```
This appends `+ human-verified:imad`, sets `review_status=reviewed`,
`review_date=<today>`, and runs the strict validator. It is idempotent and
refuses to leave you in a failing state (non-zero exit if any gold breaks).

_Sign one doc at a time if you review over multiple sessions:_
```bash
python3 scripts/sign_off_gold.py --annotator imad --doc niosh_nmam_5th_edition_ebook --apply
```

### 3. Confirm the release gate is green
```bash
make gold-pipeline      # = validate --strict + label-blindness check
# or directly:
PYTHONPATH=. python3 scripts/validate_paper_gold.py --gold-dir datasets/paper/gold --strict
```
Expect: `PASS` for all 8, exit 0. The annotator string on every gold should now
read `... + human-verified:imad`.

### 4. Commit + push
```bash
git add datasets/paper/gold/ scripts/sign_off_gold.py docs/annotation/SIGN_OFF_ISSUE.md
git commit -m "gold: human sign-off (imad) on all 8 golds — corpus is now human-verified"
git push origin main
```

---

## Definition of done

- [ ] All 8 golds read against source; label errors fixed
- [ ] `scripts/sign_off_gold.py --annotator imad --apply` run; annotator string on all 8 ends with `+ human-verified:imad`
- [ ] `make gold-pipeline` exits 0 (8× PASS)
- [ ] Committed and pushed
- [ ] (separate, after this) IAA second pass on the 3-doc subset — that closes the *agreement* claim; this issue closes the *human-verified* claim

## Explicitly NOT in scope here
- The IAA double-annotation (that's the `iaa-setup` / `iaa` workflow and its own issue)
- The 9th document (niosh_nmam_surface_sampling_guidance) — out of corpus scope
- Any change to taxonomy or validator contract
