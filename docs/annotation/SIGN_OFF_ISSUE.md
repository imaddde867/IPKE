# Human sign-off: verify and stamp included procedure golds

**Type:** gold-quality / release-gate
**Blocks:** Method-paper confirmatory evaluation (included gold must be human-verified, not just model-assisted)
**Assignee:** @imaddde867
**Estimate:** 25–40 min per included procedure after the manifest is fixed

---

## Why this exists

The current gold directory contains eight files carrying the annotator string
`model-assisted:claude-opus-4-8 + agent-adjudicated (pending human sign-off)`.
They pass the strict validator, but **no human has verified the labels yet.**
For confirmatory method-paper evidence, the annotator-of-record for every
included procedure gold must be a person. Strict validation is not sufficient.
Do not report confirmatory results until the inclusion manifest is explicit,
every included gold is signed, and the release gate passes against that same
active set.

The provenance is deliberately additive: sign-off **appends**
`+ human-verified:imad` rather than erasing the model-assisted lineage, so the
datasheet still tells the truth about how each label was produced.

---

## Current candidate inventory

The directory currently contains 256 steps / 231 constraints across eight
documents (locked `full_subprocedure` scope, 6-type taxonomy: precondition,
postcondition, guard, parameter, role_assignment, reference; enforcement ∈
must/should/may). This inventory is not the confirmatory inclusion manifest.

> **Confirmatory inclusion decision (#112, 2026-07-11).** NASA NPR 8715.3D
> section 2.5.2 is a requirements stress test, not an eligible confirmatory
> procedure. Do not append a `human-verified` procedure-gold marker to
> `nasa_npr_8715_3d_general_safety.json` until #112 resolves its exclusion or
> secondary representation. Never sign it merely to make a gate pass.

> **2026-07-06 agent verbatim-grounding + completion pass.** Every gold was
> re-read against its source `.txt` span before this sign-off: all constraint
> texts re-grounded to contiguous verbatim source wording (guidelines Verbatim
> wording rule), definitional/restatement constraints dropped, missed
> constraints added, enforcement realigned to the modal-verb mapping, and two
> mis-located spans corrected (epa_filter was annotating the MFC SOP §6.0, not
> "5.14 Repair"; usgs_groundwater was annotating the Figure-2 field form, its
> Instructions were re-authored; usgs_nfm's span cut off Steps 4d–6, now
> annotated). Details in each gold's `review_notes`. Counts below are current.

| Document | Section signed off | Steps | Constraints |
|---|---|--:|--:|
| epa_field_operations_manual_filter_sampling_sop | MFC SOP — 6.0 Calibration / Post-Calibration (6.1.1–6.1.10) | 18 | 12 |
| epa_field_sampling_measurement_procedure_validation | Procedure Development, Validation & Approval | 35 | 44 |
| epa_guidance_preparing_sops_qag6 | 2.0 SOP Process (2.1–2.6) | 36 | 33 |
| nasa_npr_8715_3d_general_safety | 2.5.2 System Safety Technical Plan | 39 | 26 |
| niosh_nmam_5th_edition_ebook | Method 2005 (Nitroaromatic compounds) | 34 | 24 |
| olsk_small_cnc_v1_workbook | 01 Electronic Box assembly | 24 | 9 |
| usgs_groundwater_technical_procedures_tm1_a1 | GWPD 1 — Instructions 1–14 + Data Recording | 29 | 20 |
| usgs_nfm_collection_water_samples_a4 | EWI sampling steps (1–6, complete) | 41 | 63 |
| **TOTAL** | | **256** | **231** |

---

## Do this, in order

### 1. Review each included gold against its source text (the actual work)

For each procedure gold named in the confirmatory inclusion manifest, open the
gold beside its source section and read for **label correctness**, not
existence. The validator guarantees structure only. Ask, per document:

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
> - **nasa** is excluded from confirmatory procedure evaluation under #112.
>   Do not stamp it as human-verified procedure gold.
> - **olsk** is constraint-light (8 across 24 steps) by nature — confirm we
>   didn't under-annotate, but don't manufacture constraints.
> - **niosh** carries numeric flow-rate/volume parameters — check the numbers.

### 2. Stamp each included gold (human reviewer only)

After personally reviewing an included document, dry-run that document first
(writes nothing, shows before/after):
```bash
python3 scripts/sign_off_gold.py --annotator imad --doc <included_doc_id>
```
Then apply only to that included document and auto-validate:
```bash
python3 scripts/sign_off_gold.py --annotator imad --doc <included_doc_id> --apply
```
This appends `+ human-verified:imad`, sets `review_status=reviewed`,
`review_date=<today>`, and runs the strict validator. It is idempotent and
refuses to leave you in a failing state (non-zero exit if any gold breaks).

Do not use the unscoped apply-all form while the excluded NASA artifact remains
in the directory. Only a human reviewer may supply this sign-off marker.

### 3. Confirm the release gate is green
```bash
make eval-paper-gate
```
Expect: `PASS` for every included gold, exit 0. This is the paper-evidence
release criterion once `PAPER_GOLD` matches the confirmatory inclusion
manifest. Every included gold's annotator string should then read
`... + human-verified:imad`.
`make eval-validate` and `make gold-pipeline` are not substitutes: strict
structural and annotation-contract validation can pass before a human signs off.
Because `eval-paper-gate` currently checks every JSON file in `PAPER_GOLD`, an
excluded NASA failure must be resolved through #112's corpus representation,
not by adding a false procedure-gold sign-off.

### 4. Commit, push the branch, and open a draft PR
```bash
git switch -c gold/human-sign-off
git add datasets/paper/gold/ docs/annotation/SIGN_OFF_ISSUE.md
git commit -m "Record human sign-off for included golds"
git push -u origin gold/human-sign-off
gh pr create --draft --base main --head gold/human-sign-off \
  --title "Record human sign-off for included golds" \
  --body "Human sign-off for the manifest-included procedure golds."
```

---

## Definition of done

- [ ] Confirmatory inclusion manifest is explicit
- [ ] Every included procedure gold read against source; label errors fixed
- [ ] Every included gold's annotator string ends with `+ human-verified:imad`
- [ ] NASA NPR 8715.3D has no procedure-gold human marker while #112 remains unresolved
- [ ] `PAPER_GOLD` matches the inclusion manifest and `make eval-paper-gate` exits 0
- [ ] Committed and pushed
- [ ] (separate, after this) IAA second pass on the 3-doc subset — that closes the *agreement* claim; this issue closes the *human-verified* claim

## Explicitly NOT in scope here
- The IAA double-annotation (that's the `iaa-setup` / `iaa` workflow and its own issue)
- The 9th document (niosh_nmam_surface_sampling_guidance) — out of corpus scope
- Any change to taxonomy or validator contract
