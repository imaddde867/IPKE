# OLSK Small CNC Manual Review

Document: `olsk_small_cnc_v1_workbook`

Gold: `datasets/paper/gold/olsk_small_cnc_v1_workbook.json`

Source: `datasets/paper/text/olsk_small_cnc_v1_workbook.txt`

Status: manual source-to-gold audit complete; current gold excluded from
confirmatory evaluation; human sign-off pending.

## Review method

The source was read directly against all 24 steps, nine constraints, 23 relations,
metadata fields, and the declared source span. Commands were used only to locate and
display source and annotation text. No script generated, classified, attached, or
transformed annotations.

## Eligibility decision

The workbook is a valid procedural source, but the current annotation is not usable.
The selected span contains five numbered workbook rows and the gold inflates these into
24 steps. Fifteen steps are actions invented from parts-list nouns, four more are only
contextually implied, and only `S16 -> S17` has explicit temporal support. The selected
span is also incomplete: electronic-box assembly continues through later panels,
wiring, testing, and closure.

Exclude the current JSON from confirmatory metrics. The source may re-enter after a
manual, source-faithful rebuild and human verification.

## High-confidence step corrections

| Items | Source lines | Required correction |
|---|---:|---|
| S2-S4 | 17-32 | Replace invented attach/insert/secure actions with one aggregate 01.1 frame-assembly step. Represent parts and counts as resources or parameters. |
| S5 | 35-38 | Keep the explicit protective-foil removal only if imperative remarks are substeps; otherwise make it a precondition of aggregate 01.2. Do not duplicate it as C2. |
| S6-S8 | 33-44 | Delete S7-S8 as parts-list inventions. Retain at most one aggregate bottom-panel installation step. |
| S9 | 45-46 | Keep the explicit red-dot marking action or represent it as an aggregate-step outcome, not both S9 and C4. |
| S10-S14 | 51-69 | Consolidate driver and power-supply placement/mounting. Delete S12-S14; the source does not assign the listed hardware to those invented actions. |
| S16-S17 | 76-77 | Keep both first/then actions. Correct S17 to include both top and bottom. |
| S18-S19 | 83-85 | Delete the nut and screw entries as steps. They are parts, not later operations. |
| S20-S25 | 99-116 | Replace six invented PCB/component actions with at most one aggregate PCB-installation step. |
| S26 | 80-81 | Keep final tightening but move it within 01.4 before 01.5. Its current position reverses source order. |

## Constraint corrections

| ID | Source lines | Required correction |
|---|---:|---|
| C10 | 23-24 | Retype as a qualitative `parameter` with `may`; attach only to aggregate frame alignment/assembly. |
| C2 | 38 | Remove if S5 remains a step; otherwise keep as a `precondition/must` on aggregate 01.2. |
| C3 | 41-42 | Keep as `guard`, change to `must`, and attach only to panel orientation/installation. |
| C5 | 50 | Treat the prepared bottom frame as a resource dependency unless a documented corpus rule classifies such entries as preconditions. |
| C4 | 45-46 | Remove if S9 remains; otherwise keep as `postcondition/must` on aggregate 01.2. |
| C6 | 57-58 | Keep as `guard`, change to `must`, and attach to consolidated driver/power-supply mounting. |
| C7 | 76-77 | Retype as `precondition/must` on S17 if the split first/then actions remain. |
| C9 | 87 | Apply the same resource-versus-precondition rule as C5. |
| C8 | 80-81 | Keep S26 as the action; narrow C8 to `at the end`, typed `precondition/should`. |

No explicit remark or warning is missing from the selected lines. The defects are
action invention, duplication, typing, attachment, scope, and ordering.

## Relation corrections

- Delete every edge derived from parts-table order.
- Delete `S25 -> S26`; it contradicts source order.
- Preserve `S16 -> S17` if the source's explicit first/then split remains.
- Rebuild macro relations only after source-numbered stages replace the invented steps.

## Metadata corrections

- Rename the procedure to the exact retained unit. The current title overstates the
  excerpt as complete electronic-box assembly.
- Replace `full_subprocedure`; the selected 01.1-01.5 range is incomplete.
- Put quantities, parts, and tools into parameters or resources rather than steps.
- Replace generic provenance sections with exact numbered row sections and offsets.
- Supersede review notes that call noun-only parts entries imperative actions.
- Keep pending human sign-off explicit and set the final review date only after repair.

## Human adjudication required

1. Choose one granularity policy: one aggregate step per numbered workbook row, or split
   only genuinely separate imperative remarks.
2. Decide whether prepared components are resources or annotation preconditions.
3. Decide whether macro `NEXT` represents numbered row progression or only explicit
   temporal language. Never mix it with parts-list order.
4. Select a complete scope. The full 64-row assembly sequence at lines 17-944 is the
   most reproducible candidate; a smaller subsystem needs defensible start and terminal
   states.
5. Decide whether the full workbook's safety content is sufficient for the primary
   corpus. Later sections include earth cable, emergency button, testing, and safety
   stickers that the current excerpt omits.

## Release criterion

- Keep the current file out of the confirmatory inclusion manifest.
- Rebuild manually after scope and granularity decisions.
- Require structural, grounding, evidence, and human-verification gates before re-entry.
