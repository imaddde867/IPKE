# EPA MFC Calibration Candidate Audit

Status: agent-audited candidate, pending complete primary-human source pass

Document: `epa_field_operations_manual_filter_sampling_sop`

This audit proposes source-grounded corrections and a compact reviewer decision packet.
It does not modify annotation JSON and does not establish human verification.

## Scope verification

- Authoritative text: `datasets/paper/text/epa_field_operations_manual_filter_sampling_sop.txt`
- Declared Unicode range: `609373:611246`, end-exclusive
- Length: 1,873 characters
- Start: section `6.0 CALIBRATION / POST-CALIBRATION`
- End: after the `ready to ship` shelf instruction and before `7.0 REFERENCES`
- Blind source: `datasets/paper/second_pass/_source/epa_field_operations_manual_filter_sampling_sop.txt`
- Substring and blind-source SHA-256:
  `6c14cfbdf5c380067a83528013f68182204ec0418db8ccb6de13e04204a5bf70`
- Source version: Revision No. 10, February 2025
- Source pages: SOP pages 4 through 6 of 6

The declared continuous boundary is correct. It includes repeated page headers and the
Figure 4 caption between items 6.1.7 and 6.1.8. Treat that material as extraction noise
and provenance context, not additional actions.

## Source action inventory

| Item | Unicode span | Source action |
|---|---:|---|
| 6.1.1 | `609408:609516` | Allow the MFM and MFC to warm up. |
| 6.1.2 | `609516:609584` | Install the Balston filter. |
| 6.1.3 | `609584:609682` | Connect the Definer suction port to the MFC inlet. |
| 6.1.4 | `609682:609775` | Connect the vacuum source to the MFC outlet. |
| 6.1.5 | `609775:609830` | Record the MFC display. |
| 6.1.6 | `609830:609859` | Turn on the flow pump. |
| 6.1.7 | `609859:610348` | Generate the flow range and record coordinated readings at each level. |
| 6.1.8 | `610684:610880` | Let the form compute values, then enter them in the logger. |
| 6.1.9 | `610880:611126` | Adjust the set point and record the result. |
| 6.1.10 | `611126:611246` | Tag, bag, and shelve the MFC. |

## Candidate diagnosis

The current 18-step candidate over-splits coordinated recording objects, invents one
executor action from an automatic form computation, and encodes several action
restatements as constraints.

High-confidence step corrections:

1. Keep S1 through S7.
2. Merge S8 through S11 into one action: record the MFC display, `Flow_Mass`, and MFM
   measurement on the Mass Flow Data Form at each flow level. Item 6.1.7 contains one
   recording action with coordinated objects. Candidate S11 is unsupported as a separate
   data-entry action.
3. Delete S12. The source states that the form computes Full Scale and Zero; it does not
   instruct the executor to calculate them.
4. Keep S13 as the executable 6.1.8 logger-entry action.
5. Keep S14 and S15.
6. Keep S16 through S18 as separate terminal operations unless the human reviewer adopts
   a coarser terminal-action policy.

The proposed source-faithful spine has 14 steps:

```text
S1 -> S2 -> S3 -> S4 -> S5 -> S6 -> S7 -> merged-S8
   -> S13 -> S14 -> S15 -> S16 -> S17 -> S18
```

Do not create a branch for the `1.50 or 3.00` choice. It is a site-dependent parameter.
Keep the `at each level` repetition inside the recording action because the current
relation vocabulary has no loop edge.

## Constraint corrections

| Candidate item | Proposal | Attachment |
|---|---|---|
| C1 | Split the action restatement into parameter `for at least 30 minutes` and guard `without the charger attached to the MFM`. | S1 |
| C10 | Keep as `parameter/must`: use latex or Teflon tubing. | S3 |
| C11 | Keep as `precondition/must`: the vacuum pump is off. | S5 |
| New | Add the missing `Flow_SetPt` control-variable and expected-setting parameter. | S7 |
| C2 | Recommended: drop as a nonbinding example. If retained, use `parameter/must`, not `guard`. | S7 or merged-S8 |
| C3 and C4 | Remove overlap. Keep narrow parameters for each flow level, voltage destination, ten-run average/STP destination, and the Mass Flow Data Form `As Left` destination. | merged-S8 |
| C5 and C6 | Merge into one complete postcondition describing the form's Full Scale and Zero outputs after the six-point audit. Remove the S12 attachment. | merged-S8 |
| C7 | Narrow and retype as `parameter/must`; the 1.50 or 3.00 flow target is quantitative, not a postcondition. | S14 |
| C12 | Keep as `parameter/must`: average ten good runs. | S14 |
| C8 | Retain only the output-location clause as `parameter/must`; do not duplicate the recording action. | S15 |
| C9 | Replace the terminal action restatement with `After completing the calibration` as `precondition/must`. Add the shelf location as a separate parameter if retained. | S16-S18 |

All supported enforcement values in this span are `must`; the text contains no `should`
or `may` signal.

## Mechanical proposals

These changes are strongly supported by the source but remain candidate proposals until
a human completes the source pass:

- preserve the overall Unicode procedure boundary;
- merge S8-S11;
- delete S12 and remove its attachments;
- split C1;
- add the `Flow_SetPt` parameter;
- remove C3/C4 overlap;
- merge C5/C6;
- narrow and retype C7 and C9;
- rebuild affected NEXT edges;
- replace null procedure and step pages with verified source pages;
- record the actual revision and item-level evidence offsets.

## Primary-human decision packet

- [ ] Approve the proposed 14-step spine and short-procedure exception.
- [ ] Drop C2 as an illustrative example, or record why it is binding.
- [ ] Approve the narrow 6.1.7 parameter clauses and merged recording attachment.
- [ ] Approve merged C5/C6 as the six-point-audit postcondition.
- [ ] Keep S16-S18 atomic, or merge the terminal handling sequence.
- [ ] Treat Figure 4 only as provenance, or add a source-supported reference constraint.
- [ ] Decide whether the form-computation postcondition also governs S13.
- [ ] Finalize and renumber IDs only after these semantic decisions.
- [ ] Verify the PDF pages and source version without adding content absent from the
      authoritative committed text.
- [ ] Regenerate the blank blind scaffold if the final step IDs change.

Do not import the out-of-span summary branch about whether post-calibration is possible.
It is not evidence within this procedure window.

## Schema and provenance gaps

- The declared JSON Schema rejects `procedure.source.page: null`; every candidate step
  also has `page: null`.
- Procedure version `CASTNET FOM` omits Revision No. 10 and February 2025.
- Step provenance repeats a broad section but has no exact offsets. Constraint objects
  have no independent evidence anchors.
- URL, checksum, retrieval date, and license live only in
  `datasets/paper/public_sources_manifest.csv`, not the candidate.
- The structured July 5 decision record omits July 6 candidate additions C10-C12 and
  subsequent metadata and relation changes.
- `quality.review_date` remains July 5 while the notes claim July 6 changes.
- Zero-constraint suppression notes for S3 and S5 are stale because C10 and C11 are now
  attached.
- The blank second-pass scaffold incorrectly says `reviewed` despite containing
  placeholder metadata and no completed annotation.

## Validation state

- Exact declared substring and blind source: identical.
- Custom strict structural validator: passes the retained candidate.
- Declared JSON Schema: fails because null page values are not allowed.
- Human-evidence gate: fails because the file remains pending human sign-off and lacks
  the complete primary-pass evidence contract.
- Corpus manifest: provisional development inclusion only.
- Candidate JSON changed by this audit: no.
