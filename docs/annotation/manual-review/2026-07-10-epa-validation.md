# EPA Procedure Validation Manual Review

Document: `epa_field_sampling_measurement_procedure_validation`

Gold: `datasets/paper/gold/epa_field_sampling_measurement_procedure_validation.json`

Source: `datasets/paper/text/epa_field_sampling_measurement_procedure_validation.txt`

Status: agent source-to-gold correction complete for the high-confidence items below;
human sign-off pending.

## Review method

The source was read directly from lines 70-238 alongside every active step, nested
constraint, attachment, and relation. Annotation decisions were applied manually. No
script or model generated, classified, or attached labels in this pass. Commands were
used only to display JSON and verify syntax or source-string grounding.

## Applied corrections

| Item | Source | Correction |
|---|---|---|
| C9 | lines 108-111 | Narrowed to the evaluation-extent parameter so it no longer duplicates S7. |
| C11 | lines 123-127 | Retyped as a parameter and narrowed to `in writing (email or memo)`. |
| C13 | lines 128-131 | Retyped as a parameter describing the required project-file location and content. |
| C14 | line 133 | Retyped as a reference and narrowed to `according to Section 4 of this procedure`. |
| C27 | lines 183-186 | Retyped as a guard and narrowed to the `as necessary` condition during validation. |
| C28 | lines 190-193 | Retyped as role assignment because the constraint identifies the validation team actor. |
| C29 | lines 190-193 | Attached to both field testing S26 and information recording S27. |
| C31 | lines 197-198 | Narrowed to the acceptance-criteria parameter used for assessment. |
| C32 | lines 202-203 | Retyped as a guard because acceptance is the condition for finalization. |
| C34 | lines 209-210 | Retyped as role assignment because it identifies the team responsible for documentation. |
| S33/C37 | lines 218-220 | Replaced the unsupported transfer action with post-validation record maintenance; C37 now captures the Quality Assurance Manager role. |
| S35/C40/C41 | lines 232-236 | Corrected the step wording, staff-scope parameter, and external training-procedure reference. |

No human-verification marker was added. Step and constraint counts are unchanged by this
pass.

## Human adjudication required

1. The declared character span contains sections 1-5 while the metadata names sections
   3-5. Decide whether the documentation-control actions S1-S3 belong in this procedure.
2. S1/C1 describe how the document was prepared in the past, not an executable action.
   Remove them if sections 1-2 remain out of the method-evaluation scope.
3. The current relation list connects standard, non-standard, and internally developed
   procedure categories as a single NEXT sequence. Rebuild the graph before using
   relation metrics.
4. The validation-retry branch and later record/training path require explicit semantics:
   failed validation may produce records but cannot authorize implementation training.
5. C46 describes validation extent generally. Decide its complete attachment set rather
   than leaving it attached only to S11.
6. Decide whether the optional new-procedure development statement at lines 103-106 is a
   distinct step or is already covered by the internally developed procedure route.

## Validation expectations

- JSON syntax must pass.
- The custom strict validator must pass after the evidence-gate implementation settles.
- The declared JSON Schema is currently expected to fail on `page: null`; fix the schema
  contract separately rather than rewriting source provenance to a false page value.
- `make eval-paper-gate` must continue to fail until a human performs sign-off.
