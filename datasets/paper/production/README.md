# Production annotations

Final annotations eligible for confirmatory evaluation belong here as
`<doc_id>.json`. The directory is intentionally empty today.

Do not copy legacy candidates from `datasets/paper/gold/`. A file enters this directory
only after a complete primary-human source pass, exact item anchors, a matching frozen
package under `datasets/paper/evidence/`, and any required blind/adjudication workflow.

`make eval-paper-gate` resolves only manifest-included production JSON files. Excluded
candidate artifacts do not need placeholder copies here, and undeclared JSON files fail
closed.
