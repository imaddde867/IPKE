# Primary annotation passes

Frozen outputs of complete primary-human source passes belong here as `<doc_id>.json`.
They remain distinct from immutable candidates in `datasets/paper/gold/` and final
annotations in `datasets/paper/production/`.

No primary pass exists yet. Do not create a file by changing only `review_status` or an
annotator marker. The matching package under `datasets/paper/evidence/` must record the
source and candidate hashes, active review time, item decisions, exact-anchor check, and
primary output hash.
