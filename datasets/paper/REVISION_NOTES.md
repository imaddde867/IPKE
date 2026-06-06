# PR #57 Revision Notes

This branch revises PR #57 to address the review of issue #53. Scope:

- Reclassify committed annotations as `pilot_gold`, not `gold`.
- Add per-batch provenance metadata under `datasets/paper/annotation_batches/`.
- Add `word_count`, `token_count`, `gold_status`, `annotation_status`,
  `annotation_scope`, and `annotator_count` columns to the manifest, and
  generate them programmatically.
- Add a stricter `ipke_paper_tiera.schema.json` and require it for new paper gold.
- Add CI tests that validate every selected paper gold file.
- Add `LICENSES.md` and `ATTRIBUTION.md`.
- Expand the IAA double-annotation subset to three documents across three
  source families, and reclassify the current IAA as a smoke check.
- Publish a combined `datasets/paper/INVENTORY.md` for archive anchor +
  paper public documents.
