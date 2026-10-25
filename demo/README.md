# IPKE static research console

First vertical slice of [issue #115](https://github.com/imaddde867/IPKE/issues/115).
Native HTML/CSS/JavaScript; no build, model, database, login, or external assets.
The console makes only a same-origin request for its committed JSON snapshot.

```sh
python scripts/export_demo_data.py
python -m http.server 8765 --directory demo
# Open http://localhost:8765
```

The graph contains 14 proposed steps, 14 embedded constraints and one
procedure-level constraint applying to three terminal steps. Select a node to
inspect type, enforcement, attachment and exact source text; step rows distinguish
step-local constraints from applicable procedure-level constraints. The summary
toggle uses an explicitly authored illustration, not a stored model output. The 15
omitted clauses sit in a collapsed disclosure; this is not a summary-quality
measurement. The audit preserves the full review packet, all 13 transformation
records, their rationales, the unverified legacy candidate and eight open human
decisions. Rejected legacy items remain inspectable with their section-level
provenance and no invented exact span. Corpus cards preserve all eight manifest
statuses, exclusions and source links; the verification count is derived from the
manifest.

## Data contract

`scripts/export_demo_data.py` uses only the Python standard library. It binds the
source, candidate, legacy candidate and audit document to the review packet's
SHA-256 hashes, checks source spans and decision-reference namespaces, derives the
corpus verification count, and writes deterministic `demo/data/epa.json`. The
manifest is authoritative for eligibility, rather than `quality.review_status`. The
export fails closed if this slice's candidate status changes. Human verification
requires the repository's independent evidence workflow; UI interactions cannot
confer it. No review edits or local acceptance are implemented in this slice.

Offsets index Unicode code points in the frozen text, end exclusive. Only the
1,873-character procedure excerpt is bundled; original absolute offsets and its
base offset are preserved. JavaScript uses `Array.from()` before slicing, so
astral characters cannot silently shift Python offsets. Counts refer to proposed
annotation structure, not extractor accuracy or human annotation effort.

To refresh the hero, update the upstream annotation/review packet through the
normal audit process, regenerate the export and run the checks below. Adding a
new graph requires item-level source anchors and an explicit evidence-status
contract in the exporter; other corpus entries currently expose metadata only.
Never silently promote legacy section anchors to exact item provenance.

## Recorded-inference investigation

Inspected at base commit `205fb904882345a1b75415068f4219e22c6d5e99`:

- No tracked per-stage inference trace or EPA MFC recorded run is present.
- `.gitignore` excludes `results/` and `tests/outputs/`.
- `scripts/experiments/README.md` describes generated predictions and logs under
  `results/experiments/`; these are instructions, not bundled measurements.
- `scripts/experiments/experiment_utils.py` records request-level `wall_time` and
  `processing_time_api`, which do not establish independently verified stage
  timings for this annotation.
- The only GitHub release (`v0.1-seed-corpus`) has no attached assets; its
  historical claims do not override the current datasheet. The connector did
  not support listing repository-wide Actions artifacts, so those were not
  established as an additional source.
- The review packet records artifact transformations, not inference events.

Accordingly the export has `replay.available: false`, with no numeric timings.
A future timed replay needs a source/model/config-bound run with stage outputs,
independently supported elapsed times and a documented relationship to the
candidate. Live inference and thesis-performance metrics remain out of scope.

## Validation

```sh
python scripts/export_demo_data.py --check
python -m unittest discover -s tests -p test_demo_export.py -v
# With the repository development environment:
uv run pytest tests/test_demo_export.py tests/test_epa_review_candidate.py
uvx ruff@0.15.10 check scripts/export_demo_data.py tests/test_demo_export.py
node --check demo/app.js
```

Optional browser regression check (Playwright must be installed/resolvable in
Node, with `npx playwright install chromium` completed): start the HTTP server
above, then `node demo/tests/browser.cjs`. `DEMO_URL` can override the default
localhost URL; `CHROMIUM_PATH` optionally selects an existing Chromium binary.
Checks all 15 span selections, the collapsed summary disclosure, procedure-level
constraint display, legacy audit inspection, corpus status, download, mobile
overflow and runtime errors. It saves screenshots under `/tmp/` for visual
inspection.

## Deployment boundary

Serve `demo/` as a static directory, with no build command. This change contains
no deployment workflow or domain configuration and does not publish production.
Before deploying to `ipke.imadlab.com`, establish the hosting project, domain/DNS
ownership, authorization and a preview review. Review source redistribution
notes in the corpus cards and dataset datasheet. Keep the unverified-evidence
banner and limitations visible in any public deployment.
