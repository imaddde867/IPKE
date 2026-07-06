# Independent Annotator Workflow

End-to-end procedure for an independent second-pass annotator producing IAA-eligible annotations for IPKE-Bench. **If you are an annotator who did not author the first-pass gold, this is the document you read.**

## What you are signing up for

- **Per document**: ~3 hours of focused work, blind to the first-pass gold.
- **Total commitment**: the IAA subset is **3 of the 8 documents** (≥30% of the corpus — see `datasets/paper/iaa_subset.json`). Two annotators splitting the subset, or one annotator taking all three, both satisfy the protocol.
- **Output**: one JSON file per assigned doc in `datasets/paper/second_pass/<doc_id>.json`, committed via PR.
- **Authorship**: ≥ 2 documents annotated qualifies for co-authorship on the ECIR 2027 Resource Paper. Single-document contributors are acknowledged.
- **Deadline**: committed annotations by **2026-08-08** (allows time for IAA computation, baseline runs, and the ECIR abstract due 2026-10-12).

## The non-negotiable rule

> You MUST NOT view the first-pass gold file (`datasets/paper/gold/<doc>.json`) or any other annotator's pass on the same document until your annotation is committed.

This rule gives κ statistical meaning. Drift correction toward gold invalidates the inter-annotator agreement claim — see `docs/annotation/guidelines.md` IAA Independence Rule.

If you accidentally see the gold (e.g. typo in a filename, autocomplete in your editor), close the file immediately, note it in your `quality.review_notes`, and continue from your current pass. Do not start over from the gold.

## Setup (one-time)

```bash
git clone https://github.com/imaddde867/IPKE.git
cd IPKE
uv sync --extra extras
uv run python -m spacy download en_core_web_sm
```

Verify the environment:

```bash
uv run python scripts/validate_paper_gold.py --gold-dir datasets/paper/gold
# Should output PASS on all 8 reviewed files.
```

Read in this order:

1. `docs/annotation/constraint-types.md` — locked 6-type × 3-enforcement vocabulary.
2. `docs/annotation/guidelines.md` — annotation decision procedure.
3. `docs/dataset/datasheet.md` — corpus context.

Do NOT yet open any file under `datasets/paper/gold/`.

## Generate your scaffolds (one command)

Instead of hand-creating each JSON, generate the blank scaffolds for the IAA subset:

```bash
uv run python scripts/setup_iaa_subset.py select    # writes datasets/paper/iaa_subset.json (the 3 ⭐ docs)
uv run python scripts/setup_iaa_subset.py scaffold  # writes blank second_pass/<doc>.json + _source/<doc>.txt
```

For each subset document this creates:

- `datasets/paper/second_pass/<doc_id>.json` — a **blank** scaffold carrying only `doc_id`, procedure title, and the exact source `char_start:char_end` span. It contains **no first-pass steps or constraints** — that is the anchoring-bias control. Fill in `steps[]`, `constraints[]`, `relations[]` yourself.
- `datasets/paper/second_pass/_source/<doc_id>.txt` — the exact source span the first pass annotated. **Read this, not the gold.** It is the same text window, so your annotation and the first pass cover identical scope (a prerequisite for a meaningful F1/κ).

Do not open `datasets/paper/gold/` at any point before you commit.

## Per-document workflow

### Step 1 — Pick a document

The IAA subset (the three documents that require a second pass) is written by `scripts/setup_iaa_subset.py select` to `datasets/paper/iaa_subset.json`. **⭐ marks the current subset.** The bounded scope below is the section each first-pass gold actually annotates under the locked `full_subprocedure` rule (`docs/annotation/guidelines.md`). All 8 documents are listed for reference; you only need the ⭐ rows unless coordinating extra coverage with Imad.

| Document | Domain | Bounded scope (locked) | IAA |
|---|---|---|---|
| `epa_field_operations_manual_filter_sampling_sop` | Field operations | MFC SOP §6.0 Calibration / Post-Calibration (6.1.1–6.1.10) | ⭐ |
| `epa_field_sampling_measurement_procedure_validation` | Quality assurance | §3–5 General Information → Procedure Implementation (whole short doc) | ⭐ |
| `nasa_npr_8715_3d_general_safety` | Safety requirements | §2.5.2 System Safety Technical Plan (SSTP) | ⭐ |
| `epa_guidance_preparing_sops_qag6` | Quality assurance | §2.0 SOP Process (2.1–2.6) | |
| `olsk_small_cnc_v1_workbook` | Mechanical assembly | §01.1–01.5 Electronic Box | |
| `niosh_nmam_5th_edition_ebook` | Analytical chemistry | Method 2005 SAMPLING / SAMPLE PREPARATION / MEASUREMENT | |
| `usgs_groundwater_technical_procedures_tm1_a1` | Hydrology | GWPD 1 — Instructions 1–14 + Data Recording | |
| `usgs_nfm_collection_water_samples_a4` | Hydrology | Steps for the EWI sampling method (Steps 1–6, complete) | |

> First-pass golds are model-assisted drafts adjudicated by an independent source-grounded critic pass (annotator field: `model-assisted:<model> + agent-adjudicated`), pending Imad's final sign-off. Your independent human pass is what turns the reported κ into a human–human agreement figure — do not treat the first pass as human-authored when reasoning about independence.

### Step 2 — Read source text

Open `datasets/paper/second_pass/_source/<doc_id>.txt` (created by the `scaffold` command above). This is the exact bounded span the first pass annotated — no need to hunt for section boundaries in the full `datasets/paper/text/<doc_id>.txt`. The bounded section names are in the table above for orientation.

You may also consult the original PDF via the `direct_url` recorded in `datasets/paper/public_sources_manifest.csv` for typographic clarity (table cells, etc.), but the `.txt` is authoritative — that's what the extractor sees.

### Step 3 — Annotate from scratch

Create a new file at `datasets/paper/second_pass/<doc_id>.json` containing your annotation. Do NOT copy from `datasets/paper/gold/<doc_id>.json`.

Schema skeleton:

```json
{
  "procedure": {
    "doc_id": "<doc_id>",
    "title": "<short title>",
    "version": "<source version>",
    "domain": "<short domain tag>",
    "source": {
      "doc_id": "<doc_id>",
      "page": "<page range>",
      "section": "<source section identifier>"
    }
  },
  "steps": [
    {
      "id": "S1",
      "label": "<imperative description of the step>",
      "action_verb": "<verb>",
      "action_object": "<object>",
      "arguments": [],
      "parameters": [],
      "constraints": [
        {
          "id": "C1",
          "type": "<one of: precondition, postcondition, guard, parameter, role_assignment, reference>",
          "enforcement": "<one of: must, should, may>",
          "text": "<verbatim or near-verbatim source text>",
          "attached_to": ["S1"]
        }
      ],
      "flags": {"safety_critical": true},
      "provenance": {"doc_id": "<doc_id>", "page": "...", "section": "..."}
    }
  ],
  "constraints": [],
  "relations": [],
  "quality": {
    "annotation_scope": "full_subprocedure",
    "review_status": "reviewed",
    "annotator": "<your-handle>",
    "review_date": "<YYYY-MM-DD>",
    "review_notes": "<1-3 sentence summary, plus any flags or ambiguities>"
  }
}
```

Follow the decision sequence in `docs/annotation/guidelines.md` for type and enforcement assignment.

### Step 4 — Validate

```bash
uv run python scripts/validate_paper_gold.py \
  --gold-dir datasets/paper/second_pass
```

You should see `PASS <doc_id>.json`. Address any ERROR before continuing. WARN messages about empty steps are acceptable if a step is a genuinely simple single action — document the choice in `review_notes`.

### Step 5 — Commit

Create a branch:

```bash
git checkout -b annotation/second-pass-<doc_id>-<yourhandle>
git add datasets/paper/second_pass/<doc_id>.json
git commit -m "dataset: independent second-pass annotation of <doc_id> (#60)"
git push -u origin annotation/second-pass-<doc_id>-<yourhandle>
gh pr create --title "..." --body "..." --base main
```

Use the PR template (link the recruitment memo / Issue #60 / declare independence from gold).

### Step 6 — IAA reveal

Once the PR is merged, Imad will:

1. Run `python scripts/setup_iaa_subset.py report` (or `make iaa`), which scores only the completed subset docs and computes step F1, constraint F1, relation F1, and token-label Cohen's κ between your pass and the first-pass gold.
2. Post the IAA result in the PR thread.
3. If κ ≥ 0.61 — congratulations, your annotation enters the paper's IAA aggregate.
4. If κ < 0.61 — open a discussion thread on the disagreements. **Do not** unilaterally update your pass to match gold; the protocol is to discuss and (if needed) update *both* the gold and your pass through a documented adjudication round.

## What to do if you get stuck

- **Step boundary unclear**: leave a question in your `review_notes` and pick the boundary closest to a source sentence break. Adjudication happens after the IAA reveal.
- **Constraint type unclear**: use the decision sequence in `docs/annotation/guidelines.md`. When the source modal verb is ambiguous, default to `must` and document the choice.
- **Step has zero constraints**: re-read the source. If the step is genuinely a single action (e.g. "Open the well"), keep it empty and note "deliberate empty: single-action step" in `review_notes`.
- **Source contains a definition or example**: drop it. Definitions and examples are not constraints. See `docs/annotation/guidelines.md` §"What to drop".

## What success looks like

A committed PR adding 1-2 files under `datasets/paper/second_pass/` with:

- `review_status: "reviewed"`
- Your handle in `annotator`
- Constraints typed from the locked 6-type vocabulary, enforced from the source modal verb
- Verbatim source text (no paraphrasing)
- `make eval-iaa` reports κ ≥ 0.61 against the first-pass gold

## Open questions to flag at the IAA reveal

- Disagreements about constraint type (especially precondition vs guard, or parameter vs guard).
- Disagreements about step boundaries.
- Disagreements about which paragraphs count as procedural vs explanatory.

These become §"Annotation Disagreements" content in the paper — even unresolved disagreement is valuable evidence about the difficulty of constraint-attachment annotation.

## Contact

- Imad: `imadeddine200507@gmail.com` (corpus owner, IAA computation)
- Issue tracker: https://github.com/imaddde867/IPKE/issues/60

## See also

- `docs/annotation/constraint-types.md`
- `docs/annotation/guidelines.md`
- `docs/dataset/datasheet.md`
- `docs/paper/ipke-bench-resource-prd.md`
- `~/Documents/2ndBrain/Projects/IPKE Paper - Thesis to Congress/07-annotator-recruitment-memo.md` (Imad's local vault — not public)
