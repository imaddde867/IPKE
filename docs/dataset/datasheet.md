# IPKE-Bench Datasheet

Datasheet for the IPKE-Bench seed corpus and benchmark artifact, following the format of Gebru et al. (2021), *Datasheets for Datasets*. Updated 2026-06-13 with the 8-document seed corpus. Will be regenerated at each corpus expansion.

## 1. Motivation

### 1.1 For what purpose was the dataset created?

IPKE-Bench evaluates procedural knowledge extraction from safety-critical industrial documents, with a focus on **constraint attachment** — the explicit link between safety rules (guards, parameters, preconditions, role assignments) and the procedural steps they govern. Existing benchmarks (PAGED, KEO, CAMB, Carriero & Celino 2024) measure step coverage and graph topology but treat constraint attachment as out of scope. IPKE-Bench fills that gap so that any LLM-based or rule-based extractor for industrial SOPs can be compared on the constraint-attachment axis.

### 1.2 Who created the dataset and on behalf of which entity?

The seed corpus is curated by Imad Eddine Ouadi (CoRe Research Engineering, Turku University of Applied Sciences) as part of the IPKE thesis-to-conference work, targeting the ECIR 2027 Resource Paper Track. CoRe is the funding/affiliation entity. Future independent annotations will list each annotator individually in the per-document `quality.annotator` field.

### 1.3 Who funded the creation?

EAKR/ERDF (regional) and CoRe lab budget. Specific grant lines to be confirmed before camera-ready submission.

## 2. Composition

### 2.1 What do the instances represent?

Each instance is one industrial / regulatory / SOP document annotated with:

- **Procedural steps** (`steps[]`) — ordered, actionable operations.
- **Constraints** (step-embedded `step.constraints[]` or procedure-level `constraints[]`) — guards, parameters, preconditions, postconditions, role assignments, or external references binding one or more steps.
- **Attachments** — explicit `attached_to` (step-embedded) or `applies_to` (procedure-level) edges between constraints and steps.
- **Quality metadata** — `review_status`, `annotator`, `review_date`, `review_notes`.

### 2.2 How many instances?

| Released | Target |
|---|---|
| 8 (seed corpus, 2026-06-13) | 12-15 (ECIR submission) |

The 8 seed documents are the original IPKE thesis selection plus public-source documents added under the Issue #53 expansion. The 4 additional documents to reach the 12-target are tracked in the resource PRD as `priority candidates`: FAA AC 43.13-1B (aviation maintenance), FDA Food Code (food safety), NIST SP 800-61 Rev. 2 (computer security incident handling), and one open-license OEM service manual.

### 2.3 What data does each instance contain?

JSON conforming to `schemas/ipke_annotation.schema.json`. See `docs/annotation/guidelines.md` for the annotation schema in human-readable form and `docs/annotation/constraint-types.md` for the locked 6-type × 3-enforcement vocabulary.

Per-instance statistics for the seed corpus (after the 2026-06-13 standards-review fixes):

| Document | Domain | Steps | Constraints |
|---|---|---:|---:|
| `nasa_npr_8715_3d_general_safety` | Aerospace safety regulation | 4 | 9 |
| `epa_guidance_preparing_sops_qag6` | SOP governance | 4 | 17 |
| `olsk_small_cnc_v1_workbook` | Mechanical assembly | 4 | 7 |
| `epa_field_operations_manual_filter_sampling_sop` | Field sampling SOP | 6 | 19 |
| `epa_field_sampling_measurement_procedure_validation` | Procedure validation | 5 | 15 |
| `niosh_nmam_5th_edition_ebook` | Industrial hygiene chemistry | 6 | 12 |
| `usgs_groundwater_technical_procedures_tm1_a1` | Field measurement | 9 | 14 |
| `usgs_nfm_collection_water_samples_a4` | Field sampling SOP | 5 | 24 |
| **TOTAL** | | **43** | **117** |

### 2.4 Is there a label or target associated with each instance?

The annotation IS the label. Tier-A evaluation metrics (`StepF1`, `ConstraintCoverage`, `ConstraintAttachmentF1`, Φ) compare extractor output to these annotations; Tier-B compares full PKG topology via SMatch.

### 2.5 Is any information missing?

- **Independent second-pass annotations**: only `epa_field_sampling_measurement_procedure_validation.json` and `olsk_small_cnc_v1_workbook.json` have `second_pass/` files, and both are marked `review_status="llm_draft"` (not independent). The paper's κ ≥ 0.61 claim is open pending recruitment.
- **Per-document license attribution inside the JSON**: currently consolidated in `datasets/paper/public_sources_manifest.csv`; will be denormalised into each gold's `procedure.source.license` field in the next manifest update.

### 2.6 Are there explicit relationships between individual instances?

Each gold file is independent at the procedural level. Some documents share `source_family` (3 EPA, 2 USGS), which the paper analyses for genre concentration (see §6).

### 2.7 Are there recommended data splits?

No standard train/dev/test split. IPKE-Bench is an evaluation-only benchmark; researchers tune on external corpora (PAGED, ProPara, OpenPI) and report on the full IPKE-Bench corpus. The seed-corpus 8 documents may all be used as evaluation; future expansion will not introduce a train split.

### 2.8 Are there errors, sources of noise, or redundancies?

- **LLM-drafted scaffolding**: every reviewed file started as an LLM-drafted JSON skeleton. The constraint-blindness baseline (`datasets/paper/reports/constraint_blindness_v2_sbert075.json`) quantifies the under-recall of the original draft: 3.66× under-production of constraints, 20.5% macro recall at the Tier-A protocol matcher (SBERT cos ≥ 0.75). This is a known property of the draft pipeline; the reviewed gold corrects it via the workflow in `docs/annotation/guidelines.md`.
- **Scope**: every gold currently uses `quality.annotation_scope = "bounded_excerpt"` (1-3 pages of a multi-step + multi-constraint section). Reviewers consuming the full source document outside the bounded section will see additional procedural content that is not annotated.

### 2.9 Is the dataset self-contained?

Yes. Each instance bundles the source text (`datasets/paper/text/<doc>.txt`), the gold JSON, the source-document checksum (in `public_sources_manifest.csv`), and (where applicable) the second_pass file. The original source PDF is not redistributed; the manifest carries a stable URL.

### 2.10 Does the dataset contain confidential data, PII, or content that might be offensive?

No. All documents are public-domain or open-licensed government/regulatory/educational SOPs. No partner-private SOPs are included.

## 3. Collection process

### 3.1 How was the data acquired?

Document selection criteria:

1. Public-domain or open-licensed (CC-BY, US federal works, open-source workbooks).
2. Stable URL on the issuing organisation's site.
3. Procedural content with multi-step + multi-constraint sections suitable for bounded-excerpt annotation.

Documents were downloaded with `scripts/download_public_sources.py` and text-extracted with `scripts/extract_public_documents.py`. SHA-256 checksums are recorded in `datasets/paper/public_sources_manifest.csv`.

### 3.2 What mechanisms were used?

- PDF → text: `pdfminer.six` via the extraction script.
- LLM-drafted gold skeleton: original IPKE draft pipeline (`tools/annotate_gold.py`).
- Human review: Imad Eddine Ouadi, 2026-06-13 sprint.
- Standards review and fixes: see `docs/plans/2026-06-13-ipke-bench-taxonomy-and-review.md` (this sprint's plan doc).

### 3.3 Who was involved and how were they compensated?

- Corpus selection + draft pipeline + human review: Imad (thesis work; not separately compensated).
- Future independent annotators: David, Mikko, and research assistants recruited from CoRe. Compensation arrangement under discussion; co-authorship offered for ≥ 2 documents annotated. See `~/Documents/2ndBrain/Projects/IPKE Paper - Thesis to Congress/07-annotator-recruitment-memo.md`.

### 3.4 Over what timeframe?

- Seed-corpus draft pipeline: April-June 2026.
- Human review of all 8 documents: 13 June 2026.
- Locked taxonomy applied: 13 June 2026.
- Planned independent IAA pass: July-August 2026.
- Corpus expansion to 12 documents: July-September 2026.

### 3.5 Were any ethical review processes conducted?

Not required for this corpus — all documents are publicly available and contain no human-subjects data. The future expert human study (P2 in PRD, optional) will need light IRB review per Turku UAS policy.

## 4. Preprocessing / cleaning / labelling

### 4.1 Was preprocessing done?

- PDF → text extraction (lossy for tables and figures).
- Normalisation of whitespace, hyphenation, and OCR artifacts.
- The LLM-drafted gold pipeline produced an initial JSON skeleton that was discarded after the human review pass — every constraint in the final gold is human-reviewed against source text.

### 4.2 Was the raw data saved?

Yes. The extracted `datasets/paper/text/*.txt` files are committed; the source PDFs are not redistributed but URLs and checksums are recorded.

### 4.3 Is the preprocessing software available?

Yes. `scripts/download_public_sources.py`, `scripts/extract_public_documents.py`, `scripts/normalize_gold_annotations.py`, and (added in the 2026-06-13 sprint) `scripts/migrate_constraint_types.py` and `scripts/validate_paper_gold.py`.

## 5. Uses

### 5.1 Has the dataset been used yet?

Yes:

- The IPKE local pipeline (Mistral-7B Q4_K_M, Llama-3.1 family) for baseline evaluation (D2 in PRD; experiments pending reviewed gold completion).
- The constraint-blindness D1 baseline (free §1 result, completed 2026-06-13).

### 5.2 What tasks could it be used for?

- LLM-based procedural extraction benchmarking.
- Constraint-attachment evaluation (the central novelty).
- Constraint-aware retrieval task (D3 in PRD, optional).
- Annotation methodology research (the locked taxonomy and verbatim rule are reusable).

### 5.3 What tasks should it NOT be used for?

- Training extractors via direct fine-tuning on the gold (the corpus is too small).
- Generalisation claims about industrial procedure extraction at scale (8 docs, US-gov-environmental/safety concentrated).
- Anything involving partner-private SOPs (the corpus does not include any).

## 6. Distribution and licensing

### 6.1 Will the dataset be distributed?

Yes. The IPKE-Bench seed corpus is part of the `personal/IPKE` repository under the repository's research-distribution license. Per-document source content remains under the original issuing organisation's license — all 8 seed documents are either US federal works (public domain) or open-licensed (CC-BY-SA for OLSK).

### 6.2 How?

GitHub repository: `https://github.com/imaddde867/IPKE`. Released under the repo's `LICENSE` for the annotation layer; per-document source content carries its own license tracked in `public_sources_manifest.csv`.

### 6.3 When?

Public during the paper's review cycle (single-blind ECIR Resource Track). Stable archived release on Zenodo (or equivalent) at camera-ready time.

### 6.4 Will there be terms of use or license?

The IPKE-Bench annotation layer (JSON files, taxonomy, guidelines, evaluation harness) is released under CC-BY 4.0. Per-document source licensing is recorded in the manifest. Users must respect the source license when redistributing source `datasets/paper/text/*.txt` files.

### 6.5 Have any third parties imposed restrictions?

No — all sources are public-licensed.

### 6.6 Are there export controls?

No.

## 7. Maintenance

### 7.1 Who is supporting / hosting the dataset?

Imad Eddine Ouadi (`imadeddine200507@gmail.com`) is the maintainer through the paper's review cycle. Long-term maintenance plan: CoRe lab GitHub organisation after ECIR submission.

### 7.2 How can contributors / users reach the maintainer?

GitHub issues on `https://github.com/imaddde867/IPKE` and email above.

### 7.3 Is there an erratum?

Future errata will be recorded as ADRs (`docs/adr/`) and per-document `quality.review_notes` entries. The 2026-06-13 standards review found and corrected 4 hard violations in the initial reviewed gold; see the sprint plan doc.

### 7.4 Will the dataset be updated?

Yes. Planned updates:

- Corpus expansion to 12-15 documents (target: pre-submission).
- Independent second-pass annotations as recruited annotators submit.
- Datasheet regeneration at every corpus change.

### 7.5 If the dataset relates to people, are there applicable retention limits?

N/A — documents do not contain personal data.

### 7.6 Will older versions remain supported?

Yes. Tagged releases (planned: `v0.1-seed-corpus` for the 8-doc seed, `v1.0-ecir-submission` for the 12-doc paper corpus) will remain reproducible.

### 7.7 Mechanism for contributing?

Pull requests against `https://github.com/imaddde867/IPKE`. New annotations must follow `docs/annotation/guidelines.md` and pass `scripts/validate_paper_gold.py`. The constraint-blindness reporter (`scripts/constraint_blindness_report.py`) will be re-run at each corpus change to keep the §1 motivating result in sync.

---

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86-92.
- IPKE-Bench resource PRD: `docs/paper/ipke-bench-resource-prd.md`
- Annotation guidelines: `docs/annotation/guidelines.md`
- Constraint taxonomy: `docs/annotation/constraint-types.md`
- Constraint-blindness baseline: `datasets/paper/reports/constraint_blindness_v2_sbert075.json`
