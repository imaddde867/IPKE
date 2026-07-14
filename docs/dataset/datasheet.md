# IPKE Evaluation Corpus Datasheet

Datasheet for the supporting IPKE evaluation corpus, following the format of Gebru et
al. (2021), *Datasheets for Datasets*. The eight-file directory is a legacy candidate
inventory, not a released benchmark or confirmatory split. Updated 2026-07-11 after
manual source audits invalidated three current golds.

## 1. Motivation

### 1.1 For what purpose was the dataset created?

The corpus supports evaluation of IPKE, a method for skeleton-conditioned,
source-grounded constraint attachment in procedural extraction. It records steps,
typed constraints, enforcement, and explicit attachment. The method, not the corpus, is
the primary paper contribution. No current file is human verified, so this datasheet
does not claim a released human-gold benchmark.

### 1.2 Who created the dataset and on behalf of which entity?

The candidate corpus is curated by Imad Eddine Elmouss at CoRe, Turku University of
Applied Sciences, as supporting infrastructure for the IPKE method paper. Future human
and independent annotations will identify each contributor in per-document provenance.

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

| Current state | Target |
|---|---|
| 8 legacy candidate files; 0 human verified; 3 currently excluded | 12 eligible procedures in a frozen confirmatory split |

Issue #87 tracks expansion and replacement. NASA NPR 8715.3D section 2.5.2 does not
count because it is a requirements block. Current OLSK and NIOSH annotations also do
not count until rebuilt manually. Final size is counted from eligible procedures, not
files present in `gold/`.

### 2.3 What data does each instance contain?

JSON conforming to `schemas/ipke_annotation.schema.json`. See `docs/annotation/guidelines.md` for the annotation schema in human-readable form and `docs/annotation/constraint-types.md` for the locked 6-type × 3-enforcement vocabulary.

Per-instance statistics for the seed corpus (after the 2026-07 full-subprocedure re-annotation + 2026-07-06 verbatim-grounding pass; annotated unit named in each gold's `procedure.source.section`):

| Document | Domain | Steps | Constraints | Current status |
|---|---|---:|---:|---|
| `nasa_npr_8715_3d_general_safety` | Aerospace requirements | 39 | 26 | Excluded: requirements stress test (#112) |
| `epa_guidance_preparing_sops_qag6` | SOP governance | 36 | 33 | Candidate; manual audit pending |
| `olsk_small_cnc_v1_workbook` | Mechanical assembly | 24 | 9 | Excluded pending manual rebuild |
| `epa_field_operations_manual_filter_sampling_sop` | Field calibration SOP (MFC) | 18 | 12 | Legacy candidate; agent packet prepared (14/15), primary-human pass pending |
| `epa_field_sampling_measurement_procedure_validation` | Procedure validation | 35 | 44 | Agent correction complete; human adjudication pending |
| `niosh_nmam_5th_edition_ebook` | Industrial hygiene chemistry | 34 | 24 | Excluded pending manual rebuild |
| `usgs_groundwater_technical_procedures_tm1_a1` | Field measurement | 29 | 20 | Candidate; manual audit pending |
| `usgs_nfm_collection_water_samples_a4` | Field sampling SOP | 41 | 63 | Candidate; manual audit pending |
| **Legacy total** | | **256** | **231** | **0 human verified; 3 excluded** |

Each candidate carries agent-authored relations. Manual audits found that document-order
`NEXT` chains manufacture execution order in NASA, OLSK, and NIOSH; these edges are not
authoritative gold. Historical bounded-excerpt files and comparisons are preserved for
provenance only.

### 2.4 Is there a label or target associated with each instance?

The annotation IS the label. Tier-A evaluation metrics (`StepF1`, `ConstraintCoverage`, `ConstraintAttachmentF1`, Φ) compare extractor output to these annotations; Tier-B compares full PKG topology via SMatch.

### 2.5 Is any information missing?

- **Independent second-pass annotations**: `second_pass/` currently holds three blank,
  anchoring-safe IAA scaffolds plus one legacy `llm_draft` file. No independent human
  second pass exists yet. At least 25% of eligible procedures must receive a frozen
  source-only pass; attachment-edge F1 ≥ 0.70 is the G0 gate and kappa is diagnostic.
- **Per-document license attribution inside the JSON**: currently consolidated in `datasets/paper/public_sources_manifest.csv`; will be denormalised into each gold's `procedure.source.license` field in the next manifest update.

### 2.6 Are there explicit relationships between individual instances?

Each gold file is independent at the procedural level. Some documents share `source_family` (3 EPA, 2 USGS), which the paper analyses for genre concentration (see §6).

### 2.7 Are there recommended data splits?

The confirmatory source-family-held-out split is not frozen yet. Development and test
membership must be explicit before tuning or full evaluation. Directory presence is not
split membership, and the legacy eight files must not all be reported as test data.

### 2.8 Are there errors, sources of noise, or redundancies?

- **Model-assisted scaffolding:** all files began with model assistance and agent review;
  none is human verified. The 2026-07-06 pass did not establish semantic correctness.
- **Known invalid current golds:** NASA is the wrong genre; OLSK invents actions from
  parts nouns; NIOSH omits safety, formula, branch, and boundary content. See
  `docs/annotation/manual-review/`.
- **Scope:** several `full_subprocedure` claims are false or mismatched with full-source
  execution. Scope must be corrected per file before inclusion.
- **Relations:** current linear chains are not safe as ordering gold until manually
  rebuilt and relation semantics are defined.

### 2.9 Is the dataset self-contained?

Yes. Each instance bundles the source text (`datasets/paper/text/<doc>.txt`), the gold JSON, the source-document checksum (in `public_sources_manifest.csv`), and (where applicable) the second_pass file. The original source PDF is not redistributed; the manifest carries a stable URL.

### 2.10 Does the dataset contain confidential data, PII, or content that might be offensive?

No. All documents are public-domain or open-licensed government/regulatory/educational SOPs. No partner-private SOPs are included.

## 3. Collection process

### 3.1 How was the data acquired?

Document selection criteria:

1. Public-domain or open-licensed (CC-BY, US federal works, open-source workbooks).
2. Stable URL on the issuing organisation's site.
3. Procedural content with a coherent, complete multi-step + multi-constraint sub-procedure suitable for full-subprocedure annotation.

SHA-256 checksums and source URLs are recorded in
`datasets/paper/public_sources_manifest.csv`. The historical acquisition and conversion
commands are incomplete; the named download/extraction scripts no longer exist. Issue
#87 requires recording the command actually used for every retained or new source.

### 3.2 What mechanisms were used?

- PDF-to-text conversion with incomplete historical command provenance.
- Model-assisted draft and agent-adjudication pipelines.
- Manual source audits recorded under `docs/annotation/manual-review/`.
- No completed human verification as of 2026-07-11.

### 3.3 Who was involved and how were they compensated?

- Corpus selection, engineering, and agent-assisted draft work: Imad.
- Future independent annotators: not yet recruited. Compensation, consent, credit, and
  authorship follow actual contribution and institutional policy; annotation count alone
  does not guarantee authorship.

### 3.4 Over what timeframe?

- Seed-corpus draft pipeline: April-June 2026.
- Agent review of the historical thin bounded excerpts: 13 June 2026.
- Locked taxonomy applied: 13 June 2026.
- Full-subprocedure re-annotation (model-assisted + agent adjudication): 4-5 July 2026.
- Source-verbatim grounding + completion pass: 6 July 2026.
- Manual source audit started: 10 July 2026; three current golds excluded by 11 July.
- Production-human evidence collection: not started; 0 frozen packages or final files.
- Planned independent IAA pass: July-August 2026.
- Corpus expansion to 12 documents: July-September 2026.

### 3.5 Were any ethical review processes conducted?

The source documents contain no intended human-subject data. Independent annotation and
any expert study still require an institutional determination covering consent, labor,
credit, and publication use before recruitment.

## 4. Preprocessing / cleaning / labelling

### 4.1 Was preprocessing done?

- PDF → text extraction (lossy for tables and figures).
- Normalisation of whitespace, hyphenation, and OCR artifacts.
- Model-assisted JSON drafts and agent adjudication produced the current candidates.
  They are not human-verified gold; manual audits have already found material defects.

### 4.2 Was the raw data saved?

Yes. The extracted `datasets/paper/text/*.txt` files are committed; the source PDFs are not redistributed but URLs and checksums are recorded.

### 4.3 Is the preprocessing software available?

Only the currently committed preprocessing and validation tools are available. The
historical download and extraction commands are not reproducible yet and must be repaired
per issue #87.

## 5. Uses

### 5.1 Has the dataset been used yet?

Only in historical and development diagnostics. Existing thesis, D1, and D2-style
outputs are not confirmatory method-paper evidence.

### 5.2 What tasks could it be used for?

- Controlled evaluation of procedural extraction methods.
- Source-grounded constraint-attachment analysis.
- Annotation-methodology research after human verification.

### 5.3 What tasks should it NOT be used for?

- Training extractors via direct fine-tuning on the gold (the corpus is too small).
- Generalisation claims about industrial procedure extraction at scale (8 docs, US-gov-environmental/safety concentrated).
- Anything involving partner-private SOPs (the corpus does not include any).

## 6. Distribution and licensing

### 6.1 Will the dataset be distributed?

The candidate files are currently in the public repository. This does not settle
redistribution rights for every historical Git blob. Issue #110 tracks the document-by-
document and history-level rights audit before any formal dataset release.

### 6.2 How?

GitHub repository: `https://github.com/imaddde867/IPKE`. Released under the repo's `LICENSE` for the annotation layer; per-document source content carries its own license tracked in `public_sources_manifest.csv`.

### 6.3 When?

No formal release date is committed. Archive only an eligible, rights-cleared corpus
after the confirmatory split and target venue are fixed.

### 6.4 Will there be terms of use or license?

The final annotation-layer license and per-document redistribution set must be confirmed
after issue #110. Users must follow each source's actual license.

### 6.5 Have any third parties imposed restrictions?

Potential source-specific restrictions and historical redistribution uncertainty remain
under review in issue #110.

### 6.6 Are there export controls?

No.

## 7. Maintenance

### 7.1 Who is supporting / hosting the dataset?

Imad Eddine Elmouss currently maintains the repository. Long-term hosting and maintenance
are not yet committed.

### 7.2 How can contributors / users reach the maintainer?

GitHub issues on `https://github.com/imaddde867/IPKE` and email above.

### 7.3 Is there an erratum?

Errata are recorded in issues, manual-review notes, commits, and per-document provenance.
Do not overwrite historical results silently.

### 7.4 Will the dataset be updated?

Yes. Planned updates:

- Corpus repair and expansion to 12 eligible procedures.
- Independent second-pass annotations as recruited annotators submit.
- Datasheet regeneration at every corpus change.

### 7.5 If the dataset relates to people, are there applicable retention limits?

N/A — documents do not contain personal data.

### 7.6 Will older versions remain supported?

Historical files and result regimes remain preserved and explicitly labeled. Release tag
names are not fixed yet.

### 7.7 Mechanism for contributing?

Pull requests against `https://github.com/imaddde867/IPKE`. New annotations must follow `docs/annotation/guidelines.md` and pass `scripts/validate_paper_gold.py`. The constraint-blindness reporter (`scripts/constraint_blindness_report.py`) will be re-run at each corpus change to keep the §1 motivating result in sync.

---

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86-92.
- Superseded IPKE-Bench resource PRD: `docs/paper/ipke-bench-resource-prd.md`
- Annotation guidelines: `docs/annotation/guidelines.md`
- Constraint taxonomy: `docs/annotation/constraint-types.md`
- Constraint-blindness baseline: `datasets/paper/reports/constraint_blindness_v2_sbert075.json`
