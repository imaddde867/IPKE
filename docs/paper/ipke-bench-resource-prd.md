# IPKE-Bench Resource Paper PRD

Updated: 2026-06-13. Reframed per advisor review: the benchmark is the contribution; baselines, retrieval, and reproducibility infrastructure are *demonstration experiments* within the artifact, not parallel contributions. Constraint-blindness, measured directly on the seed corpus, is the §1 motivating result.

Current execution direction: `docs/paper/2026-07-04-execution-direction.md`.

## Objective

Release **IPKE-Bench**: a constraint-aware benchmark and evaluation package for procedural knowledge extraction from safety-critical industrial documents. Land at ECIR 2027 Resource Paper Track.

The benchmark is the contribution. IPKE is the reference local/private baseline that exists to (a) show the benchmark can be cleared, (b) provide a strong starting score, and (c) demonstrate the artifact end-to-end. Method ablations are illustrative, not foundational.

## Venue

Primary: ECIR 2027 Resource Paper Track.

- 12 pages plus references; single-blind; LNCS.
- Track explicitly rewards datasets, labelled annotations, tools, protocols, documentation, licensing, reliability/utility/reuse demonstration, and maintenance plans.
- Abstract: 12 Oct 2026. Paper: 2 Nov 2026.

Fallback:

- ECIR 2027 Short Paper (6 pages) only if the artifact misses corpus/IAA gates.
- ESWC 2027 In-Use track if we add JSON-LD/RDF export + SHACL + SPARQL examples in time for a December deadline.

## Problem

Existing procedural-knowledge benchmarks measure step coverage, ordering, and graph topology. None of them treat **constraint attachment** — the explicit edge that binds a guard, parameter, or precondition to the step it governs — as a primary evaluation target.

Without that edge:

- A constraint without attachment is operationally useless: an executor cannot tell which step a guard protects.
- A retrieval system over an extracted PKG cannot answer "what must hold before step X" without an explicit attached_to / applies_to edge.
- An LLM-drafted gold (or extractor output) can score well on step F1 while completely missing the safety scaffolding around the procedural backbone.

The seed corpus already exhibits the phenomenon. The headline finding is matcher-independent: **LLM-drafted gold across 8 documents produced 3.66× fewer constraints than the reviewed gold (32 vs 117)**. Even when every produced constraint maps to one in gold (best case), the LLM under-produces by 73% of the reviewed total.

Recall numbers depend on the chosen matcher and threshold; the paper reports both extremes for transparency. At the Tier-A protocol matcher (SBERT cos ≥ 0.75 over the constraint text) the draft recovers 20.5%; at a loose cos ≥ 0.50 it recovers 61.0% — see `datasets/paper/reports/constraint_blindness_v2_sbert075.json` and `_sbert050.json`. The per-type breakdown is presented with the threshold attached inline (not as standalone headline numbers), because small per-type sample sizes (n=2 for `reference`, n=9 for `role_assignment`) make extreme recall values brittle.

The durable claim for §1 is the 3.66× under-production, not the per-type recall percentages.

## Intended users

- IR/NLP researchers working on procedural KG extraction, evaluating any extractor (open-source, commercial, fine-tuned, local).
- Industrial AI teams comparing local-first vs cloud extraction on safety-critical documents.
- Annotation methodology researchers — IPKE-Bench's locked constraint taxonomy and verbatim-wording rule are reusable beyond this corpus.
- CoRe internal projects (alkio, antton, riku, TEHAA, AIOP) that need a shared annotation schema for procedural extraction.

## Resource surface

The released artifact MUST include:

1. **Document manifest** (`datasets/paper/public_sources_manifest.csv`): URL, SHA-256, license, family, conversion command, review status.
2. **Source text** (`datasets/paper/text/*.txt`): pipeline-input text for every annotated document.
3. **Tier-A reviewed gold** (`datasets/paper/gold/*.json`): step + constraint + attachment + provenance, all `review_status = reviewed`.
4. **Independent second-pass annotations** (`datasets/paper/second_pass/*.json`): produced by recruited human annotators, blind to gold.
5. **Annotation guidelines** (`docs/annotation/guidelines.md`) + **locked constraint taxonomy** (`docs/annotation/constraint-types.md`).
6. **Validation tooling**: `scripts/validate_paper_gold.py` (paper-grade strict validator), `scripts/compute_iaa.py` (κ, F1), `scripts/migrate_constraint_types.py` (reproducible taxonomy migration).
7. **Constraint-blindness reporter** (`scripts/constraint_blindness_report.py`): regenerates the §1 motivating table from any LLM-draft snapshot.
8. **Evaluation harness**: Tier-A metrics (StepF1, ConstraintCoverage, ConstraintAttachmentF1 strict + fuzzy, Φ), Tier-B SMatch graph metrics, paired bootstrap CI, multi-seed runner.
9. **Datasheet** (`docs/dataset/datasheet.md`): motivation, composition, collection process, preprocessing, labelers, uses, distribution, maintenance — per Gebru et al. 2021.
10. **Reproducibility commands**: documented in `REPRODUCIBILITY.md`; one-command `make eval` regeneration of every paper table.

## Primary contribution (singular)

The IPKE-Bench dataset, taxonomy, evaluation protocol, and reproducibility package, released together under CC-BY (per-document licenses respected), with the constraint-attachment edge as the first-class evaluation target.

## Demonstration experiments (inside the contribution)

Three demonstration experiments support the benchmark, ranked by paper priority:

- **D1. Constraint-blindness baseline** — **REQUIRED, DONE.** Per-type recall of LLM-drafted gold against reviewed gold using the Tier-A protocol matcher (SBERT cos ≥ 0.75). Headline numbers in `datasets/paper/reports/constraint_blindness_v2_sbert075.json`. Shows the benchmark is non-trivial before any extractor is run, with a clean reproducible script.

- **D2. Local baseline sweep** — **EXPECTED BEFORE SUBMISSION.** 4 configs (Fixed/DSC × P0/P3) × 5 seeds × 12 documents. Φ + 95% CI + paired bootstrap. Demonstrates the benchmark discriminates configurations. Drop only if reviewed-gold + IAA gates slip past mid-September.

- **D3. Constraint-aware retrieval** — **NICE TO HAVE.** 80 queries (20 × 4 constraint types) over text-RAG vs PKG-backed retrieval. Shows the dataset enables a second downstream task beyond extraction. Drop if D2 takes longer than 2 weeks; the artifact stands without it.

The artifact is the contribution. Order of cuts under time pressure: D3 → D2 → never D1.

## Required evaluation (acceptance gates)

### P0 — must hold before submission

- **Corpus**: target 12 documents reviewed and released. Current: **8 reviewed / 12 target** (4 short). All 8 currently-included gold files have `review_status = "reviewed"` and pass `scripts/validate_paper_gold.py` with no errors.
- All constraints use the locked taxonomy (6 types) and have `enforcement ∈ {must, should, may}`. ✅ done across the 8 reviewed files.
- **≥ 4 documents (≥ 30% of the 12-doc target) have independent second-pass annotation** by a human annotator blind to gold during their pass. LLM-drafted `second_pass/*.json` files (marked `review_status="llm_draft"`) do NOT count. **Open** — recruitment pending.
- Every IAA pair has κ ≥ 0.61 (substantial, Landis & Koch). Target κ ≥ 0.70. **Open**.
- Multi-seed (N=5) baseline sweep complete; CIs and bootstrap p-values reported. **Open** (D2 in §Demonstration experiments).
- **Datasheet** (`docs/dataset/datasheet.md`, Gebru format) and annotation guidelines committed. ✅ done as of 2026-06-13 sprint; final datasheet polish remains open for the 12-document corpus.
- `make eval` regenerates every paper table on a fresh clone. ✅ wired for D1; D2/D3 wire-up follows their implementation.

### P1 — strongly recommended

- 12-15 documents, with **genre diversity** beyond US-government environmental/safety. Add at least one each: industrial maintenance OEM, aviation/transport, food safety HACCP, IT/cybersecurity SOP, or pharma.
- Per-type constraint breakdown in baseline results.
- Constraint-aware retrieval task (D3).
- PAGED metric comparison row.
- JSON-LD export example.

### P2 — nice to have

- Expert human study (Spearman ρ between Φ and trust ratings, n≈3 raters, 40-60 extractions).
- Finnish-language extension if CoRe partner provides SOPs.

## Annotation methodology (locked)

- **Constraint taxonomy**: 6 types (`precondition`, `postcondition`, `guard`, `parameter`, `role_assignment`, `reference`) × 3 enforcement levels (`must`, `should`, `may`). See `docs/annotation/constraint-types.md`.
- **Verbatim wording**: constraint text MUST be drawn from the source verbatim or near-verbatim. No paraphrasing.
- **IAA independence**: second annotators MUST NOT see gold or any other annotator's pass until their own is committed. This is the non-negotiable rule that gives κ statistical meaning.
- **Annotation scope**: bounded_excerpt of 1-3 pages with ≥ 4 steps and ≥ 6 constraints is the default. Full-procedure scope allowed when the document is small enough. Justify in `quality.review_notes`.

The `bounded_excerpt` choice is justified by (a) controlling annotation cost for the seed corpus and (b) keeping each document's procedural complexity in a single comparable band (multi-step + multi-constraint, 1-3 pages). Reviewers will see this justification in §3 of the paper.

## Non-goals

- No annotation UI for this paper.
- No fine-tuning for this paper.
- No multimodal diagrams or P&IDs.
- No partner-private SOPs in the public release.
- No expanding model families before reviewed gold + IAA gates close.
- No re-annotating second_pass to match gold (drift correction; methodologically invalid for IAA).

## Current blockers (in critical-path order)

1. **Independent annotators** not recruited. Existing `second_pass` files appear LLM-drafted; they do not satisfy P0's IAA requirement. Lead time: weeks. Recruitment outreach pending.
2. **Corpus at 8 docs** vs 12-doc P0 target. Genre concentration is US-gov environmental/safety; the 4 new documents must diversify. Candidate targets:
   - **FAA AC 43.13-1B** (aviation maintenance, public domain, US FAA)
   - **FDA Food Code** (food safety procedures, public domain, US FDA)
   - **NIST SP 800-61 Rev. 2** (computer security incident handling, US gov, public domain)
   - **Open-license OEM service manual** — candidate sources: John Deere Operator Manuals (some are open), iFixit guides (CC BY-NC-SA), or further OLSK kits (CC BY-SA).
3. **Demonstration experiments (D2, D3) not started.** D2 is now the main compute blocker after D1 is pinned: first produce one real-model, non-empty metrics row, then scale to the full multi-seed grid.
4. **Datasheet finalization.** The seed-corpus datasheet exists; update it when the corpus reaches 12 documents and per-document licensing is final.

The taxonomy + guidelines + paper-grade validator + constraint-blindness report are now committed (PR #85). The κ-grade IAA blocker is the next critical-path item.

## Schedule

See `2ndBrain/Projects/IPKE Paper - Thesis to Congress/05-timeline.md` for the 6-phase plan to Nov 2 submission.

## Source pointers

- ECIR 2027 Resource Track: https://www.ecir2027.co.uk/call-for-resource-papers
- ECIR 2027 dates: https://www.ecir2027.co.uk/
- Constraint taxonomy: `docs/annotation/constraint-types.md`
- Annotation guidelines: `docs/annotation/guidelines.md`
- Constraint-blindness reports: `datasets/paper/reports/constraint_blindness_v2_sbert075.json` (Tier-A protocol matcher) and `datasets/paper/reports/constraint_blindness_v2_sbert050.json` (loose threshold sensitivity)
- Domain glossary: `CONTEXT.md`
- Reproducibility: `REPRODUCIBILITY.md`
