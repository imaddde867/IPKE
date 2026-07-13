.PHONY: test eval eval-full eval-thesis eval-blindness eval-validate eval-paper-gate eval-iaa smoke-extract paper-table clean-artifacts \
        gold-draft gold-adjudicate gold-pipeline iaa-setup iaa repro-blindness

PYTHON := uv run python

# Paper gold dirs
PAPER_MANIFEST := datasets/paper/corpus_manifest.json
PAPER_GOLD     := datasets/paper/gold
PAPER_TEXT     := datasets/paper/text
PAPER_PRODUCTION := datasets/paper/production
PAPER_EVIDENCE := datasets/paper/evidence
PAPER_SECOND   := datasets/paper/second_pass
PAPER_REPORTS  := datasets/paper/reports
THESIS_GOLD    := datasets/archive/gold_human
THESIS_TEXT    := datasets/archive/test_data/text
D1_DRAFT_REF   := 2379c8ef8cae044c9e8b9c708c3f25faa7166ca8

test:
	uv run pytest

# Structural validator: enforces locked taxonomy + IAA-ready metadata.
eval-validate:
	$(PYTHON) scripts/validate_paper_gold.py --gold-dir $(PAPER_GOLD) --strict

# Paper-evidence release gate: exact anchors plus frozen human evidence.
eval-paper-gate:
	$(PYTHON) scripts/validate_paper_gold.py --gold-dir $(PAPER_PRODUCTION) \
		--manifest $(PAPER_MANIFEST) --require-frozen-manifest \
		--text-dir $(PAPER_TEXT) --evidence-dir $(PAPER_EVIDENCE) \
		--strict --require-production-evidence

# Constraint-blindness baseline (D1 in PRD): regenerates the §1 motivating table.
# Uses the Tier-A protocol matcher (SBERT cos >= 0.75) plus a loose-threshold
# sensitivity run at 0.50.
#
# NOTE (2026-07): the --expect-* reproducibility pins were retired here. They
# encoded the *thin-gold* numbers (reviewed_total 117, expansion 3.66x); after
# the deep re-annotation the golds are 5.9x deeper, so the fixed D1 draft
# (generated at the old scope) vs the deep golds is a CROSS-REGIME comparison.
# D1 scope DECIDED 2026-07-06 (option 2): S1 leads with corpus depth; the
# cross-regime ratio stays a labelled secondary illustration, pinned in
# `make repro-blindness` (non-gate). See docs/paper/D1_SCOPE_DECISION.md.
# This target regenerates + prints; it is not a pass/fail gate.
eval-blindness:
	mkdir -p $(PAPER_REPORTS)
	$(PYTHON) scripts/constraint_blindness_report.py \
		--draft-ref $(D1_DRAFT_REF) \
		--matcher semantic --threshold 0.75 \
		--out $(PAPER_REPORTS)/constraint_blindness_v2_sbert075.json
	$(PYTHON) scripts/constraint_blindness_report.py \
		--draft-ref $(D1_DRAFT_REF) \
		--matcher semantic --threshold 0.50 \
		--out $(PAPER_REPORTS)/constraint_blindness_v2_sbert050.json

# Pinned reproduction of the historical D1 cross-regime illustration (2026-07-06
# agent-adjudicated state: 231 constraints after the verbatim-grounding pass).
# This is a structural snapshot, not human-verified paper evidence. Fails loudly if
# the committed golds or the fixed draft drift.
repro-blindness:
	mkdir -p $(PAPER_REPORTS)
	$(PYTHON) scripts/constraint_blindness_report.py \
		--draft-ref $(D1_DRAFT_REF) \
		--matcher semantic --threshold 0.75 \
		--expect-draft-total 32 --expect-reviewed-total 231 \
		--expect-recovered 14 --expect-recall 0.0606 --expect-expansion 7.2188 \
		--out $(PAPER_REPORTS)/constraint_blindness_v2_sbert075.json
	$(PYTHON) scripts/constraint_blindness_report.py \
		--draft-ref $(D1_DRAFT_REF) \
		--matcher semantic --threshold 0.50 \
		--expect-draft-total 32 --expect-reviewed-total 231 \
		--expect-recovered 87 --expect-recall 0.3766 --expect-expansion 7.2188 \
		--out $(PAPER_REPORTS)/constraint_blindness_v2_sbert050.json
	@echo "OK: D1 cross-regime illustration reproduces (32 vs 231, 7.22x)."

# IAA: meaningful only once independent (non-llm_draft) second_pass files exist.
eval-iaa: eval-validate
	$(PYTHON) scripts/compute_iaa.py \
		--gold-dir $(PAPER_GOLD) \
		--second-dir $(PAPER_SECOND) \
		--out $(PAPER_REPORTS)/iaa_latest.json

# ---------------------------------------------------------------------------
# Historical candidate pipeline (model-assisted draft -> agent decision replay).
# The committed files under datasets/paper/gold/ are legacy candidates, not
# production gold or human evidence.
# See docs/methods/annotation-pipeline.md.
# ---------------------------------------------------------------------------

# Draft ONE procedure with the two-stage harness. Requires a model backend
# (host.llm in-session, or IPKE_LLM_BASE_URL/IPKE_LLM_MODEL/IPKE_LLM_API_KEY).
# Model outputs are non-deterministic and land as review_status="unreviewed".
#   make gold-draft DOC=<doc_id> SEG=<segments.json> CAND=<candidate_id>
gold-draft:
	@test -n "$(DOC)" && test -n "$(SEG)" && test -n "$(CAND)" || \
		{ echo "usage: make gold-draft DOC=<doc_id> SEG=<segments.json> CAND=<candidate>"; exit 2; }
	mkdir -p datasets/paper/draft
	$(PYTHON) scripts/annotate_assisted.py \
		--segments $(SEG) --candidate $(CAND) \
		--text $(PAPER_TEXT)/$(DOC).txt \
		--out datasets/paper/draft/$(DOC).json

# Deterministically re-apply a persisted agent decision log to a draft,
# reproducing a historical candidate. Requires the draft on disk first.
#   make gold-adjudicate DOC=<doc_id>
gold-adjudicate:
	@test -n "$(DOC)" || { echo "usage: make gold-adjudicate DOC=<doc_id>"; exit 2; }
	$(PYTHON) scripts/adjudicate.py replay datasets/paper/draft/$(DOC).json \
		--decisions datasets/paper/adjudication_decisions/$(DOC).json \
		--annotator "$$($(PYTHON) -c 'import json,sys;print(json.load(open("datasets/paper/adjudication_decisions/$(DOC).json"))["annotator"])')" \
		--out-dir $(PAPER_GOLD)

# Historical/development diagnostics: structurally validate the current candidate
# gold directory, then regenerate the D1 constraint-blindness numbers. Deterministic;
# needs no model. This target does not establish paper eligibility; use
# eval-paper-gate for the human-verification criterion.
gold-pipeline: eval-validate eval-blindness
	@echo "OK: development structural validation and D1 diagnostics completed."
	@echo "Paper eligibility requires: make eval-paper-gate"

# Prepare the frozen >=25% blind-annotation subset and source-only scaffolds.
iaa-setup:
	$(PYTHON) scripts/setup_iaa_subset.py select
	$(PYTHON) scripts/setup_iaa_subset.py scaffold

# Score inter-annotator agreement over completed subset second-pass files.
# Stages only completed subset docs (stale/non-subset files can't pollute).
iaa:
	$(PYTHON) scripts/setup_iaa_subset.py report --out $(PAPER_REPORTS)/iaa_latest.json

# Historical/development dry-run: plan only, no model needed. Safe on a fresh clone.
# Runs structural validation plus the legacy D1 blindness diagnostics; it is not a
# confirmatory paper-evidence command.
eval: eval-validate eval-blindness
	$(PYTHON) scripts/eval_multiseed.py \
		--gold-dir $(PAPER_GOLD) \
		--text-dir $(PAPER_TEXT) \
		--manifest $(PAPER_MANIFEST) \
		--seeds 5 \
		--phi-weights 0.5:0.3:0.2 \
		--phi-weights 0.4:0.4:0.2 \
		--phi-weights 0.6:0.2:0.2 \
		--dry-run

# Historical five-candidate development sweep from the superseded D2 design. It is not the
# current confirmatory design and cannot start until the paper-evidence gate passes.
eval-full: eval-paper-gate
	$(PYTHON) scripts/eval_multiseed.py \
		--gold-dir $(PAPER_PRODUCTION) \
		--text-dir $(PAPER_TEXT) \
		--evidence-dir $(PAPER_EVIDENCE) \
		--manifest $(PAPER_MANIFEST) \
		--seeds 5 \
		--phi-weights 0.5:0.3:0.2 \
		--phi-weights 0.4:0.4:0.2 \
		--phi-weights 0.6:0.2:0.2 \
		--out-dir results/paper/

# Thesis gold (n=3, for continuity with thesis numbers).
eval-thesis:
	$(PYTHON) -m src.evaluation.metrics \
		--gold_dir $(THESIS_GOLD) \
		--pred_dir runs/pkg_extraction/3m_marine_oem_sop \
		--out_file runs/eval/latest_tier_a.json \
		--tier A

smoke-extract:
	$(PYTHON) scripts/run_pkg_extraction.py \
		--input-path $(THESIS_TEXT)/3m_marine_oem_sop.txt \
		--doc-id 3M_OEM_SOP \
		--output-dir runs/pkg_extraction/3m_marine_oem_sop \
		--chunking-method dsc \
		--prompting-strategy P3 \
		--gpu-backend cpu

paper-table:
	$(PYTHON) scripts/experiments/build_summary_csv.py

clean-artifacts:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('runs').rglob('.DS_Store')] if Path('runs').exists() else None"
