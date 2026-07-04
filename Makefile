.PHONY: test eval eval-full eval-thesis eval-blindness eval-validate eval-iaa smoke-extract paper-table clean-artifacts \
        gold-draft gold-adjudicate gold-pipeline iaa-setup iaa

PYTHON := uv run python

# Paper gold dirs
PAPER_GOLD     := datasets/paper/gold
PAPER_TEXT     := datasets/paper/text
PAPER_SECOND   := datasets/paper/second_pass
PAPER_REPORTS  := datasets/paper/reports
THESIS_GOLD    := datasets/archive/gold_human
THESIS_TEXT    := datasets/archive/test_data/text
D1_DRAFT_REF   := 2379c8ef8cae044c9e8b9c708c3f25faa7166ca8

test:
	uv run pytest

# Paper-grade validator: enforces locked taxonomy + IAA-ready metadata.
eval-validate:
	$(PYTHON) scripts/validate_paper_gold.py --gold-dir $(PAPER_GOLD) --strict

# Constraint-blindness baseline (D1 in PRD): regenerates the §1 motivating table.
# Uses the Tier-A protocol matcher (SBERT cos >= 0.75) plus a loose-threshold
# sensitivity run at 0.50.
eval-blindness:
	mkdir -p $(PAPER_REPORTS)
	$(PYTHON) scripts/constraint_blindness_report.py \
		--draft-ref $(D1_DRAFT_REF) \
		--matcher semantic --threshold 0.75 \
		--expect-draft-total 32 \
		--expect-reviewed-total 117 \
		--expect-recovered 24 \
		--expect-recall 0.2051 \
		--expect-expansion 3.6562 \
		--out $(PAPER_REPORTS)/constraint_blindness_v2_sbert075.json
	$(PYTHON) scripts/constraint_blindness_report.py \
		--draft-ref $(D1_DRAFT_REF) \
		--matcher semantic --threshold 0.50 \
		--expect-draft-total 32 \
		--expect-reviewed-total 117 \
		--expect-recovered 72 \
		--expect-recall 0.6154 \
		--expect-expansion 3.6562 \
		--out $(PAPER_REPORTS)/constraint_blindness_v2_sbert050.json

# IAA: meaningful only once independent (non-llm_draft) second_pass files exist.
eval-iaa: eval-validate
	$(PYTHON) scripts/compute_iaa.py \
		--gold-dir $(PAPER_GOLD) \
		--second-dir $(PAPER_SECOND) \
		--out $(PAPER_REPORTS)/iaa_latest.json

# ---------------------------------------------------------------------------
# Gold-annotation pipeline (model-assisted draft -> human/critic adjudication).
# The committed golds under datasets/paper/gold/ are the source of truth;
# these targets document and re-run the process that produced them.
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

# Deterministically re-apply a persisted adjudication decision log to a draft,
# producing a reviewed gold. Requires the draft on disk (make gold-draft first).
#   make gold-adjudicate DOC=<doc_id>
gold-adjudicate:
	@test -n "$(DOC)" || { echo "usage: make gold-adjudicate DOC=<doc_id>"; exit 2; }
	$(PYTHON) scripts/adjudicate.py replay datasets/paper/draft/$(DOC).json \
		--decisions datasets/paper/adjudication_decisions/$(DOC).json \
		--annotator "$$($(PYTHON) -c 'import json,sys;print(json.load(open("datasets/paper/adjudication_decisions/$(DOC).json"))["annotator"])')" \
		--out-dir $(PAPER_GOLD)

# One-command reproducibility gate for the annotation resource: validate the
# committed golds (strict), then regenerate the D1 constraint-blindness numbers.
# Deterministic; needs no model. This is the target a reviewer runs on a fresh clone.
gold-pipeline: eval-validate eval-blindness
	@echo "OK: 8 golds pass strict validation; D1 blindness numbers regenerated."

# Prepare the >=30% IAA double-annotation subset + blank (anchoring-safe) scaffolds.
iaa-setup:
	$(PYTHON) scripts/setup_iaa_subset.py select
	$(PYTHON) scripts/setup_iaa_subset.py scaffold

# Score inter-annotator agreement over completed subset second-pass files.
# Stages only completed subset docs (stale/non-subset files can't pollute).
iaa:
	$(PYTHON) scripts/setup_iaa_subset.py report --out $(PAPER_REPORTS)/iaa_latest.json

# Dry-run: plan only, no model needed. Safe on a fresh clone.
# Runs validator + D1 blindness report so the artifact's §1 numbers regenerate
# one-command per PRD acceptance gate.
eval: eval-validate eval-blindness
	$(PYTHON) scripts/eval_multiseed.py \
		--gold-dir $(PAPER_GOLD) \
		--text-dir $(PAPER_TEXT) \
		--seeds 5 \
		--phi-weights 0.5:0.3:0.2 \
		--phi-weights 0.4:0.4:0.2 \
		--phi-weights 0.6:0.2:0.2 \
		--dry-run

# Full sweep (D2 in PRD): requires reviewed gold AND LLM_MODEL_PATH.
eval-full: eval-validate
	$(PYTHON) scripts/eval_multiseed.py \
		--gold-dir $(PAPER_GOLD) \
		--text-dir $(PAPER_TEXT) \
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
