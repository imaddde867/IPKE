.PHONY: test eval eval-full eval-thesis smoke-extract paper-table clean-artifacts

PYTHON := uv run python

# Paper gold dirs
PAPER_GOLD  := datasets/paper/gold
PAPER_TEXT  := datasets/paper/text
THESIS_GOLD := datasets/archive/gold_human
THESIS_TEXT := datasets/archive/test_data/text

test:
	uv run pytest

# Dry-run: plan only, no model needed. Safe on a fresh clone.
eval:
	$(PYTHON) scripts/eval_multiseed.py \
		--gold-dir $(PAPER_GOLD) \
		--text-dir $(PAPER_TEXT) \
		--seeds 5 \
		--phi-weights 0.5:0.3:0.2 \
		--phi-weights 0.4:0.4:0.2 \
		--phi-weights 0.6:0.2:0.2 \
		--dry-run

# Full sweep: requires LLM_MODEL_PATH and a configured backend.
eval-full:
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
