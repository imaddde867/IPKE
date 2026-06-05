.PHONY: test eval smoke-extract paper-table clean-artifacts

PYTHON := uv run python

test:
	uv run pytest

eval:
	$(PYTHON) -m src.evaluation.metrics \
		--gold_dir datasets/archive/gold_human \
		--pred_dir runs/pkg_extraction/3m_marine_oem_sop \
		--out_file runs/eval/latest_tier_a.json \
		--tier A

smoke-extract:
	$(PYTHON) scripts/run_pkg_extraction.py \
		--input-path datasets/archive/test_data/text/3m_marine_oem_sop.txt \
		--doc-id 3M_OEM_SOP \
		--output-dir runs/pkg_extraction/3m_marine_oem_sop \
		--chunking-method dsc \
		--prompting-strategy P3 \
		--gpu-backend cpu

paper-table:
	$(PYTHON) scripts/experiments/build_summary_csv.py

clean-artifacts:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('runs').rglob('.DS_Store')] if Path('runs').exists() else None"
