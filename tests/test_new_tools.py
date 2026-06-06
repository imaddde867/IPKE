"""Unit tests for tools/annotate_gold.py, tools/iaa_check.py, and scripts/eval_multiseed.py.

All tests run without a live model, GPU, or network access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.annotate_gold import _extraction_to_draft, _load_schema, _validate
from scripts.compute_iaa import compare_annotations
from scripts.eval_multiseed import _ci95, _summary_table, METRIC_COLUMNS


# ---------------------------------------------------------------------------
# annotate_gold helpers
# ---------------------------------------------------------------------------

def test_extraction_to_draft_basic():
    payload = {
        "steps": [
            {"id": "S1", "text": "Inspect the pump", "order": 1},
            {"id": "S2", "text": "Tighten the bolts", "order": 2},
        ],
        "constraints": [
            {"id": "C1", "text": "Wear gloves", "steps": ["S1"]},
        ],
    }
    draft = _extraction_to_draft("doc1", payload, "Test Procedure", "maintenance")
    assert draft["procedure"]["doc_id"] == "doc1"
    assert draft["procedure"]["title"] == "Test Procedure"
    assert len(draft["steps"]) == 2
    s1 = next(s for s in draft["steps"] if s["id"] == "S1")
    assert s1["label"] == "Inspect the pump"
    # Constraint should be nested under S1
    assert len(s1["constraints"]) == 1
    assert s1["constraints"][0]["id"] == "C1"
    # S2 has no constraints
    s2 = next(s for s in draft["steps"] if s["id"] == "S2")
    assert s2["constraints"] == []


def test_extraction_to_draft_missing_text_falls_back_to_id():
    payload = {"steps": [{"id": "S1"}], "constraints": []}
    draft = _extraction_to_draft("doc1", payload, "", "")
    assert draft["steps"][0]["label"] == "S1"


def test_extraction_to_draft_validates_against_schema(tmp_path):
    payload = {
        "steps": [{"id": "S1", "text": "Do the thing", "order": 1}],
        "constraints": [],
    }
    schema = _load_schema()
    draft = _extraction_to_draft("test_doc", payload, "A Procedure", "test")
    errors = _validate(draft, schema)
    assert errors == [], f"Unexpected validation errors: {errors}"


def test_validate_reports_missing_procedure():
    schema = _load_schema()
    bad_draft = {"steps": [{"id": "S1", "label": "step"}]}
    errors = _validate(bad_draft, schema)
    assert any("procedure" in e for e in errors)


def test_validate_reports_empty_steps():
    schema = _load_schema()
    bad_draft = {"procedure": {"doc_id": "x", "title": "y"}, "steps": []}
    errors = _validate(bad_draft, schema)
    assert any("steps" in e for e in errors)


# ---------------------------------------------------------------------------
# iaa_check — exercises compare_annotations which iaa_check wraps
# ---------------------------------------------------------------------------

def test_iaa_identical_annotations_perfect_scores():
    ann = {
        "steps": [{"id": "S1", "label": "Inspect the pump"}],
        "constraints": [{"id": "C1", "text": "Wear gloves", "applies_to": ["S1"]}],
    }
    import copy
    metrics = compare_annotations(ann, copy.deepcopy(ann))
    assert metrics["step_exact"]["f1"] == 1.0
    assert metrics["constraint_exact"]["f1"] == 1.0
    assert metrics["token_label_kappa"] == 1.0


def test_iaa_different_annotations_lower_scores():
    ann_a = {
        "steps": [
            {"id": "S1", "label": "Inspect the pump"},
            {"id": "S2", "label": "Tighten the bolts"},
        ],
        "constraints": [{"id": "C1", "text": "Wear gloves", "applies_to": ["S1"]}],
    }
    ann_b = {
        "steps": [
            {"id": "S1", "label": "Inspect the pump"},
            {"id": "S2", "label": "Close the valve"},
        ],
        "constraints": [],
    }
    metrics = compare_annotations(ann_a, ann_b)
    assert metrics["step_exact"]["f1"] < 1.0
    assert metrics["constraint_exact"]["f1"] is not None


# ---------------------------------------------------------------------------
# eval_multiseed helpers
# ---------------------------------------------------------------------------

def test_ci95_correct_interval():
    vals = [0.70, 0.75, 0.72]
    lo, hi = _ci95(vals)
    mean = sum(vals) / 3
    # Interval must be centred on mean and symmetric
    assert abs((lo + hi) / 2 - mean) < 1e-9
    assert hi > mean > lo


def test_ci95_single_value_returns_nan():
    lo, hi = _ci95([0.5])
    import math
    assert math.isnan(lo) and math.isnan(hi)


def test_summary_table_columns_and_ci():
    rows = [
        {"model_id": "m1", "chunker": "dsc", "prompter": "P3", "doc_id": "d1", "seed": i,
         "StepF1": 0.7 + i * 0.01, "AdjacencyF1": 0.5, "Kendall": 0.8,
         "ConstraintCoverage": 0.6, "ConstraintAttachmentF1": 0.4, "Phi": 0.65}
        for i in range(3)
    ]
    table = _summary_table(rows)
    for col in METRIC_COLUMNS:
        assert col in table
    assert "m1" in table
    assert "dsc" in table
    # CI bracket should appear
    assert "[" in table and "]" in table


def test_summary_table_handles_none_values():
    # A group where all metric values are None produces no data row (nothing to
    # summarise), so only the header appears in the output.
    rows = [
        {"model_id": "m1", "chunker": "dsc", "prompter": "P3", "doc_id": "d1", "seed": 0,
         "StepF1": None, "AdjacencyF1": None, "Kendall": None,
         "ConstraintCoverage": None, "ConstraintAttachmentF1": None, "Phi": None}
    ]
    table = _summary_table(rows)
    assert "StepF1" in table  # header present
    # No data row for m1 since all values are None
    assert "m1" not in table
