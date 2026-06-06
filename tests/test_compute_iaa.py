from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.compute_iaa import compare_annotations
from scripts.normalize_gold_annotations import validate_annotation_links


FIXTURE_DIR = Path("tests/fixtures/annotations")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_identical_annotations_score_perfectly() -> None:
    annotation = _load_json(FIXTURE_DIR / "annotator_a_sample.json")

    metrics = compare_annotations(annotation, copy.deepcopy(annotation))

    assert metrics["step_exact"]["f1"] == 1.0
    assert metrics["constraint_exact"]["f1"] == 1.0
    assert metrics["relation_exact"]["f1"] == 1.0
    assert metrics["token_label_kappa"] == 1.0


def test_partially_different_annotations_score_below_perfect() -> None:
    annotator_a = _load_json(FIXTURE_DIR / "annotator_a_sample.json")
    annotator_b = _load_json(FIXTURE_DIR / "annotator_b_sample.json")

    metrics = compare_annotations(annotator_a, annotator_b)

    assert metrics["step_exact"]["f1"] < 1.0
    assert metrics["constraint_exact"]["f1"] < 1.0
    assert metrics["token_label_kappa"] < 1.0


def test_empty_annotations_report_null_metrics() -> None:
    metrics = compare_annotations({"steps": [], "constraints": []}, {"steps": [], "constraints": []})

    assert metrics["step_exact"]["reference_count"] == 0
    assert metrics["step_exact"]["predicted_count"] == 0
    assert metrics["step_exact"]["precision"] is None
    assert metrics["step_exact"]["recall"] is None
    assert metrics["step_exact"]["f1"] is None
    assert metrics["constraint_exact"]["f1"] is None
    assert metrics["relation_exact"]["f1"] is None
    assert metrics["token_label_pairs"] == 0
    assert metrics["token_label_kappa"] is None


def test_missing_attachment_endpoint_fails_validation() -> None:
    annotation = _load_json(FIXTURE_DIR / "annotator_a_sample.json")
    annotation["constraints"][1]["targets"] = ["S404"]

    with pytest.raises(ValueError, match="unknown step"):
        validate_annotation_links(annotation, "sample_procedure")


def test_nested_step_constraint_fails_validation() -> None:
    annotation = _load_json(FIXTURE_DIR / "annotator_a_sample.json")
    annotation["steps"][0]["constraints"] = [
        {"id": "NC1", "text": "Use the correct PPE", "attached_to": ["S999"]}
    ]

    with pytest.raises(ValueError, match="unknown step"):
        validate_annotation_links(annotation, "sample_procedure")
