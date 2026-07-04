from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compute_iaa import compute_iaa
from scripts.compute_iaa import validate_iaa_pair


def _annotation(*, review_status: str = "reviewed", annotator: str = "ann_a") -> dict:
    return {
        "procedure": {"doc_id": "doc", "title": "Doc"},
        "steps": [
            {
                "id": "S1",
                "label": "Open valve",
                "constraints": [
                    {
                        "id": "C1",
                        "type": "guard",
                        "enforcement": "must",
                        "text": "Wear gloves",
                        "attached_to": ["S1"],
                    }
                ],
            }
        ],
        "constraints": [],
        "quality": {
            "review_status": review_status,
            "annotator": annotator,
            "review_date": "2026-06-26",
        },
    }


def test_validate_iaa_pair_accepts_reviewed_human_pair() -> None:
    issues = validate_iaa_pair(
        _annotation(annotator="gold_ann"),
        _annotation(annotator="second_ann"),
        "doc.json",
    )
    assert issues == []


def test_validate_iaa_pair_rejects_llm_draft_second_pass() -> None:
    issues = validate_iaa_pair(
        _annotation(annotator="gold_ann"),
        _annotation(review_status="llm_draft", annotator="draft_llm"),
        "doc.json",
    )
    assert any("second quality.review_status" in issue for issue in issues)


def _write_pair(tmp_path: Path, second: dict) -> tuple[Path, Path]:
    gold_dir = tmp_path / "gold"
    second_dir = tmp_path / "second"
    gold_dir.mkdir()
    second_dir.mkdir()
    (gold_dir / "doc.json").write_text(json.dumps(_annotation(annotator="gold_ann")), encoding="utf-8")
    (second_dir / "doc.json").write_text(json.dumps(second), encoding="utf-8")
    return gold_dir, second_dir


def test_validate_iaa_pair_rejects_missing_second_review_date() -> None:
    second = _annotation(annotator="second_ann")
    del second["quality"]["review_date"]
    issues = validate_iaa_pair(_annotation(annotator="gold_ann"), second, "doc.json")
    assert any("second quality.review_date missing" in issue for issue in issues)


def test_validate_iaa_pair_rejects_invalid_second_type() -> None:
    second = _annotation(annotator="second_ann")
    second["steps"][0]["constraints"][0]["type"] = "warning"
    issues = validate_iaa_pair(_annotation(annotator="gold_ann"), second, "doc.json")
    assert any("locked vocabulary" in issue for issue in issues)


def test_validate_iaa_pair_rejects_same_annotator() -> None:
    issues = validate_iaa_pair(
        _annotation(annotator="same"),
        _annotation(annotator="same"),
        "doc.json",
    )
    assert any("annotator must differ" in issue for issue in issues)


def test_compute_iaa_fails_on_invalid_second_pass(tmp_path: Path) -> None:
    gold_dir, second_dir = _write_pair(
        tmp_path,
        _annotation(review_status="llm_draft", annotator="draft_llm"),
    )
    with pytest.raises(SystemExit, match="IAA eligibility failed"):
        compute_iaa(gold_dir, second_dir)
