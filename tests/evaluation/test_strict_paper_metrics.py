from __future__ import annotations

import pytest

from src.evaluation.metrics_tier_a import normalize_doc_constraints


def _doc_with_constraint(*, ctype: str = "guard", enforcement: str | None = "must") -> dict:
    constraint = {
        "id": "C1",
        "type": ctype,
        "text": "Wear gloves",
        "attached_to": ["S1"],
    }
    if enforcement is not None:
        constraint["enforcement"] = enforcement
    return {
        "steps": [
            {
                "id": "S1",
                "label": "Handle sample",
                "constraints": [constraint],
            }
        ],
        "constraints": [],
    }


def test_strict_normalization_accepts_locked_constraint() -> None:
    constraints = normalize_doc_constraints(_doc_with_constraint(), strict_paper=True)
    assert constraints[0]["type"] == "guard"
    assert constraints[0]["enforcement"] == "must"


def test_strict_normalization_rejects_legacy_warning_type() -> None:
    with pytest.raises(ValueError, match="locked vocabulary"):
        normalize_doc_constraints(_doc_with_constraint(ctype="warning"), strict_paper=True)


def test_strict_normalization_rejects_missing_enforcement() -> None:
    with pytest.raises(ValueError, match="enforcement"):
        normalize_doc_constraints(_doc_with_constraint(enforcement=None), strict_paper=True)


def test_legacy_normalization_still_accepts_warning_type() -> None:
    constraints = normalize_doc_constraints(_doc_with_constraint(ctype="warning"), strict_paper=False)
    assert constraints[0]["type"] == "warning"
