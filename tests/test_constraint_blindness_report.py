from __future__ import annotations

from argparse import Namespace

from scripts.constraint_blindness_report import D1_DRAFT_REF, expectation_errors


def _args(**overrides):
    values = {
        "expect_draft_total": 32,
        "expect_reviewed_total": 117,
        "expect_recovered": 24,
        "expect_recall": 0.2051,
        "expect_expansion": 3.6562,
        "expect_tolerance": 0.0001,
    }
    values.update(overrides)
    return Namespace(**values)


def _report(**overrides):
    macro = {
        "draft_total": 32,
        "reviewed_total": 117,
        "recovered": 24,
        "recall": 0.2051,
        "expansion": 3.6562,
    }
    macro.update(overrides)
    return {"macro": macro}


def test_d1_draft_ref_is_pinned_to_seed_draft_commit() -> None:
    assert D1_DRAFT_REF == "2379c8ef8cae044c9e8b9c708c3f25faa7166ca8"


def test_expectation_errors_accepts_matching_macro_values() -> None:
    assert expectation_errors(_report(), _args()) == []


def test_expectation_errors_rejects_drifted_macro_counts() -> None:
    errors = expectation_errors(_report(draft_total=33, recovered=25), _args())

    assert "macro.draft_total: expected 32, got 33" in errors
    assert "macro.recovered: expected 24, got 25" in errors


def test_expectation_errors_rejects_drifted_float_values() -> None:
    errors = expectation_errors(_report(recall=0.3), _args())

    assert errors == ["macro.recall: expected 0.2051 ± 0.0001, got 0.3"]
