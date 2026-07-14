from __future__ import annotations

import pytest

from src.evaluation.evidence import assess_annotation_evidence, assess_corpus_evidence


def _annotation(annotator: str, *, status: str = "reviewed") -> dict:
    return {
        "quality": {
            "review_status": status,
            "annotator": annotator,
            "review_date": "2026-07-10",
        }
    }


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (
            _annotation(
                "model-assisted:qwen + agent-adjudicated (pending human sign-off)"
            ),
            (
                True,
                False,
                False,
                ("quality.annotator still contains pending human sign-off",),
            ),
        ),
        (
            _annotation(
                "model-assisted:qwen + agent-adjudicated + human-verified:imad"
            ),
            (True, True, False, ()),
        ),
        (
            _annotation("agent-adjudicated"),
            (
                True,
                False,
                False,
                ("quality.annotator lacks a + human-verified:<handle> marker",),
            ),
        ),
        (
            {},
            (
                False,
                False,
                False,
                (
                    "quality.review_status must be 'reviewed'",
                    "quality.annotator missing",
                    "quality.review_date missing",
                ),
            ),
        ),
    ],
)
def test_assess_annotation_evidence_classifies_review_provenance(
    annotation: dict, expected: tuple[bool, bool, bool, tuple[str, ...]]
) -> None:
    result = assess_annotation_evidence(annotation)

    assert (
        result.declared_reviewed,
        result.human_verified,
        result.evidence_eligible,
        result.issues,
    ) == expected


def test_corpus_evidence_requires_blind_coverage_for_at_least_25_percent() -> None:
    logs = {
        f"doc_{index}": {"blind_subset_selected": index == 0}
        for index in range(5)
    }

    issues = assess_corpus_evidence(logs)

    assert issues == (
        "blind second-pass coverage is 1/5; at least 2/5 required",
    )


def test_corpus_evidence_applies_attachment_gate_to_aggregate_counts() -> None:
    logs = {
        "low_pair": {"blind_subset_selected": True},
        "high_pair": {"blind_subset_selected": True},
    }
    reports = {
        "low_pair": {
            "aggregate": {
                "relation_exact": {
                    "true_positive": 1,
                    "false_positive": 1,
                    "false_negative": 1,
                    "f1": 0.5,
                }
            }
        },
        "high_pair": {
            "aggregate": {
                "relation_exact": {
                    "true_positive": 8,
                    "false_positive": 0,
                    "false_negative": 0,
                    "f1": 1.0,
                }
            }
        },
    }

    assert assess_corpus_evidence(logs, agreement_reports=reports) == ()


def test_corpus_evidence_recomputes_and_rejects_low_attachment_agreement() -> None:
    logs = {"pair": {"blind_subset_selected": True}}
    reports = {
        "pair": {
            "aggregate": {
                "relation_exact": {
                    "true_positive": 1,
                    "false_positive": 1,
                    "false_negative": 1,
                    "f1": 0.99,
                }
            }
        }
    }

    assert assess_corpus_evidence(logs, agreement_reports=reports) == (
        "attachment-edge agreement F1 0.500 is below 0.700",
    )
