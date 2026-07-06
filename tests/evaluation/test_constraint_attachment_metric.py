"""Unit tests for ConstraintAttachmentF1 fix.

Covers:
1. normalize_doc_constraints flattens nested step.constraints into a flat list
   with applies_to set to the parent step id.
2. tier_a_constraints_metrics yields > 0 attachment F1 for paraphrased constraint
   text, and ~0 for unrelated text, using an injected deterministic embedder
   (no live model download required).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from src.evaluation.metrics import (
    AlignmentResult,
    TextPreprocessor,
    evaluate_tier_a_document,
    normalize_doc_constraints,
    tier_a_constraints_metrics,
)


class _StubEmbedder:
    """Deterministic embedder that returns hand-picked vectors per string.

    Anything not in the lookup table maps to a low-overlap vector so
    unrelated text gets a near-zero cosine to known anchors.
    """

    def __init__(self, vectors: Dict[str, Sequence[float]], dim: int = 8) -> None:
        self._dim = dim
        self._lookup = {text: np.asarray(vec, dtype=float) for text, vec in vectors.items()}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim))
        out = np.zeros((len(texts), self._dim))
        for idx, text in enumerate(texts):
            if text in self._lookup:
                vec = self._lookup[text]
            else:
                # Hash to a stable orthogonal-ish slot so unseen text has low
                # similarity to anchored vectors.
                slot = hash(text) % self._dim
                vec = np.zeros(self._dim)
                vec[slot] = 1.0
            norm = np.linalg.norm(vec)
            out[idx] = vec / norm if norm else vec
        return out


def _preprocessor() -> TextPreprocessor:
    return TextPreprocessor()


def test_normalize_flattens_nested_step_constraints():
    doc = {
        "steps": [
            {
                "id": "S1",
                "label": "Inspect equipment",
                "constraints": {
                    "guard": [{"id": "C1", "text": "if pressure exceeds threshold"}],
                    "warning": [{"id": "C2", "label": "hot surface hazard"}],
                    "precondition": [],
                    "postcondition": [],
                    "acceptance_criteria": [],
                },
            },
            {
                "id": "S2",
                "label": "Lock out power",
                "constraints": {"guard": [{"id": "C3", "text": "before maintenance"}]},
            },
        ],
    }
    flat = normalize_doc_constraints(doc)
    assert len(flat) == 3
    by_id = {c["id"]: c for c in flat}
    assert by_id["C1"]["applies_to"] == "S1"
    assert by_id["C1"]["type"] == "guard"
    assert by_id["C2"]["applies_to"] == "S1"
    assert by_id["C2"]["type"] == "warning"
    assert by_id["C3"]["applies_to"] == "S2"


def test_normalize_flattens_nested_list_constraints():
    """PR #57 gold shape: steps[i].constraints is a list with attached_to."""
    doc = {
        "steps": [
            {
                "id": "S2",
                "label": "Chalk the tape",
                "constraints": [
                    {
                        "id": "C1",
                        "type": "purpose",
                        "text": "Use blue chalk so the wetted mark shows.",
                        "attached_to": ["S2"],
                    }
                ],
            },
            {
                "id": "S3",
                "label": "Lower the tape",
                "constraints": [
                    {"id": "C2", "type": "guard", "text": "Do not touch sides."},
                ],
            },
        ],
    }
    flat = normalize_doc_constraints(doc)
    assert len(flat) == 2
    by_id = {c["id"]: c for c in flat}
    # C1 already has attached_to — must be preserved, no applies_to added.
    assert by_id["C1"]["attached_to"] == ["S2"]
    assert "applies_to" not in by_id["C1"]
    # C2 has no link key — applies_to synthesised.
    assert by_id["C2"]["applies_to"] == "S3"


def test_normalize_idempotent_on_flat_shape():
    doc = {
        "steps": [{"id": "S1", "label": "do thing"}],
        "constraints": [
            {"id": "C1", "text": "if x", "steps": ["S1"]},
        ],
    }
    flat = normalize_doc_constraints(doc)
    assert flat == doc["constraints"]


def test_normalize_preserves_existing_link_keys():
    doc = {
        "steps": [
            {
                "id": "S1",
                "constraints": {
                    "guard": [{"id": "C1", "text": "x", "steps": ["S2"]}],
                },
            }
        ]
    }
    flat = normalize_doc_constraints(doc)
    # Existing link keys are not overwritten by applies_to.
    assert "applies_to" not in flat[0]
    assert flat[0]["steps"] == ["S2"]


def _step_alignment_identity(steps: List[Dict[str, str]]) -> AlignmentResult:
    ids = [s["id"] for s in steps]
    matches = [(i, i, 1.0) for i in range(len(steps))]
    return AlignmentResult(matches=matches, gold_ids=ids, pred_ids=ids)


def test_paraphrased_constraint_scores_above_zero():
    # Two paraphrases of the same idea share an anchor vector.
    preprocessor = _preprocessor()
    paraphrase_a = preprocessor("If pressure exceeds threshold")
    paraphrase_b = preprocessor("when the pressure goes above the limit")
    anchor = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    embedder = _StubEmbedder({paraphrase_a: anchor, paraphrase_b: anchor})

    gold_steps = [{"id": "S1", "label": "step"}]
    gold = [{"id": "C1", "text": "If pressure exceeds threshold", "applies_to": "S1"}]
    pred = [{"id": "PC1", "text": "when the pressure goes above the limit", "applies_to": "S1"}]
    step_alignment = _step_alignment_identity(gold_steps)
    result = tier_a_constraints_metrics(
        gold, pred, preprocessor, embedder, threshold=0.75, step_alignment=step_alignment
    )
    assert result["ConstraintCoverage"] is not None
    assert result["ConstraintCoverage"] > 0.0
    assert result["ConstraintAttachmentF1"] is not None
    assert result["ConstraintAttachmentF1"] > 0.0


def test_unrelated_constraint_scores_below_threshold():
    preprocessor = _preprocessor()
    gold_text = preprocessor("If pressure exceeds threshold")
    pred_text = preprocessor("wear blue gloves on Tuesdays")
    embedder = _StubEmbedder(
        {
            gold_text: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            pred_text: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
    gold_steps = [{"id": "S1", "label": "step"}]
    gold = [{"id": "C1", "text": "If pressure exceeds threshold", "applies_to": "S1"}]
    pred = [{"id": "PC1", "text": "wear blue gloves on Tuesdays", "applies_to": "S1"}]
    step_alignment = _step_alignment_identity(gold_steps)
    result = tier_a_constraints_metrics(
        gold, pred, preprocessor, embedder, threshold=0.75, step_alignment=step_alignment
    )
    coverage = result["ConstraintCoverage"] or 0.0
    attachment = result["ConstraintAttachmentF1"] or 0.0
    assert coverage < 0.1
    assert attachment < 0.1


def test_evaluate_tier_a_uses_nested_gold():
    preprocessor = _preprocessor()
    gold_label = preprocessor("Inspect equipment")
    pred_label = preprocessor("Inspect equipment")
    gold_c = preprocessor("if pressure exceeds threshold")
    pred_c = preprocessor("when the pressure goes above the limit")
    embedder = _StubEmbedder(
        {
            gold_label: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            pred_label: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            gold_c: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            pred_c: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    gold_doc = {
        "steps": [
            {
                "id": "S1",
                "label": "Inspect equipment",
                "constraints": {
                    "guard": [{"id": "C1", "text": "if pressure exceeds threshold"}],
                    "precondition": [],
                    "postcondition": [],
                    "acceptance_criteria": [],
                    "warning": [],
                },
            }
        ]
    }
    pred_doc = {
        "steps": [{"id": "S1", "text": "Inspect equipment"}],
        "constraints": [
            {"id": "PC1", "text": "when the pressure goes above the limit", "steps": ["S1"]}
        ],
    }
    metrics = evaluate_tier_a_document(gold_doc, pred_doc, preprocessor, embedder, threshold=0.75)
    assert metrics["ConstraintCoverage"] is not None
    assert metrics["ConstraintCoverage"] > 0.0
    assert metrics["ConstraintAttachmentF1"] is not None
    assert metrics["ConstraintAttachmentF1"] > 0.0


def test_metric_is_deterministic():
    preprocessor = _preprocessor()
    paraphrase_a = preprocessor("If pressure exceeds threshold")
    paraphrase_b = preprocessor("when the pressure goes above the limit")
    embedder = _StubEmbedder(
        {paraphrase_a: [1.0, 0.0, 0, 0, 0, 0, 0, 0], paraphrase_b: [1.0, 0.0, 0, 0, 0, 0, 0, 0]}
    )
    gold_steps = [{"id": "S1", "label": "step"}]
    gold = [{"id": "C1", "text": "If pressure exceeds threshold", "applies_to": "S1"}]
    pred = [{"id": "PC1", "text": "when the pressure goes above the limit", "applies_to": "S1"}]
    step_alignment = _step_alignment_identity(gold_steps)
    out_a = tier_a_constraints_metrics(
        gold, pred, preprocessor, embedder, threshold=0.75, step_alignment=step_alignment
    )
    out_b = tier_a_constraints_metrics(
        gold, pred, preprocessor, embedder, threshold=0.75, step_alignment=step_alignment
    )
    assert out_a == out_b
