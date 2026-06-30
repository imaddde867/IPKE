"""Tests for the paper-grade validator. The validator is the artifact's
quality contract — it MUST reject malformed annotations."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_paper_gold import validate_file

VALID_ANNOTATION: dict = {
    "procedure": {"doc_id": "test_doc", "title": "Test"},
    "steps": [
        {
            "id": "S1",
            "label": "Do the thing",
            "constraints": [
                {
                    "id": "C1",
                    "type": "guard",
                    "enforcement": "must",
                    "text": "Be careful",
                    "attached_to": ["S1"],
                }
            ],
        }
    ],
    "constraints": [],
    "relations": [],
    "quality": {
        "review_status": "reviewed",
        "annotator": "test",
        "review_date": "2026-01-01",
    },
}


def _write(tmp_path: Path, annotation: dict) -> Path:
    p = tmp_path / "annot.json"
    p.write_text(json.dumps(annotation))
    return p


def _errors(messages: list[str]) -> list[str]:
    return [m for m in messages if m.startswith("ERROR")]


def test_valid_annotation_passes(tmp_path: Path) -> None:
    msgs = validate_file(_write(tmp_path, VALID_ANNOTATION))
    assert _errors(msgs) == []


def test_rejects_unreviewed_status(tmp_path: Path) -> None:
    a = json.loads(json.dumps(VALID_ANNOTATION))
    a["quality"]["review_status"] = "unreviewed"
    msgs = validate_file(_write(tmp_path, a))
    assert any("review_status" in e for e in _errors(msgs))


def test_rejects_missing_annotator(tmp_path: Path) -> None:
    a = json.loads(json.dumps(VALID_ANNOTATION))
    del a["quality"]["annotator"]
    msgs = validate_file(_write(tmp_path, a))
    assert any("annotator" in e for e in _errors(msgs))


def test_rejects_invalid_type(tmp_path: Path) -> None:
    a = json.loads(json.dumps(VALID_ANNOTATION))
    a["steps"][0]["constraints"][0]["type"] = "requirement"
    msgs = validate_file(_write(tmp_path, a))
    assert any("locked vocabulary" in e for e in _errors(msgs))


def test_rejects_invalid_enforcement(tmp_path: Path) -> None:
    a = json.loads(json.dumps(VALID_ANNOTATION))
    a["steps"][0]["constraints"][0]["enforcement"] = "required"
    msgs = validate_file(_write(tmp_path, a))
    assert any("enforcement" in e for e in _errors(msgs))


def test_rejects_missing_enforcement(tmp_path: Path) -> None:
    a = json.loads(json.dumps(VALID_ANNOTATION))
    del a["steps"][0]["constraints"][0]["enforcement"]
    msgs = validate_file(_write(tmp_path, a))
    assert any("enforcement" in e for e in _errors(msgs))


def test_rejects_unattached_constraint(tmp_path: Path) -> None:
    a = json.loads(json.dumps(VALID_ANNOTATION))
    del a["steps"][0]["constraints"][0]["attached_to"]
    msgs = validate_file(_write(tmp_path, a))
    assert any("attached_to" in e or "applies_to" in e for e in _errors(msgs))


def test_rejects_dangling_step_ref(tmp_path: Path) -> None:
    a = json.loads(json.dumps(VALID_ANNOTATION))
    a["steps"][0]["constraints"][0]["attached_to"] = ["S99"]
    msgs = validate_file(_write(tmp_path, a))
    assert any("not in step ids" in e for e in _errors(msgs))


def test_rejects_empty_constraint_text(tmp_path: Path) -> None:
    a = json.loads(json.dumps(VALID_ANNOTATION))
    a["steps"][0]["constraints"][0]["text"] = ""
    msgs = validate_file(_write(tmp_path, a))
    assert any("empty text" in e for e in _errors(msgs))


def test_procedure_level_constraint_covers_step(tmp_path: Path) -> None:
    """Step with 0 embedded constraints but covered by procedure-level applies_to
    should NOT trigger the 'suspicious 0 constraints' warning."""
    a = json.loads(json.dumps(VALID_ANNOTATION))
    a["steps"][0]["constraints"] = []
    a["constraints"] = [
        {
            "id": "PC1",
            "type": "guard",
            "enforcement": "must",
            "text": "Procedure-level safety constraint",
            "applies_to": ["S1"],
        }
    ]
    msgs = validate_file(_write(tmp_path, a))
    # Should pass with no warnings about S1 having 0 constraints
    assert not any("0 attached constraints" in m for m in msgs)


def test_step_with_no_constraints_warns(tmp_path: Path) -> None:
    """Step with no embedded AND no procedure-level coverage should warn."""
    a = json.loads(json.dumps(VALID_ANNOTATION))
    a["steps"][0]["constraints"] = []
    msgs = validate_file(_write(tmp_path, a))
    assert any("0 attached constraints" in m for m in msgs)


def test_adjudicated_zero_constraint_step_does_not_warn(tmp_path: Path) -> None:
    a = json.loads(json.dumps(VALID_ANNOTATION))
    a["steps"][0]["constraints"] = []
    a["quality"]["review_notes"] = (
        "step:S1 zero_constraints adjudicated. S1 is a deliberate "
        "single-action step with no source-level constraints in the bounded excerpt."
    )
    msgs = validate_file(_write(tmp_path, a))
    assert not any("0 attached constraints" in m for m in msgs)


def test_adjudicated_dense_step_does_not_warn(tmp_path: Path) -> None:
    a = json.loads(json.dumps(VALID_ANNOTATION))
    a["steps"][0]["constraints"] = [
        {
            "id": f"C{i}",
            "type": "parameter",
            "enforcement": "must",
            "text": f"Parameter {i}",
            "attached_to": ["S1"],
        }
        for i in range(11)
    ]
    a["quality"]["review_notes"] = (
        "step:S1 too_many_constraints adjudicated. S1 retains 11 "
        "constraints because the source presents the rules as one procedural step."
    )
    msgs = validate_file(_write(tmp_path, a))
    assert not any("consider splitting step" in m for m in msgs)


def test_adjudication_for_other_step_does_not_suppress(tmp_path: Path) -> None:
    """Adjudication for S1 should NOT suppress warning for S2."""
    a = json.loads(json.dumps(VALID_ANNOTATION))
    a["steps"] = [
        {"id": "S1", "label": "Step 1", "constraints": [{"id": "C1", "type": "guard", "enforcement": "must", "text": "Be careful", "attached_to": ["S1"]}]},
        {"id": "S2", "label": "Step 2", "constraints": []},
    ]
    a["quality"]["review_notes"] = "step:S1 zero_constraints adjudicated"
    msgs = validate_file(_write(tmp_path, a))
    assert any("S2" in m and "0 attached constraints" in m for m in msgs)


def test_adjudication_for_other_warning_kind_does_not_suppress(tmp_path: Path) -> None:
    """Adjudicating 'too_many_constraints' for S1 should NOT suppress 'zero_constraints' for S1."""
    a = json.loads(json.dumps(VALID_ANNOTATION))
    a["steps"][0]["constraints"] = []
    a["quality"]["review_notes"] = "step:S1 too_many_constraints adjudicated"
    msgs = validate_file(_write(tmp_path, a))
    assert any("0 attached constraints" in m for m in msgs)


def test_adjudication_requires_exact_step_id(tmp_path: Path) -> None:
    """A note containing S1 as a substring (e.g., 'S10') should NOT suppress S1."""
    a = json.loads(json.dumps(VALID_ANNOTATION))
    a["steps"] = [
        {"id": "S1", "label": "Step 1", "constraints": []},
        {"id": "S10", "label": "Step 10", "constraints": []},
    ]
    a["quality"]["review_notes"] = "step:S10 zero_constraints adjudicated"
    msgs = validate_file(_write(tmp_path, a))
    assert any("S1" in m and "0 attached constraints" in m for m in msgs)


def test_seed_corpus_passes(tmp_path: Path) -> None:
    """All 8 reviewed seed-corpus files in datasets/paper/gold must pass."""
    gold_dir = Path("datasets/paper/gold")
    if not gold_dir.exists():
        return  # tests can run outside the repo root
    failures = []
    for f in sorted(gold_dir.glob("*.json")):
        msgs = validate_file(f)
        errs = _errors(msgs)
        if errs:
            failures.append((f.name, errs))
    assert not failures, f"Seed corpus failures: {failures}"
