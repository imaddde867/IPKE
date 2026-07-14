"""Tests for the paper-grade validator. The validator is the artifact's
quality contract — it MUST reject malformed annotations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.validate_paper_gold import main, validate_file

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


def _production_package(tmp_path: Path) -> tuple[Path, Path, Path]:
    paper_dir = tmp_path / "datasets" / "paper"
    gold_dir = paper_dir / "production"
    text_dir = paper_dir / "text"
    evidence_dir = paper_dir / "evidence"
    primary_dir = paper_dir / "primary_pass"
    gold_dir.mkdir(parents=True)
    text_dir.mkdir()
    evidence_dir.mkdir()
    primary_dir.mkdir()
    source_text = "Do the thing.\nBe careful.\n"
    annotation = json.loads(json.dumps(VALID_ANNOTATION))
    annotation["procedure"]["source"] = {
        "doc_id": "test_doc",
        "section": "Test",
        "char_start": 0,
        "char_end": len(source_text),
    }
    annotation["steps"][0]["provenance"] = {
        "doc_id": "test_doc",
        "char_start": 0,
        "char_end": len("Do the thing."),
    }
    constraint = annotation["steps"][0]["constraints"][0]
    constraint["provenance"] = {
        "doc_id": "test_doc",
        "char_start": source_text.index("Be careful."),
        "char_end": source_text.index("Be careful.") + len("Be careful."),
    }
    annotation["quality"]["annotator"] = "P-001"
    annotation_bytes = json.dumps(annotation, indent=2).encode("utf-8")
    source_bytes = source_text.encode("utf-8")

    gold_path = gold_dir / "test_doc.json"
    gold_path.write_bytes(annotation_bytes)
    (primary_dir / "test_doc.json").write_bytes(annotation_bytes)
    source_path = text_dir / "test_doc.txt"
    source_path.write_bytes(source_bytes)
    output_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    evidence_log = {
        "schema_version": 1,
        "protocol_id": "ipke-human-evidence-v1",
        "status": "frozen",
        "doc_id": "test_doc",
        "source": {
            "path": "datasets/paper/text/test_doc.txt",
            "url": "https://example.org/test-doc",
            "retrieval_date": "2026-07-13",
            "version": "test-v1",
            "page_range": "1",
            "section": "Test",
            "redistribution_status": "public-domain test fixture",
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "span_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "char_start": 0,
            "char_end": len(source_text),
        },
        "candidate": None,
        "primary_pass": {
            "reviewer": {
                "handle": "P-001",
                "kind": "human",
                "role": "primary_reviewer",
            },
            "started_at": "2026-07-13T09:00:00Z",
            "completed_at": "2026-07-13T09:30:00Z",
            "active_minutes": 30,
            "assistance": {
                "candidate_used": False,
                "complete_source_pass": True,
                "exact_anchors_checked": True,
            },
            "decision_counts": {
                "steps": {
                    "candidate_count": 0,
                    "accepted": 0,
                    "edited": 0,
                    "rejected": 0,
                    "added": 1,
                    "final_count": 1,
                },
                "constraints": {
                    "candidate_count": 0,
                    "accepted": 0,
                    "edited": 0,
                    "rejected": 0,
                    "added": 1,
                    "final_count": 1,
                },
                "relations": {
                    "candidate_count": 0,
                    "accepted": 0,
                    "edited": 0,
                    "rejected": 0,
                    "added": 0,
                    "final_count": 0,
                },
            },
            "decisions": [
                {
                    "decision_id": "D-S1",
                    "item_kind": "step",
                    "action": "add",
                    "input_ids": [],
                    "output_ids": ["S1"],
                    "evidence_spans": [
                        {"char_start": 0, "char_end": len("Do the thing.")}
                    ],
                    "rationale": "Source-only annotation.",
                },
                {
                    "decision_id": "D-C1",
                    "item_kind": "constraint",
                    "action": "add",
                    "input_ids": [],
                    "output_ids": ["C1"],
                    "evidence_spans": [
                        {
                            "char_start": source_text.index("Be careful."),
                            "char_end": (
                                source_text.index("Be careful.")
                                + len("Be careful.")
                            ),
                        }
                    ],
                    "rationale": "Source-only annotation.",
                },
            ],
            "unresolved_decisions": [],
            "output": {
                "path": "datasets/paper/primary_pass/test_doc.json",
                "sha256": output_sha256,
            },
        },
        "blind_subset_selected": False,
        "blind_pass": None,
        "agreement_report": None,
        "adjudication": None,
        "output": {
            "path": "datasets/paper/production/test_doc.json",
            "sha256": output_sha256,
        },
        "frozen_at": "2026-07-13T09:31:00Z",
    }
    evidence_path = evidence_dir / "test_doc.json"
    evidence_path.write_text(json.dumps(evidence_log), encoding="utf-8")
    return gold_path, source_path, evidence_path


def _errors(messages: list[str]) -> list[str]:
    return [m for m in messages if m.startswith("ERROR")]


def _manifest_entry(
    doc_id: str,
    *,
    include: bool,
    role: str = "procedure_candidate",
    status: str = "candidate",
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "source_family": "test",
        "role": role,
        "status": status,
        "include_for_evaluation": include,
        "reason": "Test fixture.",
    }


def _write_manifest(
    path: Path,
    *,
    status: str,
    documents: list[dict[str, object]],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_status": status,
                "documents": documents,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_valid_annotation_passes(tmp_path: Path) -> None:
    msgs = validate_file(_write(tmp_path, VALID_ANNOTATION))
    assert _errors(msgs) == []


def test_rejects_unreviewed_status(tmp_path: Path) -> None:
    a = json.loads(json.dumps(VALID_ANNOTATION))
    a["quality"]["review_status"] = "unreviewed"
    msgs = validate_file(_write(tmp_path, a))
    assert any("review_status" in e for e in _errors(msgs))


def test_rejects_declared_annotation_schema_failure(tmp_path: Path) -> None:
    annotation = json.loads(json.dumps(VALID_ANNOTATION))
    del annotation["procedure"]["title"]

    messages = validate_file(_write(tmp_path, annotation))

    assert any("annotation schema" in error for error in _errors(messages))


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
    assert any("step:S1:" in m and "0 attached constraints" in m for m in msgs)


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


def test_human_verified_mode_rejects_agent_review(tmp_path: Path) -> None:
    annotation = json.loads(json.dumps(VALID_ANNOTATION))
    annotation["quality"]["annotator"] = (
        "model-assisted:qwen + agent-adjudicated (pending human sign-off)"
    )

    messages = validate_file(
        _write(tmp_path, annotation), require_human_verified=True
    )

    assert any("pending human sign-off" in error for error in _errors(messages))


def test_human_verified_mode_accepts_signed_annotation(tmp_path: Path) -> None:
    annotation = json.loads(json.dumps(VALID_ANNOTATION))
    annotation["quality"]["annotator"] = (
        "model-assisted:qwen + agent-adjudicated + human-verified:imad"
    )

    messages = validate_file(
        _write(tmp_path, annotation), require_human_verified=True
    )

    assert _errors(messages) == []


def test_production_evidence_mode_accepts_complete_package(tmp_path: Path) -> None:
    gold_path, source_path, evidence_path = _production_package(tmp_path)

    messages = validate_file(
        gold_path,
        require_production_evidence=True,
        source_path=source_path,
        evidence_path=evidence_path,
    )

    assert _errors(messages) == []


def test_production_evidence_mode_rejects_marker_without_sidecar(
    tmp_path: Path,
) -> None:
    annotation = json.loads(json.dumps(VALID_ANNOTATION))
    annotation["quality"]["annotator"] = "human-verified:P-001"
    gold_path = _write(tmp_path, annotation)
    source_path = tmp_path / "annot.txt"
    source_path.write_text("Do the thing.\n", encoding="utf-8")

    messages = validate_file(
        gold_path,
        require_production_evidence=True,
        source_path=source_path,
        evidence_path=tmp_path / "missing.json",
    )

    assert any("evidence log file missing" in error for error in _errors(messages))


def test_production_evidence_cli_rejects_insufficient_blind_coverage(
    tmp_path: Path,
    capsys,
) -> None:
    gold_path, source_path, evidence_path = _production_package(tmp_path)

    rc = main(
        [
            "--gold-dir",
            str(gold_path.parent),
            "--text-dir",
            str(source_path.parent),
            "--evidence-dir",
            str(evidence_path.parent),
            "--strict",
            "--require-production-evidence",
        ]
    )

    output = capsys.readouterr().out
    assert rc == 1
    assert "blind second-pass coverage is 0/1; at least 1/1 required" in output


def test_production_cli_does_not_require_excluded_candidate_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.validate_paper_gold.assess_corpus_evidence",
        lambda _logs: (),
    )
    gold_path, source_path, evidence_path = _production_package(tmp_path)
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        status="frozen",
        documents=[
            _manifest_entry("test_doc", include=True),
            _manifest_entry(
                "excluded_doc",
                include=False,
                status="excluded_pending_reannotation",
            ),
        ],
    )

    rc = main(
        [
            "--gold-dir",
            str(gold_path.parent),
            "--text-dir",
            str(source_path.parent),
            "--evidence-dir",
            str(evidence_path.parent),
            "--manifest",
            str(manifest_path),
            "--require-frozen-manifest",
            "--strict",
            "--require-production-evidence",
        ]
    )

    assert rc == 0


def test_manifest_scoped_cli_ignores_malformed_excluded_annotation(
    tmp_path: Path,
) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    included = json.loads(json.dumps(VALID_ANNOTATION))
    included["procedure"]["doc_id"] = "included"
    included["quality"]["annotator"] = (
        "model-assisted:qwen + agent-adjudicated + human-verified:imad"
    )
    (gold_dir / "included.json").write_text(
        json.dumps(included),
        encoding="utf-8",
    )
    (gold_dir / "excluded.json").write_text("{not-json", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        status="frozen",
        documents=[
            _manifest_entry("included", include=True),
            _manifest_entry(
                "excluded",
                include=False,
                status="excluded_pending_reannotation",
            ),
        ],
    )

    rc = main(
        [
            "--gold-dir",
            str(gold_dir),
            "--manifest",
            str(manifest_path),
            "--strict",
            "--require-frozen-manifest",
            "--require-human-verified",
        ]
    )

    assert rc == 0


def test_frozen_manifest_cli_rejects_malformed_included_annotation(
    tmp_path: Path,
    capsys,
) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "included.json").write_text("{not-json", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        status="frozen",
        documents=[_manifest_entry("included", include=True)],
    )

    rc = main(
        [
            "--gold-dir",
            str(gold_dir),
            "--manifest",
            str(manifest_path),
            "--require-frozen-manifest",
        ]
    )

    output = capsys.readouterr().out
    assert rc == 1, output
    assert "FAIL included.json" in output
    assert "ERROR: JSON parse error:" in output


def test_frozen_manifest_requirement_rejects_provisional_status(
    tmp_path: Path,
    capsys,
) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    included = json.loads(json.dumps(VALID_ANNOTATION))
    included["procedure"]["doc_id"] = "included"
    included["quality"]["annotator"] = (
        "model-assisted:qwen + agent-adjudicated + human-verified:imad"
    )
    (gold_dir / "included.json").write_text(
        json.dumps(included),
        encoding="utf-8",
    )
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        status="provisional",
        documents=[_manifest_entry("included", include=True)],
    )

    rc = main(
        [
            "--gold-dir",
            str(gold_dir),
            "--manifest",
            str(manifest_path),
            "--require-frozen-manifest",
            "--require-human-verified",
        ]
    )

    output = capsys.readouterr().out
    assert rc == 1
    assert "manifest is provisional" in output
    assert "PASS included.json" in output
