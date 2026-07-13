from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

from src.evaluation.evidence import assess_production_evidence


SOURCE_TEXT = "Préparer l’échantillon.\nWear gloves.\n"
STEP_TEXT = "Préparer l’échantillon."
CONSTRAINT_TEXT = "Wear gloves."


def _annotation() -> dict:
    return {
        "procedure": {
            "doc_id": "sample_procedure",
            "title": "Sample procedure",
            "source": {
                "doc_id": "sample_procedure",
                "section": "Procedure",
                "char_start": 0,
                "char_end": len(SOURCE_TEXT),
            },
        },
        "steps": [
            {
                "id": "S1",
                "label": "Préparer l’échantillon",
                "provenance": {
                    "doc_id": "sample_procedure",
                    "char_start": 0,
                    "char_end": len(STEP_TEXT),
                },
                "constraints": [
                    {
                        "id": "C1",
                        "type": "precondition",
                        "enforcement": "must",
                        "text": CONSTRAINT_TEXT,
                        "attached_to": ["S1"],
                        "provenance": {
                            "doc_id": "sample_procedure",
                            "char_start": SOURCE_TEXT.index(CONSTRAINT_TEXT),
                            "char_end": (
                                SOURCE_TEXT.index(CONSTRAINT_TEXT)
                                + len(CONSTRAINT_TEXT)
                            ),
                        },
                    }
                ],
            }
        ],
        "constraints": [],
        "relations": [],
        "quality": {
            "review_status": "reviewed",
            "annotator": "P-001",
            "review_date": "2026-07-13",
        },
    }


def _encoded(annotation: dict) -> bytes:
    return json.dumps(annotation, indent=2).encode("utf-8")


def _evidence_log(annotation_bytes: bytes) -> dict:
    source_bytes = SOURCE_TEXT.encode("utf-8")
    output_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    return {
        "schema_version": 1,
        "protocol_id": "ipke-human-evidence-v1",
        "status": "frozen",
        "doc_id": "sample_procedure",
        "source": {
            "path": "datasets/paper/text/sample_procedure.txt",
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "span_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "char_start": 0,
            "char_end": len(SOURCE_TEXT),
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
            },
            "decisions": [
                {
                    "decision_id": "D-S1",
                    "item_kind": "step",
                    "action": "add",
                    "input_ids": [],
                    "output_ids": ["S1"],
                    "evidence_spans": [
                        {"char_start": 0, "char_end": len(STEP_TEXT)}
                    ],
                    "rationale": "Source-only annotation with no candidate.",
                },
                {
                    "decision_id": "D-C1",
                    "item_kind": "constraint",
                    "action": "add",
                    "input_ids": [],
                    "output_ids": ["C1"],
                    "evidence_spans": [
                        {
                            "char_start": SOURCE_TEXT.index(CONSTRAINT_TEXT),
                            "char_end": (
                                SOURCE_TEXT.index(CONSTRAINT_TEXT)
                                + len(CONSTRAINT_TEXT)
                            ),
                        }
                    ],
                    "rationale": "Source-only annotation with no candidate.",
                },
            ],
            "unresolved_decisions": [],
            "output": {
                "path": "datasets/paper/primary_pass/sample_procedure.json",
                "sha256": output_sha256,
            },
        },
        "blind_subset_selected": False,
        "blind_pass": None,
        "agreement_report": None,
        "adjudication": None,
        "output": {
            "path": "datasets/paper/production/sample_procedure.json",
            "sha256": output_sha256,
        },
        "frozen_at": "2026-07-13T09:31:00Z",
    }


def _artifact_loader(
    annotation_bytes: bytes,
    extra: dict[str, bytes] | None = None,
) -> Callable[[str], bytes]:
    artifacts = {
        "datasets/paper/primary_pass/sample_procedure.json": annotation_bytes,
        **(extra or {}),
    }

    def load(path: str) -> bytes:
        return artifacts[path]

    return load


def _blind_evidence_log(annotation_bytes: bytes) -> tuple[dict, dict[str, bytes]]:
    evidence_log = _evidence_log(annotation_bytes)
    report_bytes = b'{"attachment_f1": 0.8}'
    evidence_log.update(
        {
            "blind_subset_selected": True,
            "blind_pass": {
                "reviewer": {
                    "handle": "P-002",
                    "kind": "human",
                    "role": "blind_annotator",
                },
                "started_at": "2026-07-13T09:00:00Z",
                "completed_at": "2026-07-13T09:25:00Z",
                "active_minutes": 25,
                "source_only": True,
                "candidate_seen": False,
                "primary_seen": False,
                "breach_detected": False,
                "output": {
                    "path": "datasets/paper/second_pass/sample_procedure.json",
                    "sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                },
            },
            "agreement_report": {
                "path": "datasets/paper/reports/sample_procedure_agreement.json",
                "sha256": hashlib.sha256(report_bytes).hexdigest(),
            },
            "adjudication": {
                "reviewer": {
                    "handle": "P-003",
                    "kind": "human",
                    "role": "adjudicator",
                },
                "started_at": "2026-07-13T10:00:00Z",
                "completed_at": "2026-07-13T10:10:00Z",
                "active_minutes": 10,
                "unresolved_decisions": [],
                "output": {
                    "path": "datasets/paper/production/sample_procedure.json",
                    "sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                },
            },
            "frozen_at": "2026-07-13T10:11:00Z",
        }
    )
    return evidence_log, {
        "datasets/paper/second_pass/sample_procedure.json": annotation_bytes,
        "datasets/paper/reports/sample_procedure_agreement.json": report_bytes,
        "datasets/paper/production/sample_procedure.json": annotation_bytes,
    }


def test_production_evidence_accepts_frozen_human_pass_with_exact_anchors() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=_evidence_log(annotation_bytes),
        expected_doc_id="sample_procedure",
        artifact_loader=_artifact_loader(annotation_bytes),
    )

    assert result.evidence_eligible is True
    assert result.human_pass_recorded is True
    assert result.anchors_complete is True
    assert result.hashes_match is True
    assert result.issues == ()
    assert len(SOURCE_TEXT.encode("utf-8")) > len(SOURCE_TEXT)


def test_production_evidence_rejects_marker_without_sidecar() -> None:
    annotation = _annotation()
    annotation["quality"]["annotator"] = (
        "model-assisted:qwen + human-verified:P-001"
    )
    annotation_bytes = _encoded(annotation)

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=None,
    )

    assert result.evidence_eligible is False
    assert "production evidence log missing" in result.issues


def test_production_evidence_rejects_missing_item_anchor() -> None:
    annotation = _annotation()
    del annotation["steps"][0]["provenance"]
    annotation_bytes = _encoded(annotation)

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=_evidence_log(annotation_bytes),
    )

    assert result.evidence_eligible is False
    assert "step:S1 provenance missing" in result.issues


def test_production_evidence_rejects_source_hash_mismatch() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)
    evidence_log = _evidence_log(annotation_bytes)
    evidence_log["source"]["sha256"] = "0" * 64

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
    )

    assert result.evidence_eligible is False
    assert "source SHA-256 does not match evidence log" in result.issues


def test_production_evidence_rejects_output_hash_mismatch() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)
    evidence_log = _evidence_log(annotation_bytes)
    evidence_log["output"]["sha256"] = "0" * 64

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
    )

    assert "annotation SHA-256 does not match evidence log" in result.issues


def test_production_evidence_rejects_missing_embedded_constraint_anchor() -> None:
    annotation = _annotation()
    del annotation["steps"][0]["constraints"][0]["provenance"]
    annotation_bytes = _encoded(annotation)

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=_evidence_log(annotation_bytes),
    )

    assert "step:S1/constraint:C1 provenance missing" in result.issues


def test_production_evidence_rejects_doc_id_mismatch() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)
    evidence_log = _evidence_log(annotation_bytes)
    evidence_log["doc_id"] = "other_procedure"

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
    )

    assert "evidence log doc_id does not match annotation" in result.issues


def test_production_evidence_rejects_unresolved_primary_decisions() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)
    evidence_log = _evidence_log(annotation_bytes)
    evidence_log["primary_pass"]["unresolved_decisions"] = [
        {
            "decision_id": "U-1",
            "category": "taxonomy",
            "question": "Is this a guard or precondition?",
            "evidence_spans": [{"char_start": 0, "char_end": len(STEP_TEXT)}],
        }
    ]

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
    )

    assert "primary pass has unresolved decisions" in result.issues


def test_production_evidence_rejects_decision_count_drift() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)
    evidence_log = _evidence_log(annotation_bytes)
    evidence_log["primary_pass"]["decision_counts"]["steps"]["added"] = 2

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
    )

    assert "primary step decision counts do not match decision records" in result.issues


def test_production_evidence_rejects_annotation_schema_failure() -> None:
    annotation = _annotation()
    del annotation["procedure"]["title"]
    annotation_bytes = _encoded(annotation)

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=_evidence_log(annotation_bytes),
    )

    assert any("annotation schema" in issue for issue in result.issues)


def test_production_evidence_rejects_filename_identity_mismatch() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=_evidence_log(annotation_bytes),
        expected_doc_id="other_filename",
    )

    assert "annotation doc_id does not match selected filename" in result.issues


def test_production_evidence_rejects_duplicate_item_ids() -> None:
    annotation = _annotation()
    duplicate = json.loads(json.dumps(annotation["steps"][0]))
    annotation["steps"].append(duplicate)
    annotation_bytes = _encoded(annotation)

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=_evidence_log(annotation_bytes),
    )

    assert "duplicate step IDs: S1" in result.issues
    assert "duplicate constraint IDs: C1" in result.issues


def test_production_evidence_rejects_unverified_primary_artifact() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=_evidence_log(annotation_bytes),
        expected_doc_id="sample_procedure",
        artifact_loader=lambda _: (_ for _ in ()).throw(FileNotFoundError()),
    )

    assert any("primary output artifact cannot be read" in issue for issue in result.issues)


def test_production_evidence_rejects_primary_artifact_hash_mismatch() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=_evidence_log(annotation_bytes),
        expected_doc_id="sample_procedure",
        artifact_loader=_artifact_loader(b"different annotation"),
    )

    assert "primary output artifact SHA-256 does not match evidence log" in result.issues


def test_production_evidence_rejects_unanchored_primary_artifact() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)
    primary = _annotation()
    del primary["steps"][0]["provenance"]
    primary_bytes = _encoded(primary)
    evidence_log = _evidence_log(annotation_bytes)
    evidence_log["primary_pass"]["output"]["sha256"] = hashlib.sha256(
        primary_bytes
    ).hexdigest()

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
        expected_doc_id="sample_procedure",
        artifact_loader=_artifact_loader(primary_bytes),
    )

    assert "primary output step:S1 provenance missing" in result.issues


def test_production_evidence_rejects_invalid_and_reversed_timestamps() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)
    evidence_log = _evidence_log(annotation_bytes)
    evidence_log["primary_pass"]["started_at"] = "not-a-date"
    evidence_log["frozen_at"] = "2026-07-13T08:00:00Z"

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
    )

    assert any("started_at" in issue and "date-time" in issue for issue in result.issues)
    assert "evidence frozen_at precedes primary completion" in result.issues


def test_production_evidence_rejects_timezone_free_timestamp() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)
    evidence_log = _evidence_log(annotation_bytes)
    evidence_log["primary_pass"]["started_at"] = "2026-07-13T09:00:00"

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
    )

    assert any("started_at" in issue and "date-time" in issue for issue in result.issues)


def test_production_evidence_rejects_out_of_bounds_decision_span() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)
    evidence_log = _evidence_log(annotation_bytes)
    evidence_log["primary_pass"]["decisions"][0]["evidence_spans"] = [
        {"char_start": 9999, "char_end": 10000}
    ]

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
    )

    assert "primary decision D-S1 evidence span is outside source text" in result.issues


def test_production_evidence_rejects_duplicate_decision_ids() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)
    evidence_log = _evidence_log(annotation_bytes)
    evidence_log["primary_pass"]["decisions"][1]["decision_id"] = "D-S1"

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
    )

    assert "duplicate primary decision IDs: D-S1" in result.issues


def test_production_evidence_verifies_candidate_and_decision_input_ids() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)
    candidate_bytes = _encoded(annotation)
    evidence_log = _evidence_log(annotation_bytes)
    evidence_log["candidate"] = {
        "path": "datasets/paper/review_candidates/sample_procedure.json",
        "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
    }
    evidence_log["primary_pass"]["assistance"]["candidate_used"] = True
    for counts in evidence_log["primary_pass"]["decision_counts"].values():
        counts.update(
            {
                "candidate_count": 1,
                "accepted": 1,
                "added": 0,
            }
        )
    for decision in evidence_log["primary_pass"]["decisions"]:
        decision["action"] = "accept"
        decision["input_ids"] = list(decision["output_ids"])

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
        expected_doc_id="sample_procedure",
        artifact_loader=_artifact_loader(
            annotation_bytes,
            {
                "datasets/paper/review_candidates/sample_procedure.json": (
                    candidate_bytes
                )
            },
        ),
    )

    assert result.evidence_eligible is True

    evidence_log["primary_pass"]["decisions"][0]["input_ids"] = ["S99"]
    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
        expected_doc_id="sample_procedure",
        artifact_loader=_artifact_loader(
            annotation_bytes,
            {
                "datasets/paper/review_candidates/sample_procedure.json": (
                    candidate_bytes
                )
            },
        ),
    )

    assert "primary step candidate decisions are incomplete or duplicated" in result.issues


def test_production_evidence_rejects_dangling_links() -> None:
    annotation = _annotation()
    annotation["steps"][0]["constraints"][0]["attached_to"] = ["S99"]
    annotation["relations"] = [{"source": "S1", "target": "S99", "type": "NEXT"}]
    annotation_bytes = _encoded(annotation)

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=_evidence_log(annotation_bytes),
    )

    assert any("constraint:C1 references unknown step S99" in issue for issue in result.issues)
    assert any("target references unknown step S99" in issue for issue in result.issues)


def test_production_evidence_rejects_missing_attachment_and_malformed_relation() -> None:
    annotation = _annotation()
    del annotation["steps"][0]["constraints"][0]["attached_to"]
    annotation["relations"] = [{"source": "S1", "type": "NEXT"}]
    annotation_bytes = _encoded(annotation)

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=_evidence_log(annotation_bytes),
    )

    assert "step:S1/constraint:C1 has no step attachment" in result.issues
    assert "relation:0 requires source and target step endpoints" in result.issues


def test_production_evidence_verifies_complete_blind_artifact_chain() -> None:
    annotation = _annotation()
    annotation_bytes = _encoded(annotation)
    evidence_log, artifacts = _blind_evidence_log(annotation_bytes)

    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
        expected_doc_id="sample_procedure",
        artifact_loader=_artifact_loader(annotation_bytes, artifacts),
    )

    assert result.evidence_eligible is True

    evidence_log["agreement_report"]["sha256"] = "0" * 64
    result = assess_production_evidence(
        annotation,
        annotation_bytes=annotation_bytes,
        source_bytes=SOURCE_TEXT.encode("utf-8"),
        evidence_log=evidence_log,
        expected_doc_id="sample_procedure",
        artifact_loader=_artifact_loader(annotation_bytes, artifacts),
    )

    assert "agreement report artifact SHA-256 does not match evidence log" in result.issues
