from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest


SCHEMA_PATH = Path("schemas/ipke_annotation_evidence.schema.json")


def _valid_log() -> dict:
    output_sha256 = hashlib.sha256(b"annotation").hexdigest()
    return {
        "schema_version": 1,
        "protocol_id": "ipke-human-evidence-v1",
        "status": "frozen",
        "doc_id": "sample_procedure",
        "source": {
            "path": "datasets/paper/text/sample_procedure.txt",
            "sha256": hashlib.sha256(b"source").hexdigest(),
            "span_sha256": hashlib.sha256(b"source").hexdigest(),
            "char_start": 0,
            "char_end": 6,
        },
        "candidate": None,
        "primary_pass": {
            "reviewer": {
                "handle": "reviewer_a",
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
                    "evidence_spans": [{"char_start": 0, "char_end": 6}],
                    "rationale": "Source-only annotation.",
                },
                {
                    "decision_id": "D-C1",
                    "item_kind": "constraint",
                    "action": "add",
                    "input_ids": [],
                    "output_ids": ["C1"],
                    "evidence_spans": [{"char_start": 0, "char_end": 6}],
                    "rationale": "Source-only annotation.",
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


def test_annotation_evidence_schema_is_valid() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)


def test_annotation_evidence_schema_accepts_minimal_frozen_primary_pass() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(_valid_log())


def test_candidate_assistance_requires_candidate_artifact() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    evidence_log = copy.deepcopy(_valid_log())
    evidence_log["primary_pass"]["assistance"]["candidate_used"] = True

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(evidence_log)


def test_blind_subset_requires_blind_agreement_and_adjudication_artifacts() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    evidence_log = copy.deepcopy(_valid_log())
    evidence_log["blind_subset_selected"] = True

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(evidence_log)


def test_artifact_paths_are_repository_relative_dataset_paths() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    evidence_log = copy.deepcopy(_valid_log())
    evidence_log["output"]["path"] = "production/sample_procedure.json"

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(evidence_log)


def test_artifact_paths_are_bound_to_their_roles() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    evidence_log = copy.deepcopy(_valid_log())
    evidence_log["primary_pass"]["output"]["path"] = (
        "datasets/paper/production/sample_procedure.json"
    )

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(evidence_log)


def test_candidate_path_must_use_review_candidate_directory() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    evidence_log = copy.deepcopy(_valid_log())
    evidence_log["primary_pass"]["assistance"]["candidate_used"] = True
    evidence_log["candidate"] = {
        "path": "datasets/paper/gold/sample_procedure.json",
        "sha256": hashlib.sha256(b"candidate").hexdigest(),
    }

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(evidence_log)
