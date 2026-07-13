from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from scripts.normalize_gold_annotations import validate_annotation_links


SCHEMA_PATH = Path("schemas/ipke_annotation.schema.json")
FIXTURE_DIR = Path("tests/fixtures/annotations")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_annotation_schema_is_valid() -> None:
    schema = _load_json(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(
    "fixture",
    [
        "annotator_a_sample.json",
        "annotator_b_sample.json",
    ],
)
def test_annotation_fixtures_validate_against_schema(fixture: str) -> None:
    schema = _load_json(SCHEMA_PATH)
    annotation = _load_json(FIXTURE_DIR / fixture)

    jsonschema.Draft202012Validator(schema).validate(annotation)
    validate_annotation_links(annotation, "sample_procedure")


def test_annotation_link_helper_rejects_missing_attachment_endpoint() -> None:
    annotation = _load_json(FIXTURE_DIR / "annotator_a_sample.json")
    invalid = copy.deepcopy(annotation)
    invalid["constraints"][0]["applies_to"] = ["S999"]

    with pytest.raises(ValueError, match="unknown step"):
        validate_annotation_links(invalid, "sample_procedure")


def test_annotation_link_helper_rejects_missing_relation_endpoint() -> None:
    annotation = _load_json(FIXTURE_DIR / "annotator_a_sample.json")
    invalid = copy.deepcopy(annotation)
    invalid["relations"] = [{"from": "S1", "to": "S999"}]

    with pytest.raises(ValueError, match="unknown step"):
        validate_annotation_links(invalid, "sample_procedure")


def test_all_retained_candidates_validate_against_candidate_schema() -> None:
    schema = _load_json(SCHEMA_PATH)
    validator = jsonschema.Draft202012Validator(schema)
    failures: dict[str, list[str]] = {}

    for path in sorted(Path("datasets/paper/gold").glob("*.json")):
        errors = sorted(validator.iter_errors(_load_json(path)), key=lambda e: list(e.path))
        if errors:
            failures[path.name] = [error.message for error in errors]

    assert failures == {}


def test_embedded_constraint_uses_constraint_schema() -> None:
    schema = _load_json(SCHEMA_PATH)
    annotation = _load_json(FIXTURE_DIR / "annotator_a_sample.json")
    invalid = copy.deepcopy(annotation)
    constraint = copy.deepcopy(invalid["constraints"][0])
    constraint["enforcement"] = "never"
    invalid["steps"][0]["constraints"] = [constraint]

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(invalid)
