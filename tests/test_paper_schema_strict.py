from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from scripts.normalize_gold_annotations import validate_annotation_links


SCHEMA = Path("schemas/ipke_paper_tiera.schema.json")
GOLD_DIR = Path("datasets/paper/gold")


def test_strict_schema_is_valid() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("path", sorted(GOLD_DIR.glob("*.json")))
def test_pilot_gold_validates_under_strict_schema(path: Path) -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    annotation = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(annotation)
    validate_annotation_links(annotation, path.stem)


def test_strict_schema_rejects_orphan_attachment() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    annotation = json.loads((GOLD_DIR / "olsk_small_cnc_v1_workbook.json").read_text(encoding="utf-8"))
    annotation["steps"][0]["constraints"][0]["attached_to"] = ["S999"]
    with pytest.raises((jsonschema.ValidationError, ValueError)):
        jsonschema.Draft202012Validator(schema).validate(annotation)
        validate_annotation_links(annotation, "olsk_small_cnc_v1_workbook")
