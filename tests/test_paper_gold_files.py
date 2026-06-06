from __future__ import annotations

import csv
import json
from pathlib import Path

import jsonschema
import pytest

from scripts.normalize_gold_annotations import validate_annotation_links


MANIFEST = Path("datasets/paper/public_sources_manifest.csv")
GOLD_DIR = Path("datasets/paper/gold")
SECOND_DIR = Path("datasets/paper/second_pass")
LOOSE_SCHEMA = Path("schemas/ipke_annotation.schema.json")
STRICT_SCHEMA = Path("schemas/ipke_paper_tiera.schema.json")
MIN_DENSITY = 2


def _selected_gold_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["selected_for_gold"].strip().lower() == "true"]


@pytest.mark.parametrize("row", _selected_gold_rows(), ids=lambda r: r["document_id"])
def test_gold_file_present_and_validates_under_loose_schema(row: dict[str, str]) -> None:
    path = GOLD_DIR / f"{row['document_id']}.json"
    assert path.exists(), f"missing gold file for {row['document_id']}"
    schema = json.loads(LOOSE_SCHEMA.read_text(encoding="utf-8"))
    annotation = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(annotation)
    validate_annotation_links(annotation, row["document_id"])


@pytest.mark.parametrize("row", _selected_gold_rows(), ids=lambda r: r["document_id"])
def test_gold_file_has_minimum_annotation_density(row: dict[str, str]) -> None:
    path = GOLD_DIR / f"{row['document_id']}.json"
    annotation = json.loads(path.read_text(encoding="utf-8"))
    assert len(annotation["steps"]) >= MIN_DENSITY, f"{row['document_id']} has too few steps"


@pytest.mark.parametrize("row", _selected_gold_rows(), ids=lambda r: r["document_id"])
def test_gold_file_doc_id_matches_manifest(row: dict[str, str]) -> None:
    path = GOLD_DIR / f"{row['document_id']}.json"
    annotation = json.loads(path.read_text(encoding="utf-8"))
    assert annotation["procedure"]["doc_id"] == row["document_id"]


def test_second_pass_files_match_manifest_double_annotated_status() -> None:
    second_pass_doc_ids = {p.stem for p in SECOND_DIR.glob("*.json")}
    assert second_pass_doc_ids, "expected at least one second_pass file"
    for doc_id in second_pass_doc_ids:
        row = next((r for r in _selected_gold_rows() if r["document_id"] == doc_id), None)
        assert row is not None, f"{doc_id} is a second_pass file but not in manifest"
        assert row["annotation_status"] == "double_annotated", (
            f"{doc_id} has a second_pass file but manifest says {row['annotation_status']}"
        )


def test_strict_schema_validates_gold_files() -> None:
    schema = json.loads(STRICT_SCHEMA.read_text(encoding="utf-8"))
    for path in sorted(GOLD_DIR.glob("*.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(annotation)
