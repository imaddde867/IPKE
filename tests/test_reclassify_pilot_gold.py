from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reclassify_pilot_gold import (
    reclassify_annotation,
    update_manifest_columns,
    REQUIRED_AUDIT_KEYS,
)


GOLD_DIR = Path("datasets/paper/gold")
SECOND_DIR = Path("datasets/paper/second_pass")
MANIFEST = Path("datasets/paper/public_sources_manifest.csv")


@pytest.mark.parametrize("path", sorted(GOLD_DIR.glob("*.json")))
def test_every_gold_file_has_pilot_audit_block(path: Path) -> None:
    annotation = json.loads(path.read_text(encoding="utf-8"))
    audit = annotation.get("procedure", {}).get("audit")
    assert isinstance(audit, dict)
    for key in REQUIRED_AUDIT_KEYS:
        assert key in audit, f"{path.name}: missing audit.{key}"
    assert audit["gold_status"] == "pilot_gold"
    assert audit["annotation_status"] in {"draft", "reviewed", "double_annotated"}


@pytest.mark.parametrize("path", sorted(SECOND_DIR.glob("*.json")))
def test_every_second_pass_file_has_pilot_audit_block(path: Path) -> None:
    annotation = json.loads(path.read_text(encoding="utf-8"))
    audit = annotation.get("procedure", {}).get("audit")
    assert isinstance(audit, dict)
    for key in REQUIRED_AUDIT_KEYS:
        assert key in audit, f"{path.name}: missing audit.{key}"
    assert audit["gold_status"] == "pilot_gold"


def test_manifest_columns_contain_pilot_status() -> None:
    rows = list(MANIFEST.read_text(encoding="utf-8").splitlines())
    header = rows[0].split(",")
    assert "gold_status" in header
    assert "annotation_status" in header


# The 3-document ID set asserted below is a contract for the current pilot state.
# A 4th `double_annotated` document will require an explicit update to this set
# and to the test_audit_block_mirrors_manifest_annotator_count_and_scope coverage.
def test_double_annotated_gold_file_mirrors_manifest_status() -> None:
    rows = json.loads(
        Path("datasets/paper/annotation_batches/manifest_pilot_status.json").read_text(encoding="utf-8")
    )
    status_by_id = {row["document_id"]: row["annotation_status"] for row in rows}
    double_annotated_ids = {doc_id for doc_id, status in status_by_id.items() if status == "double_annotated"}
    assert double_annotated_ids == {
        "epa_field_sampling_measurement_procedure_validation",
        "niosh_nmam_5th_edition_ebook",
        "olsk_small_cnc_v1_workbook",
    }
    for doc_id in double_annotated_ids:
        annotation = json.loads((GOLD_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
        assert annotation["procedure"]["audit"]["annotation_status"] == "double_annotated", doc_id
    for path in sorted(SECOND_DIR.glob("*.json")):
        second = json.loads(path.read_text(encoding="utf-8"))
        assert second["procedure"]["audit"]["annotation_status"] == "double_annotated", path.name


def test_audit_block_mirrors_manifest_annotator_count_and_scope() -> None:
    rows = json.loads(
        Path("datasets/paper/annotation_batches/manifest_pilot_status.json").read_text(encoding="utf-8")
    )
    rows_by_id = {row["document_id"]: row for row in rows}
    for path in sorted(GOLD_DIR.glob("*.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        audit = annotation["procedure"]["audit"]
        row = rows_by_id[path.stem]
        assert audit["annotator_count"] == int(row["annotator_count"]), path.name
        assert audit["annotation_scope"] == row["annotation_scope"], path.name
