from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SECOND_PASS = REPO / "datasets/paper/second_pass/niosh_nmam_5th_edition_ebook.json"
MANIFEST = REPO / "datasets/paper/public_sources_manifest.csv"
PILOT_STATUS = REPO / "datasets/paper/annotation_batches/manifest_pilot_status.json"


def test_niosh_second_pass_file_exists() -> None:
    assert SECOND_PASS.exists(), f"missing {SECOND_PASS.name}"
    data = json.loads(SECOND_PASS.read_text(encoding="utf-8"))
    assert data["procedure"]["doc_id"] == "niosh_nmam_5th_edition_ebook"
    audit = data["procedure"]["audit"]
    assert audit["gold_status"] == "pilot_gold"
    assert audit["annotation_status"] == "double_annotated"
    assert audit["annotator_count"] >= 2
    assert len(data["steps"]) >= 3
    assert isinstance(data.get("constraints", []), list)


def test_manifest_marks_niosh_as_double_annotated() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    niosh = next(r for r in rows if r["document_id"] == "niosh_nmam_5th_edition_ebook")
    assert niosh["annotation_status"] == "double_annotated"


def test_pilot_status_lists_three_double_annotated_docs() -> None:
    status = json.loads(PILOT_STATUS.read_text(encoding="utf-8"))
    double_annotated = {row["document_id"] for row in status if row.get("annotation_status") == "double_annotated"}
    expected = {
        "epa_field_sampling_measurement_procedure_validation",
        "niosh_nmam_5th_edition_ebook",
        "olsk_small_cnc_v1_workbook",
    }
    assert expected.issubset(double_annotated)
