from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
INVENTORY = REPO / "datasets/paper/INVENTORY.md"
GOLD_DIR = REPO / "datasets/paper/gold"
SECOND_DIR = REPO / "datasets/paper/second_pass"
TEXT_DIR = REPO / "datasets/paper/text"
FAMILIES_FILE = REPO / "datasets/paper/annotation_batches/manifest_pilot_status.json"


def test_inventory_exists_and_is_substantial() -> None:
    assert INVENTORY.exists()
    text = INVENTORY.read_text(encoding="utf-8")
    assert len(text) > 1000, "inventory should not be a stub"


def test_inventory_anchors_eight_tier_a_documents() -> None:
    gold_ids = sorted(p.stem for p in GOLD_DIR.glob("*.json"))
    assert len(gold_ids) == 8
    text = INVENTORY.read_text(encoding="utf-8")
    for doc_id in gold_ids:
        assert doc_id in text, f"inventory missing {doc_id}"


def test_inventory_lists_six_source_families() -> None:
    text = INVENTORY.read_text(encoding="utf-8")
    for family in ("USGS", "FAA", "EPA", "CDC_NIOSH", "NASA", "OLSK"):
        assert family in text, f"inventory missing family {family}"


def test_inventory_references_double_annotated_set() -> None:
    text = INVENTORY.read_text(encoding="utf-8")
    for doc_id in (
        "epa_field_sampling_measurement_procedure_validation",
        "niosh_nmam_5th_edition_ebook",
        "olsk_small_cnc_v1_workbook",
    ):
        assert doc_id in text, f"inventory missing double-annotated {doc_id}"


def test_inventory_links_to_manifest_and_iaa_report() -> None:
    text = INVENTORY.read_text(encoding="utf-8")
    assert "public_sources_manifest.csv" in text
    assert "issue_53_iaa_report.json" in text
