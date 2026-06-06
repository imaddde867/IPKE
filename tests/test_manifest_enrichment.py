from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from scripts.enrich_manifest import (
    compute_text_metrics,
    merge_metrics_into_manifest,
    REQUIRED_COLUMNS,
)


MANIFEST = Path("datasets/paper/public_sources_manifest.csv")
TEXT_DIR = Path("datasets/paper/text")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_manifest_has_required_columns() -> None:
    header = MANIFEST.read_text(encoding="utf-8").splitlines()[0].split(",")
    for column in REQUIRED_COLUMNS:
        assert column in header, f"missing column {column}"


@pytest.mark.parametrize(
    "filename,expected_words",
    [
        ("epa_guidance_preparing_sops_qag6.txt", 1000),
    ],
)
def test_compute_text_metrics_skips_metadata_header(
    filename: str, expected_words: int
) -> None:
    metrics = compute_text_metrics(TEXT_DIR / filename)
    assert metrics["word_count"] >= expected_words
    assert metrics["word_count"] >= metrics["token_count"]
    assert "excerpt_word_count" in metrics


def test_merge_metrics_writes_positive_counts_for_text_rows() -> None:
    rows = _rows()
    enriched = merge_metrics_into_manifest(rows, TEXT_DIR)
    text_rows = [r for r in enriched if r["selected_for_gold"].strip().lower() == "true"]
    assert text_rows, "no selected_for_gold rows in manifest"
    for row in text_rows:
        assert int(row["word_count"]) > 0, row["document_id"]
        assert int(row["token_count"]) > 0, row["document_id"]
        assert SHA256_RE.fullmatch(row["sha256"])


def test_merge_metrics_writes_zero_counts_for_rows_without_text() -> None:
    rows = _rows()
    enriched = merge_metrics_into_manifest(rows, TEXT_DIR)
    no_text_rows = [r for r in enriched if not (TEXT_DIR / f"{r['document_id']}.txt").exists()]
    assert no_text_rows, "expected at least one manifest row with no text file"
    for row in no_text_rows:
        assert int(row["word_count"]) == 0, row["document_id"]
        assert int(row["token_count"]) == 0, row["document_id"]
        assert int(row["excerpt_word_count"]) == 0, row["document_id"]
        assert int(row["excerpt_token_count"]) == 0, row["document_id"]


def test_enrichment_preserves_existing_status_columns() -> None:
    rows = _rows()
    enriched = merge_metrics_into_manifest(rows, TEXT_DIR)
    by_id = {row["document_id"]: row for row in enriched}
    epa = by_id["epa_field_sampling_measurement_procedure_validation"]
    assert epa["annotation_status"] == "double_annotated", (
        "enrichment must not overwrite the manifest's annotation_status"
    )
    assert epa["annotator_count"] == "2", (
        "enrichment must not overwrite the manifest's annotator_count"
    )
    usgs = by_id["usgs_nfm_collection_water_samples_a4"]
    assert usgs["annotation_status"] == "reviewed"
    assert usgs["annotator_count"] == "1"
