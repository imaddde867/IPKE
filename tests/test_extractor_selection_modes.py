from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from scripts.extract_public_documents import (
    extract_public_documents,
    selected_rows,
    SELECTION_MODES,
)


MANIFEST = Path("datasets/paper/public_sources_manifest.csv")
RAW_DIR = Path("/tmp/extract-test-raw")
OUT_DIR = Path("/tmp/extract-test-out")


MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000056 00000 n \n0000000109 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\n"
    b"startxref\n171\n%%EOF\n"
)


@pytest.fixture(autouse=True)
def reset_dirs() -> None:
    for d in (RAW_DIR, OUT_DIR):
        if d.exists():
            shutil.rmtree(d)
    RAW_DIR.mkdir(parents=True)
    for name in (
        "usgs_nfm_collection_water_samples_a4.pdf",
        "faa_amt_general_handbook_2023.pdf",
    ):
        (RAW_DIR / name).write_bytes(MINIMAL_PDF)
    yield


def test_selection_modes_are_documented() -> None:
    assert SELECTION_MODES == ("gold", "download", "all")


def test_gold_mode_skips_alternates(tmp_path: Path) -> None:
    out = tmp_path / "out"
    written = extract_public_documents(MANIFEST, RAW_DIR, out, selection="gold")
    assert "usgs_nfm_collection_water_samples_a4.txt" in written
    assert "faa_amt_general_handbook_2023.txt" not in written


def test_download_mode_writes_all_downloaded(tmp_path: Path) -> None:
    out = tmp_path / "out"
    written = extract_public_documents(MANIFEST, RAW_DIR, out, selection="download")
    assert "usgs_nfm_collection_water_samples_a4.txt" in written


def test_selected_rows_isolates_by_mode() -> None:
    import csv
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    gold_rows = list(selected_rows(rows, "gold"))
    download_rows = list(selected_rows(rows, "download"))
    all_rows = list(selected_rows(rows, "all"))
    assert all(r["selected_for_gold"] == "true" for r in gold_rows)
    assert all(r["selected_for_download"] == "true" for r in download_rows)
    assert len(download_rows) >= len(gold_rows)
    assert len(all_rows) == len(rows)
