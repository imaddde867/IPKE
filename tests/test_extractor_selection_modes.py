from __future__ import annotations

import shutil
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


@pytest.fixture(autouse=True)
def reset_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    for d in (RAW_DIR, OUT_DIR):
        if d.exists():
            shutil.rmtree(d)
    RAW_DIR.mkdir(parents=True)
    for name in (
        "usgs_nfm_collection_water_samples_a4.pdf",
        "faa_amt_general_handbook_2023.pdf",
    ):
        (RAW_DIR / name).write_text("stub text\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.extract_public_documents.read_source_text",
        lambda _path: "stub text for selection test\n",
    )
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
