from __future__ import annotations

import csv
import re
from pathlib import Path


MANIFEST = Path("datasets/paper/public_sources_manifest.csv")
EXPECTED_HEADER = [
    "document_id",
    "source_family",
    "title",
    "domain",
    "document_role",
    "direct_url",
    "local_filename",
    "sha256",
    "size_bytes",
    "license_or_usage",
    "usage_notes",
    "selected_for_gold",
    "selected_for_download",
    "risk_level",
    "gold_status",
    "annotation_status",
    "annotation_scope",
    "annotator_count",
]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _rows() -> tuple[list[str], list[dict[str, str]]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_public_source_manifest_contract() -> None:
    header, rows = _rows()
    assert header == EXPECTED_HEADER
    assert rows

    document_ids = [row["document_id"] for row in rows]
    assert len(document_ids) == len(set(document_ids))

    selected_gold = 0
    for row in rows:
        assert row["direct_url"].startswith("https://")
        assert row["license_or_usage"].strip()
        assert row["usage_notes"].strip()
        assert "ifixit" not in row["direct_url"].lower()
        assert "myfixit" not in row["direct_url"].lower()
        assert "ifixit" not in row["source_family"].lower()
        assert "myfixit" not in row["source_family"].lower()
        assert SHA256_RE.fullmatch(row["sha256"])
        assert int(row["size_bytes"]) > 0
        if row["selected_for_gold"].strip().lower() == "true":
            selected_gold += 1

    assert selected_gold >= 8
