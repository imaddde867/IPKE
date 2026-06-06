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
    assert audit["annotation_status"] in {"draft", "reviewed"}


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
