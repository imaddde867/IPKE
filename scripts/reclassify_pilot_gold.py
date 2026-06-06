"""Reclassify committed paper gold annotations to pilot_gold and stamp audit metadata."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_AUDIT_KEYS: tuple[str, ...] = (
    "gold_status",
    "annotation_status",
    "annotation_scope",
    "annotator_count",
    "guideline_version",
    "model_draft_used",
    "reviewed_by",
    "reviewed_at",
    "review_notes",
)


def reclassify_annotation(annotation: dict[str, Any], *, role: str) -> dict[str, Any]:
    """Return a copy of `annotation` with the audit block populated for `role`."""
    procedure = dict(annotation.get("procedure", {}))
    audit = dict(procedure.get("audit") or {})
    audit.setdefault("gold_status", "pilot_gold")
    audit.setdefault("annotation_status", "reviewed" if role == "gold" else "second_pass")
    audit.setdefault("annotation_scope", "bounded_excerpt")
    audit.setdefault("annotator_count", 1 if role == "gold" else 1)
    audit.setdefault("guideline_version", "ipke-annotation-guideline-v0.1-pilot")
    audit.setdefault("model_draft_used", True)
    audit.setdefault("reviewed_by", "paper-author-self-review")
    audit.setdefault("reviewed_at", "2026-06-06")
    audit.setdefault(
        "review_notes",
        "Pilot annotation produced from model draft, then self-corrected. "
        "Not yet independently annotated. See datasets/paper/annotation_batches/.",
    )
    procedure["audit"] = audit
    return {**annotation, "procedure": procedure}


def update_manifest_columns(manifest: Path, rows: list[dict[str, str]]) -> None:
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing = list(reader)
    fieldnames = reader.fieldnames or []
    extra = ("gold_status", "annotation_status", "annotation_scope", "annotator_count")
    for name in extra:
        if name not in fieldnames:
            fieldnames.append(name)
    by_id = {row["document_id"]: row for row in rows}
    for row in existing:
        extra_row = by_id.get(row["document_id"], {})
        for name in extra:
            row[name] = extra_row.get(name, row.get(name, ""))
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)


def reclassify_directory(in_dir: Path, *, role: str) -> None:
    for path in sorted(in_dir.glob("*.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        updated = reclassify_annotation(annotation, role=role)
        path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{path.name}: reclassified to pilot_gold ({role})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", required=True, type=Path)
    parser.add_argument("--second-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--rows-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reclassify_directory(args.gold_dir, role="gold")
    reclassify_directory(args.second_dir, role="second_pass")
    rows = json.loads(args.rows_json.read_text(encoding="utf-8"))
    update_manifest_columns(args.manifest, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
