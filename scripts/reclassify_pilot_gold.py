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


def reclassify_annotation(
    annotation: dict[str, Any],
    *,
    annotation_status: str,
    annotation_scope: str,
    annotator_count: int,
) -> dict[str, Any]:
    """Return a copy of `annotation` with the audit block populated."""
    procedure = dict(annotation.get("procedure", {}))
    audit = dict(procedure.get("audit") or {})
    audit["gold_status"] = "pilot_gold"
    audit["annotation_status"] = annotation_status
    audit["annotation_scope"] = annotation_scope
    audit["annotator_count"] = annotator_count
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


def _check_orphan_files(
    in_dir: Path,
    *,
    role: str,
    rows_by_id: dict[str, dict[str, str]],
    rows_path: Path,
) -> list[str]:
    """Return document_ids that have a file but no rows entry. Empty list means OK."""
    orphans: list[str] = []
    for path in sorted(in_dir.glob("*.json")):
        document_id = path.stem
        if document_id not in rows_by_id:
            orphans.append(document_id)
        elif role == "second_pass" and rows_by_id[document_id]["annotation_status"] != "double_annotated":
            raise SystemExit(
                f"{document_id}: second_pass file exists but manifest says "
                f"annotation_status={rows_by_id[document_id]['annotation_status']!r}. "
                f"Move the file or update {rows_path}."
            )
    return orphans


def reclassify_directory(
    in_dir: Path,
    *,
    role: str,
    rows_by_id: dict[str, dict[str, str]],
) -> None:
    for path in sorted(in_dir.glob("*.json")):
        document_id = path.stem
        row = rows_by_id[document_id]
        annotation = json.loads(path.read_text(encoding="utf-8"))
        updated = reclassify_annotation(
            annotation,
            annotation_status=row["annotation_status"],
            annotation_scope=row["annotation_scope"],
            annotator_count=int(row["annotator_count"]),
        )
        path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"{path.name}: audit stamped "
            f"(status={row['annotation_status']}, "
            f"scope={row['annotation_scope']}, "
            f"annotators={row['annotator_count']})"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", required=True, type=Path)
    parser.add_argument("--second-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--rows-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = json.loads(args.rows_json.read_text(encoding="utf-8"))
    rows_by_id = {row["document_id"]: row for row in rows}

    for role, in_dir in (("gold", args.gold_dir), ("second_pass", args.second_dir)):
        orphans = _check_orphan_files(
            in_dir, role=role, rows_by_id=rows_by_id, rows_path=args.rows_json
        )
        if orphans:
            raise SystemExit(
                f"{in_dir}: file(s) have no entry in {args.rows_json}: {orphans}. "
                f"Add rows for them or remove the files."
            )
        reclassify_directory(in_dir, role=role, rows_by_id=rows_by_id)

    update_manifest_columns(args.manifest, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
