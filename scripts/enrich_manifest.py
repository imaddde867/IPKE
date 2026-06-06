"""Generate word_count, token_count, and other defensible columns for the paper manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


WORD_RE = re.compile(r"\b\w+\b")
METADATA_PREFIX = re.compile(r"^(document_id|source_family|title|direct_url|sha256):")


REQUIRED_COLUMNS: tuple[str, ...] = (
    "word_count",
    "token_count",
    "excerpt_word_count",
    "excerpt_token_count",
    "gold_status",
    "annotation_status",
    "annotation_scope",
    "annotator_count",
)


def _strip_metadata_header(text: str) -> str:
    lines = text.splitlines()
    body_start = 0
    for index, line in enumerate(lines):
        if METADATA_PREFIX.match(line):
            body_start = index + 1
            continue
        if line.strip() == "" and body_start > 0:
            body_start = index + 1
            continue
        break
    return "\n".join(lines[body_start:])


def compute_text_metrics(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"word_count": 0, "token_count": 0, "excerpt_word_count": 0, "excerpt_token_count": 0}
    raw = path.read_text(encoding="utf-8")
    body = _strip_metadata_header(raw)
    words = WORD_RE.findall(body)
    excerpt = body[-2000:] if len(body) > 2000 else body
    return {
        "word_count": len(words),
        "token_count": int(len(body.split())),
        "excerpt_word_count": len(WORD_RE.findall(excerpt)),
        "excerpt_token_count": int(len(excerpt.split())),
    }


def merge_metrics_into_manifest(
    rows: list[dict[str, str]], text_dir: Path
) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []
    for row in rows:
        text_path = text_dir / f"{row['document_id']}.txt"
        metrics = compute_text_metrics(text_path)
        merged = {**row}
        for key, value in metrics.items():
            merged[key] = str(value)
        enriched.append(merged)
    return enriched


def write_manifest(manifest: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for column in REQUIRED_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--text-dir", required=True, type=Path)
    parser.add_argument("--status-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    status = {
        row["document_id"]: row
        for row in json.loads(args.status_json.read_text(encoding="utf-8"))
    }
    enriched = merge_metrics_into_manifest(rows, args.text_dir)
    for row in enriched:
        doc_id = row.get("document_id")
        if doc_id in status:
            source_row = status[doc_id]
            for key in ("gold_status", "annotation_status", "annotation_scope", "annotator_count"):
                row[key] = source_row.get(key, row.get(key, ""))
    write_manifest(args.manifest, enriched)
    return 0


if __name__ == "__main__":
    sys.exit(main())
