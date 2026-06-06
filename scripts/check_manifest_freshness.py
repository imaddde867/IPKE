"""Fail CI when the paper manifest drifts from committed text and gold files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Ensure the repository root is importable when running from scripts/
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.enrich_manifest import compute_text_metrics, REQUIRED_COLUMNS  # noqa: E402


EXPECTED_STATUSES: tuple[str, ...] = (
    "gold_status",
    "annotation_status",
    "annotation_scope",
    "annotator_count",
)


def check_manifest_freshness(manifest: Path, text_dir: Path) -> dict[str, Any]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    report_rows: list[dict[str, Any]] = []
    for row in rows:
        metrics = compute_text_metrics(text_dir / f"{row['document_id']}.txt")
        drift = (
            str(metrics["word_count"]) != row.get("word_count", "")
            or str(metrics["token_count"]) != row.get("token_count", "")
        )
        missing_columns = [col for col in REQUIRED_COLUMNS if not row.get(col)]
        report_rows.append(
            {
                "document_id": row["document_id"],
                "manifest_word_count": int(row.get("word_count", 0) or 0),
                "fresh_word_count": metrics["word_count"],
                "manifest_token_count": int(row.get("token_count", 0) or 0),
                "fresh_token_count": metrics["token_count"],
                "drift": drift,
                "missing_columns": missing_columns,
                "gold_status": row.get("gold_status", ""),
                "annotation_status": row.get("annotation_status", ""),
                "annotation_scope": row.get("annotation_scope", ""),
                "annotator_count": row.get("annotator_count", ""),
            }
        )
    return {"rows": report_rows, "drift_count": sum(1 for r in report_rows if r["drift"])}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--text-dir", required=True, type=Path)
    parser.add_argument("--out", required=False, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_manifest_freshness(args.manifest, args.text_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["drift_count"]:
        print(f"manifest drift: {report['drift_count']} rows")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
