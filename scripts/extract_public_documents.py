"""Extract normalized text files for selected public paper documents."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable


SPACE_RUN = re.compile(r"[^\S\r\n]+")


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() == "true"


def selected_gold_rows(manifest: Path) -> Iterable[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        yield from (row for row in csv.DictReader(handle) if truthy(row.get("selected_for_gold")))


def normalize_text(text: str) -> str:
    normalized_lines = [SPACE_RUN.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(normalized_lines).strip() + "\n"


def extract_pdf_text(path: Path) -> str:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is required for PDF extraction. Run `uv sync --extra extras`."
        ) from exc

    pages: list[str] = []
    with fitz.open(path) as document:
        for page in document:
            pages.append(page.get_text("text"))
    return "\n".join(pages)


def read_source_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return extract_pdf_text(path)
    raise SystemExit(f"{path.name}: unsupported source extension {path.suffix!r}")


def metadata_header(row: dict[str, str]) -> str:
    fields = ("document_id", "source_family", "title", "direct_url", "sha256")
    return "\n".join(f"{field}: {row.get(field, '').strip()}" for field in fields) + "\n\n"


def extract_public_documents(manifest: Path, raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in selected_gold_rows(manifest):
        document_id = row["document_id"]
        raw_path = raw_dir / row["local_filename"]
        if not raw_path.exists():
            raise SystemExit(f"{document_id}: missing raw source {raw_path}")

        if raw_path.suffix.lower() == ".txt":
            text = raw_path.read_text(encoding="utf-8")
        else:
            text = read_source_text(raw_path)

        target = out_dir / f"{document_id}.txt"
        target.write_text(metadata_header(row) + normalize_text(text), encoding="utf-8")
        print(f"{document_id}: extracted")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extract_public_documents(args.manifest, args.raw_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
