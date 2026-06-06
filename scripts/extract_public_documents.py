"""Extract normalized text files for paper documents. Selection is mode-driven."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable


SPACE_RUN = re.compile(r"[^\S\r\n]+")


SELECTION_MODES: tuple[str, ...] = ("gold", "download", "all")


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() == "true"


def selected_rows(rows: list[dict[str, str]], selection: str) -> Iterable[dict[str, str]]:
    if selection == "gold":
        yield from (row for row in rows if truthy(row.get("selected_for_gold")))
    elif selection == "download":
        yield from (row for row in rows if truthy(row.get("selected_for_download")))
    elif selection == "all":
        yield from rows
    else:
        raise SystemExit(f"unknown selection mode: {selection!r}")


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


def extract_public_documents(
    manifest: Path, raw_dir: Path, out_dir: Path, *, selection: str = "gold"
) -> list[str]:
    if selection not in SELECTION_MODES:
        raise SystemExit(f"unknown selection mode: {selection!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in selected_rows(rows, selection):
        document_id = row["document_id"]
        raw_path = raw_dir / row["local_filename"]
        if not raw_path.exists():
            print(f"{document_id}: missing raw source {raw_path}, skipping")
            continue
        text = read_source_text(raw_path) if raw_path.suffix.lower() == ".pdf" else raw_path.read_text(encoding="utf-8")
        target = out_dir / f"{document_id}.txt"
        target.write_text(metadata_header(row) + normalize_text(text), encoding="utf-8")
        written.append(target.name)
        print(f"{document_id}: extracted (mode={selection})")
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--selection",
        choices=SELECTION_MODES,
        default="gold",
        help=(
            "Which manifest rows to extract. "
            "'gold' (default) extracts only selected_for_gold=true. "
            "'download' extracts everything in the first-wave download. "
            "'all' extracts every manifest row."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extract_public_documents(
        args.manifest, args.raw_dir, args.out_dir, selection=args.selection
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
