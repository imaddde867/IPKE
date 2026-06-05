"""Download public source documents listed in the paper dataset manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Iterable


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() == "true"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_rows(manifest: Path) -> Iterable[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        yield from (
            row for row in csv.DictReader(handle) if truthy(row.get("selected_for_download"))
        )


def download_public_sources(manifest: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in selected_rows(manifest):
        filename = row["local_filename"]
        url = row["direct_url"]
        expected_sha = row.get("sha256", "").strip().lower()
        target = out_dir / filename

        if target.exists():
            actual_sha = sha256_file(target)
            if expected_sha and actual_sha != expected_sha:
                raise SystemExit(
                    f"{filename}: existing file sha256 mismatch "
                    f"(expected {expected_sha}, got {actual_sha})"
                )
            print(f"{filename}: exists")
            continue

        urllib.request.urlretrieve(url, target)
        if expected_sha:
            actual_sha = sha256_file(target)
            if actual_sha != expected_sha:
                target.unlink(missing_ok=True)
                raise SystemExit(
                    f"{filename}: downloaded sha256 mismatch "
                    f"(expected {expected_sha}, got {actual_sha})"
                )
        print(f"{filename}: downloaded")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    download_public_sources(args.manifest, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
