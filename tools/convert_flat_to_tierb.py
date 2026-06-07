#!/usr/bin/env python3
"""
Convert a directory of flat JSON predictions/gold files into Tier-B (nodes + edges) format.

Usage:
  python -m tools.convert_flat_to_tierb \
    --in datasets/archive/gold_human \
    --out datasets/archive/gold_human_tierb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from src.graph.adapter import flat_to_tierb


def convert_dir(in_path: Path, out_path: Path) -> int:
    out_path.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(in_path.glob("*.json")):
        try:
            data: Dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {src.name}: failed to read JSON ({exc})")
            continue
        tierb = flat_to_tierb(data)
        (out_path / src.name).write_text(json.dumps(tierb, indent=2), encoding="utf-8")
        count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert flat JSONs to Tier-B nodes/edges format")
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="Input directory of flat JSON files")
    ap.add_argument("--out", dest="out", type=Path, required=True, help="Output directory for Tier-B JSON files")
    args = ap.parse_args()

    if not args.inp.exists() or not args.inp.is_dir():
        print(f"Input directory not found: {args.inp}")
        return 1

    count = convert_dir(args.inp, args.out)
    print(f"Converted {count} files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
