#!/usr/bin/env python3
"""Stamp human sign-off onto IPKE-Bench gold files.

This is the *only* step that converts a model-assisted + agent-adjudicated
gold into a human-verified one. It does NOT erase provenance: the
model-assisted lineage stays in the annotator string; a
`+ human-verified:<handle>` marker is appended and review_date is bumped to
the sign-off date. That preserves the honest datasheet trail (who/what
produced each label) while satisfying the validator's "annotator +
review_date set, review_status == reviewed" contract.

USAGE
  # dry-run every gold (default: shows the before/after annotator, writes nothing)
  python3 scripts/sign_off_gold.py --annotator imad

  # apply to ALL 8 golds, then validate
  python3 scripts/sign_off_gold.py --annotator imad --apply

  # apply to one doc only
  python3 scripts/sign_off_gold.py --annotator imad --doc olsk_small_cnc_v1_workbook --apply

  # custom date (defaults to today)
  python3 scripts/sign_off_gold.py --annotator imad --date 2026-07-10 --apply

After --apply the script runs validate_paper_gold.py --strict and refuses to
leave you in a broken state (non-zero exit if any gold fails).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOLD = REPO / "datasets" / "paper" / "gold"
PENDING_MARK = "(pending human sign-off)"
VERIFIED_PREFIX = "+ human-verified:"


def _sign_annotator(current: str, handle: str) -> str:
    """Append human-verified marker; strip the pending marker; idempotent."""
    s = (current or "").replace(PENDING_MARK, "").strip()
    s = s.rstrip("+").strip()
    marker = f"{VERIFIED_PREFIX}{handle}"
    if marker in s:  # already signed by this handle -> leave as-is
        return s
    return f"{s} {marker}".strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Stamp human sign-off onto gold files.")
    ap.add_argument("--annotator", required=True,
                    help="your handle, e.g. 'imad' -> appends '+ human-verified:imad'")
    ap.add_argument("--doc", default=None,
                    help="stem of one gold to sign (default: all 8)")
    ap.add_argument("--date", default=_dt.date.today().isoformat(),
                    help="review_date to stamp (default: today, ISO YYYY-MM-DD)")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default: dry-run)")
    args = ap.parse_args()

    files = sorted(GOLD.glob("*.json"))
    if args.doc:
        files = [p for p in files if p.stem == args.doc]
        if not files:
            print(f"ERROR: no gold named {args.doc!r} in {GOLD}", file=sys.stderr)
            return 2

    changed = 0
    for p in files:
        d = json.loads(p.read_text())
        q = d.setdefault("quality", {})
        before = q.get("annotator", "")
        after = _sign_annotator(before, args.annotator)
        q["annotator"] = after
        q["review_status"] = "reviewed"
        q["review_date"] = args.date
        tag = "SIGNED" if after != before else "noop"
        print(f"[{tag}] {p.stem}")
        print(f"        before: {before}")
        print(f"        after : {after}   review_date={args.date}")
        if args.apply:
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
            changed += 1

    if not args.apply:
        print("\nDRY-RUN — nothing written. Re-run with --apply to stamp.")
        return 0

    print(f"\nWrote {changed} file(s). Running strict validator...")
    rc = subprocess.call(
        [sys.executable, "scripts/validate_paper_gold.py",
         "--gold-dir", "datasets/paper/gold", "--strict"],
        cwd=REPO,
    )
    if rc != 0:
        print("VALIDATION FAILED — review output above before committing.", file=sys.stderr)
    else:
        print("All golds pass --strict. Safe to commit.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
