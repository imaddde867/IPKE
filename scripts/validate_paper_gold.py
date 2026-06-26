"""Paper-grade validator for IPKE-Bench gold files.

Stricter than schema-level validation. Enforces:
  - quality.review_status == "reviewed"
  - quality.annotator and quality.review_date set
  - all constraints use the locked type vocabulary (6 values)
  - all constraints have enforcement in {must, should, may}
  - all constraints have non-empty text
  - every step has at least one constraint (warn, not fail)
  - every constraint attaches to at least one valid step id

Usage:
    uv run python scripts/validate_paper_gold.py [--gold-dir datasets/paper/gold] [--strict]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from src.benchmark.taxonomy import LOCKED_CONSTRAINT_TYPES, LOCKED_ENFORCEMENT_LEVELS

LOCKED_TYPES = LOCKED_CONSTRAINT_TYPES
LOCKED_ENFORCEMENT = LOCKED_ENFORCEMENT_LEVELS


def iter_constraints(annotation: dict) -> Iterable[tuple[str, str, dict]]:
    for step in annotation.get("steps", []):
        sid = step.get("id", "?")
        for c in step.get("constraints", []) or []:
            yield f"step:{sid}", c.get("id", "?"), c
    for c in annotation.get("constraints", []) or []:
        yield "top", c.get("id", "?"), c


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return [f"JSON parse error: {e}"]

    quality = d.get("quality", {})
    if quality.get("review_status") != "reviewed":
        errors.append(f"quality.review_status != 'reviewed' (got {quality.get('review_status')!r})")
    if not quality.get("annotator"):
        errors.append("quality.annotator missing")
    if not quality.get("review_date"):
        errors.append("quality.review_date missing")

    step_ids = {s.get("id") for s in d.get("steps", []) if s.get("id")}
    if not step_ids:
        errors.append("no step ids")
        return errors

    for loc, cid, c in iter_constraints(d):
        prefix = f"{loc}/{cid}"
        t = c.get("type")
        if t not in LOCKED_TYPES:
            errors.append(f"{prefix}: type={t!r} not in locked vocabulary")
        enf = c.get("enforcement")
        if enf not in LOCKED_ENFORCEMENT:
            errors.append(f"{prefix}: enforcement={enf!r} not in {{must,should,may}}")
        if not c.get("text"):
            errors.append(f"{prefix}: empty text")
        refs = []
        for field in ("attached_to", "applies_to"):
            v = c.get(field)
            if isinstance(v, str):
                refs.append(v)
            elif isinstance(v, list):
                refs.extend(v)
        if not refs:
            errors.append(f"{prefix}: no attached_to or applies_to")
        else:
            for r in refs:
                if r not in step_ids:
                    errors.append(f"{prefix}: ref {r!r} not in step ids")

    # Count procedure-level constraints whose applies_to includes each step.
    step_procedure_counts: dict[str, int] = {sid: 0 for sid in step_ids}
    for c in d.get("constraints", []) or []:
        applies = c.get("applies_to")
        if isinstance(applies, str):
            applies = [applies]
        if isinstance(applies, list):
            for sid in applies:
                if sid in step_procedure_counts:
                    step_procedure_counts[sid] += 1

    for s in d.get("steps", []):
        sid = s.get("id", "?")
        embedded = len(s.get("constraints", []) or [])
        procedure = step_procedure_counts.get(sid, 0)
        total = embedded + procedure
        if total == 0:
            warnings.append(
                f"step:{sid}: 0 attached constraints (embedded={embedded}, procedure-level={procedure}); "
                "re-read source"
            )
        elif total > 10:
            warnings.append(
                f"step:{sid}: {total} constraints (embedded={embedded}, procedure-level={procedure}); "
                "consider splitting step"
            )

    return [f"ERROR: {e}" for e in errors] + [f"WARN: {w}" for w in warnings]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-dir", type=Path, default=Path("datasets/paper/gold"))
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    args = ap.parse_args()

    any_error = False
    any_warn = False
    for f in sorted(args.gold_dir.glob("*.json")):
        msgs = validate_file(f)
        errs = [m for m in msgs if m.startswith("ERROR")]
        warns = [m for m in msgs if m.startswith("WARN")]
        if errs:
            any_error = True
            print(f"FAIL {f.name}")
            for m in errs:
                print(f"  {m}")
        if warns:
            any_warn = True
            for m in warns:
                print(f"  {f.name}: {m}")
        if not errs and not warns:
            print(f"PASS {f.name}")
    if any_error:
        return 1
    if any_warn and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
