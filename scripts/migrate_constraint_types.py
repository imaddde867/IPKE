"""Migrate constraint types in IPKE-Bench gold files to the locked taxonomy.

Applies docs/annotation/constraint-types.md mapping table to every constraint in
datasets/paper/gold/*.json. Mechanical for unambiguous types; requires --manual
classifications for type=requirement.

Usage:
    # Audit current state (no writes):
    uv run python scripts/migrate_constraint_types.py --dry-run

    # Apply mechanical mappings + manual classifications from a JSON file:
    uv run python scripts/migrate_constraint_types.py \\
        --manual-map docs/annotation/requirement-classifications.json

    # Output the list of ambiguous constraints needing manual classification:
    uv run python scripts/migrate_constraint_types.py --emit-ambiguous \\
        > docs/annotation/requirement-classifications.template.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

LOCKED_TYPES = {
    "precondition",
    "postcondition",
    "guard",
    "parameter",
    "role_assignment",
    "reference",
}
LOCKED_ENFORCEMENT = {"must", "should", "may"}

# Mechanical mapping: ad-hoc type -> (locked_type, locked_enforcement)
# Entries marked None require --manual-map lookup.
MECHANICAL_MAP: dict[str, tuple[str, str] | None] = {
    "precondition": ("precondition", "must"),
    "postcondition": ("postcondition", "must"),
    "warning": ("guard", "should"),
    "prohibition": ("guard", "must"),
    "approval": ("precondition", "must"),
    "documentation": ("postcondition", "must"),
    "role_assignment": ("role_assignment", "must"),
    "tolerance": ("parameter", "should"),
    "parameter": ("parameter", "must"),
    "selection_rule": ("parameter", "must"),
    "location_rule": ("parameter", "must"),
    "storage": ("parameter", "must"),
    "review_cycle": ("parameter", "must"),
    "order": ("precondition", "must"),
    "recommendation": ("guard", "should"),
    "guideline": ("guard", "should"),
    "permission": ("guard", "may"),
    "reference": ("reference", "must"),
    "requirement": None,
    "definition": None,
    "purpose": None,
}
DROP_TYPES = {"definition", "purpose"}


def iter_constraints(annotation: dict) -> Iterable[tuple[str, str, dict]]:
    """Yield (location, constraint_id, constraint_dict).

    location is "step:<id>" for embedded or "top" for procedure-level.
    """
    for step in annotation.get("steps", []):
        sid = step.get("id", "?")
        for c in step.get("constraints", []) or []:
            yield f"step:{sid}", c.get("id", "?"), c
    for c in annotation.get("constraints", []) or []:
        yield "top", c.get("id", "?"), c


def make_key(doc_id: str, location: str, constraint_id: str) -> str:
    return f"{doc_id}/{location}/{constraint_id}"


def audit(gold_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in sorted(gold_dir.glob("*.json")):
        d = json.loads(f.read_text())
        for _loc, _cid, c in iter_constraints(d):
            t = c.get("type", "")
            counts[t] = counts.get(t, 0) + 1
    return counts


def emit_ambiguous(gold_dir: Path) -> list[dict]:
    """Return list of {key, doc_id, location, constraint_id, current_type, text}
    for every constraint requiring manual classification."""
    out: list[dict] = []
    for f in sorted(gold_dir.glob("*.json")):
        d = json.loads(f.read_text())
        doc_id = f.stem
        for loc, cid, c in iter_constraints(d):
            t = c.get("type", "")
            if MECHANICAL_MAP.get(t) is None and t not in DROP_TYPES:
                out.append(
                    {
                        "key": make_key(doc_id, loc, cid),
                        "doc_id": doc_id,
                        "location": loc,
                        "constraint_id": cid,
                        "current_type": t,
                        "text": c.get("text", ""),
                        "TODO_locked_type": "",
                        "TODO_enforcement": "must",
                    }
                )
    return out


def migrate(
    gold_dir: Path, manual_map: dict[str, dict[str, str]], dry_run: bool
) -> tuple[int, int, list[str]]:
    """Apply the migration. Returns (changed, dropped, errors)."""
    changed = 0
    dropped = 0
    errors: list[str] = []
    for f in sorted(gold_dir.glob("*.json")):
        d = json.loads(f.read_text())
        doc_id = f.stem
        modified = False
        # Rewrite embedded constraints
        for step in d.get("steps", []):
            new_constraints = []
            for c in step.get("constraints", []) or []:
                t = c.get("type", "")
                if t in DROP_TYPES:
                    dropped += 1
                    modified = True
                    continue
                mapping = MECHANICAL_MAP.get(t)
                if mapping is None:
                    key = make_key(doc_id, f"step:{step['id']}", c.get("id", "?"))
                    manual = manual_map.get(key)
                    if manual is None or not manual.get("TODO_locked_type"):
                        errors.append(f"Missing manual classification for {key}")
                        new_constraints.append(c)
                        continue
                    lt = manual["TODO_locked_type"]
                    le = manual.get("TODO_enforcement", "must")
                else:
                    lt, le = mapping
                if lt not in LOCKED_TYPES or le not in LOCKED_ENFORCEMENT:
                    step_id = step["id"]
                    cid = c.get("id", "?")
                    bad_key = make_key(doc_id, f"step:{step_id}", cid)
                    errors.append(f"Invalid mapping for {bad_key}: {lt}/{le}")
                    new_constraints.append(c)
                    continue
                if c.get("type") != lt or c.get("enforcement") != le:
                    c["type"] = lt
                    c["enforcement"] = le
                    changed += 1
                    modified = True
                new_constraints.append(c)
            step["constraints"] = new_constraints
        # Rewrite top-level constraints
        new_top = []
        for c in d.get("constraints", []) or []:
            t = c.get("type", "")
            if t in DROP_TYPES:
                dropped += 1
                modified = True
                continue
            mapping = MECHANICAL_MAP.get(t)
            if mapping is None:
                key = make_key(doc_id, "top", c.get("id", "?"))
                manual = manual_map.get(key)
                if manual is None or not manual.get("TODO_locked_type"):
                    errors.append(f"Missing manual classification for {key}")
                    new_top.append(c)
                    continue
                lt = manual["TODO_locked_type"]
                le = manual.get("TODO_enforcement", "must")
            else:
                lt, le = mapping
            if c.get("type") != lt or c.get("enforcement") != le:
                c["type"] = lt
                c["enforcement"] = le
                changed += 1
                modified = True
            new_top.append(c)
        d["constraints"] = new_top
        if modified and not dry_run:
            f.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    return changed, dropped, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gold-dir", type=Path, default=Path("datasets/paper/gold"),
        help="Directory containing gold JSON files.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Audit only, no writes.")
    ap.add_argument(
        "--emit-ambiguous", action="store_true",
        help="Output template JSON for manual classification.",
    )
    ap.add_argument(
        "--manual-map", type=Path,
        help="JSON file with manual classifications keyed by doc_id/location/constraint_id.",
    )
    args = ap.parse_args()

    if args.emit_ambiguous:
        items = emit_ambiguous(args.gold_dir)
        print(json.dumps({item["key"]: item for item in items}, indent=2, ensure_ascii=False))
        return 0

    counts = audit(args.gold_dir)
    print("Current type distribution:", file=sys.stderr)
    for t, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {t:25s} {n}", file=sys.stderr)
    print(f"Total: {sum(counts.values())}", file=sys.stderr)

    manual_map: dict[str, dict[str, str]] = {}
    if args.manual_map and args.manual_map.exists():
        manual_map = json.loads(args.manual_map.read_text())

    changed, dropped, errors = migrate(args.gold_dir, manual_map, args.dry_run)
    print(f"\nChanged: {changed}", file=sys.stderr)
    print(f"Dropped: {dropped}", file=sys.stderr)
    if errors:
        print(f"\nErrors ({len(errors)}):", file=sys.stderr)
        for e in errors[:20]:
            print(f"  {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
        return 1
    if args.dry_run:
        print("\n(dry-run: no files modified)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
