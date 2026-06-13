"""Constraint-blindness report: how many constraints did the LLM-drafted gold miss?

Compares the LLM-drafted gold files (at the specified git ref, default origin/main)
against the reviewed paper-grade gold (current working tree). Reports:

  * per-file constraint count delta
  * per-locked-type recall (draft constraints whose text overlaps reviewed gold)
  * macro statistics

The output is the source for the §1 motivating result in the IPKE-Bench paper:
"LLM-drafted procedural extractors systematically under-recall safety constraints
by a factor of N, with the largest gaps in [type] constraints."

This is an INTRA-DATASET measurement (draft vs reviewed gold). It is not a
substitute for the full extractor benchmark; rather it shows that the
constraint-blindness phenomenon exists even in the same LLM-drafted material the
gold was reviewed from.

Usage:
    uv run python scripts/constraint_blindness_report.py [--draft-ref origin/main]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def iter_constraints(annotation: dict) -> Iterable[tuple[str, str, dict]]:
    for step in annotation.get("steps", []):
        sid = step.get("id", "?")
        for c in step.get("constraints", []) or []:
            yield f"step:{sid}", c.get("id", "?"), c
    for c in annotation.get("constraints", []) or []:
        yield "top", c.get("id", "?"), c


def load_draft(path: Path, ref: str) -> dict | None:
    rel = path.as_posix()
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{ref}:{rel}"], stderr=subprocess.DEVNULL
        ).decode()
    except subprocess.CalledProcessError:
        return None
    return json.loads(raw)


def text_overlap(a: str, b: str, threshold: float = 0.5) -> bool:
    """Cheap surrogate: token-set Jaccard over normalised forms."""
    sa = set(normalize(a).split())
    sb = set(normalize(b).split())
    if not sa or not sb:
        return False
    inter = sa & sb
    return len(inter) / max(len(sa), len(sb)) >= threshold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-dir", type=Path, default=Path("datasets/paper/gold"))
    ap.add_argument(
        "--draft-ref", default="origin/main",
        help="Git ref where the LLM-drafted gold lives (default: origin/main)",
    )
    ap.add_argument("--out", type=Path, default=None, help="Write JSON report to this path.")
    args = ap.parse_args()

    per_file = []
    macro_draft_count = 0
    macro_reviewed_count = 0
    type_distribution_reviewed: Counter[str] = Counter()
    type_distribution_recovered: Counter[str] = Counter()

    for f in sorted(args.gold_dir.glob("*.json")):
        reviewed = json.loads(f.read_text())
        draft = load_draft(f, args.draft_ref)
        if draft is None:
            print(f"SKIP {f.name} — no draft at {args.draft_ref}", file=sys.stderr)
            continue

        reviewed_constraints = [c for _l, _i, c in iter_constraints(reviewed)]
        draft_constraints = [c for _l, _i, c in iter_constraints(draft)]
        draft_texts = [normalize(c.get("text", "")) for c in draft_constraints]

        type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"reviewed": 0, "recovered": 0})

        recovered = 0
        for rc in reviewed_constraints:
            t = rc.get("type", "unknown")
            type_distribution_reviewed[t] += 1
            type_counts[t]["reviewed"] += 1
            rt = rc.get("text", "")
            if any(text_overlap(rt, dt) for dt in draft_texts):
                recovered += 1
                type_distribution_recovered[t] += 1
                type_counts[t]["recovered"] += 1

        n_draft = len(draft_constraints)
        n_reviewed = len(reviewed_constraints)
        macro_draft_count += n_draft
        macro_reviewed_count += n_reviewed
        recall = recovered / n_reviewed if n_reviewed else 0.0
        per_file.append({
            "file": f.name,
            "n_draft_constraints": n_draft,
            "n_reviewed_constraints": n_reviewed,
            "recovered": recovered,
            "draft_recall_against_gold": round(recall, 3),
            "delta_constraints": n_reviewed - n_draft,
            "per_type": dict(type_counts),
        })

    print("=" * 80)
    print(f"Constraint-blindness report: draft @ {args.draft_ref} vs reviewed gold")
    print("=" * 80)
    print()
    print(f"{'File':55s}  draft  rev  rec  recall")
    print("-" * 80)
    for r in per_file:
        print(f"{r['file']:55s}  {r['n_draft_constraints']:5d}  {r['n_reviewed_constraints']:3d}  {r['recovered']:3d}  {r['draft_recall_against_gold']:.3f}")
    print("-" * 80)
    total_recovered = sum(r["recovered"] for r in per_file)
    macro_recall = total_recovered / macro_reviewed_count if macro_reviewed_count else 0.0
    print(f"{'TOTAL':55s}  {macro_draft_count:5d}  {macro_reviewed_count:3d}  {total_recovered:3d}  {macro_recall:.3f}")
    print()
    print("Per-type recall (reviewed gold recovered by draft):")
    for t in sorted(type_distribution_reviewed):
        rev = type_distribution_reviewed[t]
        rec = type_distribution_recovered[t]
        rec_rate = rec / rev if rev else 0.0
        print(f"  {t:20s}  {rec:3d} / {rev:3d}  ({rec_rate:.3f})")
    print()
    expansion = macro_reviewed_count / macro_draft_count if macro_draft_count else 0.0
    print(f"Expansion ratio (reviewed / draft): {expansion:.2f}x")
    print(f"Macro recall (draft against reviewed): {macro_recall:.3f}")
    print()

    if args.out:
        report = {
            "draft_ref": args.draft_ref,
            "per_file": per_file,
            "macro": {
                "draft_total": macro_draft_count,
                "reviewed_total": macro_reviewed_count,
                "recovered": total_recovered,
                "recall": round(macro_recall, 4),
                "expansion": round(expansion, 4),
            },
            "per_type_recovery": {
                t: {
                    "reviewed": type_distribution_reviewed[t],
                    "recovered": type_distribution_recovered[t],
                    "recall": round(type_distribution_recovered[t] / type_distribution_reviewed[t], 4)
                    if type_distribution_reviewed[t] else 0.0,
                }
                for t in type_distribution_reviewed
            },
        }
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"Report written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
