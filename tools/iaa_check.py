#!/usr/bin/env python3
"""Check inter-annotator agreement for a single document pair.

Wraps scripts/compute_iaa.py for single-pair use. Takes two gold JSON files
for the same document (annotator A, annotator B), computes:
  (a) step span agreement: F1 on normalised step text (presence/absence)
  (b) constraint-step assignment agreement: F1 on (constraint_text, step_id) pairs
  (c) Cohen's kappa on token labels (holistic, from compute_iaa)

Prints per-dimension scores and a PASS/FAIL flag at the given threshold.
PASS requires token_label_kappa >= threshold (default 0.7).

Usage:
    uv run python tools/iaa_check.py <annotator_a.json> <annotator_b.json>
    uv run python tools/iaa_check.py <annotator_a.json> <annotator_b.json> --threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compute_iaa import compare_annotations


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotator_a", type=Path, help="First annotator gold JSON.")
    parser.add_argument("annotator_b", type=Path, help="Second annotator gold JSON.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Kappa threshold for PASS (default: 0.7).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path_a, path_b = args.annotator_a.resolve(), args.annotator_b.resolve()

    for p in (path_a, path_b):
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            return 1

    ann_a = json.loads(path_a.read_text(encoding="utf-8"))
    ann_b = json.loads(path_b.read_text(encoding="utf-8"))

    metrics = compare_annotations(ann_a, ann_b)

    step_f1 = metrics["step_exact"].get("f1")
    constraint_f1 = metrics["constraint_exact"].get("f1")
    kappa = metrics.get("token_label_kappa")
    n_pairs = metrics.get("token_label_pairs", 0)

    t = args.threshold
    flag = ("PASS" if kappa is not None and kappa >= t else "FAIL") if kappa is not None else "N/A"

    print(f"Files:  A = {path_a.name}")
    print(f"        B = {path_b.name}")
    print(f"Threshold: {t}")
    print()
    print(f"  (a) Step span F1             = {_fmt(step_f1)}")
    print(f"      step TP/FP/FN            = {metrics['step_exact']['true_positive']} / "
          f"{metrics['step_exact']['false_positive']} / {metrics['step_exact']['false_negative']}")
    print()
    print(f"  (b) Constraint assignment F1 = {_fmt(constraint_f1)}")
    print(f"      constraint TP/FP/FN      = {metrics['constraint_exact']['true_positive']} / "
          f"{metrics['constraint_exact']['false_positive']} / {metrics['constraint_exact']['false_negative']}")
    print()
    print(f"  (c) Token-label kappa        = {_fmt(kappa)}  ({n_pairs} pairs)")
    print()
    print(f"  PASS/FAIL (kappa >= {t}): {flag}")

    return 0 if flag == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
