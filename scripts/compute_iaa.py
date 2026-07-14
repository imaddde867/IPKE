"""Compute lightweight inter-annotator agreement for IPKE gold annotations."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.benchmark.taxonomy import LOCKED_CONSTRAINT_TYPES, LOCKED_ENFORCEMENT_LEVELS
from src.evaluation.agreement import attachment_edge_set


TOKEN_RE = re.compile(r"\b\w+\b")


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def f1_counts(reference: set[Any], predicted: set[Any]) -> dict[str, float | int | None]:
    true_positive = len(reference & predicted)
    false_positive = len(predicted - reference)
    false_negative = len(reference - predicted)
    reference_count = len(reference)
    predicted_count = len(predicted)

    if not reference_count and not predicted_count:
        precision = None
        recall = None
        f1 = None
    else:
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / reference_count if reference_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "reference_count": reference_count,
        "predicted_count": predicted_count,
    }


def _text_from_fields(item: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def step_set(annotation: dict[str, Any]) -> set[str]:
    return {
        normalize_text(text)
        for step in annotation.get("steps", [])
        if isinstance(step, dict)
        for text in [_text_from_fields(step, ("label", "text"))]
        if text
    }


def _iter_constraint_dicts(annotation: dict[str, Any]) -> list[tuple[dict[str, Any], str | None]]:
    constraints: list[tuple[dict[str, Any], str | None]] = []
    for constraint in annotation.get("constraints") or []:
        if isinstance(constraint, dict):
            constraints.append((constraint, None))

    for step in annotation.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_id = step.get("id") if isinstance(step.get("id"), str) else None
        nested = step.get("constraints")
        if isinstance(nested, list):
            constraints.extend((item, step_id) for item in nested if isinstance(item, dict))
        elif isinstance(nested, dict):
            for value in nested.values():
                if isinstance(value, list):
                    constraints.extend((item, step_id) for item in value if isinstance(item, dict))
                elif isinstance(value, str) and value.strip():
                    constraints.append(({"text": value}, step_id))
    return constraints


def constraint_text(constraint: dict[str, Any]) -> str:
    return _text_from_fields(constraint, ("text", "expression", "condition", "statement"))


def constraint_set(annotation: dict[str, Any]) -> set[str]:
    return {
        normalize_text(text)
        for constraint, _step_id in _iter_constraint_dicts(annotation)
        for text in [constraint_text(constraint)]
        if text
    }


def _coerce_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        refs: list[str] = []
        for item in value:
            if isinstance(item, str):
                refs.append(item)
            elif isinstance(item, dict) and isinstance(item.get("id"), str):
                refs.append(item["id"])
        return refs
    if isinstance(value, dict) and isinstance(value.get("id"), str):
        return [value["id"]]
    return []


def relation_set(annotation: dict[str, Any]) -> set[tuple[str, str]]:
    return attachment_edge_set(annotation)


def _quality_issues(annotation: dict[str, Any], role: str) -> list[str]:
    quality = annotation.get("quality", {})
    issues: list[str] = []
    if quality.get("review_status") != "reviewed":
        issues.append(
            f"{role} quality.review_status must be 'reviewed' "
            f"(got {quality.get('review_status')!r})"
        )
    if not quality.get("annotator"):
        issues.append(f"{role} quality.annotator missing")
    if not quality.get("review_date"):
        issues.append(f"{role} quality.review_date missing")
    return issues


def validate_iaa_pair(
    reference: dict[str, Any], predicted: dict[str, Any], filename: str
) -> list[str]:
    issues: list[str] = []
    issues.extend(_quality_issues(reference, "gold"))
    issues.extend(_quality_issues(predicted, "second"))

    gold_annotator = reference.get("quality", {}).get("annotator")
    second_annotator = predicted.get("quality", {}).get("annotator")
    if gold_annotator and second_annotator and gold_annotator == second_annotator:
        issues.append(f"{filename}: annotator must differ for independent IAA")

    for role, annotation in (("gold", reference), ("second", predicted)):
        for constraint, _step_id in _iter_constraint_dicts(annotation):
            cid = constraint.get("id", "?")
            ctype = constraint.get("type")
            if ctype not in LOCKED_CONSTRAINT_TYPES:
                issues.append(
                    f"{filename}: {role} constraint {cid} type={ctype!r} "
                    "not in locked vocabulary"
                )
            enforcement = constraint.get("enforcement")
            if enforcement not in LOCKED_ENFORCEMENT_LEVELS:
                issues.append(
                    f"{filename}: {role} constraint {cid} "
                    f"enforcement={enforcement!r} invalid"
                )
            refs: list[str] = []
            for field in ("attached_to", "applies_to"):
                refs.extend(_coerce_refs(constraint.get(field)))
            if not refs:
                issues.append(f"{filename}: {role} constraint {cid} missing attachment")
    return issues


def labeled_tokens(annotation: dict[str, Any]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for text in step_set(annotation):
        for token in TOKEN_RE.findall(text):
            counts[(token, "ACTION")] += 1
    for text in constraint_set(annotation):
        for token in TOKEN_RE.findall(text):
            counts[(token, "CONSTRAINT")] += 1
    return counts


def token_label_pairs(first: dict[str, Any], second: dict[str, Any]) -> list[tuple[str, str]]:
    first_counts = labeled_tokens(first)
    second_counts = labeled_tokens(second)
    keys = sorted(set(first_counts) | set(second_counts))
    pairs: list[tuple[str, str]] = []
    for token, label in keys:
        total = max(first_counts[(token, label)], second_counts[(token, label)])
        pairs.extend((label if index < first_counts[(token, label)] else "O",
                      label if index < second_counts[(token, label)] else "O")
                     for index in range(total))
    return pairs


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(1 for first, second in pairs if first == second) / total
    first_counts = Counter(first for first, _second in pairs)
    second_counts = Counter(second for _first, second in pairs)
    labels = set(first_counts) | set(second_counts)
    expected = sum(
        (first_counts[label] / total) * (second_counts[label] / total)
        for label in labels
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1 - expected)


def compare_annotations(reference: dict[str, Any], predicted: dict[str, Any]) -> dict[str, Any]:
    pairs = token_label_pairs(reference, predicted)
    return {
        "step_exact": f1_counts(step_set(reference), step_set(predicted)),
        "constraint_exact": f1_counts(constraint_set(reference), constraint_set(predicted)),
        "relation_exact": f1_counts(relation_set(reference), relation_set(predicted)),
        "token_label_kappa": cohen_kappa(pairs),
        "token_label_pairs": len(pairs),
        "_pairs": pairs,
    }


def _sum_counts(documents: dict[str, dict[str, Any]], metric: str) -> dict[str, int]:
    totals = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "reference_count": 0,
        "predicted_count": 0,
    }
    for metrics in documents.values():
        for key in totals:
            totals[key] += int(metrics[metric][key])
    return totals


def _f1_from_totals(totals: dict[str, int]) -> dict[str, float | int | None]:
    tp = totals["true_positive"]
    reference_count = totals["reference_count"]
    predicted_count = totals["predicted_count"]

    if not reference_count and not predicted_count:
        precision = None
        recall = None
        f1 = None
    else:
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / reference_count if reference_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**totals, "precision": precision, "recall": recall, "f1": f1}


def aggregate_metrics(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for metric in ("step_exact", "constraint_exact", "relation_exact"):
        aggregate[metric] = _f1_from_totals(_sum_counts(documents, metric))

    all_pairs: list[tuple[str, str]] = []
    for metrics in documents.values():
        all_pairs.extend(metrics["_pairs"])
    aggregate["token_label_kappa"] = cohen_kappa(all_pairs)
    aggregate["token_label_pairs"] = len(all_pairs)
    return aggregate


def compute_iaa(gold_dir: Path, second_dir: Path) -> dict[str, Any]:
    documents: dict[str, dict[str, Any]] = {}
    for gold_path in sorted(gold_dir.glob("*.json")):
        second_path = second_dir / gold_path.name
        if not second_path.exists():
            continue
        document_id = gold_path.stem
        reference = json.loads(gold_path.read_text(encoding="utf-8"))
        predicted = json.loads(second_path.read_text(encoding="utf-8"))
        issues = validate_iaa_pair(reference, predicted, gold_path.name)
        if issues:
            message = "IAA eligibility failed:\n" + "\n".join(f"  - {issue}" for issue in issues)
            raise SystemExit(message)
        metrics = compare_annotations(reference, predicted)
        # _pairs already included by compare_annotations; no recomputation needed
        documents[document_id] = metrics

    if not documents:
        raise SystemExit("No matching annotation files found.")

    aggregate = aggregate_metrics(documents)
    public_documents = {
        document_id: {key: value for key, value in metrics.items() if key != "_pairs"}
        for document_id, metrics in documents.items()
    }
    return {"documents": public_documents, "aggregate": aggregate}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", required=True, type=Path)
    parser.add_argument("--second-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = compute_iaa(args.gold_dir, args.second_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
