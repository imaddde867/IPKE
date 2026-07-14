"""Shared exact agreement identities for annotation evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _constraint_records(
    annotation: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], str | None]]:
    constraints: list[tuple[Mapping[str, Any], str | None]] = []
    raw_top_constraints = annotation.get("constraints")
    if isinstance(raw_top_constraints, list):
        constraints.extend(
            (constraint, None)
            for constraint in raw_top_constraints
            if isinstance(constraint, Mapping)
        )

    raw_steps = annotation.get("steps")
    if not isinstance(raw_steps, list):
        return constraints
    for step in raw_steps:
        if not isinstance(step, Mapping):
            continue
        step_id = step.get("id") if isinstance(step.get("id"), str) else None
        nested = step.get("constraints")
        if isinstance(nested, list):
            constraints.extend(
                (constraint, step_id)
                for constraint in nested
                if isinstance(constraint, Mapping)
            )
        elif isinstance(nested, Mapping):
            for value in nested.values():
                if isinstance(value, list):
                    constraints.extend(
                        (constraint, step_id)
                        for constraint in value
                        if isinstance(constraint, Mapping)
                    )
                elif isinstance(value, str) and value.strip():
                    constraints.append(({"text": value}, step_id))
    return constraints


def _constraint_text(constraint: Mapping[str, Any]) -> str:
    for field in ("text", "expression", "condition", "statement"):
        value = constraint.get(field)
        if isinstance(value, str) and value.strip():
            return " ".join(value.lower().split())
    return ""


def _references(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [
            item if isinstance(item, str) else item["id"]
            for item in value
            if isinstance(item, str)
            or (isinstance(item, Mapping) and isinstance(item.get("id"), str))
        ]
    if isinstance(value, Mapping) and isinstance(value.get("id"), str):
        return [value["id"]]
    return []


def attachment_edge_set(
    annotation: Mapping[str, Any],
) -> set[tuple[str, str]]:
    """Return normalized exact constraint-to-step attachment identities."""
    edges: set[tuple[str, str]] = set()
    for constraint, containing_step_id in _constraint_records(annotation):
        text = _constraint_text(constraint)
        if not text:
            continue
        refs: list[str] = []
        for field in ("steps", "attached_to", "applies_to", "targets"):
            refs.extend(_references(constraint.get(field)))
        if not refs and containing_step_id:
            refs.append(containing_step_id)
        edges.update((text, ref) for ref in refs)
    return edges


def _normalized_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalized_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_normalized_json(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def _review_payload(value: Any) -> str:
    return json.dumps(
        _normalized_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def annotation_review_items(
    annotation: Mapping[str, Any],
) -> set[tuple[str, str]]:
    """Project all adjudication-relevant annotation dimensions to stable items."""
    items: set[tuple[str, str]] = set()
    raw_steps = annotation.get("steps")
    if isinstance(raw_steps, list):
        for step in raw_steps:
            if not isinstance(step, Mapping):
                continue
            step_record = dict(step)
            step_record.pop("constraints", None)
            items.add(("step", _review_payload(step_record)))

    for constraint, containing_step_id in _constraint_records(annotation):
        items.add(
            (
                "constraint",
                _review_payload(
                    {
                        "containing_step_id": containing_step_id,
                        "constraint": constraint,
                    }
                ),
            )
        )

    raw_relations = annotation.get("relations")
    if isinstance(raw_relations, list):
        items.update(
            ("relation", _review_payload(relation))
            for relation in raw_relations
            if isinstance(relation, Mapping)
        )

    items.update(
        ("attachment", _review_payload(edge))
        for edge in attachment_edge_set(annotation)
    )
    return items
