"""Schema validation helpers for extraction payloads."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jsonschema import Draft7Validator

SchemaDict = Dict[str, Any]

_SCHEMA_CACHE: SchemaDict | None = None


def load_schema() -> SchemaDict:
    """Load and cache the procedural graph schema."""

    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        schema_path = Path(__file__).resolve().parents[1] / "graph" / "schema.json"
        _SCHEMA_CACHE = json.loads(schema_path.read_text(encoding="utf-8"))
    return _SCHEMA_CACHE


def validate_extraction(payload: Dict[str, Any], autofix: bool = True) -> Tuple[bool, List[str]]:
    """Validate a prediction payload against the schema.

    Args:
        payload: Graph-like prediction dictionary.
        autofix: Whether to attempt auto-fixing common issues.

    Returns:
        Tuple of (is_valid, issues). Issues include both auto-fix notes and
        JSON-schema validation errors.
    """

    working = payload if autofix else deepcopy(payload)
    notes: List[str] = _apply_autofixes(working) if autofix else []
    validator = Draft7Validator(load_schema())
    error_messages = _collect_errors(validator, working)
    return len(error_messages) == 0, notes + error_messages


def _collect_errors(validator: Draft7Validator, candidate: Dict[str, Any]) -> List[str]:
    errors = []
    for error in sorted(validator.iter_errors(candidate), key=lambda e: list(e.path)):
        path = "/".join(str(part) for part in error.path) or "<root>"
        errors.append(f"{path}: {error.message}")
    return errors


def _apply_autofixes(payload: Dict[str, Any]) -> List[str]:
    fixes: List[str] = []
    if not payload.get("document_id"):
        payload["document_id"] = "unknown_document"
        fixes.append("AUTO-FIX: document_id missing -> set default")

    for key in ("steps", "conditions", "equipment", "parameters", "edges"):
        if key not in payload or payload[key] is None:
            payload[key] = []
            fixes.append(f"AUTO-FIX: Added empty list for '{key}'")

    steps = payload.get("steps", [])
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            payload["steps"][idx - 1] = {"id": f"S{idx}", "text": ""}
            fixes.append(f"AUTO-FIX: Step {idx} invalid -> replaced with default")
            continue
        if not step.get("id"):
            step["id"] = f"S{idx}"
            fixes.append(f"AUTO-FIX: Step {idx} missing id -> assigned S{idx}")
        if "text" not in step or step["text"] is None:
            step["text"] = ""
            fixes.append(f"AUTO-FIX: Step {idx} missing text -> set empty string")

    conditions = payload.get("conditions", [])
    for idx, condition in enumerate(conditions, start=1):
        if not isinstance(condition, dict):
            payload["conditions"][idx - 1] = {"id": f"C{idx}", "expression": ""}
            fixes.append(f"AUTO-FIX: Condition {idx} invalid -> replaced with default")
            continue
        if not condition.get("id"):
            condition["id"] = f"C{idx}"
            fixes.append(f"AUTO-FIX: Condition {idx} missing id -> assigned C{idx}")
        if "expression" not in condition or condition["expression"] is None:
            condition["expression"] = ""
            fixes.append(f"AUTO-FIX: Condition {idx} missing expression -> set empty string")

    edges = payload.get("edges", [])
    for idx, edge in enumerate(edges, start=1):
        if not isinstance(edge, dict):
            payload["edges"][idx - 1] = {"from_id": "", "to_id": "", "type": "NEXT"}
            fixes.append(f"AUTO-FIX: Edge {idx} invalid -> replaced with default")
            continue
        if not edge.get("type"):
            edge["type"] = "NEXT"
            fixes.append(f"AUTO-FIX: Edge {idx} missing type -> default NEXT")
        if "from_id" not in edge or edge["from_id"] is None:
            edge["from_id"] = ""
            fixes.append(f"AUTO-FIX: Edge {idx} missing from_id -> set empty string")
        if "to_id" not in edge or edge["to_id"] is None:
            edge["to_id"] = ""
            fixes.append(f"AUTO-FIX: Edge {idx} missing to_id -> set empty string")

    metadata = payload.get("metadata")
    if metadata is None or not isinstance(metadata, dict):
        payload["metadata"] = {}
        fixes.append("AUTO-FIX: metadata missing -> initialized empty object")

    return fixes


__all__ = ["load_schema", "validate_extraction"]
