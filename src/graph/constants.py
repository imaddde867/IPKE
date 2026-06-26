"""Shared constants and utilities for the graph module.

Consolidated from builder.py and adapter.py to eliminate duplication of:
- Step reference key sets
- Edge type mappings (upper-case -> lower-case)
- Condition type constraints
- Reference flattening utilities

Usage:
    from src.graph.constants import STEP_REF_KEYS, flatten_refs, LOWER_REL_MAP
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.benchmark.taxonomy import LOCKED_CONSTRAINT_TYPES


STEP_REF_KEYS: tuple[str, ...] = (
    "step",
    "steps",
    "step_id",
    "attached_to",
    "attached_step",
    "attached_steps",
    "targets",
    "scope",
    "applies_to",
)


ALLOWED_CONDITION_TYPES: set[str] = set(LOCKED_CONSTRAINT_TYPES)


LOWER_REL_MAP: dict[str, str] = {
    "NEXT": "next",
    "CONDITION_ON": "condition_on",
    "USES": "uses",
    "HAS_PARAMETER": "has_parameter",
    "REQUIRES": "requires",
    "PRODUCES": "produces",
    "REFERENCES": "references",
    "ALTERNATIVE_TO": "alternative_to",
}


def normalize_id(value: Any) -> str:
    """Convert a raw ID value to a stripped string, returning '' for None/empty."""
    return str(value).strip() if value is not None else ""


def flatten_refs(value: Any) -> List[str]:
    """Recursively flatten a value (str, dict, list, set) into a list of string refs.

    Handles the same shapes as both builder.py's _flatten_ref_value and
    adapter.py's _flatten_refs:
    - str -> [str]
    - dict -> extracts id/step_id/step values
    - list/set -> recursive flatten of each element
    - other -> [str(value)]
    """
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        potential = value.get("id") or value.get("step_id") or value.get("step")
        if potential:
            return [str(potential)]
        result: List[str] = []
        for nested_value in value.values():
            result.extend(flatten_refs(nested_value))
        return result
    if isinstance(value, (list, tuple, set)):
        refs: List[str] = []
        for item in value:
            refs.extend(flatten_refs(item))
        return refs
    return [str(value)]


def collect_step_refs(record: Dict[str, Any]) -> List[str]:
    """Collect all step ID references from a constraint/entity record using STEP_REF_KEYS."""
    refs: List[str] = []
    for key in STEP_REF_KEYS:
        raw = record.get(key)
        refs.extend(flatten_refs(raw))
    return refs
