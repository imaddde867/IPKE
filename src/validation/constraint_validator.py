"""Semantic constraint validation helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


TEMPORAL_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "daily": ("daily", "every day", "each day"),
    "weekly": ("weekly", "each week", "every week"),
    "monthly": ("monthly", "each month", "every month"),
}

HAZARD_KEYWORDS = (
    "hazard",
    "danger",
    "risk",
    "injury",
    "toxic",
    "flammable",
    "explosive",
)


@dataclass
class ValidationReport:
    passed: List[str] = field(default_factory=list)
    warnings: List[Tuple[str, str]] = field(default_factory=list)
    errors: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Iterable]:
        return {
            "passed": list(self.passed),
            "warnings": [(cid, msg) for cid, msg in self.warnings],
            "errors": [(cid, msg) for cid, msg in self.errors],
        }


def has_attached_steps(constraint: Dict[str, object]) -> bool:
    refs = _extract_refs(constraint)
    return len(refs) > 0


def validate_temporal_consistency(constraints: Sequence[Dict[str, object]]) -> List[Tuple[str, str]]:
    conflicts: List[Tuple[str, str]] = []
    step_freqs: Dict[str, Dict[str, str]] = defaultdict(dict)
    for constraint in constraints:
        cid = _constraint_id(constraint)
        frequency = _detect_frequency(constraint)
        if not frequency:
            continue
        for step_ref in _extract_refs(constraint):
            seen = step_freqs[step_ref]
            for existing_freq, existing_cid in seen.items():
                if existing_freq != frequency:
                    conflicts.append(
                        (
                            cid,
                            f"Conflicting cadence '{frequency}' vs '{existing_freq}' on step {step_ref}",
                        )
                    )
            seen[frequency] = cid
    return conflicts


def validate_safety_types(constraints: Sequence[Dict[str, object]]) -> List[Tuple[str, str]]:
    warnings: List[Tuple[str, str]] = []
    for constraint in constraints:
        ctype = str(constraint.get("type") or "").lower()
        if ctype != "safety":
            continue
        expression = _constraint_text(constraint)
        if not any(keyword in expression.lower() for keyword in HAZARD_KEYWORDS):
            warnings.append((_constraint_id(constraint), "Safety constraint lacks hazard keywords"))
    return warnings


def validate_parameter_ranges(constraints: Sequence[Dict[str, object]]) -> List[Tuple[str, str]]:
    errors: List[Tuple[str, str]] = []
    number_pattern = re.compile(r"-?\d+(?:\.\d+)?")
    for constraint in constraints:
        expression = _constraint_text(constraint)
        numbers = [float(match) for match in number_pattern.findall(expression)]
        if len(numbers) < 2:
            continue
        first, second = numbers[0], numbers[1]
        if first > second:
            errors.append(
                (
                    _constraint_id(constraint),
                    f"Invalid parameter range: {first} exceeds {second}",
                )
            )
    return errors


def validate_constraints(constraints: Sequence[Dict[str, object]]) -> ValidationReport:
    report = ValidationReport()
    base_passed: List[str] = []
    invalid_ids = set()
    warning_ids = set()

    for idx, constraint in enumerate(constraints, start=1):
        cid = _constraint_id(constraint, fallback=f"C{idx}")
        if not has_attached_steps(constraint):
            report.errors.append((cid, "Constraint missing attached steps"))
            invalid_ids.add(cid)
            continue
        base_passed.append(cid)

    for cid, message in validate_temporal_consistency(constraints):
        report.errors.append((cid, message))
        invalid_ids.add(cid)

    for cid, message in validate_parameter_ranges(constraints):
        report.errors.append((cid, message))
        invalid_ids.add(cid)

    for cid, message in validate_safety_types(constraints):
        warning_ids.add(cid)
        report.warnings.append((cid, message))

    report.passed = [cid for cid in base_passed if cid not in invalid_ids and cid not in warning_ids]
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _constraint_id(constraint: Dict[str, object], fallback: str = "unknown") -> str:
    cid = constraint.get("id")
    if cid is None:
        return fallback
    return str(cid)


def _extract_refs(constraint: Dict[str, object]) -> List[str]:
    keys = ("attached_to", "steps", "targets")
    refs: List[str] = []
    for key in keys:
        values = constraint.get(key)
        if not values:
            continue
        if not isinstance(values, list):
            values = [values]
        for value in values:
            if isinstance(value, dict):
                value = value.get("id")
            if value:
                refs.append(str(value))
    return refs


def _detect_frequency(constraint: Dict[str, object]) -> str | None:
    text = _constraint_text(constraint).lower()
    for freq, keywords in TEMPORAL_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return freq
    return None


def _constraint_text(constraint: Dict[str, object]) -> str:
    return (
        str(constraint.get("text") or "")
        or str(constraint.get("expression") or "")
        or str(constraint.get("description") or "")
    )


__all__ = [
    "ValidationReport",
    "has_attached_steps",
    "validate_temporal_consistency",
    "validate_safety_types",
    "validate_parameter_ranges",
    "validate_constraints",
]
