"""Unit tests for src/validation/constraint_validator.py."""
from __future__ import annotations

from src.validation.constraint_validator import (
    ValidationReport,
    has_attached_steps,
    validate_constraints,
    validate_parameter_ranges,
    validate_safety_types,
    validate_temporal_consistency,
)


def test_has_attached_steps_true_when_steps_present():
    assert has_attached_steps({"id": "C1", "steps": ["S1"]}) is True


def test_has_attached_steps_true_for_attached_to_alias():
    assert has_attached_steps({"id": "C1", "attached_to": ["S2"]}) is True


def test_has_attached_steps_false_when_no_refs():
    assert has_attached_steps({"id": "C1", "text": "Wear gloves"}) is False


def test_validate_temporal_consistency_flags_conflict():
    constraints = [
        {"id": "C1", "steps": ["S1"], "text": "Inspect daily"},
        {"id": "C2", "steps": ["S1"], "text": "Inspect weekly"},
    ]
    conflicts = validate_temporal_consistency(constraints)
    assert len(conflicts) >= 1


def test_validate_temporal_consistency_no_conflict_same_frequency():
    constraints = [
        {"id": "C1", "steps": ["S1"], "text": "Inspect daily"},
        {"id": "C2", "steps": ["S1"], "text": "Test daily"},
    ]
    assert validate_temporal_consistency(constraints) == []


def test_validate_safety_types_flags_missing_hazard_keyword():
    # A constraint typed as "safety" without any hazard keyword should be flagged.
    constraints = [{"id": "C1", "type": "safety", "steps": ["S1"], "text": "Follow the procedure"}]
    warnings = validate_safety_types(constraints)
    assert len(warnings) >= 1
    assert all(isinstance(w, tuple) and len(w) == 2 for w in warnings)


def test_validate_parameter_ranges_returns_list():
    constraints = [{"id": "C1", "steps": ["S1"], "text": "Maintain 20-25 degrees C"}]
    result = validate_parameter_ranges(constraints)
    assert isinstance(result, list)


def test_validate_constraints_report_has_expected_shape():
    constraints = [
        {"id": "C1", "steps": ["S1"], "text": "Maintain temperature"},
        {"id": "C2", "text": "Orphan constraint with no step attachment"},
    ]
    report = validate_constraints(constraints)
    assert isinstance(report, ValidationReport)
    assert isinstance(report.passed, list)
    assert isinstance(report.warnings, list)
    assert isinstance(report.errors, list)


def test_validate_constraints_to_dict_is_serialisable():
    report = validate_constraints([{"id": "C1", "steps": ["S1"], "text": "Wear PPE"}])
    d = report.to_dict()
    assert "passed" in d
    assert "warnings" in d
    assert "errors" in d
