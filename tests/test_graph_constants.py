"""Tests for src/graph/constants.py — shared constants and utilities."""

from __future__ import annotations

from src.graph.constants import (
    STEP_REF_KEYS,
    LOWER_REL_MAP,
    ALLOWED_CONDITION_TYPES,
    flatten_refs,
    collect_step_refs,
    normalize_id,
)


class TestConstants:
    def test_step_ref_keys_contains_expected(self):
        assert "step" in STEP_REF_KEYS
        assert "steps" in STEP_REF_KEYS
        assert "attached_to" in STEP_REF_KEYS
        assert "applies_to" in STEP_REF_KEYS
        assert "scope" in STEP_REF_KEYS
        assert "targets" in STEP_REF_KEYS

    def test_step_ref_keys_is_tuple(self):
        assert isinstance(STEP_REF_KEYS, tuple)

    def test_lower_rel_map_contains_core_types(self):
        assert LOWER_REL_MAP["NEXT"] == "next"
        assert LOWER_REL_MAP["CONDITION_ON"] == "condition_on"
        assert LOWER_REL_MAP["USES"] == "uses"
        assert LOWER_REL_MAP["HAS_PARAMETER"] == "has_parameter"

    def test_allowed_condition_types_is_set(self):
        assert isinstance(ALLOWED_CONDITION_TYPES, set)
        assert "precondition" in ALLOWED_CONDITION_TYPES
        assert "postcondition" in ALLOWED_CONDITION_TYPES
        assert "safety" in ALLOWED_CONDITION_TYPES
        assert "exception" in ALLOWED_CONDITION_TYPES


class TestNormalizeId:
    def test_returns_stripped_string(self):
        assert normalize_id("  S1  ") == "S1"

    def test_returns_empty_for_none(self):
        assert normalize_id(None) == ""

    def test_returns_empty_for_empty_string(self):
        assert normalize_id("") == ""

    def test_converts_int_to_string(self):
        assert normalize_id(42) == "42"

    def test_converts_float_to_string(self):
        assert normalize_id(3.14) == "3.14"


class TestFlattenRefs:
    def test_string_returns_single_element_list(self):
        assert flatten_refs("S1") == ["S1"]

    def test_none_returns_empty_list(self):
        assert flatten_refs(None) == []

    def test_empty_string_returns_empty_list(self):
        assert flatten_refs("") == []

    def test_dict_with_id_extracts_id(self):
        assert flatten_refs({"id": "S1"}) == ["S1"]

    def test_dict_with_step_id_extracts_step_id(self):
        assert flatten_refs({"step_id": "S2"}) == ["S2"]

    def test_dict_with_step_extracts_step(self):
        assert flatten_refs({"step": "S3"}) == ["S3"]

    def test_dict_prefers_id_over_step_id(self):
        assert flatten_refs({"id": "S1", "step_id": "S2"}) == ["S1"]

    def test_list_of_strings(self):
        assert flatten_refs(["S1", "S2", "S3"]) == ["S1", "S2", "S3"]

    def test_list_of_dicts(self):
        result = flatten_refs([{"id": "S1"}, {"id": "S2"}])
        assert result == ["S1", "S2"]

    def test_nested_lists(self):
        assert flatten_refs([["S1", "S2"], "S3"]) == ["S1", "S2", "S3"]

    def test_set_of_strings(self):
        assert sorted(flatten_refs({"S1", "S2"})) == sorted(["S1", "S2"])

    def test_non_string_non_dict(self):
        assert flatten_refs(True) == ["True"]

    def test_dict_values_scanned_when_id_missing(self):
        d = {"a": {"id": "S1"}, "b": {"step": "S2"}}
        result = flatten_refs(d)
        assert "S1" in result
        assert "S2" in result


class TestCollectStepRefs:
    def test_collects_from_step_key(self):
        assert collect_step_refs({"step": "S1"}) == ["S1"]

    def test_collects_from_steps_key(self):
        assert collect_step_refs({"steps": "S1"}) == ["S1"]

    def test_collects_from_attached_to(self):
        assert collect_step_refs({"attached_to": "S1"}) == ["S1"]

    def test_collects_from_multiple_keys(self):
        result = collect_step_refs({"step": "S1", "attached_to": "S2"})
        assert sorted(result) == sorted(["S1", "S2"])

    def test_empty_record_returns_empty_list(self):
        assert collect_step_refs({}) == []

    def test_none_values_are_skipped(self):
        assert collect_step_refs({"step": None}) == []
