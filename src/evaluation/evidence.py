"""Evidence eligibility metadata for research annotations."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema

PENDING_HUMAN_SIGN_OFF = "(pending human sign-off)"
HUMAN_VERIFIED_RE = re.compile(
    r"(?:^|\s)\+\s*human-verified:([A-Za-z0-9][A-Za-z0-9._-]*)"
)
EVIDENCE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "ipke_annotation_evidence.schema.json"
)
ANNOTATION_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "ipke_annotation.schema.json"
)

ArtifactLoader = Callable[[str], bytes]


@dataclass(frozen=True, slots=True)
class AnnotationEvidence:
    declared_reviewed: bool
    human_verified: bool
    evidence_eligible: bool
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProductionEvidence:
    human_pass_recorded: bool
    anchors_complete: bool
    hashes_match: bool
    evidence_eligible: bool
    issues: tuple[str, ...]


def assess_annotation_evidence(annotation: Mapping[str, Any]) -> AnnotationEvidence:
    raw_quality = annotation.get("quality")
    quality = raw_quality if isinstance(raw_quality, Mapping) else {}
    declared_reviewed = quality.get("review_status") == "reviewed"
    raw_annotator = quality.get("annotator")
    annotator = raw_annotator.strip() if isinstance(raw_annotator, str) else ""
    review_date = quality.get("review_date")
    pending = PENDING_HUMAN_SIGN_OFF in annotator.lower()
    human_verified = bool(HUMAN_VERIFIED_RE.search(annotator)) and not pending

    issues: list[str] = []
    if not declared_reviewed:
        issues.append("quality.review_status must be 'reviewed'")
    if not annotator:
        issues.append("quality.annotator missing")
    elif pending:
        issues.append("quality.annotator still contains pending human sign-off")
    elif not human_verified:
        issues.append("quality.annotator lacks a + human-verified:<handle> marker")
    if not review_date:
        issues.append("quality.review_date missing")

    return AnnotationEvidence(
        declared_reviewed=declared_reviewed,
        human_verified=human_verified,
        evidence_eligible=not issues,
        issues=tuple(issues),
    )


@lru_cache(maxsize=1)
def _evidence_validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(EVIDENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    )


@lru_cache(maxsize=1)
def _annotation_validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(ANNOTATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _schema_issues(evidence_log: Mapping[str, Any]) -> list[str]:
    errors = sorted(
        _evidence_validator().iter_errors(evidence_log),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    issues: list[str] = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        issues.append(f"evidence log schema at {location}: {error.message}")
    return issues


def _annotation_schema_issues(annotation: Mapping[str, Any]) -> list[str]:
    errors = sorted(
        _annotation_validator().iter_errors(annotation),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    issues: list[str] = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        issues.append(f"annotation schema at {location}: {error.message}")
    return issues


def _integer_offset(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _span_issues(
    label: str,
    provenance: Any,
    *,
    doc_id: str,
    procedure_start: int,
    procedure_end: int,
    source_text: str,
) -> list[str]:
    if not isinstance(provenance, Mapping):
        return [f"{label} provenance missing"]

    issues: list[str] = []
    if provenance.get("doc_id") != doc_id:
        issues.append(f"{label} provenance.doc_id does not match procedure")
    start = provenance.get("char_start")
    end = provenance.get("char_end")
    if not _integer_offset(start) or not _integer_offset(end):
        issues.append(f"{label} provenance requires integer char_start and char_end")
        return issues
    if not 0 <= start < end <= len(source_text):
        issues.append(f"{label} provenance span is outside source text")
        return issues
    if start < procedure_start or end > procedure_end:
        issues.append(f"{label} provenance span is outside procedure source span")
    if not source_text[start:end].strip():
        issues.append(f"{label} provenance span contains only whitespace")
    return issues


def _annotation_anchor_issues(
    annotation: Mapping[str, Any], source_text: str
) -> list[str]:
    raw_procedure = annotation.get("procedure")
    procedure = raw_procedure if isinstance(raw_procedure, Mapping) else {}
    raw_doc_id = procedure.get("doc_id")
    doc_id = raw_doc_id if isinstance(raw_doc_id, str) else ""
    source = procedure.get("source")
    if not isinstance(source, Mapping):
        return ["procedure.source provenance missing"]

    issues: list[str] = []
    if source.get("doc_id") != doc_id:
        issues.append("procedure.source.doc_id does not match procedure.doc_id")
    start = source.get("char_start")
    end = source.get("char_end")
    if not _integer_offset(start) or not _integer_offset(end):
        return issues + [
            "procedure.source requires integer char_start and char_end"
        ]
    if not 0 <= start < end <= len(source_text):
        return issues + ["procedure.source span is outside source text"]
    if not source_text[start:end].strip():
        issues.append("procedure.source span contains only whitespace")

    raw_steps = annotation.get("steps")
    steps = raw_steps if isinstance(raw_steps, list) else []
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            issues.append(f"step:{index} is not an object")
            continue
        step_id = raw_step.get("id", index)
        step_label = f"step:{step_id}"
        issues.extend(
            _span_issues(
                step_label,
                raw_step.get("provenance"),
                doc_id=doc_id,
                procedure_start=start,
                procedure_end=end,
                source_text=source_text,
            )
        )
        embedded = raw_step.get("constraints", []) or []
        if not isinstance(embedded, list):
            issues.append(f"{step_label} constraints must be an array in production")
            continue
        for constraint_index, raw_constraint in enumerate(embedded):
            if not isinstance(raw_constraint, Mapping):
                issues.append(
                    f"{step_label}/constraint:{constraint_index} is not an object"
                )
                continue
            constraint_id = raw_constraint.get("id", constraint_index)
            issues.extend(
                _span_issues(
                    f"{step_label}/constraint:{constraint_id}",
                    raw_constraint.get("provenance"),
                    doc_id=doc_id,
                    procedure_start=start,
                    procedure_end=end,
                    source_text=source_text,
                )
            )

    top_constraints = annotation.get("constraints", []) or []
    if not isinstance(top_constraints, list):
        issues.append("top-level constraints must be an array in production")
        return issues
    for index, raw_constraint in enumerate(top_constraints):
        if not isinstance(raw_constraint, Mapping):
            issues.append(f"constraint:{index} is not an object")
            continue
        constraint_id = raw_constraint.get("id", index)
        issues.extend(
            _span_issues(
                f"constraint:{constraint_id}",
                raw_constraint.get("provenance"),
                doc_id=doc_id,
                procedure_start=start,
                procedure_end=end,
                source_text=source_text,
            )
        )
    return issues


def _coerce_refs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _duplicate_issue(label: str, values: list[str]) -> str | None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if not duplicates:
        return None
    return f"duplicate {label} IDs: {', '.join(duplicates)}"


def _annotation_integrity_issues(
    annotation: Mapping[str, Any],
    *,
    expected_doc_id: str | None = None,
) -> list[str]:
    issues = _annotation_schema_issues(annotation)
    raw_procedure = annotation.get("procedure")
    procedure = raw_procedure if isinstance(raw_procedure, Mapping) else {}
    doc_id = procedure.get("doc_id")
    if expected_doc_id is not None and doc_id != expected_doc_id:
        issues.append("annotation doc_id does not match selected filename")

    raw_steps = annotation.get("steps")
    steps = raw_steps if isinstance(raw_steps, list) else []
    step_ids = [
        step["id"]
        for step in steps
        if isinstance(step, Mapping) and isinstance(step.get("id"), str)
    ]
    duplicate_steps = _duplicate_issue("step", step_ids)
    if duplicate_steps:
        issues.append(duplicate_steps)
    valid_step_ids = set(step_ids)

    constraint_ids: list[str] = []
    constraints: list[tuple[str, Mapping[str, Any]]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            continue
        step_id = step.get("id", index)
        raw_constraints = step.get("constraints")
        if isinstance(raw_constraints, list):
            for constraint_index, constraint in enumerate(raw_constraints):
                if not isinstance(constraint, Mapping):
                    continue
                constraint_id = constraint.get("id", constraint_index)
                constraints.append((f"step:{step_id}/constraint:{constraint_id}", constraint))
                if isinstance(constraint.get("id"), str):
                    constraint_ids.append(constraint["id"])
    raw_top_constraints = annotation.get("constraints")
    if isinstance(raw_top_constraints, list):
        for index, constraint in enumerate(raw_top_constraints):
            if not isinstance(constraint, Mapping):
                continue
            constraint_id = constraint.get("id", index)
            constraints.append((f"constraint:{constraint_id}", constraint))
            if isinstance(constraint.get("id"), str):
                constraint_ids.append(constraint["id"])
    duplicate_constraints = _duplicate_issue("constraint", constraint_ids)
    if duplicate_constraints:
        issues.append(duplicate_constraints)

    for label, constraint in constraints:
        refs: list[str] = []
        for field in ("attached_to", "applies_to"):
            refs.extend(_coerce_refs(constraint.get(field)))
        if not refs:
            issues.append(f"{label} has no step attachment")
        for ref in refs:
            if ref not in valid_step_ids:
                issues.append(f"{label} references unknown step {ref}")

    raw_relations = annotation.get("relations")
    relations = raw_relations if isinstance(raw_relations, list) else []
    relation_ids = [
        relation["id"]
        for relation in relations
        if isinstance(relation, Mapping) and isinstance(relation.get("id"), str)
    ]
    duplicate_relations = _duplicate_issue("relation", relation_ids)
    if duplicate_relations:
        issues.append(duplicate_relations)
    for index, relation in enumerate(relations):
        if not isinstance(relation, Mapping):
            continue
        relation_label = f"relation:{relation.get('id', index)}"
        source_target = (relation.get("source"), relation.get("target"))
        from_to = (relation.get("from"), relation.get("to"))
        if not (
            all(isinstance(value, str) for value in source_target)
            or all(isinstance(value, str) for value in from_to)
        ):
            issues.append(f"{relation_label} requires source and target step endpoints")
        for field in ("from", "to", "source", "target"):
            endpoint = relation.get(field)
            if isinstance(endpoint, str) and endpoint not in valid_step_ids:
                issues.append(
                    f"{relation_label} {field} references unknown step {endpoint}"
                )
    return issues


def _artifact_path_issue(path: Any, expected: str, label: str) -> str | None:
    if not isinstance(path, str):
        return f"{label} artifact path missing"
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or "." in parsed.parts or ".." in parsed.parts:
        return f"{label} artifact path is not canonical"
    if path != expected:
        return f"{label} artifact path does not match selected document"
    return None


def _load_and_verify_artifact(
    artifact: Any,
    *,
    expected_path: str,
    label: str,
    artifact_loader: ArtifactLoader | None,
) -> tuple[bytes | None, list[str]]:
    if not isinstance(artifact, Mapping):
        return None, [f"{label} artifact record missing"]
    path_issue = _artifact_path_issue(artifact.get("path"), expected_path, label)
    issues = [path_issue] if path_issue else []
    if path_issue or artifact_loader is None:
        if artifact_loader is None:
            issues.append(f"{label} artifact loader missing")
        return None, issues
    try:
        content = artifact_loader(expected_path)
    except (OSError, KeyError) as exc:
        return None, [*issues, f"{label} artifact cannot be read: {exc}"]
    digest = hashlib.sha256(content).hexdigest()
    if artifact.get("sha256") != digest:
        issues.append(f"{label} artifact SHA-256 does not match evidence log")
    return content, issues


def _parse_annotation_artifact(
    content: bytes | None,
    *,
    label: str,
    expected_doc_id: str,
) -> tuple[Mapping[str, Any] | None, list[str]]:
    if content is None:
        return None, []
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, [f"{label} artifact is not valid JSON: {exc}"]
    if not isinstance(parsed, Mapping):
        return None, [f"{label} artifact must contain a JSON object"]
    return parsed, [
        f"{label} {issue}" for issue in _annotation_integrity_issues(
            parsed,
            expected_doc_id=expected_doc_id,
        )
    ]


def _decision_span_issues(
    primary_pass: Mapping[str, Any],
    *,
    procedure_start: int,
    procedure_end: int,
    source_text: str,
) -> list[str]:
    issues: list[str] = []
    for group_name in ("decisions", "unresolved_decisions"):
        raw_records = primary_pass.get(group_name)
        records = raw_records if isinstance(raw_records, list) else []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                continue
            decision_id = record.get("decision_id", index)
            raw_spans = record.get("evidence_spans")
            spans = raw_spans if isinstance(raw_spans, list) else []
            for span in spans:
                if not isinstance(span, Mapping):
                    continue
                start = span.get("char_start")
                end = span.get("char_end")
                label = f"primary decision {decision_id} evidence span"
                if not _integer_offset(start) or not _integer_offset(end):
                    issues.append(f"{label} requires integer offsets")
                    continue
                if not 0 <= start < end <= len(source_text):
                    issues.append(f"{label} is outside source text")
                    continue
                if start < procedure_start or end > procedure_end:
                    issues.append(f"{label} is outside procedure source span")
                if not source_text[start:end].strip():
                    issues.append(f"{label} contains only whitespace")
    return issues


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _timing_issues(evidence_log: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    raw_primary = evidence_log.get("primary_pass")
    primary = raw_primary if isinstance(raw_primary, Mapping) else {}
    raw_blind = evidence_log.get("blind_pass")
    blind = raw_blind if isinstance(raw_blind, Mapping) else {}
    raw_adjudication = evidence_log.get("adjudication")
    adjudication = raw_adjudication if isinstance(raw_adjudication, Mapping) else {}
    timestamp_fields: list[tuple[str, Any]] = [
        ("primary_pass.started_at", primary.get("started_at")),
        ("primary_pass.completed_at", primary.get("completed_at")),
        ("frozen_at", evidence_log.get("frozen_at")),
    ]
    if evidence_log.get("blind_subset_selected") is True:
        timestamp_fields.extend(
            [
                ("blind_pass.started_at", blind.get("started_at")),
                ("blind_pass.completed_at", blind.get("completed_at")),
                ("adjudication.started_at", adjudication.get("started_at")),
                ("adjudication.completed_at", adjudication.get("completed_at")),
            ]
        )
    for location, value in timestamp_fields:
        if isinstance(value, str) and _parse_timestamp(value) is None:
            issues.append(f"evidence log {location} is not a valid date-time")
    primary_start = _parse_timestamp(primary.get("started_at"))
    primary_end = _parse_timestamp(primary.get("completed_at"))
    frozen_at = _parse_timestamp(evidence_log.get("frozen_at"))
    if primary_start is not None and primary_end is not None:
        if primary_end < primary_start:
            issues.append("primary completion precedes primary start")
        active_minutes = primary.get("active_minutes")
        elapsed_minutes = (primary_end - primary_start).total_seconds() / 60
        if isinstance(active_minutes, int) and active_minutes > elapsed_minutes:
            issues.append("primary active minutes exceed elapsed review time")
    if primary_end is not None and frozen_at is not None and frozen_at < primary_end:
        issues.append("evidence frozen_at precedes primary completion")
    if evidence_log.get("blind_subset_selected") is True:
        blind_start = _parse_timestamp(blind.get("started_at"))
        blind_end = _parse_timestamp(blind.get("completed_at"))
        adjudication_start = _parse_timestamp(adjudication.get("started_at"))
        adjudication_end = _parse_timestamp(adjudication.get("completed_at"))
        if blind_start is not None and blind_end is not None and blind_end < blind_start:
            issues.append("blind completion precedes blind start")
        if adjudication_start is not None and adjudication_end is not None:
            if adjudication_end < adjudication_start:
                issues.append("adjudication completion precedes adjudication start")
            completed_passes = [
                value for value in (primary_end, blind_end) if value is not None
            ]
            if completed_passes and adjudication_start < max(completed_passes):
                issues.append("adjudication starts before annotation passes are frozen")
        if (
            adjudication_end is not None
            and frozen_at is not None
            and frozen_at < adjudication_end
        ):
            issues.append("evidence frozen_at precedes adjudication completion")
    return issues


def _annotation_item_ids(annotation: Mapping[str, Any]) -> dict[str, set[str]]:
    step_ids: set[str] = set()
    constraint_ids: set[str] = set()
    raw_steps = annotation.get("steps")
    if isinstance(raw_steps, list):
        for step in raw_steps:
            if not isinstance(step, Mapping):
                continue
            if isinstance(step.get("id"), str):
                step_ids.add(step["id"])
            constraints = step.get("constraints")
            if isinstance(constraints, list):
                constraint_ids.update(
                    constraint["id"]
                    for constraint in constraints
                    if isinstance(constraint, Mapping)
                    and isinstance(constraint.get("id"), str)
                )
    top_constraints = annotation.get("constraints")
    if isinstance(top_constraints, list):
        constraint_ids.update(
            constraint["id"]
            for constraint in top_constraints
            if isinstance(constraint, Mapping)
            and isinstance(constraint.get("id"), str)
        )
    return {"step": step_ids, "constraint": constraint_ids}


def _decision_issues(
    annotation: Mapping[str, Any],
    primary_pass: Mapping[str, Any],
    *,
    candidate_ids: Mapping[str, set[str]],
) -> list[str]:
    raw_decisions = primary_pass.get("decisions")
    decisions = raw_decisions if isinstance(raw_decisions, list) else []
    raw_counts = primary_pass.get("decision_counts")
    counts = raw_counts if isinstance(raw_counts, Mapping) else {}
    final_ids = _annotation_item_ids(annotation)
    issues: list[str] = []

    decision_ids = [
        decision["decision_id"]
        for decision in decisions
        if isinstance(decision, Mapping)
        and isinstance(decision.get("decision_id"), str)
    ]
    duplicate_decisions = _duplicate_issue("primary decision", decision_ids)
    if duplicate_decisions:
        issues.append(duplicate_decisions)

    for item_kind, count_key in (("step", "steps"), ("constraint", "constraints")):
        kind_decisions = [
            decision
            for decision in decisions
            if isinstance(decision, Mapping)
            and decision.get("item_kind") == item_kind
        ]
        actual = {action: 0 for action in ("accepted", "edited", "rejected", "added")}
        input_ids: list[str] = []
        output_ids: list[str] = []
        for decision in kind_decisions:
            action = decision.get("action")
            raw_inputs = decision.get("input_ids")
            raw_outputs = decision.get("output_ids")
            decision_inputs = raw_inputs if isinstance(raw_inputs, list) else []
            decision_outputs = raw_outputs if isinstance(raw_outputs, list) else []
            input_ids.extend(value for value in decision_inputs if isinstance(value, str))
            output_ids.extend(
                value for value in decision_outputs if isinstance(value, str)
            )
            if action == "accept":
                actual["accepted"] += len(decision_inputs)
            elif action == "edit":
                actual["edited"] += len(decision_inputs)
            elif action == "reject":
                actual["rejected"] += len(decision_inputs)
            elif action == "add":
                actual["added"] += len(decision_outputs)

        declared_raw = counts.get(count_key)
        declared = declared_raw if isinstance(declared_raw, Mapping) else {}
        if any(declared.get(key) != value for key, value in actual.items()):
            issues.append(
                f"primary {item_kind} decision counts do not match decision records"
            )
        candidate_count = declared.get("candidate_count")
        expected_candidate_ids = candidate_ids[item_kind]
        if (
            candidate_count != len(expected_candidate_ids)
            or set(input_ids) != expected_candidate_ids
            or len(input_ids) != len(set(input_ids))
        ):
            issues.append(
                f"primary {item_kind} candidate decisions are incomplete or duplicated"
            )
        if len(output_ids) != len(set(output_ids)):
            issues.append(f"primary {item_kind} output decisions are duplicated")
        if set(output_ids) != final_ids[item_kind]:
            issues.append(
                f"primary {item_kind} decisions do not cover final annotation IDs"
            )
        if declared.get("final_count") != len(final_ids[item_kind]):
            issues.append(
                f"primary {item_kind} final count does not match annotation"
            )
    return issues


def assess_production_evidence(
    annotation: Mapping[str, Any],
    *,
    annotation_bytes: bytes,
    source_bytes: bytes,
    evidence_log: Mapping[str, Any] | None,
    expected_doc_id: str | None = None,
    artifact_loader: ArtifactLoader | None = None,
) -> ProductionEvidence:
    issues: list[str] = []
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError:
        source_text = ""
        issues.append("source text is not valid UTF-8")

    integrity_issues = _annotation_integrity_issues(
        annotation,
        expected_doc_id=expected_doc_id,
    )
    issues.extend(integrity_issues)

    quality_raw = annotation.get("quality")
    quality = quality_raw if isinstance(quality_raw, Mapping) else {}
    if quality.get("review_status") != "reviewed":
        issues.append("quality.review_status must be 'reviewed'")
    raw_annotator = quality.get("annotator")
    annotator = raw_annotator.strip() if isinstance(raw_annotator, str) else ""
    if not annotator:
        issues.append("quality.annotator missing")
    elif PENDING_HUMAN_SIGN_OFF in annotator.lower():
        issues.append("quality.annotator still contains pending human sign-off")
    if not quality.get("review_date"):
        issues.append("quality.review_date missing")

    anchor_issues = _annotation_anchor_issues(annotation, source_text)
    issues.extend(anchor_issues)

    if evidence_log is None:
        issues.append("production evidence log missing")
        return ProductionEvidence(
            human_pass_recorded=False,
            anchors_complete=not anchor_issues,
            hashes_match=False,
            evidence_eligible=False,
            issues=tuple(issues),
        )

    schema_issues = _schema_issues(evidence_log)
    issues.extend(schema_issues)
    issues.extend(_timing_issues(evidence_log))
    raw_procedure = annotation.get("procedure")
    procedure = raw_procedure if isinstance(raw_procedure, Mapping) else {}
    doc_id = procedure.get("doc_id")
    if evidence_log.get("doc_id") != doc_id:
        issues.append("evidence log doc_id does not match annotation")
    selected_doc_id = expected_doc_id or (doc_id if isinstance(doc_id, str) else "")
    if evidence_log.get("status") != "frozen":
        issues.append("production evidence log must be frozen")

    raw_primary = evidence_log.get("primary_pass")
    primary = raw_primary if isinstance(raw_primary, Mapping) else {}
    raw_reviewer = primary.get("reviewer")
    reviewer = raw_reviewer if isinstance(raw_reviewer, Mapping) else {}
    reviewer_handle = reviewer.get("handle")
    marker = HUMAN_VERIFIED_RE.search(annotator)
    marker_handle = marker.group(1) if marker else None
    annotator_matches = reviewer_handle in {annotator, marker_handle}
    human_pass_recorded = (
        evidence_log.get("status") == "frozen"
        and reviewer.get("kind") == "human"
        and reviewer.get("role") == "primary_reviewer"
        and isinstance(reviewer_handle, str)
        and annotator_matches
    )
    if isinstance(reviewer_handle, str) and not annotator_matches:
        issues.append("primary reviewer does not match quality.annotator")
    unresolved = primary.get("unresolved_decisions")
    if isinstance(unresolved, list) and unresolved:
        issues.append("primary pass has unresolved decisions")

    raw_source = evidence_log.get("source")
    logged_source = raw_source if isinstance(raw_source, Mapping) else {}
    raw_output = evidence_log.get("output")
    logged_output = raw_output if isinstance(raw_output, Mapping) else {}
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    annotation_digest = hashlib.sha256(annotation_bytes).hexdigest()
    hash_issues: list[str] = []
    expected_source_path = f"datasets/paper/text/{selected_doc_id}.txt"
    source_path_issue = _artifact_path_issue(
        logged_source.get("path"),
        expected_source_path,
        "source",
    )
    if source_path_issue:
        hash_issues.append(source_path_issue)
    if logged_source.get("sha256") != source_digest:
        hash_issues.append("source SHA-256 does not match evidence log")
    expected_output_path = f"datasets/paper/production/{selected_doc_id}.json"
    output_path_issue = _artifact_path_issue(
        logged_output.get("path"),
        expected_output_path,
        "production output",
    )
    if output_path_issue:
        hash_issues.append(output_path_issue)
    if logged_output.get("sha256") != annotation_digest:
        hash_issues.append("annotation SHA-256 does not match evidence log")

    source_provenance_raw = procedure.get("source")
    source_provenance = (
        source_provenance_raw
        if isinstance(source_provenance_raw, Mapping)
        else {}
    )
    start = source_provenance.get("char_start")
    end = source_provenance.get("char_end")
    if logged_source.get("char_start") != start or logged_source.get("char_end") != end:
        hash_issues.append("source span does not match evidence log")
    if _integer_offset(start) and _integer_offset(end) and 0 <= start < end <= len(source_text):
        span_digest = hashlib.sha256(source_text[start:end].encode("utf-8")).hexdigest()
        if logged_source.get("span_sha256") != span_digest:
            hash_issues.append("source span SHA-256 does not match evidence log")

    candidate_ids: Mapping[str, set[str]] = {"step": set(), "constraint": set()}
    raw_assistance = primary.get("assistance")
    assistance = raw_assistance if isinstance(raw_assistance, Mapping) else {}
    if assistance.get("candidate_used") is True:
        expected_candidate_path = (
            f"datasets/paper/review_candidates/{selected_doc_id}.json"
        )
        candidate_bytes, candidate_artifact_issues = _load_and_verify_artifact(
            evidence_log.get("candidate"),
            expected_path=expected_candidate_path,
            label="candidate",
            artifact_loader=artifact_loader,
        )
        hash_issues.extend(candidate_artifact_issues)
        candidate, candidate_parse_issues = _parse_annotation_artifact(
            candidate_bytes,
            label="candidate",
            expected_doc_id=selected_doc_id,
        )
        hash_issues.extend(candidate_parse_issues)
        if candidate is not None:
            hash_issues.extend(
                f"candidate {issue}"
                for issue in _annotation_anchor_issues(candidate, source_text)
            )
            candidate_ids = _annotation_item_ids(candidate)

    expected_primary_path = (
        f"datasets/paper/primary_pass/{selected_doc_id}.json"
    )
    primary_bytes, primary_artifact_issues = _load_and_verify_artifact(
        primary.get("output"),
        expected_path=expected_primary_path,
        label="primary output",
        artifact_loader=artifact_loader,
    )
    hash_issues.extend(primary_artifact_issues)
    primary_annotation, primary_parse_issues = _parse_annotation_artifact(
        primary_bytes,
        label="primary output",
        expected_doc_id=selected_doc_id,
    )
    hash_issues.extend(primary_parse_issues)
    if primary_annotation is not None:
        hash_issues.extend(
            f"primary output {issue}"
            for issue in _annotation_anchor_issues(
                primary_annotation,
                source_text,
            )
        )
    blind_selected = evidence_log.get("blind_subset_selected") is True
    if (
        not blind_selected
        and primary_bytes is not None
        and primary_bytes != annotation_bytes
    ):
        hash_issues.append(
            "primary output bytes do not match unadjudicated production annotation"
        )

    source_procedure_start = (
        start if _integer_offset(start) else 0
    )
    source_procedure_end = (
        end if _integer_offset(end) else len(source_text)
    )
    issues.extend(
        _decision_span_issues(
            primary,
            procedure_start=source_procedure_start,
            procedure_end=source_procedure_end,
            source_text=source_text,
        )
    )
    issues.extend(
        _decision_issues(
            primary_annotation or annotation,
            primary,
            candidate_ids=candidate_ids,
        )
    )

    if blind_selected:
        raw_blind = evidence_log.get("blind_pass")
        blind = raw_blind if isinstance(raw_blind, Mapping) else {}
        raw_adjudication = evidence_log.get("adjudication")
        adjudication = (
            raw_adjudication if isinstance(raw_adjudication, Mapping) else {}
        )
        blind_reviewer_raw = blind.get("reviewer")
        blind_reviewer = (
            blind_reviewer_raw if isinstance(blind_reviewer_raw, Mapping) else {}
        )
        adjudicator_raw = adjudication.get("reviewer")
        adjudicator = (
            adjudicator_raw if isinstance(adjudicator_raw, Mapping) else {}
        )
        handles = {
            reviewer_handle,
            blind_reviewer.get("handle"),
            adjudicator.get("handle"),
        }
        if None in handles or len(handles) != 3:
            issues.append("primary, blind, and adjudication actors must be distinct")

        expected_blind_path = (
            f"datasets/paper/second_pass/{selected_doc_id}.json"
        )
        blind_bytes, blind_artifact_issues = _load_and_verify_artifact(
            blind.get("output"),
            expected_path=expected_blind_path,
            label="blind output",
            artifact_loader=artifact_loader,
        )
        hash_issues.extend(blind_artifact_issues)
        blind_annotation, blind_parse_issues = _parse_annotation_artifact(
            blind_bytes,
            label="blind output",
            expected_doc_id=selected_doc_id,
        )
        hash_issues.extend(blind_parse_issues)
        if blind_annotation is not None:
            hash_issues.extend(
                f"blind output {issue}"
                for issue in _annotation_anchor_issues(
                    blind_annotation,
                    source_text,
                )
            )

        expected_report_path = (
            f"datasets/paper/reports/{selected_doc_id}_agreement.json"
        )
        _, agreement_artifact_issues = _load_and_verify_artifact(
            evidence_log.get("agreement_report"),
            expected_path=expected_report_path,
            label="agreement report",
            artifact_loader=artifact_loader,
        )
        hash_issues.extend(agreement_artifact_issues)

        adjudication_bytes, adjudication_artifact_issues = (
            _load_and_verify_artifact(
                adjudication.get("output"),
                expected_path=expected_output_path,
                label="adjudication output",
                artifact_loader=artifact_loader,
            )
        )
        hash_issues.extend(adjudication_artifact_issues)
        if adjudication_bytes is not None and adjudication_bytes != annotation_bytes:
            hash_issues.append(
                "adjudication output bytes do not match production annotation"
            )

    issues.extend(hash_issues)

    return ProductionEvidence(
        human_pass_recorded=human_pass_recorded,
        anchors_complete=not anchor_issues,
        hashes_match=not hash_issues,
        evidence_eligible=not issues,
        issues=tuple(issues),
    )
