"""Evidence eligibility metadata for research annotations."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PENDING_HUMAN_SIGN_OFF = "(pending human sign-off)"
HUMAN_VERIFIED_RE = re.compile(
    r"(?:^|\s)\+\s*human-verified:([A-Za-z0-9][A-Za-z0-9._-]*)"
)


@dataclass(frozen=True, slots=True)
class AnnotationEvidence:
    declared_reviewed: bool
    human_verified: bool
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
