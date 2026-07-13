"""Paper-grade validator for IPKE-Bench gold files.

Stricter than schema-level validation. Enforces:
  - quality.review_status == "reviewed"
  - quality.annotator and quality.review_date set
  - all constraints use the locked type vocabulary (6 values)
  - all constraints have enforcement in {must, should, may}
  - all constraints have non-empty text
  - every step has at least one constraint (warn, not fail)
  - every constraint attaches to at least one valid step id

Usage:
    uv run python scripts/validate_paper_gold.py [--gold-dir datasets/paper/gold] [--strict]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import jsonschema
from pydantic import ValidationError

from src.benchmark.taxonomy import LOCKED_CONSTRAINT_TYPES, LOCKED_ENFORCEMENT_LEVELS
from src.evaluation.corpus_manifest import (
    load_corpus_manifest,
    select_manifest_gold_files,
    select_manifest_production_files,
)
from src.evaluation.evidence import (
    ArtifactLoader,
    assess_annotation_evidence,
    assess_production_evidence,
)

LOCKED_TYPES = LOCKED_CONSTRAINT_TYPES
LOCKED_ENFORCEMENT = LOCKED_ENFORCEMENT_LEVELS
REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_SCHEMA_PATH = REPO_ROOT / "schemas" / "ipke_annotation.schema.json"


@lru_cache(maxsize=1)
def _annotation_validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(ANNOTATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _annotation_schema_errors(annotation: dict) -> list[str]:
    errors = sorted(
        _annotation_validator().iter_errors(annotation),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    messages: list[str] = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        messages.append(f"annotation schema at {location}: {error.message}")
    return messages


def _artifact_loader_for_source(source_path: Path) -> ArtifactLoader:
    resolved_source = source_path.resolve()
    relative_source = Path("datasets/paper/text") / source_path.name
    artifact_root = REPO_ROOT
    for parent in resolved_source.parents:
        if (parent / relative_source).resolve() == resolved_source:
            artifact_root = parent
            break

    def load(path: str) -> bytes:
        return (artifact_root / path).read_bytes()

    return load


def iter_constraints(annotation: dict) -> Iterable[tuple[str, str, dict]]:
    for step in annotation.get("steps", []):
        sid = step.get("id", "?")
        for c in step.get("constraints", []) or []:
            yield f"step:{sid}", c.get("id", "?"), c
    for c in annotation.get("constraints", []) or []:
        yield "top", c.get("id", "?"), c


def _adjudicates_warning(review_notes: str, step_id: str, warning_kind: str) -> bool:
    marker = f"step:{step_id} {warning_kind} adjudicated"
    return marker in review_notes


def validate_file(
    path: Path,
    *,
    require_human_verified: bool = False,
    require_production_evidence: bool = False,
    source_path: Path | None = None,
    evidence_path: Path | None = None,
    artifact_loader: ArtifactLoader | None = None,
) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        annotation_bytes = path.read_bytes()
        d = json.loads(annotation_bytes)
    except Exception as e:
        return [f"ERROR: JSON parse error: {e}"]

    schema_errors = _annotation_schema_errors(d)
    if schema_errors:
        return [f"ERROR: {error}" for error in schema_errors]

    quality = d.get("quality", {})
    review_notes = str(quality.get("review_notes") or "")
    if quality.get("review_status") != "reviewed":
        errors.append(f"quality.review_status != 'reviewed' (got {quality.get('review_status')!r})")
    if not quality.get("annotator"):
        errors.append("quality.annotator missing")
    if not quality.get("review_date"):
        errors.append("quality.review_date missing")
    if require_human_verified:
        evidence = assess_annotation_evidence(d)
        if not evidence.human_verified:
            errors.extend(
                issue
                for issue in evidence.issues
                if "human" in issue or "pending human sign-off" in issue
            )
    if require_production_evidence:
        if source_path is None:
            errors.append("production source path missing")
        elif not source_path.is_file():
            errors.append(f"production source file missing: {source_path}")
        elif source_path.stem != path.stem:
            errors.append("production source filename does not match annotation filename")
        if evidence_path is None:
            errors.append("production evidence log path missing")
        elif not evidence_path.is_file():
            errors.append(f"production evidence log file missing: {evidence_path}")
        elif evidence_path.stem != path.stem:
            errors.append("production evidence filename does not match annotation filename")
        if (
            source_path is not None
            and source_path.is_file()
            and evidence_path is not None
            and evidence_path.is_file()
        ):
            try:
                source_bytes = source_path.read_bytes()
                evidence_log = json.loads(evidence_path.read_bytes())
            except Exception as exc:
                errors.append(f"production evidence read error: {exc}")
            else:
                production = assess_production_evidence(
                    d,
                    annotation_bytes=annotation_bytes,
                    source_bytes=source_bytes,
                    evidence_log=evidence_log,
                    expected_doc_id=path.stem,
                    artifact_loader=(
                        artifact_loader or _artifact_loader_for_source(source_path)
                    ),
                )
                errors.extend(
                    issue for issue in production.issues if issue not in errors
                )

    step_ids = {s.get("id") for s in d.get("steps", []) if s.get("id")}
    if not step_ids:
        errors.append("no step ids")
        return errors

    for loc, cid, c in iter_constraints(d):
        prefix = f"{loc}/{cid}"
        t = c.get("type")
        if t not in LOCKED_TYPES:
            errors.append(f"{prefix}: type={t!r} not in locked vocabulary")
        enf = c.get("enforcement")
        if enf not in LOCKED_ENFORCEMENT:
            errors.append(f"{prefix}: enforcement={enf!r} not in {{must,should,may}}")
        if not c.get("text"):
            errors.append(f"{prefix}: empty text")
        refs = []
        for field in ("attached_to", "applies_to"):
            v = c.get(field)
            if isinstance(v, str):
                refs.append(v)
            elif isinstance(v, list):
                refs.extend(v)
        if not refs:
            errors.append(f"{prefix}: no attached_to or applies_to")
        else:
            for r in refs:
                if r not in step_ids:
                    errors.append(f"{prefix}: ref {r!r} not in step ids")

    # Count procedure-level constraints whose applies_to includes each step.
    step_procedure_counts: dict[str, int] = {sid: 0 for sid in step_ids}
    for c in d.get("constraints", []) or []:
        applies = c.get("applies_to")
        if isinstance(applies, str):
            applies = [applies]
        if isinstance(applies, list):
            for sid in applies:
                if sid in step_procedure_counts:
                    step_procedure_counts[sid] += 1

    for s in d.get("steps", []):
        sid = s.get("id", "?")
        embedded = len(s.get("constraints", []) or [])
        procedure = step_procedure_counts.get(sid, 0)
        total = embedded + procedure
        if total == 0:
            if not _adjudicates_warning(review_notes, sid, "zero_constraints"):
                warnings.append(
                    f"step:{sid}: 0 attached constraints (embedded={embedded}, procedure-level={procedure}); "
                    "re-read source"
                )
        elif total > 10:
            if not _adjudicates_warning(review_notes, sid, "too_many_constraints"):
                warnings.append(
                    f"step:{sid}: {total} constraints (embedded={embedded}, procedure-level={procedure}); "
                    "consider splitting step"
                )

    return [f"ERROR: {e}" for e in errors] + [f"WARN: {w}" for w in warnings]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-dir", type=Path, default=Path("datasets/paper/gold"))
    ap.add_argument("--text-dir", type=Path)
    ap.add_argument("--evidence-dir", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    ap.add_argument(
        "--require-frozen-manifest",
        action="store_true",
        help="Require the supplied corpus manifest to be frozen.",
    )
    ap.add_argument(
        "--require-human-verified",
        action="store_true",
        help="Require the legacy non-pending + human-verified:<handle> marker.",
    )
    ap.add_argument(
        "--require-production-evidence",
        action="store_true",
        help=(
            "Require exact item anchors and a frozen human-evidence sidecar. "
            "Also requires --text-dir and --evidence-dir."
        ),
    )
    args = ap.parse_args(argv)

    if args.require_frozen_manifest and args.manifest is None:
        print("ERROR: --require-frozen-manifest requires --manifest")
        return 1
    if args.require_production_evidence:
        if args.text_dir is None or args.evidence_dir is None:
            print(
                "ERROR: --require-production-evidence requires "
                "--text-dir and --evidence-dir"
            )
            return 1
        for directory in (args.text_dir, args.evidence_dir):
            if not directory.is_dir():
                print(f"ERROR: directory not found: {directory}")
                return 1

    manifest_error = False
    gold_files = tuple(sorted(args.gold_dir.glob("*.json")))
    if args.manifest is not None:
        try:
            manifest = load_corpus_manifest(args.manifest)
            selector = (
                select_manifest_production_files
                if args.require_production_evidence
                else select_manifest_gold_files
            )
            gold_files = selector(manifest, args.gold_dir)
        except (OSError, ValueError, ValidationError) as exc:
            print(f"ERROR: invalid corpus manifest: {exc}")
            return 1
        if manifest.manifest_status == "provisional":
            if args.require_frozen_manifest:
                print("ERROR: corpus manifest is provisional; frozen manifest required")
                manifest_error = True
            else:
                print("WARN: corpus manifest is provisional")

    any_error = manifest_error
    any_warn = False
    for f in gold_files:
        msgs = validate_file(
            f,
            require_human_verified=args.require_human_verified,
            require_production_evidence=args.require_production_evidence,
            source_path=(
                args.text_dir / f"{f.stem}.txt" if args.text_dir is not None else None
            ),
            evidence_path=(
                args.evidence_dir / f"{f.stem}.json"
                if args.evidence_dir is not None
                else None
            ),
        )
        errs = [m for m in msgs if m.startswith("ERROR")]
        warns = [m for m in msgs if m.startswith("WARN")]
        if errs:
            any_error = True
            print(f"FAIL {f.name}")
            for m in errs:
                print(f"  {m}")
        if warns:
            any_warn = True
            for m in warns:
                print(f"  {f.name}: {m}")
        if not errs and not warns:
            print(f"PASS {f.name}")
    if any_error:
        return 1
    if any_warn and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
