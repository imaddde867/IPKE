"""Validate and normalize public paper gold annotations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import jsonschema


ATTACHMENT_FIELDS = ("steps", "attached_to", "applies_to", "targets")
RELATION_ENDPOINT_FIELDS = ("from", "to", "source", "target")


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() == "true"


def natural_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def selected_gold_document_ids(manifest: Path) -> list[str]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        return [
            row["document_id"]
            for row in csv.DictReader(handle)
            if truthy(row.get("selected_for_gold"))
        ]


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


def _constraint_text(constraint: dict[str, Any]) -> str:
    for field in ("text", "expression", "condition", "statement"):
        value = constraint.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _iter_constraint_dicts(annotation: dict[str, Any]) -> Iterable[tuple[dict[str, Any], str | None]]:
    for constraint in annotation.get("constraints") or []:
        if isinstance(constraint, dict):
            yield constraint, None

    for step in annotation.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_id = step.get("id") if isinstance(step.get("id"), str) else None
        nested = step.get("constraints")
        if isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    yield item, step_id
        elif isinstance(nested, dict):
            for value in nested.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item, step_id
                elif isinstance(value, str) and value.strip():
                    yield {"text": value}, step_id


def validate_annotation_links(annotation: dict[str, Any], document_id: str | None = None) -> None:
    """Validate cross-field links that JSON Schema cannot express."""
    procedure = annotation.get("procedure", {})
    if document_id is not None and procedure.get("doc_id") != document_id:
        raise ValueError(
            f"procedure.doc_id {procedure.get('doc_id')!r} does not match {document_id!r}"
        )

    steps = annotation.get("steps", [])
    step_ids = {step["id"] for step in steps if isinstance(step, dict) and "id" in step}

    for constraint, containing_step_id in _iter_constraint_dicts(annotation):
        constraint_id = constraint.get("id") or _constraint_text(constraint) or "<unknown>"
        refs: list[str] = []
        for field in ATTACHMENT_FIELDS:
            refs.extend(_coerce_refs(constraint.get(field)))
        if not refs and containing_step_id:
            refs = [containing_step_id]
        for ref in refs:
            if ref not in step_ids:
                raise ValueError(
                    f"constraint {constraint_id!r} references unknown step {ref!r}"
                )

    relations = annotation.get("relations")
    relation_items: list[dict[str, Any]] = []
    if isinstance(relations, list):
        relation_items = [item for item in relations if isinstance(item, dict)]
    elif isinstance(relations, dict):
        for value in relations.values():
            if isinstance(value, list):
                relation_items.extend(item for item in value if isinstance(item, dict))

    for relation in relation_items:
        for field in RELATION_ENDPOINT_FIELDS:
            value = relation.get(field)
            if isinstance(value, str) and value.startswith("S") and value not in step_ids:
                raise ValueError(f"relation field {field!r} references unknown step {value!r}")


def normalize_annotation(annotation: dict[str, Any], document_id: str) -> dict[str, Any]:
    validate_annotation_links(annotation, document_id)
    normalized = dict(annotation)
    normalized["steps"] = sorted(
        annotation["steps"],
        key=lambda step: natural_key(str(step.get("id", ""))),
    )
    if isinstance(annotation.get("constraints"), list):
        normalized["constraints"] = sorted(
            annotation["constraints"],
            key=lambda constraint: natural_key(str(constraint.get("id", ""))),
        )
    return normalized


def normalize_gold_annotations(schema: Path, manifest: Path, in_dir: Path, out_dir: Path) -> None:
    schema_doc = json.loads(schema.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema_doc)
    out_dir.mkdir(parents=True, exist_ok=True)

    for document_id in selected_gold_document_ids(manifest):
        source = in_dir / f"{document_id}.json"
        if not source.exists():
            raise SystemExit(f"{document_id}: missing annotation {source}")

        annotation = json.loads(source.read_text(encoding="utf-8"))
        validator.validate(annotation)
        normalized = normalize_annotation(annotation, document_id)

        target = out_dir / f"{document_id}.json"
        target.write_text(
            json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"{document_id}: normalized")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--in-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    normalize_gold_annotations(args.schema, args.manifest, args.in_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
