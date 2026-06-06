#!/usr/bin/env python3
"""Draft a gold annotation file from a source text document.

Runs the extraction pipeline (P3 two-stage prompting, backend from the
environment) on the given text file, converts the output to the flat
ipke_annotation schema, saves a draft under datasets/paper/gold_drafts/,
validates it against schemas/ipke_annotation.schema.json, and prints all
validation errors to stdout so they can be corrected by hand.

The draft is saved regardless of validation result so partial output can be
reviewed. Human correction happens outside this tool; do not commit drafts as
reviewed gold.

Usage:
    uv run python tools/annotate_gold.py <doc_path> [--doc-id DOC_ID]
                                                     [--out-dir DIR]

Options:
    doc_path        Path to the source text file.
    --doc-id        Override the document id (default: stem of the input file).
    --out-dir       Directory to write the draft (default: datasets/paper/gold_drafts/).
    --title         Procedure title written into the draft.
    --domain        Domain tag written into the draft.
    --skip-model    Validate a pre-existing draft without re-running extraction.
                    When set, doc_path is treated as an existing JSON draft.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import jsonschema
except ImportError:
    print("ERROR: jsonschema not installed. Run: uv pip install jsonschema", file=sys.stderr)
    sys.exit(1)

SCHEMA_PATH = REPO_ROOT / "schemas" / "ipke_annotation.schema.json"


def _load_schema() -> dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Annotation schema not found: {SCHEMA_PATH}")
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validate(draft: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(draft), key=lambda e: list(e.path))
    return [f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors]


def _extraction_to_draft(
    doc_id: str,
    payload: dict[str, Any],
    title: str,
    domain: str,
) -> dict[str, Any]:
    """Convert a pipeline extraction payload to the ipke_annotation schema.

    Input ``payload`` has the shape produced by ``extraction_payload()`` in
    ``src/pipelines/baseline.py``:
      steps: [{id, text, order, ...}]
      constraints: [{id, text, steps: [step_id], ...}]

    Output matches the gold annotation schema:
      procedure: {doc_id, title, domain}
      steps: [{id, label, action_verb, constraints: []}]
      constraints: []   (top-level, schema-canonical flat list NOT used here;
                         constraints remain nested per step for annotation)
    """
    raw_steps: list[dict[str, Any]] = payload.get("steps") or []
    raw_constraints: list[dict[str, Any]] = payload.get("constraints") or []

    # Build a step-id -> list-of-constraint index for nesting.
    # Constraints with no step reference are placed in an __orphan__ bucket so
    # the caller can warn the user rather than silently losing them.
    step_constraint_index: dict[str, list[dict[str, Any]]] = {}
    orphan_constraints: list[str] = []
    for i, c in enumerate(raw_constraints):
        cid = c.get("id") or f"C{i + 1}"
        refs: list[str] = []
        for key in ("steps", "attached_to", "applies_to", "targets", "step_id"):
            val = c.get(key)
            if isinstance(val, str):
                refs = [val]
                break
            if isinstance(val, list):
                refs = [v for v in val if isinstance(v, str)]
                break
        text = c.get("text") or c.get("condition") or c.get("statement") or ""
        entry: dict[str, Any] = {
            "id": cid,
            "type": c.get("type", "guard"),
            "text": text,
        }
        if refs:
            entry["attached_to"] = refs
            for sid in refs:
                step_constraint_index.setdefault(sid, []).append(entry)
        else:
            orphan_constraints.append(cid)

    draft_steps: list[dict[str, Any]] = []
    for i, s in enumerate(raw_steps):
        sid = s.get("id") or f"S{i + 1}"
        label = s.get("text") or s.get("label") or s.get("description") or sid
        draft_step: dict[str, Any] = {
            "id": sid,
            "label": label,
        }
        verb = s.get("action_verb")
        if verb:
            draft_step["action_verb"] = verb
        nested = step_constraint_index.get(sid, [])
        draft_step["constraints"] = nested
        if s.get("order") is not None:
            draft_step["provenance"] = {"order": s["order"]}
        draft_steps.append(draft_step)

    if orphan_constraints:
        print(
            f"WARNING: {len(orphan_constraints)} constraint(s) had no step reference "
            f"and were dropped from draft: {orphan_constraints}",
            file=__import__("sys").stderr,
        )

    return {
        "procedure": {
            "doc_id": doc_id,
            "title": title or doc_id,
            "version": None,
            "domain": domain or "unknown",
            "source": {"doc_id": doc_id},
        },
        "steps": draft_steps,
        "constraints": [],
        "relations": [],
        "quality": {
            "annotator": "draft_llm",
            "status": "unreviewed",
        },
    }


async def _run_extraction(doc_path: Path, doc_id: str) -> dict[str, Any]:
    from src.core.unified_config import reload_config
    from src.pipelines.baseline import extraction_payload
    from src.processors.streamlined_processor import StreamlinedDocumentProcessor

    # Default to P3 + DSC for annotation drafting unless caller overrides via env.
    os.environ.setdefault("PROMPTING_STRATEGY", "P3")
    os.environ.setdefault("CHUNKING_METHOD", "dsc")
    config = reload_config()
    processor = StreamlinedDocumentProcessor(config=config)
    result = await processor.process_document(file_path=str(doc_path), document_id=doc_id)
    return extraction_payload(doc_id, result)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("doc_path", type=Path, help="Source text file (or existing draft JSON if --skip-model).")
    parser.add_argument("--doc-id", default=None, help="Document id (default: file stem).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "paper" / "gold_drafts",
        help="Output directory for the draft.",
    )
    parser.add_argument("--title", default="", help="Procedure title for the draft.")
    parser.add_argument("--domain", default="", help="Domain tag for the draft.")
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Validate an existing draft JSON instead of running extraction.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    doc_path = args.doc_path.resolve()
    doc_id = args.doc_id or doc_path.stem

    if not doc_path.exists():
        print(f"ERROR: path not found: {doc_path}", file=sys.stderr)
        return 1

    schema = _load_schema()

    if args.skip_model:
        if args.title or args.domain:
            print("NOTE: --title/--domain ignored in --skip-model mode (draft loaded as-is)")
        print(f"Loading existing draft: {doc_path}")
        draft = json.loads(doc_path.read_text(encoding="utf-8"))
    else:
        print(f"Running P3+DSC extraction on: {doc_path}")
        print(f"Document id: {doc_id}")
        payload = asyncio.run(_run_extraction(doc_path, doc_id))
        n_steps = len(payload.get("steps") or [])
        n_constraints = len(payload.get("constraints") or [])
        print(f"Extraction: {n_steps} steps, {n_constraints} constraints")
        draft = _extraction_to_draft(doc_id, payload, args.title, args.domain)

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc_id}.json"
    out_path.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Draft saved: {out_path}")

    errors = _validate(draft, schema)
    if errors:
        print(f"\nVALIDATION: {len(errors)} error(s) — correct before promoting to gold/")
        for err in errors:
            print(f"  {err}")
        return 2
    else:
        print("\nVALIDATION: PASS — draft conforms to ipke_annotation schema")
        return 0


if __name__ == "__main__":
    sys.exit(main())
