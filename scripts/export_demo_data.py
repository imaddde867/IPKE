"""Export the source-bound static research console; no model or optional dependencies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_ID = "epa_field_operations_manual_filter_sampling_sop"


def read_json(root: Path, path: str) -> dict:
    return json.loads((root / path).read_text(encoding="utf-8"))


def _decision_signature(item: dict, kind: str) -> tuple:
    if kind == "step":
        return item.get("label"), item.get("action_verb"), item.get("action_object")
    return item.get("text"), item.get("type"), item.get("enforcement")


def _decision_terms(item: dict, kind: str) -> set[str]:
    fields = ("label", "action_verb", "action_object") if kind == "step" else ("text",)
    return {
        term
        for field in fields
        for term in re.findall(r"[a-z0-9]+", str(item.get(field) or "").lower())
    }


def _validate_decisions(
    decisions: list[dict],
    *,
    kind: str,
    legacy_items: dict[str, dict],
    review_items: dict[str, dict],
) -> None:
    seen_input_ids: list[str] = []
    seen_output_ids: list[str] = []
    for decision in decisions:
        action = decision.get("action")
        input_refs = decision.get("input_refs", [])
        output_refs = decision.get("output_refs", [])
        for refs, label in ((input_refs, "input"), (output_refs, "output")):
            if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
                raise ValueError(f"Decision {label} references must be strings")
            if len(refs) != len(set(refs)):
                raise ValueError(f"Duplicate decision {label} reference")

        inputs: list[dict] = []
        for ref in input_refs:
            prefix, _, item_id = ref.partition(":")
            if prefix != "legacy" or item_id not in legacy_items:
                raise ValueError(f"Unresolved decision input reference: {ref}")
            inputs.append(legacy_items[item_id])
            seen_input_ids.append(item_id)

        outputs: list[dict] = []
        for ref in output_refs:
            prefix, _, item_id = ref.partition(":")
            if prefix != "review" or item_id not in review_items:
                raise ValueError(f"Unresolved decision output reference: {ref}")
            outputs.append(review_items[item_id])
            seen_output_ids.append(item_id)

        if action == "accept":
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("Accept decisions require exactly one input and one output")
            if _decision_signature(inputs[0], kind) != _decision_signature(outputs[0], kind):
                raise ValueError("Accepted item does not match its review output")
        elif action == "edit":
            if not inputs or not outputs:
                raise ValueError("Edit decisions require input and output items")
            input_terms = [_decision_terms(item, kind) for item in inputs]
            output_terms = [_decision_terms(item, kind) for item in outputs]
            if any(not terms for terms in input_terms + output_terms):
                raise ValueError("Edit decision items must contain text for correspondence checks")
            if any(not terms.intersection(set.union(*output_terms)) for terms in input_terms):
                raise ValueError("Edit decision input has no corresponding output")
            if any(not terms.intersection(set.union(*input_terms)) for terms in output_terms):
                raise ValueError("Edit decision output has no corresponding input")
        elif action == "reject":
            if not inputs or outputs:
                raise ValueError("Reject decisions require input items and no outputs")
        elif action == "add":
            if inputs or not outputs:
                raise ValueError("Add decisions require output items and no inputs")
        else:
            raise ValueError(f"Unknown decision action: {action!r}")

    if set(seen_input_ids) != set(legacy_items) or len(seen_input_ids) != len(legacy_items):
        raise ValueError(f"Decision inputs do not cover each legacy {kind} exactly once")
    if set(seen_output_ids) != set(review_items) or len(seen_output_ids) != len(review_items):
        raise ValueError(f"Decision outputs do not cover each review {kind} exactly once")


def build_data(root: Path = ROOT) -> dict:
    packet = read_json(root, f"datasets/paper/review_packets/{DOC_ID}.json")
    hashes = {}
    for key in ("source", "audit", "legacy_candidate", "review_candidate"):
        artifact = packet[key]
        digest = hashlib.sha256((root / artifact["path"]).read_bytes()).hexdigest()
        if digest != artifact["sha256"]:
            raise ValueError(f"Review packet hash mismatch: {key}")
        hashes[artifact["path"]] = digest
    annotation = read_json(root, packet["review_candidate"]["path"])
    legacy = read_json(root, packet["legacy_candidate"]["path"])
    source = (root / packet["source"]["path"]).read_text(encoding="utf-8")
    start, end = packet["source"]["char_start"], packet["source"]["char_end"]
    excerpt = source[start:end]
    if hashlib.sha256(excerpt.encode()).hexdigest() != packet["source"]["span_sha256"]:
        raise ValueError("Source excerpt hash mismatch")
    constraints = [c for s in annotation["steps"] for c in s["constraints"]]
    constraints += annotation["constraints"]
    steps = {s["id"] for s in annotation["steps"]}
    items = annotation["steps"] + constraints
    if len({i["id"] for i in items}) != len(items):
        raise ValueError("Duplicate item IDs")
    legacy_constraints = [c for s in legacy["steps"] for c in s.get("constraints", [])]
    legacy_constraints += legacy.get("constraints", [])
    legacy_items = {
        "step": {item["id"]: item for item in legacy["steps"]},
        "constraint": {item["id"]: item for item in legacy_constraints},
    }
    review_items = {
        "step": {item["id"]: item for item in annotation["steps"]},
        "constraint": {item["id"]: item for item in constraints},
    }
    for kind in ("step", "constraint"):
        _validate_decisions(
            packet[f"proposed_{kind}_decisions"],
            kind=kind,
            legacy_items=legacy_items[kind],
            review_items=review_items[kind],
        )
    for item in items:
        p = item["provenance"]
        if p["doc_id"] != DOC_ID or not start <= p["char_start"] < p["char_end"] <= end:
            raise ValueError(f"Invalid source span: {item['id']}")
        quote = source[p["char_start"]:p["char_end"]]
        if not quote.strip() or ("text" in item and quote != item["text"]):
            raise ValueError(f"Source text mismatch: {item['id']}")
    for c in constraints:
        targets = []
        for field in ("attached_to", "applies_to"):
            refs = c.get(field)
            if refs is None:
                continue
            if isinstance(refs, str):
                refs = [refs]
            if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
                raise ValueError(f"Invalid {field} references: {c['id']}")
            targets.extend(refs)
        if not targets or not set(targets) <= steps:
            raise ValueError(f"Broken attachment: {c['id']}")
    for edge in annotation["relations"]:
        if edge["source"] not in steps or edge["target"] not in steps:
            raise ValueError("Broken relation")
    expected_spine = [
        {"source": a["id"], "target": b["id"], "type": "NEXT"}
        for a, b in zip(annotation["steps"], annotation["steps"][1:])
    ]
    if annotation["relations"] != expected_spine:
        raise ValueError("Console requires a linear NEXT spine")
    for label, candidate in (("legacy_candidate", legacy), ("review_candidate", annotation)):
        count = sum(len(s["constraints"]) for s in candidate["steps"]) + len(candidate["constraints"])
        if (len(candidate["steps"]), count) != (packet[label]["step_count"], packet[label]["constraint_count"]):
            raise ValueError(f"Packet counts mismatch: {label}")
    manifest_path = "datasets/paper/corpus_manifest.json"
    manifest = read_json(root, manifest_path)
    hashes[manifest_path] = hashlib.sha256((root / manifest_path).read_bytes()).hexdigest()
    status = next(d for d in manifest["documents"] if d["doc_id"] == DOC_ID)
    # Fail closed: this slice represents an agent candidate, never human evidence.
    if status["status"] != "candidate" or annotation["artifact_role"] != "agent_review_candidate" or packet["status"] != "prepared_not_human_evidence":
        raise ValueError("Annotation evidence status changed; review console contract")
    with (root / "datasets/paper/public_sources_manifest.csv").open(encoding="utf-8", newline="") as f:
        sources = {r["document_id"]: r for r in csv.DictReader(f)}
    corpus = [{**d, "title": sources[d["doc_id"]]["title"],
               "source_url": sources[d["doc_id"]]["direct_url"],
               "usage": sources[d["doc_id"]]["license_or_usage"]} for d in manifest["documents"]]
    verified = sum(d["status"] == "human_verified" for d in manifest["documents"])
    return {
        "schema_version": 1, "mode": "static_snapshot", "human_verified": False,
        "status": status, "input_sha256": hashes,
        "source": {"text": excerpt, "char_start": start, "char_end": end,
                   "offset_unit": "Unicode code points; end exclusive"},
        "annotation": annotation, "legacy": legacy, "constraints": constraints,
        "review_packet": packet, "corpus": corpus,
        "corpus_summary": {"documents": len(manifest["documents"]),
                           "human_verified": verified},
        "summary": {"kind": "authored_illustration_not_model_output",
                    "text": "Warm up the instruments, connect the calibration equipment, and record the initial display. Generate calibration flows and record the measurements. Enter the computed calibration values, adjust the set point, record the result, and prepare the controller for shipment.",
                    "omitted_constraint_ids": [c["id"] for c in constraints]},
        "replay": {"available": False, "reason": "No complete recorded inference trace with independently supported stage timings is bundled. This is an annotation snapshot, not an extraction replay."},
    }


def serialize(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if committed export differs")
    args = parser.parse_args()
    path = ROOT / "demo/data/epa.json"
    content = serialize(build_data())
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise SystemExit("Demo export is stale; run python scripts/export_demo_data.py")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
