"""Export the source-bound static research console; no model or optional dependencies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_ID = "epa_field_operations_manual_filter_sampling_sop"


def read_json(root: Path, path: str) -> dict:
    return json.loads((root / path).read_text(encoding="utf-8"))


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
    for item in items:
        p = item["provenance"]
        if p["doc_id"] != DOC_ID or not start <= p["char_start"] < p["char_end"] <= end:
            raise ValueError(f"Invalid source span: {item['id']}")
        quote = source[p["char_start"]:p["char_end"]]
        if not quote.strip() or ("text" in item and quote != item["text"]):
            raise ValueError(f"Source text mismatch: {item['id']}")
    for c in constraints:
        targets = c.get("attached_to", c.get("applies_to", []))
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
    return {
        "schema_version": 1, "mode": "static_snapshot", "human_verified": False,
        "status": status, "input_sha256": hashes,
        "source": {"text": excerpt, "char_start": start, "char_end": end,
                   "offset_unit": "Unicode code points; end exclusive"},
        "annotation": annotation, "legacy": legacy, "constraints": constraints,
        "review_packet": packet, "corpus": corpus,
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
