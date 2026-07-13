from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema

from scripts.normalize_gold_annotations import validate_annotation_links


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_epa_review_candidate_is_exactly_source_grounded() -> None:
    path = Path(
        "datasets/paper/review_candidates/"
        "epa_field_operations_manual_filter_sampling_sop.json"
    )
    annotation = _load_json(path)
    schema = _load_json(Path("schemas/ipke_annotation.schema.json"))
    jsonschema.Draft202012Validator(schema).validate(annotation)
    validate_annotation_links(
        annotation, "epa_field_operations_manual_filter_sampling_sop"
    )

    source = Path(
        "datasets/paper/text/epa_field_operations_manual_filter_sampling_sop.txt"
    ).read_text(encoding="utf-8")
    procedure_span = annotation["procedure"]["source"]
    assert source[
        procedure_span["char_start"] : procedure_span["char_end"]
    ].startswith("6.0\nCALIBRATION / POST-CALIBRATION")
    assert len(annotation["steps"]) == 14

    constraints = list(annotation["constraints"])
    for step in annotation["steps"]:
        provenance = step["provenance"]
        assert (
            procedure_span["char_start"]
            <= provenance["char_start"]
            < provenance["char_end"]
            <= procedure_span["char_end"]
        )
        assert source[provenance["char_start"] : provenance["char_end"]].strip()
        constraints.extend(step["constraints"])

    assert len(constraints) == 15
    for constraint in constraints:
        provenance = constraint["provenance"]
        assert constraint["text"] == source[
            provenance["char_start"] : provenance["char_end"]
        ]


def test_epa_review_packet_hashes_and_decisions_reconcile() -> None:
    packet = _load_json(
        Path(
            "datasets/paper/review_packets/"
            "epa_field_operations_manual_filter_sampling_sop.json"
        )
    )
    packet_schema = _load_json(Path("schemas/ipke_review_packet.schema.json"))
    jsonschema.Draft202012Validator(packet_schema).validate(packet)
    assert packet["decision_semantics"]["counts_are_human_effort_evidence"] is False

    for key in ("source", "audit", "legacy_candidate", "review_candidate"):
        artifact = packet[key]
        assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == (
            artifact["sha256"]
        )

    source_text = Path(packet["source"]["path"]).read_text(encoding="utf-8")
    source = packet["source"]
    assert hashlib.sha256(
        source_text[source["char_start"] : source["char_end"]].encode("utf-8")
    ).hexdigest() == source["span_sha256"]

    annotation = _load_json(Path(packet["review_candidate"]["path"]))
    legacy = _load_json(Path(packet["legacy_candidate"]["path"]))
    final_ids = {
        "steps": {step["id"] for step in annotation["steps"]},
        "constraints": {
            constraint["id"] for constraint in annotation["constraints"]
        }
        | {
            constraint["id"]
            for step in annotation["steps"]
            for constraint in step["constraints"]
        },
    }
    legacy_ids = {
        "steps": {step["id"] for step in legacy["steps"]},
        "constraints": {
            constraint["id"] for constraint in legacy["constraints"]
        }
        | {
            constraint["id"]
            for step in legacy["steps"]
            for constraint in step["constraints"]
        },
    }
    for kind in ("step", "constraint"):
        decisions = packet[f"proposed_{kind}_decisions"]
        counts = packet["proposed_counts"][f"{kind}s"]
        input_refs = [item for decision in decisions for item in decision["input_refs"]]
        output_refs = [
            item for decision in decisions for item in decision["output_refs"]
        ]
        assert all(ref.startswith("legacy:") for ref in input_refs)
        assert all(ref.startswith("review:") for ref in output_refs)
        input_ids = {ref.removeprefix("legacy:") for ref in input_refs}
        output_ids = {ref.removeprefix("review:") for ref in output_refs}
        assert input_ids == legacy_ids[f"{kind}s"]
        assert len(input_refs) == len(input_ids) == counts["candidate_count"]
        assert output_ids == final_ids[f"{kind}s"]
        assert len(output_refs) == len(output_ids) == counts["final_count"]
        assert counts["accepted"] == sum(
            len(decision["input_refs"])
            for decision in decisions
            if decision["action"] == "accept"
        )
        assert counts["edited"] == sum(
            len(decision["input_refs"])
            for decision in decisions
            if decision["action"] == "edit"
        )
        assert counts["rejected"] == sum(
            len(decision["input_refs"])
            for decision in decisions
            if decision["action"] == "reject"
        )
        assert counts["added"] == sum(
            len(decision["output_refs"])
            for decision in decisions
            if decision["action"] == "add"
        )
