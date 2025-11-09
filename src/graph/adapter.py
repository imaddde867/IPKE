"""
Adapter to convert flat baseline predictions (steps, constraints, entities)
into Tier-B graph shape expected by evaluate.py:

- nodes: list of { id, type: "step"|"condition"|"equipment"|"parameter", ... }
- edges: list of { source, target, type: lowercase relation name }

Notes
- We keep the current baseline flat extractor unchanged; this is a non-invasive utility.
- Edge types are lowercased for the evaluator (e.g., "next", "condition_on").
- For now, we emit:
  - step nodes from `steps`
  - condition nodes from `constraints`
  - "next" edges between consecutive steps
  - "condition_on" edges from each condition to referenced step(s) when available
- Equipment/parameter nodes can be added later when entity typing is reliable.
"""
from __future__ import annotations

from typing import Any, Dict, List


LOWER_REL_MAP = {
    "NEXT": "next",
    "CONDITION_ON": "condition_on",
    "USES": "uses",
    "HAS_PARAMETER": "has_parameter",
    "REQUIRES": "requires",
    "PRODUCES": "produces",
    "REFERENCES": "references",
    "ALTERNATIVE_TO": "alternative_to",
}


def _norm_id(value: Any) -> str:
    return str(value) if value is not None else ""


def flat_to_tierb(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a flat prediction payload (as produced by baseline.extraction_payload)
    into the Tier-B format expected by evaluate.py.

    Input shape (subset):
      {
        "document_id": str,
        "steps": [{"id": "S1", "text": "...", "order": 1, ...}, ...],
        "constraints": [{"id": "C1", "text": "...", "steps": ["S1"], ...}, ...],
        "entities": [...],
        ...
      }

    Output shape:
      {
        "document_id": str,
        "nodes": [
          {"id": "S1", "type": "step", "text": "..."},
          {"id": "C1", "type": "condition", "text": "..."},
          ...
        ],
        "edges": [
          {"source": "S1", "target": "S2", "type": "next"},
          {"source": "C1", "target": "S1", "type": "condition_on"},
          ...
        ]
      }
    """
    document_id = str(doc.get("document_id", ""))
    steps: List[Dict[str, Any]] = list(doc.get("steps", []) or [])
    constraints: List[Dict[str, Any]] = list(doc.get("constraints", []) or [])

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # Step nodes (preserve text for label extraction in evaluator)
    for s in steps:
        sid = _norm_id(s.get("id")) or f"S{len(nodes)+1}"
        nodes.append({
            "id": sid,
            "type": "step",
            "text": (s.get("text") or s.get("description") or "").strip(),
            "order": s.get("order"),
        })

    # Condition nodes
    for c in constraints:
        cid = _norm_id(c.get("id")) or f"C{len(nodes)+1}"
        nodes.append({
            "id": cid,
            "type": "condition",
            "text": (c.get("text") or c.get("description") or "").strip(),
        })

    # Build a simple step index for NEXT edges and constraint attachments
    step_ids: List[str] = [n["id"] for n in nodes if n.get("type") == "step"]

    # NEXT edges from step order (sequential)
    for i in range(len(step_ids) - 1):
        edges.append({
            "source": step_ids[i],
            "target": step_ids[i + 1],
            "type": "next",
        })

    # condition_on edges if constraint references exist
    # We read references from constraint["steps"] which may be list[str] or list[dict]
    constraint_nodes = [n for n in nodes if n.get("type") == "condition"]
    constraint_by_id = {n["id"]: n for n in constraint_nodes}

    for c in constraints:
        cid = _norm_id(c.get("id"))
        if not cid or cid not in constraint_by_id:
            continue
        raw_refs = c.get("steps") or c.get("attached_to") or c.get("scope") or []
        if isinstance(raw_refs, dict):
            raw_refs = [raw_refs]
        if isinstance(raw_refs, str):
            raw_refs = [raw_refs]
        for ref in raw_refs:
            if isinstance(ref, dict):
                sid = _norm_id(ref.get("id"))
            else:
                sid = _norm_id(ref)
            if sid in step_ids:
                edges.append({
                    "source": cid,
                    "target": sid,
                    "type": "condition_on",
                })

    return {
        "document_id": document_id,
        "nodes": nodes,
        "edges": edges,
    }
