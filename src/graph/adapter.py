"""
Adapter to convert flat baseline predictions (steps, constraints, entities)
into the Tier-B graph shape expected by evaluate.py:

- nodes: list of { id, type: "step"|"condition"|"equipment"|"parameter", ... }
- edges: list of { source, target, type: lowercase relation name }

Notes
- We keep the current baseline flat extractor unchanged; this is a non-invasive utility.
- Edge types are lowercased for the evaluator (e.g., "next", "condition_on").
- Emitted structure now includes:
  - step nodes from `steps`
  - condition nodes from `constraints`
  - equipment + parameter nodes derived from either top-level lists or step fields (tools/materials/parameters)
  - "next" edges between consecutive steps
  - "condition_on" edges from each condition to referenced step(s) when available
  - "uses"/"requires" edges for step-to-equipment relations
  - "has_parameter" edges linking steps (or constraints) to parameter nodes when data is available
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


logger = logging.getLogger(__name__)

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


STEP_REF_KEYS = (
    "step",
    "steps",
    "step_id",
    "attached_to",
    "attached_step",
    "attached_steps",
    "targets",
    "scope",
    "applies_to",
)


def _norm_id(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _unique_id(prefix: str, existing: Set[str]) -> str:
    candidate = prefix
    counter = 1
    while candidate in existing:
        candidate = f"{prefix}_{counter}"
        counter += 1
    existing.add(candidate)
    return candidate


def _collect_step_refs(record: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for key in STEP_REF_KEYS:
        raw = record.get(key)
        refs.extend(_flatten_refs(raw))
    return refs


def _flatten_refs(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        potential = value.get("id") or value.get("step_id") or value.get("step")
        if potential:
            return [str(potential)]
        nested: List[str] = []
        for nested_value in value.values():
            nested.extend(_flatten_refs(nested_value))
        return nested
    if isinstance(value, (list, tuple, set)):
        refs: List[str] = []
        for item in value:
            refs.extend(_flatten_refs(item))
        return refs
    return [str(value)]


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_stringify(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_stringify(item) for item in value)
    return str(value)


def _infer_equipment_relation(record: Dict[str, Any], default: str) -> str:
    relation = str(record.get("relation") or record.get("edge_type") or record.get("type") or "").lower()
    if "require" in relation:
        return "requires"
    if "use" in relation:
        return "uses"
    if relation in LOWER_REL_MAP:
        return LOWER_REL_MAP[relation]
    return default


def convert_to_tierb(doc: Dict[str, Any]) -> Dict[str, Any]:
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
    equipment: List[Dict[str, Any]] = list(doc.get("equipment", []) or [])
    parameters: List[Dict[str, Any]] = list(doc.get("parameters", []) or [])

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    existing_ids: Set[str] = set()
    logger.debug("convert_to_tierb: document %s includes %d constraint(s)", document_id or "unknown", len(constraints))

    # Step nodes (preserve text for label extraction in evaluator)
    step_alias_map: Dict[str, str] = {}
    for s in steps:
        raw_sid = _norm_id(s.get("id"))
        sid = raw_sid or f"S{len(nodes)+1}"
        sid = _unique_id(sid, existing_ids)
        if raw_sid and raw_sid not in step_alias_map:
            step_alias_map[raw_sid] = sid
        nodes.append({
            "id": sid,
            "type": "step",
            "text": (s.get("text") or s.get("description") or "").strip(),
            "order": s.get("order"),
        })

    # Condition nodes
    constraint_node_ids: Dict[int, str] = {}
    for idx, c in enumerate(constraints):
        cid = _norm_id(c.get("id")) or f"C{len(nodes)+1}"
        cid = _unique_id(cid, existing_ids)
        constraint_node_ids[idx] = cid
        nodes.append({
            "id": cid,
            "type": "condition",
            "text": (c.get("text") or c.get("description") or "").strip(),
        })

    # Build a simple step index for NEXT edges and constraint attachments
    step_ids: List[str] = [n["id"] for n in nodes if n.get("type") == "step"]
    step_id_set = set(step_ids)

    # NEXT edges from step order (sequential)
    for i in range(len(step_ids) - 1):
        edges.append({
            "source": step_ids[i],
            "target": step_ids[i + 1],
            "type": "next",
        })

    # condition_on edges if constraint references exist
    # We read references from constraint["steps"] which may be list[str] or list[dict]
    condition_edge_type = LOWER_REL_MAP["CONDITION_ON"]
    condition_edge_count = 0
    constraints_missing_valid: List[str] = []
    constraints_missing_attachment_field: List[str] = []

    for idx, c in enumerate(constraints):
        cid = constraint_node_ids.get(idx)
        if not cid:
            continue

        attached_field_present = "attached_to" in c
        if not attached_field_present:
            constraints_missing_attachment_field.append(cid)
        attached_field = c.get("attached_to")
        attached_refs = _flatten_refs(attached_field)
        # Fall back to other reference keys if attached_to is missing or empty.
        refs = attached_refs or _collect_step_refs(c)
        valid_targets: List[str] = []

        for sid in refs:
            sid_norm = _norm_id(sid)
            if not sid_norm:
                continue
            resolved = step_alias_map.get(sid_norm, sid_norm)
            if resolved not in step_id_set:
                continue
            valid_targets.append(resolved)
            edges.append({
                "source": cid,
                "target": resolved,
                "type": condition_edge_type,
            })
            condition_edge_count += 1

        if attached_field_present and (not attached_refs or not valid_targets):
            constraints_missing_valid.append(cid)

    if constraints_missing_attachment_field:
        sample_missing = ", ".join(constraints_missing_attachment_field[:5])
        if len(constraints_missing_attachment_field) > 5:
            sample_missing = f"{sample_missing}, ..."
        logger.debug(
            "convert_to_tierb: %d constraint(s) lacked an 'attached_to' field (ids: %s)",
            len(constraints_missing_attachment_field),
            sample_missing,
        )
    if constraints_missing_valid:
        sample = ", ".join(constraints_missing_valid[:5])
        if len(constraints_missing_valid) > 5:
            sample = f"{sample}, ..."
        logger.debug(
            "convert_to_tierb: %d constraint(s) lacked valid attachments (ids: %s)",
            len(constraints_missing_valid),
            sample,
        )
    logger.debug("convert_to_tierb: created %d '%s' edge(s)", condition_edge_count, condition_edge_type)

    # Equipment / parameter nodes from top-level payloads
    equipment_nodes = _convert_equipment_entries(equipment, step_id_set, existing_ids, edges)
    parameter_nodes = _convert_parameter_entries(parameters, step_id_set, existing_ids, edges)
    nodes.extend(equipment_nodes)
    nodes.extend(parameter_nodes)

    # Step-level equipment/parameter fields
    step_equipment, step_parameters = _convert_step_level_entities(steps, existing_ids, edges)
    nodes.extend(step_equipment)
    nodes.extend(step_parameters)

    return {
        "document_id": document_id,
        "nodes": nodes,
        "edges": edges,
    }


# Backwards compatibility helper
flat_to_tierb = convert_to_tierb


def _convert_equipment_entries(
    equipment: Sequence[Any],
    valid_steps: Set[str],
    existing_ids: Set[str],
    edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for idx, entry in enumerate(equipment):
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, dict):
            continue
        eq_id = _norm_id(entry.get("id")) or f"EQUIP_{idx + 1}"
        eq_id = _unique_id(eq_id, existing_ids)
        label = entry.get("name") or entry.get("text") or entry.get("content") or eq_id
        nodes.append({
            "id": eq_id,
            "type": "equipment",
            "text": str(label),
            "name": entry.get("name") or str(label),
            "metadata": entry.get("metadata"),
        })
        relation = _infer_equipment_relation(entry, default="uses")
        for ref in _collect_step_refs(entry):
            sid = _norm_id(ref)
            if sid in valid_steps:
                edges.append({
                    "source": sid,
                    "target": eq_id,
                    "type": relation,
                })
    return nodes


def _convert_parameter_entries(
    parameters: Sequence[Any],
    valid_steps: Set[str],
    existing_ids: Set[str],
    edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for idx, entry in enumerate(parameters):
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, dict):
            continue
        param_id = _norm_id(entry.get("id")) or f"PARAM_{idx + 1}"
        param_id = _unique_id(param_id, existing_ids)
        text = entry.get("text") or entry.get("name") or entry.get("label") or _stringify(entry.get("value")) or param_id
        nodes.append({
            "id": param_id,
            "type": "parameter",
            "text": str(text),
            "name": entry.get("name"),
            "value": entry.get("value"),
        })
        relation = _infer_equipment_relation(entry, default="has_parameter")
        for ref in _collect_step_refs(entry):
            sid = _norm_id(ref)
            if sid in valid_steps:
                edges.append({
                    "source": sid,
                    "target": param_id,
                    "type": relation if relation in {"requires", "has_parameter", "uses"} else "has_parameter",
                })
    return nodes


def _convert_step_level_entities(
    steps: Sequence[Dict[str, Any]],
    existing_ids: Set[str],
    edges: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    equipment_nodes: List[Dict[str, Any]] = []
    parameter_nodes: List[Dict[str, Any]] = []
    for step in steps:
        step_id = _norm_id(step.get("id"))
        if not step_id:
            continue
        equipment_values = _gather_items(step, ["tools", "equipment", "materials"])
        for idx, (label, relation) in enumerate(equipment_values, start=1):
            node_id = _unique_id(f"{step_id}_equip_{idx}", existing_ids)
            equipment_nodes.append({
                "id": node_id,
                "type": "equipment",
                "text": label,
            })
            edges.append({
                "source": step_id,
                "target": node_id,
                "type": relation,
            })
        params = step.get("parameters")
        if isinstance(params, dict):
            for idx, (name, value) in enumerate(params.items(), start=1):
                node_id = _unique_id(f"{step_id}_param_{idx}", existing_ids)
                text_value = _stringify(value)
                parameter_nodes.append({
                    "id": node_id,
                    "type": "parameter",
                    "text": f"{name}: {text_value}" if name else text_value,
                    "name": name,
                    "value": text_value,
                })
                edges.append({
                    "source": step_id,
                    "target": node_id,
                    "type": "has_parameter",
                })
    return equipment_nodes, parameter_nodes


def _gather_items(step: Dict[str, Any], fields: Sequence[str]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for field in fields:
        entries = step.get(field)
        if not entries:
            continue
        relation = "requires" if field == "materials" else "uses"
        if isinstance(entries, str):
            items.append((entries, relation))
        elif isinstance(entries, (list, tuple, set)):
            for entry in entries:
                if entry:
                    items.append((str(entry), relation))
    return items
