"""Quick CLI helper to summarize PKG JSONs."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON file.") from exc


def _normalize_refs(attached: Any) -> List[str]:
    if not attached:
        return []
    if isinstance(attached, str):
        return [attached]
    if isinstance(attached, list):
        return [str(item) for item in attached if item]
    return [str(attached)]


def print_graph_structure(file_path: str) -> None:
    path = Path(file_path)
    try:
        data = _load_payload(path)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return

    steps = data.get("steps", [])
    constraints = data.get("constraints", [])
    entities = data.get("entities", [])
    steps_lookup = {str(s.get("id")): s.get("text", "") for s in steps if s.get("id")}

    print(f"\n=== KNOWLEDGE GRAPH: {path.stem} ===")
    print(f"Stats: {len(steps_lookup)} Steps | {len(constraints)} Constraints | {len(entities)} Entities")

    print("\n--- PROCEDURAL FLOW (Step -> Next Step) ---")
    sorted_steps = sorted(steps, key=lambda x: x.get("order", 999))
    for curr, nxt in zip(sorted_steps, sorted_steps[1:]):
        print(f"  [Step {curr.get('id')}] --> [Step {nxt.get('id')}]")
        print(f"     \"{curr.get('text', '')[:60]}...\"")

    print("\n--- LOGIC LAYER (Constraint -> Step) ---")
    linked_count = 0
    for constraint in constraints:
        c_id = constraint.get("id", "??")
        c_text = constraint.get("expression", constraint.get("text", "Unknown"))
        c_text = str(c_text).replace("\n", " ")[:50]

        attached = _normalize_refs(constraint.get("attached_to") or constraint.get("steps"))
        if not attached:
            print(f"  [Constraint {c_id}] (Unlinked): \"{c_text}...\"")
            continue

        linked_count += 1
        for target in attached:
            if target in steps_lookup:
                print(f"  [Constraint {c_id}] --guards--> [Step {target}]")
                print(f"     Rule: \"{c_text}...\"")
            else:
                print(f"  [Constraint {c_id}] --(broken link)--> {target}")

    print(f"\nTotal Linked Constraints: {linked_count} / {len(constraints)}")


if __name__ == "__main__":
    base_path = "logs/prompting_grid/P3_two_stage/3M_OEM_SOP/predictions.json"
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    print_graph_structure(base_path)
