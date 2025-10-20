"""
Evaluate structured procedural extraction

Compares a predicted procedural graph to a ground-truth graph and reports
precision/recall/F1 for entities and relations. Uses simple text-based
normalization and exact matching for a lightweight, reproducible baseline.

Usage
  python tests/evaluation/evaluate_structured.py \
    --ground-truth path/to/gt.json \
    --predictions path/to/pred.json \
    [--pretty]

Expected JSON format follows src/graph/schema.json and is loadable via
src.graph.models.Graph.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from src.graph.models import Graph, normalize_text


@dataclass
class PRF:
    tp: int
    fp: int
    fn: int

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    def to_dict(self):
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision(), 4),
            "recall": round(self.recall(), 4),
            "f1": round(self.f1(), 4),
        }


def load_graph(path: str) -> Graph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Graph.model_validate(data)


def set_entities(g: Graph) -> Dict[str, Set[str]]:
    """Build normalized entity sets per type for matching by value.

    Matching keys
    - steps: normalized step text
    - conditions: normalized condition expression
    - equipment: normalized "name [model]" if model present else name
    - parameters: normalized "name:value" (value optional)
    """
    steps = {normalize_text(s.text) for s in g.steps if normalize_text(s.text)}
    conds = {normalize_text(c.expression) for c in g.conditions if normalize_text(c.expression)}

    equip_vals = set()
    for e in g.equipment:
        name = normalize_text(e.name)
        model = normalize_text(e.model)
        if name:
            equip_vals.add(f"{name} [{model}]" if model else name)

    params = set()
    for p in g.parameters:
        name = normalize_text(p.name)
        value = normalize_text(p.value)
        if name:
            params.add(f"{name}:{value}" if value else name)

    return {
        "step": steps,
        "condition": conds,
        "equipment": equip_vals,
        "parameter": params,
    }


def prf_for_sets(gt: Set[str], pred: Set[str]) -> PRF:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    return PRF(tp, fp, fn)


def edge_key(g: Graph, from_id: str, to_id: str) -> Tuple[str, str]:
    """Resolve edge ends to normalized textual keys for robust matching.

    Priority of text for each node type when available:
    - Step: text
    - Condition: expression
    - Equipment: name [model]
    - Parameter: name:value
    Fallback to raw ID if resolution fails.
    """
    idx = g.id_index()
    def node_norm(nid: str) -> str:
        n = idx.get(nid)
        if n is None:
            return nid
        # duck-typing by available fields
        if hasattr(n, "text"):
            return normalize_text(getattr(n, "text", None)) or nid
        if hasattr(n, "expression"):
            return normalize_text(getattr(n, "expression", None)) or nid
        if hasattr(n, "name"):
            name = normalize_text(getattr(n, "name", None))
            model = normalize_text(getattr(n, "model", None))
            return (f"{name} [{model}]" if model else name) or nid
        if hasattr(n, "value") and hasattr(n, "name"):
            name = normalize_text(getattr(n, "name", None))
            value = normalize_text(getattr(n, "value", None))
            return (f"{name}:{value}" if value else name) or nid
        return nid

    return node_norm(from_id), node_norm(to_id)


def relation_set(g: Graph) -> Set[Tuple[str, str, str]]:
    """Return set of (type, from_key, to_key) for relation matching.
    Uses normalized textual keys from node resolution to be robust to differing IDs.
    """
    rels: Set[Tuple[str, str, str]] = set()
    for e in g.edges:
        fkey, tkey = edge_key(g, e.from_id, e.to_id)
        if fkey and tkey:
            rels.add((e.type, fkey, tkey))
    return rels


def adjacency_from_next(g: Graph) -> Set[Tuple[str, str]]:
    """Extract adjacency pairs from NEXT edges (from_key -> to_key)."""
    adj: Set[Tuple[str, str]] = set()
    for e in g.edges:
        if e.type == "NEXT":
            adj.add(edge_key(g, e.from_id, e.to_id))
    return adj


def evaluate(gt: Graph, pred: Graph) -> Dict:
    gt_sets = set_entities(gt)
    pr_sets = set_entities(pred)

    entity_metrics = {}
    for k in ("step", "condition", "equipment", "parameter"):
        entity_metrics[k] = prf_for_sets(gt_sets[k], pr_sets[k]).to_dict()

    # Relations overall (typed, normalized endpoints)
    rel_gt = relation_set(gt)
    rel_pr = relation_set(pred)
    relation_metrics = prf_for_sets(rel_gt, rel_pr).to_dict()

    # NEXT adjacency (ordering proxy)
    next_gt = adjacency_from_next(gt)
    next_pr = adjacency_from_next(pred)
    ordering_metrics = prf_for_sets(next_gt, next_pr).to_dict()

    # Macro-averaged entity F1
    macro_f1 = sum(m["f1"] for m in entity_metrics.values()) / 4.0

    return {
        "summary": {
            "macro_entity_f1": round(macro_f1, 4),
            "relation_f1": relation_metrics["f1"],
            "ordering_f1": ordering_metrics["f1"],
        },
        "entities": entity_metrics,
        "relations": relation_metrics,
        "ordering": ordering_metrics,
        "ground_truth_counts": gt.summarize(),
        "prediction_counts": pred.summarize(),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate structured procedural graph")
    ap.add_argument("--ground-truth", required=True, help="Path to ground-truth graph JSON")
    ap.add_argument("--predictions", required=True, help="Path to predicted graph JSON")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = ap.parse_args()

    gt = load_graph(args.ground_truth)
    pr = load_graph(args.predictions)

    if gt.document_id != pr.document_id:
        # Not fatal; warn via console since this is a skeleton script
        print(f"Warning: document_id mismatch: gt={gt.document_id} pred={pr.document_id}")

    res = evaluate(gt, pr)
    if args.pretty:
        print(json.dumps(res, indent=2))
    else:
        print(json.dumps(res, separators=(",", ":")))


if __name__ == "__main__":
    main()

