import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import networkx as nx

from src.evaluation.smatch import compute_smatch
from src.evaluation.core import (
    collect_constraint_links,
    compute_prf,
    extract_constraint_text,
    extract_node_label,
    extract_step_text,
    normalize_field,
    round3,
)
from src.evaluation.alignment import (
    AlignmentResult,
    EmbeddingCache,
    TextPreprocessor,
    align_by_text,
)


def is_step_type(node_type: str) -> bool:
    return node_type.lower() == "step"


def is_constraint_type(node_type: str) -> bool:
    node_type = node_type.lower()
    return any(key in node_type for key in ("constraint", "guard", "condition"))


def sort_steps_for_graph(steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered: List[Tuple[Tuple[float, int, int], Dict[str, Any]]] = []
    for idx, step in enumerate(steps):
        raw_order = step.get("order")
        order_value: Optional[float] = None
        if isinstance(raw_order, (int, float)):
            order_value = float(raw_order)
        elif isinstance(raw_order, str):
            try:
                order_value = float(raw_order.strip())
            except ValueError:
                order_value = None
        if order_value is not None:
            key = (0.0, int(order_value), idx)
        else:
            key = (1.0, idx, 0)
        ordered.append((key, step))
    ordered.sort(key=lambda item: item[0])
    return [item[1] for item in ordered]


def build_tier_b_graph(doc: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes_field = doc.get("nodes")
    edges_field = doc.get("edges")
    if nodes_field:
        nodes: List[Dict[str, Any]] = []
        for idx, node in enumerate(nodes_field or []):
            cleaned = dict(node)
            cleaned_id = normalize_field(cleaned.get("id") or f"node_{idx + 1}")
            cleaned_type = normalize_field(cleaned.get("type", "") or cleaned.get("category", ""))
            cleaned["id"] = cleaned_id
            cleaned["type"] = cleaned_type or "node"
            if not cleaned.get("text"):
                cleaned["text"] = extract_node_label(cleaned)
            nodes.append(cleaned)
        edges: List[Dict[str, Any]] = []
        for edge in edges_field or []:
            src, tgt = get_edge_endpoints(edge)
            relation = normalize_field(edge.get("type") or edge.get("relation", "")).lower()
            if not relation or not src or not tgt:
                continue
            edges.append({"type": relation, "source": src, "target": tgt})
        return nodes, edges

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    steps = doc.get("steps") or []
    constraints = doc.get("constraints") or []

    ordered_steps = sort_steps_for_graph(steps)
    step_ids: List[str] = []
    for idx, step in enumerate(ordered_steps):
        raw_id = step.get("id") or f"step_{idx + 1}"
        step_id = normalize_field(raw_id) or f"step_{idx + 1}"
        step_ids.append(step_id)
        nodes.append(
            {
                "id": step_id,
                "type": "step",
                "text": extract_step_text(step),
            },
        )
    for src, tgt in zip(step_ids, step_ids[1:]):
        edges.append({"type": "next", "source": src, "target": tgt})

    step_id_set = set(step_ids)
    seen_condition_edges: Set[Tuple[str, str]] = set()
    for idx, constraint in enumerate(constraints):
        raw_id = constraint.get("id") or f"constraint_{idx + 1}"
        constraint_id = normalize_field(raw_id) or f"constraint_{idx + 1}"
        nodes.append(
            {
                "id": constraint_id,
                "type": "constraint",
                "text": extract_constraint_text(constraint),
            },
        )
        for target in sorted(collect_constraint_links(constraint)):
            target_id = normalize_field(target)
            if target_id in step_id_set and (constraint_id, target_id) not in seen_condition_edges:
                edges.append({"type": "condition_on", "source": constraint_id, "target": target_id})
                seen_condition_edges.add((constraint_id, target_id))

    return nodes, edges


def get_edge_endpoints(edge: Dict[str, Any]) -> Tuple[str, str]:
    src = edge.get("source") or edge.get("from") or ""
    tgt = edge.get("target") or edge.get("to") or ""
    return normalize_field(src), normalize_field(tgt)


def build_step_lookup(steps: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {normalize_field(step.get("id", str(idx))): idx for idx, step in enumerate(steps)}


def adjacency_from_edges(edges: Sequence[Dict[str, Any]], id_to_index: Dict[str, int]) -> Set[Tuple[int, int]]:
    adjacency: Set[Tuple[int, int]] = set()
    for edge in edges:
        relation = normalize_field(edge.get("type", "")).lower()
        if relation != "next":
            continue
        src, tgt = get_edge_endpoints(edge)
        if src in id_to_index and tgt in id_to_index:
            adjacency.add((id_to_index[src], id_to_index[tgt]))
    return adjacency


def derive_order_from_next_edges(step_ids: List[str], edges: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    graph = nx.DiGraph()
    graph.add_nodes_from(step_ids)
    for edge in edges:
        relation = normalize_field(edge.get("type", "")).lower()
        if relation != "next":
            continue
        src, tgt = get_edge_endpoints(edge)
        if src in graph and tgt in graph:
            graph.add_edge(src, tgt)
    try:
        topo_order = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        topo_order = []
    if len(topo_order) < len(step_ids):
        topo_order.extend([sid for sid in step_ids if sid not in topo_order])
    return {sid: idx for idx, sid in enumerate(topo_order)}


def align_constraint_nodes(
    gold_nodes: Sequence[Dict[str, Any]],
    pred_nodes: Sequence[Dict[str, Any]],
    preprocessor: TextPreprocessor,
    embedder: EmbeddingCache,
    threshold: float,
) -> AlignmentResult:
    gold_constraints = [node for node in gold_nodes if is_constraint_type(normalize_field(node.get("type", "")))]
    pred_constraints = [node for node in pred_nodes if is_constraint_type(normalize_field(node.get("type", "")))]
    return align_by_text(gold_constraints, pred_constraints, extract_constraint_text, preprocessor, embedder, threshold)


def expand_node_mapping(
    gold_nodes: Sequence[Dict[str, Any]],
    pred_nodes: Sequence[Dict[str, Any]],
    preprocessor: TextPreprocessor,
    embedder: EmbeddingCache,
    threshold: float,
    initial_mapping: Dict[str, str],
) -> Dict[str, str]:
    mapping = dict(initial_mapping)
    used_gold = set(initial_mapping.values())
    used_pred = set(initial_mapping.keys())
    gold_by_type: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    pred_by_type: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)

    for node in gold_nodes:
        node_id = normalize_field(node.get("id", ""))
        if node_id in used_gold:
            continue
        node_type = normalize_field(node.get("type", ""))
        gold_by_type[node_type].append((len(gold_by_type[node_type]), node))

    for node in pred_nodes:
        node_id = normalize_field(node.get("id", ""))
        if node_id in used_pred:
            continue
        node_type = normalize_field(node.get("type", ""))
        pred_by_type[node_type].append((len(pred_by_type[node_type]), node))

    for node_type, gold_items in gold_by_type.items():
        pred_items = pred_by_type.get(node_type, [])
        if not pred_items:
            continue
        gold_seq = [item for _, item in gold_items]
        pred_seq = [item for _, item in pred_items]
        alignment = align_by_text(gold_seq, pred_seq, extract_node_label, preprocessor, embedder, threshold)
        for gold_idx, pred_idx, _ in alignment.matches:
            gold_id = normalize_field(gold_seq[gold_idx].get("id", ""))
            pred_id = normalize_field(pred_seq[pred_idx].get("id", ""))
            if gold_id and pred_id:
                mapping[pred_id] = gold_id
                used_gold.add(gold_id)
                used_pred.add(pred_id)

    gold_ids = {normalize_field(node.get("id", "")) for node in gold_nodes}
    for node in pred_nodes:
        pred_id = normalize_field(node.get("id", ""))
        if pred_id in used_pred:
            continue
        if pred_id in gold_ids and pred_id not in used_gold:
            mapping[pred_id] = pred_id
            used_gold.add(pred_id)
            used_pred.add(pred_id)

    return mapping


def graph_to_smatch_triples(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
) -> Tuple[Set[Tuple[str, str, str]], Set[str]]:
    triples: Set[Tuple[str, str, str]] = set()
    node_ids: Set[str] = set()
    for node in nodes:
        node_id = normalize_field(node.get("id", ""))
        if not node_id:
            continue
        node_ids.add(node_id)
        node_type = normalize_field(node.get("type", ""))
        label = extract_node_label(node)
        descriptor = f"{node_type}:{label}" if node_type else label
        triples.add((node_id, "instance", descriptor))

    for edge in edges:
        relation = normalize_field(edge.get("type", "")).lower()
        if not relation:
            continue
        src, tgt = get_edge_endpoints(edge)
        if not src or not tgt:
            continue
        triples.add((src, relation, tgt))

    return triples, node_ids


def compute_edge_metrics(
    edges_gold: Sequence[Dict[str, Any]],
    edges_pred: Sequence[Dict[str, Any]],
    node_mapping: Dict[str, str],
    relation_filter: Callable[[str], bool],
) -> float:
    gold_set: Set[Tuple[str, str, str]] = set()
    for edge in edges_gold:
        relation = normalize_field(edge.get("type", "")).lower()
        if not relation_filter(relation):
            continue
        src, tgt = get_edge_endpoints(edge)
        if src and tgt:
            gold_set.add((src, relation, tgt))

    pred_set: Set[Tuple[str, str, str]] = set()
    unmatched_pred = 0
    for edge in edges_pred:
        relation = normalize_field(edge.get("type", "")).lower()
        if not relation_filter(relation):
            continue
        src, tgt = get_edge_endpoints(edge)
        if not src or not tgt:
            unmatched_pred += 1
            continue
        mapped_src = node_mapping.get(src)
        mapped_tgt = node_mapping.get(tgt)
        if mapped_src and mapped_tgt:
            pred_set.add((mapped_src, relation, mapped_tgt))
        else:
            unmatched_pred += 1

    tp = len(gold_set & pred_set)
    total_pred = len(pred_set) + unmatched_pred
    _, _, f1 = compute_prf(tp, total_pred, len(gold_set))
    return round3(f1)


def compute_constraint_attachment_f1_tier_b(
    nodes_gold: Sequence[Dict[str, Any]],
    edges_gold: Sequence[Dict[str, Any]],
    nodes_pred: Sequence[Dict[str, Any]],
    edges_pred: Sequence[Dict[str, Any]],
    node_mapping: Dict[str, str],
) -> Optional[float]:
    type_map_gold = {normalize_field(node.get("id", "")): normalize_field(node.get("type", "")) for node in nodes_gold}
    type_map_pred = {normalize_field(node.get("id", "")): normalize_field(node.get("type", "")) for node in nodes_pred}

    attachments_gold: Set[Tuple[str, str, str]] = set()
    for edge in edges_gold:
        relation = normalize_field(edge.get("type", "")).lower()
        src, tgt = get_edge_endpoints(edge)
        if not src or not tgt:
            continue
        src_type = type_map_gold.get(src, "")
        tgt_type = type_map_gold.get(tgt, "")
        if is_constraint_type(src_type) and is_step_type(tgt_type):
            attachments_gold.add((src, relation, tgt))
        elif is_step_type(src_type) and is_constraint_type(tgt_type):
            attachments_gold.add((tgt, relation, src))

    attachments_pred: Set[Tuple[str, str, str]] = set()
    unmatched_pred = 0
    for edge in edges_pred:
        relation = normalize_field(edge.get("type", "")).lower()
        src, tgt = get_edge_endpoints(edge)
        if not src or not tgt:
            unmatched_pred += 1
            continue
        src_type = type_map_pred.get(src, "")
        tgt_type = type_map_pred.get(tgt, "")
        if is_constraint_type(src_type) and is_step_type(tgt_type):
            mapped_src = node_mapping.get(src)
            mapped_tgt = node_mapping.get(tgt)
        elif is_step_type(src_type) and is_constraint_type(tgt_type):
            mapped_src = node_mapping.get(tgt)
            mapped_tgt = node_mapping.get(src)
        else:
            continue
        if mapped_src and mapped_tgt:
            attachments_pred.add((mapped_src, relation, mapped_tgt))
        else:
            unmatched_pred += 1

    if not attachments_gold and not attachments_pred:
        return None

    tp = len(attachments_gold & attachments_pred)
    total_pred = len(attachments_pred) + unmatched_pred
    _, _, f1 = compute_prf(tp, total_pred, len(attachments_gold))
    return round3(f1)


def evaluate_tier_b_document(
    gold_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    preprocessor: TextPreprocessor,
    embedder: EmbeddingCache,
    threshold: float,
    step_id_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[float]]:
    gold_nodes, gold_edges = build_tier_b_graph(gold_doc)
    pred_nodes, pred_edges = build_tier_b_graph(pred_doc)
    gold_triples, gold_node_ids = graph_to_smatch_triples(gold_nodes, gold_edges)
    pred_triples, pred_node_ids = graph_to_smatch_triples(pred_nodes, pred_edges)

    initial_mapping: Dict[str, str] = {}
    if step_id_map:
        for pred_id, gold_id in step_id_map.items():
            if pred_id in pred_node_ids and gold_id in gold_node_ids:
                initial_mapping[pred_id] = gold_id

    smatch_result = compute_smatch(
        gold_triples,
        pred_triples,
        gold_node_ids,
        pred_node_ids,
        initial_mapping=initial_mapping,
    )
    node_mapping = smatch_result.mapping

    metrics: Dict[str, Optional[float]] = {
        "GraphPrecision": round3(smatch_result.precision),
        "GraphRecall": round3(smatch_result.recall),
        "GraphF1": round3(smatch_result.f1),
    }
    metrics["NEXT_EdgeF1"] = compute_edge_metrics(
        gold_edges,
        pred_edges,
        node_mapping,
        relation_filter=lambda rel: rel == "next",
    )
    metrics["Logic_EdgeF1"] = compute_edge_metrics(
        gold_edges,
        pred_edges,
        node_mapping,
        relation_filter=lambda rel: rel != "next",
    )
    attachment_f1 = compute_constraint_attachment_f1_tier_b(
        gold_nodes,
        gold_edges,
        pred_nodes,
        pred_edges,
        node_mapping,
    )
    if attachment_f1 is not None:
        metrics["ConstraintAttachmentF1_TierB"] = attachment_f1
    return metrics
