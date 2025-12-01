import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import networkx as nx
import numpy as np
import spacy
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.evaluation.smatch import compute_smatch


HEADLINE_METRICS_ORDER = [
    "StepF1",
    "AdjacencyF1",
    "Kendall",
    "ConstraintCoverage",
    "ConstraintAttachmentF1",
    "ConstraintAttachmentF1_TierB",
    "A_score",
    "GraphPrecision",
    "GraphRecall",
    "GraphF1",
    "AlignedGraphF1",
    "AlignedEdgeAccuracy",
    "NEXT_EdgeF1",
    "Logic_EdgeF1",
    "B_score",
]


def round3(value: Optional[float]) -> Optional[float]:
    if value is None or np.isnan(value):
        return None
    return float(np.round(value + 1e-12, 3))


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_prf(tp: int, total_pred: int, total_gold: int) -> Tuple[float, float, float]:
    precision = safe_ratio(tp, total_pred)
    recall = safe_ratio(tp, total_gold)
    f1 = safe_ratio(2 * precision * recall, precision + recall) if precision + recall else 0.0
    return precision, recall, f1


class TextPreprocessor:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self._nlp, self._use_lemma = self._load(model_name)

    @staticmethod
    def _load(model_name: str):
        try:
            nlp = spacy.load(model_name)
            use_lemma = "lemmatizer" in nlp.pipe_names
            if not use_lemma:
                logging.warning("spaCy model '%s' missing lemmatizer; defaulting to surface forms.", model_name)
            return nlp, use_lemma
        except Exception:  # noqa: BLE001
            logging.warning("spaCy model '%s' unavailable; falling back to blank English model.", model_name)
            nlp = spacy.blank("en")
            use_lemma = False
            if "lemmatizer" not in nlp.pipe_names:
                try:
                    nlp.add_pipe("lemmatizer")
                    nlp.initialize()
                    use_lemma = True
                except Exception:  # noqa: BLE001
                    logging.warning("Failed to initialise spaCy lemmatizer; defaulting to surface forms.")
                    if "lemmatizer" in nlp.pipe_names:
                        nlp.remove_pipe("lemmatizer")
            return nlp, use_lemma

    def __call__(self, text: str) -> str:
        text = (text or "").lower()
        doc = self._nlp(text)
        tokens = [
            (token.lemma_ if self._use_lemma else token.text)
            for token in doc
            if not token.is_space and not token.is_punct
        ]
        return " ".join(tokens) if tokens else ""


class EmbeddingCache:
    def __init__(self, model_name: str = "all-mpnet-base-v2", device: Optional[str] = None) -> None:
        self._model = SentenceTransformer(model_name, device=device)
        self._cache: Dict[str, np.ndarray] = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()))
        unique: Dict[str, List[int]] = defaultdict(list)
        for idx, text in enumerate(texts):
            unique[text].append(idx)
        to_compute = [text for text in unique.keys() if text not in self._cache]
        if to_compute:
            vectors = self._model.encode(
                to_compute,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vector in zip(to_compute, vectors):
                self._cache[text] = vector
        dim = self._model.get_sentence_embedding_dimension()
        embeddings = np.zeros((len(texts), dim))
        for text, indices in unique.items():
            vector = self._cache[text]
            for idx in indices:
                embeddings[idx] = vector
        return embeddings


def normalize_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(normalize_field(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k} {normalize_field(v)}" for k, v in sorted(value.items()))
    return str(value)


def extract_step_text(step: Dict[str, Any]) -> str:
    for key in ("text", "description", "name", "label", "summary"):
        if step.get(key):
            return normalize_field(step[key])
    candidates = []
    for key in ("action", "object", "target", "tool", "result"):
        if step.get(key):
            candidates.append(normalize_field(step[key]))
    return " ".join(candidates) if candidates else normalize_field(step.get("id", ""))


def extract_constraint_text(constraint: Dict[str, Any]) -> str:
    for key in ("text", "description", "condition", "statement"):
        if constraint.get(key):
            return normalize_field(constraint[key])
    return normalize_field(constraint.get("id", ""))


def extract_node_label(node: Dict[str, Any]) -> str:
    if node.get("type", "").lower() == "step":
        return extract_step_text(node)
    if node.get("type"):
        type_hint = normalize_field(node.get("type"))
    else:
        type_hint = ""
    for key in ("text", "name", "label", "description"):
        if node.get(key):
            return normalize_field(node[key])
    return " ".join(filter(None, [type_hint, normalize_field(node.get("id", ""))]))


def collect_constraint_links(constraint: Dict[str, Any]) -> Set[str]:
    keys = [
        "step_id",
        "step",
        "steps",
        "attached_step",
        "attached_steps",
        "attached_to",
        "applies_to",
        "scope",
        "targets",
    ]
    links: Set[str] = set()
    for key in keys:
        value = constraint.get(key)
        if not value:
            continue
        if isinstance(value, str):
            links.add(value)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, dict):
                    candidate = item.get("id") or item.get("step_id") or normalize_field(item)
                    if candidate:
                        links.add(candidate)
                elif isinstance(item, str):
                    links.add(item)
        elif isinstance(value, dict):
            candidate = value.get("id") or value.get("step_id")
            if candidate:
                links.add(candidate)
    return links


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    return np.clip(a @ b.T, -1.0, 1.0)


@dataclass
class AlignmentResult:
    matches: List[Tuple[int, int, float]]
    gold_ids: List[str]
    pred_ids: List[str]

    @property
    def gold_size(self) -> int:
        return len(self.gold_ids)

    @property
    def pred_size(self) -> int:
        return len(self.pred_ids)

    @property
    def gold_to_pred(self) -> Dict[int, int]:
        return {g: p for g, p, _ in self.matches}

    @property
    def pred_to_gold(self) -> Dict[int, int]:
        return {p: g for g, p, _ in self.matches}


def alignment_to_id_map(alignment: AlignmentResult) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for gold_idx, pred_idx, _ in alignment.matches:
        if gold_idx >= len(alignment.gold_ids) or pred_idx >= len(alignment.pred_ids):
            continue
        gold_id = normalize_field(alignment.gold_ids[gold_idx])
        pred_id = normalize_field(alignment.pred_ids[pred_idx])
        if pred_id:
            mapping[pred_id] = gold_id
    return mapping


def align_by_text(
    gold_items: Sequence[Dict[str, Any]],
    pred_items: Sequence[Dict[str, Any]],
    text_fn: Callable[[Dict[str, Any]], str],
    preprocessor: TextPreprocessor,
    embedder: EmbeddingCache,
    threshold: float,
) -> AlignmentResult:
    gold_texts = [preprocessor(text_fn(item)) for item in gold_items]
    pred_texts = [preprocessor(text_fn(item)) for item in pred_items]
    gold_emb = embedder.encode(gold_texts)
    pred_emb = embedder.encode(pred_texts)
    sim = cosine_similarity_matrix(gold_emb, pred_emb)
    if not sim.size:
        return AlignmentResult([], [normalize_field(item.get("id", str(idx))) for idx, item in enumerate(gold_items)],
                               [normalize_field(item.get("id", str(idx))) for idx, item in enumerate(pred_items)])
    cost = 1 - sim
    row_ind, col_ind = linear_sum_assignment(cost)
    matches: List[Tuple[int, int, float]] = []
    for r, c in zip(row_ind, col_ind):
        if sim[r, c] >= threshold:
            matches.append((int(r), int(c), float(sim[r, c])))
    gold_ids = [normalize_field(item.get("id", str(idx))) for idx, item in enumerate(gold_items)]
    pred_ids = [normalize_field(item.get("id", str(idx))) for idx, item in enumerate(pred_items)]
    return AlignmentResult(matches, gold_ids, pred_ids)


def compute_step_metrics(
    alignment: AlignmentResult,
    gold_adj_pairs: Set[Tuple[int, int]],
    pred_adj_pairs: Set[Tuple[int, int]],
    gold_order: Dict[str, int],
    pred_order: Dict[str, int],
) -> Dict[str, Optional[float]]:
    step_matches = len(alignment.matches)
    step_precision = safe_ratio(step_matches, alignment.pred_size)
    step_recall = safe_ratio(step_matches, alignment.gold_size)
    step_f1 = safe_ratio(2 * step_precision * step_recall, step_precision + step_recall) if (step_precision + step_recall) else 0.0

    gold_to_pred = alignment.gold_to_pred
    pred_to_gold = alignment.pred_to_gold

    filtered_gold_pairs = {pair for pair in gold_adj_pairs if pair[0] in gold_to_pred and pair[1] in gold_to_pred}
    pred_pairs_mapped = {
        (pred_to_gold[pair[0]], pred_to_gold[pair[1]])
        for pair in pred_adj_pairs
        if pair[0] in pred_to_gold and pair[1] in pred_to_gold
    }
    tp = len(filtered_gold_pairs & pred_pairs_mapped)
    adj_precision, adj_recall, adj_f1 = compute_prf(tp, len(pred_pairs_mapped), len(filtered_gold_pairs))

    gold_ranks: List[int] = []
    pred_ranks: List[int] = []
    for g_idx, p_idx, _ in alignment.matches:
        gold_id = alignment.gold_ids[g_idx]
        pred_id = alignment.pred_ids[p_idx]
        if gold_id in gold_order and pred_id in pred_order:
            gold_ranks.append(gold_order[gold_id])
            pred_ranks.append(pred_order[pred_id])
    if len(gold_ranks) >= 2:
        tau, _ = kendalltau(gold_ranks, pred_ranks)
        kendall = (tau + 1) / 2 if tau is not None and not np.isnan(tau) else 0.0
    else:
        kendall = None

    return {
        "StepF1": round3(step_f1),
        "AdjacencyF1": round3(adj_f1),
        "Kendall": round3(kendall) if kendall is not None else None,
    }


def tier_a_constraints_metrics(
    gold_constraints: Sequence[Dict[str, Any]],
    pred_constraints: Sequence[Dict[str, Any]],
    preprocessor: TextPreprocessor,
    embedder: EmbeddingCache,
    threshold: float,
    step_alignment: AlignmentResult,
) -> Dict[str, Optional[float]]:
    constraint_alignment = align_by_text(
        gold_constraints,
        pred_constraints,
        extract_constraint_text,
        preprocessor,
        embedder,
        threshold,
    )
    matched_constraints = len(constraint_alignment.matches)
    coverage = None if constraint_alignment.gold_size == 0 else safe_ratio(matched_constraints, constraint_alignment.gold_size)

    gold_step_ids = step_alignment.gold_ids
    pred_step_ids = step_alignment.pred_ids
    pred_idx_to_gold_idx = step_alignment.pred_to_gold

    gold_links_cache: Dict[int, Set[str]] = {}
    for idx, constraint in enumerate(gold_constraints):
        links = collect_constraint_links(constraint)
        if links:
            gold_links_cache[idx] = links

    pred_links_cache: Dict[int, Set[str]] = {}
    for idx, constraint in enumerate(pred_constraints):
        links = collect_constraint_links(constraint)
        if links:
            pred_links_cache[idx] = links

    gold_step_id_to_index = {sid: idx for idx, sid in enumerate(gold_step_ids)}
    pred_step_id_to_index = {sid: idx for idx, sid in enumerate(pred_step_ids)}

    gold_triplets: Set[Tuple[int, int]] = set()
    for gold_idx, links in gold_links_cache.items():
        for link in links:
            gold_step_index = gold_step_id_to_index.get(link)
            if gold_step_index is not None:
                gold_triplets.add((gold_idx, gold_step_index))

    pred_to_gold_constraint = {pred_idx: gold_idx for gold_idx, pred_idx, _ in constraint_alignment.matches}
    pred_triplets: Set[Tuple[int, int]] = set()
    unmatched_pred = 0
    for pred_idx, links in pred_links_cache.items():
        mapped_constraint = pred_to_gold_constraint.get(pred_idx)
        for link in links:
            pred_step_index = pred_step_id_to_index.get(link)
            gold_step_index = pred_idx_to_gold_idx.get(pred_step_index) if pred_step_index is not None else None
            if mapped_constraint is not None and gold_step_index is not None:
                pred_triplets.add((mapped_constraint, gold_step_index))
            else:
                unmatched_pred += 1

    total_gold = len(gold_triplets)
    total_pred = len(pred_triplets) + unmatched_pred
    tp = len(gold_triplets & pred_triplets)

    if total_gold == 0 and total_pred == 0:
        attachment_metric: Optional[float] = None
    else:
        _, _, attachment_f1 = compute_prf(tp, total_pred, total_gold)
        attachment_metric = round3(attachment_f1)

    return {
        "ConstraintCoverage": round3(coverage) if coverage is not None else None,
        "ConstraintAttachmentF1": attachment_metric,
    }


def derive_sequence_adjacency(length: int) -> Set[Tuple[int, int]]:
    return {(idx, idx + 1) for idx in range(length - 1)}


def derive_sequence_order(ids: List[str]) -> Dict[str, int]:
    return {identifier: idx for idx, identifier in enumerate(ids)}


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


def evaluate_tier_a_document(
    gold_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    preprocessor: TextPreprocessor,
    embedder: EmbeddingCache,
    threshold: float,
    return_alignment_map: bool = False,
) -> Union[Dict[str, Optional[float]], Tuple[Dict[str, Optional[float]], Dict[str, str]]]:
    gold_steps = gold_doc.get("steps", [])
    pred_steps = pred_doc.get("steps", [])
    step_alignment = align_by_text(gold_steps, pred_steps, extract_step_text, preprocessor, embedder, threshold)

    gold_adj = derive_sequence_adjacency(len(gold_steps))
    pred_adj = derive_sequence_adjacency(len(pred_steps))
    gold_order = derive_sequence_order(step_alignment.gold_ids)
    pred_order = derive_sequence_order(step_alignment.pred_ids)

    metrics = compute_step_metrics(step_alignment, gold_adj, pred_adj, gold_order, pred_order)

    constraints_metrics = tier_a_constraints_metrics(
        gold_doc.get("constraints", []),
        pred_doc.get("constraints", []),
        preprocessor,
        embedder,
        threshold,
        step_alignment,
    )
    metrics.update(constraints_metrics)

    # New Procedural_Fidelity_Score: 0.5*ConstraintCoverage + 0.3*StepF1 + 0.2*Kendall
    if all(k in metrics for k in ["ConstraintCoverage", "StepF1", "Kendall"]):
        pfs = (
            0.5 * metrics.get("ConstraintCoverage", 0) +
            0.3 * metrics.get("StepF1", 0) +
            0.2 * (metrics.get("Kendall") or 0)
        )
        metrics["Procedural_Fidelity_Score"] = round3(pfs)
    else:
        metrics["Procedural_Fidelity_Score"] = None
    if return_alignment_map:
        return metrics, alignment_to_id_map(step_alignment)
    return metrics


def is_step_type(node_type: str) -> bool:
    return node_type.lower() == "step"


def is_constraint_type(node_type: str) -> bool:
    node_type = node_type.lower()
    return any(key in node_type for key in ("constraint", "guard", "condition"))


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
    metrics["B_score"] = metrics.get("GraphF1")
    return metrics


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pairs(
    gold_path: str,
    pred_path: str,
    subset: Optional[float] = None,
    seed: int = 13,
) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    gold_root = Path(gold_path)
    pred_root = Path(pred_path)

    if not gold_root.exists():
        raise FileNotFoundError(f"Gold path not found: {gold_path}")
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction path not found: {pred_path}")

    if gold_root.is_file():
        gold_files = [gold_root]
    else:
        gold_files = sorted(list(gold_root.glob("*.json")))

    if pred_root.is_file():
        pred_files = [pred_root]
    else:
        pred_files = sorted(list(pred_root.glob("*.json")))

    pred_by_name = {path.name: path for path in pred_files}
    pairs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for gold_file in gold_files:
        pred_file = pred_by_name.get(gold_file.name)
        if not pred_file:
            logging.warning("Missing prediction for %s", gold_file.name)
            continue
        doc_id = gold_file.stem
        try:
            gold_doc = load_json(gold_file)
            pred_doc = load_json(pred_file)
        except json.JSONDecodeError as exc:
            logging.error("Failed to read %s or %s: %s", gold_file, pred_file, exc)
            continue
        pairs.append((doc_id, gold_doc, pred_doc))

    if subset and 0 < subset < 1 and len(pairs) > 1:
        sample_size = max(1, int(round(len(pairs) * subset)))
        rng = random.Random(seed)
        pairs = rng.sample(pairs, sample_size)

    return pairs


def pretty_print(results: Dict[str, Dict[str, Optional[float]]]) -> None:
    metrics_present: List[str] = []
    for metric in HEADLINE_METRICS_ORDER:
        if any(metric in doc_metrics for doc_metrics in results.values()):
            metrics_present.append(metric)

    header = ["doc_id"] + metrics_present
    column_widths = {key: max(len(key), 10) for key in header}

    rows: List[List[str]] = []
    for doc_id, metrics in results.items():
        row = [doc_id]
        for metric in metrics_present:
            value = metrics.get(metric)
            if value is None:
                cell = "-"
            else:
                cell = f"{value:.3f}"
            column_widths[metric] = max(column_widths[metric], len(cell))
            row.append(cell)
        column_widths["doc_id"] = max(column_widths["doc_id"], len(doc_id))
        rows.append(row)

    def format_row(values: List[str]) -> str:
        return "  ".join(
            value.ljust(column_widths[key])
            for value, key in zip(values, header)
        )

    print(format_row(header))
    print("-" * sum(column_widths[key] + 2 for key in header))
    for row in rows:
        print(format_row(row))


def compute_macro_average(results: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    aggregates: Dict[str, List[float]] = defaultdict(list)
    for doc_id, metrics in results.items():
        if doc_id == "macro_avg":
            continue
        for metric_name, value in metrics.items():
            if value is not None:
                aggregates[metric_name].append(value)
    return {metric: round3(float(np.mean(values))) if values else None for metric, values in aggregates.items()}


def prepare_evaluator(
    spacy_model: str = "en_core_web_sm",
    embedding_model: str = "all-mpnet-base-v2",
    *,
    device: Optional[str] = None,
) -> Tuple[TextPreprocessor, EmbeddingCache]:
    """Instantiate reusable evaluation helpers."""
    preprocessor = TextPreprocessor(spacy_model)
    embedder = EmbeddingCache(model_name=embedding_model, device=device)
    return preprocessor, embedder


def run_evaluation(
    gold: Any,
    pred: Any,
    *,
    tiers: Iterable[str] = ("A", "B"),
    threshold: float = 0.75,
    preprocessor: Optional[TextPreprocessor] = None,
    embedder: Optional[EmbeddingCache] = None,
    spacy_model: str = "en_core_web_sm",
    embedding_model: str = "all-mpnet-base-v2",
    device: Optional[str] = None,
    preserve_conflicts: bool = False,
) -> Dict[str, Optional[float]]:
    """Evaluate a single prediction/gold pair for the requested tiers."""
    gold_doc = load_json(Path(gold)) if isinstance(gold, (str, Path)) else gold
    pred_doc = load_json(Path(pred)) if isinstance(pred, (str, Path)) else pred
    tiers_upper = {tier.upper() for tier in tiers} or {"A", "B"}
    evaluator_pre = preprocessor or TextPreprocessor(spacy_model)
    evaluator_emb = embedder or EmbeddingCache(model_name=embedding_model, device=device)

    metrics: Dict[str, Optional[float]] = {}
    step_alignment_map: Optional[Dict[str, str]] = None
    if "A" in tiers_upper:
        tier_a_result = evaluate_tier_a_document(
            gold_doc,
            pred_doc,
            evaluator_pre,
            evaluator_emb,
            threshold,
            return_alignment_map="B" in tiers_upper,
        )
        if isinstance(tier_a_result, tuple):
            tier_a_metrics, step_alignment_map = tier_a_result
        else:
            tier_a_metrics = tier_a_result
        metrics.update(tier_a_metrics)
    if "B" in tiers_upper:
        tier_b_metrics = evaluate_tier_b_document(
            gold_doc,
            pred_doc,
            evaluator_pre,
            evaluator_emb,
            threshold,
            step_id_map=step_alignment_map,
        )
        for key, value in tier_b_metrics.items():
            if preserve_conflicts and key in metrics:
                metrics[f"{key}_TierB"] = value
            else:
                metrics[key] = value
    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate procedural extraction quality.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory containing predictions/gold.")
    parser.add_argument("--gold_dir", type=str, default=None, help="Directory or file containing gold JSONs.")
    parser.add_argument("--pred_dir", type=str, default=None, help="Directory or file containing prediction JSONs.")
    parser.add_argument("--tier", type=str, choices=["A", "B", "both"], default="both", help="Which tier to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold for alignment.")
    parser.add_argument("--subset", type=float, default=None, help="Optional fraction to sample for sanity checks.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed used when sampling a subset.")
    parser.add_argument("--out_file", type=str, default=None, help="Path to save JSON report.")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm", help="spaCy model name for lemmatization.")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-mpnet-base-v2",
        help="SentenceTransformer model to use for semantic alignment.",
    )
    parser.add_argument("--device", type=str, default=None, help="SentenceTransformer device override.")
    args = parser.parse_args(argv)
    if not args.run_dir and (args.gold_dir is None or args.pred_dir is None):
        parser.error("Provide --run-dir or both --gold_dir and --pred_dir.")
    return args


def _guess_path(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
        json_candidate = root / f"{name}.json"
        if json_candidate.exists():
            return json_candidate
    return None


def resolve_io_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    gold_path = Path(args.gold_dir).resolve() if args.gold_dir else None
    pred_path = Path(args.pred_dir).resolve() if args.pred_dir else None

    if run_dir:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        if gold_path is None:
            gold_path = _guess_path(run_dir, ["gold", "references", "labels", "ground_truth"])
        if pred_path is None:
            pred_path = _guess_path(run_dir, ["predictions", "preds", "outputs", "results"])
        out_file = Path(args.out_file) if args.out_file else run_dir / f"evaluation_{args.tier.lower()}.json"
    else:
        out_file = Path(args.out_file or "evaluation_report.json")

    if gold_path is None or pred_path is None:
        raise FileNotFoundError("Unable to resolve gold/prediction paths; specify --gold_dir and --pred_dir explicitly.")
    return gold_path, pred_path, out_file


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        gold_dir, pred_dir, out_file = resolve_io_paths(args)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1
    args.gold_dir = str(gold_dir)
    args.pred_dir = str(pred_dir)
    args.out_file = str(out_file)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pairs = load_pairs(args.gold_dir, args.pred_dir, args.subset, args.seed)
    if not pairs:
        logging.error("No document pairs to evaluate.")
        return 1

    preprocessor, embedder = prepare_evaluator(args.spacy_model, args.embedding_model, device=args.device)

    doc_results: Dict[str, Dict[str, Optional[float]]] = {}
    if args.tier == "both":
        tier_tuple: Tuple[str, ...] = ("A", "B")
    else:
        tier_tuple = (args.tier,)

    for doc_id, gold_doc, pred_doc in tqdm(pairs, desc="Evaluating documents"):
        metrics = run_evaluation(
            gold_doc,
            pred_doc,
            tiers=tier_tuple,
            threshold=args.threshold,
            preprocessor=preprocessor,
            embedder=embedder,
        )
        doc_results[doc_id] = metrics

    macro = compute_macro_average(doc_results)
    doc_results["macro_avg"] = macro

    output_path = Path(args.out_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(doc_results, handle, indent=2)
    logging.info("Saved evaluation report to %s", output_path)

    pretty_print(doc_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
