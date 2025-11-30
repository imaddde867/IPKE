"""Lightweight Smatch implementation for procedural graphs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


Triples = Sequence[Tuple[str, str, str]]


@dataclass
class SmatchResult:
    precision: float
    recall: float
    f1: float
    mapping: Dict[str, str]


def compute_smatch(
    gold_triples: Iterable[Tuple[str, str, str]],
    pred_triples: Iterable[Tuple[str, str, str]],
    gold_nodes: Iterable[str],
    pred_nodes: Iterable[str],
    *,
    initial_mapping: Dict[str, str] | None = None,
    iterations: int = 10,
    seed: int = 13,
) -> SmatchResult:
    gold_set = set(gold_triples)
    pred_list = list(pred_triples)
    pred_node_list = [node for node in pred_nodes if node]
    gold_node_list = [node for node in gold_nodes if node]
    if not gold_node_list:
        gold_node_list = ["__SMATCH_NULL__"]

    if not gold_set and not pred_list:
        return SmatchResult(precision=1.0, recall=1.0, f1=1.0, mapping={})

    rng = random.Random(seed)
    pred_node_set = set(pred_node_list)
    base_mapping = {
        key: value
        for key, value in (initial_mapping or {}).items()
        if key in pred_node_set and value in gold_node_list
    }

    best_mapping: Dict[str, str] = {}
    best_match = 0

    for _ in range(max(iterations, 1)):
        mapping = base_mapping.copy()
        for node in pred_node_list:
            mapping.setdefault(node, rng.choice(gold_node_list))
        match_count = _score_mapping(mapping, gold_set, pred_list, pred_node_set)
        improved = True
        while improved:
            improved = False
            for node in pred_node_list:
                current_target = mapping.get(node)
                best_target = current_target
                best_local = match_count
                for candidate in gold_node_list:
                    if candidate == current_target:
                        continue
                    mapping[node] = candidate
                    score = _score_mapping(mapping, gold_set, pred_list, pred_node_set)
                    if score > best_local:
                        best_local = score
                        best_target = candidate
                mapping[node] = best_target
                if best_local > match_count:
                    match_count = best_local
                    improved = True
        if match_count > best_match:
            best_match = match_count
            best_mapping = mapping.copy()

    precision = _safe_ratio(best_match, len(pred_list))
    recall = _safe_ratio(best_match, len(gold_set))
    f1 = _safe_ratio(2 * precision * recall, precision + recall) if precision + recall else 0.0
    return SmatchResult(precision=precision, recall=recall, f1=f1, mapping=best_mapping)


def _score_mapping(
    mapping: Dict[str, str],
    gold_triples: Set[Tuple[str, str, str]],
    pred_triples: Sequence[Tuple[str, str, str]],
    pred_nodes: Set[str],
) -> int:
    mapped: Set[Tuple[str, str, str]] = set()
    for src, relation, tgt in pred_triples:
        mapped_src = mapping.get(src, src)
        mapped_tgt = mapping.get(tgt, tgt) if tgt in pred_nodes else tgt
        mapped.add((mapped_src, relation, mapped_tgt))
    return len(gold_triples & mapped)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


__all__ = ["SmatchResult", "compute_smatch"]
