import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.stats import kendalltau

from src.evaluation.core import (
    collect_constraint_links,
    compute_prf,
    derive_sequence_adjacency,
    derive_sequence_order,
    extract_constraint_text,
    extract_step_text,
    normalize_doc_constraints,
    round3,
    safe_ratio,
)
from src.evaluation.alignment import (
    AlignmentResult,
    EmbeddingCache,
    TextPreprocessor,
    align_by_text,
    alignment_to_id_map,
)


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


def compute_phi(
    constraint_coverage: float | None,
    step_f1: float | None,
    kendall: float | None,
    w_cov: float = 0.5,
    w_step: float = 0.3,
    w_tau: float = 0.2,
    round_result: bool = False,
) -> float:
    c = 0.0 if constraint_coverage is None else constraint_coverage
    s = 0.0 if step_f1 is None else step_f1
    k = 0.0 if kendall is None else kendall
    raw = w_cov * c + w_step * s + w_tau * k
    return round3(raw) if round_result else raw


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
        normalize_doc_constraints(gold_doc),
        normalize_doc_constraints(pred_doc),
        preprocessor,
        embedder,
        threshold,
        step_alignment,
    )
    metrics.update(constraints_metrics)

    if all(k in metrics for k in ["ConstraintCoverage", "StepF1", "Kendall"]):
        metrics["Phi"] = compute_phi(
            constraint_coverage=metrics["ConstraintCoverage"],
            step_f1=metrics["StepF1"],
            kendall=metrics["Kendall"],
            round_result=True,
        )
    else:
        metrics["Phi"] = None
    if return_alignment_map:
        return metrics, alignment_to_id_map(step_alignment)
    return metrics
