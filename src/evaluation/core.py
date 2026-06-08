import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

HEADLINE_METRICS_ORDER = [
    "StepF1",
    "AdjacencyF1",
    "Kendall",
    "ConstraintCoverage",
    "ConstraintAttachmentF1",
    "ConstraintAttachmentF1_TierB",
    "Phi",
    "GraphPrecision",
    "GraphRecall",
    "GraphF1",
    "AlignedGraphF1",
    "AlignedEdgeAccuracy",
    "NEXT_EdgeF1",
    "Logic_EdgeF1",
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
    for key in ("text", "description", "condition", "statement", "label", "expression"):
        if constraint.get(key):
            return normalize_field(constraint[key])
    return normalize_field(constraint.get("id", ""))


CONSTRAINT_LINK_KEYS = (
    "step_id",
    "step",
    "steps",
    "attached_step",
    "attached_steps",
    "attached_to",
    "applies_to",
    "scope",
    "targets",
)


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


def compute_macro_average(results: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    aggregates: Dict[str, List[float]] = defaultdict(list)
    for doc_id, metrics in results.items():
        if doc_id == "macro_avg":
            continue
        for metric_name, value in metrics.items():
            if value is not None:
                aggregates[metric_name].append(value)
    return {metric: round3(float(np.mean(values))) if values else None for metric, values in aggregates.items()}

