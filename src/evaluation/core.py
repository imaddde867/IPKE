import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import spacy
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer


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
        except (OSError, ValueError):
            logging.warning("spaCy model '%s' unavailable; falling back to blank English model.", model_name)
            nlp = spacy.blank("en")
            use_lemma = False
            if "lemmatizer" not in nlp.pipe_names:
                try:
                    nlp.add_pipe("lemmatizer")
                    nlp.initialize()
                    use_lemma = True
                except (OSError, ValueError, RuntimeError):
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
    for key in ("text", "description", "condition", "statement", "label", "expression"):
        if constraint.get(key):
            return normalize_field(constraint[key])
    return normalize_field(constraint.get("id", ""))


NESTED_CONSTRAINT_KINDS = (
    "precondition",
    "postcondition",
    "guard",
    "acceptance_criteria",
    "warning",
    "exception",
    "safety",
    "environment",
    "quality",
)

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


def normalize_doc_constraints(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    top = doc.get("constraints")
    if isinstance(top, list) and top:
        return list(top)

    flat: List[Dict[str, Any]] = []
    for step in doc.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        sid = step.get("id")
        nested = step.get("constraints")
        if isinstance(nested, list):
            for item in nested:
                if not isinstance(item, dict):
                    continue
                new_item = dict(item)
                has_link = any(new_item.get(k) for k in CONSTRAINT_LINK_KEYS)
                if sid and not has_link:
                    new_item["applies_to"] = sid
                flat.append(new_item)
            continue
        if not isinstance(nested, dict):
            continue
        for kind in NESTED_CONSTRAINT_KINDS:
            items = nested.get(kind)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                new_item = dict(item)
                new_item.setdefault("type", kind)
                has_link = any(new_item.get(k) for k in CONSTRAINT_LINK_KEYS)
                if sid and not has_link:
                    new_item["applies_to"] = sid
                flat.append(new_item)
        for kind, items in nested.items():
            if kind in NESTED_CONSTRAINT_KINDS or not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                if any(item.get(k) for k in CONSTRAINT_LINK_KEYS):
                    new_item = dict(item)
                    new_item.setdefault("type", kind)
                    flat.append(new_item)
    return flat


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


def derive_sequence_adjacency(length: int) -> Set[Tuple[int, int]]:
    return {(idx, idx + 1) for idx in range(length - 1)}


def derive_sequence_order(ids: List[str]) -> Dict[str, int]:
    return {identifier: idx for idx, identifier in enumerate(ids)}


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


def prepare_evaluator(
    spacy_model: str = "en_core_web_sm",
    embedding_model: str = "all-mpnet-base-v2",
    *,
    device: Optional[str] = None,
) -> Tuple[TextPreprocessor, EmbeddingCache]:
    preprocessor = TextPreprocessor(spacy_model)
    embedder = EmbeddingCache(model_name=embedding_model, device=device)
    return preprocessor, embedder
