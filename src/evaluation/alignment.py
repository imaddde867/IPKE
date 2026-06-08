import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import spacy
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

from src.evaluation.core import normalize_field


class TextPreprocessor:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self._nlp: spacy.Language
        self._use_lemma: bool
        try:
            nlp = spacy.load(model_name)
            use_lemma = "lemmatizer" in nlp.pipe_names
            if not use_lemma:
                logging.warning("spaCy model '%s' missing lemmatizer; defaulting to surface forms.", model_name)
            self._nlp = nlp
            self._use_lemma = use_lemma
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
            self._nlp = nlp
            self._use_lemma = use_lemma

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


def prepare_evaluator(
    spacy_model: str = "en_core_web_sm",
    embedding_model: str = "all-mpnet-base-v2",
    *,
    device: Optional[str] = None,
) -> Tuple[TextPreprocessor, EmbeddingCache]:
    preprocessor = TextPreprocessor(spacy_model)
    embedder = EmbeddingCache(model_name=embedding_model, device=device)
    return preprocessor, embedder
