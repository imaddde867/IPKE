"""Pluggable text chunking strategies for the knowledge engine."""

from __future__ import annotations
import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.core.unified_config import UnifiedConfig, Environment
from src.logging_config import get_logger

logger = get_logger(__name__)

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None  # type: ignore

SentenceTransformer = None  # Lazily imported

_SPACY_CACHE = {}
_EMBEDDER_CACHE: Dict[str, Any] = {}
_HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+\.)[\s:-]+|^\s*[A-Z][\w\s]{0,60}:"
)


class _SimpleSentence:
    """Minimal sentence span compatible with spaCy's interface."""

    __slots__ = ("text", "start_char", "end_char")

    def __init__(self, text: str, start_char: int, end_char: int):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char


class _SimpleSentenceDoc:
    """Container exposing a `.sents` iterable like spaCy Doc."""

    __slots__ = ("sents",)

    def __init__(self, sentences: List[_SimpleSentence]):
        self.sents = sentences


class _RegexSentenceSplitter:
    """Lightweight fallback tokenizer when spaCy isn't available."""

    _pattern = re.compile(r"[^.!?\n]+[.!?\n]*")

    def __call__(self, text: str) -> _SimpleSentenceDoc:
        spans: List[_SimpleSentence] = []
        for match in self._pattern.finditer(text):
            snippet = match.group()
            if not snippet.strip():
                continue
            spans.append(_SimpleSentence(snippet, match.start(), match.end()))
        return _SimpleSentenceDoc(spans)


class _DeterministicEmbedder:
    """Deterministic, dependency-free sentence embeddings."""

    def encode(
        self,
        sentences: List[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **_: Any,
    ):
        vectors: List[np.ndarray] = []
        for sentence in sentences:
            normalized = sentence.strip().lower().encode("utf-8")
            digest = hashlib.md5(normalized).digest()
            base_features = [
                digest[0] / 255.0 + 0.1,
                digest[1] / 255.0 + 0.1,
                float(len(sentence)) / 100.0 + 0.1,
                float(len(sentence.split())) / 10.0 + 0.1,
            ]
            vectors.append(np.asarray(base_features, dtype=np.float32))
        if not vectors:
            return np.zeros((0, 4), dtype=np.float32)
        arr = np.vstack(vectors)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-8)
            arr = arr / norms
        return arr


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int
    sentence_span: Tuple[int, int]
    meta: Dict[str, Any]


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        ...


class FixedChunker(BaseChunker):
    """Legacy fixed-size chunker with whitespace-aware boundaries."""

    def __init__(self, max_chars: int = 2000):
        self.max_chars = max_chars

    def chunk(self, text: str) -> List[Chunk]:
        if not text:
            return []

        chunks: List[Chunk] = []
        idx = 0
        length = len(text)

        while idx < length:
            cap = min(idx + self.max_chars, length)
            split = cap
            if cap < length:
                soft = text.rfind(" ", idx, cap)
                if soft > idx + int(self.max_chars * 0.5):
                    split = soft
            snippet = text[idx:split].strip()
            if snippet:
                chunk = Chunk(
                    text=snippet,
                    start_char=idx,
                    end_char=split,
                    sentence_span=(-1, -1),
                    meta={
                        "chunking_method": "fixed",
                        "sentence_count": None,
                        "cohesion": None,
                        "char_length": len(snippet),
                    },
                )
                chunks.append(chunk)
            idx = split
            while idx < length and text[idx].isspace():
                idx += 1

        return chunks


class BreakpointSemanticChunker(BaseChunker):
    """Sentence-level dynamic-programming chunker."""

    def __init__(
        self,
        cfg: UnifiedConfig,
        *,
        nlp=None,
        embedder=None,
    ):
        self.cfg = cfg
        if cfg.environment == Environment.TESTING:
            if nlp is None:
                nlp = _RegexSentenceSplitter()
            if embedder is None:
                embedder = _DeterministicEmbedder()
        self._nlp = nlp
        self._embedder = embedder

    def chunk(self, text: str) -> List[Chunk]:
        sentences, spans = self._sentences(text)
        if not sentences:
            return []
        embeddings = self._embeddings(sentences)
        return self._chunk_from_sentences(
            sentences=sentences,
            spans=spans,
            embeddings=embeddings,
            sentence_offset=0,
            text=text,
            method_label="breakpoint_semantic",
        )

    # --- Shared helpers -------------------------------------------------

    def _chunk_from_sentences(
        self,
        sentences: List[str],
        spans: List[Tuple[int, int]],
        embeddings: np.ndarray,
        *,
        sentence_offset: int,
        text: str,
        method_label: str,
    ) -> List[Chunk]:
        n = len(sentences)
        if n == 0:
            return []
        sims = self._pairwise_cosine(embeddings)
        pref = np.concatenate(([0.0], np.cumsum(sims)))
        boundaries = self._dp_boundaries(n, pref)
        return self._build_chunks(
            boundaries,
            sentences,
            spans,
            pref,
            sentence_offset=sentence_offset,
            text=text,
            method_label=method_label,
        )

    def _dp_boundaries(self, n: int, pref: np.ndarray) -> List[Tuple[int, int]]:
        cfg = self.cfg
        min_len = max(1, cfg.sem_min_sentences_per_chunk)
        max_len = max(min_len, cfg.sem_max_sentences_per_chunk)
        window = max(max_len, cfg.sem_window_w)
        lam = cfg.sem_lambda

        dp = [-1e9] * (n + 1)
        prev = [-1] * (n + 1)
        dp[0] = 0.0

        for j in range(1, n + 1):
            i_start = max(0, j - window)
            best_val = -1e9
            best_i = -1
            for i in range(i_start, j):
                length = j - i
                if length < min_len or length > max_len:
                    continue
                pairs = max(1, length - 1)
                seg_sum = pref[j - 1] - pref[i]
                cohesion = seg_sum / pairs
                candidate = dp[i] + cohesion - lam
                if candidate > best_val:
                    best_val = candidate
                    best_i = i
            if best_i >= 0:
                dp[j] = best_val
                prev[j] = best_i

        boundaries: List[Tuple[int, int]] = []
        j = n
        while j > 0:
            i = prev[j]
            if i < 0:
                i = max(0, j - max_len)
            boundaries.append((i, j))
            j = i
        boundaries.reverse()
        if not boundaries:
            return [(0, n)]
        first_start, first_end = boundaries[0]
        if first_start != 0:
            boundaries[0] = (0, first_end)
        for idx in range(1, len(boundaries)):
            prev_end = boundaries[idx - 1][1]
            start, end = boundaries[idx]
            start = max(prev_end, start)
            end = max(start + 1, end)
            boundaries[idx] = (start, min(end, n))
        if boundaries[-1][1] < n:
            boundaries.append((boundaries[-1][1], n))
        return boundaries

    def _build_chunks(
        self,
        boundaries: List[Tuple[int, int]],
        sentences: List[str],
        spans: List[Tuple[int, int]],
        pref: np.ndarray,
        *,
        sentence_offset: int,
        text: str,
        method_label: str,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        for idx, (start, end) in enumerate(boundaries):
            if start >= end:
                continue
            for sub_start, sub_end in self._apply_char_cap(start, end, spans):
                start_char = spans[sub_start][0]
                end_char = spans[sub_end - 1][1]
                snippet = text[start_char:end_char].strip()
                if not snippet:
                    continue
                length = sub_end - sub_start
                cohesion = self._cohesion(pref, sub_start, sub_end)
                meta = {
                    "chunking_method": method_label,
                    "sentence_count": length,
                    "cohesion": cohesion,
                    "char_length": len(snippet),
                    "chunk_index": len(chunks),
                }
                if self.cfg.debug_chunking:
                    meta["sentence_offsets"] = (sentence_offset + sub_start, sentence_offset + sub_end - 1)
                chunk = Chunk(
                    text=snippet,
                    start_char=start_char,
                    end_char=end_char,
                    sentence_span=(sentence_offset + sub_start, sentence_offset + sub_end - 1),
                    meta=meta,
                )
                chunks.append(chunk)
        return chunks

    def _apply_char_cap(
        self,
        start: int,
        end: int,
        spans: List[Tuple[int, int]],
    ) -> Iterable[Tuple[int, int]]:
        cap = max(64, self.cfg.chunk_max_chars)
        current_start = start
        accum = 0

        for i in range(start, end):
            sentence_len = spans[i][1] - spans[i][0]
            if accum + sentence_len > cap and i > current_start:
                yield current_start, i
                current_start = i
                accum = 0
            accum += sentence_len

        yield current_start, end

    def _cohesion(self, pref: np.ndarray, start: int, end: int) -> float:
        length = end - start
        if length <= 1:
            return 1.0
        seg_sum = pref[end - 1] - pref[start]
        return float(seg_sum / (length - 1))

    def _pairwise_cosine(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.shape[0] < 2:
            return np.zeros(0, dtype=np.float32)
        sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        return sims.astype(np.float32)

    def _sentences(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        if not text:
            return [], []
        nlp = self._get_nlp()
        doc = nlp(text)
        sentences: List[str] = []
        spans: List[Tuple[int, int]] = []
        for sent in doc.sents:
            snippet = sent.text.strip()
            if not snippet:
                continue
            sentences.append(snippet)
            spans.append((sent.start_char, sent.end_char))
        return sentences, spans

    def _embeddings(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, 1), dtype=np.float32)
        model = self._get_embedder()
        try:
            vectors = model.encode(
                sentences,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Sentence embedding failed using model {self.cfg.embedding_model_path}: {exc}"
            ) from exc
        embeddings = np.asarray(vectors, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[:, None]
        return embeddings

    def _get_nlp(self):
        if self._nlp is not None:
            return self._nlp
        if self.cfg.environment == Environment.TESTING:
            self._nlp = _RegexSentenceSplitter()
            return self._nlp
        if spacy is None:
            logger.warning(
                "spaCy is not available; falling back to a regex-based sentence splitter for chunking."
            )
            self._nlp = _RegexSentenceSplitter()
            return self._nlp
        cache_key = "en_core_web_sm"
        if cache_key not in _SPACY_CACHE:
            nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer", "morphologizer"])
            if "senter" not in nlp.pipe_names:
                try:
                    nlp.add_pipe("senter")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Unable to add spaCy 'senter' pipe (%s); proceeding with existing pipeline.", exc)
            _SPACY_CACHE[cache_key] = nlp
        self._nlp = _SPACY_CACHE[cache_key]
        return self._nlp

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        model_path = self.cfg.embedding_model_path or "default"
        if model_path not in _EMBEDDER_CACHE:
            if self.cfg.environment == Environment.TESTING:
                _EMBEDDER_CACHE[model_path] = _DeterministicEmbedder()
            else:
                device = self._resolve_device()
                try:
                    sentence_transformer_cls = _import_sentence_transformer()
                except RuntimeError as exc:
                    logger.warning(
                        "SentenceTransformer unavailable (%s); using deterministic fallback embeddings.",
                        exc,
                    )
                    _EMBEDDER_CACHE[model_path] = _DeterministicEmbedder()
                else:
                    try:
                        _EMBEDDER_CACHE[model_path] = sentence_transformer_cls(model_path, device=device)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to load embedding model at %s (%s). Using deterministic fallback encoder.",
                            model_path,
                            exc,
                        )
                        _EMBEDDER_CACHE[model_path] = _DeterministicEmbedder()
        if model_path not in _EMBEDDER_CACHE:
            # Ensure we always have a fallback, even if previous branches failed silently.
            _EMBEDDER_CACHE[model_path] = _DeterministicEmbedder()
        self._embedder = _EMBEDDER_CACHE[model_path]
        return self._embedder

    def _resolve_device(self) -> str:
        backend = self.cfg.detect_gpu_backend()
        if not self.cfg.enable_gpu:
            return "cpu"
        if backend == "cuda":
            return "cuda"
        if backend == "metal":
            return "mps"
        return "cpu"


class DualSemanticChunker(BaseChunker):
    """Two-stage chunker with coarse parent blocks and fine semantic splits."""

    def __init__(self, cfg: UnifiedConfig, *, nlp=None, embedder=None):
        self.cfg = cfg
        self.base = BreakpointSemanticChunker(cfg, nlp=nlp, embedder=embedder)

    def chunk(self, text: str) -> List[Chunk]:
        sentences, spans = self.base._sentences(text)
        if not sentences:
            return []
        embeddings = self.base._embeddings(sentences)
        distances = 1.0 - self.base._pairwise_cosine(embeddings)
        parents = self._parent_boundaries(distances, sentences)
        chunks: List[Chunk] = []
        for parent_idx, (start, end) in enumerate(parents):
            subset = self.base._chunk_from_sentences(
                sentences[start:end],
                spans[start:end],
                embeddings[start:end],
                sentence_offset=start,
                text=text,
                method_label="dsc",
            )
            for ch in subset:
                ch.meta["parent_block"] = parent_idx
                ch.meta["parent_sentence_span"] = (start, end - 1)
                chunks.append(ch)
        return chunks

    def _parent_boundaries(self, distances: np.ndarray, sentences: List[str]) -> List[Tuple[int, int]]:
        cfg = self.cfg
        min_parent = max(cfg.dsc_parent_min_sentences, 2)
        max_parent = max(min_parent + 1, cfg.dsc_parent_max_sentences)
        window = max(3, cfg.dsc_delta_window)
        k = cfg.dsc_threshold_k
        use_headings = cfg.dsc_use_headings

        parents: List[Tuple[int, int]] = []
        start = 0
        t = 0
        n_sentences = len(sentences)
        while t < n_sentences - 1:
            span_len = t - start + 1
            d_t = distances[t] if t < len(distances) else 0.0
            left = max(0, t - window)
            right = min(len(distances), t + window + 1)
            window_vals = distances[left:right]
            mu = float(window_vals.mean()) if len(window_vals) else 0.0
            sigma = float(window_vals.std()) if len(window_vals) else 0.0
            theta = mu + k * sigma

            should_break = False
            heading_hint = False
            adjusted_theta = theta
            if use_headings and t + 1 < n_sentences:
                heading_hint = self._looks_like_heading(sentences[t + 1])
                if heading_hint:
                    adjusted_theta = theta * 0.85
            if span_len >= min_parent and d_t > adjusted_theta:
                should_break = True
            if span_len >= max_parent:
                should_break = True
            if heading_hint and span_len >= max(2, min_parent // 2) and (d_t > mu or span_len >= min_parent):
                should_break = True

            if should_break:
                parents.append((start, t + 1))
                start = t + 1
            t += 1

        parents.append((start, n_sentences))
        return parents

    @staticmethod
    def _looks_like_heading(sentence: str) -> bool:
        candidate = sentence.strip()
        if not candidate:
            return False
        return bool(_HEADING_PATTERN.match(candidate))


def get_chunker(config: UnifiedConfig) -> BaseChunker:
    method = (config.chunking_method or "fixed").strip().lower()
    if method == "fixed":
        return FixedChunker(max_chars=config.chunk_max_chars)
    if method == "breakpoint_semantic":
        return BreakpointSemanticChunker(config)
    if method == "dsc":
        return DualSemanticChunker(config)
    logger.warning("Unknown chunking method '%s'; falling back to fixed.", method)
    return FixedChunker(max_chars=config.chunk_max_chars)
def _import_sentence_transformer():
    global SentenceTransformer
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for semantic chunking but is not installed."
            ) from exc
        SentenceTransformer = _SentenceTransformer
    return SentenceTransformer
