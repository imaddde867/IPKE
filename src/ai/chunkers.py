"""
Pluggable text chunking strategies for IPKE.

Provides three methods:
1. FixedChunker: simple character-based chunking (default)
2. BreakpointSemanticChunker: sentence-level DP segmentation with embeddings
3. DualSemanticChunker (DSC): two-stage coarse-to-fine semantic chunking
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import re
import numpy as np
from pathlib import Path

from src.logging_config import get_logger
from src.core.unified_config import UnifiedConfig

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a single text chunk with metadata."""
    text: str
    start_char: int
    end_char: int
    sentence_span: Tuple[int, int]  # [i, j] inclusive sentence indices
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Chunk the given text and return a list of Chunk objects."""
        pass


class FixedChunker(BaseChunker):
    """
    Simple fixed-size chunking that splits on whitespace boundaries.
    Preserves existing behavior.
    """
    
    def __init__(self, max_chars: int = 2000):
        self.max_chars = max_chars
    
    def chunk(self, text: str) -> List[Chunk]:
        """Split text into fixed-size chunks at whitespace boundaries."""
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end position
            end = start + self.max_chars
            
            if end >= len(text):
                # Last chunk
                chunk_text = text[start:]
                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=len(text),
                    sentence_span=(-1, -1),  # Not sentence-based
                    meta={'method': 'fixed'}
                ))
                break
            
            # Try to break at whitespace
            # Look backwards from end to find whitespace
            break_pos = end
            for i in range(end - 1, max(start, end - 100), -1):
                if text[i].isspace():
                    break_pos = i
                    break
            
            chunk_text = text[start:break_pos].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=break_pos,
                    sentence_span=(-1, -1),
                    meta={'method': 'fixed'}
                ))
            
            start = break_pos
            # Skip whitespace
            while start < len(text) and text[start].isspace():
                start += 1
        
        return chunks


class BreakpointSemanticChunker(BaseChunker):
    """
    Sentence-level DP segmentation using embeddings.
    
    Maximizes average within-segment similarity minus a break penalty (λ).
    Uses dynamic programming with window constraint.
    """
    
    def __init__(self, cfg: UnifiedConfig):
        self.cfg = cfg
        self.nlp = None
        self.model = None
        self._initialized = False
    
    def _load_spacy(self):
        """Load spaCy model with sentencizer."""
        if self.nlp is not None:
            return self.nlp
        
        try:
            import spacy
            self.nlp = spacy.load(self.cfg.spacy_model)
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
            logger.info(f"Loaded spaCy model: {self.cfg.spacy_model}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise RuntimeError(f"spaCy model '{self.cfg.spacy_model}' not available. "
                             f"Install via: python -m spacy download {self.cfg.spacy_model}") from e
        return self.nlp
    
    def _load_embedder(self):
        """Load SentenceTransformer model."""
        if self.model is not None:
            return self.model
        
        try:
            from sentence_transformers import SentenceTransformer
            
            model_path = Path(self.cfg.embedding_model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Embedding model not found at {model_path}. "
                    f"Please ensure the model is downloaded to this path."
                )
            
            # Determine device based on config
            backend = self.cfg.detect_gpu_backend()
            if backend == "cuda":
                device = "cuda"
            elif backend == "metal":
                device = "mps"  # PyTorch device for Metal
            else:
                device = "cpu"
            
            # Try to use the detected device, fall back to CPU if needed
            try:
                self.model = SentenceTransformer(str(model_path), device=device)
            except Exception as e:
                logger.warning(f"Failed to load model on {device}, falling back to CPU: {e}")
                self.model = SentenceTransformer(str(model_path), device="cpu")
            
            logger.info(f"Loaded embedding model from {model_path} on device {self.model.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(
                f"Embedding model not available at {self.cfg.embedding_model_path}. "
                f"Please download the model or check the path."
            ) from e
        
        return self.model
    
    def _sentences(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Tokenize text into sentences using spaCy.
        Returns: (list of sentence strings, list of (start, end) char spans)
        """
        nlp = self._load_spacy()
        doc = nlp(text)
        
        sents = []
        spans = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                sents.append(sent_text)
                spans.append((sent.start_char, sent.end_char))
        
        return sents, spans
    
    def _embeddings(self, sents: List[str]) -> np.ndarray:
        """
        Compute L2-normalized embeddings for sentences.
        Returns: array of shape (n, d)
        """
        model = self._load_embedder()
        
        # Encode sentences
        embeddings = model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings
    
    def _splice_chunks(
        self,
        boundaries: List[Tuple[int, int]],
        sents: List[str],
        spans: List[Tuple[int, int]],
        text: str
    ) -> List[Chunk]:
        """
        Create Chunk objects from sentence boundaries.
        
        Args:
            boundaries: List of (start_sent_idx, end_sent_idx) tuples
            sents: List of sentence strings
            spans: List of (start_char, end_char) for each sentence
            text: Original text
        """
        chunks = []
        
        for i, j in boundaries:
            if i >= j or i >= len(sents):
                continue
            
            # Adjust j to be inclusive max index
            j = min(j, len(sents))
            
            # Get character span
            start_char = spans[i][0]
            end_char = spans[j - 1][1]
            
            chunk_text = text[start_char:end_char].strip()
            
            if not chunk_text:
                continue
            
            # Calculate cohesion if DEBUG_CHUNKING is enabled
            meta = {
                'method': 'breakpoint_semantic',
                'num_sentences': j - i,
            }
            
            if self.cfg.debug_chunking:
                meta['sentence_indices'] = (i, j - 1)
            
            chunks.append(Chunk(
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                sentence_span=(i, j - 1),
                meta=meta
            ))
        
        return chunks
    
    def _enforce_char_cap(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Split any chunks exceeding CHUNK_MAX_CHARS on sentence boundaries.
        """
        max_chars = self.cfg.chunk_max_chars
        result = []
        
        for chunk in chunks:
            if len(chunk.text) <= max_chars:
                result.append(chunk)
                continue
            
            # Need to split this chunk
            # Re-tokenize into sentences
            sents, spans_local = self._sentences(chunk.text)
            
            if not sents:
                result.append(chunk)
                continue
            
            # Greedily pack sentences into sub-chunks
            current_sents = []
            current_start_idx = 0
            
            for idx, sent in enumerate(sents):
                test_text = ' '.join(current_sents + [sent])
                
                if len(test_text) <= max_chars:
                    current_sents.append(sent)
                else:
                    # Flush current sub-chunk
                    if current_sents:
                        sub_text = ' '.join(current_sents)
                        sub_start = chunk.start_char + spans_local[current_start_idx][0]
                        sub_end = chunk.start_char + spans_local[idx - 1][1]
                        result.append(Chunk(
                            text=sub_text,
                            start_char=sub_start,
                            end_char=sub_end,
                            sentence_span=(chunk.sentence_span[0] + current_start_idx,
                                         chunk.sentence_span[0] + idx - 1),
                            meta={'method': 'breakpoint_semantic', 'split_from_large': True}
                        ))
                    
                    # Start new sub-chunk
                    current_sents = [sent]
                    current_start_idx = idx
            
            # Flush remaining
            if current_sents:
                sub_text = ' '.join(current_sents)
                sub_start = chunk.start_char + spans_local[current_start_idx][0]
                sub_end = chunk.start_char + spans_local[-1][1]
                result.append(Chunk(
                    text=sub_text,
                    start_char=sub_start,
                    end_char=sub_end,
                    sentence_span=(chunk.sentence_span[0] + current_start_idx,
                                 chunk.sentence_span[0] + len(sents) - 1),
                    meta={'method': 'breakpoint_semantic', 'split_from_large': True}
                ))
        
        return result
    
    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk text using DP-based semantic segmentation.
        """
        if not text or not text.strip():
            return []
        
        sents, sent_spans = self._sentences(text)
        
        if not sents:
            return []
        
        # Single sentence: return as one chunk
        if len(sents) == 1:
            return [Chunk(
                text=sents[0],
                start_char=sent_spans[0][0],
                end_char=sent_spans[0][1],
                sentence_span=(0, 0),
                meta={'method': 'breakpoint_semantic', 'num_sentences': 1}
            )]
        
        # Compute embeddings
        E = self._embeddings(sents)
        n = len(sents)
        
        # Cosine similarity between consecutive sentences
        sim = (E[:-1] @ E[1:].T).diagonal()
        
        # Prefix sums for O(1) range queries
        pref = np.concatenate(([0.0], np.cumsum(sim)))
        
        # DP parameters
        w = self.cfg.sem_window_w
        lam = self.cfg.sem_lambda
        min_len = self.cfg.sem_min_sentences_per_chunk
        max_len = self.cfg.sem_max_sentences_per_chunk
        
        # DP arrays
        DP = np.full(n + 1, -1e9)
        DP[0] = 0.0
        prev = np.full(n + 1, -1, dtype=int)
        
        for j in range(1, n + 1):
            i_start = max(0, j - w)
            for i in range(i_start, j):
                length = j - i
                if length < min_len or length > max_len:
                    continue
                
                # c(i,j) = mean similarity over consecutive pairs in [i, j-1]
                pairs = max(1, length - 1)
                seg_sum = pref[j - 1] - pref[i]
                c_ij = seg_sum / pairs
                
                val = DP[i] + c_ij - lam
                if val > DP[j]:
                    DP[j] = val
                    prev[j] = i
        
        # Backtrack to find boundaries
        boundaries = []
        j = n
        while j > 0:
            i = prev[j]
            if i < 0:
                # Fallback: force a chunk
                i = max(0, j - max_len)
            boundaries.append((i, j))
            j = i
        
        boundaries.reverse()
        
        # Build chunks
        chunks = self._splice_chunks(boundaries, sents, sent_spans, text)
        
        # Enforce character cap
        return self._enforce_char_cap(chunks)


class DualSemanticChunker(BaseChunker):
    """
    Two-stage chunking:
    1. Coarse parent blocks via adaptive distance threshold (θ = μ + k·σ)
    2. Fine-grained breakpoint semantic within each parent block
    """
    
    def __init__(self, cfg: UnifiedConfig):
        self.cfg = cfg
        self.base = BreakpointSemanticChunker(cfg)
    
    def _is_heading(self, sent: str) -> bool:
        """
        Check if a sentence looks like a heading/section marker.
        Patterns: numbered lists, Roman numerals, capitalized short phrases.
        """
        sent = sent.strip()
        if len(sent) > 100:
            return False
        
        # Patterns: "1.2.3 Title", "IV. Section", "SECTION TITLE:", etc.
        patterns = [
            r'^\s*\d+(\.\d+)*[\s:]',  # 1. or 1.2.3
            r'^\s*[IVXLC]+\.?\s',  # Roman numerals
            r'^[A-Z][A-Z\s]{2,60}:\s*$',  # ALL CAPS TITLE:
        ]
        
        for pattern in patterns:
            if re.match(pattern, sent):
                return True
        
        return False
    
    def _parent_boundaries(
        self,
        d: np.ndarray,
        sents: List[str],
        spans: List[Tuple[int, int]],
        text: str
    ) -> List[Tuple[int, int]]:
        """
        Compute coarse parent boundaries using adaptive threshold.
        
        Args:
            d: distance array (1 - cosine similarity) between consecutive sentences
            sents: list of sentence strings
            spans: list of (start_char, end_char) for each sentence
            text: original text
        
        Returns:
            List of (start_sent_idx, end_sent_idx) for parent blocks
        """
        n = len(sents)
        if n == 0:
            return []
        
        min_parent = self.cfg.dsc_parent_min_sentences
        max_parent = self.cfg.dsc_parent_max_sentences
        delta_window = self.cfg.dsc_delta_window
        k = self.cfg.dsc_threshold_k
        use_headings = self.cfg.dsc_use_headings
        
        boundaries = []
        last_boundary = 0
        
        for t in range(len(d)):
            # Local statistics over distance deltas
            window_start = max(0, t - delta_window)
            window_end = min(len(d), t + delta_window)
            local_d = d[window_start:window_end]
            
            mu = np.mean(local_d)
            sigma = np.std(local_d)
            threshold = mu + k * sigma
            
            # Check if we should trigger a boundary
            since_last = (t + 1) - last_boundary
            
            # Force break if we exceed max_parent
            if since_last >= max_parent:
                boundaries.append((last_boundary, t + 1))
                last_boundary = t + 1
                continue
            
            # Don't break before min_parent
            if since_last < min_parent:
                continue
            
            # Check distance threshold
            if d[t] > threshold:
                # Optional: bias with heading detection
                if use_headings and (t + 1 < n):
                    # Check if next sentence looks like a heading
                    if self._is_heading(sents[t + 1]):
                        boundaries.append((last_boundary, t + 1))
                        last_boundary = t + 1
                        continue
                
                # Regular threshold-based break
                boundaries.append((last_boundary, t + 1))
                last_boundary = t + 1
        
        # Final boundary
        if last_boundary < n:
            boundaries.append((last_boundary, n))
        
        return boundaries
    
    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk text using two-stage DSC approach.
        """
        if not text or not text.strip():
            return []
        
        # Get sentences and embeddings (reuse base chunker's methods)
        sents, spans = self.base._sentences(text)
        
        if not sents:
            return []
        
        if len(sents) == 1:
            return [Chunk(
                text=sents[0],
                start_char=spans[0][0],
                end_char=spans[0][1],
                sentence_span=(0, 0),
                meta={'method': 'dsc', 'num_sentences': 1}
            )]
        
        # Compute embeddings
        E = self.base._embeddings(sents)
        
        # Compute distances (1 - cosine similarity)
        d = 1.0 - (E[:-1] @ E[1:].T).diagonal()
        
        # Find parent boundaries
        parents = self._parent_boundaries(d, sents, spans, text)
        
        # Refine each parent with breakpoint semantic
        out: List[Chunk] = []
        for i, j in parents:
            # Extract sub-text for this parent
            sub_start_char = spans[i][0]
            sub_end_char = spans[j - 1][1]
            sub_text = text[sub_start_char:sub_end_char]
            
            # Chunk within this parent
            sub_chunks = self.base.chunk(sub_text)
            
            # Adjust offsets to global coordinates
            for ch in sub_chunks:
                ch.start_char += sub_start_char
                ch.end_char += sub_start_char
                ch.meta['method'] = 'dsc'
                ch.meta['parent_block'] = (i, j - 1)
                out.append(ch)
        
        # Enforce char cap (already done by base, but double-check)
        return self.base._enforce_char_cap(out)


def get_chunker(cfg: UnifiedConfig) -> BaseChunker:
    """
    Factory function to create the appropriate chunker based on config.
    
    Args:
        cfg: UnifiedConfig instance
    
    Returns:
        BaseChunker instance
    """
    method = cfg.chunking_method.lower()
    
    if method == "fixed":
        return FixedChunker(max_chars=cfg.chunk_max_chars)
    elif method == "breakpoint_semantic":
        return BreakpointSemanticChunker(cfg)
    elif method == "dsc":
        return DualSemanticChunker(cfg)
    else:
        logger.warning(f"Unknown chunking method '{method}', falling back to 'fixed'")
        return FixedChunker(max_chars=cfg.chunk_max_chars)
