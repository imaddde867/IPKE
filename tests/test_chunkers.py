"""
Tests for the pluggable chunking strategies.
"""

import pytest
from src.ai.chunkers import (
    FixedChunker,
    BreakpointSemanticChunker,
    DualSemanticChunker,
    get_chunker,

)
from src.core.unified_config import UnifiedConfig


class TestFixedChunker:
    """Tests for FixedChunker."""
    
    def test_fixed_chunker_respects_char_cap(self):
        """Verify that FixedChunker respects the character limit."""
        chunker = FixedChunker(max_chars=100)
        
        text = "This is a sentence. " * 50  # 1000 chars
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 1, "Should create multiple chunks"
        for chunk in chunks:
            assert len(chunk.text) <= 100, f"Chunk exceeds limit: {len(chunk.text)} chars"
    
    def test_fixed_chunker_empty_text(self):
        """Empty text should return no chunks."""
        chunker = FixedChunker(max_chars=100)
        chunks = chunker.chunk("")
        assert len(chunks) == 0
        
        chunks = chunker.chunk("   ")
        assert len(chunks) == 0
    
    def test_fixed_chunker_short_text(self):
        """Short text should return a single chunk."""
        chunker = FixedChunker(max_chars=100)
        text = "Short text."
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
    
    def test_fixed_chunker_metadata(self):
        """Check that metadata is set correctly."""
        chunker = FixedChunker(max_chars=100)
        text = "Sample text for testing."
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0].meta['method'] == 'fixed'
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(text)


@pytest.mark.integration
class TestBreakpointSemanticChunker:
    """Tests for BreakpointSemanticChunker (requires embeddings)."""
    
    @pytest.fixture
    def config(self):
        """Create a test config."""
        cfg = UnifiedConfig.from_environment()
        # Override to use CPU and ensure we have reasonable params
        cfg.enable_gpu = False
        cfg.gpu_backend = "cpu"
        cfg.sem_min_sentences_per_chunk = 2
        cfg.sem_max_sentences_per_chunk = 10
        cfg.sem_lambda = 0.15
        cfg.sem_window_w = 20
        cfg.chunk_max_chars = 1000
        return cfg
    
    def test_breakpoint_semantic_basic(self, config):
        """Basic functionality test."""
        chunker = BreakpointSemanticChunker(config)
        
        text = (
            "This is the first sentence. "
            "This is the second sentence. "
            "This is the third sentence. "
            "This is the fourth sentence."
        )
        
        chunks = chunker.chunk(text)
        
        # Should produce at least one chunk
        assert len(chunks) >= 1
        
        # Check metadata
        for chunk in chunks:
            assert chunk.meta['method'] == 'breakpoint_semantic'
            assert 'num_sentences' in chunk.meta
    
    def test_breakpoint_semantic_aligns_on_toy_shifts(self, config):
        """Test that semantic chunker detects topic shifts."""
        chunker = BreakpointSemanticChunker(config)
        
        # Create text with clear topic shifts
        topic_a = " ".join([
            "The car is red.",
            "The vehicle has four wheels.",
            "The automobile runs on gasoline.",
            "The sedan is parked outside.",
            "The motorized transport needs maintenance.",
        ])
        
        topic_b = " ".join([
            "The recipe requires flour.",
            "Cooking involves mixing ingredients.",
            "Baking needs precise temperatures.",
            "The kitchen has modern appliances.",
            "Culinary arts demand patience.",
        ])
        
        topic_c = " ".join([
            "Programming languages enable software development.",
            "Code must be debugged carefully.",
            "Algorithms solve computational problems.",
            "Software engineering requires planning.",
            "Testing ensures code quality.",
        ])
        
        text = f"{topic_a} {topic_b} {topic_c}"
        
        chunks = chunker.chunk(text)
        
        # Should create multiple chunks due to topic shifts
        # With λ=0.15 and clear topic differences, expect at least 2 chunks
        assert len(chunks) >= 2, "Should detect topic boundaries"
        
        # Total sentences should be preserved
        total_sents = sum(c.meta.get('num_sentences', 0) for c in chunks)
        assert total_sents >= 10, "Should preserve most sentences"
    
    def test_semantic_fallbacks_when_short_or_unembeddable(self, config):
        """Handle edge cases gracefully."""
        chunker = BreakpointSemanticChunker(config)
        
        # Very short text - single sentence
        short_text = "Hello world."
        chunks = chunker.chunk(short_text)
        assert len(chunks) == 1
        
        # Empty text
        chunks = chunker.chunk("")
        assert len(chunks) == 0
        
        # Noisy text (shouldn't crash)
        noisy_text = "!@#$ %^&* ()_+ 123 456 789"
        chunks = chunker.chunk(noisy_text)
        # Should return something without crashing
        assert isinstance(chunks, list)
    
    def test_breakpoint_enforces_char_cap(self, config):
        """Ensure chunks don't exceed character limit."""
        config.chunk_max_chars = 200
        chunker = BreakpointSemanticChunker(config)
        
        # Create long sentences
        long_sent = "This is a very long sentence with many words. " * 10
        text = " ".join([long_sent] * 5)
        
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            # Allow small margin for splitting logic (20 chars, consistent with other tests)
            assert len(chunk.text) <= config.chunk_max_chars + 20, \
                f"Chunk too long: {len(chunk.text)} chars"


@pytest.mark.integration
class TestDualSemanticChunker:
    """Tests for DualSemanticChunker (requires embeddings)."""
    
    @pytest.fixture
    def config(self):
        """Create a test config for DSC."""
        cfg = UnifiedConfig.from_environment()
        cfg.enable_gpu = False
        cfg.gpu_backend = "cpu"
        cfg.sem_min_sentences_per_chunk = 2
        cfg.sem_max_sentences_per_chunk = 10
        cfg.dsc_parent_min_sentences = 5
        cfg.dsc_parent_max_sentences = 30
        cfg.dsc_delta_window = 10
        cfg.dsc_threshold_k = 1.0
        cfg.dsc_use_headings = True
        cfg.chunk_max_chars = 1000
        return cfg
    
    def test_dsc_basic(self, config):
        """Basic DSC functionality."""
        chunker = DualSemanticChunker(config)
        
        text = " ".join([
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
            "Fourth sentence.",
            "Fifth sentence.",
            "Sixth sentence.",
        ])
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.meta['method'] == 'dsc'
    
    def test_dsc_produces_coarser_then_finer(self, config):
        """
        Test that DSC creates coarse parent blocks and refines them.
        Craft a document with 3 sections, each with two subtopics.
        """
        chunker = DualSemanticChunker(config)
        
        # Section 1: Transportation - Cars
        s1a = " ".join([
            "The car is red.",
            "The vehicle has four wheels.",
            "The automobile is fast.",
        ])
        s1b = " ".join([
            "The train is long.",
            "The locomotive runs on tracks.",
            "Railway transport is efficient.",
        ])
        
        # Section 2: Food - Cooking
        s2a = " ".join([
            "The recipe needs flour.",
            "Cooking requires heat.",
            "Baking is an art.",
        ])
        s2b = " ".join([
            "The restaurant serves meals.",
            "Dining experiences vary.",
            "Cuisine has many styles.",
        ])
        
        # Section 3: Technology - Programming
        s3a = " ".join([
            "Programming uses code.",
            "Software runs on computers.",
            "Algorithms solve problems.",
        ])
        s3b = " ".join([
            "The internet connects people.",
            "Networks transmit data.",
            "Connectivity is essential.",
        ])
        
        text = f"{s1a} {s1b} {s2a} {s2b} {s3a} {s3b}"
        
        chunks = chunker.chunk(text)
        
        # DSC should create multiple chunks
        assert len(chunks) >= 2, "DSC should create multiple chunks"
        
        # Check that parent_block metadata exists
        has_parent_meta = any('parent_block' in c.meta for c in chunks)
        assert has_parent_meta, "Should include parent block metadata"
    
    def test_dsc_heading_detection(self, config):
        """Test that DSC can use heading hints."""
        config.dsc_use_headings = True
        chunker = DualSemanticChunker(config)
        
        text = (
            "Introduction. This is the intro. "
            "We discuss many things. "
            "1. First Section "
            "This is the first section content. "
            "It has multiple sentences. "
            "We elaborate here. "
            "2. Second Section "
            "This is the second section content. "
            "It also has many sentences. "
            "More elaboration here."
        )
        
        chunks = chunker.chunk(text)
        
        # Should create chunks, possibly influenced by headings
        assert len(chunks) >= 1
    
    def test_dsc_empty_and_short(self, config):
        """DSC handles edge cases."""
        chunker = DualSemanticChunker(config)
        
        # Empty
        chunks = chunker.chunk("")
        assert len(chunks) == 0
        
        # Single sentence
        chunks = chunker.chunk("Only one sentence here.")
        assert len(chunks) == 1


class TestChunkerFactory:
    """Test the get_chunker factory function."""
    
    def test_factory_fixed(self):
        """Factory returns FixedChunker for 'fixed'."""
        cfg = UnifiedConfig.from_environment()
        cfg.chunking_method = "fixed"
        
        chunker = get_chunker(cfg)
        assert isinstance(chunker, FixedChunker)
    
    def test_factory_breakpoint_semantic(self):
        """Factory returns BreakpointSemanticChunker for 'breakpoint_semantic'."""
        cfg = UnifiedConfig.from_environment()
        cfg.chunking_method = "breakpoint_semantic"
        cfg.enable_gpu = False
        
        chunker = get_chunker(cfg)
        assert isinstance(chunker, BreakpointSemanticChunker)
    
    def test_factory_dsc(self):
        """Factory returns DualSemanticChunker for 'dsc'."""
        cfg = UnifiedConfig.from_environment()
        cfg.chunking_method = "dsc"
        cfg.enable_gpu = False
        
        chunker = get_chunker(cfg)
        assert isinstance(chunker, DualSemanticChunker)
    
    def test_factory_fallback(self):
        """Factory falls back to FixedChunker for unknown methods."""
        cfg = UnifiedConfig.from_environment()
        cfg.chunking_method = "unknown_method"
        
        chunker = get_chunker(cfg)
        assert isinstance(chunker, FixedChunker)
