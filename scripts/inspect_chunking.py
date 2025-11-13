#!/usr/bin/env python3
"""
CLI tool to inspect and visualize chunking strategies.

Usage:
    python scripts/inspect_chunking.py --method fixed --text-file sample.txt
    python scripts/inspect_chunking.py --method breakpoint_semantic --text "Your text here"
    python scripts/inspect_chunking.py --method dsc --text-file doc.txt --lambda 0.2
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.unified_config import UnifiedConfig, get_config, reload_config
from src.ai.chunkers import get_chunker
import os


def main():
    parser = argparse.ArgumentParser(
        description="Inspect text chunking strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--method",
        choices=["fixed", "breakpoint_semantic", "dsc"],
        default="fixed",
        help="Chunking method to use"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Text to chunk (inline)")
    input_group.add_argument("--text-file", type=Path, help="Path to text file")
    
    # Override parameters
    parser.add_argument("--max-chars", type=int, help="Override CHUNK_MAX_CHARS")
    parser.add_argument("--lambda", dest="sem_lambda", type=float, help="Override SEM_LAMBDA")
    parser.add_argument("--window", type=int, help="Override SEM_WINDOW_W")
    parser.add_argument("--min-sentences", type=int, help="Override SEM_MIN_SENTENCES_PER_CHUNK")
    parser.add_argument("--max-sentences", type=int, help="Override SEM_MAX_SENTENCES_PER_CHUNK")
    parser.add_argument("--dsc-k", type=float, help="Override DSC_THRESHOLD_K")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Get text
    if args.text:
        text = args.text
    else:
        if not args.text_file.exists():
            print(f"Error: File not found: {args.text_file}", file=sys.stderr)
            return 1
        text = args.text_file.read_text(encoding='utf-8')
    
    # Set environment overrides
    os.environ['CHUNKING_METHOD'] = args.method
    
    if args.max_chars:
        os.environ['CHUNK_MAX_CHARS'] = str(args.max_chars)
    if args.sem_lambda is not None:
        os.environ['SEM_LAMBDA'] = str(args.sem_lambda)
    if args.window:
        os.environ['SEM_WINDOW_W'] = str(args.window)
    if args.min_sentences:
        os.environ['SEM_MIN_SENTENCES_PER_CHUNK'] = str(args.min_sentences)
    if args.max_sentences:
        os.environ['SEM_MAX_SENTENCES_PER_CHUNK'] = str(args.max_sentences)
    if args.dsc_k is not None:
        os.environ['DSC_THRESHOLD_K'] = str(args.dsc_k)
    
    # Force CPU for inspection
    os.environ['ENABLE_GPU'] = 'false'
    os.environ['GPU_BACKEND'] = 'cpu'
    
    # Reload config to pick up env vars
    config = reload_config()
    
    # Create chunker
    print(f"Using chunking method: {config.chunking_method}")
    if args.verbose:
        print(f"Config: chunk_max_chars={config.chunk_max_chars}")
        if config.chunking_method != "fixed":
            print(f"  sem_lambda={config.sem_lambda}")
            print(f"  sem_window_w={config.sem_window_w}")
            print(f"  sem_min_sentences={config.sem_min_sentences_per_chunk}")
            print(f"  sem_max_sentences={config.sem_max_sentences_per_chunk}")
        if config.chunking_method == "dsc":
            print(f"  dsc_threshold_k={config.dsc_threshold_k}")
            print(f"  dsc_parent_min={config.dsc_parent_min_sentences}")
            print(f"  dsc_parent_max={config.dsc_parent_max_sentences}")
    
    print(f"\nInput text length: {len(text)} characters")
    print("=" * 60)
    
    chunker = get_chunker(config)
    chunks = chunker.chunk(text)
    
    print(f"\nGenerated {len(chunks)} chunks:\n")
    
    for i, chunk in enumerate(chunks, 1):
        num_sents = chunk.meta.get('num_sentences', 'N/A')
        method = chunk.meta.get('method', 'unknown')
        
        print(f"Chunk #{i} ({method}):")
        print(f"  Characters: {len(chunk.text)}")
        print(f"  Sentences: {num_sents}")
        print(f"  Span: [{chunk.start_char}, {chunk.end_char})")
        print(f"  Sentence span: {chunk.sentence_span}")
        
        if args.verbose and chunk.meta:
            print(f"  Metadata: {chunk.meta}")
        
        # Preview
        preview = chunk.text[:100]
        if len(chunk.text) > 100:
            preview += "..."
        print(f"  Preview: {preview}")
        print()
    
    # Summary statistics
    if chunks:
        avg_len = sum(len(c.text) for c in chunks) / len(chunks)
        min_len = min(len(c.text) for c in chunks)
        max_len = max(len(c.text) for c in chunks)
        
        print("=" * 60)
        print("Summary Statistics:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Avg chunk length: {avg_len:.0f} chars")
        print(f"  Min chunk length: {min_len} chars")
        print(f"  Max chunk length: {max_len} chars")
        
        # Sentence stats if available
        sent_counts = [c.meta.get('num_sentences', 0) for c in chunks if 'num_sentences' in c.meta]
        if sent_counts:
            avg_sents = sum(sent_counts) / len(sent_counts)
            print(f"  Avg sentences/chunk: {avg_sents:.1f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
