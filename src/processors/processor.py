"""
EXPLAINIUM - Main Processor (Redirects to Streamlined Version)

This file now redirects to the new streamlined processor for backward compatibility.
For new development, use StreamlinedDocumentProcessor directly.
"""

# Import the new streamlined processor
from src.processors.streamlined_processor import (
    StreamlinedDocumentProcessor, 
    ProcessingResult,
    create_document_processor
)

# Backward compatibility alias
OptimizedDocumentProcessor = StreamlinedDocumentProcessor

# Legacy support - redirect old class name to new implementation
class DocumentProcessor(StreamlinedDocumentProcessor):
    """Legacy compatibility class that redirects to StreamlinedDocumentProcessor"""
    
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "DocumentProcessor is deprecated. Use StreamlinedDocumentProcessor instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

# Export for backward compatibility
__all__ = [
    'StreamlinedDocumentProcessor',
    'OptimizedDocumentProcessor', 
    'DocumentProcessor',
    'ProcessingResult',
    'create_document_processor'
]