"""
EXPLAINIUM - Document Processor (Compatibility Wrapper)

Backward compatibility wrapper for document processing that now uses
the advanced IntelligentDocumentProcessor with multi-modal capabilities.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio

# Internal imports
from src.logging_config import get_logger
from src.processors.intelligent_document_processor import IntelligentDocumentProcessor
from src.exceptions import ProcessingError

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Compatibility wrapper for the old DocumentProcessor that now uses
    the advanced IntelligentDocumentProcessor with multi-modal capabilities.
    
    This maintains backward compatibility while providing sophisticated
    document processing and knowledge extraction.
    """
    
    def __init__(self):
        """Initialize the intelligent document processor"""
        try:
            self.intelligent_processor = IntelligentDocumentProcessor()
            logger.info("DocumentProcessor initialized with intelligent multi-modal processing")
        except Exception as e:
            logger.error(f"Failed to initialize intelligent processor: {e}")
            self.intelligent_processor = None
    
    def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        """
        Synchronous wrapper for document processing (backward compatibility)
        
        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            
        Returns:
            Dictionary containing processing results
        """
        try:
            if not self.intelligent_processor:
                logger.error("Intelligent processor not available")
                return self._fallback_processing(file_path, document_id)
            
            # Run the async method in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.intelligent_processor.process_document(file_path, document_id)
                )
                
                # Convert to legacy format for backward compatibility
                legacy_result = self._convert_to_legacy_format(result)
                return legacy_result
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return self._fallback_processing(file_path, document_id)
    
    async def process_document_async(self, file_path: str, document_id: int, 
                                   company_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Asynchronous document processing with enhanced capabilities
        
        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            company_context: Optional company context for better extraction
            
        Returns:
            Dictionary containing comprehensive processing results
        """
        try:
            if not self.intelligent_processor:
                return self._fallback_processing(file_path, document_id)
            
            result = await self.intelligent_processor.process_document(
                file_path, document_id, company_context
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Async document processing failed: {e}")
            return self._fallback_processing(file_path, document_id)
    
    def _convert_to_legacy_format(self, intelligent_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert intelligent processor results to legacy format for backward compatibility
        """
        try:
            # Extract key information for legacy format
            legacy_result = {
                'document_id': intelligent_result.get('document_id'),
                'filename': intelligent_result.get('filename'),
                'file_type': intelligent_result.get('file_type'),
                'content': intelligent_result.get('extraction_metadata', {}).get('text', ''),
                'extracted_text': intelligent_result.get('extraction_metadata', {}).get('text', ''),
                'processing_status': 'completed' if not intelligent_result.get('error') else 'error',
                'processing_timestamp': intelligent_result.get('processing_timestamp'),
                'file_size': os.path.getsize(intelligent_result.get('filename', '')) if intelligent_result.get('filename') and os.path.exists(intelligent_result.get('filename', '')) else 0,
                'page_count': intelligent_result.get('extraction_metadata', {}).get('metadata', {}).get('pages', 1),
                'word_count': len(intelligent_result.get('extraction_metadata', {}).get('text', '').split()),
                'extraction_method': intelligent_result.get('extraction_metadata', {}).get('method', 'intelligent'),
                'confidence': intelligent_result.get('extraction_metadata', {}).get('confidence', 1.0),
                'error': intelligent_result.get('error'),
                
                # Enhanced fields from intelligent processing
                'knowledge_extraction': intelligent_result.get('knowledge_extraction', {}),
                'operational_intelligence': intelligent_result.get('operational_intelligence', {}),
                'knowledge_graph_summary': intelligent_result.get('knowledge_graph_summary', {}),
                'confidence_scores': intelligent_result.get('confidence_scores', {}),
                
                # Legacy compatibility fields
                'entities': [],
                'processes': [],
                'relationships': [],
                'insights': {}
            }
            
            # Extract entities, processes, and relationships for legacy compatibility
            knowledge_extraction = intelligent_result.get('knowledge_extraction', {})
            
            # Convert entities
            for entity in knowledge_extraction.get('entities', []):
                legacy_entity = {
                    'id': getattr(entity, 'id', ''),
                    'name': getattr(entity, 'name', ''),
                    'type': getattr(entity, 'type', 'unknown'),
                    'description': getattr(entity, 'description', ''),
                    'confidence': getattr(entity, 'confidence', 0.0)
                }
                legacy_result['entities'].append(legacy_entity)
            
            # Convert processes
            for process in knowledge_extraction.get('processes', []):
                legacy_process = {
                    'id': getattr(process, 'id', ''),
                    'name': getattr(process, 'name', ''),
                    'description': getattr(process, 'description', ''),
                    'steps': getattr(process, 'steps', [])
                }
                legacy_result['processes'].append(legacy_process)
            
            # Convert relationships
            for relationship in knowledge_extraction.get('relationships', []):
                legacy_relationship = {
                    'source_id': getattr(relationship, 'source_id', ''),
                    'target_id': getattr(relationship, 'target_id', ''),
                    'relationship_type': getattr(relationship, 'relationship_type', 'unknown'),
                    'confidence': getattr(relationship, 'confidence', 0.0)
                }
                legacy_result['relationships'].append(legacy_relationship)
            
            # Extract insights
            legacy_result['insights'] = knowledge_extraction.get('insights', {})
            
            return legacy_result
            
        except Exception as e:
            logger.error(f"Error converting to legacy format: {e}")
            return self._fallback_processing("", 0)
    
    def _fallback_processing(self, file_path: str, document_id: int) -> Dict[str, Any]:
        """
        Simple fallback processing when intelligent processor fails
        """
        try:
            # Basic text extraction as fallback
            content = ""
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read()
                    except Exception:
                        content = "Unable to read file content"
            
            return {
                'document_id': document_id,
                'filename': os.path.basename(file_path) if file_path else 'unknown',
                'file_type': 'text',
                'content': content,
                'extracted_text': content,
                'processing_status': 'completed_fallback',
                'processing_timestamp': '',
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'page_count': 1,
                'word_count': len(content.split()),
                'extraction_method': 'fallback',
                'confidence': 0.5,
                'error': None,
                'entities': [],
                'processes': [],
                'relationships': [],
                'insights': {},
                'knowledge_extraction': {},
                'operational_intelligence': {},
                'knowledge_graph_summary': {},
                'confidence_scores': {}
            }
            
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            return {
                'document_id': document_id,
                'error': str(e),
                'processing_status': 'error',
                'entities': [],
                'processes': [],
                'relationships': [],
                'insights': {}
            }
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats
        """
        if self.intelligent_processor:
            formats = self.intelligent_processor.get_supported_formats()
            # Flatten the format dictionary for legacy compatibility
            all_formats = []
            for format_list in formats.values():
                all_formats.extend(format_list)
            return all_formats
        else:
            return ['.txt', '.pdf', '.doc', '.docx']
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics and capabilities
        """
        if self.intelligent_processor:
            return self.intelligent_processor.get_processing_stats()
        else:
            return {
                'supported_formats': self.get_supported_formats(),
                'processor_type': 'fallback',
                'capabilities': ['basic_text_extraction']
            }
    
    async def process_batch_documents(self, file_paths: List[str], 
                                    company_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process multiple documents in optimized batches
        """
        if self.intelligent_processor:
            return await self.intelligent_processor.process_batch_documents(
                file_paths, company_context
            )
        else:
            # Fallback batch processing
            results = []
            for i, file_path in enumerate(file_paths):
                result = self.process_document(file_path, i)
                results.append(result)
            return results
    
    def extract_text(self, file_path: str) -> str:
        """
        Simple text extraction for backward compatibility
        """
        try:
            result = self.process_document(file_path, 0)
            return result.get('extracted_text', '')
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def is_supported_format(self, file_path: str) -> bool:
        """
        Check if file format is supported
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            supported_formats = self.get_supported_formats()
            return file_ext in supported_formats
        except Exception:
            return False
    
    def cleanup(self):
        """
        Clean up resources used by the processor
        """
        try:
            if self.intelligent_processor:
                # The intelligent processor doesn't have cleanup, but we can clear references
                pass
            logger.info("DocumentProcessor cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Factory function for backward compatibility
def create_document_processor() -> DocumentProcessor:
    """
    Factory function to create a document processor instance
    """
    return DocumentProcessor()