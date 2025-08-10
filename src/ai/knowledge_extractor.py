"""
EXPLAINIUM - Knowledge Extractor (Compatibility Wrapper)

Backward compatibility wrapper that replaces the old superficial BERT/BART models
with the new AdvancedKnowledgeEngine while maintaining existing API compatibility.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

# Internal imports
from src.logging_config import get_logger
from src.ai.advanced_knowledge_engine import (
    AdvancedKnowledgeEngine, 
    KnowledgeEntity, 
    KnowledgeRelationship, 
    WorkflowProcess, 
    TacitKnowledge
)

logger = get_logger(__name__)


class KnowledgeExtractor:
    """
    Compatibility wrapper for the old KnowledgeExtractor that now uses
    the advanced AI-powered knowledge processing engine.
    
    This maintains backward compatibility while providing sophisticated
    knowledge extraction capabilities.
    """
    
    def __init__(self):
        """Initialize the advanced knowledge extraction engine"""
        try:
            self.advanced_engine = AdvancedKnowledgeEngine()
            logger.info("KnowledgeExtractor initialized with advanced AI engine")
        except Exception as e:
            logger.error(f"Failed to initialize advanced knowledge engine: {e}")
            self.advanced_engine = None
    
    async def extract_knowledge(self, content: str, document_id: str = None) -> Dict[str, Any]:
        """
        Extract knowledge from content using advanced AI models
        
        Args:
            content: Text content to analyze
            document_id: Optional document identifier
            
        Returns:
            Dictionary containing extracted knowledge with backward compatibility
        """
        try:
            if not self.advanced_engine:
                logger.error("Advanced knowledge engine not available")
                return self._fallback_extraction(content)
            
            # Prepare document for advanced processing
            document = {
                'id': document_id or 'unknown',
                'content': content,
                'metadata': {
                    'processing_timestamp': datetime.now().isoformat(),
                    'extractor_version': 'advanced_v2.0'
                }
            }
            
            # Use advanced knowledge extraction
            results = await self.advanced_engine.extract_deep_knowledge(document)
            
            # Convert to backward-compatible format
            compatible_results = self._convert_to_legacy_format(results)
            
            logger.info(f"Successfully extracted knowledge from document {document_id}")
            return compatible_results
            
        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            return self._fallback_extraction(content)
    
    def extract_knowledge_sync(self, content: str, document_id: str = None) -> Dict[str, Any]:
        """
        Synchronous wrapper for knowledge extraction
        """
        try:
            # Run the async method in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.extract_knowledge(content, document_id)
                )
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Synchronous knowledge extraction failed: {e}")
            return self._fallback_extraction(content)
    
    def _convert_to_legacy_format(self, advanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert advanced extraction results to legacy format for backward compatibility
        """
        try:
            legacy_results = {
                'entities': [],
                'processes': [],
                'relationships': [],
                'insights': {},
                'confidence_scores': {},
                'extraction_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'engine': 'advanced_ai',
                    'version': '2.0'
                }
            }
            
            # Convert entities
            for entity in advanced_results.get('entities', []):
                legacy_entity = {
                    'id': getattr(entity, 'id', ''),
                    'name': getattr(entity, 'name', ''),
                    'type': getattr(entity, 'type', 'unknown'),
                    'description': getattr(entity, 'description', ''),
                    'confidence': getattr(entity, 'confidence', 0.0),
                    'source_documents': getattr(entity, 'source_documents', []),
                    'properties': getattr(entity, 'properties', {}),
                    'created_at': getattr(entity, 'created_at', datetime.now()).isoformat() if hasattr(entity, 'created_at') else datetime.now().isoformat()
                }
                legacy_results['entities'].append(legacy_entity)
            
            # Convert processes
            for process in advanced_results.get('processes', []):
                legacy_process = {
                    'id': getattr(process, 'id', ''),
                    'name': getattr(process, 'name', ''),
                    'description': getattr(process, 'description', ''),
                    'steps': getattr(process, 'steps', []),
                    'roles_involved': getattr(process, 'roles_involved', []),
                    'complexity_score': getattr(process, 'complexity_score', 0.0),
                    'automation_potential': getattr(process, 'automation_potential', 0.0)
                }
                legacy_results['processes'].append(legacy_process)
            
            # Convert relationships
            for relationship in advanced_results.get('relationships', []):
                legacy_relationship = {
                    'source_id': getattr(relationship, 'source_id', ''),
                    'target_id': getattr(relationship, 'target_id', ''),
                    'relationship_type': getattr(relationship, 'relationship_type', 'unknown'),
                    'strength': getattr(relationship, 'strength', 0.0),
                    'confidence': getattr(relationship, 'confidence', 0.0),
                    'context': getattr(relationship, 'context', ''),
                    'properties': getattr(relationship, 'properties', {})
                }
                legacy_results['relationships'].append(legacy_relationship)
            
            # Extract insights
            legacy_results['insights'] = advanced_results.get('insights', {})
            
            # Calculate confidence scores
            entity_confidences = [e['confidence'] for e in legacy_results['entities']]
            relationship_confidences = [r['confidence'] for r in legacy_results['relationships']]
            
            legacy_results['confidence_scores'] = {
                'average_entity_confidence': sum(entity_confidences) / len(entity_confidences) if entity_confidences else 0.0,
                'average_relationship_confidence': sum(relationship_confidences) / len(relationship_confidences) if relationship_confidences else 0.0,
                'high_confidence_entities': len([c for c in entity_confidences if c > 0.8]),
                'high_confidence_relationships': len([c for c in relationship_confidences if c > 0.8])
            }
            
            return legacy_results
            
        except Exception as e:
            logger.error(f"Error converting to legacy format: {e}")
            return self._fallback_extraction("")
    
    def _fallback_extraction(self, content: str) -> Dict[str, Any]:
        """
        Simple fallback extraction when advanced engine fails
        """
        return {
            'entities': [],
            'processes': [],
            'relationships': [],
            'insights': {},
            'confidence_scores': {
                'average_entity_confidence': 0.0,
                'average_relationship_confidence': 0.0,
                'high_confidence_entities': 0,
                'high_confidence_relationships': 0
            },
            'extraction_metadata': {
                'timestamp': datetime.now().isoformat(),
                'engine': 'fallback',
                'version': '1.0',
                'error': 'Advanced engine unavailable'
            }
        }
    
    def get_supported_domains(self) -> List[str]:
        """
        Get list of supported knowledge domains
        """
        return [
            'operational',
            'safety_compliance',
            'equipment_technology',
            'human_resources',
            'quality_assurance',
            'maintenance',
            'training',
            'regulatory',
            'environmental',
            'financial'
        ]
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge extraction capabilities
        """
        if self.advanced_engine:
            return {
                'engine_type': 'advanced_ai',
                'version': '2.0',
                'capabilities': [
                    'deep_entity_extraction',
                    'process_workflow_analysis',
                    'relationship_mapping',
                    'tacit_knowledge_discovery',
                    'semantic_understanding',
                    'knowledge_graph_building'
                ],
                'supported_domains': self.get_supported_domains(),
                'model_info': self.advanced_engine.get_knowledge_summary()
            }
        else:
            return {
                'engine_type': 'fallback',
                'version': '1.0',
                'capabilities': ['basic_text_processing'],
                'supported_domains': [],
                'model_info': {}
            }
    
    async def extract_processes(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract business processes from content
        """
        try:
            results = await self.extract_knowledge(content)
            return results.get('processes', [])
        except Exception as e:
            logger.error(f"Process extraction failed: {e}")
            return []
    
    async def extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract entities from content
        """
        try:
            results = await self.extract_knowledge(content)
            return results.get('entities', [])
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def extract_relationships(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract relationships from content
        """
        try:
            results = await self.extract_knowledge(content)
            return results.get('relationships', [])
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []
    
    def cleanup(self):
        """
        Clean up resources used by the knowledge extractor
        """
        try:
            if self.advanced_engine:
                self.advanced_engine.cleanup_memory()
            logger.info("KnowledgeExtractor cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Backward compatibility aliases
# These ensure existing code continues to work
ExtractedProcess = dict
ExtractedDecisionPoint = dict
ExtractedComplianceItem = dict
ExtractedRiskAssessment = dict


def create_knowledge_extractor() -> KnowledgeExtractor:
    """
    Factory function to create a knowledge extractor instance
    """
    return KnowledgeExtractor()


# Legacy enum classes for backward compatibility
class KnowledgeDomain:
    OPERATIONAL = "operational"
    SAFETY_COMPLIANCE = "safety_compliance"
    EQUIPMENT_TECHNOLOGY = "equipment_technology"
    HUMAN_RESOURCES = "human_resources"
    QUALITY_ASSURANCE = "quality_assurance"
    MAINTENANCE = "maintenance"
    TRAINING = "training"
    REGULATORY = "regulatory"
    ENVIRONMENTAL = "environmental"
    FINANCIAL = "financial"


class HierarchyLevel:
    CORE_FUNCTION = "core_function"
    OPERATION = "operation"
    PROCEDURE = "procedure"
    SPECIFIC_STEP = "specific_step"


class CriticalityLevel:
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"