"""
EXPLAINIUM - Database Output Generator

Phase 3: Database-Optimized Output Generation
Generate clean, normalized data entries for direct database ingestion with
synthesis over extraction, context preservation, semantic clarity, and business relevance.
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
from enum import Enum

# Database imports
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

# Internal imports
from src.logging_config import get_logger
from src.core.config import AIConfig
from src.database.models import (
    Process, DecisionPoint, ComplianceItem, RiskAssessment, 
    KnowledgeEntity as DBKnowledgeEntity, Document,
    KnowledgeDomain, HierarchyLevel, CriticalityLevel, 
    ComplianceStatus, RiskLevel
)
from src.ai.document_intelligence_analyzer import DocumentIntelligence, DocumentType
from src.ai.knowledge_categorization_engine import (
    KnowledgeEntity, KnowledgeCategory, EntityType, PriorityLevel
)

logger = get_logger(__name__)


@dataclass
class DatabaseEntry:
    """Represents a clean, normalized database entry"""
    table_name: str
    data: Dict[str, Any]
    relationships: List[Dict[str, Any]]
    quality_score: float
    business_relevance: float
    synthesis_notes: str


@dataclass
class ProcessedKnowledgeUnit:
    """Complete, actionable knowledge unit for database ingestion"""
    primary_entry: DatabaseEntry
    related_entries: List[DatabaseEntry]
    summary: str
    confidence_score: float
    completeness_score: float
    actionability_score: float
    extraction_metadata: Dict[str, Any]


class DatabaseOutputGenerator:
    """Advanced database output generator for Phase 3 processing"""
    
    def __init__(self, config: AIConfig, db_session: Session):
        self.config = config
        self.db_session = db_session
        self.logger = get_logger(__name__)
        
        # Domain mapping for knowledge categorization
        self.domain_mapping = {
            'safety': KnowledgeDomain.SAFETY_COMPLIANCE,
            'compliance': KnowledgeDomain.SAFETY_COMPLIANCE,
            'operational': KnowledgeDomain.OPERATIONAL,
            'process': KnowledgeDomain.OPERATIONAL,
            'technical': KnowledgeDomain.EQUIPMENT_TECHNOLOGY,
            'quality': KnowledgeDomain.QUALITY_ASSURANCE,
            'risk': KnowledgeDomain.SAFETY_COMPLIANCE,
            'financial': KnowledgeDomain.FINANCIAL,
            'organizational': KnowledgeDomain.HUMAN_RESOURCES,
            'training': KnowledgeDomain.TRAINING,
            'maintenance': KnowledgeDomain.MAINTENANCE,
            'regulatory': KnowledgeDomain.REGULATORY,
            'environmental': KnowledgeDomain.ENVIRONMENTAL
        }
        
        # Criticality mapping
        self.criticality_mapping = {
            PriorityLevel.HIGH: CriticalityLevel.CRITICAL,
            PriorityLevel.MEDIUM: CriticalityLevel.MEDIUM,
            PriorityLevel.LOW: CriticalityLevel.LOW
        }
        
        # Quality standards for database integration
        self.quality_thresholds = {
            'minimum_confidence': 0.6,
            'minimum_completeness': 0.5,
            'minimum_clarity': 0.4,
            'minimum_actionability': 0.3,
            'minimum_business_relevance': 0.4
        }
    
    async def generate_database_entries(self, 
                                      knowledge_entities: List[KnowledgeEntity],
                                      document_intelligence: DocumentIntelligence,
                                      document_id: int) -> List[ProcessedKnowledgeUnit]:
        """
        Generate clean, normalized database entries from knowledge entities
        
        Args:
            knowledge_entities: List of categorized knowledge entities
            document_intelligence: Phase 1 analysis results  
            document_id: Database ID of source document
            
        Returns:
            List of processed knowledge units ready for database ingestion
        """
        logger.info("Starting database-optimized output generation")
        
        processed_units = []
        
        # Phase 3A: Quality Filtering
        high_quality_entities = await self._filter_high_quality_entities(knowledge_entities)
        
        # Phase 3B: Content Synthesis
        synthesized_entities = await self._synthesize_content(high_quality_entities)
        
        # Phase 3C: Context Preservation
        contextualized_entities = await self._preserve_context(
            synthesized_entities, document_intelligence
        )
        
        # Phase 3D: Semantic Clarity Enhancement
        clarified_entities = await self._enhance_semantic_clarity(contextualized_entities)
        
        # Phase 3E: Business Relevance Assessment
        relevant_entities = await self._assess_business_relevance(clarified_entities)
        
        # Phase 3F: Database Entry Generation
        for entity in relevant_entities:
            processed_unit = await self._generate_database_entry(
                entity, document_intelligence, document_id
            )
            if processed_unit:
                processed_units.append(processed_unit)
        
        # Phase 3G: Relationship Mapping
        final_units = await self._map_database_relationships(processed_units)
        
        # Phase 3H: Quality Validation
        validated_units = await self._validate_quality_standards(final_units)
        
        logger.info(f"Database output generation completed: {len(validated_units)} knowledge units")
        
        return validated_units
    
    async def _filter_high_quality_entities(self, 
                                          entities: List[KnowledgeEntity]) -> List[KnowledgeEntity]:
        """Filter entities based on quality thresholds"""
        high_quality = []
        
        for entity in entities:
            quality_score = await self._calculate_overall_quality(entity)
            
            if (entity.confidence_score >= self.quality_thresholds['minimum_confidence'] and
                entity.completeness_score >= self.quality_thresholds['minimum_completeness'] and
                entity.clarity_score >= self.quality_thresholds['minimum_clarity'] and
                quality_score >= 0.5):
                
                high_quality.append(entity)
                logger.debug(f"Entity {entity.key_identifier} passed quality filter")
            else:
                logger.debug(f"Entity {entity.key_identifier} filtered out due to low quality")
        
        logger.info(f"Quality filtering: {len(high_quality)}/{len(entities)} entities retained")
        return high_quality
    
    async def _synthesize_content(self, 
                                entities: List[KnowledgeEntity]) -> List[KnowledgeEntity]:
        """Combine fragmented information into complete, actionable units"""
        synthesized = []
        
        # Group related entities for synthesis
        entity_groups = await self._group_related_entities(entities)
        
        for group in entity_groups:
            if len(group) == 1:
                # Single entity - no synthesis needed
                synthesized.append(group[0])
            else:
                # Multiple related entities - synthesize into single unit
                synthesized_entity = await self._synthesize_entity_group(group)
                synthesized.append(synthesized_entity)
        
        logger.info(f"Content synthesis: {len(entities)} entities -> {len(synthesized)} synthesized units")
        return synthesized
    
    async def _preserve_context(self, 
                              entities: List[KnowledgeEntity],
                              intelligence: DocumentIntelligence) -> List[KnowledgeEntity]:
        """Maintain logical relationships between related data points"""
        contextualized = []
        
        for entity in entities:
            # Add document context
            entity.structured_data['document_context'] = {
                'document_type': intelligence.document_type.value,
                'target_audience': [aud.value for aud in intelligence.target_audience],
                'information_architecture': intelligence.information_architecture.value,
                'priority_contexts': [ctx.value for ctx in intelligence.priority_contexts]
            }
            
            # Add relational context
            entity.structured_data['relational_context'] = await self._extract_relational_context(
                entity, entities
            )
            
            # Add semantic context
            entity.structured_data['semantic_context'] = await self._extract_semantic_context(
                entity, intelligence
            )
            
            contextualized.append(entity)
        
        logger.info("Context preservation completed for all entities")
        return contextualized
    
    async def _enhance_semantic_clarity(self, 
                                      entities: List[KnowledgeEntity]) -> List[KnowledgeEntity]:
        """Ensure each database entry is self-contained and unambiguous"""
        clarified = []
        
        for entity in entities:
            # Expand abbreviations and acronyms
            clarified_content = await self._expand_abbreviations(entity.core_content)
            
            # Add disambiguating context
            clarified_content = await self._add_disambiguating_context(
                clarified_content, entity
            )
            
            # Improve readability
            clarified_content = await self._improve_readability(clarified_content)
            
            # Update entity with clarified content
            entity.core_content = clarified_content
            entity.clarity_score = await self._recalculate_clarity_score(clarified_content)
            
            clarified.append(entity)
        
        logger.info("Semantic clarity enhancement completed")
        return clarified
    
    async def _assess_business_relevance(self, 
                                       entities: List[KnowledgeEntity]) -> List[KnowledgeEntity]:
        """Focus on information that drives decisions or actions"""
        relevant_entities = []
        
        for entity in entities:
            business_relevance = await self._calculate_business_relevance(entity)
            
            if business_relevance >= self.quality_thresholds['minimum_business_relevance']:
                entity.structured_data['business_relevance'] = business_relevance
                relevant_entities.append(entity)
                logger.debug(f"Entity {entity.key_identifier} has high business relevance: {business_relevance:.2f}")
            else:
                logger.debug(f"Entity {entity.key_identifier} filtered due to low business relevance: {business_relevance:.2f}")
        
        logger.info(f"Business relevance filtering: {len(relevant_entities)}/{len(entities)} entities retained")
        return relevant_entities
    
    async def _generate_database_entry(self, 
                                     entity: KnowledgeEntity,
                                     intelligence: DocumentIntelligence,
                                     document_id: int) -> Optional[ProcessedKnowledgeUnit]:
        """Generate database entry based on entity type"""
        
        try:
            if entity.entity_type == EntityType.PROCESS:
                return await self._generate_process_entry(entity, intelligence, document_id)
            elif entity.entity_type == EntityType.COMPLIANCE_ITEM:
                return await self._generate_compliance_entry(entity, intelligence, document_id)
            elif entity.entity_type == EntityType.RISK:
                return await self._generate_risk_entry(entity, intelligence, document_id)
            elif entity.entity_type == EntityType.DECISION_POINT:
                return await self._generate_decision_entry(entity, intelligence, document_id)
            elif entity.entity_type in [EntityType.DEFINITION, EntityType.METRIC, EntityType.ROLE]:
                return await self._generate_knowledge_entity_entry(entity, intelligence, document_id)
            else:
                logger.warning(f"Unsupported entity type: {entity.entity_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate database entry for {entity.key_identifier}: {e}")
            return None
    
    async def _generate_process_entry(self, 
                                    entity: KnowledgeEntity,
                                    intelligence: DocumentIntelligence,
                                    document_id: int) -> ProcessedKnowledgeUnit:
        """Generate Process table entry"""
        
        # Determine domain from context tags
        domain = await self._determine_domain(entity.context_tags)
        
        # Determine hierarchy level from content complexity
        hierarchy_level = await self._determine_hierarchy_level(entity)
        
        # Extract process-specific data
        process_data = entity.structured_data
        
        primary_entry = DatabaseEntry(
            table_name="processes",
            data={
                'document_id': document_id,
                'name': entity.key_identifier,
                'description': entity.core_content,
                'domain': domain,
                'hierarchy_level': hierarchy_level,
                'criticality': self.criticality_mapping[entity.priority_level],
                'steps': process_data.get('workflow_steps', []),
                'prerequisites': process_data.get('dependencies', []),
                'success_criteria': process_data.get('completion_criteria', []),
                'responsible_parties': process_data.get('responsible_parties', []),
                'estimated_duration': process_data.get('estimated_duration'),
                'frequency': process_data.get('frequency'),
                'confidence': entity.confidence_score,
                'extraction_method': entity.extraction_method,
                'source_text': entity.core_content,
                'source_section': entity.source_section
            },
            relationships=[],
            quality_score=await self._calculate_overall_quality(entity),
            business_relevance=entity.structured_data.get('business_relevance', 0.5),
            synthesis_notes=f"Synthesized from {entity.extraction_method} with {entity.confidence_score:.2f} confidence"
        )
        
        # Generate related entries for decision points
        related_entries = []
        decision_points = process_data.get('decision_points', [])
        for i, decision in enumerate(decision_points):
            decision_entry = DatabaseEntry(
                table_name="decision_points",
                data={
                    'document_id': document_id,
                    'name': f"{entity.key_identifier}_decision_{i+1}",
                    'description': decision.get('condition', ''),
                    'decision_type': decision.get('type', 'conditional'),
                    'criteria': {'condition': decision.get('condition', '')},
                    'outcomes': {'type': decision.get('type', 'conditional')},
                    'confidence': entity.confidence_score,
                    'source_text': decision.get('condition', ''),
                    'source_page': 1  # Default value
                },
                relationships=[{'type': 'belongs_to_process', 'target': entity.key_identifier}],
                quality_score=await self._calculate_overall_quality(entity),
                business_relevance=entity.structured_data.get('business_relevance', 0.5),
                synthesis_notes="Generated from process decision points"
            )
            related_entries.append(decision_entry)
        
        return ProcessedKnowledgeUnit(
            primary_entry=primary_entry,
            related_entries=related_entries,
            summary=await self._generate_summary(entity),
            confidence_score=entity.confidence_score,
            completeness_score=entity.completeness_score,
            actionability_score=entity.actionability_score,
            extraction_metadata={
                'source_entity_type': entity.entity_type.value,
                'source_category': entity.category.value,
                'context_tags': entity.context_tags,
                'relationships': entity.relationships
            }
        )
    
    async def _generate_compliance_entry(self, 
                                       entity: KnowledgeEntity,
                                       intelligence: DocumentIntelligence,
                                       document_id: int) -> ProcessedKnowledgeUnit:
        """Generate ComplianceItem table entry"""
        
        compliance_data = entity.structured_data
        
        primary_entry = DatabaseEntry(
            table_name="compliance_items",
            data={
                'document_id': document_id,
                'regulation_name': entity.key_identifier,
                'regulation_section': entity.source_section,
                'regulation_authority': compliance_data.get('authority_level', 'Unknown'),
                'requirement': entity.core_content,
                'requirement_type': 'mandatory' if entity.priority_level == PriorityLevel.HIGH else 'recommended',
                'status': ComplianceStatus.PENDING,
                'responsible_party': ', '.join(compliance_data.get('responsible_parties', [])),
                'review_frequency': compliance_data.get('review_frequency', 'annually'),
                'confidence': entity.confidence_score,
                'source_text': entity.core_content,
                'source_section': entity.source_section
            },
            relationships=[],
            quality_score=await self._calculate_overall_quality(entity),
            business_relevance=entity.structured_data.get('business_relevance', 0.8),  # Compliance typically high relevance
            synthesis_notes=f"Compliance requirement extracted with {entity.confidence_score:.2f} confidence"
        )
        
        return ProcessedKnowledgeUnit(
            primary_entry=primary_entry,
            related_entries=[],
            summary=await self._generate_summary(entity),
            confidence_score=entity.confidence_score,
            completeness_score=entity.completeness_score,
            actionability_score=entity.actionability_score,
            extraction_metadata={
                'source_entity_type': entity.entity_type.value,
                'source_category': entity.category.value,
                'context_tags': entity.context_tags,
                'relationships': entity.relationships
            }
        )
    
    async def _generate_risk_entry(self, 
                                 entity: KnowledgeEntity,
                                 intelligence: DocumentIntelligence,
                                 document_id: int) -> ProcessedKnowledgeUnit:
        """Generate RiskAssessment table entry"""
        
        risk_data = entity.structured_data
        
        # Determine risk level from priority
        risk_level_mapping = {
            PriorityLevel.HIGH: RiskLevel.HIGH,
            PriorityLevel.MEDIUM: RiskLevel.MEDIUM,
            PriorityLevel.LOW: RiskLevel.LOW
        }
        
        primary_entry = DatabaseEntry(
            table_name="risk_assessments",
            data={
                'document_id': document_id,
                'hazard': entity.key_identifier,
                'hazard_category': await self._determine_hazard_category(entity),
                'risk_description': entity.core_content,
                'likelihood': 'medium',  # Default - could be enhanced with specific extraction
                'impact': 'medium',  # Default - could be enhanced with specific extraction
                'overall_risk_level': risk_level_mapping[entity.priority_level],
                'mitigation_strategies': risk_data.get('prevention_strategies', []),
                'control_measures': risk_data.get('control_measures', []),
                'monitoring_requirements': '\n'.join(risk_data.get('monitoring_requirements', [])),
                'risk_owner': ', '.join(risk_data.get('responsible_parties', [])),
                'confidence': entity.confidence_score,
                'source_text': entity.core_content,
                'source_section': entity.source_section
            },
            relationships=[],
            quality_score=await self._calculate_overall_quality(entity),
            business_relevance=entity.structured_data.get('business_relevance', 0.9),  # Risk typically very high relevance
            synthesis_notes=f"Risk assessment synthesized with {entity.confidence_score:.2f} confidence"
        )
        
        return ProcessedKnowledgeUnit(
            primary_entry=primary_entry,
            related_entries=[],
            summary=await self._generate_summary(entity),
            confidence_score=entity.confidence_score,
            completeness_score=entity.completeness_score,
            actionability_score=entity.actionability_score,
            extraction_metadata={
                'source_entity_type': entity.entity_type.value,
                'source_category': entity.category.value,
                'context_tags': entity.context_tags,
                'relationships': entity.relationships
            }
        )
    
    async def _generate_decision_entry(self, 
                                     entity: KnowledgeEntity,
                                     intelligence: DocumentIntelligence,
                                     document_id: int) -> ProcessedKnowledgeUnit:
        """Generate DecisionPoint table entry"""
        
        decision_data = entity.structured_data
        
        primary_entry = DatabaseEntry(
            table_name="decision_points",
            data={
                'document_id': document_id,
                'name': entity.key_identifier,
                'description': entity.core_content,
                'decision_type': decision_data.get('type', 'conditional'),
                'criteria': decision_data.get('criteria', {}),
                'outcomes': decision_data.get('outcomes', {}),
                'authority_level': decision_data.get('authority_level', 'manager'),
                'escalation_path': decision_data.get('escalation_path', ''),
                'confidence': entity.confidence_score,
                'source_text': entity.core_content,
                'source_section': entity.source_section
            },
            relationships=[],
            quality_score=await self._calculate_overall_quality(entity),
            business_relevance=entity.structured_data.get('business_relevance', 0.7),
            synthesis_notes=f"Decision point extracted with {entity.confidence_score:.2f} confidence"
        )
        
        return ProcessedKnowledgeUnit(
            primary_entry=primary_entry,
            related_entries=[],
            summary=await self._generate_summary(entity),
            confidence_score=entity.confidence_score,
            completeness_score=entity.completeness_score,
            actionability_score=entity.actionability_score,
            extraction_metadata={
                'source_entity_type': entity.entity_type.value,
                'source_category': entity.category.value,
                'context_tags': entity.context_tags,
                'relationships': entity.relationships
            }
        )
    
    async def _generate_knowledge_entity_entry(self, 
                                             entity: KnowledgeEntity,
                                             intelligence: DocumentIntelligence,
                                             document_id: int) -> ProcessedKnowledgeUnit:
        """Generate KnowledgeEntity table entry for definitions, metrics, etc."""
        
        # Map entity type to label
        label_mapping = {
            EntityType.DEFINITION: 'DEFINITION',
            EntityType.METRIC: 'METRIC',
            EntityType.ROLE: 'ROLE',
            EntityType.SPECIFICATION: 'SPECIFICATION'
        }
        
        primary_entry = DatabaseEntry(
            table_name="knowledge_entities",
            data={
                'document_id': document_id,
                'text': entity.key_identifier,
                'label': label_mapping.get(entity.entity_type, 'OTHER'),
                'confidence': entity.confidence_score,
                'context': entity.core_content,
                'extraction_method': entity.extraction_method
            },
            relationships=[],
            quality_score=await self._calculate_overall_quality(entity),
            business_relevance=entity.structured_data.get('business_relevance', 0.5),
            synthesis_notes=f"Knowledge entity extracted with {entity.confidence_score:.2f} confidence"
        )
        
        return ProcessedKnowledgeUnit(
            primary_entry=primary_entry,
            related_entries=[],
            summary=await self._generate_summary(entity),
            confidence_score=entity.confidence_score,
            completeness_score=entity.completeness_score,
            actionability_score=entity.actionability_score,
            extraction_metadata={
                'source_entity_type': entity.entity_type.value,
                'source_category': entity.category.value,
                'context_tags': entity.context_tags,
                'relationships': entity.relationships
            }
        )
    
    # Helper methods
    async def _calculate_overall_quality(self, entity: KnowledgeEntity) -> float:
        """Calculate overall quality score for an entity"""
        return (entity.confidence_score + entity.completeness_score + 
                entity.clarity_score + entity.actionability_score) / 4
    
    async def _group_related_entities(self, 
                                    entities: List[KnowledgeEntity]) -> List[List[KnowledgeEntity]]:
        """Group related entities for synthesis"""
        groups = []
        processed = set()
        
        for entity in entities:
            if entity.key_identifier in processed:
                continue
            
            group = [entity]
            processed.add(entity.key_identifier)
            
            # Find related entities
            for other_entity in entities:
                if (other_entity.key_identifier not in processed and
                    other_entity.key_identifier in entity.relationships):
                    group.append(other_entity)
                    processed.add(other_entity.key_identifier)
            
            groups.append(group)
        
        return groups
    
    async def _synthesize_entity_group(self, 
                                     group: List[KnowledgeEntity]) -> KnowledgeEntity:
        """Synthesize multiple related entities into a single comprehensive unit"""
        # Use the highest quality entity as the base
        base_entity = max(group, key=lambda e: e.confidence_score)
        
        # Combine content from all entities
        combined_content = base_entity.core_content
        for entity in group:
            if entity != base_entity:
                combined_content += f"\n\nAdditional context: {entity.core_content}"
        
        # Combine structured data
        combined_structured_data = base_entity.structured_data.copy()
        for entity in group:
            if entity != base_entity:
                for key, value in entity.structured_data.items():
                    if key in combined_structured_data:
                        if isinstance(value, list):
                            combined_structured_data[key].extend(value)
                        elif isinstance(value, dict):
                            combined_structured_data[key].update(value)
                    else:
                        combined_structured_data[key] = value
        
        # Create synthesized entity
        synthesized = KnowledgeEntity(
            entity_type=base_entity.entity_type,
            key_identifier=f"synthesized_{base_entity.key_identifier}",
            core_content=combined_content,
            context_tags=list(set(tag for entity in group for tag in entity.context_tags)),
            priority_level=max(entity.priority_level for entity in group),
            category=base_entity.category,
            confidence_score=sum(entity.confidence_score for entity in group) / len(group),
            source_section=base_entity.source_section,
            extraction_method="synthesis",
            relationships=list(set(rel for entity in group for rel in entity.relationships)),
            structured_data=combined_structured_data,
            completeness_score=max(entity.completeness_score for entity in group),
            clarity_score=sum(entity.clarity_score for entity in group) / len(group),
            actionability_score=max(entity.actionability_score for entity in group)
        )
        
        return synthesized
    
    async def _extract_relational_context(self, 
                                        entity: KnowledgeEntity,
                                        all_entities: List[KnowledgeEntity]) -> Dict[str, Any]:
        """Extract relational context for an entity"""
        related_entities = [e for e in all_entities if e.key_identifier in entity.relationships]
        
        return {
            'related_count': len(related_entities),
            'related_categories': list(set(e.category.value for e in related_entities)),
            'related_types': list(set(e.entity_type.value for e in related_entities))
        }
    
    async def _extract_semantic_context(self, 
                                      entity: KnowledgeEntity,
                                      intelligence: DocumentIntelligence) -> Dict[str, Any]:
        """Extract semantic context for an entity"""
        return {
            'document_type_context': intelligence.document_type.value,
            'audience_context': [aud.value for aud in intelligence.target_audience],
            'priority_context': [ctx.value for ctx in intelligence.priority_contexts],
            'complexity_level': intelligence.complexity_level
        }
    
    async def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations and acronyms"""
        # Common business abbreviations
        abbreviations = {
            r'\bSOP\b': 'Standard Operating Procedure',
            r'\bKPI\b': 'Key Performance Indicator',
            r'\bROI\b': 'Return on Investment',
            r'\bCEO\b': 'Chief Executive Officer',
            r'\bCTO\b': 'Chief Technology Officer',
            r'\bHR\b': 'Human Resources',
            r'\bIT\b': 'Information Technology',
            r'\bQA\b': 'Quality Assurance',
            r'\bQC\b': 'Quality Control'
        }
        
        expanded_text = text
        for abbrev, expansion in abbreviations.items():
            # Avoid f-string backslash parsing issues by using format()
            expanded_text = re.sub(abbrev, "{} ({})".format(expansion, abbrev.strip('\\b')), expanded_text)
        
        return expanded_text
    
    async def _add_disambiguating_context(self, 
                                        text: str, 
                                        entity: KnowledgeEntity) -> str:
        """Add context to disambiguate meaning"""
        context_prefix = ""
        
        if entity.category == KnowledgeCategory.PROCESS_INTELLIGENCE:
            context_prefix = f"Process: "
        elif entity.category == KnowledgeCategory.COMPLIANCE_GOVERNANCE:
            context_prefix = f"Compliance Requirement: "
        elif entity.category == KnowledgeCategory.RISK_MITIGATION_INTELLIGENCE:
            context_prefix = f"Risk Assessment: "
        elif entity.category == KnowledgeCategory.ORGANIZATIONAL_INTELLIGENCE:
            context_prefix = f"Organizational Element: "
        
        return context_prefix + text
    
    async def _improve_readability(self, text: str) -> str:
        """Improve text readability"""
        # Basic readability improvements
        improved = text
        
        # Add proper spacing after periods
        improved = re.sub(r'\.([A-Z])', r'. \1', improved)
        
        # Ensure sentences end with periods
        if improved and not improved.endswith(('.', '!', '?')):
            improved += '.'
        
        # Clean up multiple spaces
        improved = re.sub(r'\s+', ' ', improved)
        
        return improved.strip()
    
    async def _recalculate_clarity_score(self, text: str) -> float:
        """Recalculate clarity score after improvements"""
        # Simple heuristic based on readability improvements
        sentences = text.split('.')
        avg_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if avg_length < 15:
            return 0.9
        elif avg_length < 25:
            return 0.7
        else:
            return 0.5
    
    async def _calculate_business_relevance(self, entity: KnowledgeEntity) -> float:
        """Calculate business relevance score"""
        relevance = 0.5  # Base score
        
        # High relevance keywords
        high_relevance_keywords = [
            'critical', 'mandatory', 'required', 'must', 'compliance', 'safety',
            'revenue', 'cost', 'profit', 'customer', 'quality', 'performance'
        ]
        
        # Medium relevance keywords
        medium_relevance_keywords = [
            'important', 'should', 'recommended', 'process', 'procedure',
            'standard', 'guideline', 'training', 'review'
        ]
        
        text_lower = entity.core_content.lower()
        
        # Count keyword matches
        high_matches = sum(1 for keyword in high_relevance_keywords if keyword in text_lower)
        medium_matches = sum(1 for keyword in medium_relevance_keywords if keyword in text_lower)
        
        # Adjust relevance based on matches
        relevance += high_matches * 0.15
        relevance += medium_matches * 0.1
        
        # Adjust based on priority level
        if entity.priority_level == PriorityLevel.HIGH:
            relevance += 0.2
        elif entity.priority_level == PriorityLevel.MEDIUM:
            relevance += 0.1
        
        # Adjust based on category
        high_relevance_categories = [
            KnowledgeCategory.COMPLIANCE_GOVERNANCE,
            KnowledgeCategory.RISK_MITIGATION_INTELLIGENCE,
            KnowledgeCategory.PROCESS_INTELLIGENCE
        ]
        
        if entity.category in high_relevance_categories:
            relevance += 0.15
        
        return min(1.0, relevance)
    
    async def _determine_domain(self, context_tags: List[str]) -> KnowledgeDomain:
        """Determine knowledge domain from context tags"""
        for tag in context_tags:
            if tag in self.domain_mapping:
                return self.domain_mapping[tag]
        
        return KnowledgeDomain.OPERATIONAL  # Default
    
    async def _determine_hierarchy_level(self, entity: KnowledgeEntity) -> HierarchyLevel:
        """Determine hierarchy level based on content complexity"""
        content_length = len(entity.core_content.split())
        
        if content_length > 100:
            return HierarchyLevel.CORE_FUNCTION
        elif content_length > 50:
            return HierarchyLevel.OPERATION
        elif content_length > 20:
            return HierarchyLevel.PROCEDURE
        else:
            return HierarchyLevel.SPECIFIC_STEP
    
    async def _determine_hazard_category(self, entity: KnowledgeEntity) -> str:
        """Determine hazard category for risk assessments"""
        text_lower = entity.core_content.lower()
        
        if any(word in text_lower for word in ['chemical', 'toxic', 'exposure']):
            return 'chemical'
        elif any(word in text_lower for word in ['physical', 'injury', 'equipment']):
            return 'physical'
        elif any(word in text_lower for word in ['biological', 'infection', 'contamination']):
            return 'biological'
        elif any(word in text_lower for word in ['fire', 'explosion', 'electrical']):
            return 'fire_explosion'
        else:
            return 'general'
    
    async def _generate_summary(self, entity: KnowledgeEntity) -> str:
        """Generate a brief summary for UI context"""
        summary = entity.core_content[:150]
        if len(entity.core_content) > 150:
            summary += "..."
        
        category_name = entity.category.value.replace('_', ' ').title()
        priority_name = entity.priority_level.value.title()
        
        return f"{category_name} ({priority_name} Priority): {summary}"
    
    async def _map_database_relationships(self, 
                                        units: List[ProcessedKnowledgeUnit]) -> List[ProcessedKnowledgeUnit]:
        """Map relationships between database entries"""
        # Simple implementation - could be enhanced with foreign key mapping
        for unit in units:
            # Map relationships based on extraction metadata
            related_identifiers = unit.extraction_metadata.get('relationships', [])
            for related_id in related_identifiers:
                unit.primary_entry.relationships.append({
                    'type': 'references',
                    'target': related_id
                })
        
        return units
    
    async def _validate_quality_standards(self, 
                                        units: List[ProcessedKnowledgeUnit]) -> List[ProcessedKnowledgeUnit]:
        """Final quality validation before database insertion"""
        validated = []
        
        for unit in units:
            # Check quality thresholds
            if (unit.confidence_score >= self.quality_thresholds['minimum_confidence'] and
                unit.completeness_score >= self.quality_thresholds['minimum_completeness'] and
                unit.primary_entry.business_relevance >= self.quality_thresholds['minimum_business_relevance']):
                
                validated.append(unit)
                logger.debug(f"Knowledge unit validated: {unit.primary_entry.data.get('name', 'unnamed')}")
            else:
                logger.debug(f"Knowledge unit failed validation: {unit.primary_entry.data.get('name', 'unnamed')}")
        
        logger.info(f"Quality validation: {len(validated)}/{len(units)} units passed")
        return validated