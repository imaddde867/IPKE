#!/usr/bin/env python3
"""
LLM-First Processing Engine - Primary Intelligence Layer
========================================================

This engine implements the LLM-first processing hierarchy where offline AI models
are the primary source for knowledge extraction, ensuring superior results through
intelligent document understanding and structured knowledge generation.

PROCESSING HIERARCHY RULES:
1. LLM Primary Analysis (Priority: CRITICAL)
2. Enhanced Pattern Recognition (Priority: HIGH) 
3. Traditional NLP Fallback (Priority: LOW)

QUALITY THRESHOLDS:
- LLM Confidence Minimum: 0.75
- Pattern Recognition Minimum: 0.60
- Combined Analysis Threshold: 0.80
- Entity Validation Score: 0.70

PROCESSING RULES & VALUES:
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import re
from pathlib import Path

# Import LLM backends
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    Llama = None

from src.logging_config import get_logger
from src.core.config import AIConfig
from src.ai.enhanced_extraction_engine import EnhancedExtractionEngine, ExtractedEntity

logger = get_logger(__name__)

class ProcessingPriority(Enum):
    """Processing priority levels for different extraction methods"""
    CRITICAL = "critical"    # LLM-based processing (0.95+ reliability)
    HIGH = "high"           # Enhanced pattern + LLM validation (0.80+ reliability)
    MEDIUM = "medium"       # Pattern recognition with NLP (0.65+ reliability)
    LOW = "low"             # Basic pattern matching (0.50+ reliability)

class QualityThreshold(Enum):
    """Quality thresholds for different processing stages"""
    LLM_MINIMUM = 0.75          # Minimum LLM confidence
    ENHANCED_MINIMUM = 0.60     # Enhanced extraction minimum
    COMBINED_MINIMUM = 0.80     # Combined analysis threshold
    ENTITY_VALIDATION = 0.70    # Entity validation score
    PRODUCTION_READY = 0.85     # Production deployment threshold

@dataclass
class ProcessingRule:
    """Defines a processing rule with conditions and actions"""
    name: str
    condition: str
    action: str
    priority: ProcessingPriority
    threshold: float
    description: str

@dataclass
class ProcessingResult:
    """Comprehensive processing result with quality metrics"""
    entities: List[ExtractedEntity]
    processing_method: str
    confidence_score: float
    quality_metrics: Dict[str, float]
    processing_time: float
    llm_enhanced: bool
    validation_passed: bool
    metadata: Dict[str, Any]

class LLMProcessingEngine:
    """
    LLM-First Processing Engine - The Primary Intelligence Layer
    
    CORE PRINCIPLES:
    ===============
    1. LLM SUPREMACY: Offline LLM models are the primary processing source
    2. QUALITY FIRST: High confidence thresholds ensure superior results
    3. HIERARCHICAL FALLBACK: Graceful degradation through processing layers
    4. VALIDATION REQUIRED: All extractions must pass validation thresholds
    5. STRUCTURED OUTPUT: Clean, categorized, database-ready results
    
    PROCESSING RULES:
    ================
    """
    
    # Core Processing Rules - THE FUNDAMENTAL LAWS
    PROCESSING_RULES = [
        ProcessingRule(
            name="LLM_PRIMARY_RULE",
            condition="llm_available == True",
            action="use_llm_primary_processing",
            priority=ProcessingPriority.CRITICAL,
            threshold=0.75,
            description="LLM must be primary processor for ALL documents when available"
        ),
        ProcessingRule(
            name="QUALITY_THRESHOLD_RULE", 
            condition="extraction_confidence < 0.70",
            action="escalate_to_higher_priority_method",
            priority=ProcessingPriority.HIGH,
            threshold=0.70,
            description="Low confidence extractions must be re-processed with higher priority method"
        ),
        ProcessingRule(
            name="VALIDATION_GATE_RULE",
            condition="entity_count > 0",
            action="validate_all_entities",
            priority=ProcessingPriority.CRITICAL,
            threshold=0.70,
            description="All extracted entities must pass validation before acceptance"
        ),
        ProcessingRule(
            name="STRUCTURED_OUTPUT_RULE",
            condition="processing_complete == True",
            action="ensure_structured_categorized_output",
            priority=ProcessingPriority.CRITICAL,
            threshold=0.80,
            description="Final output must be clean, structured, and categorized"
        ),
        ProcessingRule(
            name="FALLBACK_HIERARCHY_RULE",
            condition="primary_method_failed == True",
            action="cascade_through_processing_hierarchy",
            priority=ProcessingPriority.HIGH,
            threshold=0.60,
            description="Failed processing must cascade through hierarchy until success"
        )
    ]
    
    def __init__(self, config: AIConfig = None):
        self.config = config or AIConfig()
        self.llm_model = None
        self.enhanced_engine = None
        self.initialized = False
        self.processing_stats = {
            "llm_primary_used": 0,
            "enhanced_pattern_used": 0,
            "fallback_used": 0,
            "total_processed": 0,
            "average_confidence": 0.0,
            "validation_pass_rate": 0.0
        }
        
    async def initialize(self):
        """Initialize the LLM-first processing engine"""
        try:
            logger.info("Initializing LLM-First Processing Engine")
            
            # Priority 1: Initialize Primary LLM
            await self._initialize_primary_llm()
            
            # Priority 2: Initialize Enhanced Pattern Engine
            self.enhanced_engine = EnhancedExtractionEngine(llm_model=self.llm_model)
            
            # Priority 3: Validate initialization
            self._validate_initialization()
            
            self.initialized = True
            logger.info("✅ LLM-First Processing Engine initialized successfully")
            logger.info(f"🧠 Primary LLM Available: {self.llm_model is not None}")
            logger.info(f"🔧 Enhanced Engine Ready: {self.enhanced_engine is not None}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM Processing Engine: {e}")
            self.initialized = False
            raise
    
    async def _initialize_primary_llm(self):
        """Initialize the primary LLM model with optimized settings"""
        if not LLAMA_AVAILABLE:
            logger.warning("⚠️ llama-cpp-python not available. LLM processing disabled.")
            return
        
        try:
            # Load model configuration
            model_config_path = Path("models/setup_config.json")
            if model_config_path.exists():
                with open(model_config_path, 'r') as f:
                    model_config = json.load(f)
                
                llm_config = model_config.get("models", {}).get("llm", {})
                optimization = llm_config.get("m4_optimization", {})
                
                # Extract optimal settings
                model_path = optimization.get("model_path", "models/llm/Mistral-7B-Instruct-v0.2-GGUF")
                context_length = optimization.get("context_length", 4096)
                threads = optimization.get("threads", 8)
                
                # Initialize with optimal settings for knowledge extraction
                model_file = Path(model_path) / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
                
                if model_file.exists():
                    self.llm_model = Llama(
                        model_path=str(model_file),
                        n_ctx=context_length,
                        n_threads=threads,
                        n_gpu_layers=-1,  # Use Metal on M4
                        verbose=False,
                        seed=42,  # Reproducible results
                        n_batch=4,
                        temperature=0.3,  # Conservative for knowledge extraction
                        top_p=0.9
                    )
                    logger.info(f"🧠 Primary LLM loaded: {model_file.name}")
                else:
                    logger.warning(f"⚠️ LLM model file not found: {model_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize primary LLM: {e}")
            self.llm_model = None
    
    def _validate_initialization(self):
        """Validate that the processing engine meets minimum requirements"""
        requirements = {
            "enhanced_engine": self.enhanced_engine is not None,
            "processing_rules": len(self.PROCESSING_RULES) >= 5,
            "quality_thresholds": True  # Always available
        }
        
        failed_requirements = [req for req, met in requirements.items() if not met]
        
        if failed_requirements:
            raise RuntimeError(f"Initialization validation failed: {failed_requirements}")
        
        logger.info("✅ Processing engine validation passed")
    
    async def process_document(self, content: str, document_type: str = "unknown", 
                             metadata: Dict[str, Any] = None) -> ProcessingResult:
        """
        PRIMARY PROCESSING FUNCTION - Implements LLM-First Hierarchy
        
        PROCESSING FLOW:
        1. Document Analysis & Complexity Assessment
        2. LLM Primary Processing (if available & complex)
        3. Enhanced Pattern Processing (high-quality fallback)
        4. Entity Validation & Quality Control
        5. Structured Output Generation
        
        Args:
            content: Document content to process
            document_type: Type of document for optimized processing
            metadata: Additional context and processing hints
            
        Returns:
            ProcessingResult with comprehensive extraction results
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = datetime.now()
        metadata = metadata or {}
        
        logger.info(f"🚀 Starting LLM-First processing for {document_type} document")
        
        # STEP 1: Document Complexity Assessment
        complexity_score = await self._assess_document_complexity(content, document_type)
        
        # STEP 2: Apply Processing Rules to Determine Method
        processing_method = self._determine_processing_method(complexity_score, metadata)
        
        # STEP 3: Execute Primary Processing
        entities = await self._execute_processing(content, document_type, processing_method, metadata)
        
        # STEP 4: Validate and Enhance Results
        validated_entities = await self._validate_and_enhance_entities(entities, content)
        
        # STEP 5: Calculate Quality Metrics
        quality_metrics = self._calculate_quality_metrics(validated_entities, processing_method)
        
        # STEP 6: Generate Final Result
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = ProcessingResult(
            entities=validated_entities,
            processing_method=processing_method,
            confidence_score=quality_metrics["overall_confidence"],
            quality_metrics=quality_metrics,
            processing_time=processing_time,
            llm_enhanced=processing_method.startswith("llm"),
            validation_passed=quality_metrics["validation_score"] >= QualityThreshold.ENTITY_VALIDATION.value,
            metadata={
                "document_type": document_type,
                "complexity_score": complexity_score,
                "rules_applied": self._get_applied_rules(),
                "processing_stats": self._update_processing_stats(processing_method, quality_metrics)
            }
        )
        
        logger.info(f"✅ Processing complete: {len(validated_entities)} entities, "
                   f"{quality_metrics['overall_confidence']:.2f} confidence, "
                   f"{processing_time:.2f}s")
        
        return result
    
    async def _assess_document_complexity(self, content: str, document_type: str) -> float:
        """Assess document complexity to determine optimal processing method"""
        complexity_indicators = {
            "length": min(len(content) / 10000, 1.0) * 0.3,  # Longer = more complex
            "technical_terms": self._count_technical_terms(content) * 0.2,
            "structured_elements": self._count_structured_elements(content) * 0.2,
            "domain_specificity": self._assess_domain_specificity(content, document_type) * 0.3
        }
        
        complexity_score = sum(complexity_indicators.values())
        logger.info(f"📊 Document complexity: {complexity_score:.2f} {complexity_indicators}")
        
        return min(complexity_score, 1.0)
    
    def _determine_processing_method(self, complexity_score: float, metadata: Dict[str, Any]) -> str:
        """Apply processing rules to determine optimal method"""
        
        # Rule 1: LLM Primary when available (prioritize local AI models)
        # Use LLM for ALL documents if available since it's more accurate
        if self.llm_model is not None:
            return "llm_primary_analysis"
        
        # Rule 2: Enhanced processing for medium complexity (fallback)
        if complexity_score >= 0.3:
            return "enhanced_pattern_processing"
        
        # Rule 3: Basic processing for simple documents (last resort)
        return "basic_pattern_processing"
    
    async def _execute_processing(self, content: str, document_type: str, 
                                method: str, metadata: Dict[str, Any]) -> List[ExtractedEntity]:
        """Execute the determined processing method"""
        
        if method == "llm_primary_analysis":
            return await self._llm_primary_processing(content, document_type)
        elif method == "enhanced_pattern_processing":
            return await self._enhanced_pattern_processing(content, document_type)
        else:
            return await self._basic_pattern_processing(content, document_type)
    
    async def _llm_primary_processing(self, content: str, document_type: str) -> List[ExtractedEntity]:
        """LLM-first processing - the primary intelligence method"""
        logger.info("🧠 Executing LLM Primary Processing")
        
        if not self.llm_model:
            logger.warning("⚠️ LLM not available, falling back to enhanced processing")
            return await self._enhanced_pattern_processing(content, document_type)
        
        # Use enhanced engine with LLM for maximum intelligence
        entities = self.enhanced_engine.extract_comprehensive_knowledge(content, document_type)
        
        # LLM-specific enhancements
        enhanced_entities = []
        for entity in entities:
            # Boost confidence for LLM-enhanced entities
            if entity.metadata.get("llm_generated") or entity.metadata.get("enhanced_by_llm"):
                entity.confidence = min(0.95, entity.confidence + 0.10)
                entity.metadata["processing_method"] = "llm_primary"
                entity.metadata["quality_boost"] = True
            
            enhanced_entities.append(entity)
        
        return enhanced_entities
    
    async def _enhanced_pattern_processing(self, content: str, document_type: str) -> List[ExtractedEntity]:
        """Enhanced pattern processing with intelligent extraction"""
        logger.info("🔧 Executing Enhanced Pattern Processing")
        
        if self.enhanced_engine:
            entities = self.enhanced_engine.extract_comprehensive_knowledge(content, document_type)
            
            # Mark as enhanced processing
            for entity in entities:
                entity.metadata["processing_method"] = "enhanced_pattern"
            
            return entities
        else:
            return await self._basic_pattern_processing(content, document_type)
    
    async def _basic_pattern_processing(self, content: str, document_type: str) -> List[ExtractedEntity]:
        """Basic pattern processing - fallback method"""
        logger.info("📝 Executing Basic Pattern Processing")
        
        # Simple pattern-based extraction as absolute fallback
        entities = []
        
        # Basic patterns for critical information
        patterns = {
            "specifications": r'(\d+\.?\d*)\s*(HP|V|PSI|GPM|°F|°C)\s*([^\.]{0,50})',
            "procedures": r'(must|shall|should|required)\s+([^\.]{20,100})',
            "personnel": r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*-\s*([^,\n]{10,50})',
        }
        
        for pattern_type, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entity = ExtractedEntity(
                    content=match.group(0).strip(),
                    entity_type=pattern_type,
                    category="basic_extraction",
                    confidence=0.50,  # Lower confidence for basic patterns
                    context=content[max(0, match.start()-50):match.end()+50],
                    metadata={
                        "processing_method": "basic_pattern",
                        "pattern_type": pattern_type,
                        "fallback_used": True
                    },
                    relationships=[],
                    source_location=f"chars_{match.start()}_{match.end()}"
                )
                entities.append(entity)
        
        return entities
    
    async def _validate_and_enhance_entities(self, entities: List[ExtractedEntity], 
                                          content: str) -> List[ExtractedEntity]:
        """Validate and enhance extracted entities according to quality rules"""
        validated_entities = []
        
        for entity in entities:
            # Apply validation rules
            if self._validate_entity(entity):
                # Apply enhancement rules
                enhanced_entity = await self._enhance_entity(entity, content)
                validated_entities.append(enhanced_entity)
            else:
                logger.debug(f"Entity failed validation: {entity.content[:50]}...")
        
        logger.info(f"✅ Validation complete: {len(validated_entities)}/{len(entities)} entities passed")
        return validated_entities
    
    def _validate_entity(self, entity: ExtractedEntity) -> bool:
        """Validate entity against quality thresholds"""
        validations = {
            "confidence_threshold": entity.confidence >= QualityThreshold.ENTITY_VALIDATION.value,
            "content_length": len(entity.content) >= 10,
            "meaningful_content": not self._is_meaningless_content(entity.content),
            "category_valid": entity.category != "unknown"
        }
        
        return all(validations.values())
    
    async def _enhance_entity(self, entity: ExtractedEntity, content: str) -> ExtractedEntity:
        """Enhance entity with additional intelligence and context"""
        
        # Add structured metadata
        entity.metadata.update({
            "enhanced_timestamp": datetime.now().isoformat(),
            "validation_passed": True,
            "enhancement_level": self._calculate_enhancement_level(entity),
            "business_relevance": self._calculate_business_relevance(entity),
            "actionability": self._calculate_actionability(entity)
        })
        
        return entity
    
    def _calculate_quality_metrics(self, entities: List[ExtractedEntity], 
                                 processing_method: str) -> Dict[str, float]:
        """Calculate comprehensive quality metrics"""
        if not entities:
            return {
                "overall_confidence": 0.0,
                "validation_score": 0.0,
                "completeness": 0.0,
                "diversity": 0.0,
                "business_value": 0.0
            }
        
        # Core metrics
        confidences = [e.confidence for e in entities]
        overall_confidence = sum(confidences) / len(confidences)
        
        # Validation metrics
        llm_enhanced_count = sum(1 for e in entities if e.metadata.get("llm_generated") or e.metadata.get("enhanced_by_llm"))
        validation_score = llm_enhanced_count / len(entities) if entities else 0.0
        
        # Diversity metrics
        categories = set(e.category for e in entities)
        entity_types = set(e.entity_type for e in entities)
        diversity = (len(categories) + len(entity_types)) / (len(entities) + 1)  # Avoid division by zero
        
        # Business value estimation
        high_value_entities = sum(1 for e in entities if e.confidence > 0.8)
        business_value = high_value_entities / len(entities) if entities else 0.0
        
        # Completeness based on processing method
        method_completeness = {
            "llm_primary_analysis": 0.95,
            "enhanced_pattern_processing": 0.80,
            "basic_pattern_processing": 0.60
        }
        completeness = method_completeness.get(processing_method, 0.50)
        
        return {
            "overall_confidence": overall_confidence,
            "validation_score": validation_score,
            "completeness": completeness,
            "diversity": min(diversity, 1.0),
            "business_value": business_value,
            "entity_count": len(entities),
            "llm_enhanced_count": llm_enhanced_count,
            "processing_method": processing_method
        }
    
    # Helper methods for complexity assessment
    def _count_technical_terms(self, content: str) -> float:
        """Count technical terms in content"""
        technical_patterns = [
            r'\b\d+\.?\d*\s*(HP|PSI|GPM|RPM|kW|MHz|GHz|°F|°C|V|A|W)\b',
            r'\b(specification|procedure|protocol|standard|regulation|compliance)\b',
            r'\b[A-Z]{2,}[-\d]*\b',  # Acronyms
        ]
        
        count = 0
        for pattern in technical_patterns:
            count += len(re.findall(pattern, content, re.IGNORECASE))
        
        return min(count / 100, 1.0)  # Normalize to 0-1
    
    def _count_structured_elements(self, content: str) -> float:
        """Count structured elements (lists, tables, sections)"""
        structured_patterns = [
            r'^\s*\d+\.', r'^\s*[a-z]\)', r'^\s*[-*•]',  # Lists
            r'\|.*\|', r'\t.*\t',  # Tables
            r'^[A-Z\s]+:$', r'#{1,6}\s'  # Headers
        ]
        
        count = 0
        for line in content.split('\n'):
            for pattern in structured_patterns:
                if re.search(pattern, line, re.MULTILINE):
                    count += 1
                    break
        
        return min(count / 50, 1.0)  # Normalize to 0-1
    
    def _assess_domain_specificity(self, content: str, document_type: str) -> float:
        """Assess domain-specific complexity"""
        domain_indicators = {
            "safety": ["hazard", "risk", "danger", "safety", "protection", "warning"],
            "technical": ["specification", "parameter", "measurement", "calibration"],
            "regulatory": ["compliance", "standard", "regulation", "requirement", "certification"],
            "operational": ["procedure", "process", "workflow", "maintenance", "operation"]
        }
        
        content_lower = content.lower()
        max_score = 0
        
        for domain, terms in domain_indicators.items():
            score = sum(1 for term in terms if term in content_lower)
            max_score = max(max_score, score)
        
        return min(max_score / 20, 1.0)  # Normalize to 0-1
    
    def _is_meaningless_content(self, content: str) -> bool:
        """Check if content is meaningless or too generic"""
        meaningless_patterns = [
            r'^[^a-zA-Z]*$',  # No letters
            r'^(the|and|or|but|in|on|at|to|for|of|with|by).*$',  # Starts with common words only
            r'^.{1,5}$',  # Too short
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, content.strip(), re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_enhancement_level(self, entity: ExtractedEntity) -> str:
        """Calculate enhancement level for entity"""
        if entity.metadata.get("llm_generated") and entity.confidence > 0.85:
            return "maximum"
        elif entity.metadata.get("enhanced_by_llm") or entity.confidence > 0.75:
            return "high"
        elif entity.confidence > 0.60:
            return "medium"
        else:
            return "basic"
    
    def _calculate_business_relevance(self, entity: ExtractedEntity) -> float:
        """Calculate business relevance score"""
        relevance_indicators = {
            "safety": 0.9,
            "compliance": 0.8,
            "technical": 0.7,
            "operational": 0.6,
            "general": 0.4
        }
        
        entity_category = entity.category.lower()
        for indicator, score in relevance_indicators.items():
            if indicator in entity_category:
                return score
        
        return 0.5  # Default relevance
    
    def _calculate_actionability(self, entity: ExtractedEntity) -> float:
        """Calculate how actionable the entity is"""
        actionable_terms = ["must", "shall", "should", "required", "procedure", "step", "process"]
        content_lower = entity.content.lower()
        
        actionability_score = sum(0.1 for term in actionable_terms if term in content_lower)
        return min(actionability_score, 1.0)
    
    def _get_applied_rules(self) -> List[str]:
        """Get list of applied processing rules"""
        return [rule.name for rule in self.PROCESSING_RULES]
    
    def _update_processing_stats(self, method: str, quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Update and return processing statistics"""
        self.processing_stats["total_processed"] += 1
        
        if method.startswith("llm"):
            self.processing_stats["llm_primary_used"] += 1
        elif method.startswith("enhanced"):
            self.processing_stats["enhanced_pattern_used"] += 1
        else:
            self.processing_stats["fallback_used"] += 1
        
        # Update running averages
        total = self.processing_stats["total_processed"]
        self.processing_stats["average_confidence"] = (
            (self.processing_stats["average_confidence"] * (total - 1) + 
             quality_metrics["overall_confidence"]) / total
        )
        
        self.processing_stats["validation_pass_rate"] = (
            (self.processing_stats["validation_pass_rate"] * (total - 1) + 
             quality_metrics["validation_score"]) / total
        )
        
        return self.processing_stats.copy()
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing engine summary"""
        return {
            "engine_status": "initialized" if self.initialized else "not_initialized",
            "llm_available": self.llm_model is not None,
            "enhanced_engine_available": self.enhanced_engine is not None,
            "processing_rules_count": len(self.PROCESSING_RULES),
            "quality_thresholds": {
                "llm_minimum": QualityThreshold.LLM_MINIMUM.value,
                "enhanced_minimum": QualityThreshold.ENHANCED_MINIMUM.value,
                "combined_minimum": QualityThreshold.COMBINED_MINIMUM.value,
                "entity_validation": QualityThreshold.ENTITY_VALIDATION.value,
                "production_ready": QualityThreshold.PRODUCTION_READY.value
            },
            "processing_statistics": self.processing_stats,
            "core_principles": [
                "LLM Supremacy - Offline models are primary source",
                "Quality First - High confidence thresholds",
                "Hierarchical Fallback - Graceful degradation",
                "Validation Required - All entities validated",
                "Structured Output - Clean categorized results"
            ]
        }

# Export the main processing engine
__all__ = ["LLMProcessingEngine", "ProcessingResult", "ProcessingPriority", "QualityThreshold"]
