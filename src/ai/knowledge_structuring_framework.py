"""
Knowledge Structuring Framework for EXPLAINIUM

This module implements the framework for organizing extracted knowledge into
hierarchical process structures, domain classifications, metadata enrichment,
and version control for enterprise knowledge management.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from datetime import datetime
from enum import Enum
import hashlib

from src.logging_config import get_logger

logger = get_logger(__name__)

# Enums for standardized classifications
class KnowledgeDomain(Enum):
    """Enterprise knowledge domains."""
    OPERATIONAL = "operational"
    SAFETY_COMPLIANCE = "safety_compliance"
    EQUIPMENT_TECHNOLOGY = "equipment_technology"
    HUMAN_RESOURCES = "human_resources"
    QUALITY_ASSURANCE = "quality_assurance"
    ENVIRONMENTAL = "environmental"
    FINANCIAL = "financial"
    REGULATORY = "regulatory"

class HierarchyLevel(Enum):
    """Process hierarchy levels."""
    CORE_BUSINESS_FUNCTION = 1  # Level 1: Core Business Functions
    DEPARTMENT_OPERATION = 2    # Level 2: Department Operations
    INDIVIDUAL_PROCEDURE = 3    # Level 3: Individual Procedures
    SPECIFIC_STEP = 4          # Level 4: Specific Steps

class SourceQuality(Enum):
    """Source quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class CriticalityLevel(Enum):
    """Criticality levels for knowledge items."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# Data structures for structured knowledge
@dataclass
class ProcessHierarchyNode:
    """Represents a node in the process hierarchy."""
    process_id: str
    name: str
    description: str
    level: HierarchyLevel
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StructuredKnowledgeItem:
    """Represents a structured knowledge item with enriched metadata."""
    process_id: str
    name: str
    description: str
    knowledge_type: str  # tacit, explicit, procedural
    domain: KnowledgeDomain
    hierarchy_level: HierarchyLevel
    confidence_score: float
    source_quality: SourceQuality
    completeness_index: float
    criticality_level: CriticalityLevel
    access_level: str
    source_document_id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    change_history: List[Dict[str, Any]] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    relationships: List[str] = field(default_factory=list)

@dataclass
class DomainClassification:
    """Represents domain classification results."""
    domain: KnowledgeDomain
    confidence: float
    keywords: List[str]
    reasoning: str

@dataclass
class VersionInfo:
    """Version information for knowledge items."""
    version: int
    timestamp: datetime
    changes: List[str]
    author: str
    checksum: str

class HierarchicalProcessOrganizer:
    """Organizes processes into a 4-level hierarchy."""
    
    def __init__(self):
        self.hierarchy_patterns = self._initialize_hierarchy_patterns()
        self.process_graph = {}
        
    def _initialize_hierarchy_patterns(self) -> Dict[HierarchyLevel, List[str]]:
        """Initialize patterns for identifying hierarchy levels."""
        return {
            HierarchyLevel.CORE_BUSINESS_FUNCTION: [
                r'\b(?:manufacturing|production|operations|sales|marketing|finance|hr|it|quality|safety|maintenance|logistics|procurement)\b',
                r'\b(?:business function|core process|primary operation|main activity)\b'
            ],
            HierarchyLevel.DEPARTMENT_OPERATION: [
                r'\b(?:department|division|unit|team|group|section)\b.*\b(?:operation|process|activity)\b',
                r'\b(?:assembly|inspection|testing|packaging|shipping|receiving)\b.*\b(?:operation|process)\b'
            ],
            HierarchyLevel.INDIVIDUAL_PROCEDURE: [
                r'\b(?:procedure|protocol|method|routine|workflow|process)\b',
                r'\b(?:sop|standard operating procedure|work instruction)\b'
            ],
            HierarchyLevel.SPECIFIC_STEP: [
                r'\b(?:step|action|task|instruction|operation)\b\s*\d+',
                r'^\s*\d+\.\s+',
                r'\b(?:first|second|third|next|then|finally)\b.*\b(?:step|action)\b'
            ]
        }
    
    def organize_process_hierarchy(self, raw_processes: List[Dict[str, Any]]) -> Dict[str, ProcessHierarchyNode]:
        """
        Organize raw processes into a hierarchical structure.
        
        Args:
            raw_processes: List of raw process data
            
        Returns:
            Dictionary of organized process hierarchy nodes
        """
        hierarchy = {}
        
        try:
            # First pass: classify each process by hierarchy level
            classified_processes = []
            for process in raw_processes:
                level = self._classify_hierarchy_level(process)
                process_id = self._generate_process_id(process, level)
                
                node = ProcessHierarchyNode(
                    process_id=process_id,
                    name=process.get('name', process.get('text', 'Unknown Process'))[:100],
                    description=process.get('description', process.get('text', ''))[:500],
                    level=level,
                    metadata=process
                )
                
                classified_processes.append(node)
                hierarchy[process_id] = node
            
            # Second pass: establish parent-child relationships
            self._establish_relationships(classified_processes, hierarchy)
            
            logger.info(f"Organized {len(hierarchy)} processes into hierarchy")
            
        except Exception as e:
            logger.error(f"Error organizing process hierarchy: {e}")
        
        return hierarchy
    
    def _classify_hierarchy_level(self, process: Dict[str, Any]) -> HierarchyLevel:
        """Classify a process into the appropriate hierarchy level."""
        
        text = f"{process.get('name', '')} {process.get('description', '')} {process.get('text', '')}".lower()
        
        # Check patterns for each level (from most specific to most general)
        for level in [HierarchyLevel.SPECIFIC_STEP, HierarchyLevel.INDIVIDUAL_PROCEDURE, 
                     HierarchyLevel.DEPARTMENT_OPERATION, HierarchyLevel.CORE_BUSINESS_FUNCTION]:
            
            patterns = self.hierarchy_patterns[level]
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return level
        
        # Default to individual procedure if no pattern matches
        return HierarchyLevel.INDIVIDUAL_PROCEDURE
    
    def _generate_process_id(self, process: Dict[str, Any], level: HierarchyLevel) -> str:
        """Generate a hierarchical process ID."""
        
        # Create a base identifier from the process content
        content = f"{process.get('name', '')} {process.get('text', '')}"
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        # Level prefixes
        level_prefixes = {
            HierarchyLevel.CORE_BUSINESS_FUNCTION: "CBF",
            HierarchyLevel.DEPARTMENT_OPERATION: "DO",
            HierarchyLevel.INDIVIDUAL_PROCEDURE: "IP",
            HierarchyLevel.SPECIFIC_STEP: "SS"
        }
        
        prefix = level_prefixes[level]
        return f"{prefix}-{content_hash}"
    
    def _establish_relationships(self, processes: List[ProcessHierarchyNode], 
                               hierarchy: Dict[str, ProcessHierarchyNode]):
        """Establish parent-child relationships between processes."""
        
        # Group processes by level
        by_level = defaultdict(list)
        for process in processes:
            by_level[process.level].append(process)
        
        # Establish relationships from top to bottom
        levels = [HierarchyLevel.CORE_BUSINESS_FUNCTION, HierarchyLevel.DEPARTMENT_OPERATION,
                 HierarchyLevel.INDIVIDUAL_PROCEDURE, HierarchyLevel.SPECIFIC_STEP]
        
        for i in range(len(levels) - 1):
            parent_level = levels[i]
            child_level = levels[i + 1]
            
            parents = by_level[parent_level]
            children = by_level[child_level]
            
            # Simple heuristic: match based on content similarity
            for child in children:
                best_parent = self._find_best_parent(child, parents)
                if best_parent:
                    child.parent_id = best_parent.process_id
                    best_parent.children.append(child.process_id)
    
    def _find_best_parent(self, child: ProcessHierarchyNode, 
                         candidates: List[ProcessHierarchyNode]) -> Optional[ProcessHierarchyNode]:
        """Find the best parent for a child process."""
        
        if not candidates:
            return None
        
        child_text = f"{child.name} {child.description}".lower()
        best_score = 0
        best_parent = None
        
        for candidate in candidates:
            candidate_text = f"{candidate.name} {candidate.description}".lower()
            
            # Simple similarity based on common words
            child_words = set(child_text.split())
            candidate_words = set(candidate_text.split())
            
            if child_words and candidate_words:
                similarity = len(child_words & candidate_words) / len(child_words | candidate_words)
                
                if similarity > best_score:
                    best_score = similarity
                    best_parent = candidate
        
        # Only return parent if similarity is above threshold
        return best_parent if best_score > 0.1 else None

class DomainClassifier:
    """Classifies knowledge into enterprise domains."""
    
    def __init__(self):
        self.domain_keywords = self._initialize_domain_keywords()
        self.domain_patterns = self._initialize_domain_patterns()
    
    def _initialize_domain_keywords(self) -> Dict[KnowledgeDomain, List[str]]:
        """Initialize keywords for each domain."""
        return {
            KnowledgeDomain.OPERATIONAL: [
                'production', 'manufacturing', 'assembly', 'operation', 'process', 'workflow',
                'procedure', 'routine', 'schedule', 'planning', 'execution', 'performance'
            ],
            KnowledgeDomain.SAFETY_COMPLIANCE: [
                'safety', 'hazard', 'risk', 'accident', 'incident', 'ppe', 'emergency',
                'evacuation', 'first aid', 'osha', 'compliance', 'regulation', 'standard'
            ],
            KnowledgeDomain.EQUIPMENT_TECHNOLOGY: [
                'equipment', 'machine', 'tool', 'device', 'system', 'technology', 'software',
                'hardware', 'maintenance', 'repair', 'calibration', 'specification'
            ],
            KnowledgeDomain.HUMAN_RESOURCES: [
                'employee', 'staff', 'personnel', 'training', 'skill', 'competency',
                'certification', 'performance', 'evaluation', 'development', 'hiring'
            ],
            KnowledgeDomain.QUALITY_ASSURANCE: [
                'quality', 'inspection', 'testing', 'validation', 'verification', 'audit',
                'defect', 'nonconformance', 'corrective', 'preventive', 'iso', 'standard'
            ],
            KnowledgeDomain.ENVIRONMENTAL: [
                'environmental', 'waste', 'emission', 'pollution', 'sustainability',
                'recycling', 'energy', 'conservation', 'green', 'eco', 'carbon'
            ],
            KnowledgeDomain.FINANCIAL: [
                'cost', 'budget', 'expense', 'revenue', 'profit', 'financial', 'accounting',
                'invoice', 'payment', 'procurement', 'purchasing', 'vendor'
            ],
            KnowledgeDomain.REGULATORY: [
                'regulation', 'compliance', 'audit', 'inspection', 'permit', 'license',
                'approval', 'certification', 'standard', 'requirement', 'law', 'policy'
            ]
        }
    
    def _initialize_domain_patterns(self) -> Dict[KnowledgeDomain, List[str]]:
        """Initialize regex patterns for each domain."""
        return {
            KnowledgeDomain.OPERATIONAL: [
                r'\b(?:production|manufacturing|assembly)\s+(?:process|procedure|operation)\b',
                r'\b(?:workflow|routine|schedule)\b'
            ],
            KnowledgeDomain.SAFETY_COMPLIANCE: [
                r'\b(?:safety|hazard|risk)\s+(?:assessment|procedure|protocol)\b',
                r'\b(?:emergency|evacuation)\s+(?:procedure|plan)\b'
            ],
            KnowledgeDomain.EQUIPMENT_TECHNOLOGY: [
                r'\b(?:equipment|machine|system)\s+(?:maintenance|operation|specification)\b',
                r'\b(?:calibration|repair|troubleshooting)\s+(?:procedure|guide)\b'
            ],
            KnowledgeDomain.HUMAN_RESOURCES: [
                r'\b(?:training|skill|competency)\s+(?:program|development|assessment)\b',
                r'\b(?:employee|personnel)\s+(?:evaluation|performance|development)\b'
            ],
            KnowledgeDomain.QUALITY_ASSURANCE: [
                r'\b(?:quality|inspection|testing)\s+(?:procedure|protocol|standard)\b',
                r'\b(?:audit|validation|verification)\s+(?:process|procedure)\b'
            ],
            KnowledgeDomain.ENVIRONMENTAL: [
                r'\b(?:environmental|waste|emission)\s+(?:management|control|procedure)\b',
                r'\b(?:sustainability|conservation)\s+(?:program|initiative)\b'
            ],
            KnowledgeDomain.FINANCIAL: [
                r'\b(?:cost|budget|financial)\s+(?:analysis|management|control)\b',
                r'\b(?:procurement|purchasing)\s+(?:procedure|process)\b'
            ],
            KnowledgeDomain.REGULATORY: [
                r'\b(?:regulatory|compliance)\s+(?:requirement|procedure|audit)\b',
                r'\b(?:permit|license|certification)\s+(?:process|procedure)\b'
            ]
        }
    
    def classify_knowledge_domains(self, knowledge_items: List[Dict[str, Any]]) -> List[DomainClassification]:
        """
        Classify knowledge items into enterprise domains.
        
        Args:
            knowledge_items: List of knowledge items to classify
            
        Returns:
            List of domain classifications
        """
        classifications = []
        
        try:
            for item in knowledge_items:
                classification = self._classify_single_item(item)
                if classification:
                    classifications.append(classification)
            
            logger.info(f"Classified {len(classifications)} knowledge items into domains")
            
        except Exception as e:
            logger.error(f"Error classifying knowledge domains: {e}")
        
        return classifications
    
    def _classify_single_item(self, item: Dict[str, Any]) -> Optional[DomainClassification]:
        """Classify a single knowledge item."""
        
        # Combine all text content
        text_content = f"{item.get('name', '')} {item.get('description', '')} {item.get('text', '')}"
        text_content = text_content.lower()
        
        if not text_content.strip():
            return None
        
        domain_scores = {}
        
        # Score based on keywords
        for domain, keywords in self.domain_keywords.items():
            keyword_score = sum(1 for keyword in keywords if keyword in text_content)
            domain_scores[domain] = keyword_score
        
        # Score based on patterns
        for domain, patterns in self.domain_patterns.items():
            pattern_score = sum(1 for pattern in patterns if re.search(pattern, text_content))
            domain_scores[domain] = domain_scores.get(domain, 0) + pattern_score * 2  # Weight patterns higher
        
        # Find best domain
        if not domain_scores or max(domain_scores.values()) == 0:
            return None
        
        best_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[best_domain] / sum(domain_scores.values())
        
        # Extract relevant keywords
        relevant_keywords = [kw for kw in self.domain_keywords[best_domain] if kw in text_content]
        
        return DomainClassification(
            domain=best_domain,
            confidence=confidence,
            keywords=relevant_keywords,
            reasoning=f"Matched {domain_scores[best_domain]} indicators for {best_domain.value}"
        )

class MetadataEnricher:
    """Enriches knowledge items with confidence scores and source quality."""
    
    def __init__(self):
        self.quality_indicators = self._initialize_quality_indicators()
        self.completeness_factors = self._initialize_completeness_factors()
        self.criticality_indicators = self._initialize_criticality_indicators()
    
    def _initialize_quality_indicators(self) -> Dict[SourceQuality, List[str]]:
        """Initialize indicators for source quality assessment."""
        return {
            SourceQuality.HIGH: [
                'official', 'standard', 'certified', 'approved', 'validated',
                'documented', 'procedure', 'protocol', 'specification'
            ],
            SourceQuality.MEDIUM: [
                'guideline', 'recommendation', 'practice', 'method', 'routine'
            ],
            SourceQuality.LOW: [
                'informal', 'draft', 'preliminary', 'temporary', 'workaround'
            ]
        }
    
    def _initialize_completeness_factors(self) -> List[str]:
        """Initialize factors that indicate completeness."""
        return [
            'step', 'procedure', 'instruction', 'requirement', 'specification',
            'input', 'output', 'result', 'outcome', 'criteria', 'standard'
        ]
    
    def _initialize_criticality_indicators(self) -> Dict[CriticalityLevel, List[str]]:
        """Initialize indicators for criticality assessment."""
        return {
            CriticalityLevel.CRITICAL: [
                'critical', 'essential', 'mandatory', 'required', 'must',
                'emergency', 'safety', 'hazard', 'risk', 'failure'
            ],
            CriticalityLevel.HIGH: [
                'important', 'significant', 'major', 'key', 'primary',
                'should', 'recommended', 'preferred'
            ],
            CriticalityLevel.MEDIUM: [
                'standard', 'normal', 'typical', 'regular', 'routine'
            ],
            CriticalityLevel.LOW: [
                'optional', 'minor', 'supplementary', 'additional', 'nice to have'
            ]
        }
    
    def enrich_metadata(self, knowledge_item: Dict[str, Any]) -> StructuredKnowledgeItem:
        """
        Enrich a knowledge item with metadata.
        
        Args:
            knowledge_item: Raw knowledge item data
            
        Returns:
            Enriched structured knowledge item
        """
        try:
            # Extract basic information
            process_id = knowledge_item.get('process_id', self._generate_process_id(knowledge_item))
            name = knowledge_item.get('name', knowledge_item.get('text', 'Unknown'))[:100]
            description = knowledge_item.get('description', knowledge_item.get('text', ''))[:500]
            knowledge_type = knowledge_item.get('knowledge_type', 'explicit')
            
            # Classify domain
            domain_classification = self._classify_domain(knowledge_item)
            domain = domain_classification.domain if domain_classification else KnowledgeDomain.OPERATIONAL
            
            # Determine hierarchy level
            hierarchy_level = self._determine_hierarchy_level(knowledge_item)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(knowledge_item)
            
            # Assess source quality
            source_quality = self._assess_source_quality(knowledge_item)
            
            # Calculate completeness index
            completeness_index = self._calculate_completeness_index(knowledge_item)
            
            # Determine criticality level
            criticality_level = self._determine_criticality_level(knowledge_item)
            
            # Extract tags
            tags = self._extract_tags(knowledge_item)
            
            return StructuredKnowledgeItem(
                process_id=process_id,
                name=name,
                description=description,
                knowledge_type=knowledge_type,
                domain=domain,
                hierarchy_level=hierarchy_level,
                confidence_score=confidence_score,
                source_quality=source_quality,
                completeness_index=completeness_index,
                criticality_level=criticality_level,
                access_level=knowledge_item.get('access_level', 'internal'),
                source_document_id=knowledge_item.get('source_document_id'),
                tags=tags
            )
            
        except Exception as e:
            logger.error(f"Error enriching metadata: {e}")
            # Return minimal structured item on error
            return StructuredKnowledgeItem(
                process_id=self._generate_process_id(knowledge_item),
                name="Error Processing Item",
                description=str(e),
                knowledge_type="unknown",
                domain=KnowledgeDomain.OPERATIONAL,
                hierarchy_level=HierarchyLevel.INDIVIDUAL_PROCEDURE,
                confidence_score=0.0,
                source_quality=SourceQuality.UNKNOWN,
                completeness_index=0.0,
                criticality_level=CriticalityLevel.LOW,
                access_level='internal'
            )
    
    def _generate_process_id(self, item: Dict[str, Any]) -> str:
        """Generate a process ID for an item."""
        content = f"{item.get('name', '')} {item.get('text', '')}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _classify_domain(self, item: Dict[str, Any]) -> Optional[DomainClassification]:
        """Classify the domain of a knowledge item."""
        classifier = DomainClassifier()
        classifications = classifier.classify_knowledge_domains([item])
        return classifications[0] if classifications else None
    
    def _determine_hierarchy_level(self, item: Dict[str, Any]) -> HierarchyLevel:
        """Determine the hierarchy level of a knowledge item."""
        organizer = HierarchicalProcessOrganizer()
        return organizer._classify_hierarchy_level(item)
    
    def _calculate_confidence_score(self, item: Dict[str, Any]) -> float:
        """Calculate confidence score based on various factors."""
        score = 0.0
        
        # Base score from existing confidence if available
        if 'confidence' in item:
            score += float(item['confidence']) * 0.5
        
        # Boost for structured content
        text = f"{item.get('name', '')} {item.get('description', '')} {item.get('text', '')}"
        
        # Structured indicators
        if re.search(r'\d+\.\s+', text):  # Numbered steps
            score += 0.2
        if re.search(r'\b(?:procedure|protocol|standard)\b', text, re.IGNORECASE):
            score += 0.2
        if len(text.split()) > 20:  # Sufficient detail
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_source_quality(self, item: Dict[str, Any]) -> SourceQuality:
        """Assess the quality of the source."""
        text = f"{item.get('name', '')} {item.get('description', '')} {item.get('text', '')}".lower()
        
        # Check for quality indicators
        for quality, indicators in self.quality_indicators.items():
            if any(indicator in text for indicator in indicators):
                return quality
        
        return SourceQuality.MEDIUM  # Default
    
    def _calculate_completeness_index(self, item: Dict[str, Any]) -> float:
        """Calculate how complete the knowledge item is."""
        text = f"{item.get('name', '')} {item.get('description', '')} {item.get('text', '')}".lower()
        
        completeness_score = 0.0
        
        # Check for completeness factors
        factor_count = sum(1 for factor in self.completeness_factors if factor in text)
        completeness_score += min(0.6, factor_count * 0.1)
        
        # Boost for structured content
        if re.search(r'\d+\.\s+', text):  # Numbered items
            completeness_score += 0.2
        
        # Boost for detailed content
        word_count = len(text.split())
        if word_count > 50:
            completeness_score += 0.2
        
        return min(1.0, completeness_score)
    
    def _determine_criticality_level(self, item: Dict[str, Any]) -> CriticalityLevel:
        """Determine the criticality level of the knowledge item."""
        text = f"{item.get('name', '')} {item.get('description', '')} {item.get('text', '')}".lower()
        
        # Check for criticality indicators (from highest to lowest)
        for level in [CriticalityLevel.CRITICAL, CriticalityLevel.HIGH, 
                     CriticalityLevel.MEDIUM, CriticalityLevel.LOW]:
            indicators = self.criticality_indicators[level]
            if any(indicator in text for indicator in indicators):
                return level
        
        return CriticalityLevel.MEDIUM  # Default
    
    def _extract_tags(self, item: Dict[str, Any]) -> Set[str]:
        """Extract relevant tags from the knowledge item."""
        text = f"{item.get('name', '')} {item.get('description', '')} {item.get('text', '')}".lower()
        
        tags = set()
        
        # Extract domain-related tags
        all_keywords = []
        classifier = DomainClassifier()
        for domain_keywords in classifier.domain_keywords.values():
            all_keywords.extend(domain_keywords)
        
        for keyword in all_keywords:
            if keyword in text:
                tags.add(keyword)
        
        # Extract process-related tags
        process_tags = ['procedure', 'process', 'operation', 'task', 'step', 'workflow']
        for tag in process_tags:
            if tag in text:
                tags.add(tag)
        
        return tags

class VersionController:
    """Manages versions and change history for knowledge items."""
    
    def __init__(self):
        self.version_history = {}
    
    def manage_versions(self, knowledge_item: StructuredKnowledgeItem, 
                       changes: List[str], author: str = "system") -> StructuredKnowledgeItem:
        """
        Manage versions for a knowledge item.
        
        Args:
            knowledge_item: The knowledge item to version
            changes: List of changes made
            author: Author of the changes
            
        Returns:
            Updated knowledge item with version information
        """
        try:
            process_id = knowledge_item.process_id
            
            # Calculate checksum for content integrity
            content = f"{knowledge_item.name}{knowledge_item.description}"
            checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            # Create version info
            version_info = VersionInfo(
                version=knowledge_item.version,
                timestamp=datetime.utcnow(),
                changes=changes,
                author=author,
                checksum=checksum
            )
            
            # Update change history
            change_record = {
                'version': knowledge_item.version,
                'timestamp': version_info.timestamp.isoformat(),
                'changes': changes,
                'author': author,
                'checksum': checksum
            }
            
            knowledge_item.change_history.append(change_record)
            knowledge_item.updated_at = datetime.utcnow()
            
            # Store in version history
            if process_id not in self.version_history:
                self.version_history[process_id] = []
            
            self.version_history[process_id].append(version_info)
            
            logger.info(f"Updated version for {process_id} to v{knowledge_item.version}")
            
        except Exception as e:
            logger.error(f"Error managing versions: {e}")
        
        return knowledge_item
    
    def increment_version(self, knowledge_item: StructuredKnowledgeItem, 
                         changes: List[str], author: str = "system") -> StructuredKnowledgeItem:
        """Increment version and record changes."""
        knowledge_item.version += 1
        return self.manage_versions(knowledge_item, changes, author)
    
    def get_version_history(self, process_id: str) -> List[VersionInfo]:
        """Get version history for a process."""
        return self.version_history.get(process_id, [])
    
    def validate_integrity(self, knowledge_item: StructuredKnowledgeItem) -> bool:
        """Validate the integrity of a knowledge item."""
        content = f"{knowledge_item.name}{knowledge_item.description}"
        current_checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        if knowledge_item.change_history:
            last_checksum = knowledge_item.change_history[-1].get('checksum')
            return current_checksum == last_checksum
        
        return True

class KnowledgeStructuringFramework:
    """Main framework for structuring extracted knowledge."""
    
    def __init__(self):
        self.hierarchical_organizer = HierarchicalProcessOrganizer()
        self.domain_classifier = DomainClassifier()
        self.metadata_enricher = MetadataEnricher()
        self.version_controller = VersionController()
    
    def structure_knowledge(self, raw_knowledge_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Structure raw knowledge items into organized, enriched knowledge.
        
        Args:
            raw_knowledge_items: List of raw knowledge items
            
        Returns:
            Dictionary containing structured knowledge components
        """
        logger.info("Starting knowledge structuring process")
        
        results = {
            'structured_items': [],
            'process_hierarchy': {},
            'domain_classifications': [],
            'structuring_metadata': {
                'total_items': len(raw_knowledge_items),
                'processing_timestamp': datetime.utcnow().isoformat()
            }
        }
        
        try:
            # Step 1: Enrich metadata for all items
            structured_items = []
            for item in raw_knowledge_items:
                enriched_item = self.metadata_enricher.enrich_metadata(item)
                structured_items.append(enriched_item)
            
            results['structured_items'] = structured_items
            
            # Step 2: Organize into process hierarchy
            hierarchy_input = [asdict(item) for item in structured_items]
            process_hierarchy = self.hierarchical_organizer.organize_process_hierarchy(hierarchy_input)
            results['process_hierarchy'] = {k: asdict(v) for k, v in process_hierarchy.items()}
            
            # Step 3: Classify domains
            domain_classifications = self.domain_classifier.classify_knowledge_domains(raw_knowledge_items)
            results['domain_classifications'] = [asdict(dc) for dc in domain_classifications]
            
            # Step 4: Initialize version control for all items
            for item in structured_items:
                self.version_controller.manage_versions(item, ["Initial creation"], "system")
            
            # Update metadata
            results['structuring_metadata'].update({
                'structured_items_count': len(structured_items),
                'hierarchy_nodes_count': len(process_hierarchy),
                'domain_classifications_count': len(domain_classifications),
                'domains_identified': list(set(dc.domain.value for dc in domain_classifications))
            })
            
            logger.info(f"Knowledge structuring completed: {results['structuring_metadata']}")
            
        except Exception as e:
            logger.error(f"Error in knowledge structuring: {e}")
            results['structuring_metadata']['error'] = str(e)
        
        return results

# Global instance for use in other modules
knowledge_structuring_framework = KnowledgeStructuringFramework()