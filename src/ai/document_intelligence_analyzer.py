"""
EXPLAINIUM - Document Intelligence Analyzer

Phase 1: Document Intelligence Assessment
Rapidly analyze the complete file to determine document type, target audience,
information architecture, and priority contexts for intelligent knowledge extraction.
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

# Core AI libraries
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np

# Local model support
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    Llama = None

# Internal imports
from src.logging_config import get_logger
from src.core.config import AIConfig

logger = get_logger(__name__)


class DocumentType(Enum):
    """Document type classification"""
    MANUAL = "manual"
    CONTRACT = "contract"
    REPORT = "report"
    POLICY = "policy"
    SPECIFICATION = "specification"
    PROCEDURE = "procedure"
    TRAINING_MATERIAL = "training_material"
    COMPLIANCE_DOCUMENT = "compliance_document"
    SAFETY_DOCUMENT = "safety_document"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    FINANCIAL_DOCUMENT = "financial_document"
    ORGANIZATIONAL_CHART = "organizational_chart"
    MEETING_MINUTES = "meeting_minutes"
    PRESENTATION = "presentation"
    FORM = "form"
    CHECKLIST = "checklist"
    UNKNOWN = "unknown"


class TargetAudience(Enum):
    """Target audience classification"""
    TECHNICAL_STAFF = "technical_staff"
    MANAGEMENT = "management"
    END_USERS = "end_users"
    COMPLIANCE_OFFICERS = "compliance_officers"
    EXECUTIVES = "executives"
    OPERATIONS_TEAM = "operations_team"
    SAFETY_PERSONNEL = "safety_personnel"
    EXTERNAL_STAKEHOLDERS = "external_stakeholders"
    GENERAL_WORKFORCE = "general_workforce"
    SPECIALISTS = "specialists"
    MULTIPLE_AUDIENCES = "multiple_audiences"


class InformationArchitecture(Enum):
    """Information organization patterns"""
    HIERARCHICAL = "hierarchical"
    SEQUENTIAL = "sequential"
    MATRIX = "matrix"
    NETWORK = "network"
    CATEGORICAL = "categorical"
    CHRONOLOGICAL = "chronological"
    PROCESS_FLOW = "process_flow"
    REFERENCE = "reference"
    NARRATIVE = "narrative"
    MIXED = "mixed"


class PriorityContext(Enum):
    """Priority contexts for knowledge extraction"""
    SAFETY_CRITICAL = "safety_critical"
    COMPLIANCE_MANDATORY = "compliance_mandatory"
    OPERATIONAL_ESSENTIAL = "operational_essential"
    BUSINESS_CRITICAL = "business_critical"
    TECHNICAL_SPECIFICATIONS = "technical_specifications"
    DECISION_MAKING = "decision_making"
    QUALITY_STANDARDS = "quality_standards"
    RISK_MANAGEMENT = "risk_management"
    PERFORMANCE_METRICS = "performance_metrics"
    TRAINING_REQUIREMENTS = "training_requirements"


@dataclass
class DocumentIntelligence:
    """Complete document intelligence assessment"""
    document_type: DocumentType
    confidence_score: float
    target_audience: List[TargetAudience]
    information_architecture: InformationArchitecture
    priority_contexts: List[PriorityContext]
    
    # Detailed analysis
    complexity_level: str  # "basic", "intermediate", "advanced", "expert"
    content_density: str  # "light", "moderate", "dense", "very_dense"
    technical_depth: str  # "overview", "detailed", "comprehensive", "exhaustive"
    regulatory_focus: bool
    process_oriented: bool
    
    # Structure insights
    section_count: int
    has_tables: bool
    has_diagrams: bool
    has_checklists: bool
    has_forms: bool
    
    # Extraction strategy
    recommended_extraction_approach: str
    key_extraction_patterns: List[str]
    context_preservation_requirements: List[str]
    
    # Metadata
    analysis_timestamp: datetime
    analysis_confidence: float


@dataclass
class ContentSection:
    """Represents a section of content with its characteristics"""
    title: str
    content: str
    section_type: str
    importance_level: str
    extraction_priority: int
    contains_processes: bool
    contains_requirements: bool
    contains_metrics: bool
    contains_risks: bool


class DocumentIntelligenceAnalyzer:
    """Advanced document intelligence analyzer for Phase 1 assessment"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.llm = None
        self.embedder = None
        self.initialized = False
        
        # Document type indicators
        self.type_indicators = {
            DocumentType.MANUAL: [
                "user guide", "manual", "instructions", "how to", "step by step",
                "operating instructions", "user manual", "instruction manual"
            ],
            DocumentType.CONTRACT: [
                "agreement", "contract", "terms and conditions", "service level",
                "vendor agreement", "supplier contract", "purchase order"
            ],
            DocumentType.REPORT: [
                "report", "analysis", "findings", "summary", "executive summary",
                "quarterly report", "annual report", "assessment report"
            ],
            DocumentType.POLICY: [
                "policy", "governance", "standard", "guideline", "directive",
                "corporate policy", "company policy", "organizational policy"
            ],
            DocumentType.SPECIFICATION: [
                "specification", "requirements", "technical spec", "design spec",
                "functional requirements", "system requirements"
            ],
            DocumentType.PROCEDURE: [
                "procedure", "process", "workflow", "standard operating",
                "sop", "operating procedure", "work instruction"
            ],
            DocumentType.TRAINING_MATERIAL: [
                "training", "course", "learning", "curriculum", "educational",
                "training manual", "course material", "learning guide"
            ],
            DocumentType.COMPLIANCE_DOCUMENT: [
                "compliance", "regulatory", "audit", "certification",
                "compliance manual", "regulatory requirements"
            ],
            DocumentType.SAFETY_DOCUMENT: [
                "safety", "hazard", "risk assessment", "safety manual",
                "safety procedures", "emergency procedures", "msds"
            ]
        }
        
        # Audience indicators
        self.audience_indicators = {
            TargetAudience.TECHNICAL_STAFF: [
                "technical", "implementation", "configuration", "troubleshooting",
                "system administration", "technical specifications"
            ],
            TargetAudience.MANAGEMENT: [
                "management", "executive", "oversight", "governance", "strategic",
                "management review", "executive summary", "business case"
            ],
            TargetAudience.END_USERS: [
                "user", "employee", "staff", "how to use", "getting started",
                "user guide", "end user", "employee handbook"
            ],
            TargetAudience.COMPLIANCE_OFFICERS: [
                "compliance", "audit", "regulatory", "legal", "certification",
                "compliance officer", "audit trail", "regulatory requirements"
            ]
        }
    
    async def initialize(self):
        """Initialize the analyzer with AI models"""
        try:
            if hasattr(self.config, 'local_model_path') and self.config.local_model_path:
                # Initialize local LLM if available
                if LLAMA_AVAILABLE:
                    self.llm = Llama(
                        model_path=self.config.local_model_path,
                        n_ctx=4096,
                        n_threads=4,
                        verbose=False
                    )
            
            # Initialize embedding model
            self.embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
            self.initialized = True
            logger.info("Document Intelligence Analyzer initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize models, using fallback methods: {e}")
            self.initialized = False
    
    async def analyze_document_intelligence(self, content: str, filename: str = "", 
                                           metadata: Dict[str, Any] = None) -> DocumentIntelligence:
        """
        Perform comprehensive document intelligence assessment
        
        Args:
            content: Full document content
            filename: Document filename for additional context
            metadata: Additional document metadata
            
        Returns:
            DocumentIntelligence object with complete assessment
        """
        if not self.initialized:
            await self.initialize()
        
        logger.info(f"Starting document intelligence analysis for {filename}")
        
        # Extract document sections for analysis
        sections = await self._extract_document_sections(content)
        
        # Phase 1A: Document Type Detection
        doc_type, type_confidence = await self._detect_document_type(content, filename, sections)
        
        # Phase 1B: Target Audience Analysis
        target_audiences = await self._analyze_target_audience(content, sections)
        
        # Phase 1C: Information Architecture Analysis
        info_architecture = await self._analyze_information_architecture(content, sections)
        
        # Phase 1D: Priority Context Identification
        priority_contexts = await self._identify_priority_contexts(content, sections)
        
        # Phase 1E: Content Characteristics
        characteristics = await self._analyze_content_characteristics(content, sections)
        
        # Phase 1F: Structural Analysis
        structure = await self._analyze_document_structure(content, sections)
        
        # Phase 1G: Extraction Strategy
        extraction_strategy = await self._determine_extraction_strategy(
            doc_type, target_audiences, info_architecture, priority_contexts
        )
        
        # Compile complete intelligence assessment
        intelligence = DocumentIntelligence(
            document_type=doc_type,
            confidence_score=type_confidence,
            target_audience=target_audiences,
            information_architecture=info_architecture,
            priority_contexts=priority_contexts,
            
            complexity_level=characteristics['complexity_level'],
            content_density=characteristics['content_density'],
            technical_depth=characteristics['technical_depth'],
            regulatory_focus=characteristics['regulatory_focus'],
            process_oriented=characteristics['process_oriented'],
            
            section_count=structure['section_count'],
            has_tables=structure['has_tables'],
            has_diagrams=structure['has_diagrams'],
            has_checklists=structure['has_checklists'],
            has_forms=structure['has_forms'],
            
            recommended_extraction_approach=extraction_strategy['approach'],
            key_extraction_patterns=extraction_strategy['patterns'],
            context_preservation_requirements=extraction_strategy['context_requirements'],
            
            analysis_timestamp=datetime.now(),
            analysis_confidence=self._calculate_overall_confidence(
                type_confidence, characteristics, structure
            )
        )
        
        logger.info(f"Document intelligence analysis completed: {doc_type.value} "
                   f"({type_confidence:.2f} confidence)")
        
        return intelligence
    
    async def _extract_document_sections(self, content: str) -> List[ContentSection]:
        """Extract and analyze document sections"""
        sections = []
        
        # Split content into logical sections
        section_patterns = [
            r'\n#+\s*(.*?)\n',  # Markdown headers
            r'\n(\d+\.?\s+.*?)\n',  # Numbered sections
            r'\n([A-Z][A-Z\s]{3,})\n',  # ALL CAPS headers
            r'\n(SECTION.*?)\n',  # Section headers
            r'\n(CHAPTER.*?)\n',  # Chapter headers
        ]
        
        current_section = ""
        current_title = "Introduction"
        
        for line in content.split('\n'):
            line_stripped = line.strip()
            
            # Check if this line is a section header
            is_header = False
            for pattern in section_patterns:
                if re.match(pattern, f'\n{line}\n'):
                    is_header = True
                    # Save previous section
                    if current_section.strip():
                        section = await self._analyze_section_content(
                            current_title, current_section
                        )
                        sections.append(section)
                    
                    # Start new section
                    current_title = line_stripped
                    current_section = ""
                    break
            
            if not is_header:
                current_section += line + '\n'
        
        # Add final section
        if current_section.strip():
            section = await self._analyze_section_content(current_title, current_section)
            sections.append(section)
        
        return sections
    
    async def _analyze_section_content(self, title: str, content: str) -> ContentSection:
        """Analyze individual section content"""
        
        # Determine section type
        section_type = "general"
        if any(keyword in title.lower() for keyword in ["process", "procedure", "step"]):
            section_type = "process"
        elif any(keyword in title.lower() for keyword in ["requirement", "standard", "compliance"]):
            section_type = "requirement"
        elif any(keyword in title.lower() for keyword in ["risk", "safety", "hazard"]):
            section_type = "risk"
        elif any(keyword in title.lower() for keyword in ["metric", "measure", "kpi"]):
            section_type = "metric"
        
        # Determine importance level
        importance_level = "medium"
        if any(keyword in content.lower() for keyword in ["critical", "mandatory", "required", "must"]):
            importance_level = "high"
        elif any(keyword in content.lower() for keyword in ["optional", "suggested", "recommended"]):
            importance_level = "low"
        
        # Calculate extraction priority
        priority_score = 50  # Base priority
        if "critical" in content.lower(): priority_score += 30
        if "mandatory" in content.lower(): priority_score += 25
        if "required" in content.lower(): priority_score += 20
        if section_type in ["process", "requirement"]: priority_score += 15
        
        extraction_priority = min(100, priority_score)
        
        # Content analysis
        contains_processes = bool(re.search(r'\b(step|procedure|process|workflow)\b', content, re.I))
        contains_requirements = bool(re.search(r'\b(must|shall|required|mandatory)\b', content, re.I))
        contains_metrics = bool(re.search(r'\b(measure|metric|kpi|target|threshold)\b', content, re.I))
        contains_risks = bool(re.search(r'\b(risk|hazard|danger|safety)\b', content, re.I))
        
        return ContentSection(
            title=title,
            content=content,
            section_type=section_type,
            importance_level=importance_level,
            extraction_priority=extraction_priority,
            contains_processes=contains_processes,
            contains_requirements=contains_requirements,
            contains_metrics=contains_metrics,
            contains_risks=contains_risks
        )
    
    async def _detect_document_type(self, content: str, filename: str, 
                                   sections: List[ContentSection]) -> Tuple[DocumentType, float]:
        """Detect document type with confidence score"""
        
        # Combine content sources for analysis
        analysis_text = f"{filename} {content[:2000]}".lower()
        
        # Score each document type
        type_scores = {}
        
        for doc_type, indicators in self.type_indicators.items():
            score = 0
            for indicator in indicators:
                # Count occurrences with position weighting
                occurrences = len(re.findall(rf'\b{re.escape(indicator)}\b', analysis_text))
                score += occurrences * 10
                
                # Bonus for filename matches
                if indicator in filename.lower():
                    score += 25
                
                # Bonus for early occurrence in content
                early_match = analysis_text[:500].count(indicator)
                score += early_match * 15
            
            type_scores[doc_type] = score
        
        # Special pattern-based detection
        if re.search(r'table of contents|chapter \d+|section \d+', content, re.I):
            type_scores[DocumentType.MANUAL] += 20
        
        if re.search(r'whereas|hereby|party|agreement|contract', content, re.I):
            type_scores[DocumentType.CONTRACT] += 30
        
        if re.search(r'step \d+|procedure:|process:|workflow', content, re.I):
            type_scores[DocumentType.PROCEDURE] += 25
        
        # Find highest scoring type
        best_type = max(type_scores.items(), key=lambda x: x[1])
        
        if best_type[1] == 0:
            return DocumentType.UNKNOWN, 0.0
        
        # Calculate confidence score
        total_score = sum(type_scores.values())
        confidence = min(0.95, best_type[1] / max(total_score, 1))
        
        return best_type[0], confidence
    
    async def _analyze_target_audience(self, content: str, 
                                     sections: List[ContentSection]) -> List[TargetAudience]:
        """Analyze target audience from content characteristics"""
        
        analysis_text = content.lower()
        audience_scores = {}
        
        for audience, indicators in self.audience_indicators.items():
            score = 0
            for indicator in indicators:
                score += len(re.findall(rf'\b{re.escape(indicator)}\b', analysis_text))
            audience_scores[audience] = score
        
        # Additional analysis based on content complexity and language
        technical_terms = len(re.findall(r'\b(configure|implement|troubleshoot|administrator)\b', analysis_text))
        if technical_terms > 5:
            audience_scores[TargetAudience.TECHNICAL_STAFF] += 20
        
        management_terms = len(re.findall(r'\b(strategic|governance|oversight|management)\b', analysis_text))
        if management_terms > 3:
            audience_scores[TargetAudience.MANAGEMENT] += 15
        
        user_terms = len(re.findall(r'\b(user|employee|staff|worker)\b', analysis_text))
        if user_terms > 5:
            audience_scores[TargetAudience.END_USERS] += 15
        
        # Select audiences with significant scores
        threshold = max(5, max(audience_scores.values()) * 0.3)
        target_audiences = [
            audience for audience, score in audience_scores.items() 
            if score >= threshold
        ]
        
        if not target_audiences:
            target_audiences = [TargetAudience.GENERAL_WORKFORCE]
        
        return target_audiences
    
    async def _analyze_information_architecture(self, content: str, 
                                              sections: List[ContentSection]) -> InformationArchitecture:
        """Analyze how information is organized"""
        
        # Check for hierarchical structure
        hierarchical_indicators = len(re.findall(r'\d+\.\d+|\d+\.\d+\.\d+|chapter|section', content, re.I))
        
        # Check for sequential structure
        sequential_indicators = len(re.findall(r'step \d+|first|next|then|finally|procedure', content, re.I))
        
        # Check for process flow
        process_indicators = len(re.findall(r'workflow|process|flowchart|diagram', content, re.I))
        
        # Check for reference structure
        reference_indicators = len(re.findall(r'see|refer to|reference|index|appendix', content, re.I))
        
        # Check for categorical structure
        categorical_indicators = len(re.findall(r'category|type|classification|group', content, re.I))
        
        # Determine primary architecture
        scores = {
            InformationArchitecture.HIERARCHICAL: hierarchical_indicators,
            InformationArchitecture.SEQUENTIAL: sequential_indicators,
            InformationArchitecture.PROCESS_FLOW: process_indicators,
            InformationArchitecture.REFERENCE: reference_indicators,
            InformationArchitecture.CATEGORICAL: categorical_indicators
        }
        
        max_score = max(scores.values())
        if max_score < 3:
            return InformationArchitecture.NARRATIVE
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    async def _identify_priority_contexts(self, content: str, 
                                        sections: List[ContentSection]) -> List[PriorityContext]:
        """Identify priority contexts for extraction"""
        
        contexts = []
        analysis_text = content.lower()
        
        # Safety critical indicators
        if re.search(r'safety|hazard|danger|critical|emergency', analysis_text):
            contexts.append(PriorityContext.SAFETY_CRITICAL)
        
        # Compliance mandatory indicators
        if re.search(r'compliance|regulatory|mandatory|required|legal', analysis_text):
            contexts.append(PriorityContext.COMPLIANCE_MANDATORY)
        
        # Operational essential indicators
        if re.search(r'operation|procedure|process|workflow|essential', analysis_text):
            contexts.append(PriorityContext.OPERATIONAL_ESSENTIAL)
        
        # Business critical indicators
        if re.search(r'business|critical|strategic|revenue|customer', analysis_text):
            contexts.append(PriorityContext.BUSINESS_CRITICAL)
        
        # Technical specifications indicators
        if re.search(r'specification|technical|requirement|standard', analysis_text):
            contexts.append(PriorityContext.TECHNICAL_SPECIFICATIONS)
        
        # Decision making indicators
        if re.search(r'decision|approval|authority|escalation', analysis_text):
            contexts.append(PriorityContext.DECISION_MAKING)
        
        # Quality standards indicators
        if re.search(r'quality|standard|metric|measure|kpi', analysis_text):
            contexts.append(PriorityContext.QUALITY_STANDARDS)
        
        # Risk management indicators
        if re.search(r'risk|mitigation|control|assessment|management', analysis_text):
            contexts.append(PriorityContext.RISK_MANAGEMENT)
        
        return contexts if contexts else [PriorityContext.OPERATIONAL_ESSENTIAL]
    
    async def _analyze_content_characteristics(self, content: str, 
                                             sections: List[ContentSection]) -> Dict[str, Any]:
        """Analyze content complexity and characteristics"""
        
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?]+', content))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Complexity analysis
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b|\btechnical\b|\bspecification\b', content))
        complexity_level = "basic"
        if technical_terms > 20:
            complexity_level = "expert"
        elif technical_terms > 10:
            complexity_level = "advanced"
        elif technical_terms > 5:
            complexity_level = "intermediate"
        
        # Content density
        content_density = "moderate"
        if word_count > 10000:
            content_density = "very_dense"
        elif word_count > 5000:
            content_density = "dense"
        elif word_count < 1000:
            content_density = "light"
        
        # Technical depth
        technical_depth = "overview"
        if avg_sentence_length > 25:
            technical_depth = "exhaustive"
        elif avg_sentence_length > 20:
            technical_depth = "comprehensive"
        elif avg_sentence_length > 15:
            technical_depth = "detailed"
        
        # Regulatory focus
        regulatory_terms = len(re.findall(r'regulatory|compliance|legal|standard|requirement', content, re.I))
        regulatory_focus = regulatory_terms > 10
        
        # Process orientation
        process_terms = len(re.findall(r'process|procedure|step|workflow|operation', content, re.I))
        process_oriented = process_terms > 15
        
        return {
            'complexity_level': complexity_level,
            'content_density': content_density,
            'technical_depth': technical_depth,
            'regulatory_focus': regulatory_focus,
            'process_oriented': process_oriented
        }
    
    async def _analyze_document_structure(self, content: str, 
                                        sections: List[ContentSection]) -> Dict[str, Any]:
        """Analyze document structural elements"""
        
        return {
            'section_count': len(sections),
            'has_tables': bool(re.search(r'\|.*\||\btable\b', content, re.I)),
            'has_diagrams': bool(re.search(r'diagram|figure|chart|graph', content, re.I)),
            'has_checklists': bool(re.search(r'checklist|\[\s*\]|\☐|\✓', content)),
            'has_forms': bool(re.search(r'form|field|input|signature', content, re.I))
        }
    
    async def _determine_extraction_strategy(self, doc_type: DocumentType, 
                                           audiences: List[TargetAudience],
                                           architecture: InformationArchitecture,
                                           contexts: List[PriorityContext]) -> Dict[str, Any]:
        """Determine optimal extraction strategy"""
        
        approach = "comprehensive"
        patterns = []
        context_requirements = []
        
        # Adjust approach based on document type
        if doc_type == DocumentType.PROCEDURE:
            approach = "process_focused"
            patterns.extend(["step_extraction", "workflow_mapping", "decision_points"])
            context_requirements.append("maintain_sequential_order")
        
        elif doc_type == DocumentType.COMPLIANCE_DOCUMENT:
            approach = "requirement_focused"
            patterns.extend(["requirement_extraction", "compliance_mapping", "audit_points"])
            context_requirements.append("preserve_regulatory_context")
        
        elif doc_type == DocumentType.MANUAL:
            approach = "user_focused"
            patterns.extend(["instruction_extraction", "troubleshooting", "feature_mapping"])
            context_requirements.append("maintain_user_context")
        
        # Adjust based on priority contexts
        if PriorityContext.SAFETY_CRITICAL in contexts:
            patterns.append("safety_extraction")
            context_requirements.append("preserve_safety_warnings")
        
        if PriorityContext.COMPLIANCE_MANDATORY in contexts:
            patterns.append("compliance_extraction")
            context_requirements.append("maintain_regulatory_traceability")
        
        # Default patterns for comprehensive extraction
        if not patterns:
            patterns = ["general_extraction", "entity_recognition", "relationship_mapping"]
        
        if not context_requirements:
            context_requirements = ["preserve_document_structure"]
        
        return {
            'approach': approach,
            'patterns': patterns,
            'context_requirements': context_requirements
        }
    
    def _calculate_overall_confidence(self, type_confidence: float, 
                                    characteristics: Dict[str, Any],
                                    structure: Dict[str, Any]) -> float:
        """Calculate overall analysis confidence"""
        
        confidence_factors = [type_confidence]
        
        # Structure confidence
        if structure['section_count'] > 0:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Content confidence based on characteristics
        if characteristics['complexity_level'] != "basic":
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        return sum(confidence_factors) / len(confidence_factors)
