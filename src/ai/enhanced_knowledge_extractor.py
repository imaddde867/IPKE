"""
Enhanced Knowledge Extraction Engine for EXPLAINIUM

This module implements advanced algorithms for extracting comprehensive organizational 
knowledge from various document types, including implicit knowledge patterns, 
process hierarchies, decision flows, compliance requirements, and operational structures.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from datetime import datetime
import spacy
import networkx as nx
from transformers import pipeline
import numpy as np

from src.logging_config import get_logger
from src.database.models import (
    KnowledgeDomain, HierarchyLevel, CriticalityLevel, 
    ComplianceStatus, RiskLevel
)

logger = get_logger(__name__)

@dataclass
class ExtractedProcess:
    """Represents an extracted organizational process"""
    process_id: str
    name: str
    description: str
    domain: KnowledgeDomain
    hierarchy_level: HierarchyLevel
    parent_process_id: Optional[str] = None
    estimated_duration: Optional[str] = None
    frequency: Optional[str] = None
    prerequisites: List[str] = None
    success_criteria: List[str] = None
    required_skills: List[str] = None
    required_certifications: List[str] = None
    quality_standards: List[str] = None
    compliance_requirements: List[str] = None
    criticality_level: CriticalityLevel = CriticalityLevel.MEDIUM
    confidence_score: float = 0.0

@dataclass
class ExtractedDecisionPoint:
    """Represents a decision point within a process"""
    name: str
    description: str
    decision_type: str  # binary, multiple_choice, conditional, threshold
    criteria: Dict[str, Any]
    outcomes: List[Dict[str, Any]]
    authority_level: str
    escalation_path: List[str]
    confidence: float

@dataclass
class ExtractedComplianceItem:
    """Represents a compliance requirement"""
    regulation_name: str
    section: str
    description: str
    status: ComplianceStatus
    responsible_party: str
    evidence_required: List[str]
    review_frequency: str
    confidence: float

@dataclass
class ExtractedRiskAssessment:
    """Represents a risk assessment"""
    category: str
    description: str
    likelihood: RiskLevel
    impact: RiskLevel
    mitigation_strategies: List[str]
    monitoring_requirements: List[str]
    responsible_party: str
    confidence: float

class EnhancedKnowledgeExtractor:
    """Advanced knowledge extraction engine for organizational documents"""
    
    def __init__(self):
        self.nlp = None
        self.classifier = None
        self.sentiment_analyzer = None
        self._load_models()
        self._initialize_patterns()
    
    def _load_models(self):
        """Load required NLP models and transformers"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Load classification pipeline
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            # Load sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            logger.info("Successfully loaded NLP models for enhanced knowledge extraction")
        except Exception as e:
            logger.error(f"Failed to load NLP models: {e}")
            # Set fallback modes
            self.nlp = None
            self.classifier = None
            self.sentiment_analyzer = None
    
    def _initialize_patterns(self):
        """Initialize pattern libraries for knowledge extraction"""
        
        # Process identification patterns
        self.process_patterns = {
            'action_verbs': [
                'perform', 'execute', 'conduct', 'implement', 'operate', 'maintain',
                'inspect', 'verify', 'check', 'monitor', 'control', 'manage',
                'start', 'stop', 'begin', 'complete', 'finish', 'prepare',
                'install', 'remove', 'replace', 'repair', 'calibrate', 'test'
            ],
            'process_indicators': [
                'procedure', 'process', 'operation', 'workflow', 'task',
                'step', 'phase', 'stage', 'activity', 'function', 'routine'
            ],
            'hierarchy_indicators': {
                HierarchyLevel.CORE_BUSINESS_FUNCTION: [
                    'core function', 'business function', 'primary operation',
                    'main process', 'strategic', 'enterprise'
                ],
                HierarchyLevel.DEPARTMENT_OPERATION: [
                    'department', 'division', 'unit', 'section', 'area',
                    'operational', 'functional'
                ],
                HierarchyLevel.INDIVIDUAL_PROCEDURE: [
                    'procedure', 'protocol', 'method', 'technique',
                    'standard operating procedure', 'sop'
                ],
                HierarchyLevel.SPECIFIC_STEP: [
                    'step', 'action', 'task', 'instruction', 'directive'
                ]
            }
        }
        
        # Decision point patterns
        self.decision_patterns = {
            'decision_indicators': [
                'if', 'when', 'unless', 'provided that', 'in case of',
                'should', 'must', 'may', 'can', 'decide', 'determine',
                'choose', 'select', 'evaluate', 'assess', 'consider'
            ],
            'binary_decisions': [
                'yes/no', 'pass/fail', 'accept/reject', 'approve/deny',
                'go/no-go', 'continue/stop'
            ],
            'threshold_decisions': [
                'greater than', 'less than', 'exceeds', 'below',
                'above', 'within', 'outside', 'between'
            ]
        }
        
        # Compliance patterns
        self.compliance_patterns = {
            'regulatory_bodies': [
                'OSHA', 'EPA', 'FDA', 'ISO', 'ANSI', 'NIST', 'DOT',
                'NIOSH', 'ASTM', 'IEEE', 'API', 'ASME'
            ],
            'compliance_indicators': [
                'shall', 'must', 'required', 'mandatory', 'obligatory',
                'compliance', 'regulation', 'standard', 'code',
                'requirement', 'specification'
            ],
            'standards_patterns': [
                r'ISO\s+\d+', r'OSHA\s+\d+', r'ANSI\s+[A-Z]\d+',
                r'ASTM\s+[A-Z]\d+', r'IEEE\s+\d+', r'API\s+\d+'
            ]
        }
        
        # Risk assessment patterns
        self.risk_patterns = {
            'risk_indicators': [
                'risk', 'hazard', 'danger', 'threat', 'vulnerability',
                'exposure', 'safety', 'security', 'failure'
            ],
            'severity_levels': {
                RiskLevel.VERY_HIGH: ['catastrophic', 'critical', 'severe', 'major'],
                RiskLevel.HIGH: ['high', 'significant', 'serious', 'important'],
                RiskLevel.MEDIUM: ['moderate', 'medium', 'average', 'typical'],
                RiskLevel.LOW: ['low', 'minor', 'small', 'negligible'],
                RiskLevel.VERY_LOW: ['very low', 'minimal', 'insignificant']
            },
            'mitigation_indicators': [
                'prevent', 'mitigate', 'control', 'reduce', 'minimize',
                'eliminate', 'avoid', 'protect', 'safeguard'
            ]
        }
        
        # Domain classification patterns
        self.domain_patterns = {
            KnowledgeDomain.OPERATIONAL: [
                'operation', 'production', 'manufacturing', 'workflow',
                'process', 'procedure', 'task', 'activity'
            ],
            KnowledgeDomain.SAFETY_COMPLIANCE: [
                'safety', 'hazard', 'risk', 'protection', 'emergency',
                'compliance', 'regulation', 'standard'
            ],
            KnowledgeDomain.EQUIPMENT_TECHNOLOGY: [
                'equipment', 'machine', 'device', 'tool', 'instrument',
                'technology', 'system', 'component'
            ],
            KnowledgeDomain.HUMAN_RESOURCES: [
                'personnel', 'staff', 'employee', 'training', 'skill',
                'certification', 'qualification', 'competency'
            ],
            KnowledgeDomain.QUALITY_ASSURANCE: [
                'quality', 'inspection', 'testing', 'verification',
                'validation', 'audit', 'review', 'assessment'
            ],
            KnowledgeDomain.MAINTENANCE: [
                'maintenance', 'repair', 'service', 'calibration',
                'preventive', 'corrective', 'overhaul'
            ],
            KnowledgeDomain.TRAINING: [
                'training', 'education', 'instruction', 'learning',
                'development', 'course', 'program', 'curriculum'
            ]
        }
    
    def extract_comprehensive_knowledge(self, text: str, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to extract comprehensive organizational knowledge from text
        
        Args:
            text: Document text content
            document_metadata: Document metadata and context
            
        Returns:
            Dictionary containing all extracted knowledge items
        """
        logger.info("Starting comprehensive knowledge extraction")
        
        # Initialize results structure
        results = {
            'processes': [],
            'decision_points': [],
            'compliance_items': [],
            'risk_assessments': [],
            'equipment_relationships': [],
            'personnel_relationships': [],
            'knowledge_hierarchy': {},
            'implicit_knowledge': [],
            'confidence_scores': {}
        }
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Extract different types of knowledge
            results['processes'] = self._extract_processes(processed_text, document_metadata)
            results['decision_points'] = self._extract_decision_points(processed_text)
            results['compliance_items'] = self._extract_compliance_items(processed_text)
            results['risk_assessments'] = self._extract_risk_assessments(processed_text)
            results['equipment_relationships'] = self._extract_equipment_relationships(processed_text)
            results['personnel_relationships'] = self._extract_personnel_relationships(processed_text)
            results['knowledge_hierarchy'] = self._build_knowledge_hierarchy(results['processes'])
            results['implicit_knowledge'] = self._extract_implicit_knowledge(processed_text)
            
            # Calculate overall confidence scores
            results['confidence_scores'] = self._calculate_confidence_scores(results)
            
            logger.info(f"Knowledge extraction completed. Extracted {len(results['processes'])} processes, "
                       f"{len(results['decision_points'])} decision points, "
                       f"{len(results['compliance_items'])} compliance items")
            
        except Exception as e:
            logger.error(f"Error during knowledge extraction: {e}")
            # Return partial results with error information
            results['extraction_errors'] = [str(e)]
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better extraction"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize section headers
        text = re.sub(r'^(\d+\.?\d*)\s+([A-Z][A-Za-z\s]+)$', r'\1 \2', text, flags=re.MULTILINE)
        
        # Clean up formatting artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/]', ' ', text)
        
        return text.strip()
    
    def _extract_processes(self, text: str, metadata: Dict[str, Any]) -> List[ExtractedProcess]:
        """Extract organizational processes from text"""
        processes = []
        
        if not self.nlp:
            return self._extract_processes_fallback(text, metadata)
        
        try:
            doc = self.nlp(text)
            
            # Find process-related sentences
            process_sentences = []
            for sent in doc.sents:
                if self._is_process_sentence(sent.text):
                    process_sentences.append(sent.text)
            
            # Extract processes from identified sentences
            for i, sentence in enumerate(process_sentences):
                process = self._extract_process_from_sentence(sentence, i, metadata)
                if process:
                    processes.append(process)
            
        except Exception as e:
            logger.error(f"Error in process extraction: {e}")
            return self._extract_processes_fallback(text, metadata)
        
        return processes
    
    def _is_process_sentence(self, sentence: str) -> bool:
        """Determine if a sentence describes a process"""
        sentence_lower = sentence.lower()
        
        # Check for action verbs
        has_action_verb = any(verb in sentence_lower for verb in self.process_patterns['action_verbs'])
        
        # Check for process indicators
        has_process_indicator = any(indicator in sentence_lower for indicator in self.process_patterns['process_indicators'])
        
        # Check for procedural language
        has_procedural_language = any(word in sentence_lower for word in ['step', 'procedure', 'process', 'operation'])
        
        return has_action_verb or has_process_indicator or has_procedural_language
    
    def _extract_process_from_sentence(self, sentence: str, index: int, metadata: Dict[str, Any]) -> Optional[ExtractedProcess]:
        """Extract a process from a single sentence"""
        try:
            # Generate process ID
            process_id = f"PROC_{metadata.get('document_id', 'UNK')}_{index:03d}"
            
            # Extract process name (first noun phrase or key terms)
            name = self._extract_process_name(sentence)
            
            # Classify domain
            domain = self._classify_domain(sentence)
            
            # Determine hierarchy level
            hierarchy_level = self._determine_hierarchy_level(sentence)
            
            # Extract other attributes
            duration = self._extract_duration(sentence)
            frequency = self._extract_frequency(sentence)
            prerequisites = self._extract_prerequisites(sentence)
            success_criteria = self._extract_success_criteria(sentence)
            
            # Calculate confidence score
            confidence = self._calculate_process_confidence(sentence)
            
            return ExtractedProcess(
                process_id=process_id,
                name=name,
                description=sentence,
                domain=domain,
                hierarchy_level=hierarchy_level,
                estimated_duration=duration,
                frequency=frequency,
                prerequisites=prerequisites,
                success_criteria=success_criteria,
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.error(f"Error extracting process from sentence: {e}")
            return None
    
    def _extract_process_name(self, sentence: str) -> str:
        """Extract process name from sentence"""
        if self.nlp:
            doc = self.nlp(sentence)
            # Look for noun phrases that might be process names
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 4:  # Reasonable process name length
                    return chunk.text.strip()
        
        # Fallback: use first few words
        words = sentence.split()[:5]
        return ' '.join(words).strip()
    
    def _classify_domain(self, text: str) -> KnowledgeDomain:
        """Classify the knowledge domain of the text"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return KnowledgeDomain.OPERATIONAL  # Default
    
    def _determine_hierarchy_level(self, text: str) -> HierarchyLevel:
        """Determine the hierarchy level of a process"""
        text_lower = text.lower()
        
        for level, indicators in self.process_patterns['hierarchy_indicators'].items():
            if any(indicator in text_lower for indicator in indicators):
                return level
        
        # Default based on content analysis
        if len(text.split()) > 20:  # Longer descriptions tend to be higher level
            return HierarchyLevel.DEPARTMENT_OPERATION
        else:
            return HierarchyLevel.SPECIFIC_STEP
    
    def _extract_duration(self, text: str) -> Optional[str]:
        """Extract estimated duration from text"""
        duration_patterns = [
            r'(\d+)\s*(hour|hr|minute|min|day|week|month)s?',
            r'(approximately|about|around)\s+(\d+)\s*(hour|hr|minute|min|day|week|month)s?',
            r'takes?\s+(\d+)\s*(hour|hr|minute|min|day|week|month)s?'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        
        return None
    
    def _extract_frequency(self, text: str) -> Optional[str]:
        """Extract frequency information from text"""
        frequency_patterns = [
            r'(daily|weekly|monthly|annually|yearly)',
            r'(every|each)\s+(day|week|month|year|hour)',
            r'(\d+)\s+times?\s+(per|a)\s+(day|week|month|year)',
            r'(as needed|when required|on demand)'
        ]
        
        for pattern in frequency_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        
        return None
    
    def _extract_prerequisites(self, text: str) -> List[str]:
        """Extract prerequisites from text"""
        prerequisites = []
        prerequisite_patterns = [
            r'before\s+([^,.]+)',
            r'requires?\s+([^,.]+)',
            r'must\s+([^,.]+)\s+before',
            r'prerequisite:?\s*([^,.]+)'
        ]
        
        for pattern in prerequisite_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                prerequisites.append(match.group(1).strip())
        
        return prerequisites
    
    def _extract_success_criteria(self, text: str) -> List[str]:
        """Extract success criteria from text"""
        criteria = []
        criteria_patterns = [
            r'success(?:ful)?\s+when\s+([^,.]+)',
            r'complete(?:d)?\s+when\s+([^,.]+)',
            r'result(?:s)?\s+in\s+([^,.]+)',
            r'achieve(?:s)?\s+([^,.]+)'
        ]
        
        for pattern in criteria_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                criteria.append(match.group(1).strip())
        
        return criteria
    
    def _calculate_process_confidence(self, text: str) -> float:
        """Calculate confidence score for process extraction"""
        confidence_factors = []
        
        # Check for specific process indicators
        if any(indicator in text.lower() for indicator in self.process_patterns['process_indicators']):
            confidence_factors.append(0.3)
        
        # Check for action verbs
        if any(verb in text.lower() for verb in self.process_patterns['action_verbs']):
            confidence_factors.append(0.2)
        
        # Check for structured language
        if re.search(r'\d+\.', text):  # Numbered steps
            confidence_factors.append(0.2)
        
        # Check for procedural language
        if any(word in text.lower() for word in ['shall', 'must', 'should', 'will']):
            confidence_factors.append(0.2)
        
        # Check for completeness
        if len(text.split()) >= 10:  # Reasonable detail
            confidence_factors.append(0.1)
        
        return min(sum(confidence_factors), 1.0)
    
    def _extract_processes_fallback(self, text: str, metadata: Dict[str, Any]) -> List[ExtractedProcess]:
        """Fallback method for process extraction without NLP models"""
        processes = []
        
        # Simple pattern-based extraction
        sentences = text.split('.')
        for i, sentence in enumerate(sentences):
            if self._is_process_sentence(sentence):
                process = ExtractedProcess(
                    process_id=f"PROC_{metadata.get('document_id', 'UNK')}_{i:03d}",
                    name=sentence.split()[:5],  # First 5 words as name
                    description=sentence.strip(),
                    domain=self._classify_domain(sentence),
                    hierarchy_level=self._determine_hierarchy_level(sentence),
                    confidence_score=0.5  # Lower confidence for fallback
                )
                processes.append(process)
        
        return processes
    
    def _extract_decision_points(self, text: str) -> List[ExtractedDecisionPoint]:
        """Extract decision points from text"""
        decision_points = []
        
        # Find sentences with decision indicators
        sentences = text.split('.')
        for sentence in sentences:
            if self._is_decision_sentence(sentence):
                decision_point = self._extract_decision_from_sentence(sentence)
                if decision_point:
                    decision_points.append(decision_point)
        
        return decision_points
    
    def _is_decision_sentence(self, sentence: str) -> bool:
        """Determine if a sentence contains a decision point"""
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in self.decision_patterns['decision_indicators'])
    
    def _extract_decision_from_sentence(self, sentence: str) -> Optional[ExtractedDecisionPoint]:
        """Extract decision point from sentence"""
        try:
            # Extract decision name
            name = sentence.split()[:8]  # First 8 words as name
            name = ' '.join(name).strip()
            
            # Determine decision type
            decision_type = self._determine_decision_type(sentence)
            
            # Extract criteria and outcomes
            criteria = self._extract_decision_criteria(sentence)
            outcomes = self._extract_decision_outcomes(sentence)
            
            # Extract authority information
            authority_level = self._extract_authority_level(sentence)
            
            # Calculate confidence
            confidence = self._calculate_decision_confidence(sentence)
            
            return ExtractedDecisionPoint(
                name=name,
                description=sentence,
                decision_type=decision_type,
                criteria=criteria,
                outcomes=outcomes,
                authority_level=authority_level,
                escalation_path=[],
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error extracting decision point: {e}")
            return None
    
    def _determine_decision_type(self, sentence: str) -> str:
        """Determine the type of decision"""
        sentence_lower = sentence.lower()
        
        if any(binary in sentence_lower for binary in self.decision_patterns['binary_decisions']):
            return 'binary'
        elif any(threshold in sentence_lower for threshold in self.decision_patterns['threshold_decisions']):
            return 'threshold'
        elif 'choose' in sentence_lower or 'select' in sentence_lower:
            return 'multiple_choice'
        else:
            return 'conditional'
    
    def _extract_decision_criteria(self, sentence: str) -> Dict[str, Any]:
        """Extract decision criteria from sentence"""
        criteria = {}
        
        # Look for conditional statements
        if_match = re.search(r'if\s+([^,]+)', sentence.lower())
        if if_match:
            criteria['condition'] = if_match.group(1).strip()
        
        # Look for threshold values
        threshold_match = re.search(r'(greater than|less than|exceeds|below)\s+([^\s,]+)', sentence.lower())
        if threshold_match:
            criteria['threshold_operator'] = threshold_match.group(1)
            criteria['threshold_value'] = threshold_match.group(2)
        
        return criteria
    
    def _extract_decision_outcomes(self, sentence: str) -> List[Dict[str, Any]]:
        """Extract possible outcomes from decision"""
        outcomes = []
        
        # Look for then/else patterns
        then_match = re.search(r'then\s+([^,]+)', sentence.lower())
        if then_match:
            outcomes.append({'action': then_match.group(1).strip(), 'type': 'positive'})
        
        else_match = re.search(r'else\s+([^,]+)', sentence.lower())
        if else_match:
            outcomes.append({'action': else_match.group(1).strip(), 'type': 'negative'})
        
        return outcomes
    
    def _extract_authority_level(self, sentence: str) -> str:
        """Extract authority level required for decision"""
        sentence_lower = sentence.lower()
        
        authority_indicators = {
            'supervisor': ['supervisor', 'manager', 'lead'],
            'management': ['management', 'director', 'executive'],
            'operator': ['operator', 'technician', 'worker'],
            'engineer': ['engineer', 'specialist', 'expert']
        }
        
        for level, indicators in authority_indicators.items():
            if any(indicator in sentence_lower for indicator in indicators):
                return level
        
        return 'operator'  # Default
    
    def _calculate_decision_confidence(self, sentence: str) -> float:
        """Calculate confidence score for decision extraction"""
        confidence_factors = []
        
        # Check for decision indicators
        if any(indicator in sentence.lower() for indicator in self.decision_patterns['decision_indicators']):
            confidence_factors.append(0.4)
        
        # Check for structured decision language
        if any(pattern in sentence.lower() for pattern in ['if', 'then', 'else', 'when']):
            confidence_factors.append(0.3)
        
        # Check for specific outcomes
        if any(word in sentence.lower() for word in ['approve', 'reject', 'continue', 'stop']):
            confidence_factors.append(0.2)
        
        # Check for authority mentions
        if any(word in sentence.lower() for word in ['supervisor', 'manager', 'operator']):
            confidence_factors.append(0.1)
        
        return min(sum(confidence_factors), 1.0)
    
    def _extract_compliance_items(self, text: str) -> List[ExtractedComplianceItem]:
        """Extract compliance requirements from text"""
        compliance_items = []
        
        # Find sentences with compliance indicators
        sentences = text.split('.')
        for sentence in sentences:
            if self._is_compliance_sentence(sentence):
                compliance_item = self._extract_compliance_from_sentence(sentence)
                if compliance_item:
                    compliance_items.append(compliance_item)
        
        return compliance_items
    
    def _is_compliance_sentence(self, sentence: str) -> bool:
        """Determine if sentence contains compliance information"""
        sentence_lower = sentence.lower()
        
        # Check for regulatory bodies
        has_regulatory_body = any(body in sentence_lower for body in self.compliance_patterns['regulatory_bodies'])
        
        # Check for compliance indicators
        has_compliance_indicator = any(indicator in sentence_lower for indicator in self.compliance_patterns['compliance_indicators'])
        
        # Check for standards patterns
        has_standard_pattern = any(re.search(pattern, sentence) for pattern in self.compliance_patterns['standards_patterns'])
        
        return has_regulatory_body or has_compliance_indicator or has_standard_pattern
    
    def _extract_compliance_from_sentence(self, sentence: str) -> Optional[ExtractedComplianceItem]:
        """Extract compliance item from sentence"""
        try:
            # Extract regulation name
            regulation_name = self._extract_regulation_name(sentence)
            
            # Extract section if present
            section = self._extract_regulation_section(sentence)
            
            # Determine compliance status
            status = self._determine_compliance_status(sentence)
            
            # Extract responsible party
            responsible_party = self._extract_responsible_party(sentence)
            
            # Calculate confidence
            confidence = self._calculate_compliance_confidence(sentence)
            
            return ExtractedComplianceItem(
                regulation_name=regulation_name,
                section=section or '',
                description=sentence,
                status=status,
                responsible_party=responsible_party or 'Not specified',
                evidence_required=[],
                review_frequency='Annual',  # Default
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error extracting compliance item: {e}")
            return None
    
    def _extract_regulation_name(self, sentence: str) -> str:
        """Extract regulation name from sentence"""
        # Look for known regulatory patterns
        for pattern in self.compliance_patterns['standards_patterns']:
            match = re.search(pattern, sentence)
            if match:
                return match.group(0)
        
        # Look for regulatory bodies
        for body in self.compliance_patterns['regulatory_bodies']:
            if body in sentence:
                return body
        
        return 'General Compliance'
    
    def _extract_regulation_section(self, sentence: str) -> Optional[str]:
        """Extract regulation section from sentence"""
        section_patterns = [
            r'section\s+(\d+(?:\.\d+)*)',
            r'part\s+(\d+(?:\.\d+)*)',
            r'clause\s+(\d+(?:\.\d+)*)'
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, sentence.lower())
            if match:
                return match.group(1)
        
        return None
    
    def _determine_compliance_status(self, sentence: str) -> ComplianceStatus:
        """Determine compliance status from sentence"""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['compliant', 'meets', 'satisfies']):
            return ComplianceStatus.COMPLIANT
        elif any(word in sentence_lower for word in ['non-compliant', 'violates', 'fails']):
            return ComplianceStatus.NON_COMPLIANT
        elif any(word in sentence_lower for word in ['review', 'assess', 'evaluate']):
            return ComplianceStatus.UNDER_REVIEW
        else:
            return ComplianceStatus.NOT_APPLICABLE
    
    def _extract_responsible_party(self, sentence: str) -> Optional[str]:
        """Extract responsible party from sentence"""
        responsibility_patterns = [
            r'responsible\s+(?:party|person)?:?\s*([^,\.]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:is|will be)\s+responsible',
            r'(?:manager|supervisor|operator|engineer)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        ]
        
        for pattern in responsibility_patterns:
            match = re.search(pattern, sentence)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _calculate_compliance_confidence(self, sentence: str) -> float:
        """Calculate confidence score for compliance extraction"""
        confidence_factors = []
        
        # Check for regulatory bodies
        if any(body in sentence.lower() for body in self.compliance_patterns['regulatory_bodies']):
            confidence_factors.append(0.4)
        
        # Check for compliance indicators
        if any(indicator in sentence.lower() for indicator in self.compliance_patterns['compliance_indicators']):
            confidence_factors.append(0.3)
        
        # Check for standards patterns
        if any(re.search(pattern, sentence) for pattern in self.compliance_patterns['standards_patterns']):
            confidence_factors.append(0.2)
        
        # Check for specific compliance language
        if any(word in sentence.lower() for word in ['shall', 'must', 'required']):
            confidence_factors.append(0.1)
        
        return min(sum(confidence_factors), 1.0)
    
    def _extract_risk_assessments(self, text: str) -> List[ExtractedRiskAssessment]:
        """Extract risk assessments from text"""
        risk_assessments = []
        
        # Find sentences with risk indicators
        sentences = text.split('.')
        for sentence in sentences:
            if self._is_risk_sentence(sentence):
                risk_assessment = self._extract_risk_from_sentence(sentence)
                if risk_assessment:
                    risk_assessments.append(risk_assessment)
        
        return risk_assessments
    
    def _is_risk_sentence(self, sentence: str) -> bool:
        """Determine if sentence contains risk information"""
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in self.risk_patterns['risk_indicators'])
    
    def _extract_risk_from_sentence(self, sentence: str) -> Optional[ExtractedRiskAssessment]:
        """Extract risk assessment from sentence"""
        try:
            # Determine risk category
            category = self._determine_risk_category(sentence)
            
            # Extract likelihood and impact
            likelihood = self._extract_risk_likelihood(sentence)
            impact = self._extract_risk_impact(sentence)
            
            # Extract mitigation strategies
            mitigation_strategies = self._extract_mitigation_strategies(sentence)
            
            # Extract responsible party
            responsible_party = self._extract_responsible_party(sentence)
            
            # Calculate confidence
            confidence = self._calculate_risk_confidence(sentence)
            
            return ExtractedRiskAssessment(
                category=category,
                description=sentence,
                likelihood=likelihood,
                impact=impact,
                mitigation_strategies=mitigation_strategies,
                monitoring_requirements=[],
                responsible_party=responsible_party or 'Not specified',
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error extracting risk assessment: {e}")
            return None
    
    def _determine_risk_category(self, sentence: str) -> str:
        """Determine risk category from sentence"""
        sentence_lower = sentence.lower()
        
        categories = {
            'safety': ['safety', 'injury', 'accident', 'hazard'],
            'operational': ['operational', 'process', 'production', 'downtime'],
            'financial': ['financial', 'cost', 'budget', 'economic'],
            'regulatory': ['regulatory', 'compliance', 'legal', 'violation'],
            'environmental': ['environmental', 'pollution', 'contamination', 'emission']
        }
        
        for category, keywords in categories.items():
            if any(keyword in sentence_lower for keyword in keywords):
                return category
        
        return 'operational'  # Default
    
    def _extract_risk_likelihood(self, sentence: str) -> RiskLevel:
        """Extract risk likelihood from sentence"""
        sentence_lower = sentence.lower()
        
        for level, indicators in self.risk_patterns['severity_levels'].items():
            if any(indicator in sentence_lower for indicator in indicators):
                return level
        
        return RiskLevel.MEDIUM  # Default
    
    def _extract_risk_impact(self, sentence: str) -> RiskLevel:
        """Extract risk impact from sentence"""
        # Similar to likelihood extraction
        return self._extract_risk_likelihood(sentence)
    
    def _extract_mitigation_strategies(self, sentence: str) -> List[str]:
        """Extract mitigation strategies from sentence"""
        strategies = []
        
        # Look for mitigation indicators
        for indicator in self.risk_patterns['mitigation_indicators']:
            if indicator in sentence.lower():
                # Extract text after the indicator
                pattern = rf'{indicator}\s+([^,\.]+)'
                match = re.search(pattern, sentence.lower())
                if match:
                    strategies.append(match.group(1).strip())
        
        return strategies
    
    def _calculate_risk_confidence(self, sentence: str) -> float:
        """Calculate confidence score for risk extraction"""
        confidence_factors = []
        
        # Check for risk indicators
        if any(indicator in sentence.lower() for indicator in self.risk_patterns['risk_indicators']):
            confidence_factors.append(0.4)
        
        # Check for severity levels
        if any(any(indicator in sentence.lower() for indicator in indicators) 
               for indicators in self.risk_patterns['severity_levels'].values()):
            confidence_factors.append(0.3)
        
        # Check for mitigation indicators
        if any(indicator in sentence.lower() for indicator in self.risk_patterns['mitigation_indicators']):
            confidence_factors.append(0.2)
        
        # Check for quantitative information
        if re.search(r'\d+%|\d+\.\d+', sentence):
            confidence_factors.append(0.1)
        
        return min(sum(confidence_factors), 1.0)
    
    def _extract_equipment_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract equipment relationships and dependencies"""
        relationships = []
        
        # Find equipment mentions and their relationships
        equipment_patterns = [
            r'(pump|motor|valve|sensor|compressor|tank)\s+([A-Z0-9-]+)',
            r'([A-Z][a-z]+\s+(?:pump|motor|valve|sensor|compressor|tank))'
        ]
        
        for pattern in equipment_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                equipment_name = match.group(0)
                # Extract context around the equipment mention
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                # Analyze relationships in context
                relationship = self._analyze_equipment_context(equipment_name, context)
                if relationship:
                    relationships.append(relationship)
        
        return relationships
    
    def _analyze_equipment_context(self, equipment_name: str, context: str) -> Optional[Dict[str, Any]]:
        """Analyze equipment context for relationships"""
        relationships = []
        
        # Look for operational relationships
        if 'operates' in context.lower() or 'controls' in context.lower():
            relationships.append('controls')
        
        if 'connected to' in context.lower() or 'linked to' in context.lower():
            relationships.append('connected')
        
        if 'depends on' in context.lower() or 'requires' in context.lower():
            relationships.append('depends')
        
        if relationships:
            return {
                'equipment': equipment_name,
                'relationships': relationships,
                'context': context,
                'confidence': 0.7
            }
        
        return None
    
    def _extract_personnel_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract personnel relationships and responsibilities"""
        relationships = []
        
        # Find personnel mentions
        personnel_patterns = [
            r'(operator|technician|engineer|supervisor|manager)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+),?\s+(operator|technician|engineer|supervisor|manager)'
        ]
        
        for pattern in personnel_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                person_info = match.group(0)
                # Extract context around the person mention
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                # Analyze relationships in context
                relationship = self._analyze_personnel_context(person_info, context)
                if relationship:
                    relationships.append(relationship)
        
        return relationships
    
    def _analyze_personnel_context(self, person_info: str, context: str) -> Optional[Dict[str, Any]]:
        """Analyze personnel context for relationships"""
        responsibilities = []
        
        # Look for responsibility indicators
        if 'responsible for' in context.lower():
            responsibilities.append('responsible')
        
        if 'reports to' in context.lower():
            responsibilities.append('reports_to')
        
        if 'supervises' in context.lower() or 'manages' in context.lower():
            responsibilities.append('supervises')
        
        if 'authorized to' in context.lower():
            responsibilities.append('authorized')
        
        if responsibilities:
            return {
                'person': person_info,
                'responsibilities': responsibilities,
                'context': context,
                'confidence': 0.7
            }
        
        return None
    
    def _build_knowledge_hierarchy(self, processes: List[ExtractedProcess]) -> Dict[str, Any]:
        """Build hierarchical structure of knowledge"""
        hierarchy = {
            'core_functions': [],
            'departments': [],
            'procedures': [],
            'steps': []
        }
        
        for process in processes:
            level_map = {
                HierarchyLevel.CORE_BUSINESS_FUNCTION: 'core_functions',
                HierarchyLevel.DEPARTMENT_OPERATION: 'departments',
                HierarchyLevel.INDIVIDUAL_PROCEDURE: 'procedures',
                HierarchyLevel.SPECIFIC_STEP: 'steps'
            }
            
            level_key = level_map.get(process.hierarchy_level, 'procedures')
            hierarchy[level_key].append({
                'process_id': process.process_id,
                'name': process.name,
                'domain': process.domain.value,
                'confidence': process.confidence_score
            })
        
        return hierarchy
    
    def _extract_implicit_knowledge(self, text: str) -> List[Dict[str, Any]]:
        """Extract implicit knowledge patterns"""
        implicit_knowledge = []
        
        # Look for implicit knowledge indicators
        implicit_patterns = [
            r'usually\s+([^,\.]+)',
            r'typically\s+([^,\.]+)',
            r'normally\s+([^,\.]+)',
            r'often\s+([^,\.]+)',
            r'sometimes\s+([^,\.]+)',
            r'experience\s+shows\s+([^,\.]+)',
            r'best\s+practice\s+([^,\.]+)'
        ]
        
        for pattern in implicit_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                implicit_knowledge.append({
                    'type': 'experiential',
                    'description': match.group(0),
                    'extracted_knowledge': match.group(1).strip(),
                    'confidence': 0.6
                })
        
        return implicit_knowledge
    
    def _calculate_confidence_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall confidence scores for extracted knowledge"""
        confidence_scores = {}
        
        # Calculate average confidence for each knowledge type
        for knowledge_type, items in results.items():
            if isinstance(items, list) and items:
                if hasattr(items[0], 'confidence') or (isinstance(items[0], dict) and 'confidence' in items[0]):
                    confidences = []
                    for item in items:
                        if hasattr(item, 'confidence'):
                            confidences.append(item.confidence)
                        elif isinstance(item, dict) and 'confidence' in item:
                            confidences.append(item['confidence'])
                    
                    if confidences:
                        confidence_scores[knowledge_type] = sum(confidences) / len(confidences)
        
        # Calculate overall confidence
        if confidence_scores:
            confidence_scores['overall'] = sum(confidence_scores.values()) / len(confidence_scores)
        else:
            confidence_scores['overall'] = 0.0
        
        return confidence_scores
