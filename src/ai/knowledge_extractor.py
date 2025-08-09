"""
EXPLAINIUM - Consolidated Knowledge Extractor

A clean, professional implementation that consolidates all knowledge extraction
functionality from multiple AI modules into a single, efficient system.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# NLP and AI libraries
import spacy
from transformers import pipeline
import torch

# Internal imports
from src.logging_config import get_logger
from src.config import config_manager

logger = get_logger(__name__)


class KnowledgeDomain(Enum):
    """Knowledge domains for classification"""
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


class HierarchyLevel(Enum):
    """Process hierarchy levels"""
    CORE_FUNCTION = "core_function"
    OPERATION = "operation"
    PROCEDURE = "procedure"
    SPECIFIC_STEP = "specific_step"


class CriticalityLevel(Enum):
    """Criticality levels for processes and requirements"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ExtractedProcess:
    """Represents an extracted organizational process"""
    name: str
    description: str
    domain: KnowledgeDomain
    hierarchy_level: HierarchyLevel
    steps: List[str]
    prerequisites: List[str]
    success_criteria: List[str]
    responsible_parties: List[str]
    duration: Optional[str]
    frequency: Optional[str]
    criticality: CriticalityLevel
    confidence: float


@dataclass
class ExtractedDecisionPoint:
    """Represents an extracted decision point"""
    name: str
    description: str
    criteria: Dict[str, Any]
    outcomes: List[Dict[str, Any]]
    authority_level: str
    escalation_path: Optional[str]
    confidence: float


@dataclass
class ExtractedComplianceItem:
    """Represents an extracted compliance requirement"""
    regulation_name: str
    section: Optional[str]
    requirement: str
    responsible_party: Optional[str]
    review_frequency: Optional[str]
    status: str
    confidence: float


@dataclass
class ExtractedRiskAssessment:
    """Represents an extracted risk assessment"""
    hazard: str
    risk_description: str
    likelihood: str
    impact: str
    overall_risk_level: str
    mitigation_strategies: List[str]
    confidence: float


class KnowledgeExtractor:
    """
    Consolidated knowledge extraction engine that processes text and extracts
    structured organizational knowledge including processes, decisions, compliance,
    and risk assessments.
    """
    
    def __init__(self):
        self.nlp = None
        self.classifier = None
        self.ner_pipeline = None
        
        # Initialize AI models
        self._init_models()
        
        # Define extraction patterns
        self._init_patterns()
    
    def _init_models(self):
        """Initialize AI models for knowledge extraction"""
        try:
            # Load spaCy model for NLP processing
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found, using fallback methods")
            self.nlp = None
        
        try:
            # Initialize classification pipeline
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            logger.info("Classification model loaded successfully")
        except Exception as e:
            logger.warning(f"Classification model failed to load: {e}")
            self.classifier = None
        
        try:
            # Initialize NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
            logger.info("NER model loaded successfully")
        except Exception as e:
            logger.warning(f"NER model failed to load: {e}")
            self.ner_pipeline = None
    
    def _init_patterns(self):
        """Initialize regex patterns for knowledge extraction"""
        self.process_patterns = {
            'action_verbs': [
                r'\b(perform|execute|conduct|carry out|implement|complete|follow|ensure|verify|check|inspect|test|measure|record|document|report|review|approve|authorize|validate|confirm)\b',
                r'\b(must|shall|should|will|need to|required to|responsible for)\b',
                r'\b(step|procedure|process|operation|task|activity|action|method)\b'
            ],
            'sequence_indicators': [
                r'\b(first|second|third|next|then|after|before|finally|lastly)\b',
                r'\b(step \d+|phase \d+|\d+\.|^\d+\))',
                r'\b(initially|subsequently|following|preceding)\b'
            ]
        }
        
        self.decision_patterns = {
            'decision_indicators': [
                r'\b(if|when|unless|provided that|in case|should|whether)\b',
                r'\b(decide|determine|choose|select|evaluate|assess|consider)\b',
                r'\b(approval|authorization|permission|clearance)\b'
            ],
            'criteria_indicators': [
                r'\b(criteria|requirements|conditions|standards|specifications)\b',
                r'\b(greater than|less than|equal to|exceeds|below|above)\b',
                r'\b(acceptable|unacceptable|satisfactory|compliant)\b'
            ]
        }
        
        self.compliance_patterns = {
            'regulatory_bodies': [
                r'\b(OSHA|EPA|FDA|ANSI|ISO|NFPA|ASTM|API|ASME|IEEE)\b',
                r'\b(regulation|standard|code|requirement|guideline)\b',
                r'\b(compliance|conformance|adherence|accordance)\b'
            ],
            'compliance_indicators': [
                r'\b(must comply|shall meet|required by|mandated by|according to)\b',
                r'\b(violation|non-compliance|breach|infringement)\b'
            ]
        }
        
        self.risk_patterns = {
            'hazard_indicators': [
                r'\b(hazard|danger|risk|threat|peril|unsafe|dangerous)\b',
                r'\b(injury|accident|incident|exposure|contamination)\b',
                r'\b(fire|explosion|chemical|electrical|mechanical|biological)\b'
            ],
            'risk_levels': [
                r'\b(high risk|medium risk|low risk|critical|severe|moderate|minor)\b',
                r'\b(likely|unlikely|probable|possible|rare|frequent)\b'
            ]
        }
    
    def extract_knowledge(self, text: str, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to extract comprehensive knowledge from text
        
        Args:
            text: Input text to process
            document_metadata: Metadata about the source document
            
        Returns:
            Dictionary containing all extracted knowledge
        """
        if not text or not isinstance(text, str):
            return self._empty_knowledge_result()
        
        # Clean and preprocess text
        cleaned_text = self._clean_text(text)
        
        # Extract entities if NER is available
        entities = self._extract_entities(cleaned_text) if self.ner_pipeline else []
        
        # Classify document domain
        domain = self._classify_domain(cleaned_text)
        
        # Extract different types of knowledge
        processes = self._extract_processes(cleaned_text, document_metadata)
        decision_points = self._extract_decision_points(cleaned_text)
        compliance_items = self._extract_compliance_items(cleaned_text)
        risk_assessments = self._extract_risk_assessments(cleaned_text)
        
        # Calculate overall confidence
        all_items = processes + decision_points + compliance_items + risk_assessments
        overall_confidence = (
            sum(item.confidence for item in all_items) / len(all_items)
            if all_items else 0.0
        )
        
        return {
            'document_metadata': document_metadata,
            'domain': domain.value if domain else 'general',
            'entities': entities,
            'processes': [self._process_to_dict(p) for p in processes],
            'decision_points': [self._decision_to_dict(d) for d in decision_points],
            'compliance_items': [self._compliance_to_dict(c) for c in compliance_items],
            'risk_assessments': [self._risk_to_dict(r) for r in risk_assessments],
            'overall_confidence': overall_confidence,
            'extraction_timestamp': datetime.now().isoformat(),
            'total_items_extracted': len(all_items)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/]', ' ', text)
        
        return text.strip()
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        if not self.ner_pipeline:
            return []
        
        try:
            entities = self.ner_pipeline(text)
            return [
                {
                    'text': entity['word'],
                    'label': entity['entity_group'],
                    'confidence': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                }
                for entity in entities
                if entity['score'] > 0.7  # Filter low-confidence entities
            ]
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _classify_domain(self, text: str) -> Optional[KnowledgeDomain]:
        """Classify the document domain"""
        if not self.classifier:
            return None
        
        try:
            domain_labels = [domain.value.replace('_', ' ') for domain in KnowledgeDomain]
            result = self.classifier(text[:1000], domain_labels)  # Limit text length
            
            if result['scores'][0] > 0.5:  # Confidence threshold
                domain_name = result['labels'][0].replace(' ', '_')
                return KnowledgeDomain(domain_name)
        except Exception as e:
            logger.error(f"Domain classification failed: {e}")
        
        return None
    
    def _extract_processes(self, text: str, metadata: Dict[str, Any]) -> List[ExtractedProcess]:
        """Extract organizational processes from text"""
        processes = []
        sentences = self._split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            if self._is_process_sentence(sentence):
                process = self._extract_process_from_sentence(sentence, i, metadata)
                if process:
                    processes.append(process)
        
        return processes
    
    def _extract_decision_points(self, text: str) -> List[ExtractedDecisionPoint]:
        """Extract decision points from text"""
        decision_points = []
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            if self._is_decision_sentence(sentence):
                decision = self._extract_decision_from_sentence(sentence)
                if decision:
                    decision_points.append(decision)
        
        return decision_points
    
    def _extract_compliance_items(self, text: str) -> List[ExtractedComplianceItem]:
        """Extract compliance requirements from text"""
        compliance_items = []
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            if self._is_compliance_sentence(sentence):
                compliance = self._extract_compliance_from_sentence(sentence)
                if compliance:
                    compliance_items.append(compliance)
        
        return compliance_items
    
    def _extract_risk_assessments(self, text: str) -> List[ExtractedRiskAssessment]:
        """Extract risk assessments from text"""
        risk_assessments = []
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            if self._is_risk_sentence(sentence):
                risk = self._extract_risk_from_sentence(sentence)
                if risk:
                    risk_assessments.append(risk)
        
        return risk_assessments
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _is_process_sentence(self, sentence: str) -> bool:
        """Check if sentence describes a process"""
        sentence_lower = sentence.lower()
        
        # Check for action verbs
        for pattern in self.process_patterns['action_verbs']:
            if re.search(pattern, sentence_lower):
                return True
        
        # Check for sequence indicators
        for pattern in self.process_patterns['sequence_indicators']:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _is_decision_sentence(self, sentence: str) -> bool:
        """Check if sentence describes a decision point"""
        sentence_lower = sentence.lower()
        
        for pattern in self.decision_patterns['decision_indicators']:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _is_compliance_sentence(self, sentence: str) -> bool:
        """Check if sentence describes compliance requirements"""
        sentence_lower = sentence.lower()
        
        for pattern in self.compliance_patterns['regulatory_bodies']:
            if re.search(pattern, sentence_lower):
                return True
        
        for pattern in self.compliance_patterns['compliance_indicators']:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _is_risk_sentence(self, sentence: str) -> bool:
        """Check if sentence describes risks or hazards"""
        sentence_lower = sentence.lower()
        
        for pattern in self.risk_patterns['hazard_indicators']:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _extract_process_from_sentence(self, sentence: str, index: int, metadata: Dict[str, Any]) -> Optional[ExtractedProcess]:
        """Extract a process from a single sentence"""
        try:
            name = self._extract_process_name(sentence)
            description = sentence.strip()
            
            # Determine hierarchy level
            hierarchy_level = self._determine_hierarchy_level(sentence)
            
            # Extract steps (simplified)
            steps = [sentence.strip()]
            
            # Extract other attributes with fallbacks
            prerequisites = self._extract_prerequisites(sentence)
            success_criteria = self._extract_success_criteria(sentence)
            responsible_parties = self._extract_responsible_parties(sentence)
            duration = self._extract_duration(sentence)
            frequency = self._extract_frequency(sentence)
            
            # Determine criticality
            criticality = self._determine_criticality(sentence)
            
            # Calculate confidence
            confidence = self._calculate_process_confidence(sentence)
            
            # Determine domain
            domain = self._determine_process_domain(sentence)
            
            return ExtractedProcess(
                name=name,
                description=description,
                domain=domain,
                hierarchy_level=hierarchy_level,
                steps=steps,
                prerequisites=prerequisites,
                success_criteria=success_criteria,
                responsible_parties=responsible_parties,
                duration=duration,
                frequency=frequency,
                criticality=criticality,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error extracting process from sentence: {e}")
            return None
    
    def _extract_process_name(self, sentence: str) -> str:
        """Extract process name from sentence"""
        # Simple heuristic: use first few words or the whole sentence if short
        words = sentence.split()
        if len(words) <= 5:
            return sentence.strip()
        else:
            return ' '.join(words[:5]) + "..."
    
    def _determine_hierarchy_level(self, sentence: str) -> HierarchyLevel:
        """Determine the hierarchy level of a process"""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['step', 'action', 'task']):
            return HierarchyLevel.SPECIFIC_STEP
        elif any(word in sentence_lower for word in ['procedure', 'method', 'protocol']):
            return HierarchyLevel.PROCEDURE
        elif any(word in sentence_lower for word in ['operation', 'process', 'workflow']):
            return HierarchyLevel.OPERATION
        else:
            return HierarchyLevel.CORE_FUNCTION
    
    def _determine_process_domain(self, sentence: str) -> KnowledgeDomain:
        """Determine the domain of a process"""
        sentence_lower = sentence.lower()
        
        domain_keywords = {
            KnowledgeDomain.SAFETY_COMPLIANCE: ['safety', 'hazard', 'ppe', 'protection', 'emergency'],
            KnowledgeDomain.MAINTENANCE: ['maintenance', 'repair', 'service', 'calibration', 'inspection'],
            KnowledgeDomain.QUALITY_ASSURANCE: ['quality', 'test', 'validation', 'verification', 'standard'],
            KnowledgeDomain.TRAINING: ['training', 'education', 'certification', 'competency', 'skill'],
            KnowledgeDomain.EQUIPMENT_TECHNOLOGY: ['equipment', 'machine', 'device', 'system', 'technology'],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in sentence_lower for keyword in keywords):
                return domain
        
        return KnowledgeDomain.OPERATIONAL
    
    def _determine_criticality(self, sentence: str) -> CriticalityLevel:
        """Determine the criticality level"""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['critical', 'essential', 'vital', 'mandatory']):
            return CriticalityLevel.CRITICAL
        elif any(word in sentence_lower for word in ['important', 'significant', 'required']):
            return CriticalityLevel.HIGH
        elif any(word in sentence_lower for word in ['recommended', 'suggested', 'preferred']):
            return CriticalityLevel.MEDIUM
        else:
            return CriticalityLevel.LOW
    
    def _extract_prerequisites(self, sentence: str) -> List[str]:
        """Extract prerequisites from sentence"""
        prerequisites = []
        sentence_lower = sentence.lower()
        
        prerequisite_patterns = [
            r'before\s+(.+?)(?:\.|,|$)',
            r'requires?\s+(.+?)(?:\.|,|$)',
            r'must\s+have\s+(.+?)(?:\.|,|$)',
            r'needs?\s+(.+?)(?:\.|,|$)'
        ]
        
        for pattern in prerequisite_patterns:
            matches = re.findall(pattern, sentence_lower)
            prerequisites.extend(matches)
        
        return [p.strip() for p in prerequisites if p.strip()]
    
    def _extract_success_criteria(self, sentence: str) -> List[str]:
        """Extract success criteria from sentence"""
        criteria = []
        sentence_lower = sentence.lower()
        
        criteria_patterns = [
            r'until\s+(.+?)(?:\.|,|$)',
            r'when\s+(.+?)(?:\.|,|$)',
            r'success\s+(?:is\s+)?(.+?)(?:\.|,|$)',
            r'complete\s+when\s+(.+?)(?:\.|,|$)'
        ]
        
        for pattern in criteria_patterns:
            matches = re.findall(pattern, sentence_lower)
            criteria.extend(matches)
        
        return [c.strip() for c in criteria if c.strip()]
    
    def _extract_responsible_parties(self, sentence: str) -> List[str]:
        """Extract responsible parties from sentence"""
        parties = []
        
        # Look for role-based patterns
        role_patterns = [
            r'\b(operator|technician|supervisor|manager|engineer|inspector|auditor)\b',
            r'\b(responsible\s+(?:person|party|individual))\b',
            r'\b(assigned\s+(?:to|by))\s+(\w+)'
        ]
        
        for pattern in role_patterns:
            matches = re.findall(pattern, sentence.lower())
            if isinstance(matches[0], tuple) if matches else False:
                parties.extend([match[-1] for match in matches])
            else:
                parties.extend(matches)
        
        return list(set(parties))  # Remove duplicates
    
    def _extract_duration(self, sentence: str) -> Optional[str]:
        """Extract duration from sentence"""
        duration_patterns = [
            r'(\d+)\s*(minutes?|hours?|days?|weeks?|months?)',
            r'(approximately|about|roughly)\s+(\d+)\s*(minutes?|hours?|days?)',
            r'takes?\s+(\d+)\s*(minutes?|hours?|days?)'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, sentence.lower())
            if match:
                return match.group(0)
        
        return None
    
    def _extract_frequency(self, sentence: str) -> Optional[str]:
        """Extract frequency from sentence"""
        frequency_patterns = [
            r'\b(daily|weekly|monthly|quarterly|annually|yearly)\b',
            r'\b(every\s+\d+\s+(?:days?|weeks?|months?))\b',
            r'\b(once\s+(?:per|a)\s+(?:day|week|month|year))\b'
        ]
        
        for pattern in frequency_patterns:
            match = re.search(pattern, sentence.lower())
            if match:
                return match.group(0)
        
        return None
    
    def _calculate_process_confidence(self, sentence: str) -> float:
        """Calculate confidence score for process extraction"""
        confidence_factors = []
        
        # Check for action verbs
        action_verb_count = sum(
            1 for pattern in self.process_patterns['action_verbs']
            if re.search(pattern, sentence.lower())
        )
        confidence_factors.append(min(action_verb_count * 0.3, 1.0))
        
        # Check for sequence indicators
        sequence_count = sum(
            1 for pattern in self.process_patterns['sequence_indicators']
            if re.search(pattern, sentence.lower())
        )
        confidence_factors.append(min(sequence_count * 0.2, 1.0))
        
        # Check sentence length (reasonable processes have moderate length)
        word_count = len(sentence.split())
        length_factor = 1.0 if 5 <= word_count <= 30 else 0.5
        confidence_factors.append(length_factor)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _extract_decision_from_sentence(self, sentence: str) -> Optional[ExtractedDecisionPoint]:
        """Extract decision point from sentence"""
        try:
            name = self._extract_decision_name(sentence)
            description = sentence.strip()
            criteria = self._extract_decision_criteria(sentence)
            outcomes = self._extract_decision_outcomes(sentence)
            authority_level = self._extract_authority_level(sentence)
            confidence = self._calculate_decision_confidence(sentence)
            
            return ExtractedDecisionPoint(
                name=name,
                description=description,
                criteria=criteria,
                outcomes=outcomes,
                authority_level=authority_level,
                escalation_path=None,  # Could be enhanced
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error extracting decision from sentence: {e}")
            return None
    
    def _extract_decision_name(self, sentence: str) -> str:
        """Extract decision name from sentence"""
        words = sentence.split()
        return ' '.join(words[:6]) + ("..." if len(words) > 6 else "")
    
    def _extract_decision_criteria(self, sentence: str) -> Dict[str, Any]:
        """Extract decision criteria from sentence"""
        criteria = {}
        sentence_lower = sentence.lower()
        
        # Look for numerical criteria
        number_patterns = [
            r'(\w+)\s+(?:is\s+)?(?:greater than|>|exceeds)\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+(?:is\s+)?(?:less than|<|below)\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+(?:equals?|=|is)\s+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, sentence_lower)
            for match in matches:
                criteria[match[0]] = match[1]
        
        return criteria
    
    def _extract_decision_outcomes(self, sentence: str) -> List[Dict[str, Any]]:
        """Extract decision outcomes from sentence"""
        outcomes = []
        
        # Look for conditional outcomes
        if 'if' in sentence.lower() and 'then' in sentence.lower():
            outcomes.append({
                'condition': 'if_condition',
                'action': 'then_action',
                'description': sentence
            })
        
        return outcomes
    
    def _extract_authority_level(self, sentence: str) -> str:
        """Extract authority level from sentence"""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['manager', 'supervisor', 'director']):
            return 'management'
        elif any(word in sentence_lower for word in ['operator', 'technician', 'worker']):
            return 'operational'
        else:
            return 'general'
    
    def _calculate_decision_confidence(self, sentence: str) -> float:
        """Calculate confidence for decision extraction"""
        confidence_factors = []
        
        # Check for decision indicators
        decision_count = sum(
            1 for pattern in self.decision_patterns['decision_indicators']
            if re.search(pattern, sentence.lower())
        )
        confidence_factors.append(min(decision_count * 0.4, 1.0))
        
        # Check for criteria indicators
        criteria_count = sum(
            1 for pattern in self.decision_patterns['criteria_indicators']
            if re.search(pattern, sentence.lower())
        )
        confidence_factors.append(min(criteria_count * 0.3, 1.0))
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _extract_compliance_from_sentence(self, sentence: str) -> Optional[ExtractedComplianceItem]:
        """Extract compliance item from sentence"""
        try:
            regulation_name = self._extract_regulation_name(sentence)
            section = self._extract_regulation_section(sentence)
            requirement = sentence.strip()
            responsible_party = self._extract_responsible_party(sentence)
            confidence = self._calculate_compliance_confidence(sentence)
            
            return ExtractedComplianceItem(
                regulation_name=regulation_name,
                section=section,
                requirement=requirement,
                responsible_party=responsible_party,
                review_frequency=None,  # Could be enhanced
                status='identified',
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error extracting compliance from sentence: {e}")
            return None
    
    def _extract_regulation_name(self, sentence: str) -> str:
        """Extract regulation name from sentence"""
        # Look for known regulatory patterns
        regulatory_patterns = [
            r'\b(OSHA\s+\d+(?:\.\d+)*)\b',
            r'\b(EPA\s+\d+(?:\.\d+)*)\b',
            r'\b(ISO\s+\d+(?:-\d+)*)\b',
            r'\b(ANSI\s+\w+(?:\.\d+)*)\b'
        ]
        
        for pattern in regulatory_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return match.group(1)
        
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
    
    def _extract_responsible_party(self, sentence: str) -> Optional[str]:
        """Extract responsible party from sentence"""
        responsibility_patterns = [
            r'(?:responsible|accountable|assigned)\s+(?:to\s+)?(\w+)',
            r'(\w+)\s+(?:is\s+)?responsible',
            r'(\w+)\s+must\s+ensure'
        ]
        
        for pattern in responsibility_patterns:
            match = re.search(pattern, sentence.lower())
            if match:
                return match.group(1)
        
        return None
    
    def _calculate_compliance_confidence(self, sentence: str) -> float:
        """Calculate confidence for compliance extraction"""
        confidence_factors = []
        
        # Check for regulatory body mentions
        regulatory_count = sum(
            1 for pattern in self.compliance_patterns['regulatory_bodies']
            if re.search(pattern, sentence.lower())
        )
        confidence_factors.append(min(regulatory_count * 0.4, 1.0))
        
        # Check for compliance indicators
        compliance_count = sum(
            1 for pattern in self.compliance_patterns['compliance_indicators']
            if re.search(pattern, sentence.lower())
        )
        confidence_factors.append(min(compliance_count * 0.3, 1.0))
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _extract_risk_from_sentence(self, sentence: str) -> Optional[ExtractedRiskAssessment]:
        """Extract risk assessment from sentence"""
        try:
            hazard = self._extract_hazard(sentence)
            risk_description = sentence.strip()
            likelihood = self._extract_likelihood(sentence)
            impact = self._extract_impact(sentence)
            overall_risk_level = self._calculate_risk_level(likelihood, impact)
            mitigation_strategies = self._extract_mitigation_strategies(sentence)
            confidence = self._calculate_risk_confidence(sentence)
            
            return ExtractedRiskAssessment(
                hazard=hazard,
                risk_description=risk_description,
                likelihood=likelihood,
                impact=impact,
                overall_risk_level=overall_risk_level,
                mitigation_strategies=mitigation_strategies,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error extracting risk from sentence: {e}")
            return None
    
    def _extract_hazard(self, sentence: str) -> str:
        """Extract hazard from sentence"""
        hazard_patterns = [
            r'\b(fire|explosion|chemical|electrical|mechanical|biological)\s+hazard\b',
            r'\b(toxic|corrosive|flammable|explosive|radioactive)\b',
            r'\b(slip|fall|cut|burn|shock)\b'
        ]
        
        for pattern in hazard_patterns:
            match = re.search(pattern, sentence.lower())
            if match:
                return match.group(0)
        
        return 'General Hazard'
    
    def _extract_likelihood(self, sentence: str) -> str:
        """Extract likelihood from sentence"""
        likelihood_patterns = [
            r'\b(very likely|highly likely|likely|probable)\b',
            r'\b(unlikely|improbable|rare|remote)\b',
            r'\b(possible|potential|may occur)\b'
        ]
        
        for pattern in likelihood_patterns:
            match = re.search(pattern, sentence.lower())
            if match:
                return match.group(0)
        
        return 'unknown'
    
    def _extract_impact(self, sentence: str) -> str:
        """Extract impact from sentence"""
        impact_patterns = [
            r'\b(severe|serious|significant|major)\b',
            r'\b(minor|negligible|minimal)\b',
            r'\b(catastrophic|critical|moderate)\b'
        ]
        
        for pattern in impact_patterns:
            match = re.search(pattern, sentence.lower())
            if match:
                return match.group(0)
        
        return 'unknown'
    
    def _calculate_risk_level(self, likelihood: str, impact: str) -> str:
        """Calculate overall risk level"""
        # Simple risk matrix
        high_likelihood = likelihood.lower() in ['very likely', 'highly likely', 'likely', 'probable']
        high_impact = impact.lower() in ['severe', 'serious', 'significant', 'major', 'catastrophic', 'critical']
        
        if high_likelihood and high_impact:
            return 'high'
        elif high_likelihood or high_impact:
            return 'medium'
        else:
            return 'low'
    
    def _extract_mitigation_strategies(self, sentence: str) -> List[str]:
        """Extract mitigation strategies from sentence"""
        strategies = []
        
        mitigation_patterns = [
            r'(?:prevent|avoid|mitigate|reduce|control)\s+(.+?)(?:\.|,|$)',
            r'(?:use|wear|implement)\s+(.+?)(?:\.|,|$)',
            r'(?:ensure|maintain|check)\s+(.+?)(?:\.|,|$)'
        ]
        
        for pattern in mitigation_patterns:
            matches = re.findall(pattern, sentence.lower())
            strategies.extend(matches)
        
        return [s.strip() for s in strategies if s.strip()]
    
    def _calculate_risk_confidence(self, sentence: str) -> float:
        """Calculate confidence for risk extraction"""
        confidence_factors = []
        
        # Check for hazard indicators
        hazard_count = sum(
            1 for pattern in self.risk_patterns['hazard_indicators']
            if re.search(pattern, sentence.lower())
        )
        confidence_factors.append(min(hazard_count * 0.4, 1.0))
        
        # Check for risk level indicators
        risk_level_count = sum(
            1 for pattern in self.risk_patterns['risk_levels']
            if re.search(pattern, sentence.lower())
        )
        confidence_factors.append(min(risk_level_count * 0.3, 1.0))
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _empty_knowledge_result(self) -> Dict[str, Any]:
        """Return empty knowledge extraction result"""
        return {
            'document_metadata': {},
            'domain': 'general',
            'entities': [],
            'processes': [],
            'decision_points': [],
            'compliance_items': [],
            'risk_assessments': [],
            'overall_confidence': 0.0,
            'extraction_timestamp': datetime.now().isoformat(),
            'total_items_extracted': 0
        }
    
    def _process_to_dict(self, process: ExtractedProcess) -> Dict[str, Any]:
        """Convert ExtractedProcess to dictionary"""
        return {
            'name': process.name,
            'description': process.description,
            'domain': process.domain.value,
            'hierarchy_level': process.hierarchy_level.value,
            'steps': process.steps,
            'prerequisites': process.prerequisites,
            'success_criteria': process.success_criteria,
            'responsible_parties': process.responsible_parties,
            'duration': process.duration,
            'frequency': process.frequency,
            'criticality': process.criticality.value,
            'confidence': process.confidence
        }
    
    def _decision_to_dict(self, decision: ExtractedDecisionPoint) -> Dict[str, Any]:
        """Convert ExtractedDecisionPoint to dictionary"""
        return {
            'name': decision.name,
            'description': decision.description,
            'criteria': decision.criteria,
            'outcomes': decision.outcomes,
            'authority_level': decision.authority_level,
            'escalation_path': decision.escalation_path,
            'confidence': decision.confidence
        }
    
    def _compliance_to_dict(self, compliance: ExtractedComplianceItem) -> Dict[str, Any]:
        """Convert ExtractedComplianceItem to dictionary"""
        return {
            'regulation_name': compliance.regulation_name,
            'section': compliance.section,
            'requirement': compliance.requirement,
            'responsible_party': compliance.responsible_party,
            'review_frequency': compliance.review_frequency,
            'status': compliance.status,
            'confidence': compliance.confidence
        }
    
    def _risk_to_dict(self, risk: ExtractedRiskAssessment) -> Dict[str, Any]:
        """Convert ExtractedRiskAssessment to dictionary"""
        return {
            'hazard': risk.hazard,
            'risk_description': risk.risk_description,
            'likelihood': risk.likelihood,
            'impact': risk.impact,
            'overall_risk_level': risk.overall_risk_level,
            'mitigation_strategies': risk.mitigation_strategies,
            'confidence': risk.confidence
        }