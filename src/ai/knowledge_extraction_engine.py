"""
Core Knowledge Extraction Engine for EXPLAINIUM

This module implements sophisticated algorithms for extracting tacit knowledge
from enterprise documents, including workflow dependencies, decision patterns,
optimization opportunities, and communication flows.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import spacy
from transformers import pipeline
import networkx as nx

from src.logging_config import get_logger

logger = get_logger(__name__)

# Data structures for knowledge extraction results
@dataclass
class WorkflowDependency:
    """Represents a dependency between workflow processes."""
    source_process: str
    target_process: str
    dependency_type: str  # prerequisite, parallel, downstream, conditional
    strength: float  # 0.0-1.0
    conditions: Dict[str, Any]
    confidence: float

@dataclass
class DecisionPattern:
    """Represents an implicit decision-making pattern."""
    decision_point: str
    decision_type: str  # binary, multiple_choice, conditional, threshold
    conditions: Dict[str, Any]
    outcomes: List[Dict[str, Any]]
    confidence: float
    context: str

@dataclass
class OptimizationPattern:
    """Represents a resource optimization pattern."""
    pattern_type: str  # resource, time, quality, cost, safety
    description: str
    conditions: Dict[str, Any]
    improvements: Dict[str, Any]
    success_metrics: List[str]
    confidence: float
    impact_level: str

@dataclass
class CommunicationFlow:
    """Represents information flow between roles."""
    source_role: str
    target_role: str
    information_type: str
    communication_method: str
    frequency: str
    criticality: str
    formal_protocol: bool
    confidence: float

class TacitKnowledgeDetector:
    """Detects tacit knowledge patterns in enterprise documents."""
    
    def __init__(self):
        self.nlp = None
        self.sentiment_analyzer = None
        self._load_models()
        
        # Pattern libraries for different types of tacit knowledge
        self.workflow_patterns = self._initialize_workflow_patterns()
        self.decision_patterns = self._initialize_decision_patterns()
        self.optimization_patterns = self._initialize_optimization_patterns()
        self.communication_patterns = self._initialize_communication_patterns()
    
    def _load_models(self):
        """Load required NLP models."""
        try:
            # Load spaCy model for advanced NLP processing
            self.nlp = spacy.load("en_core_web_sm")
            
            # Load sentiment analysis for detecting optimization opportunities
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            logger.info("Successfully loaded NLP models for tacit knowledge detection")
        except Exception as e:
            logger.error(f"Failed to load NLP models: {e}")
            # Fallback to basic processing
            self.nlp = None
            self.sentiment_analyzer = None
    
    def _initialize_workflow_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting workflow dependencies."""
        return {
            'prerequisite': [
                r'\b(?:before|prior to|prerequisite|must be completed before|requires)\b',
                r'\b(?:after completing|once|following|subsequent to)\b',
                r'\b(?:depends on|dependent on|contingent on)\b'
            ],
            'parallel': [
                r'\b(?:simultaneously|concurrently|at the same time|in parallel)\b',
                r'\b(?:while|during|as)\b.*\b(?:process|procedure|operation)\b'
            ],
            'downstream': [
                r'\b(?:then|next|subsequently|following|after)\b',
                r'\b(?:leads to|results in|triggers|initiates)\b'
            ],
            'conditional': [
                r'\b(?:if|when|unless|provided that|in case of)\b',
                r'\b(?:depending on|based on|according to)\b'
            ]
        }
    
    def _initialize_decision_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting decision points."""
        return {
            'binary': [
                r'\b(?:yes or no|pass or fail|accept or reject)\b',
                r'\b(?:either|or|whether)\b.*\b(?:or|not)\b'
            ],
            'multiple_choice': [
                r'\b(?:options include|alternatives are|choices are)\b',
                r'\b(?:select from|choose between|pick one of)\b'
            ],
            'conditional': [
                r'\b(?:if.*then|when.*do|unless.*otherwise)\b',
                r'\b(?:in case of|depending on|based on)\b'
            ],
            'threshold': [
                r'\b(?:exceeds|below|above|greater than|less than)\b.*\b(?:limit|threshold|maximum|minimum)\b',
                r'\b(?:\d+%|\d+\.\d+|[\d,]+)\b.*\b(?:acceptable|required|critical)\b'
            ]
        }
    
    def _initialize_optimization_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting optimization opportunities."""
        return {
            'resource': [
                r'\b(?:waste|inefficient|redundant|duplicate|unnecessary)\b',
                r'\b(?:optimize|streamline|consolidate|reduce)\b.*\b(?:resources|materials|supplies)\b'
            ],
            'time': [
                r'\b(?:delay|bottleneck|waiting|idle time|downtime)\b',
                r'\b(?:faster|quicker|accelerate|expedite|reduce time)\b'
            ],
            'quality': [
                r'\b(?:defect|error|rework|rejection|failure)\b',
                r'\b(?:improve quality|enhance|upgrade|refine)\b'
            ],
            'cost': [
                r'\b(?:expensive|costly|budget|reduce cost|save money)\b',
                r'\b(?:cost-effective|economical|affordable)\b'
            ],
            'safety': [
                r'\b(?:hazard|risk|danger|unsafe|accident)\b',
                r'\b(?:safety improvement|risk reduction|safer)\b'
            ]
        }
    
    def _initialize_communication_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting communication flows."""
        return {
            'verbal': [
                r'\b(?:discuss|meeting|call|conversation|briefing)\b',
                r'\b(?:inform|notify|tell|communicate verbally)\b'
            ],
            'written': [
                r'\b(?:report|document|email|memo|letter)\b',
                r'\b(?:write|document|record|log)\b'
            ],
            'digital': [
                r'\b(?:system|database|software|application|portal)\b',
                r'\b(?:enter|input|update|access)\b.*\b(?:system|database)\b'
            ],
            'visual': [
                r'\b(?:diagram|chart|graph|display|monitor)\b',
                r'\b(?:show|display|visualize|indicate)\b'
            ]
        }

class WorkflowDependencyAnalyzer:
    """Analyzes workflow dependencies and hidden process connections."""
    
    def __init__(self, detector: TacitKnowledgeDetector):
        self.detector = detector
        self.dependency_graph = nx.DiGraph()
    
    def analyze_workflow_dependencies(self, content: str, entities: List[Dict]) -> List[WorkflowDependency]:
        """
        Identify hidden process connections and workflow dependencies.
        
        Args:
            content: Document text content
            entities: Extracted entities from NER
            
        Returns:
            List of WorkflowDependency objects
        """
        dependencies = []
        
        try:
            # Extract process mentions and their contexts
            processes = self._extract_process_mentions(content, entities)
            
            # Analyze relationships between processes
            for i, process1 in enumerate(processes):
                for j, process2 in enumerate(processes):
                    if i != j:
                        dependency = self._analyze_process_relationship(
                            process1, process2, content
                        )
                        if dependency and dependency.confidence > 0.3:
                            dependencies.append(dependency)
            
            # Build dependency graph for validation
            self._build_dependency_graph(dependencies)
            
            logger.info(f"Identified {len(dependencies)} workflow dependencies")
            
        except Exception as e:
            logger.error(f"Error analyzing workflow dependencies: {e}")
        
        return dependencies
    
    def _extract_process_mentions(self, content: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Extract process mentions from content."""
        processes = []
        
        # Process-related keywords
        process_keywords = [
            'procedure', 'process', 'operation', 'task', 'step', 'activity',
            'workflow', 'protocol', 'method', 'routine', 'sequence'
        ]
        
        # Find process mentions using regex and NLP
        for keyword in process_keywords:
            pattern = rf'\b{keyword}\b[^.]*?(?:\.|$)'
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                process_text = match.group(0).strip()
                if len(process_text) > 10:  # Filter out very short matches
                    processes.append({
                        'text': process_text,
                        'start': match.start(),
                        'end': match.end(),
                        'keyword': keyword,
                        'context': self._get_context(content, match.start(), match.end())
                    })
        
        # Add processes from entities (if any are classified as procedures)
        for entity in entities:
            if entity.get('entity_group') in ['PROCEDURE', 'PROCESS', 'TASK']:
                processes.append({
                    'text': entity['word'],
                    'start': entity.get('start', 0),
                    'end': entity.get('end', 0),
                    'keyword': 'entity',
                    'context': self._get_context(content, entity.get('start', 0), entity.get('end', 0))
                })
        
        return processes
    
    def _get_context(self, content: str, start: int, end: int, window: int = 200) -> str:
        """Get context around a text span."""
        context_start = max(0, start - window)
        context_end = min(len(content), end + window)
        return content[context_start:context_end]
    
    def _analyze_process_relationship(self, process1: Dict, process2: Dict, content: str) -> Optional[WorkflowDependency]:
        """Analyze relationship between two processes."""
        
        # Get combined context
        combined_context = f"{process1['context']} {process2['context']}"
        
        # Check for dependency patterns
        for dep_type, patterns in self.detector.workflow_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_context, re.IGNORECASE):
                    # Calculate confidence based on pattern strength and context
                    confidence = self._calculate_dependency_confidence(
                        process1, process2, dep_type, combined_context
                    )
                    
                    if confidence > 0.3:
                        return WorkflowDependency(
                            source_process=process1['text'][:100],  # Truncate for storage
                            target_process=process2['text'][:100],
                            dependency_type=dep_type,
                            strength=confidence,
                            conditions=self._extract_conditions(combined_context),
                            confidence=confidence
                        )
        
        return None
    
    def _calculate_dependency_confidence(self, process1: Dict, process2: Dict, 
                                       dep_type: str, context: str) -> float:
        """Calculate confidence score for a dependency relationship."""
        confidence = 0.0
        
        # Base confidence from pattern match
        confidence += 0.4
        
        # Boost confidence if processes are mentioned close together
        distance = abs(process1['start'] - process2['start'])
        if distance < 500:  # Within 500 characters
            confidence += 0.2
        
        # Boost confidence for specific dependency types
        if dep_type == 'prerequisite' and 'before' in context.lower():
            confidence += 0.2
        elif dep_type == 'conditional' and any(word in context.lower() for word in ['if', 'when', 'unless']):
            confidence += 0.2
        
        # Reduce confidence for very generic processes
        generic_terms = ['process', 'procedure', 'step']
        if any(term in process1['text'].lower() for term in generic_terms):
            confidence -= 0.1
        if any(term in process2['text'].lower() for term in generic_terms):
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _extract_conditions(self, context: str) -> Dict[str, Any]:
        """Extract conditions from context."""
        conditions = {}
        
        # Look for conditional phrases
        condition_patterns = [
            r'if\s+([^,\.]+)',
            r'when\s+([^,\.]+)',
            r'unless\s+([^,\.]+)',
            r'provided\s+that\s+([^,\.]+)'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            if matches:
                conditions['conditional_phrases'] = matches
        
        return conditions
    
    def _build_dependency_graph(self, dependencies: List[WorkflowDependency]):
        """Build a graph representation of dependencies for validation."""
        self.dependency_graph.clear()
        
        for dep in dependencies:
            self.dependency_graph.add_edge(
                dep.source_process,
                dep.target_process,
                type=dep.dependency_type,
                strength=dep.strength,
                confidence=dep.confidence
            )

class DecisionTreeExtractor:
    """Extracts implicit decision-making patterns from content."""
    
    def __init__(self, detector: TacitKnowledgeDetector):
        self.detector = detector
    
    def extract_decision_patterns(self, content: str, entities: List[Dict]) -> List[DecisionPattern]:
        """
        Extract implicit decision-making patterns from content.
        
        Args:
            content: Document text content
            entities: Extracted entities from NER
            
        Returns:
            List of DecisionPattern objects
        """
        patterns = []
        
        try:
            # Find decision points in the content
            decision_points = self._find_decision_points(content)
            
            # Analyze each decision point
            for point in decision_points:
                pattern = self._analyze_decision_point(point, content)
                if pattern and pattern.confidence > 0.3:
                    patterns.append(pattern)
            
            logger.info(f"Extracted {len(patterns)} decision patterns")
            
        except Exception as e:
            logger.error(f"Error extracting decision patterns: {e}")
        
        return patterns
    
    def _find_decision_points(self, content: str) -> List[Dict[str, Any]]:
        """Find potential decision points in content."""
        decision_points = []
        
        # Decision indicators
        decision_indicators = [
            r'\b(?:decide|decision|choose|select|determine|evaluate)\b',
            r'\b(?:if|when|whether|should|must|can)\b',
            r'\b(?:option|alternative|choice|possibility)\b',
            r'\b(?:approve|reject|accept|deny)\b'
        ]
        
        for indicator in decision_indicators:
            matches = re.finditer(indicator, content, re.IGNORECASE)
            for match in matches:
                context = self._get_context(content, match.start(), match.end(), 300)
                decision_points.append({
                    'indicator': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'context': context
                })
        
        return decision_points
    
    def _get_context(self, content: str, start: int, end: int, window: int = 200) -> str:
        """Get context around a text span."""
        context_start = max(0, start - window)
        context_end = min(len(content), end + window)
        return content[context_start:context_end]
    
    def _analyze_decision_point(self, point: Dict, content: str) -> Optional[DecisionPattern]:
        """Analyze a specific decision point."""
        
        context = point['context']
        
        # Determine decision type
        decision_type = self._classify_decision_type(context)
        
        # Extract conditions and outcomes
        conditions = self._extract_decision_conditions(context)
        outcomes = self._extract_decision_outcomes(context)
        
        # Calculate confidence
        confidence = self._calculate_decision_confidence(context, conditions, outcomes)
        
        if confidence > 0.3:
            return DecisionPattern(
                decision_point=point['indicator'],
                decision_type=decision_type,
                conditions=conditions,
                outcomes=outcomes,
                confidence=confidence,
                context=context[:200]  # Truncate for storage
            )
        
        return None
    
    def _classify_decision_type(self, context: str) -> str:
        """Classify the type of decision based on context."""
        
        for decision_type, patterns in self.detector.decision_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    return decision_type
        
        return 'general'
    
    def _extract_decision_conditions(self, context: str) -> Dict[str, Any]:
        """Extract conditions that influence the decision."""
        conditions = {}
        
        # Look for conditional phrases
        condition_patterns = [
            r'if\s+([^,\.]+)',
            r'when\s+([^,\.]+)',
            r'based\s+on\s+([^,\.]+)',
            r'depending\s+on\s+([^,\.]+)'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            if matches:
                conditions['conditions'] = matches
        
        # Look for criteria
        criteria_patterns = [
            r'criteria\s*:?\s*([^\.]+)',
            r'requirements\s*:?\s*([^\.]+)',
            r'standards\s*:?\s*([^\.]+)'
        ]
        
        for pattern in criteria_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            if matches:
                conditions['criteria'] = matches
        
        return conditions
    
    def _extract_decision_outcomes(self, context: str) -> List[Dict[str, Any]]:
        """Extract possible outcomes of the decision."""
        outcomes = []
        
        # Look for outcome indicators
        outcome_patterns = [
            r'then\s+([^,\.]+)',
            r'result\s+in\s+([^,\.]+)',
            r'leads\s+to\s+([^,\.]+)',
            r'approve\s*:?\s*([^,\.]+)',
            r'reject\s*:?\s*([^,\.]+)'
        ]
        
        for pattern in outcome_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            for match in matches:
                outcomes.append({
                    'description': match.strip(),
                    'type': 'consequence'
                })
        
        return outcomes
    
    def _calculate_decision_confidence(self, context: str, conditions: Dict, outcomes: List) -> float:
        """Calculate confidence score for a decision pattern."""
        confidence = 0.0
        
        # Base confidence
        confidence += 0.3
        
        # Boost for explicit conditions
        if conditions:
            confidence += 0.2
        
        # Boost for explicit outcomes
        if outcomes:
            confidence += 0.2
        
        # Boost for decision keywords
        decision_keywords = ['decide', 'choose', 'select', 'determine', 'evaluate']
        if any(keyword in context.lower() for keyword in decision_keywords):
            confidence += 0.2
        
        # Boost for structured decision language
        if re.search(r'\b(?:if.*then|when.*do)\b', context, re.IGNORECASE):
            confidence += 0.1
        
        return min(1.0, confidence)

class ResourceOptimizationDetector:
    """Detects resource optimization patterns and efficiency opportunities."""
    
    def __init__(self, detector: TacitKnowledgeDetector):
        self.detector = detector
    
    def detect_optimization_patterns(self, content: str, entities: List[Dict]) -> List[OptimizationPattern]:
        """
        Detect resource optimization patterns and efficiency opportunities.
        
        Args:
            content: Document text content
            entities: Extracted entities from NER
            
        Returns:
            List of OptimizationPattern objects
        """
        patterns = []
        
        try:
            # Analyze content for optimization opportunities
            for pattern_type, pattern_regexes in self.detector.optimization_patterns.items():
                type_patterns = self._find_optimization_patterns_by_type(
                    content, pattern_type, pattern_regexes
                )
                patterns.extend(type_patterns)
            
            # Analyze sentiment for implicit optimization opportunities
            sentiment_patterns = self._analyze_sentiment_for_optimization(content)
            patterns.extend(sentiment_patterns)
            
            logger.info(f"Detected {len(patterns)} optimization patterns")
            
        except Exception as e:
            logger.error(f"Error detecting optimization patterns: {e}")
        
        return patterns
    
    def _find_optimization_patterns_by_type(self, content: str, pattern_type: str, 
                                          pattern_regexes: List[str]) -> List[OptimizationPattern]:
        """Find optimization patterns of a specific type."""
        patterns = []
        
        for regex in pattern_regexes:
            matches = re.finditer(regex, content, re.IGNORECASE)
            
            for match in matches:
                context = self._get_context(content, match.start(), match.end(), 300)
                
                pattern = self._analyze_optimization_opportunity(
                    pattern_type, match.group(0), context
                )
                
                if pattern and pattern.confidence > 0.3:
                    patterns.append(pattern)
        
        return patterns
    
    def _get_context(self, content: str, start: int, end: int, window: int = 200) -> str:
        """Get context around a text span."""
        context_start = max(0, start - window)
        context_end = min(len(content), end + window)
        return content[context_start:context_end]
    
    def _analyze_optimization_opportunity(self, pattern_type: str, trigger_text: str, 
                                        context: str) -> Optional[OptimizationPattern]:
        """Analyze a specific optimization opportunity."""
        
        # Extract improvement opportunities
        improvements = self._extract_improvements(context, pattern_type)
        
        # Extract success metrics
        metrics = self._extract_success_metrics(context, pattern_type)
        
        # Extract conditions
        conditions = self._extract_optimization_conditions(context)
        
        # Calculate confidence and impact
        confidence = self._calculate_optimization_confidence(context, pattern_type)
        impact_level = self._assess_impact_level(context, pattern_type)
        
        if confidence > 0.3:
            return OptimizationPattern(
                pattern_type=pattern_type,
                description=f"Optimization opportunity: {trigger_text}",
                conditions=conditions,
                improvements=improvements,
                success_metrics=metrics,
                confidence=confidence,
                impact_level=impact_level
            )
        
        return None
    
    def _extract_improvements(self, context: str, pattern_type: str) -> Dict[str, Any]:
        """Extract potential improvements from context."""
        improvements = {}
        
        improvement_patterns = {
            'resource': [r'reduce\s+([^,\.]+)', r'save\s+([^,\.]+)', r'optimize\s+([^,\.]+)'],
            'time': [r'faster\s+([^,\.]+)', r'reduce\s+time\s+([^,\.]+)', r'accelerate\s+([^,\.]+)'],
            'quality': [r'improve\s+([^,\.]+)', r'enhance\s+([^,\.]+)', r'better\s+([^,\.]+)'],
            'cost': [r'reduce\s+cost\s+([^,\.]+)', r'save\s+money\s+([^,\.]+)', r'cheaper\s+([^,\.]+)'],
            'safety': [r'safer\s+([^,\.]+)', r'reduce\s+risk\s+([^,\.]+)', r'prevent\s+([^,\.]+)']
        }
        
        if pattern_type in improvement_patterns:
            for pattern in improvement_patterns[pattern_type]:
                matches = re.findall(pattern, context, re.IGNORECASE)
                if matches:
                    improvements['potential_improvements'] = matches
        
        return improvements
    
    def _extract_success_metrics(self, context: str, pattern_type: str) -> List[str]:
        """Extract success metrics from context."""
        metrics = []
        
        # Look for quantitative metrics
        metric_patterns = [
            r'(\d+%)\s+(?:improvement|reduction|increase)',
            r'(\d+)\s+(?:minutes|hours|days)\s+(?:saved|reduced)',
            r'\$(\d+(?:,\d+)*)\s+(?:saved|reduced)',
            r'(\d+)\s+(?:defects|errors|incidents)\s+(?:reduced|prevented)'
        ]
        
        for pattern in metric_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            metrics.extend(matches)
        
        # Add pattern-specific metrics
        pattern_metrics = {
            'resource': ['resource_utilization', 'waste_reduction'],
            'time': ['cycle_time', 'throughput', 'lead_time'],
            'quality': ['defect_rate', 'customer_satisfaction', 'rework_rate'],
            'cost': ['cost_per_unit', 'total_cost_savings', 'roi'],
            'safety': ['incident_rate', 'near_miss_frequency', 'safety_score']
        }
        
        if pattern_type in pattern_metrics:
            metrics.extend(pattern_metrics[pattern_type])
        
        return metrics
    
    def _extract_optimization_conditions(self, context: str) -> Dict[str, Any]:
        """Extract conditions for optimization."""
        conditions = {}
        
        # Look for conditional phrases
        condition_patterns = [
            r'if\s+([^,\.]+)',
            r'when\s+([^,\.]+)',
            r'requires\s+([^,\.]+)',
            r'needs\s+([^,\.]+)'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            if matches:
                conditions['requirements'] = matches
        
        return conditions
    
    def _calculate_optimization_confidence(self, context: str, pattern_type: str) -> float:
        """Calculate confidence score for optimization pattern."""
        confidence = 0.0
        
        # Base confidence
        confidence += 0.4
        
        # Boost for specific optimization keywords
        optimization_keywords = ['optimize', 'improve', 'enhance', 'reduce', 'increase', 'streamline']
        keyword_count = sum(1 for keyword in optimization_keywords if keyword in context.lower())
        confidence += min(0.3, keyword_count * 0.1)
        
        # Boost for quantitative mentions
        if re.search(r'\d+%|\$\d+|\d+\s+(?:minutes|hours|days)', context):
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _assess_impact_level(self, context: str, pattern_type: str) -> str:
        """Assess the potential impact level of the optimization."""
        
        # High impact indicators
        high_impact_keywords = ['critical', 'significant', 'major', 'substantial', 'dramatic']
        if any(keyword in context.lower() for keyword in high_impact_keywords):
            return 'high'
        
        # Low impact indicators
        low_impact_keywords = ['minor', 'small', 'slight', 'marginal']
        if any(keyword in context.lower() for keyword in low_impact_keywords):
            return 'low'
        
        return 'medium'
    
    def _analyze_sentiment_for_optimization(self, content: str) -> List[OptimizationPattern]:
        """Analyze sentiment to identify implicit optimization opportunities."""
        patterns = []
        
        if not self.detector.sentiment_analyzer:
            return patterns
        
        try:
            # Split content into sentences for sentiment analysis
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                if len(sentence.strip()) < 20:  # Skip very short sentences
                    continue
                
                # Analyze sentiment
                result = self.detector.sentiment_analyzer(sentence.strip())
                
                # Look for negative sentiment that might indicate optimization opportunities
                if result[0]['label'] == 'NEGATIVE' and result[0]['score'] > 0.7:
                    pattern = self._create_sentiment_optimization_pattern(sentence, result[0]['score'])
                    if pattern:
                        patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Error in sentiment analysis for optimization: {e}")
        
        return patterns
    
    def _create_sentiment_optimization_pattern(self, sentence: str, sentiment_score: float) -> Optional[OptimizationPattern]:
        """Create optimization pattern from negative sentiment."""
        
        # Look for optimization-related keywords in negative sentences
        optimization_keywords = ['slow', 'inefficient', 'waste', 'problem', 'issue', 'difficult', 'expensive']
        
        if any(keyword in sentence.lower() for keyword in optimization_keywords):
            return OptimizationPattern(
                pattern_type='general',
                description=f"Potential optimization opportunity identified: {sentence[:100]}",
                conditions={'sentiment_trigger': sentence},
                improvements={'sentiment_based': 'Address negative sentiment indicators'},
                success_metrics=['sentiment_improvement', 'user_satisfaction'],
                confidence=min(0.8, sentiment_score),
                impact_level='medium'
            )
        
        return None

class CommunicationProtocolMapper:
    """Maps communication protocols and information flows."""
    
    def __init__(self, detector: TacitKnowledgeDetector):
        self.detector = detector
    
    def map_communication_flows(self, content: str, entities: List[Dict]) -> List[CommunicationFlow]:
        """
        Map communication protocols and information flows.
        
        Args:
            content: Document text content
            entities: Extracted entities from NER
            
        Returns:
            List of CommunicationFlow objects
        """
        flows = []
        
        try:
            # Extract roles and communication patterns
            roles = self._extract_roles(content, entities)
            communications = self._find_communication_patterns(content)
            
            # Map flows between roles
            for comm in communications:
                flow = self._analyze_communication_flow(comm, roles, content)
                if flow and flow.confidence > 0.3:
                    flows.append(flow)
            
            logger.info(f"Mapped {len(flows)} communication flows")
            
        except Exception as e:
            logger.error(f"Error mapping communication flows: {e}")
        
        return flows
    
    def _extract_roles(self, content: str, entities: List[Dict]) -> List[str]:
        """Extract organizational roles from content."""
        roles = set()
        
        # Common organizational roles
        role_keywords = [
            'manager', 'supervisor', 'operator', 'technician', 'engineer',
            'coordinator', 'specialist', 'analyst', 'director', 'lead',
            'administrator', 'inspector', 'auditor', 'trainer'
        ]
        
        # Find roles in content
        for keyword in role_keywords:
            pattern = rf'\b{keyword}\b'
            if re.search(pattern, content, re.IGNORECASE):
                roles.add(keyword.title())
        
        # Extract roles from entities
        for entity in entities:
            if entity.get('entity_group') == 'PER':
                # Look for role context around person names
                context = self._get_context(content, entity.get('start', 0), entity.get('end', 0))
                for keyword in role_keywords:
                    if keyword in context.lower():
                        roles.add(keyword.title())
        
        return list(roles)
    
    def _get_context(self, content: str, start: int, end: int, window: int = 100) -> str:
        """Get context around a text span."""
        context_start = max(0, start - window)
        context_end = min(len(content), end + window)
        return content[context_start:context_end]
    
    def _find_communication_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Find communication patterns in content."""
        communications = []
        
        # Communication verbs and patterns
        comm_patterns = [
            r'\b(?:inform|notify|report|communicate|discuss|meet|call)\b',
            r'\b(?:send|receive|transmit|share|distribute)\b',
            r'\b(?:document|record|log|write|email)\b',
            r'\b(?:coordinate|collaborate|consult|brief)\b'
        ]
        
        for pattern in comm_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                context = self._get_context(content, match.start(), match.end(), 200)
                communications.append({
                    'verb': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'context': context
                })
        
        return communications
    
    def _analyze_communication_flow(self, comm: Dict, roles: List[str], content: str) -> Optional[CommunicationFlow]:
        """Analyze a specific communication flow."""
        
        context = comm['context']
        
        # Try to identify source and target roles
        source_role, target_role = self._identify_communication_roles(context, roles)
        
        if not source_role or not target_role:
            return None
        
        # Determine communication method
        method = self._determine_communication_method(context)
        
        # Determine information type
        info_type = self._determine_information_type(context)
        
        # Determine frequency
        frequency = self._determine_frequency(context)
        
        # Determine criticality
        criticality = self._determine_criticality(context)
        
        # Check if formal protocol
        formal_protocol = self._is_formal_protocol(context)
        
        # Calculate confidence
        confidence = self._calculate_communication_confidence(context, source_role, target_role)
        
        if confidence > 0.3:
            return CommunicationFlow(
                source_role=source_role,
                target_role=target_role,
                information_type=info_type,
                communication_method=method,
                frequency=frequency,
                criticality=criticality,
                formal_protocol=formal_protocol,
                confidence=confidence
            )
        
        return None
    
    def _identify_communication_roles(self, context: str, roles: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Identify source and target roles in communication."""
        
        found_roles = []
        for role in roles:
            if role.lower() in context.lower():
                found_roles.append(role)
        
        # Simple heuristic: first role is source, second is target
        if len(found_roles) >= 2:
            return found_roles[0], found_roles[1]
        elif len(found_roles) == 1:
            # Try to infer the other role from context
            if 'to' in context.lower():
                return found_roles[0], 'Unknown'
            else:
                return 'Unknown', found_roles[0]
        
        return None, None
    
    def _determine_communication_method(self, context: str) -> str:
        """Determine the communication method."""
        
        for method, patterns in self.detector.communication_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    return method
        
        return 'unknown'
    
    def _determine_information_type(self, context: str) -> str:
        """Determine the type of information being communicated."""
        
        info_types = {
            'status_update': ['status', 'progress', 'update', 'report'],
            'instruction': ['instruction', 'direction', 'order', 'command'],
            'request': ['request', 'ask', 'need', 'require'],
            'notification': ['notify', 'inform', 'alert', 'announce'],
            'data': ['data', 'information', 'results', 'findings'],
            'approval': ['approve', 'authorize', 'confirm', 'sign-off']
        }
        
        for info_type, keywords in info_types.items():
            if any(keyword in context.lower() for keyword in keywords):
                return info_type
        
        return 'general'
    
    def _determine_frequency(self, context: str) -> str:
        """Determine communication frequency."""
        
        frequency_patterns = {
            'continuous': ['continuous', 'ongoing', 'constant'],
            'daily': ['daily', 'every day', 'each day'],
            'weekly': ['weekly', 'every week', 'each week'],
            'monthly': ['monthly', 'every month', 'each month'],
            'as_needed': ['as needed', 'when required', 'if necessary']
        }
        
        for frequency, patterns in frequency_patterns.items():
            if any(pattern in context.lower() for pattern in patterns):
                return frequency
        
        return 'as_needed'
    
    def _determine_criticality(self, context: str) -> str:
        """Determine communication criticality."""
        
        critical_keywords = ['critical', 'urgent', 'emergency', 'immediate']
        high_keywords = ['important', 'priority', 'significant']
        low_keywords = ['routine', 'standard', 'regular']
        
        if any(keyword in context.lower() for keyword in critical_keywords):
            return 'critical'
        elif any(keyword in context.lower() for keyword in high_keywords):
            return 'high'
        elif any(keyword in context.lower() for keyword in low_keywords):
            return 'low'
        
        return 'medium'
    
    def _is_formal_protocol(self, context: str) -> bool:
        """Determine if communication follows formal protocol."""
        
        formal_indicators = ['protocol', 'procedure', 'standard', 'policy', 'regulation', 'requirement']
        return any(indicator in context.lower() for indicator in formal_indicators)
    
    def _calculate_communication_confidence(self, context: str, source_role: str, target_role: str) -> float:
        """Calculate confidence score for communication flow."""
        confidence = 0.0
        
        # Base confidence
        confidence += 0.4
        
        # Boost for clear role identification
        if source_role != 'Unknown' and target_role != 'Unknown':
            confidence += 0.3
        elif source_role != 'Unknown' or target_role != 'Unknown':
            confidence += 0.1
        
        # Boost for communication verbs
        comm_verbs = ['inform', 'notify', 'report', 'communicate', 'discuss']
        if any(verb in context.lower() for verb in comm_verbs):
            confidence += 0.2
        
        # Boost for directional indicators
        if any(indicator in context.lower() for indicator in ['to', 'from', 'between']):
            confidence += 0.1
        
        return min(1.0, confidence)

class KnowledgeExtractionEngine:
    """Main engine for extracting tacit knowledge from enterprise documents."""
    
    def __init__(self):
        self.detector = TacitKnowledgeDetector()
        self.workflow_analyzer = WorkflowDependencyAnalyzer(self.detector)
        self.decision_extractor = DecisionTreeExtractor(self.detector)
        self.optimization_detector = ResourceOptimizationDetector(self.detector)
        self.communication_mapper = CommunicationProtocolMapper(self.detector)
    
    def extract_tacit_knowledge(self, content: str, entities: List[Dict] = None) -> Dict[str, Any]:
        """
        Extract all types of tacit knowledge from content.
        
        Args:
            content: Document text content
            entities: Optional list of extracted entities
            
        Returns:
            Dictionary containing all extracted knowledge types
        """
        if entities is None:
            entities = []
        
        logger.info("Starting tacit knowledge extraction")
        
        from datetime import datetime
        
        results = {
            'workflow_dependencies': [],
            'decision_patterns': [],
            'optimization_patterns': [],
            'communication_flows': [],
            'extraction_metadata': {
                'content_length': len(content),
                'entity_count': len(entities),
                'extraction_timestamp': datetime.utcnow().isoformat()
            }
        }
        
        try:
            # Extract workflow dependencies
            results['workflow_dependencies'] = self.workflow_analyzer.analyze_workflow_dependencies(content, entities)
            
            # Extract decision patterns
            results['decision_patterns'] = self.decision_extractor.extract_decision_patterns(content, entities)
            
            # Detect optimization patterns
            results['optimization_patterns'] = self.optimization_detector.detect_optimization_patterns(content, entities)
            
            # Map communication flows
            results['communication_flows'] = self.communication_mapper.map_communication_flows(content, entities)
            
            # Update metadata
            results['extraction_metadata'].update({
                'workflow_dependencies_count': len(results['workflow_dependencies']),
                'decision_patterns_count': len(results['decision_patterns']),
                'optimization_patterns_count': len(results['optimization_patterns']),
                'communication_flows_count': len(results['communication_flows'])
            })
            
            logger.info(f"Tacit knowledge extraction completed: {results['extraction_metadata']}")
            
        except Exception as e:
            logger.error(f"Error in tacit knowledge extraction: {e}")
            results['extraction_metadata']['error'] = str(e)
        
        return results

# Global instance for use in other modules
knowledge_extraction_engine = KnowledgeExtractionEngine()