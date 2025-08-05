"""
Relationship Mapping System for EXPLAINIUM

This module implements algorithms to identify and map relationships between
different knowledge elements, including process dependencies, equipment-maintenance
correlations, skill-function linkages, and compliance-procedure connections.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import networkx as nx
from datetime import datetime

from src.logging_config import get_logger
from src.database.models import (
    KnowledgeItem, WorkflowDependency, Equipment, Procedure, Personnel,
    SafetyInformation, TechnicalSpecification
)

logger = get_logger(__name__)

@dataclass
class ProcessDependency:
    """Represents a dependency between processes."""
    source_process_id: str
    target_process_id: str
    dependency_type: str  # prerequisite, parallel, downstream, conditional
    strength: float  # 0.0-1.0
    conditions: Dict[str, Any]
    confidence: float
    evidence: List[str]

@dataclass
class EquipmentMaintenanceCorrelation:
    """Represents correlation between equipment and maintenance patterns."""
    equipment_id: int
    maintenance_pattern: str
    correlation_type: str  # preventive, corrective, predictive
    frequency: str
    conditions: Dict[str, Any]
    confidence: float
    evidence: List[str]

@dataclass
class SkillFunctionLink:
    """Represents link between skills and job functions."""
    skill_name: str
    function_name: str
    proficiency_level: str  # basic, intermediate, advanced, expert
    criticality: str  # critical, important, helpful
    confidence: float
    evidence: List[str]

@dataclass
class ComplianceProcedureConnection:
    """Represents connection between compliance requirements and procedures."""
    regulation_reference: str
    procedure_id: int
    compliance_type: str  # mandatory, recommended, best_practice
    risk_level: str  # high, medium, low
    confidence: float
    evidence: List[str]

class ProcessDependencyMapper:
    """Maps dependencies and connections between processes."""
    
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.process_patterns = self._initialize_process_patterns()
        self.dependency_indicators = self._initialize_dependency_indicators()
    
    def _initialize_process_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for identifying processes."""
        return {
            'process_identifiers': [
                r'\b(?:process|procedure|operation|task|step|activity|workflow|protocol|method|routine)\b',
                r'\b(?:phase|stage|cycle|sequence|series|chain)\b',
                r'\b(?:inspection|testing|verification|validation|review)\b',
                r'\b(?:startup|shutdown|maintenance|calibration|cleaning)\b'
            ],
            'process_boundaries': [
                r'\b(?:begins|starts|initiates|commences)\b',
                r'\b(?:ends|completes|finishes|concludes|terminates)\b',
                r'\b(?:input|output|prerequisite|requirement|deliverable)\b'
            ]
        }
    
    def _initialize_dependency_indicators(self) -> Dict[str, List[str]]:
        """Initialize indicators for different dependency types."""
        return {
            'prerequisite': [
                r'\b(?:before|prior to|prerequisite|must be completed before|requires)\b',
                r'\b(?:after completing|once|following|subsequent to)\b',
                r'\b(?:depends on|dependent on|contingent on|relies on)\b',
                r'\b(?:cannot proceed until|wait for|pending)\b'
            ],
            'parallel': [
                r'\b(?:simultaneously|concurrently|at the same time|in parallel)\b',
                r'\b(?:while|during|as)\b.*\b(?:process|procedure|operation)\b',
                r'\b(?:concurrent|simultaneous|parallel)\b.*\b(?:execution|processing)\b'
            ],
            'downstream': [
                r'\b(?:then|next|subsequently|following|after)\b',
                r'\b(?:leads to|results in|triggers|initiates|causes)\b',
                r'\b(?:followed by|succeeded by|continued with)\b'
            ],
            'conditional': [
                r'\b(?:if|when|unless|provided that|in case of)\b',
                r'\b(?:depending on|based on|according to|subject to)\b',
                r'\b(?:conditional|contingent|optional)\b'
            ]
        }
    
    def identify_process_dependencies(self, knowledge_items: List[KnowledgeItem], 
                                    procedures: List[Procedure]) -> List[ProcessDependency]:
        """
        Identify dependencies between processes.
        
        Args:
            knowledge_items: List of knowledge items from database
            procedures: List of procedures from database
            
        Returns:
            List of ProcessDependency objects
        """
        dependencies = []
        
        try:
            # Create process mapping
            process_map = self._create_process_map(knowledge_items, procedures)
            
            # Analyze pairwise relationships
            process_ids = list(process_map.keys())
            for i, source_id in enumerate(process_ids):
                for j, target_id in enumerate(process_ids):
                    if i != j:
                        dependency = self._analyze_process_pair(
                            source_id, target_id, process_map
                        )
                        if dependency and dependency.confidence > 0.3:
                            dependencies.append(dependency)
            
            # Build and validate dependency graph
            self._build_dependency_graph(dependencies)
            
            # Remove circular dependencies and validate
            dependencies = self._validate_dependencies(dependencies)
            
            logger.info(f"Identified {len(dependencies)} process dependencies")
            
        except Exception as e:
            logger.error(f"Error identifying process dependencies: {e}")
        
        return dependencies
    
    def _create_process_map(self, knowledge_items: List[KnowledgeItem], 
                          procedures: List[Procedure]) -> Dict[str, Dict[str, Any]]:
        """Create a comprehensive map of all processes."""
        process_map = {}
        
        # Add knowledge items
        for item in knowledge_items:
            process_map[item.process_id] = {
                'type': 'knowledge_item',
                'name': item.name,
                'description': item.description or '',
                'domain': item.domain,
                'hierarchy_level': item.hierarchy_level,
                'object': item
            }
        
        # Add procedures
        for procedure in procedures:
            proc_id = f"PROC_{procedure.id}"
            process_map[proc_id] = {
                'type': 'procedure',
                'name': procedure.title,
                'description': self._get_procedure_description(procedure),
                'domain': procedure.category or 'operational',
                'hierarchy_level': 3,  # Procedures are typically level 3
                'object': procedure
            }
        
        return process_map
    
    def _get_procedure_description(self, procedure: Procedure) -> str:
        """Get description from procedure steps."""
        if procedure.steps:
            step_descriptions = [step.description for step in procedure.steps[:3]]
            return " ".join(step_descriptions)[:500]
        return ""
    
    def _analyze_process_pair(self, source_id: str, target_id: str, 
                            process_map: Dict[str, Dict[str, Any]]) -> Optional[ProcessDependency]:
        """Analyze relationship between two processes."""
        
        source_process = process_map[source_id]
        target_process = process_map[target_id]
        
        # Combine text content for analysis
        source_text = f"{source_process['name']} {source_process['description']}"
        target_text = f"{target_process['name']} {target_process['description']}"
        combined_text = f"{source_text} {target_text}"
        
        # Check for dependency patterns
        for dep_type, patterns in self.dependency_indicators.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    # Calculate confidence and extract evidence
                    confidence, evidence = self._calculate_dependency_confidence(
                        source_process, target_process, dep_type, combined_text
                    )
                    
                    if confidence > 0.3:
                        return ProcessDependency(
                            source_process_id=source_id,
                            target_process_id=target_id,
                            dependency_type=dep_type,
                            strength=confidence,
                            conditions=self._extract_dependency_conditions(combined_text),
                            confidence=confidence,
                            evidence=evidence
                        )
        
        # Check for semantic relationships
        semantic_dependency = self._analyze_semantic_relationship(
            source_process, target_process
        )
        
        return semantic_dependency
    
    def _calculate_dependency_confidence(self, source_process: Dict, target_process: Dict,
                                       dep_type: str, combined_text: str) -> Tuple[float, List[str]]:
        """Calculate confidence score and extract evidence for dependency."""
        confidence = 0.0
        evidence = []
        
        # Base confidence from pattern match
        confidence += 0.4
        evidence.append(f"Pattern match for {dep_type} dependency")
        
        # Boost for domain similarity
        if source_process['domain'] == target_process['domain']:
            confidence += 0.1
            evidence.append("Same domain processes")
        
        # Boost for hierarchy level relationships
        source_level = source_process.get('hierarchy_level', 0)
        target_level = target_process.get('hierarchy_level', 0)
        
        if abs(source_level - target_level) == 1:
            confidence += 0.15
            evidence.append("Adjacent hierarchy levels")
        elif source_level == target_level:
            confidence += 0.1
            evidence.append("Same hierarchy level")
        
        # Boost for specific dependency types
        if dep_type == 'prerequisite':
            prerequisite_keywords = ['before', 'prior', 'prerequisite', 'requires']
            if any(keyword in combined_text.lower() for keyword in prerequisite_keywords):
                confidence += 0.2
                evidence.append("Strong prerequisite indicators")
        
        elif dep_type == 'conditional':
            conditional_keywords = ['if', 'when', 'unless', 'depending']
            if any(keyword in combined_text.lower() for keyword in conditional_keywords):
                confidence += 0.2
                evidence.append("Strong conditional indicators")
        
        # Reduce confidence for very generic processes
        generic_terms = ['process', 'procedure', 'step', 'task']
        if any(term in source_process['name'].lower() for term in generic_terms):
            confidence -= 0.05
        if any(term in target_process['name'].lower() for term in generic_terms):
            confidence -= 0.05
        
        return min(1.0, max(0.0, confidence)), evidence
    
    def _extract_dependency_conditions(self, text: str) -> Dict[str, Any]:
        """Extract conditions that govern the dependency."""
        conditions = {}
        
        # Extract conditional phrases
        condition_patterns = [
            r'if\s+([^,\.]+)',
            r'when\s+([^,\.]+)',
            r'unless\s+([^,\.]+)',
            r'provided\s+that\s+([^,\.]+)',
            r'subject\s+to\s+([^,\.]+)'
        ]
        
        conditional_phrases = []
        for pattern in condition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            conditional_phrases.extend(matches)
        
        if conditional_phrases:
            conditions['conditional_phrases'] = conditional_phrases
        
        # Extract timing conditions
        timing_patterns = [
            r'within\s+(\d+\s+(?:minutes|hours|days|weeks))',
            r'after\s+(\d+\s+(?:minutes|hours|days|weeks))',
            r'before\s+(\d+\s+(?:minutes|hours|days|weeks))'
        ]
        
        timing_conditions = []
        for pattern in timing_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            timing_conditions.extend(matches)
        
        if timing_conditions:
            conditions['timing'] = timing_conditions
        
        return conditions
    
    def _analyze_semantic_relationship(self, source_process: Dict, 
                                     target_process: Dict) -> Optional[ProcessDependency]:
        """Analyze semantic relationships between processes."""
        
        # Simple semantic analysis based on name similarity and common terms
        source_words = set(re.findall(r'\b\w+\b', source_process['name'].lower()))
        target_words = set(re.findall(r'\b\w+\b', target_process['name'].lower()))
        
        # Calculate word overlap
        common_words = source_words.intersection(target_words)
        if len(common_words) > 0:
            overlap_ratio = len(common_words) / max(len(source_words), len(target_words))
            
            if overlap_ratio > 0.3:  # Significant overlap
                confidence = min(0.6, overlap_ratio)
                
                return ProcessDependency(
                    source_process_id=source_process.get('process_id', ''),
                    target_process_id=target_process.get('process_id', ''),
                    dependency_type='related',
                    strength=confidence,
                    conditions={'semantic_similarity': overlap_ratio},
                    confidence=confidence,
                    evidence=[f"Semantic similarity: {overlap_ratio:.2f}"]
                )
        
        return None
    
    def _build_dependency_graph(self, dependencies: List[ProcessDependency]):
        """Build a graph representation of dependencies."""
        self.dependency_graph.clear()
        
        for dep in dependencies:
            self.dependency_graph.add_edge(
                dep.source_process_id,
                dep.target_process_id,
                type=dep.dependency_type,
                strength=dep.strength,
                confidence=dep.confidence,
                conditions=dep.conditions
            )
    
    def _validate_dependencies(self, dependencies: List[ProcessDependency]) -> List[ProcessDependency]:
        """Validate dependencies and remove circular references."""
        validated_dependencies = []
        
        for dep in dependencies:
            # Check for circular dependencies
            if not self._creates_cycle(dep):
                validated_dependencies.append(dep)
            else:
                logger.warning(f"Circular dependency detected: {dep.source_process_id} -> {dep.target_process_id}")
        
        return validated_dependencies
    
    def _creates_cycle(self, dependency: ProcessDependency) -> bool:
        """Check if adding this dependency would create a cycle."""
        try:
            # Temporarily add the edge
            self.dependency_graph.add_edge(
                dependency.source_process_id,
                dependency.target_process_id
            )
            
            # Check for cycles
            has_cycle = not nx.is_directed_acyclic_graph(self.dependency_graph)
            
            # Remove the temporary edge
            self.dependency_graph.remove_edge(
                dependency.source_process_id,
                dependency.target_process_id
            )
            
            return has_cycle
            
        except Exception:
            return False

class EquipmentMaintenanceCorrelator:
    """Correlates equipment with maintenance patterns."""
    
    def __init__(self):
        self.maintenance_patterns = self._initialize_maintenance_patterns()
    
    def _initialize_maintenance_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for maintenance correlation."""
        return {
            'preventive': [
                r'\b(?:scheduled|routine|regular|periodic|planned)\b.*\b(?:maintenance|service|inspection)\b',
                r'\b(?:preventive|preventative|proactive)\b.*\b(?:maintenance|care)\b',
                r'\b(?:daily|weekly|monthly|quarterly|annual)\b.*\b(?:check|inspection|service)\b'
            ],
            'corrective': [
                r'\b(?:repair|fix|replace|correct|restore)\b',
                r'\b(?:breakdown|failure|malfunction|defect)\b.*\b(?:maintenance|repair)\b',
                r'\b(?:emergency|urgent|immediate)\b.*\b(?:repair|service)\b'
            ],
            'predictive': [
                r'\b(?:condition|vibration|temperature|pressure)\b.*\b(?:monitoring|analysis)\b',
                r'\b(?:predictive|condition-based|data-driven)\b.*\b(?:maintenance|monitoring)\b',
                r'\b(?:sensor|measurement|diagnostic)\b.*\b(?:data|analysis)\b'
            ]
        }
    
    def correlate_equipment_maintenance(self, equipment_list: List[Equipment], 
                                      procedures: List[Procedure]) -> List[EquipmentMaintenanceCorrelation]:
        """
        Correlate equipment with maintenance patterns.
        
        Args:
            equipment_list: List of equipment from database
            procedures: List of procedures from database
            
        Returns:
            List of EquipmentMaintenanceCorrelation objects
        """
        correlations = []
        
        try:
            for equipment in equipment_list:
                equipment_correlations = self._analyze_equipment_maintenance(
                    equipment, procedures
                )
                correlations.extend(equipment_correlations)
            
            logger.info(f"Found {len(correlations)} equipment-maintenance correlations")
            
        except Exception as e:
            logger.error(f"Error correlating equipment maintenance: {e}")
        
        return correlations
    
    def _analyze_equipment_maintenance(self, equipment: Equipment, 
                                     procedures: List[Procedure]) -> List[EquipmentMaintenanceCorrelation]:
        """Analyze maintenance patterns for specific equipment."""
        correlations = []
        
        equipment_name = equipment.name.lower()
        equipment_type = (equipment.type or '').lower()
        
        for procedure in procedures:
            # Check if procedure is related to this equipment
            procedure_text = f"{procedure.title} {self._get_procedure_text(procedure)}".lower()
            
            # Look for equipment mentions
            if (equipment_name in procedure_text or 
                equipment_type in procedure_text or
                self._check_equipment_procedure_link(equipment, procedure)):
                
                # Determine maintenance pattern
                for pattern_type, patterns in self.maintenance_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, procedure_text, re.IGNORECASE):
                            correlation = self._create_maintenance_correlation(
                                equipment, procedure, pattern_type, procedure_text
                            )
                            if correlation and correlation.confidence > 0.3:
                                correlations.append(correlation)
                            break
        
        return correlations
    
    def _get_procedure_text(self, procedure: Procedure) -> str:
        """Get full text content from procedure."""
        if procedure.steps:
            return " ".join([step.description for step in procedure.steps])
        return ""
    
    def _check_equipment_procedure_link(self, equipment: Equipment, procedure: Procedure) -> bool:
        """Check if equipment and procedure are linked through relationships."""
        # Check if there's a direct relationship in the database
        for proc_equip in procedure.procedure_equipments:
            if proc_equip.equipment_id == equipment.id:
                return True
        return False
    
    def _create_maintenance_correlation(self, equipment: Equipment, procedure: Procedure,
                                      pattern_type: str, procedure_text: str) -> Optional[EquipmentMaintenanceCorrelation]:
        """Create a maintenance correlation object."""
        
        # Extract frequency information
        frequency = self._extract_maintenance_frequency(procedure_text)
        
        # Extract conditions
        conditions = self._extract_maintenance_conditions(procedure_text)
        
        # Calculate confidence
        confidence = self._calculate_maintenance_confidence(
            equipment, procedure, pattern_type, procedure_text
        )
        
        # Extract evidence
        evidence = self._extract_maintenance_evidence(procedure_text, pattern_type)
        
        if confidence > 0.3:
            return EquipmentMaintenanceCorrelation(
                equipment_id=equipment.id,
                maintenance_pattern=f"{pattern_type}_{procedure.id}",
                correlation_type=pattern_type,
                frequency=frequency,
                conditions=conditions,
                confidence=confidence,
                evidence=evidence
            )
        
        return None
    
    def _extract_maintenance_frequency(self, text: str) -> str:
        """Extract maintenance frequency from text."""
        frequency_patterns = [
            r'\b(?:daily|every day)\b',
            r'\b(?:weekly|every week)\b',
            r'\b(?:monthly|every month)\b',
            r'\b(?:quarterly|every quarter)\b',
            r'\b(?:annually|yearly|every year)\b',
            r'\bevery\s+(\d+)\s+(days?|weeks?|months?|years?)\b',
            r'\b(\d+)\s+times?\s+per\s+(day|week|month|year)\b'
        ]
        
        for pattern in frequency_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return 'as_needed'
    
    def _extract_maintenance_conditions(self, text: str) -> Dict[str, Any]:
        """Extract conditions for maintenance."""
        conditions = {}
        
        # Temperature conditions
        temp_pattern = r'(?:temperature|temp)\s*(?:above|below|exceeds|drops)\s*(\d+)'
        temp_matches = re.findall(temp_pattern, text, re.IGNORECASE)
        if temp_matches:
            conditions['temperature_thresholds'] = temp_matches
        
        # Operating hours conditions
        hours_pattern = r'(?:after|every)\s*(\d+)\s*(?:hours?|hrs?)\s*(?:of operation)?'
        hours_matches = re.findall(hours_pattern, text, re.IGNORECASE)
        if hours_matches:
            conditions['operating_hours'] = hours_matches
        
        # Condition-based triggers
        condition_patterns = [
            r'when\s+([^,\.]+)',
            r'if\s+([^,\.]+)',
            r'upon\s+([^,\.]+)'
        ]
        
        triggers = []
        for pattern in condition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            triggers.extend(matches)
        
        if triggers:
            conditions['triggers'] = triggers
        
        return conditions
    
    def _calculate_maintenance_confidence(self, equipment: Equipment, procedure: Procedure,
                                        pattern_type: str, text: str) -> float:
        """Calculate confidence for maintenance correlation."""
        confidence = 0.0
        
        # Base confidence
        confidence += 0.4
        
        # Boost for direct equipment mention
        if equipment.name.lower() in text.lower():
            confidence += 0.2
        
        # Boost for equipment type mention
        if equipment.type and equipment.type.lower() in text.lower():
            confidence += 0.15
        
        # Boost for specific maintenance keywords
        maintenance_keywords = ['maintenance', 'service', 'repair', 'inspection', 'check']
        keyword_count = sum(1 for keyword in maintenance_keywords if keyword in text.lower())
        confidence += min(0.2, keyword_count * 0.05)
        
        # Boost for pattern-specific indicators
        if pattern_type == 'preventive' and any(word in text.lower() for word in ['scheduled', 'routine', 'regular']):
            confidence += 0.1
        elif pattern_type == 'corrective' and any(word in text.lower() for word in ['repair', 'fix', 'breakdown']):
            confidence += 0.1
        elif pattern_type == 'predictive' and any(word in text.lower() for word in ['condition', 'monitoring', 'sensor']):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _extract_maintenance_evidence(self, text: str, pattern_type: str) -> List[str]:
        """Extract evidence for maintenance correlation."""
        evidence = []
        
        # Add pattern type as evidence
        evidence.append(f"Maintenance pattern: {pattern_type}")
        
        # Extract specific evidence phrases
        evidence_patterns = {
            'preventive': [r'scheduled.*maintenance', r'routine.*inspection', r'regular.*service'],
            'corrective': [r'repair.*procedure', r'fix.*problem', r'breakdown.*response'],
            'predictive': [r'condition.*monitoring', r'predictive.*analysis', r'sensor.*data']
        }
        
        if pattern_type in evidence_patterns:
            for pattern in evidence_patterns[pattern_type]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                evidence.extend(matches)
        
        return evidence[:5]  # Limit evidence to top 5 items

class SkillFunctionLinker:
    """Links skills to job functions and performance metrics."""
    
    def __init__(self):
        self.skill_patterns = self._initialize_skill_patterns()
        self.proficiency_indicators = self._initialize_proficiency_indicators()
    
    def _initialize_skill_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for skill identification."""
        return {
            'technical_skills': [
                r'\b(?:programming|coding|software|database|network|system)\b',
                r'\b(?:welding|machining|electrical|mechanical|hydraulic)\b',
                r'\b(?:CAD|CAM|CNC|PLC|SCADA|HMI)\b'
            ],
            'safety_skills': [
                r'\b(?:safety|hazard|risk|emergency|first aid|CPR)\b',
                r'\b(?:OSHA|lockout|tagout|confined space|fall protection)\b',
                r'\b(?:PPE|personal protective equipment|safety procedures)\b'
            ],
            'quality_skills': [
                r'\b(?:quality|inspection|testing|calibration|measurement)\b',
                r'\b(?:ISO|Six Sigma|lean|continuous improvement)\b',
                r'\b(?:SPC|statistical process control|quality assurance)\b'
            ],
            'leadership_skills': [
                r'\b(?:leadership|management|supervision|coordination)\b',
                r'\b(?:communication|teamwork|problem solving|decision making)\b',
                r'\b(?:training|mentoring|coaching|development)\b'
            ]
        }
    
    def _initialize_proficiency_indicators(self) -> Dict[str, List[str]]:
        """Initialize indicators for proficiency levels."""
        return {
            'expert': [
                r'\b(?:expert|master|advanced|senior|lead|specialist)\b',
                r'\b(?:certified|licensed|qualified|experienced)\b.*\b(?:\d+\+?\s*years?)\b'
            ],
            'advanced': [
                r'\b(?:advanced|proficient|skilled|competent)\b',
                r'\b(?:\d+\-\d+|\d+)\s*years?\s*(?:experience|exp)\b'
            ],
            'intermediate': [
                r'\b(?:intermediate|moderate|working knowledge|familiar)\b',
                r'\b(?:some experience|basic understanding)\b'
            ],
            'basic': [
                r'\b(?:basic|beginner|entry level|novice|learning)\b',
                r'\b(?:minimal experience|new to|training)\b'
            ]
        }
    
    def link_skills_functions(self, personnel_list: List[Personnel], 
                            procedures: List[Procedure]) -> List[SkillFunctionLink]:
        """
        Link skills to job functions and performance metrics.
        
        Args:
            personnel_list: List of personnel from database
            procedures: List of procedures from database
            
        Returns:
            List of SkillFunctionLink objects
        """
        links = []
        
        try:
            for person in personnel_list:
                person_links = self._analyze_person_skills(person, procedures)
                links.extend(person_links)
            
            logger.info(f"Created {len(links)} skill-function links")
            
        except Exception as e:
            logger.error(f"Error linking skills to functions: {e}")
        
        return links
    
    def _analyze_person_skills(self, person: Personnel, 
                             procedures: List[Procedure]) -> List[SkillFunctionLink]:
        """Analyze skills for a specific person."""
        links = []
        
        # Extract skills from person data
        person_text = f"{person.role} {person.responsibilities or ''}"
        if person.certifications:
            cert_text = " ".join(person.certifications)
            person_text += f" {cert_text}"
        
        # Find related procedures
        related_procedures = self._find_related_procedures(person, procedures)
        
        # Extract skills from person data and related procedures
        skills = self._extract_skills(person_text)
        
        for procedure in related_procedures:
            procedure_text = f"{procedure.title} {self._get_procedure_text(procedure)}"
            procedure_skills = self._extract_skills(procedure_text)
            
            # Create links between skills and functions
            for skill in skills + procedure_skills:
                link = self._create_skill_function_link(
                    skill, person, procedure, person_text + " " + procedure_text
                )
                if link and link.confidence > 0.3:
                    links.append(link)
        
        return links
    
    def _find_related_procedures(self, person: Personnel, 
                               procedures: List[Procedure]) -> List[Procedure]:
        """Find procedures related to a person."""
        related = []
        
        for procedure in procedures:
            # Check direct relationships
            for proc_person in procedure.procedure_personnels:
                if proc_person.personnel_id == person.id:
                    related.append(procedure)
                    break
            
            # Check role-based relationships
            if person.role and person.role.lower() in procedure.title.lower():
                related.append(procedure)
        
        return related
    
    def _get_procedure_text(self, procedure: Procedure) -> str:
        """Get full text content from procedure."""
        if procedure.steps:
            return " ".join([step.description for step in procedure.steps])
        return ""
    
    def _extract_skills(self, text: str) -> List[Dict[str, Any]]:
        """Extract skills from text."""
        skills = []
        
        for skill_category, patterns in self.skill_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    skill_name = match.group(0)
                    proficiency = self._determine_proficiency(text, match.start(), match.end())
                    
                    skills.append({
                        'name': skill_name,
                        'category': skill_category,
                        'proficiency': proficiency,
                        'context': self._get_context(text, match.start(), match.end())
                    })
        
        return skills
    
    def _determine_proficiency(self, text: str, start: int, end: int) -> str:
        """Determine proficiency level from context."""
        context = self._get_context(text, start, end, 100)
        
        for level, indicators in self.proficiency_indicators.items():
            for indicator in indicators:
                if re.search(indicator, context, re.IGNORECASE):
                    return level
        
        return 'intermediate'  # Default proficiency
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around a text span."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _create_skill_function_link(self, skill: Dict[str, Any], person: Personnel,
                                  procedure: Procedure, full_text: str) -> Optional[SkillFunctionLink]:
        """Create a skill-function link."""
        
        # Determine criticality
        criticality = self._assess_skill_criticality(skill, procedure, full_text)
        
        # Calculate confidence
        confidence = self._calculate_skill_confidence(skill, person, procedure, full_text)
        
        # Extract evidence
        evidence = self._extract_skill_evidence(skill, full_text)
        
        if confidence > 0.3:
            return SkillFunctionLink(
                skill_name=skill['name'],
                function_name=procedure.title,
                proficiency_level=skill['proficiency'],
                criticality=criticality,
                confidence=confidence,
                evidence=evidence
            )
        
        return None
    
    def _assess_skill_criticality(self, skill: Dict[str, Any], procedure: Procedure, text: str) -> str:
        """Assess criticality of skill for function."""
        
        # Critical indicators
        critical_keywords = ['critical', 'essential', 'required', 'must', 'mandatory']
        if any(keyword in text.lower() for keyword in critical_keywords):
            return 'critical'
        
        # Important indicators
        important_keywords = ['important', 'necessary', 'needed', 'should']
        if any(keyword in text.lower() for keyword in important_keywords):
            return 'important'
        
        # Safety skills are typically critical
        if skill['category'] == 'safety_skills':
            return 'critical'
        
        return 'helpful'
    
    def _calculate_skill_confidence(self, skill: Dict[str, Any], person: Personnel,
                                  procedure: Procedure, text: str) -> float:
        """Calculate confidence for skill-function link."""
        confidence = 0.0
        
        # Base confidence
        confidence += 0.4
        
        # Boost for direct skill mention
        if skill['name'].lower() in text.lower():
            confidence += 0.2
        
        # Boost for role alignment
        if person.role and any(word in person.role.lower() for word in skill['name'].lower().split()):
            confidence += 0.15
        
        # Boost for certification alignment
        if person.certifications:
            cert_text = " ".join(person.certifications).lower()
            if any(word in cert_text for word in skill['name'].lower().split()):
                confidence += 0.2
        
        # Boost for skill category relevance
        if skill['category'] == 'safety_skills' and 'safety' in procedure.title.lower():
            confidence += 0.1
        elif skill['category'] == 'quality_skills' and 'quality' in procedure.title.lower():
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _extract_skill_evidence(self, skill: Dict[str, Any], text: str) -> List[str]:
        """Extract evidence for skill-function link."""
        evidence = []
        
        # Add skill category as evidence
        evidence.append(f"Skill category: {skill['category']}")
        
        # Add proficiency level as evidence
        evidence.append(f"Proficiency: {skill['proficiency']}")
        
        # Extract context around skill mention
        if skill['name'].lower() in text.lower():
            evidence.append(f"Skill mentioned: {skill['name']}")
        
        return evidence[:3]  # Limit to top 3 evidence items

class ComplianceProcedureConnector:
    """Connects regulatory requirements to compliance procedures."""
    
    def __init__(self):
        self.regulation_patterns = self._initialize_regulation_patterns()
        self.compliance_indicators = self._initialize_compliance_indicators()
    
    def _initialize_regulation_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for regulation identification."""
        return {
            'osha': [
                r'\bOSHA\b.*\b(?:\d+\.\d+|\d+)\b',
                r'\b(?:29 CFR|Code of Federal Regulations)\b.*\b\d+\b',
                r'\b(?:occupational safety|workplace safety|industrial hygiene)\b'
            ],
            'epa': [
                r'\bEPA\b.*\b(?:\d+\.\d+|\d+)\b',
                r'\b(?:40 CFR|environmental protection)\b',
                r'\b(?:clean air act|clean water act|RCRA|CERCLA)\b'
            ],
            'iso': [
                r'\bISO\b.*\b\d+\b',
                r'\b(?:ISO 9001|ISO 14001|ISO 45001)\b',
                r'\b(?:quality management|environmental management)\b'
            ],
            'fda': [
                r'\bFDA\b.*\b(?:\d+\.\d+|\d+)\b',
                r'\b(?:21 CFR|food and drug)\b',
                r'\b(?:GMP|good manufacturing practice|HACCP)\b'
            ]
        }
    
    def _initialize_compliance_indicators(self) -> Dict[str, List[str]]:
        """Initialize indicators for compliance types."""
        return {
            'mandatory': [
                r'\b(?:required|mandatory|must|shall|obligatory)\b',
                r'\b(?:regulation|law|statute|code)\b',
                r'\b(?:compliance|conform|adhere)\b'
            ],
            'recommended': [
                r'\b(?:recommended|suggested|advised|should)\b',
                r'\b(?:best practice|guideline|standard)\b',
                r'\b(?:preferred|optimal|ideal)\b'
            ],
            'best_practice': [
                r'\b(?:best practice|industry standard|benchmark)\b',
                r'\b(?:excellence|optimization|improvement)\b',
                r'\b(?:leading practice|state of the art)\b'
            ]
        }
    
    def connect_compliance_procedures(self, procedures: List[Procedure], 
                                    safety_info: List[SafetyInformation]) -> List[ComplianceProcedureConnection]:
        """
        Connect regulatory requirements to compliance procedures.
        
        Args:
            procedures: List of procedures from database
            safety_info: List of safety information from database
            
        Returns:
            List of ComplianceProcedureConnection objects
        """
        connections = []
        
        try:
            for procedure in procedures:
                procedure_connections = self._analyze_procedure_compliance(
                    procedure, safety_info
                )
                connections.extend(procedure_connections)
            
            logger.info(f"Created {len(connections)} compliance-procedure connections")
            
        except Exception as e:
            logger.error(f"Error connecting compliance procedures: {e}")
        
        return connections
    
    def _analyze_procedure_compliance(self, procedure: Procedure, 
                                    safety_info: List[SafetyInformation]) -> List[ComplianceProcedureConnection]:
        """Analyze compliance requirements for a specific procedure."""
        connections = []
        
        procedure_text = f"{procedure.title} {self._get_procedure_text(procedure)}"
        
        # Find regulation references
        regulation_refs = self._extract_regulation_references(procedure_text)
        
        # Analyze each regulation reference
        for reg_ref in regulation_refs:
            connection = self._create_compliance_connection(
                reg_ref, procedure, procedure_text, safety_info
            )
            if connection and connection.confidence > 0.3:
                connections.append(connection)
        
        return connections
    
    def _get_procedure_text(self, procedure: Procedure) -> str:
        """Get full text content from procedure."""
        if procedure.steps:
            return " ".join([step.description for step in procedure.steps])
        return ""
    
    def _extract_regulation_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract regulation references from text."""
        references = []
        
        for reg_type, patterns in self.regulation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    references.append({
                        'type': reg_type,
                        'reference': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'context': self._get_context(text, match.start(), match.end())
                    })
        
        return references
    
    def _get_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Get context around a text span."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _create_compliance_connection(self, reg_ref: Dict[str, Any], procedure: Procedure,
                                    procedure_text: str, safety_info: List[SafetyInformation]) -> Optional[ComplianceProcedureConnection]:
        """Create a compliance-procedure connection."""
        
        # Determine compliance type
        compliance_type = self._determine_compliance_type(reg_ref['context'])
        
        # Assess risk level
        risk_level = self._assess_compliance_risk(reg_ref, procedure_text, safety_info)
        
        # Calculate confidence
        confidence = self._calculate_compliance_confidence(reg_ref, procedure, procedure_text)
        
        # Extract evidence
        evidence = self._extract_compliance_evidence(reg_ref, procedure_text)
        
        if confidence > 0.3:
            return ComplianceProcedureConnection(
                regulation_reference=reg_ref['reference'],
                procedure_id=procedure.id,
                compliance_type=compliance_type,
                risk_level=risk_level,
                confidence=confidence,
                evidence=evidence
            )
        
        return None
    
    def _determine_compliance_type(self, context: str) -> str:
        """Determine type of compliance requirement."""
        
        for comp_type, indicators in self.compliance_indicators.items():
            for indicator in indicators:
                if re.search(indicator, context, re.IGNORECASE):
                    return comp_type
        
        return 'recommended'  # Default compliance type
    
    def _assess_compliance_risk(self, reg_ref: Dict[str, Any], procedure_text: str,
                              safety_info: List[SafetyInformation]) -> str:
        """Assess risk level of compliance requirement."""
        
        # High risk indicators
        high_risk_keywords = ['critical', 'fatal', 'severe', 'major', 'emergency']
        if any(keyword in procedure_text.lower() for keyword in high_risk_keywords):
            return 'high'
        
        # Check safety information for risk indicators
        for safety in safety_info:
            if safety.severity and safety.severity.lower() in ['high', 'critical']:
                return 'high'
        
        # Medium risk indicators
        medium_risk_keywords = ['important', 'significant', 'moderate', 'injury']
        if any(keyword in procedure_text.lower() for keyword in medium_risk_keywords):
            return 'medium'
        
        return 'low'
    
    def _calculate_compliance_confidence(self, reg_ref: Dict[str, Any], procedure: Procedure,
                                       procedure_text: str) -> float:
        """Calculate confidence for compliance connection."""
        confidence = 0.0
        
        # Base confidence for regulation reference
        confidence += 0.5
        
        # Boost for specific regulation types
        if reg_ref['type'] in ['osha', 'epa', 'fda']:
            confidence += 0.2
        
        # Boost for compliance keywords
        compliance_keywords = ['compliance', 'regulation', 'requirement', 'standard']
        keyword_count = sum(1 for keyword in compliance_keywords if keyword in procedure_text.lower())
        confidence += min(0.2, keyword_count * 0.05)
        
        # Boost for safety-related procedures
        if 'safety' in procedure.title.lower() or procedure.category == 'safety':
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _extract_compliance_evidence(self, reg_ref: Dict[str, Any], text: str) -> List[str]:
        """Extract evidence for compliance connection."""
        evidence = []
        
        # Add regulation reference as evidence
        evidence.append(f"Regulation reference: {reg_ref['reference']}")
        
        # Add regulation type as evidence
        evidence.append(f"Regulation type: {reg_ref['type']}")
        
        # Extract compliance-related phrases
        compliance_phrases = re.findall(
            r'\b(?:compliance|conform|adhere|requirement|regulation)\b[^.]*',
            text, re.IGNORECASE
        )
        evidence.extend(compliance_phrases[:2])  # Add up to 2 phrases
        
        return evidence[:4]  # Limit to top 4 evidence items