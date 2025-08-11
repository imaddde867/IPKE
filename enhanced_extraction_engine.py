#!/usr/bin/env python3
"""
Enhanced Knowledge Extraction Engine
Significantly improved extraction capabilities for better knowledge quality
"""

import re
import spacy
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class ExtractedEntity:
    """Enhanced entity with comprehensive information"""
    content: str
    entity_type: str
    category: str
    confidence: float
    context: str
    metadata: Dict[str, Any]
    relationships: List[str]
    source_location: str

class EnhancedExtractionEngine:
    """Enhanced extraction engine with comprehensive pattern recognition and LLM integration"""
    
    def __init__(self, llm_model=None):
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Some features may be limited.")
            self.nlp = None
        
        # LLM model for intelligent extraction
        self.llm_model = llm_model
        self.llm_available = llm_model is not None
    
    def extract_comprehensive_knowledge(self, content: str, document_type: str = "unknown") -> List[ExtractedEntity]:
        """Extract comprehensive knowledge from document content"""
        entities = []
        
        # Clean and prepare content
        content = self._clean_content(content)
        
        # Multi-pattern extraction
        entities.extend(self._extract_technical_specifications(content))
        entities.extend(self._extract_procedures_and_processes(content))
        entities.extend(self._extract_safety_requirements(content))
        entities.extend(self._extract_personnel_and_roles(content))
        entities.extend(self._extract_equipment_information(content))
        entities.extend(self._extract_maintenance_schedules(content))
        entities.extend(self._extract_regulatory_compliance(content))
        entities.extend(self._extract_quantitative_data(content))
        entities.extend(self._extract_definitions_and_terms(content))
        entities.extend(self._extract_warnings_and_cautions(content))
        
        # Apply NLP enhancement if available
        if self.nlp:
            entities = self._enhance_with_nlp(entities, content)
        
        # Apply LLM enhancement for deeper understanding
        if self.llm_available:
            entities = self._enhance_with_llm(entities, content, document_type)
        
        # Filter and score entities
        entities = self._filter_and_score_entities(entities)
        
        return entities
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content for better extraction"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        # Remove special characters that interfere with extraction
        content = re.sub(r'[^\w\s\-\.\,\:\;\(\)\[\]\/\\\'"°%$#@&\+\=\|\<\>\?]', '', content)
        return content.strip()
    
    def _extract_technical_specifications(self, content: str) -> List[ExtractedEntity]:
        """Extract technical specifications and parameters"""
        entities = []
        
        # Equipment specifications patterns
        spec_patterns = [
            # Motor specifications
            r'(\d+\.?\d*)\s*(HP|hp|horsepower)\s*(.*?)(?:motor|Motor)',
            # Voltage specifications  
            r'(\d+\.?\d*)\s*(V|volt|volts|voltage)\s*[,\s]*(\d*\.?\d*\s*phase|\d*-phase|3-phase|single-phase)?',
            # Pressure specifications
            r'(\d+\.?\d*)\s*(PSI|psi|bar|Pa|pascal|kPa)\s*(max|maximum|min|minimum|operating)?',
            # Temperature specifications
            r'(\d+\.?\d*)\s*°?\s*(F|C|fahrenheit|celsius)\s*(to|-)?\s*(\d+\.?\d*)\s*°?\s*(F|C)?',
            # Flow rates
            r'(\d+\.?\d*)\s*(GPM|gpm|LPM|lpm|cfm|CFM)\s*(flow|rate)?',
            # Capacity specifications
            r'(\d+\.?\d*)\s*(gallon|liter|cubic|ft³|m³|tons?)\s*(capacity|volume)?'
        ]
        
        for pattern in spec_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end].strip()
                
                entities.append(ExtractedEntity(
                    content=match.group(0).strip(),
                    entity_type="specification",
                    category="technical_specifications",
                    confidence=0.85,
                    context=context,
                    metadata={
                        "specification_type": "technical_parameter",
                        "full_match": match.group(0),
                        "groups": match.groups()
                    },
                    relationships=[],
                    source_location=f"chars_{match.start()}_{match.end()}"
                ))
        
        return entities
    
    def _extract_procedures_and_processes(self, content: str) -> List[ExtractedEntity]:
        """Extract detailed procedures and step-by-step processes"""
        entities = []
        
        # Procedure patterns
        procedure_patterns = [
            # Step-by-step procedures
            r'(?:step\s*\d+|procedure|process|method):\s*([^\.!?]+[\.!?])',
            # Daily/Weekly/Monthly procedures
            r'(daily|weekly|monthly|annually|quarterly):\s*([^\.!?]+[\.!?])',
            # Maintenance procedures
            r'(check|inspect|replace|lubricate|clean|test|verify|ensure|maintain)\s+([^\.!?]+[\.!?])',
            # Safety procedures
            r'(always|never|must|shall|should|required)\s+([^\.!?]+[\.!?])',
            # Sequential actions
            r'(first|next|then|finally|last|before|after)\s+([^\.!?]+[\.!?])'
        ]
        
        for pattern in procedure_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                procedure_text = match.group(0).strip()
                if len(procedure_text) > 30:  # Filter meaningful procedures
                    
                    # Determine procedure type
                    proc_type = self._classify_procedure_type(procedure_text)
                    
                    entities.append(ExtractedEntity(
                        content=procedure_text,
                        entity_type="procedure",
                        category="process_intelligence", 
                        confidence=0.75,
                        context=self._get_surrounding_context(content, match),
                        metadata={
                            "procedure_type": proc_type,
                            "action_words": self._extract_action_words(procedure_text),
                            "contains_sequence": "step" in procedure_text.lower() or any(word in procedure_text.lower() for word in ["first", "next", "then", "finally"])
                        },
                        relationships=[],
                        source_location=f"chars_{match.start()}_{match.end()}"
                    ))
        
        return entities
    
    def _extract_safety_requirements(self, content: str) -> List[ExtractedEntity]:
        """Extract safety requirements and protocols"""
        entities = []
        
        safety_patterns = [
            # Safety equipment requirements
            r'(?:wear|use|require|need)\s+([^\.!?]*(?:goggles|gloves|helmet|shoes|equipment|protection|PPE)[^\.!?]*[\.!?])',
            # Emergency procedures
            r'(?:emergency|shutdown|alarm|evacuation)\s+([^\.!?]+[\.!?])',
            # Safety warnings
            r'(?:danger|warning|caution|hazard|risk)\s*:?\s*([^\.!?]+[\.!?])',
            # Prohibited actions
            r'(?:do not|never|prohibited|forbidden|avoid)\s+([^\.!?]+[\.!?])',
            # Safety requirements
            r'(?:must|shall|required|mandatory)\s+([^\.!?]*(?:safety|secure|protection)[^\.!?]*[\.!?])'
        ]
        
        for pattern in safety_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                safety_text = match.group(0).strip()
                
                # Classify safety level
                safety_level = self._classify_safety_level(safety_text)
                
                entities.append(ExtractedEntity(
                    content=safety_text,
                    entity_type="safety_requirement",
                    category="risk_mitigation_intelligence",
                    confidence=0.80,
                    context=self._get_surrounding_context(content, match),
                    metadata={
                        "safety_level": safety_level,
                        "requirement_type": "mandatory" if any(word in safety_text.lower() for word in ["must", "shall", "required"]) else "recommended",
                        "emergency_related": "emergency" in safety_text.lower()
                    },
                    relationships=[],
                    source_location=f"chars_{match.start()}_{match.end()}"
                ))
        
        return entities
    
    def _extract_personnel_and_roles(self, content: str) -> List[ExtractedEntity]:
        """Extract personnel information and role definitions"""
        entities = []
        
        # Personnel patterns
        personnel_patterns = [
            # Name with title/certification
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*-\s*([^,\n]+(?:engineer|manager|supervisor|director|technician|specialist|coordinator|analyst|certified)[^,\n]*)',
            # Role definitions
            r'(chief|senior|lead|head|principal|assistant)\s+(engineer|manager|supervisor|director|technician|specialist|coordinator|analyst)',
            # Certifications
            r'([A-Z]+\s*\d*\s*certified|PE\s+certified|OSHA\s+\d+|certification|licensed)'
        ]
        
        for pattern in personnel_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                personnel_text = match.group(0).strip()
                
                entities.append(ExtractedEntity(
                    content=personnel_text,
                    entity_type="personnel",
                    category="organizational_intelligence",
                    confidence=0.70,
                    context=self._get_surrounding_context(content, match),
                    metadata={
                        "contains_name": bool(re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', personnel_text)),
                        "contains_title": bool(re.search(r'(engineer|manager|supervisor|director|technician)', personnel_text, re.IGNORECASE)),
                        "contains_certification": bool(re.search(r'(certified|PE|OSHA|license)', personnel_text, re.IGNORECASE))
                    },
                    relationships=[],
                    source_location=f"chars_{match.start()}_{match.end()}"
                ))
        
        return entities
    
    def _extract_equipment_information(self, content: str) -> List[ExtractedEntity]:
        """Extract equipment and asset information"""
        entities = []
        
        equipment_patterns = [
            # Equipment with specifications
            r'(motor|pump|valve|sensor|controller|compressor|heater|cooler|tank|vessel)\s*:?\s*([^\.!?\n]+[\.!?])',
            # Equipment models/types
            r'(model|type|series)\s*:?\s*([A-Z0-9\-]+)',
            # Equipment conditions
            r'(operating|maximum|minimum|normal|optimal)\s+(pressure|temperature|speed|voltage|current|flow)\s*:?\s*([^\.!?\n]+)'
        ]
        
        for pattern in equipment_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                equipment_text = match.group(0).strip()
                
                entities.append(ExtractedEntity(
                    content=equipment_text,
                    entity_type="equipment",
                    category="asset_intelligence",
                    confidence=0.75,
                    context=self._get_surrounding_context(content, match),
                    metadata={
                        "equipment_type": match.group(1).lower() if match.groups() else "unknown",
                        "has_specifications": bool(re.search(r'\d+', equipment_text)),
                        "condition_related": any(word in equipment_text.lower() for word in ["operating", "maximum", "minimum"])
                    },
                    relationships=[],
                    source_location=f"chars_{match.start()}_{match.end()}"
                ))
        
        return entities
    
    def _extract_maintenance_schedules(self, content: str) -> List[ExtractedEntity]:
        """Extract maintenance schedules and frequencies"""
        entities = []
        
        maintenance_patterns = [
            # Scheduled maintenance
            r'(daily|weekly|monthly|quarterly|annually|yearly)\s*:?\s*([^\.!?\n]*(?:check|inspect|replace|clean|lubricate|test|maintain)[^\.!?\n]*[\.!?])',
            # Maintenance intervals
            r'(?:every|each)\s+(\d+)\s+(day|week|month|year|hour|cycle)s?\s*[,:]?\s*([^\.!?\n]+[\.!?])',
            # Maintenance actions
            r'(replace|change|inspect|check|clean|lubricate|adjust|calibrate|test)\s+([^\.!?\n]+(?:filter|oil|belt|bearing|connection|level)[^\.!?\n]*[\.!?])'
        ]
        
        for pattern in maintenance_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                maintenance_text = match.group(0).strip()
                
                # Extract frequency if present
                frequency = self._extract_frequency(maintenance_text)
                
                entities.append(ExtractedEntity(
                    content=maintenance_text,
                    entity_type="maintenance",
                    category="maintenance_intelligence",
                    confidence=0.80,
                    context=self._get_surrounding_context(content, match),
                    metadata={
                        "frequency": frequency,
                        "action_type": self._extract_maintenance_action(maintenance_text),
                        "is_scheduled": bool(re.search(r'(daily|weekly|monthly|quarterly|annually)', maintenance_text, re.IGNORECASE))
                    },
                    relationships=[],
                    source_location=f"chars_{match.start()}_{match.end()}"
                ))
        
        return entities
    
    def _extract_regulatory_compliance(self, content: str) -> List[ExtractedEntity]:
        """Extract regulatory and compliance information"""
        entities = []
        
        compliance_patterns = [
            # Standards and regulations
            r'(OSHA|ANSI|ISO|ASTM|NFPA|EPA|FDA|HACCP|GMP)\s*[0-9\-]*\s*[:]?\s*([^\.!?\n]+[\.!?])',
            # Compliance requirements
            r'(?:comply|compliance|accordance|conform)\s+(?:with|to)\s+([^\.!?\n]+[\.!?])',
            # Regulatory actions
            r'(?:must|shall|required|mandatory)\s+([^\.!?\n]*(?:comply|meet|follow|adhere)[^\.!?\n]*[\.!?])',
            # Certification requirements
            r'(?:certified|certification|licensed|approved)\s+(?:by|for|under)\s+([^\.!?\n]+[\.!?])'
        ]
        
        for pattern in compliance_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                compliance_text = match.group(0).strip()
                
                entities.append(ExtractedEntity(
                    content=compliance_text,
                    entity_type="compliance",
                    category="compliance_governance",
                    confidence=0.75,
                    context=self._get_surrounding_context(content, match),
                    metadata={
                        "regulation_type": self._identify_regulation_type(compliance_text),
                        "mandatory": any(word in compliance_text.lower() for word in ["must", "shall", "required", "mandatory"]),
                        "certification_related": "certif" in compliance_text.lower()
                    },
                    relationships=[],
                    source_location=f"chars_{match.start()}_{match.end()}"
                ))
        
        return entities
    
    def _extract_quantitative_data(self, content: str) -> List[ExtractedEntity]:
        """Extract quantitative data and measurements"""
        entities = []
        
        quantitative_patterns = [
            # Measurements with units
            r'(\d+\.?\d*)\s*(mm|cm|m|km|in|ft|yd|mil|micron|nm|kg|g|lb|oz|ton|gallon|liter|ml|°|degrees?|rpm|hz|khz|mhz|watts?|amps?|volts?)',
            # Ranges and limits
            r'(?:between|from)\s+(\d+\.?\d*)\s*(?:to|and|-)\s*(\d+\.?\d*)\s*(\w+)',
            # Percentages
            r'(\d+\.?\d*)\s*%\s*([^\.!?\n]*)',
            # Ratios and rates
            r'(\d+\.?\d*)\s*(?::|per|/)\s*(\d+\.?\d*)\s*(\w+)?'
        ]
        
        for pattern in quantitative_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                quant_text = match.group(0).strip()
                
                entities.append(ExtractedEntity(
                    content=quant_text,
                    entity_type="measurement",
                    category="quantitative_intelligence", 
                    confidence=0.85,
                    context=self._get_surrounding_context(content, match),
                    metadata={
                        "value": self._extract_numeric_value(quant_text),
                        "unit": self._extract_unit(quant_text),
                        "is_range": "to" in quant_text or "between" in quant_text,
                        "measurement_type": self._classify_measurement_type(quant_text)
                    },
                    relationships=[],
                    source_location=f"chars_{match.start()}_{match.end()}"
                ))
        
        return entities
    
    def _extract_definitions_and_terms(self, content: str) -> List[ExtractedEntity]:
        """Extract definitions and technical terms"""
        entities = []
        
        definition_patterns = [
            # Explicit definitions
            r'([A-Z][A-Za-z\s]+)\s*(?:is|means|refers to|defined as)\s+([^\.!?\n]+[\.!?])',
            # Acronym definitions
            r'([A-Z]{2,})\s*(?:\(|\-)\s*([^)]+)(?:\)|$)',
            # Technical terms with descriptions
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[:\-]\s*([^\.!?\n]+[\.!?])'
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                def_text = match.group(0).strip()
                
                entities.append(ExtractedEntity(
                    content=def_text,
                    entity_type="definition",
                    category="knowledge_definitions",
                    confidence=0.70,
                    context=self._get_surrounding_context(content, match),
                    metadata={
                        "term": match.group(1) if match.groups() else "unknown",
                        "definition": match.group(2) if len(match.groups()) > 1 else "unknown",
                        "is_acronym": bool(re.search(r'^[A-Z]{2,}', match.group(1))) if match.groups() else False
                    },
                    relationships=[],
                    source_location=f"chars_{match.start()}_{match.end()}"
                ))
        
        return entities
    
    def _extract_warnings_and_cautions(self, content: str) -> List[ExtractedEntity]:
        """Extract warnings, cautions, and alerts"""
        entities = []
        
        warning_patterns = [
            # Warning labels
            r'(?:WARNING|DANGER|CAUTION|ALERT|NOTICE)\s*:?\s*([^\.!?\n]+[\.!?])',
            # Risk statements
            r'(?:risk|hazard|danger)\s+(?:of|from)\s+([^\.!?\n]+[\.!?])',
            # Conditional warnings
            r'(?:if|when|unless)\s+([^,]+),?\s*([^\.!?\n]*(?:danger|risk|hazard|warning|caution)[^\.!?\n]*[\.!?])'
        ]
        
        for pattern in warning_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                warning_text = match.group(0).strip()
                
                # Classify warning severity
                severity = self._classify_warning_severity(warning_text)
                
                entities.append(ExtractedEntity(
                    content=warning_text,
                    entity_type="warning",
                    category="risk_mitigation_intelligence",
                    confidence=0.80,
                    context=self._get_surrounding_context(content, match),
                    metadata={
                        "severity": severity,
                        "conditional": "if" in warning_text.lower() or "when" in warning_text.lower(),
                        "warning_type": self._classify_warning_type(warning_text)
                    },
                    relationships=[],
                    source_location=f"chars_{match.start()}_{match.end()}"
                ))
        
        return entities
    
    # Helper methods
    def _get_surrounding_context(self, content: str, match, context_size: int = 100) -> str:
        """Get surrounding context for a match"""
        start = max(0, match.start() - context_size)
        end = min(len(content), match.end() + context_size)
        return content[start:end].strip()
    
    def _classify_procedure_type(self, text: str) -> str:
        """Classify the type of procedure"""
        text_lower = text.lower()
        if any(word in text_lower for word in ["daily", "weekly", "monthly"]):
            return "maintenance"
        elif any(word in text_lower for word in ["emergency", "shutdown", "alarm"]):
            return "emergency"
        elif any(word in text_lower for word in ["safety", "protective", "wear"]):
            return "safety"
        elif any(word in text_lower for word in ["check", "inspect", "test", "verify"]):
            return "inspection"
        else:
            return "operational"
    
    def _classify_safety_level(self, text: str) -> str:
        """Classify safety requirement level"""
        text_lower = text.lower()
        if "danger" in text_lower:
            return "critical"
        elif "warning" in text_lower:
            return "high"
        elif "caution" in text_lower:
            return "medium"
        else:
            return "standard"
    
    def _extract_action_words(self, text: str) -> List[str]:
        """Extract action words from text"""
        action_words = []
        actions = ["check", "inspect", "replace", "clean", "lubricate", "test", "verify", "ensure", "maintain", "wear", "use", "follow"]
        for action in actions:
            if action in text.lower():
                action_words.append(action)
        return action_words
    
    def _extract_frequency(self, text: str) -> str:
        """Extract frequency information"""
        text_lower = text.lower()
        frequencies = ["daily", "weekly", "monthly", "quarterly", "annually", "hourly"]
        for freq in frequencies:
            if freq in text_lower:
                return freq
        return "unspecified"
    
    def _extract_maintenance_action(self, text: str) -> str:
        """Extract maintenance action type"""
        text_lower = text.lower()
        if "replace" in text_lower or "change" in text_lower:
            return "replacement"
        elif "inspect" in text_lower or "check" in text_lower:
            return "inspection"
        elif "clean" in text_lower:
            return "cleaning"
        elif "lubricate" in text_lower:
            return "lubrication"
        elif "adjust" in text_lower or "calibrate" in text_lower:
            return "calibration"
        else:
            return "general"
    
    def _identify_regulation_type(self, text: str) -> str:
        """Identify regulation/standard type"""
        text_upper = text.upper()
        if "OSHA" in text_upper:
            return "occupational_safety"
        elif "ISO" in text_upper:
            return "international_standard"
        elif "ANSI" in text_upper:
            return "american_standard"
        elif "EPA" in text_upper:
            return "environmental"
        elif "FDA" in text_upper:
            return "food_drug"
        elif "HACCP" in text_upper or "GMP" in text_upper:
            return "food_safety"
        else:
            return "general_compliance"
    
    def _extract_numeric_value(self, text: str) -> float:
        """Extract numeric value from text"""
        match = re.search(r'(\d+\.?\d*)', text)
        return float(match.group(1)) if match else 0.0
    
    def _extract_unit(self, text: str) -> str:
        """Extract unit from text"""
        units = ["mm", "cm", "m", "km", "in", "ft", "yd", "kg", "g", "lb", "oz", "ton", "gallon", "liter", "ml", "°", "degrees", "rpm", "hz", "khz", "mhz", "watts", "amps", "volts", "psi", "bar", "pa", "kpa"]
        for unit in units:
            if unit.lower() in text.lower():
                return unit
        return "unknown"
    
    def _classify_measurement_type(self, text: str) -> str:
        """Classify type of measurement"""
        text_lower = text.lower()
        if any(unit in text_lower for unit in ["mm", "cm", "m", "km", "in", "ft", "yd"]):
            return "length"
        elif any(unit in text_lower for unit in ["kg", "g", "lb", "oz", "ton"]):
            return "weight"
        elif any(unit in text_lower for unit in ["gallon", "liter", "ml"]):
            return "volume"
        elif any(unit in text_lower for unit in ["°", "degrees"]):
            return "temperature"
        elif any(unit in text_lower for unit in ["psi", "bar", "pa", "kpa"]):
            return "pressure"
        elif any(unit in text_lower for unit in ["rpm", "hz", "khz", "mhz"]):
            return "frequency"
        elif any(unit in text_lower for unit in ["watts", "amps", "volts"]):
            return "electrical"
        else:
            return "general"
    
    def _classify_warning_severity(self, text: str) -> str:
        """Classify warning severity"""
        text_upper = text.upper()
        if "DANGER" in text_upper:
            return "danger"
        elif "WARNING" in text_upper:
            return "warning"
        elif "CAUTION" in text_upper:
            return "caution"
        else:
            return "notice"
    
    def _classify_warning_type(self, text: str) -> str:
        """Classify warning type"""
        text_lower = text.lower()
        if any(word in text_lower for word in ["electrical", "shock", "voltage"]):
            return "electrical"
        elif any(word in text_lower for word in ["chemical", "toxic", "poison"]):
            return "chemical"
        elif any(word in text_lower for word in ["mechanical", "crushing", "pinch"]):
            return "mechanical"
        elif any(word in text_lower for word in ["fire", "explosion", "flammable"]):
            return "fire"
        else:
            return "general"
    
    def _enhance_with_nlp(self, entities: List[ExtractedEntity], content: str) -> List[ExtractedEntity]:
        """Enhance entities using NLP processing"""
        if not self.nlp:
            return entities
        
        # Process content with spaCy
        doc = self.nlp(content)
        
        # Extract additional entities from NLP
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "PRODUCT", "QUANTITY", "CARDINAL"]:
                entities.append(ExtractedEntity(
                    content=ent.text,
                    entity_type=ent.label_.lower(),
                    category="nlp_enhanced",
                    confidence=0.65,
                    context=ent.sent.text if ent.sent else "",
                    metadata={
                        "nlp_label": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char
                    },
                    relationships=[],
                    source_location=f"chars_{ent.start_char}_{ent.end_char}"
                ))
        
        return entities
    
    def _enhance_with_llm(self, entities: List[ExtractedEntity], content: str, document_type: str) -> List[ExtractedEntity]:
        """Enhance entities using LLM for deeper understanding and context"""
        if not self.llm_available:
            return entities
        
        enhanced_entities = []
        
        # LLM-powered content analysis
        llm_entities = self._llm_extract_knowledge(content, document_type)
        enhanced_entities.extend(llm_entities)
        
        # Enhance existing entities with LLM insights
        for entity in entities:
            enhanced_entity = self._llm_enhance_entity(entity, content)
            enhanced_entities.append(enhanced_entity)
        
        return enhanced_entities
    
    def _llm_extract_knowledge(self, content: str, document_type: str) -> List[ExtractedEntity]:
        """Use LLM to extract high-level knowledge and insights"""
        llm_entities = []
        
        # Create focused prompts for different types of knowledge extraction
        prompts = self._create_llm_extraction_prompts(content, document_type)
        
        for prompt_type, prompt in prompts.items():
            try:
                # Query the LLM
                response = self._query_llm(prompt)
                
                # Parse LLM response into entities
                parsed_entities = self._parse_llm_response(response, prompt_type, content)
                llm_entities.extend(parsed_entities)
                
            except Exception as e:
                print(f"LLM extraction failed for {prompt_type}: {e}")
                continue
        
        return llm_entities
    
    def _create_llm_extraction_prompts(self, content: str, document_type: str) -> Dict[str, str]:
        """Create targeted prompts for LLM-based knowledge extraction"""
        # Truncate content if too long
        max_content_length = 2000
        truncated_content = content[:max_content_length] + "..." if len(content) > max_content_length else content
        
        prompts = {
            "key_processes": f"""
            Analyze this {document_type} document and extract the key processes, procedures, and workflows described.
            For each process, identify:
            1. The main steps or phases
            2. Required inputs or prerequisites  
            3. Expected outputs or results
            4. Responsible parties or roles
            5. Critical decision points
            
            Document content:
            {truncated_content}
            
            Format your response as a structured list with clear process names and details.
            """,
            
            "safety_requirements": f"""
            Analyze this {document_type} document and extract all safety requirements, hazards, and risk mitigation measures.
            For each safety item, identify:
            1. The specific requirement or hazard
            2. Severity level (critical, high, medium, low)
            3. Required protective equipment or measures
            4. Consequences of non-compliance
            5. Applicable scenarios or conditions
            
            Document content:
            {truncated_content}
            
            Format your response as a structured list with clear safety items and details.
            """,
            
            "technical_specifications": f"""
            Analyze this {document_type} document and extract all technical specifications, parameters, and measurements.
            For each specification, identify:
            1. The component or system being specified
            2. Numerical values with units
            3. Operating ranges or limits
            4. Performance criteria
            5. Testing or verification methods
            
            Document content:
            {truncated_content}
            
            Format your response as a structured list with clear specifications and values.
            """,
            
            "compliance_requirements": f"""
            Analyze this {document_type} document and extract all compliance requirements, standards, and regulatory information.
            For each requirement, identify:
            1. The specific standard or regulation
            2. Mandatory vs. recommended requirements
            3. Compliance verification methods
            4. Documentation requirements
            5. Responsible parties
            
            Document content:
            {truncated_content}
            
            Format your response as a structured list with clear compliance items.
            """,
            
            "organizational_info": f"""
            Analyze this {document_type} document and extract organizational information including roles, responsibilities, and personnel.
            For each organizational element, identify:
            1. Specific roles or positions
            2. Responsibilities and authorities
            3. Required qualifications or certifications
            4. Reporting relationships
            5. Contact information if available
            
            Document content:
            {truncated_content}
            
            Format your response as a structured list with clear organizational details.
            """
        }
        
        return prompts
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM model with a prompt"""
        if not self.llm_available:
            return ""
        
        try:
            # Query using llama-cpp-python interface
            response = self.llm_model(
                prompt,
                max_tokens=1000,
                temperature=0.3,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["\n\n", "Document content:", "---"]
            )
            
            return response["choices"][0]["text"].strip()
            
        except Exception as e:
            print(f"LLM query failed: {e}")
            return ""
    
    def _parse_llm_response(self, response: str, prompt_type: str, content: str) -> List[ExtractedEntity]:
        """Parse LLM response into structured entities"""
        entities = []
        
        if not response:
            return entities
        
        # Split response into individual items
        lines = response.split('\n')
        current_item = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for list markers or structured patterns
            if re.match(r'^\d+\.|\-|\*', line):
                # Process previous item if exists
                if current_item:
                    entity = self._create_llm_entity(current_item, prompt_type, content)
                    if entity:
                        entities.append(entity)
                
                # Start new item
                current_item = line
            else:
                # Continue current item
                current_item += " " + line
        
        # Process final item
        if current_item:
            entity = self._create_llm_entity(current_item, prompt_type, content)
            if entity:
                entities.append(entity)
        
        return entities
    
    def _create_llm_entity(self, item_text: str, prompt_type: str, content: str) -> Optional[ExtractedEntity]:
        """Create an entity from LLM extracted text"""
        if len(item_text) < 20:  # Skip very short items
            return None
        
        # Map prompt types to categories
        category_mapping = {
            "key_processes": "process_intelligence",
            "safety_requirements": "risk_mitigation_intelligence", 
            "technical_specifications": "technical_specifications",
            "compliance_requirements": "compliance_governance",
            "organizational_info": "organizational_intelligence"
        }
        
        # Determine entity type based on content
        entity_type = self._determine_llm_entity_type(item_text, prompt_type)
        
        # Extract confidence based on LLM response quality
        confidence = self._calculate_llm_confidence(item_text)
        
        return ExtractedEntity(
            content=item_text.strip(),
            entity_type=entity_type,
            category=category_mapping.get(prompt_type, "llm_extracted"),
            confidence=confidence,
            context=self._find_context_in_content(item_text, content),
            metadata={
                "extraction_method": "llm_analysis",
                "prompt_type": prompt_type,
                "llm_generated": True,
                "quality_indicators": self._analyze_llm_item_quality(item_text)
            },
            relationships=[],
            source_location="llm_extracted"
        )
    
    def _determine_llm_entity_type(self, text: str, prompt_type: str) -> str:
        """Determine entity type from LLM extracted text"""
        text_lower = text.lower()
        
        if prompt_type == "key_processes":
            if any(word in text_lower for word in ["step", "procedure", "process", "workflow"]):
                return "process"
            elif any(word in text_lower for word in ["decision", "choice", "option"]):
                return "decision_point"
            else:
                return "procedure"
        
        elif prompt_type == "safety_requirements":
            if any(word in text_lower for word in ["wear", "use", "equipment"]):
                return "safety_equipment"
            elif any(word in text_lower for word in ["danger", "warning", "caution"]):
                return "hazard"
            else:
                return "safety_requirement"
        
        elif prompt_type == "technical_specifications":
            if re.search(r'\d+', text):
                return "specification"
            else:
                return "technical_parameter"
        
        elif prompt_type == "compliance_requirements":
            return "compliance_requirement"
        
        elif prompt_type == "organizational_info":
            if any(word in text_lower for word in ["role", "position", "title"]):
                return "role"
            else:
                return "personnel"
        
        return "extracted_knowledge"
    
    def _calculate_llm_confidence(self, text: str) -> float:
        """Calculate confidence score for LLM extracted content"""
        base_confidence = 0.75  # Base confidence for LLM extractions
        
        # Boost confidence for specific indicators
        if len(text) > 50:
            base_confidence += 0.05
        
        if re.search(r'\d+', text):  # Contains numbers
            base_confidence += 0.05
        
        if any(word in text.lower() for word in ["must", "shall", "required", "critical"]):
            base_confidence += 0.05
        
        if len(text.split()) > 10:  # Substantial content
            base_confidence += 0.05
        
        return min(0.90, base_confidence)
    
    def _find_context_in_content(self, item_text: str, content: str) -> str:
        """Find where this extracted item appears in the original content"""
        # Look for partial matches in content
        words = item_text.split()[:5]  # Use first 5 words
        search_text = " ".join(words)
        
        # Find approximate location
        for i in range(0, len(content) - len(search_text), 50):
            chunk = content[i:i+200]
            if any(word.lower() in chunk.lower() for word in words[:3]):
                return chunk.strip()
        
        return "Context not found in original document"
    
    def _analyze_llm_item_quality(self, text: str) -> Dict[str, Any]:
        """Analyze quality indicators of LLM extracted item"""
        return {
            "word_count": len(text.split()),
            "has_numbers": bool(re.search(r'\d+', text)),
            "has_action_words": any(word in text.lower() for word in ["check", "inspect", "ensure", "verify", "maintain"]),
            "has_technical_terms": any(word in text.lower() for word in ["system", "equipment", "component", "procedure"]),
            "specificity_score": len(re.findall(r'\b[A-Z][a-z]+\b', text)) / max(len(text.split()), 1)
        }
    
    def _llm_enhance_entity(self, entity: ExtractedEntity, content: str) -> ExtractedEntity:
        """Enhance an existing entity with LLM insights"""
        if not self.llm_available:
            return entity
        
        # Create enhancement prompt
        enhancement_prompt = f"""
        Analyze this extracted knowledge item and provide additional context, relationships, and insights:
        
        Extracted Item: {entity.content}
        Category: {entity.category}
        Entity Type: {entity.entity_type}
        
        From the document context, identify:
        1. Related procedures or processes
        2. Dependencies or prerequisites
        3. Potential risks or considerations
        4. Implementation details
        5. Quality or performance criteria
        
        Provide a concise analysis focusing on practical insights.
        """
        
        try:
            llm_enhancement = self._query_llm(enhancement_prompt)
            
            if llm_enhancement:
                # Update entity with LLM insights
                entity.metadata["llm_enhancement"] = llm_enhancement
                entity.metadata["enhanced_by_llm"] = True
                
                # Extract relationships from LLM response
                relationships = self._extract_relationships_from_llm(llm_enhancement)
                entity.relationships.extend(relationships)
                
                # Boost confidence for enhanced entities
                entity.confidence = min(0.95, entity.confidence + 0.05)
        
        except Exception as e:
            print(f"LLM enhancement failed: {e}")
        
        return entity
    
    def _extract_relationships_from_llm(self, llm_text: str) -> List[str]:
        """Extract relationships from LLM enhancement text"""
        relationships = []
        
        # Look for relationship indicators
        relationship_patterns = [
            r'related to ([^\.]+)',
            r'depends on ([^\.]+)',
            r'requires ([^\.]+)',
            r'connected to ([^\.]+)',
            r'part of ([^\.]+)'
        ]
        
        for pattern in relationship_patterns:
            matches = re.finditer(pattern, llm_text, re.IGNORECASE)
            for match in matches:
                relationship = match.group(1).strip()
                if len(relationship) > 5 and len(relationship) < 100:
                    relationships.append(relationship)
        
        return relationships[:5]  # Limit to 5 relationships
    
    def _filter_and_score_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Filter and improve scoring of entities"""
        filtered_entities = []
        
        for entity in entities:
            # Skip very short or meaningless content
            if len(entity.content) < 10:
                continue
            
            # Skip duplicates (simple check)
            if not any(e.content.lower() == entity.content.lower() for e in filtered_entities):
                # Adjust confidence based on content quality
                entity.confidence = self._calculate_enhanced_confidence(entity)
                filtered_entities.append(entity)
        
        # Sort by confidence
        filtered_entities.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered_entities
    
    def _calculate_enhanced_confidence(self, entity: ExtractedEntity) -> float:
        """Calculate enhanced confidence score"""
        base_confidence = entity.confidence
        
        # Boost confidence for specific patterns
        if entity.entity_type == "specification" and any(char.isdigit() for char in entity.content):
            base_confidence += 0.1
        
        if entity.category == "technical_specifications" and len(entity.content) > 30:
            base_confidence += 0.05
        
        if entity.entity_type == "procedure" and len(entity.metadata.get("action_words", [])) > 1:
            base_confidence += 0.05
        
        # Ensure confidence stays within bounds
        return min(0.95, max(0.1, base_confidence))

# Test the enhanced extraction
if __name__ == "__main__":
    engine = EnhancedExtractionEngine()
    
    # Test with sample content
    test_content = """
    INDUSTRIAL EQUIPMENT MANUAL
    
    SAFETY PROCEDURES
    Always wear safety goggles and protective gloves when operating equipment.
    Emergency shutdown procedures must be followed in case of malfunction.
    
    EQUIPMENT SPECIFICATIONS
    Primary Motor: 10 HP electric motor, 480V, 3-phase
    Cooling Pump: Centrifugal pump, 200 GPM flow rate
    Operating Pressure: 200 PSI maximum
    Temperature Range: 32°F to 180°F
    
    MAINTENANCE SCHEDULE
    Daily: Check fluid levels and pressure readings
    Weekly: Inspect electrical connections and belts
    Monthly: Replace air filters and lubricate bearings
    
    PERSONNEL
    Sarah Johnson - Chief Engineer, PE certified
    Mike Davis - Maintenance Supervisor, OSHA 30 certified
    """
    
    entities = engine.extract_comprehensive_knowledge(test_content)
    
    print(f"Extracted {len(entities)} entities:")
    for i, entity in enumerate(entities[:10]):  # Show first 10
        print(f"{i+1}. [{entity.category}] {entity.content[:80]}... (Confidence: {entity.confidence:.2f})")
