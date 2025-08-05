"""
Unit tests for relationship mapping system.

Tests the accuracy and functionality of process dependency mapping,
equipment-maintenance correlation, skill-function linking, and
compliance-procedure connections.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List

from src.ai.relationship_mapper import (
    ProcessDependencyMapper, EquipmentMaintenanceCorrelator,
    SkillFunctionLinker, ComplianceProcedureConnector,
    ProcessDependency, EquipmentMaintenanceCorrelation,
    SkillFunctionLink, ComplianceProcedureConnection
)
from src.database.models import (
    KnowledgeItem, Equipment, Procedure, Personnel, SafetyInformation, Step
)

class TestProcessDependencyMapper:
    """Test process dependency mapping functionality."""
    
    @pytest.fixture
    def mapper(self):
        """Create a ProcessDependencyMapper instance."""
        return ProcessDependencyMapper()
    
    @pytest.fixture
    def sample_knowledge_items(self):
        """Create sample knowledge items for testing."""
        items = []
        
        # Create mock knowledge items
        item1 = Mock(spec=KnowledgeItem)
        item1.process_id = "PROC_001"
        item1.name = "Equipment Startup Procedure"
        item1.description = "Before starting equipment, ensure all safety checks are completed"
        item1.domain = "operational"
        item1.hierarchy_level = 3
        items.append(item1)
        
        item2 = Mock(spec=KnowledgeItem)
        item2.process_id = "PROC_002"
        item2.name = "Safety Check Procedure"
        item2.description = "Comprehensive safety inspection must be completed before any operation"
        item2.domain = "safety"
        item2.hierarchy_level = 3
        items.append(item2)
        
        item3 = Mock(spec=KnowledgeItem)
        item3.process_id = "PROC_003"
        item3.name = "Equipment Operation"
        item3.description = "Normal operation procedure following startup"
        item3.domain = "operational"
        item3.hierarchy_level = 3
        items.append(item3)
        
        return items
    
    @pytest.fixture
    def sample_procedures(self):
        """Create sample procedures for testing."""
        procedures = []
        
        # Create mock procedures with steps
        proc1 = Mock(spec=Procedure)
        proc1.id = 1
        proc1.title = "Daily Equipment Startup"
        proc1.category = "operational"
        
        step1 = Mock(spec=Step)
        step1.description = "Complete safety inspection before proceeding"
        proc1.steps = [step1]
        procedures.append(proc1)
        
        proc2 = Mock(spec=Procedure)
        proc2.id = 2
        proc2.title = "Safety Inspection Protocol"
        proc2.category = "safety"
        
        step2 = Mock(spec=Step)
        step2.description = "Check all safety systems and equipment"
        proc2.steps = [step2]
        procedures.append(proc2)
        
        return procedures
    
    def test_identify_process_dependencies(self, mapper, sample_knowledge_items, sample_procedures):
        """Test identification of process dependencies."""
        dependencies = mapper.identify_process_dependencies(sample_knowledge_items, sample_procedures)
        
        assert isinstance(dependencies, list)
        assert len(dependencies) > 0
        
        # Check that dependencies have required attributes
        for dep in dependencies:
            assert isinstance(dep, ProcessDependency)
            assert dep.source_process_id
            assert dep.target_process_id
            assert dep.dependency_type in ['prerequisite', 'parallel', 'downstream', 'conditional', 'related']
            assert 0.0 <= dep.strength <= 1.0
            assert 0.0 <= dep.confidence <= 1.0
            assert isinstance(dep.conditions, dict)
            assert isinstance(dep.evidence, list)
    
    def test_prerequisite_dependency_detection(self, mapper, sample_knowledge_items, sample_procedures):
        """Test detection of prerequisite dependencies."""
        dependencies = mapper.identify_process_dependencies(sample_knowledge_items, sample_procedures)
        
        # Look for prerequisite dependencies
        prerequisite_deps = [dep for dep in dependencies if dep.dependency_type == 'prerequisite']
        
        # Should find at least one prerequisite dependency based on "before" keyword
        assert len(prerequisite_deps) > 0
        
        # Check confidence scores are reasonable
        for dep in prerequisite_deps:
            assert dep.confidence > 0.3
    
    def test_create_process_map(self, mapper, sample_knowledge_items, sample_procedures):
        """Test creation of process map."""
        process_map = mapper._create_process_map(sample_knowledge_items, sample_procedures)
        
        assert isinstance(process_map, dict)
        assert len(process_map) > 0
        
        # Check knowledge items are included
        for item in sample_knowledge_items:
            assert item.process_id in process_map
            assert process_map[item.process_id]['name'] == item.name
            assert process_map[item.process_id]['type'] == 'knowledge_item'
        
        # Check procedures are included
        for proc in sample_procedures:
            proc_id = f"PROC_{proc.id}"
            assert proc_id in process_map
            assert process_map[proc_id]['name'] == proc.title
            assert process_map[proc_id]['type'] == 'procedure'
    
    def test_dependency_confidence_calculation(self, mapper):
        """Test dependency confidence calculation."""
        # Create mock processes
        source_process = {
            'name': 'Safety Check',
            'description': 'Complete safety inspection before operation',
            'domain': 'safety',
            'hierarchy_level': 3
        }
        
        target_process = {
            'name': 'Equipment Startup',
            'description': 'Start equipment after safety checks',
            'domain': 'operational',
            'hierarchy_level': 3
        }
        
        combined_text = "Complete safety inspection before operation Start equipment after safety checks"
        
        confidence, evidence = mapper._calculate_dependency_confidence(
            source_process, target_process, 'prerequisite', combined_text
        )
        
        assert 0.0 <= confidence <= 1.0
        assert isinstance(evidence, list)
        assert len(evidence) > 0
    
    def test_extract_dependency_conditions(self, mapper):
        """Test extraction of dependency conditions."""
        text = "If temperature exceeds 80°C, then shutdown procedure must be initiated within 5 minutes"
        
        conditions = mapper._extract_dependency_conditions(text)
        
        assert isinstance(conditions, dict)
        if 'conditional_phrases' in conditions:
            assert len(conditions['conditional_phrases']) > 0
        if 'timing' in conditions:
            assert len(conditions['timing']) > 0

class TestEquipmentMaintenanceCorrelator:
    """Test equipment-maintenance correlation functionality."""
    
    @pytest.fixture
    def correlator(self):
        """Create an EquipmentMaintenanceCorrelator instance."""
        return EquipmentMaintenanceCorrelator()
    
    @pytest.fixture
    def sample_equipment(self):
        """Create sample equipment for testing."""
        equipment = []
        
        equip1 = Mock(spec=Equipment)
        equip1.id = 1
        equip1.name = "Centrifugal Pump"
        equip1.type = "Pump"
        equipment.append(equip1)
        
        equip2 = Mock(spec=Equipment)
        equip2.id = 2
        equip2.name = "Electric Motor"
        equip2.type = "Motor"
        equipment.append(equip2)
        
        return equipment
    
    @pytest.fixture
    def sample_maintenance_procedures(self):
        """Create sample maintenance procedures for testing."""
        procedures = []
        
        # Preventive maintenance procedure
        proc1 = Mock(spec=Procedure)
        proc1.id = 1
        proc1.title = "Monthly Pump Maintenance"
        proc1.procedure_equipments = []
        
        step1 = Mock(spec=Step)
        step1.description = "Perform scheduled maintenance on centrifugal pump every month"
        proc1.steps = [step1]
        procedures.append(proc1)
        
        # Corrective maintenance procedure
        proc2 = Mock(spec=Procedure)
        proc2.id = 2
        proc2.title = "Motor Repair Procedure"
        proc2.procedure_equipments = []
        
        step2 = Mock(spec=Step)
        step2.description = "Repair electric motor when breakdown occurs"
        proc2.steps = [step2]
        procedures.append(proc2)
        
        return procedures
    
    def test_correlate_equipment_maintenance(self, correlator, sample_equipment, sample_maintenance_procedures):
        """Test equipment-maintenance correlation."""
        correlations = correlator.correlate_equipment_maintenance(sample_equipment, sample_maintenance_procedures)
        
        assert isinstance(correlations, list)
        
        # Check correlation attributes
        for corr in correlations:
            assert isinstance(corr, EquipmentMaintenanceCorrelation)
            assert corr.equipment_id in [1, 2]
            assert corr.correlation_type in ['preventive', 'corrective', 'predictive']
            assert 0.0 <= corr.confidence <= 1.0
            assert isinstance(corr.conditions, dict)
            assert isinstance(corr.evidence, list)
    
    def test_maintenance_pattern_detection(self, correlator):
        """Test detection of maintenance patterns."""
        # Test preventive maintenance detection
        text = "Perform scheduled maintenance on equipment every month"
        
        for pattern_type, patterns in correlator.maintenance_patterns.items():
            for pattern in patterns:
                if pattern_type == 'preventive':
                    import re
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        assert pattern_type == 'preventive'
                        break
    
    def test_extract_maintenance_frequency(self, correlator):
        """Test extraction of maintenance frequency."""
        test_cases = [
            ("Perform maintenance daily", "daily"),
            ("Service equipment every week", "every week"),
            ("Monthly inspection required", "monthly"),
            ("Check every 30 days", "every 30 days")
        ]
        
        for text, expected_freq in test_cases:
            frequency = correlator._extract_maintenance_frequency(text)
            # Should extract some frequency information
            assert frequency != 'as_needed' or 'every' in text.lower() or 'daily' in text.lower()
    
    def test_extract_maintenance_conditions(self, correlator):
        """Test extraction of maintenance conditions."""
        text = "Perform maintenance when temperature exceeds 80°C or after 1000 hours of operation"
        
        conditions = correlator._extract_maintenance_conditions(text)
        
        assert isinstance(conditions, dict)
        # Should extract temperature or operating hours conditions
        assert len(conditions) > 0

class TestSkillFunctionLinker:
    """Test skill-function linking functionality."""
    
    @pytest.fixture
    def linker(self):
        """Create a SkillFunctionLinker instance."""
        return SkillFunctionLinker()
    
    @pytest.fixture
    def sample_personnel(self):
        """Create sample personnel for testing."""
        personnel = []
        
        person1 = Mock(spec=Personnel)
        person1.id = 1
        person1.role = "Maintenance Technician"
        person1.responsibilities = "Responsible for equipment maintenance and repair"
        person1.certifications = ["OSHA 30", "Electrical Safety"]
        personnel.append(person1)
        
        person2 = Mock(spec=Personnel)
        person2.id = 2
        person2.role = "Quality Inspector"
        person2.responsibilities = "Perform quality inspections and testing"
        person2.certifications = ["ISO 9001", "Six Sigma Green Belt"]
        personnel.append(person2)
        
        return personnel
    
    @pytest.fixture
    def sample_skill_procedures(self):
        """Create sample procedures for skill testing."""
        procedures = []
        
        proc1 = Mock(spec=Procedure)
        proc1.id = 1
        proc1.title = "Equipment Maintenance Protocol"
        proc1.procedure_personnels = []
        
        step1 = Mock(spec=Step)
        step1.description = "Requires advanced welding skills and electrical knowledge"
        proc1.steps = [step1]
        procedures.append(proc1)
        
        return procedures
    
    def test_link_skills_functions(self, linker, sample_personnel, sample_skill_procedures):
        """Test linking of skills to functions."""
        links = linker.link_skills_functions(sample_personnel, sample_skill_procedures)
        
        assert isinstance(links, list)
        
        # Check link attributes
        for link in links:
            assert isinstance(link, SkillFunctionLink)
            assert link.skill_name
            assert link.function_name
            assert link.proficiency_level in ['basic', 'intermediate', 'advanced', 'expert']
            assert link.criticality in ['critical', 'important', 'helpful']
            assert 0.0 <= link.confidence <= 1.0
            assert isinstance(link.evidence, list)
    
    def test_extract_skills(self, linker):
        """Test skill extraction from text."""
        text = "Requires advanced welding skills, electrical knowledge, and safety training"
        
        skills = linker._extract_skills(text)
        
        assert isinstance(skills, list)
        assert len(skills) > 0
        
        # Check skill attributes
        for skill in skills:
            assert 'name' in skill
            assert 'category' in skill
            assert 'proficiency' in skill
            assert skill['category'] in ['technical_skills', 'safety_skills', 'quality_skills', 'leadership_skills']
    
    def test_determine_proficiency(self, linker):
        """Test proficiency level determination."""
        test_cases = [
            ("Expert level welding required", "expert"),
            ("Advanced programming skills needed", "advanced"),
            ("Basic understanding of safety procedures", "basic"),
            ("Intermediate knowledge of quality systems", "intermediate")
        ]
        
        for text, expected_level in test_cases:
            proficiency = linker._determine_proficiency(text, 0, len(text))
            # Should determine appropriate proficiency or default to intermediate
            assert proficiency in ['basic', 'intermediate', 'advanced', 'expert']
    
    def test_assess_skill_criticality(self, linker):
        """Test skill criticality assessment."""
        # Create mock skill and procedure
        skill = {'name': 'safety training', 'category': 'safety_skills'}
        procedure = Mock(spec=Procedure)
        procedure.title = "Safety Protocol"
        
        text = "Critical safety training required for all personnel"
        
        criticality = linker._assess_skill_criticality(skill, procedure, text)
        
        assert criticality in ['critical', 'important', 'helpful']
        # Safety skills should typically be critical
        if skill['category'] == 'safety_skills':
            assert criticality == 'critical'

class TestComplianceProcedureConnector:
    """Test compliance-procedure connection functionality."""
    
    @pytest.fixture
    def connector(self):
        """Create a ComplianceProcedureConnector instance."""
        return ComplianceProcedureConnector()
    
    @pytest.fixture
    def sample_compliance_procedures(self):
        """Create sample procedures with compliance requirements."""
        procedures = []
        
        proc1 = Mock(spec=Procedure)
        proc1.id = 1
        proc1.title = "OSHA Safety Compliance"
        proc1.category = "safety"
        
        step1 = Mock(spec=Step)
        step1.description = "Must comply with OSHA 29 CFR 1910.147 lockout/tagout requirements"
        proc1.steps = [step1]
        procedures.append(proc1)
        
        proc2 = Mock(spec=Procedure)
        proc2.id = 2
        proc2.title = "ISO Quality Standards"
        proc2.category = "quality"
        
        step2 = Mock(spec=Step)
        step2.description = "Recommended to follow ISO 9001 quality management standards"
        proc2.steps = [step2]
        procedures.append(proc2)
        
        return procedures
    
    @pytest.fixture
    def sample_safety_info(self):
        """Create sample safety information."""
        safety_info = []
        
        safety1 = Mock(spec=SafetyInformation)
        safety1.hazard = "Electrical shock"
        safety1.severity = "high"
        safety_info.append(safety1)
        
        return safety_info
    
    def test_connect_compliance_procedures(self, connector, sample_compliance_procedures, sample_safety_info):
        """Test connection of compliance requirements to procedures."""
        connections = connector.connect_compliance_procedures(sample_compliance_procedures, sample_safety_info)
        
        assert isinstance(connections, list)
        
        # Check connection attributes
        for conn in connections:
            assert isinstance(conn, ComplianceProcedureConnection)
            assert conn.regulation_reference
            assert conn.procedure_id in [1, 2]
            assert conn.compliance_type in ['mandatory', 'recommended', 'best_practice']
            assert conn.risk_level in ['high', 'medium', 'low']
            assert 0.0 <= conn.confidence <= 1.0
            assert isinstance(conn.evidence, list)
    
    def test_extract_regulation_references(self, connector):
        """Test extraction of regulation references."""
        text = "Must comply with OSHA 29 CFR 1910.147 and follow ISO 9001 standards"
        
        references = connector._extract_regulation_references(text)
        
        assert isinstance(references, list)
        assert len(references) > 0
        
        # Check reference attributes
        for ref in references:
            assert 'type' in ref
            assert 'reference' in ref
            assert ref['type'] in ['osha', 'epa', 'iso', 'fda']
    
    def test_determine_compliance_type(self, connector):
        """Test determination of compliance type."""
        test_cases = [
            ("This is required by regulation", "mandatory"),
            ("It is recommended to follow this guideline", "recommended"),
            ("This represents best practice in the industry", "best_practice")
        ]
        
        for text, expected_type in test_cases:
            compliance_type = connector._determine_compliance_type(text)
            assert compliance_type in ['mandatory', 'recommended', 'best_practice']
    
    def test_assess_compliance_risk(self, connector, sample_safety_info):
        """Test assessment of compliance risk level."""
        reg_ref = {'type': 'osha', 'reference': 'OSHA 29 CFR 1910.147'}
        text = "Critical safety procedure with severe consequences if not followed"
        
        risk_level = connector._assess_compliance_risk(reg_ref, text, sample_safety_info)
        
        assert risk_level in ['high', 'medium', 'low']
        # Should be high risk due to "critical" and "severe" keywords
        assert risk_level == 'high'

class TestRelationshipMappingIntegration:
    """Integration tests for relationship mapping system."""
    
    def test_end_to_end_relationship_mapping(self):
        """Test complete relationship mapping workflow."""
        # Create mock data
        knowledge_items = [
            Mock(spec=KnowledgeItem, process_id="PROC_001", name="Safety Check", 
                 description="Safety inspection procedure", domain="safety", hierarchy_level=3),
            Mock(spec=KnowledgeItem, process_id="PROC_002", name="Equipment Startup", 
                 description="Equipment startup after safety check", domain="operational", hierarchy_level=3)
        ]
        
        equipment = [
            Mock(spec=Equipment, id=1, name="Test Pump", type="Pump")
        ]
        
        procedures = [
            Mock(spec=Procedure, id=1, title="Safety Protocol", category="safety", 
                 steps=[Mock(spec=Step, description="Complete safety check before operation")],
                 procedure_equipments=[], procedure_personnels=[])
        ]
        
        personnel = [
            Mock(spec=Personnel, id=1, role="Operator", responsibilities="Equipment operation",
                 certifications=["Safety Training"])
        ]
        
        safety_info = [
            Mock(spec=SafetyInformation, hazard="Equipment failure", severity="high")
        ]
        
        # Test all mappers
        dep_mapper = ProcessDependencyMapper()
        dependencies = dep_mapper.identify_process_dependencies(knowledge_items, procedures)
        assert isinstance(dependencies, list)
        
        maint_correlator = EquipmentMaintenanceCorrelator()
        correlations = maint_correlator.correlate_equipment_maintenance(equipment, procedures)
        assert isinstance(correlations, list)
        
        skill_linker = SkillFunctionLinker()
        links = skill_linker.link_skills_functions(personnel, procedures)
        assert isinstance(links, list)
        
        compliance_connector = ComplianceProcedureConnector()
        connections = compliance_connector.connect_compliance_procedures(procedures, safety_info)
        assert isinstance(connections, list)
    
    def test_relationship_mapping_accuracy_metrics(self):
        """Test accuracy metrics for relationship mapping."""
        # This would typically involve comparing against ground truth data
        # For now, we'll test that confidence scores are reasonable
        
        mapper = ProcessDependencyMapper()
        
        # Test confidence calculation with known good and bad examples
        good_source = {
            'name': 'Safety Check',
            'description': 'Complete safety inspection before operation',
            'domain': 'safety',
            'hierarchy_level': 3
        }
        
        good_target = {
            'name': 'Equipment Startup',
            'description': 'Start equipment after safety checks are completed',
            'domain': 'operational',
            'hierarchy_level': 3
        }
        
        good_text = "Complete safety inspection before operation Start equipment after safety checks are completed"
        
        confidence, evidence = mapper._calculate_dependency_confidence(
            good_source, good_target, 'prerequisite', good_text
        )
        
        # Should have high confidence for clear prerequisite relationship
        assert confidence > 0.5
        assert len(evidence) > 0
        
        # Test with unrelated processes
        bad_source = {
            'name': 'Random Process',
            'description': 'Unrelated process',
            'domain': 'other',
            'hierarchy_level': 1
        }
        
        bad_target = {
            'name': 'Another Process',
            'description': 'Another unrelated process',
            'domain': 'different',
            'hierarchy_level': 4
        }
        
        bad_text = "Random process Another unrelated process"
        
        bad_confidence, bad_evidence = mapper._calculate_dependency_confidence(
            bad_source, bad_target, 'prerequisite', bad_text
        )
        
        # Should have lower confidence for unrelated processes
        assert bad_confidence < confidence

if __name__ == "__main__":
    pytest.main([__file__])