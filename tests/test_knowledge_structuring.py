"""
Unit tests for knowledge structuring framework.

Tests the hierarchical process organizer, domain classifier, metadata enricher,
and version controller components.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from src.ai.knowledge_structuring_framework import (
    HierarchicalProcessOrganizer,
    DomainClassifier,
    MetadataEnricher,
    VersionController,
    KnowledgeStructuringFramework,
    KnowledgeDomain,
    HierarchyLevel,
    SourceQuality,
    CriticalityLevel,
    ProcessHierarchyNode,
    StructuredKnowledgeItem,
    DomainClassification,
    VersionInfo
)

class TestHierarchicalProcessOrganizer:
    """Test the hierarchical process organizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.organizer = HierarchicalProcessOrganizer()
    
    def test_initialization(self):
        """Test organizer initialization."""
        assert self.organizer.hierarchy_patterns is not None
        assert HierarchyLevel.CORE_BUSINESS_FUNCTION in self.organizer.hierarchy_patterns
        assert HierarchyLevel.SPECIFIC_STEP in self.organizer.hierarchy_patterns
    
    def test_classify_hierarchy_level_core_function(self):
        """Test classification of core business functions."""
        process = {
            'name': 'Manufacturing Operations',
            'description': 'Core manufacturing business function',
            'text': 'Primary manufacturing operations for the facility'
        }
        
        level = self.organizer._classify_hierarchy_level(process)
        assert level == HierarchyLevel.CORE_BUSINESS_FUNCTION
    
    def test_classify_hierarchy_level_department_operation(self):
        """Test classification of department operations."""
        process = {
            'name': 'Assembly Department Operation',
            'description': 'Assembly department daily operations',
            'text': 'Department operation for assembly line management'
        }
        
        level = self.organizer._classify_hierarchy_level(process)
        assert level == HierarchyLevel.DEPARTMENT_OPERATION
    
    def test_classify_hierarchy_level_individual_procedure(self):
        """Test classification of individual procedures."""
        process = {
            'name': 'Equipment Startup Procedure',
            'description': 'Standard operating procedure for equipment startup',
            'text': 'SOP for starting up manufacturing equipment'
        }
        
        level = self.organizer._classify_hierarchy_level(process)
        assert level == HierarchyLevel.INDIVIDUAL_PROCEDURE
    
    def test_classify_hierarchy_level_specific_step(self):
        """Test classification of specific steps."""
        process = {
            'name': 'Step 1: Check Power Supply',
            'description': 'First step in the startup sequence',
            'text': '1. Verify that power supply is connected and operational'
        }
        
        level = self.organizer._classify_hierarchy_level(process)
        assert level == HierarchyLevel.SPECIFIC_STEP
    
    def test_generate_process_id(self):
        """Test process ID generation."""
        process = {'name': 'Test Process', 'text': 'Test content'}
        level = HierarchyLevel.INDIVIDUAL_PROCEDURE
        
        process_id = self.organizer._generate_process_id(process, level)
        
        assert process_id.startswith('IP-')
        assert len(process_id) == 11  # IP- + 8 character hash
    
    def test_organize_process_hierarchy(self):
        """Test organizing processes into hierarchy."""
        raw_processes = [
            {
                'name': 'Manufacturing',
                'description': 'Core manufacturing function',
                'text': 'Primary manufacturing operations'
            },
            {
                'name': 'Assembly Procedure',
                'description': 'Standard assembly procedure',
                'text': 'SOP for product assembly'
            },
            {
                'name': 'Step 1: Prepare Components',
                'description': 'First assembly step',
                'text': '1. Gather all required components'
            }
        ]
        
        hierarchy = self.organizer.organize_process_hierarchy(raw_processes)
        
        assert len(hierarchy) == 3
        
        # Check that all nodes are ProcessHierarchyNode instances
        for node in hierarchy.values():
            assert isinstance(node, ProcessHierarchyNode)
            assert node.process_id is not None
            assert node.name is not None
            assert node.level in HierarchyLevel
    
    def test_find_best_parent(self):
        """Test finding best parent for a child process."""
        child = ProcessHierarchyNode(
            process_id='child-1',
            name='Assembly Step',
            description='Step in assembly process',
            level=HierarchyLevel.SPECIFIC_STEP
        )
        
        candidates = [
            ProcessHierarchyNode(
                process_id='parent-1',
                name='Assembly Procedure',
                description='Assembly procedure documentation',
                level=HierarchyLevel.INDIVIDUAL_PROCEDURE
            ),
            ProcessHierarchyNode(
                process_id='parent-2',
                name='Quality Control',
                description='Quality control procedures',
                level=HierarchyLevel.INDIVIDUAL_PROCEDURE
            )
        ]
        
        best_parent = self.organizer._find_best_parent(child, candidates)
        
        assert best_parent is not None
        assert best_parent.process_id == 'parent-1'  # Should match assembly-related parent

class TestDomainClassifier:
    """Test the domain classifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = DomainClassifier()
    
    def test_initialization(self):
        """Test classifier initialization."""
        assert self.classifier.domain_keywords is not None
        assert self.classifier.domain_patterns is not None
        assert KnowledgeDomain.OPERATIONAL in self.classifier.domain_keywords
        assert KnowledgeDomain.SAFETY_COMPLIANCE in self.classifier.domain_keywords
    
    def test_classify_operational_domain(self):
        """Test classification of operational knowledge."""
        knowledge_items = [{
            'name': 'Production Process',
            'description': 'Manufacturing production workflow',
            'text': 'Standard production procedure for assembly line operations'
        }]
        
        classifications = self.classifier.classify_knowledge_domains(knowledge_items)
        
        assert len(classifications) == 1
        assert classifications[0].domain == KnowledgeDomain.OPERATIONAL
        assert classifications[0].confidence > 0
        assert 'production' in classifications[0].keywords
    
    def test_classify_safety_domain(self):
        """Test classification of safety knowledge."""
        knowledge_items = [{
            'name': 'Emergency Evacuation',
            'description': 'Safety evacuation procedure',
            'text': 'Emergency evacuation protocol for hazardous situations'
        }]
        
        classifications = self.classifier.classify_knowledge_domains(knowledge_items)
        
        assert len(classifications) == 1
        assert classifications[0].domain == KnowledgeDomain.SAFETY_COMPLIANCE
        assert 'emergency' in classifications[0].keywords or 'safety' in classifications[0].keywords
    
    def test_classify_equipment_domain(self):
        """Test classification of equipment knowledge."""
        knowledge_items = [{
            'name': 'Equipment Maintenance',
            'description': 'Machine maintenance procedure',
            'text': 'Equipment maintenance and repair guidelines'
        }]
        
        classifications = self.classifier.classify_knowledge_domains(knowledge_items)
        
        assert len(classifications) == 1
        assert classifications[0].domain == KnowledgeDomain.EQUIPMENT_TECHNOLOGY
        assert 'equipment' in classifications[0].keywords or 'maintenance' in classifications[0].keywords
    
    def test_classify_multiple_items(self):
        """Test classification of multiple knowledge items."""
        knowledge_items = [
            {
                'name': 'Production Line',
                'description': 'Manufacturing operations',
                'text': 'Production line operational procedures'
            },
            {
                'name': 'Safety Training',
                'description': 'Employee safety training',
                'text': 'Safety training program for new employees'
            }
        ]
        
        classifications = self.classifier.classify_knowledge_domains(knowledge_items)
        
        assert len(classifications) == 2
        domains = [c.domain for c in classifications]
        assert KnowledgeDomain.OPERATIONAL in domains
        assert KnowledgeDomain.SAFETY_COMPLIANCE in domains or KnowledgeDomain.HUMAN_RESOURCES in domains
    
    def test_classify_empty_content(self):
        """Test classification with empty content."""
        knowledge_items = [{'name': '', 'description': '', 'text': ''}]
        
        classifications = self.classifier.classify_knowledge_domains(knowledge_items)
        
        assert len(classifications) == 0

class TestMetadataEnricher:
    """Test the metadata enricher."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.enricher = MetadataEnricher()
    
    def test_initialization(self):
        """Test enricher initialization."""
        assert self.enricher.quality_indicators is not None
        assert self.enricher.completeness_factors is not None
        assert self.enricher.criticality_indicators is not None
    
    def test_enrich_metadata_basic(self):
        """Test basic metadata enrichment."""
        knowledge_item = {
            'name': 'Test Procedure',
            'description': 'A test procedure for validation',
            'text': 'This is a standard operating procedure with multiple steps',
            'knowledge_type': 'procedural'
        }
        
        enriched = self.enricher.enrich_metadata(knowledge_item)
        
        assert isinstance(enriched, StructuredKnowledgeItem)
        assert enriched.name == 'Test Procedure'
        assert enriched.knowledge_type == 'procedural'
        assert isinstance(enriched.domain, KnowledgeDomain)
        assert isinstance(enriched.hierarchy_level, HierarchyLevel)
        assert 0.0 <= enriched.confidence_score <= 1.0
        assert isinstance(enriched.source_quality, SourceQuality)
        assert 0.0 <= enriched.completeness_index <= 1.0
        assert isinstance(enriched.criticality_level, CriticalityLevel)
    
    def test_assess_high_source_quality(self):
        """Test assessment of high source quality."""
        item = {
            'name': 'Official Standard Procedure',
            'description': 'Certified and approved procedure',
            'text': 'This is an official standard operating procedure'
        }
        
        quality = self.enricher._assess_source_quality(item)
        assert quality == SourceQuality.HIGH
    
    def test_assess_low_source_quality(self):
        """Test assessment of low source quality."""
        item = {
            'name': 'Informal Draft',
            'description': 'Preliminary draft notes',
            'text': 'This is an informal draft of a temporary workaround'
        }
        
        quality = self.enricher._assess_source_quality(item)
        assert quality == SourceQuality.LOW
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        item = {
            'name': 'Detailed Procedure',
            'description': 'A comprehensive procedure with numbered steps',
            'text': '1. First step in the procedure 2. Second step with details 3. Final step',
            'confidence': 0.8
        }
        
        confidence = self.enricher._calculate_confidence_score(item)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high due to structured content
    
    def test_determine_critical_level(self):
        """Test criticality level determination."""
        critical_item = {
            'name': 'Emergency Safety Procedure',
            'description': 'Critical safety procedure',
            'text': 'This is a critical emergency procedure that must be followed'
        }
        
        criticality = self.enricher._determine_criticality_level(critical_item)
        assert criticality == CriticalityLevel.CRITICAL
    
    def test_calculate_completeness_index(self):
        """Test completeness index calculation."""
        complete_item = {
            'name': 'Complete Procedure',
            'description': 'A complete procedure with all steps and requirements',
            'text': '1. Step one with input requirements 2. Step two with output specifications 3. Final step with success criteria and expected results'
        }
        
        completeness = self.enricher._calculate_completeness_index(complete_item)
        assert 0.0 <= completeness <= 1.0
        assert completeness > 0.5  # Should be high due to detailed content
    
    def test_extract_tags(self):
        """Test tag extraction."""
        item = {
            'name': 'Manufacturing Process',
            'description': 'Production procedure for quality control',
            'text': 'This procedure covers manufacturing operations and quality assurance'
        }
        
        tags = self.enricher._extract_tags(item)
        assert isinstance(tags, set)
        assert len(tags) > 0
        # Should contain relevant keywords
        expected_tags = {'manufacturing', 'production', 'procedure', 'quality', 'operations'}
        assert len(tags & expected_tags) > 0

class TestVersionController:
    """Test the version controller."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.controller = VersionController()
    
    def test_initialization(self):
        """Test controller initialization."""
        assert self.controller.version_history == {}
    
    def test_manage_versions(self):
        """Test version management."""
        knowledge_item = StructuredKnowledgeItem(
            process_id='test-001',
            name='Test Item',
            description='Test description',
            knowledge_type='procedural',
            domain=KnowledgeDomain.OPERATIONAL,
            hierarchy_level=HierarchyLevel.INDIVIDUAL_PROCEDURE,
            confidence_score=0.8,
            source_quality=SourceQuality.HIGH,
            completeness_index=0.9,
            criticality_level=CriticalityLevel.MEDIUM,
            access_level='internal'
        )
        
        changes = ['Initial creation', 'Added detailed description']
        updated_item = self.controller.manage_versions(knowledge_item, changes, 'test_user')
        
        assert len(updated_item.change_history) == 1
        assert updated_item.change_history[0]['changes'] == changes
        assert updated_item.change_history[0]['author'] == 'test_user'
        assert 'test-001' in self.controller.version_history
    
    def test_increment_version(self):
        """Test version incrementing."""
        knowledge_item = StructuredKnowledgeItem(
            process_id='test-002',
            name='Test Item 2',
            description='Test description',
            knowledge_type='procedural',
            domain=KnowledgeDomain.OPERATIONAL,
            hierarchy_level=HierarchyLevel.INDIVIDUAL_PROCEDURE,
            confidence_score=0.8,
            source_quality=SourceQuality.HIGH,
            completeness_index=0.9,
            criticality_level=CriticalityLevel.MEDIUM,
            access_level='internal',
            version=1
        )
        
        changes = ['Updated description']
        updated_item = self.controller.increment_version(knowledge_item, changes, 'test_user')
        
        assert updated_item.version == 2
        assert len(updated_item.change_history) == 1
    
    def test_get_version_history(self):
        """Test getting version history."""
        knowledge_item = StructuredKnowledgeItem(
            process_id='test-003',
            name='Test Item 3',
            description='Test description',
            knowledge_type='procedural',
            domain=KnowledgeDomain.OPERATIONAL,
            hierarchy_level=HierarchyLevel.INDIVIDUAL_PROCEDURE,
            confidence_score=0.8,
            source_quality=SourceQuality.HIGH,
            completeness_index=0.9,
            criticality_level=CriticalityLevel.MEDIUM,
            access_level='internal'
        )
        
        # Add some versions
        self.controller.manage_versions(knowledge_item, ['Initial'], 'user1')
        self.controller.increment_version(knowledge_item, ['Update 1'], 'user2')
        
        history = self.controller.get_version_history('test-003')
        assert len(history) == 2
        assert all(isinstance(v, VersionInfo) for v in history)
    
    def test_validate_integrity(self):
        """Test integrity validation."""
        knowledge_item = StructuredKnowledgeItem(
            process_id='test-004',
            name='Test Item 4',
            description='Test description',
            knowledge_type='procedural',
            domain=KnowledgeDomain.OPERATIONAL,
            hierarchy_level=HierarchyLevel.INDIVIDUAL_PROCEDURE,
            confidence_score=0.8,
            source_quality=SourceQuality.HIGH,
            completeness_index=0.9,
            criticality_level=CriticalityLevel.MEDIUM,
            access_level='internal'
        )
        
        # Add version history
        self.controller.manage_versions(knowledge_item, ['Initial'], 'user1')
        
        # Should validate successfully
        assert self.controller.validate_integrity(knowledge_item) is True

class TestKnowledgeStructuringFramework:
    """Test the main knowledge structuring framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.framework = KnowledgeStructuringFramework()
    
    def test_initialization(self):
        """Test framework initialization."""
        assert self.framework.hierarchical_organizer is not None
        assert self.framework.domain_classifier is not None
        assert self.framework.metadata_enricher is not None
        assert self.framework.version_controller is not None
    
    def test_structure_knowledge_basic(self):
        """Test basic knowledge structuring."""
        raw_knowledge_items = [
            {
                'name': 'Manufacturing Process',
                'description': 'Core manufacturing process',
                'text': 'Standard manufacturing procedure for production line',
                'knowledge_type': 'procedural'
            },
            {
                'name': 'Safety Check',
                'description': 'Safety verification step',
                'text': '1. Verify all safety equipment is in place',
                'knowledge_type': 'procedural'
            }
        ]
        
        results = self.framework.structure_knowledge(raw_knowledge_items)
        
        # Verify structure
        assert 'structured_items' in results
        assert 'process_hierarchy' in results
        assert 'domain_classifications' in results
        assert 'structuring_metadata' in results
        
        # Verify content
        assert len(results['structured_items']) == 2
        assert results['structuring_metadata']['total_items'] == 2
        assert results['structuring_metadata']['structured_items_count'] == 2
    
    def test_structure_knowledge_empty_input(self):
        """Test structuring with empty input."""
        raw_knowledge_items = []
        
        results = self.framework.structure_knowledge(raw_knowledge_items)
        
        assert results['structuring_metadata']['total_items'] == 0
        assert len(results['structured_items']) == 0
    
    def test_structure_knowledge_integration(self):
        """Test integration of all structuring components."""
        raw_knowledge_items = [
            {
                'name': 'Equipment Maintenance',
                'description': 'Critical equipment maintenance procedure',
                'text': 'This is a critical maintenance procedure with detailed steps: 1. Shutdown equipment 2. Perform inspection 3. Replace components',
                'knowledge_type': 'procedural',
                'confidence': 0.9
            }
        ]
        
        results = self.framework.structure_knowledge(raw_knowledge_items)
        
        # Verify all components worked
        structured_item = results['structured_items'][0]
        assert isinstance(structured_item, StructuredKnowledgeItem)
        assert structured_item.domain == KnowledgeDomain.EQUIPMENT_TECHNOLOGY
        assert structured_item.criticality_level == CriticalityLevel.CRITICAL
        assert structured_item.confidence_score > 0.5
        assert len(structured_item.change_history) == 1  # Version control applied
        
        # Verify hierarchy was created
        assert len(results['process_hierarchy']) > 0
        
        # Verify domain classification
        assert len(results['domain_classifications']) > 0

# Integration tests
class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.framework = KnowledgeStructuringFramework()
    
    def test_manufacturing_knowledge_structuring(self):
        """Test structuring of manufacturing knowledge."""
        raw_knowledge_items = [
            {
                'name': 'Manufacturing Operations',
                'description': 'Core manufacturing business function',
                'text': 'Primary manufacturing operations including production planning and execution',
                'knowledge_type': 'explicit'
            },
            {
                'name': 'Assembly Line Procedure',
                'description': 'Standard assembly procedure',
                'text': 'SOP for assembly line operations with quality checkpoints',
                'knowledge_type': 'procedural'
            },
            {
                'name': 'Quality Inspection Step',
                'description': 'Quality control inspection',
                'text': '1. Inspect product for defects 2. Record inspection results 3. Route to next station',
                'knowledge_type': 'procedural'
            }
        ]
        
        results = self.framework.structure_knowledge(raw_knowledge_items)
        
        # Should have different hierarchy levels
        hierarchy_levels = set()
        for item in results['structured_items']:
            hierarchy_levels.add(item.hierarchy_level)
        
        assert len(hierarchy_levels) > 1  # Multiple levels represented
        
        # Should classify as operational domain primarily
        operational_items = [item for item in results['structured_items'] 
                           if item.domain == KnowledgeDomain.OPERATIONAL]
        assert len(operational_items) > 0
    
    def test_safety_knowledge_structuring(self):
        """Test structuring of safety knowledge."""
        raw_knowledge_items = [
            {
                'name': 'Emergency Response Protocol',
                'description': 'Critical emergency response procedure',
                'text': 'Emergency response protocol for fire, chemical spills, and evacuations. This is a critical safety procedure that must be followed immediately.',
                'knowledge_type': 'procedural'
            },
            {
                'name': 'PPE Requirements',
                'description': 'Personal protective equipment requirements',
                'text': 'Required PPE includes safety glasses, hard hat, steel-toed boots, and gloves for all personnel',
                'knowledge_type': 'explicit'
            }
        ]
        
        results = self.framework.structure_knowledge(raw_knowledge_items)
        
        # Should classify as safety domain
        safety_items = [item for item in results['structured_items'] 
                       if item.domain == KnowledgeDomain.SAFETY_COMPLIANCE]
        assert len(safety_items) > 0
        
        # Should identify critical items
        critical_items = [item for item in results['structured_items'] 
                         if item.criticality_level == CriticalityLevel.CRITICAL]
        assert len(critical_items) > 0

if __name__ == '__main__':
    pytest.main([__file__])