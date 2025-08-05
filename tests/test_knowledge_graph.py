"""
Unit tests for knowledge graph generation and query system.

Tests the functionality of knowledge graph building, traversal algorithms,
visualization data preparation, and complex relationship queries.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
from datetime import datetime
from typing import List, Dict, Any

from src.ai.knowledge_graph import (
    KnowledgeGraphBuilder, GraphTraversalEngine, GraphVisualizationPreparer,
    GraphQueryInterface, GraphNode, GraphEdge, GraphPath, DependencyAnalysis,
    GraphVisualizationData
)
from src.ai.relationship_mapper import (
    ProcessDependency, EquipmentMaintenanceCorrelation,
    SkillFunctionLink, ComplianceProcedureConnection
)
from src.database.models import KnowledgeItem, Equipment, Personnel

class TestKnowledgeGraphBuilder:
    """Test knowledge graph building functionality."""
    
    @pytest.fixture
    def builder(self):
        """Create a KnowledgeGraphBuilder instance."""
        return KnowledgeGraphBuilder()
    
    @pytest.fixture
    def sample_knowledge_items(self):
        """Create sample knowledge items."""
        items = []
        
        item1 = Mock(spec=KnowledgeItem)
        item1.process_id = "PROC_001"
        item1.name = "Safety Check"
        item1.description = "Safety inspection procedure"
        item1.domain = "safety"
        item1.hierarchy_level = 3
        item1.knowledge_type = "procedural"
        item1.confidence_score = 0.8
        item1.completeness_index = 0.9
        item1.criticality_level = "high"
        item1.source_quality = "high"
        item1.created_at = datetime.now()
        item1.updated_at = datetime.now()
        items.append(item1)
        
        item2 = Mock(spec=KnowledgeItem)
        item2.process_id = "PROC_002"
        item2.name = "Equipment Startup"
        item2.description = "Equipment startup procedure"
        item2.domain = "operational"
        item2.hierarchy_level = 3
        item2.knowledge_type = "procedural"
        item2.confidence_score = 0.7
        item2.completeness_index = 0.8
        item2.criticality_level = "medium"
        item2.source_quality = "medium"
        item2.created_at = datetime.now()
        item2.updated_at = datetime.now()
        items.append(item2)
        
        return items
    
    @pytest.fixture
    def sample_process_dependencies(self):
        """Create sample process dependencies."""
        return [
            ProcessDependency(
                source_process_id="PROC_001",
                target_process_id="PROC_002",
                dependency_type="prerequisite",
                strength=0.8,
                conditions={"timing": "before startup"},
                confidence=0.9,
                evidence=["safety check required before startup"]
            )
        ]
    
    @pytest.fixture
    def sample_equipment(self):
        """Create sample equipment."""
        equipment = []
        
        equip1 = Mock(spec=Equipment)
        equip1.id = 1
        equip1.name = "Test Pump"
        equip1.type = "Pump"
        equip1.specifications = {"power": "10HP"}
        equip1.location = "Building A"
        equip1.confidence = 0.8
        equip1.document_id = 1
        equipment.append(equip1)
        
        return equipment
    
    @pytest.fixture
    def sample_personnel(self):
        """Create sample personnel."""
        personnel = []
        
        person1 = Mock(spec=Personnel)
        person1.id = 1
        person1.name = "John Doe"
        person1.role = "Operator"
        person1.responsibilities = "Equipment operation"
        person1.certifications = ["Safety Training"]
        person1.confidence = 0.9
        person1.document_id = 1
        personnel.append(person1)
        
        return personnel
    
    def test_build_knowledge_graph(self, builder, sample_knowledge_items, 
                                 sample_process_dependencies, sample_equipment, sample_personnel):
        """Test building a complete knowledge graph."""
        graph = builder.build_knowledge_graph(
            knowledge_items=sample_knowledge_items,
            process_dependencies=sample_process_dependencies,
            equipment_correlations=[],
            skill_links=[],
            compliance_connections=[],
            equipment_list=sample_equipment,
            personnel_list=sample_personnel
        )
        
        assert isinstance(graph, nx.MultiDiGraph)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() >= 0
        
        # Check that knowledge items were added as nodes
        for item in sample_knowledge_items:
            assert item.process_id in graph.nodes()
            node_data = graph.nodes[item.process_id]
            assert node_data['label'] == item.name
            assert node_data['node_type'] == 'process'
        
        # Check that equipment was added as nodes
        for equipment in sample_equipment:
            equip_id = f"EQUIP_{equipment.id}"
            assert equip_id in graph.nodes()
            node_data = graph.nodes[equip_id]
            assert node_data['label'] == equipment.name
            assert node_data['node_type'] == 'equipment'
        
        # Check that personnel was added as nodes
        for person in sample_personnel:
            person_id = f"PERSON_{person.id}"
            assert person_id in graph.nodes()
            node_data = graph.nodes[person_id]
            assert node_data['label'] == person.name
            assert node_data['node_type'] == 'person'
    
    def test_add_knowledge_item_nodes(self, builder, sample_knowledge_items):
        """Test adding knowledge item nodes."""
        builder._add_knowledge_item_nodes(sample_knowledge_items)
        
        assert len(builder.node_registry) == len(sample_knowledge_items)
        assert builder.graph.number_of_nodes() == len(sample_knowledge_items)
        
        for item in sample_knowledge_items:
            assert item.process_id in builder.node_registry
            node = builder.node_registry[item.process_id]
            assert node.label == item.name
            assert node.node_type == 'process'
    
    def test_add_process_dependency_edges(self, builder, sample_knowledge_items, sample_process_dependencies):
        """Test adding process dependency edges."""
        # First add nodes
        builder._add_knowledge_item_nodes(sample_knowledge_items)
        
        # Then add edges
        builder._add_process_dependency_edges(sample_process_dependencies)
        
        assert builder.graph.number_of_edges() > 0
        
        # Check edge properties
        for dep in sample_process_dependencies:
            if builder.graph.has_edge(dep.source_process_id, dep.target_process_id):
                edge_data = builder.graph.get_edge_data(dep.source_process_id, dep.target_process_id)
                assert len(edge_data) > 0
                
                # Check first edge data
                first_edge = list(edge_data.values())[0]
                assert first_edge['confidence'] == dep.confidence
                assert first_edge['weight'] == dep.strength
    
    def test_equipment_correlation_edges(self, builder, sample_equipment):
        """Test adding equipment correlation edges."""
        # Add equipment nodes first
        builder._add_equipment_nodes(sample_equipment)
        
        # Create sample correlation
        correlations = [
            EquipmentMaintenanceCorrelation(
                equipment_id=1,
                maintenance_pattern="preventive_maintenance",
                correlation_type="preventive",
                frequency="monthly",
                conditions={},
                confidence=0.8,
                evidence=["scheduled maintenance"]
            )
        ]
        
        builder._add_equipment_correlation_edges(correlations)
        
        # Should have equipment node
        assert "EQUIP_1" in builder.graph.nodes()
    
    def test_skill_function_edges(self, builder):
        """Test adding skill-function edges."""
        # Create sample skill links
        skill_links = [
            SkillFunctionLink(
                skill_name="Welding",
                function_name="Equipment Maintenance",
                proficiency_level="advanced",
                criticality="critical",
                confidence=0.9,
                evidence=["welding required for repairs"]
            )
        ]
        
        builder._add_skill_function_edges(skill_links)
        
        # Should create skill node
        skill_id = "SKILL_Welding"
        assert skill_id in builder.node_registry
        assert builder.node_registry[skill_id].node_type == 'skill'
    
    def test_compliance_connection_edges(self, builder):
        """Test adding compliance connection edges."""
        # Create sample compliance connections
        connections = [
            ComplianceProcedureConnection(
                regulation_reference="OSHA 29 CFR 1910.147",
                procedure_id=1,
                compliance_type="mandatory",
                risk_level="high",
                confidence=0.9,
                evidence=["OSHA requirement"]
            )
        ]
        
        builder._add_compliance_connection_edges(connections)
        
        # Should create regulation node
        reg_id = "REG_OSHA_29_CFR_1910_147"
        assert reg_id in builder.node_registry
        assert builder.node_registry[reg_id].node_type == 'regulation'
    
    def test_graph_validation(self, builder, sample_knowledge_items):
        """Test graph validation functionality."""
        builder._add_knowledge_item_nodes(sample_knowledge_items)
        
        # Should not raise any exceptions
        builder._validate_graph()
        
        # Test with isolated node
        isolated_node = GraphNode(
            id="ISOLATED",
            label="Isolated Node",
            node_type="process",
            properties={},
            metadata={}
        )
        builder._add_node_to_graph(isolated_node)
        
        # Should log warning about isolated nodes
        builder._validate_graph()
    
    def test_graph_optimization(self, builder, sample_knowledge_items):
        """Test graph optimization functionality."""
        builder._add_knowledge_item_nodes(sample_knowledge_items)
        
        # Add duplicate edges with different confidence
        edge1 = GraphEdge(
            source_id="PROC_001",
            target_id="PROC_002",
            relationship_type="depends_on",
            properties={},
            weight=0.5,
            confidence=0.6
        )
        
        edge2 = GraphEdge(
            source_id="PROC_001",
            target_id="PROC_002",
            relationship_type="depends_on",
            properties={},
            weight=0.7,
            confidence=0.8
        )
        
        builder._add_edge_to_graph(edge1)
        builder._add_edge_to_graph(edge2)
        
        initial_edge_count = builder.graph.number_of_edges()
        
        # Optimize should remove lower confidence edge
        builder._optimize_graph()
        
        final_edge_count = builder.graph.number_of_edges()
        assert final_edge_count <= initial_edge_count

class TestGraphTraversalEngine:
    """Test graph traversal and dependency analysis."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = nx.MultiDiGraph()
        
        # Add nodes
        nodes = [
            ("A", {"label": "Node A", "node_type": "process"}),
            ("B", {"label": "Node B", "node_type": "process"}),
            ("C", {"label": "Node C", "node_type": "process"}),
            ("D", {"label": "Node D", "node_type": "equipment"})
        ]
        
        for node_id, data in nodes:
            graph.add_node(node_id, **data)
        
        # Add edges
        edges = [
            ("A", "B", {"relationship_type": "depends_prerequisite", "weight": 0.8, "confidence": 0.9}),
            ("B", "C", {"relationship_type": "depends_downstream", "weight": 0.7, "confidence": 0.8}),
            ("A", "D", {"relationship_type": "uses_equipment", "weight": 0.6, "confidence": 0.7})
        ]
        
        for source, target, data in edges:
            graph.add_edge(source, target, **data)
        
        return graph
    
    @pytest.fixture
    def traversal_engine(self, sample_graph):
        """Create a GraphTraversalEngine instance."""
        return GraphTraversalEngine(sample_graph)
    
    def test_analyze_dependencies(self, traversal_engine):
        """Test comprehensive dependency analysis."""
        analysis = traversal_engine.analyze_dependencies()
        
        assert isinstance(analysis, DependencyAnalysis)
        assert isinstance(analysis.critical_paths, list)
        assert isinstance(analysis.bottlenecks, list)
        assert isinstance(analysis.circular_dependencies, list)
        assert isinstance(analysis.dependency_depth, dict)
        assert isinstance(analysis.impact_analysis, dict)
    
    def test_find_critical_paths(self, traversal_engine):
        """Test finding critical paths."""
        critical_paths = traversal_engine._find_critical_paths()
        
        assert isinstance(critical_paths, list)
        
        for path in critical_paths:
            assert isinstance(path, GraphPath)
            assert len(path.nodes) > 0
            assert len(path.edges) == len(path.nodes) - 1
            assert 0.0 <= path.confidence <= 1.0
    
    def test_find_critical_paths_with_target(self, traversal_engine):
        """Test finding critical paths to specific target."""
        critical_paths = traversal_engine._find_critical_paths("C")
        
        assert isinstance(critical_paths, list)
        
        # Should find paths ending at node C
        for path in critical_paths:
            if path.nodes:
                assert path.nodes[-1].id == "C"
    
    def test_identify_bottlenecks(self, traversal_engine):
        """Test bottleneck identification."""
        bottlenecks = traversal_engine._identify_bottlenecks()
        
        assert isinstance(bottlenecks, list)
        
        for bottleneck in bottlenecks:
            assert isinstance(bottleneck, GraphNode)
            assert 'centrality_score' in bottleneck.properties
            assert 'in_degree' in bottleneck.properties
            assert 'out_degree' in bottleneck.properties
    
    def test_detect_circular_dependencies(self, traversal_engine):
        """Test circular dependency detection."""
        circular_deps = traversal_engine._detect_circular_dependencies()
        
        assert isinstance(circular_deps, list)
        
        # Add a cycle to test detection
        traversal_engine.graph.add_edge("C", "A", relationship_type="feedback", weight=0.5, confidence=0.6)
        
        circular_deps = traversal_engine._detect_circular_dependencies()
        
        # Should detect the cycle
        assert len(circular_deps) > 0
    
    def test_calculate_dependency_depth(self, traversal_engine):
        """Test dependency depth calculation."""
        depth_map = traversal_engine._calculate_dependency_depth()
        
        assert isinstance(depth_map, dict)
        assert len(depth_map) > 0
        
        # All nodes should have depth values
        for node in traversal_engine.graph.nodes():
            assert node in depth_map
            assert isinstance(depth_map[node], int)
            assert depth_map[node] >= 0
    
    def test_perform_impact_analysis(self, traversal_engine):
        """Test impact analysis."""
        impact_analysis = traversal_engine._perform_impact_analysis()
        
        assert isinstance(impact_analysis, dict)
        
        for node, analysis in impact_analysis.items():
            assert 'direct_successors' in analysis
            assert 'direct_predecessors' in analysis
            assert 'downstream_impact' in analysis
            assert 'upstream_dependencies' in analysis
            assert 'criticality' in analysis
            assert analysis['criticality'] in ['critical', 'high', 'medium', 'low']
    
    def test_assess_node_criticality(self, traversal_engine):
        """Test node criticality assessment."""
        # Test different criticality levels
        assert traversal_engine._assess_node_criticality("test", 15, 5) == 'critical'
        assert traversal_engine._assess_node_criticality("test", 8, 3) == 'high'
        assert traversal_engine._assess_node_criticality("test", 3, 2) == 'medium'
        assert traversal_engine._assess_node_criticality("test", 1, 1) == 'low'

class TestGraphVisualizationPreparer:
    """Test graph visualization data preparation."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for visualization testing."""
        graph = nx.MultiDiGraph()
        
        # Add nodes with visualization properties
        nodes = [
            ("A", {
                "label": "Process A", 
                "node_type": "process",
                "color": "#4CAF50",
                "shape": "box",
                "properties": {"hierarchy_level": 1, "criticality_level": "high"}
            }),
            ("B", {
                "label": "Equipment B", 
                "node_type": "equipment",
                "color": "#FF9800",
                "shape": "ellipse",
                "properties": {"hierarchy_level": 2, "criticality_level": "medium"}
            })
        ]
        
        for node_id, data in nodes:
            graph.add_node(node_id, **data)
        
        # Add edges
        graph.add_edge("A", "B", 
                      relationship_type="uses_equipment", 
                      weight=0.8, 
                      confidence=0.9)
        
        return graph
    
    @pytest.fixture
    def viz_preparer(self, sample_graph):
        """Create a GraphVisualizationPreparer instance."""
        return GraphVisualizationPreparer(sample_graph)
    
    def test_prepare_visualization_data(self, viz_preparer):
        """Test preparation of visualization data."""
        viz_data = viz_preparer.prepare_visualization_data()
        
        assert isinstance(viz_data, GraphVisualizationData)
        assert isinstance(viz_data.nodes, list)
        assert isinstance(viz_data.edges, list)
        assert isinstance(viz_data.layout, dict)
        assert isinstance(viz_data.metadata, dict)
        
        # Check nodes
        assert len(viz_data.nodes) > 0
        for node in viz_data.nodes:
            assert 'id' in node
            assert 'label' in node
            assert 'x' in node
            assert 'y' in node
            assert 'type' in node
            assert 'size' in node
        
        # Check edges
        for edge in viz_data.edges:
            assert 'source' in edge
            assert 'target' in edge
            assert 'type' in edge
            assert 'confidence' in edge
    
    def test_apply_filters(self, viz_preparer):
        """Test applying filters to graph."""
        # Test node type filter
        filter_criteria = {'node_types': ['process']}
        filtered_graph = viz_preparer._apply_filters(filter_criteria)
        
        # Should only have process nodes
        for node, data in filtered_graph.nodes(data=True):
            assert data.get('node_type') == 'process'
        
        # Test confidence filter
        filter_criteria = {'min_confidence': 0.95}
        filtered_graph = viz_preparer._apply_filters(filter_criteria)
        
        # Should only have high-confidence edges
        for u, v, k, data in filtered_graph.edges(keys=True, data=True):
            assert data.get('confidence', 0) >= 0.95
    
    def test_calculate_layout(self, viz_preparer):
        """Test layout calculation."""
        # Test spring layout
        layout = viz_preparer._calculate_layout(viz_preparer.graph, 'spring')
        assert isinstance(layout, dict)
        assert len(layout) == viz_preparer.graph.number_of_nodes()
        
        for node_id, (x, y) in layout.items():
            assert isinstance(x, (int, float))
            assert isinstance(y, (int, float))
        
        # Test circular layout
        layout = viz_preparer._calculate_layout(viz_preparer.graph, 'circular')
        assert isinstance(layout, dict)
        
        # Test hierarchical layout
        layout = viz_preparer._calculate_layout(viz_preparer.graph, 'hierarchical')
        assert isinstance(layout, dict)
    
    def test_prepare_nodes_for_visualization(self, viz_preparer):
        """Test node preparation for visualization."""
        layout = {'A': (0, 0), 'B': (1, 1)}
        vis_nodes = viz_preparer._prepare_nodes_for_visualization(viz_preparer.graph, layout)
        
        assert isinstance(vis_nodes, list)
        assert len(vis_nodes) == viz_preparer.graph.number_of_nodes()
        
        for node in vis_nodes:
            assert 'id' in node
            assert 'label' in node
            assert 'x' in node
            assert 'y' in node
            assert 'type' in node
            assert 'size' in node
            assert node['size'] > 0
    
    def test_prepare_edges_for_visualization(self, viz_preparer):
        """Test edge preparation for visualization."""
        vis_edges = viz_preparer._prepare_edges_for_visualization(viz_preparer.graph)
        
        assert isinstance(vis_edges, list)
        assert len(vis_edges) == viz_preparer.graph.number_of_edges()
        
        for edge in vis_edges:
            assert 'source' in edge
            assert 'target' in edge
            assert 'type' in edge
            assert 'confidence' in edge
            assert 'width' in edge
            assert edge['width'] > 0
    
    def test_calculate_node_size(self, viz_preparer):
        """Test node size calculation."""
        size = viz_preparer._calculate_node_size(viz_preparer.graph, 'A')
        assert isinstance(size, float)
        assert size > 0
        
        # Node with higher degree should have larger size
        viz_preparer.graph.add_edge('C', 'A')
        viz_preparer.graph.add_edge('D', 'A')
        new_size = viz_preparer._calculate_node_size(viz_preparer.graph, 'A')
        assert new_size >= size
    
    def test_get_edge_color(self, viz_preparer):
        """Test edge color assignment."""
        color = viz_preparer._get_edge_color('depends_prerequisite')
        assert isinstance(color, str)
        assert color.startswith('#')
        
        # Unknown relationship type should get default color
        default_color = viz_preparer._get_edge_color('unknown_relationship')
        assert default_color == '#999999'
    
    def test_prepare_visualization_metadata(self, viz_preparer):
        """Test visualization metadata preparation."""
        metadata = viz_preparer._prepare_visualization_metadata(viz_preparer.graph)
        
        assert isinstance(metadata, dict)
        assert 'total_nodes' in metadata
        assert 'total_edges' in metadata
        assert 'node_type_counts' in metadata
        assert 'relationship_type_counts' in metadata
        assert 'is_connected' in metadata
        assert 'has_cycles' in metadata
        
        assert metadata['total_nodes'] == viz_preparer.graph.number_of_nodes()
        assert metadata['total_edges'] == viz_preparer.graph.number_of_edges()

class TestGraphQueryInterface:
    """Test graph query interface functionality."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for query testing."""
        graph = nx.MultiDiGraph()
        
        # Add nodes
        nodes = [
            ("PROC_001", {
                "label": "Safety Check", 
                "node_type": "process",
                "properties": {"domain": "safety", "criticality": "high"}
            }),
            ("PROC_002", {
                "label": "Equipment Startup", 
                "node_type": "process",
                "properties": {"domain": "operational", "criticality": "medium"}
            }),
            ("EQUIP_001", {
                "label": "Test Pump", 
                "node_type": "equipment",
                "properties": {"type": "pump", "location": "Building A"}
            })
        ]
        
        for node_id, data in nodes:
            graph.add_node(node_id, **data)
        
        # Add edges
        edges = [
            ("PROC_001", "PROC_002", {
                "relationship_type": "depends_prerequisite", 
                "weight": 0.8, 
                "confidence": 0.9
            }),
            ("PROC_002", "EQUIP_001", {
                "relationship_type": "uses_equipment", 
                "weight": 0.7, 
                "confidence": 0.8
            })
        ]
        
        for source, target, data in edges:
            graph.add_edge(source, target, **data)
        
        return graph
    
    @pytest.fixture
    def query_interface(self, sample_graph):
        """Create a GraphQueryInterface instance."""
        return GraphQueryInterface(sample_graph)
    
    def test_find_nodes_query(self, query_interface):
        """Test finding nodes by criteria."""
        query = {
            'type': 'find_nodes',
            'criteria': {
                'node_type': 'process'
            }
        }
        
        results = query_interface.query_relationships(query)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # All results should be process nodes
        for result in results:
            assert result['type'] == 'process'
            assert 'node_id' in result
            assert 'label' in result
            assert 'properties' in result
    
    def test_find_nodes_with_properties(self, query_interface):
        """Test finding nodes by property criteria."""
        query = {
            'type': 'find_nodes',
            'criteria': {
                'node_type': 'process',
                'properties': {
                    'domain': 'safety'
                }
            }
        }
        
        results = query_interface.query_relationships(query)
        
        assert isinstance(results, list)
        
        # All results should be safety processes
        for result in results:
            assert result['type'] == 'process'
            assert result['properties']['domain'] == 'safety'
    
    def test_find_paths_query(self, query_interface):
        """Test finding paths between nodes."""
        query = {
            'type': 'find_paths',
            'source': 'PROC_001',
            'target': 'EQUIP_001',
            'max_length': 5
        }
        
        results = query_interface.query_relationships(query)
        
        assert isinstance(results, list)
        
        for result in results:
            assert 'path' in result
            assert 'length' in result
            assert 'edges' in result
            assert result['path'][0] == 'PROC_001'
            assert result['path'][-1] == 'EQUIP_001'
    
    def test_find_neighbors_query(self, query_interface):
        """Test finding neighbors of a node."""
        query = {
            'type': 'find_neighbors',
            'node_id': 'PROC_001',
            'direction': 'out',
            'max_distance': 1
        }
        
        results = query_interface.query_relationships(query)
        
        assert isinstance(results, list)
        
        for result in results:
            assert 'node_id' in result
            assert 'label' in result
            assert 'type' in result
            assert 'relationship_to_source' in result
    
    def test_find_neighbors_both_directions(self, query_interface):
        """Test finding neighbors in both directions."""
        query = {
            'type': 'find_neighbors',
            'node_id': 'PROC_002',
            'direction': 'both',
            'max_distance': 1
        }
        
        results = query_interface.query_relationships(query)
        
        assert isinstance(results, list)
        # Should find both predecessors and successors
        assert len(results) >= 1
    
    def test_analyze_subgraph_query(self, query_interface):
        """Test subgraph analysis."""
        query = {
            'type': 'analyze_subgraph',
            'node_ids': ['PROC_001', 'PROC_002']
        }
        
        results = query_interface.query_relationships(query)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        analysis = results[0]
        assert 'node_count' in analysis
        assert 'edge_count' in analysis
        assert 'is_connected' in analysis
        assert 'has_cycles' in analysis
        assert 'density' in analysis
        assert 'node_types' in analysis
        assert 'relationship_types' in analysis
    
    def test_matches_criteria(self, query_interface):
        """Test criteria matching functionality."""
        node_data = {
            'node_type': 'process',
            'properties': {
                'domain': 'safety',
                'criticality': 'high'
            }
        }
        
        # Should match exact criteria
        criteria1 = {'node_type': 'process'}
        assert query_interface._matches_criteria(node_data, criteria1)
        
        # Should match property criteria
        criteria2 = {
            'node_type': 'process',
            'properties': {'domain': 'safety'}
        }
        assert query_interface._matches_criteria(node_data, criteria2)
        
        # Should not match wrong criteria
        criteria3 = {'node_type': 'equipment'}
        assert not query_interface._matches_criteria(node_data, criteria3)
    
    def test_get_relationship_to_source(self, query_interface):
        """Test getting relationship information."""
        # Test outgoing relationship
        rel_info = query_interface._get_relationship_to_source('PROC_001', 'PROC_002')
        assert rel_info['direction'] == 'outgoing'
        assert rel_info['type'] == 'depends_prerequisite'
        assert rel_info['confidence'] == 0.9
        
        # Test incoming relationship
        rel_info = query_interface._get_relationship_to_source('PROC_002', 'PROC_001')
        assert rel_info['direction'] == 'incoming'
        
        # Test no relationship
        rel_info = query_interface._get_relationship_to_source('PROC_001', 'EQUIP_001')
        assert rel_info['direction'] == 'none'
    
    def test_invalid_query_type(self, query_interface):
        """Test handling of invalid query types."""
        query = {
            'type': 'invalid_query_type'
        }
        
        with pytest.raises(ValueError):
            query_interface.query_relationships(query)

class TestKnowledgeGraphIntegration:
    """Integration tests for the complete knowledge graph system."""
    
    def test_end_to_end_knowledge_graph_workflow(self):
        """Test complete knowledge graph workflow."""
        # Create sample data
        knowledge_items = [
            Mock(spec=KnowledgeItem, 
                 process_id="PROC_001", name="Safety Check", description="Safety procedure",
                 domain="safety", hierarchy_level=3, knowledge_type="procedural",
                 confidence_score=0.8, completeness_index=0.9, criticality_level="high",
                 source_quality="high", created_at=datetime.now(), updated_at=datetime.now())
        ]
        
        process_dependencies = [
            ProcessDependency(
                source_process_id="PROC_001", target_process_id="PROC_002",
                dependency_type="prerequisite", strength=0.8, conditions={},
                confidence=0.9, evidence=["safety first"]
            )
        ]
        
        # Build graph
        builder = KnowledgeGraphBuilder()
        graph = builder.build_knowledge_graph(
            knowledge_items=knowledge_items,
            process_dependencies=process_dependencies,
            equipment_correlations=[],
            skill_links=[],
            compliance_connections=[]
        )
        
        # Analyze dependencies
        traversal_engine = GraphTraversalEngine(graph)
        analysis = traversal_engine.analyze_dependencies()
        
        assert isinstance(analysis, DependencyAnalysis)
        
        # Prepare visualization
        viz_preparer = GraphVisualizationPreparer(graph)
        viz_data = viz_preparer.prepare_visualization_data()
        
        assert isinstance(viz_data, GraphVisualizationData)
        
        # Query graph
        query_interface = GraphQueryInterface(graph)
        results = query_interface.query_relationships({
            'type': 'find_nodes',
            'criteria': {'node_type': 'process'}
        })
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    def test_graph_performance_with_large_dataset(self):
        """Test graph performance with larger dataset."""
        # Create larger dataset
        knowledge_items = []
        for i in range(100):
            item = Mock(spec=KnowledgeItem)
            item.process_id = f"PROC_{i:03d}"
            item.name = f"Process {i}"
            item.description = f"Description for process {i}"
            item.domain = "operational"
            item.hierarchy_level = (i % 4) + 1
            item.knowledge_type = "procedural"
            item.confidence_score = 0.8
            item.completeness_index = 0.9
            item.criticality_level = "medium"
            item.source_quality = "medium"
            item.created_at = datetime.now()
            item.updated_at = datetime.now()
            knowledge_items.append(item)
        
        # Create dependencies
        dependencies = []
        for i in range(50):
            dep = ProcessDependency(
                source_process_id=f"PROC_{i:03d}",
                target_process_id=f"PROC_{(i+1):03d}",
                dependency_type="downstream",
                strength=0.7,
                conditions={},
                confidence=0.8,
                evidence=[]
            )
            dependencies.append(dep)
        
        # Build and analyze graph
        builder = KnowledgeGraphBuilder()
        graph = builder.build_knowledge_graph(
            knowledge_items=knowledge_items,
            process_dependencies=dependencies,
            equipment_correlations=[],
            skill_links=[],
            compliance_connections=[]
        )
        
        # Should handle large graph efficiently
        assert graph.number_of_nodes() == 100
        assert graph.number_of_edges() == 50
        
        # Test traversal performance
        traversal_engine = GraphTraversalEngine(graph)
        analysis = traversal_engine.analyze_dependencies()
        
        # Should complete analysis without errors
        assert isinstance(analysis, DependencyAnalysis)

if __name__ == "__main__":
    pytest.main([__file__])