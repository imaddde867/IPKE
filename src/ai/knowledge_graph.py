"""
Knowledge Graph Generation and Query System for EXPLAINIUM

This module implements knowledge graph generation from extracted relationships,
graph traversal algorithms for dependency analysis, graph visualization data
preparation, and complex relationship query interfaces.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import networkx as nx
from datetime import datetime

from src.logging_config import get_logger
from src.ai.relationship_mapper import (
    ProcessDependency, EquipmentMaintenanceCorrelation,
    SkillFunctionLink, ComplianceProcedureConnection
)
from src.database.models import (
    KnowledgeItem, WorkflowDependency, KnowledgeRelationship,
    Equipment, Procedure, Personnel
)

logger = get_logger(__name__)

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    node_type: str  # process, equipment, person, regulation, skill
    properties: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]
    weight: float
    confidence: float

@dataclass
class GraphPath:
    """Represents a path through the knowledge graph."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    path_type: str
    total_weight: float
    confidence: float

@dataclass
class DependencyAnalysis:
    """Results of dependency analysis."""
    critical_paths: List[GraphPath]
    bottlenecks: List[GraphNode]
    circular_dependencies: List[List[str]]
    dependency_depth: Dict[str, int]
    impact_analysis: Dict[str, Dict[str, Any]]

@dataclass
class GraphVisualizationData:
    """Data structure for graph visualization."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    layout: Dict[str, Any]
    metadata: Dict[str, Any]

class KnowledgeGraphBuilder:
    """Builds knowledge graphs from extracted relationships."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_registry = {}
        self.edge_registry = {}
        self.node_types = {
            'process': {'color': '#4CAF50', 'shape': 'box'},
            'equipment': {'color': '#FF9800', 'shape': 'ellipse'},
            'person': {'color': '#2196F3', 'shape': 'circle'},
            'regulation': {'color': '#F44336', 'shape': 'diamond'},
            'skill': {'color': '#9C27B0', 'shape': 'triangle'}
        }
    
    def build_knowledge_graph(self, 
                            knowledge_items: List[KnowledgeItem],
                            process_dependencies: List[ProcessDependency],
                            equipment_correlations: List[EquipmentMaintenanceCorrelation],
                            skill_links: List[SkillFunctionLink],
                            compliance_connections: List[ComplianceProcedureConnection],
                            equipment_list: List[Equipment] = None,
                            personnel_list: List[Personnel] = None) -> nx.MultiDiGraph:
        """
        Build a comprehensive knowledge graph from all relationship types.
        
        Args:
            knowledge_items: List of knowledge items
            process_dependencies: List of process dependencies
            equipment_correlations: List of equipment-maintenance correlations
            skill_links: List of skill-function links
            compliance_connections: List of compliance-procedure connections
            equipment_list: Optional list of equipment
            personnel_list: Optional list of personnel
            
        Returns:
            NetworkX MultiDiGraph representing the knowledge graph
        """
        try:
            # Clear existing graph
            self.graph.clear()
            self.node_registry.clear()
            self.edge_registry.clear()
            
            # Add nodes from knowledge items
            self._add_knowledge_item_nodes(knowledge_items)
            
            # Add equipment nodes
            if equipment_list:
                self._add_equipment_nodes(equipment_list)
            
            # Add personnel nodes
            if personnel_list:
                self._add_personnel_nodes(personnel_list)
            
            # Add process dependency edges
            self._add_process_dependency_edges(process_dependencies)
            
            # Add equipment correlation edges
            self._add_equipment_correlation_edges(equipment_correlations)
            
            # Add skill-function edges
            self._add_skill_function_edges(skill_links)
            
            # Add compliance connection edges
            self._add_compliance_connection_edges(compliance_connections)
            
            # Validate and optimize graph
            self._validate_graph()
            self._optimize_graph()
            
            logger.info(f"Built knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
            return self.graph
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            raise
    
    def _add_knowledge_item_nodes(self, knowledge_items: List[KnowledgeItem]):
        """Add knowledge item nodes to the graph."""
        for item in knowledge_items:
            node = GraphNode(
                id=item.process_id,
                label=item.name,
                node_type='process',
                properties={
                    'description': item.description or '',
                    'domain': item.domain,
                    'hierarchy_level': item.hierarchy_level,
                    'knowledge_type': item.knowledge_type,
                    'confidence_score': item.confidence_score,
                    'completeness_index': item.completeness_index,
                    'criticality_level': item.criticality_level
                },
                metadata={
                    'created_at': item.created_at.isoformat() if item.created_at else None,
                    'updated_at': item.updated_at.isoformat() if item.updated_at else None,
                    'source_quality': item.source_quality
                }
            )
            
            self._add_node_to_graph(node)
    
    def _add_equipment_nodes(self, equipment_list: List[Equipment]):
        """Add equipment nodes to the graph."""
        for equipment in equipment_list:
            node = GraphNode(
                id=f"EQUIP_{equipment.id}",
                label=equipment.name,
                node_type='equipment',
                properties={
                    'type': equipment.type or '',
                    'specifications': equipment.specifications or {},
                    'location': equipment.location or '',
                    'confidence': equipment.confidence or 0.0
                },
                metadata={
                    'equipment_id': equipment.id,
                    'document_id': equipment.document_id
                }
            )
            
            self._add_node_to_graph(node)
    
    def _add_personnel_nodes(self, personnel_list: List[Personnel]):
        """Add personnel nodes to the graph."""
        for person in personnel_list:
            node = GraphNode(
                id=f"PERSON_{person.id}",
                label=person.name,
                node_type='person',
                properties={
                    'role': person.role,
                    'responsibilities': person.responsibilities or '',
                    'certifications': person.certifications or [],
                    'confidence': person.confidence or 0.0
                },
                metadata={
                    'person_id': person.id,
                    'document_id': person.document_id
                }
            )
            
            self._add_node_to_graph(node)
    
    def _add_node_to_graph(self, node: GraphNode):
        """Add a node to the NetworkX graph."""
        self.node_registry[node.id] = node
        
        # Prepare node attributes for NetworkX
        node_attrs = {
            'label': node.label,
            'node_type': node.node_type,
            'properties': node.properties,
            'metadata': node.metadata
        }
        
        # Add visual attributes
        if node.node_type in self.node_types:
            node_attrs.update(self.node_types[node.node_type])
        
        self.graph.add_node(node.id, **node_attrs)
    
    def _add_process_dependency_edges(self, dependencies: List[ProcessDependency]):
        """Add process dependency edges to the graph."""
        for dep in dependencies:
            if dep.source_process_id in self.node_registry and dep.target_process_id in self.node_registry:
                edge = GraphEdge(
                    source_id=dep.source_process_id,
                    target_id=dep.target_process_id,
                    relationship_type=f"depends_{dep.dependency_type}",
                    properties={
                        'dependency_type': dep.dependency_type,
                        'conditions': dep.conditions,
                        'evidence': dep.evidence
                    },
                    weight=dep.strength,
                    confidence=dep.confidence
                )
                
                self._add_edge_to_graph(edge)
    
    def _add_equipment_correlation_edges(self, correlations: List[EquipmentMaintenanceCorrelation]):
        """Add equipment-maintenance correlation edges to the graph."""
        for corr in correlations:
            equipment_id = f"EQUIP_{corr.equipment_id}"
            
            # Find related process nodes (maintenance procedures)
            maintenance_nodes = [node_id for node_id, node in self.node_registry.items() 
                               if node.node_type == 'process' and 'maintenance' in node.label.lower()]
            
            for maintenance_node in maintenance_nodes:
                edge = GraphEdge(
                    source_id=equipment_id,
                    target_id=maintenance_node,
                    relationship_type=f"requires_{corr.correlation_type}_maintenance",
                    properties={
                        'correlation_type': corr.correlation_type,
                        'frequency': corr.frequency,
                        'conditions': corr.conditions,
                        'evidence': corr.evidence
                    },
                    weight=corr.confidence,
                    confidence=corr.confidence
                )
                
                self._add_edge_to_graph(edge)
    
    def _add_skill_function_edges(self, skill_links: List[SkillFunctionLink]):
        """Add skill-function edges to the graph."""
        for link in skill_links:
            # Create skill node if it doesn't exist
            skill_id = f"SKILL_{link.skill_name.replace(' ', '_')}"
            if skill_id not in self.node_registry:
                skill_node = GraphNode(
                    id=skill_id,
                    label=link.skill_name,
                    node_type='skill',
                    properties={
                        'proficiency_level': link.proficiency_level,
                        'criticality': link.criticality
                    },
                    metadata={}
                )
                self._add_node_to_graph(skill_node)
            
            # Find function node
            function_nodes = [node_id for node_id, node in self.node_registry.items() 
                            if node.node_type == 'process' and link.function_name.lower() in node.label.lower()]
            
            for function_node in function_nodes:
                edge = GraphEdge(
                    source_id=skill_id,
                    target_id=function_node,
                    relationship_type="required_for_function",
                    properties={
                        'proficiency_level': link.proficiency_level,
                        'criticality': link.criticality,
                        'evidence': link.evidence
                    },
                    weight=link.confidence,
                    confidence=link.confidence
                )
                
                self._add_edge_to_graph(edge)
    
    def _add_compliance_connection_edges(self, connections: List[ComplianceProcedureConnection]):
        """Add compliance-procedure connection edges to the graph."""
        for conn in connections:
            # Create regulation node if it doesn't exist
            regulation_id = f"REG_{conn.regulation_reference.replace(' ', '_').replace('.', '_')}"
            if regulation_id not in self.node_registry:
                regulation_node = GraphNode(
                    id=regulation_id,
                    label=conn.regulation_reference,
                    node_type='regulation',
                    properties={
                        'compliance_type': conn.compliance_type,
                        'risk_level': conn.risk_level
                    },
                    metadata={}
                )
                self._add_node_to_graph(regulation_node)
            
            # Find procedure node
            procedure_id = f"PROC_{conn.procedure_id}"
            if procedure_id in self.node_registry:
                edge = GraphEdge(
                    source_id=regulation_id,
                    target_id=procedure_id,
                    relationship_type=f"requires_{conn.compliance_type}_compliance",
                    properties={
                        'compliance_type': conn.compliance_type,
                        'risk_level': conn.risk_level,
                        'evidence': conn.evidence
                    },
                    weight=conn.confidence,
                    confidence=conn.confidence
                )
                
                self._add_edge_to_graph(edge)
    
    def _add_edge_to_graph(self, edge: GraphEdge):
        """Add an edge to the NetworkX graph."""
        edge_key = f"{edge.source_id}_{edge.target_id}_{edge.relationship_type}"
        self.edge_registry[edge_key] = edge
        
        # Prepare edge attributes for NetworkX
        edge_attrs = {
            'relationship_type': edge.relationship_type,
            'properties': edge.properties,
            'weight': edge.weight,
            'confidence': edge.confidence
        }
        
        self.graph.add_edge(edge.source_id, edge.target_id, key=edge_key, **edge_attrs)
    
    def _validate_graph(self):
        """Validate the constructed graph."""
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(self.graph))
        if isolated_nodes:
            logger.warning(f"Found {len(isolated_nodes)} isolated nodes in knowledge graph")
        
        # Check for self-loops
        self_loops = list(nx.nodes_with_selfloops(self.graph))
        if self_loops:
            logger.warning(f"Found {len(self_loops)} self-loops in knowledge graph")
        
        # Check graph connectivity
        if not nx.is_weakly_connected(self.graph):
            components = list(nx.weakly_connected_components(self.graph))
            logger.info(f"Knowledge graph has {len(components)} weakly connected components")
    
    def _optimize_graph(self):
        """Optimize the graph structure."""
        # Remove duplicate edges with lower confidence
        edges_to_remove = []
        
        for source, target in self.graph.edges():
            edge_data = self.graph.get_edge_data(source, target)
            if len(edge_data) > 1:  # Multiple edges between same nodes
                # Keep only the edge with highest confidence
                best_edge = max(edge_data.items(), key=lambda x: x[1].get('confidence', 0))
                for key, data in edge_data.items():
                    if key != best_edge[0]:
                        edges_to_remove.append((source, target, key))
        
        for source, target, key in edges_to_remove:
            self.graph.remove_edge(source, target, key)
        
        logger.info(f"Removed {len(edges_to_remove)} duplicate edges during optimization")

class GraphTraversalEngine:
    """Implements graph traversal algorithms for dependency analysis."""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
    
    def analyze_dependencies(self, target_node: str = None) -> DependencyAnalysis:
        """
        Perform comprehensive dependency analysis on the knowledge graph.
        
        Args:
            target_node: Optional specific node to analyze
            
        Returns:
            DependencyAnalysis object with analysis results
        """
        try:
            # Find critical paths
            critical_paths = self._find_critical_paths(target_node)
            
            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks()
            
            # Detect circular dependencies
            circular_dependencies = self._detect_circular_dependencies()
            
            # Calculate dependency depth
            dependency_depth = self._calculate_dependency_depth()
            
            # Perform impact analysis
            impact_analysis = self._perform_impact_analysis()
            
            return DependencyAnalysis(
                critical_paths=critical_paths,
                bottlenecks=bottlenecks,
                circular_dependencies=circular_dependencies,
                dependency_depth=dependency_depth,
                impact_analysis=impact_analysis
            )
            
        except Exception as e:
            logger.error(f"Error analyzing dependencies: {e}")
            raise
    
    def _find_critical_paths(self, target_node: str = None) -> List[GraphPath]:
        """Find critical paths in the dependency graph."""
        critical_paths = []
        
        try:
            # If target node specified, find paths to it
            if target_node and target_node in self.graph:
                # Find all paths to target node
                for source in self.graph.nodes():
                    if source != target_node:
                        try:
                            paths = list(nx.all_simple_paths(self.graph, source, target_node, cutoff=10))
                            for path in paths:
                                graph_path = self._create_graph_path(path, 'critical')
                                if graph_path.confidence > 0.5:  # Only high-confidence paths
                                    critical_paths.append(graph_path)
                        except nx.NetworkXNoPath:
                            continue
            else:
                # Find longest paths in the graph
                try:
                    # Use topological sort to find longest paths
                    if nx.is_directed_acyclic_graph(self.graph):
                        topo_order = list(nx.topological_sort(self.graph))
                        longest_paths = self._find_longest_paths(topo_order)
                        
                        for path in longest_paths:
                            graph_path = self._create_graph_path(path, 'critical')
                            critical_paths.append(graph_path)
                except nx.NetworkXError:
                    # Graph has cycles, use alternative approach
                    logger.warning("Graph contains cycles, using alternative critical path detection")
                    critical_paths = self._find_critical_paths_with_cycles()
            
            # Sort by confidence and weight
            critical_paths.sort(key=lambda x: (x.confidence, x.total_weight), reverse=True)
            
            return critical_paths[:10]  # Return top 10 critical paths
            
        except Exception as e:
            logger.error(f"Error finding critical paths: {e}")
            return []
    
    def _create_graph_path(self, node_path: List[str], path_type: str) -> GraphPath:
        """Create a GraphPath object from a list of node IDs."""
        nodes = []
        edges = []
        total_weight = 0.0
        confidence_scores = []
        
        # Get nodes
        for node_id in node_path:
            if node_id in self.graph:
                node_data = self.graph.nodes[node_id]
                node = GraphNode(
                    id=node_id,
                    label=node_data.get('label', node_id),
                    node_type=node_data.get('node_type', 'unknown'),
                    properties=node_data.get('properties', {}),
                    metadata=node_data.get('metadata', {})
                )
                nodes.append(node)
        
        # Get edges
        for i in range(len(node_path) - 1):
            source = node_path[i]
            target = node_path[i + 1]
            
            if self.graph.has_edge(source, target):
                edge_data = self.graph.get_edge_data(source, target)
                # Get the best edge if multiple exist
                best_edge_data = max(edge_data.values(), key=lambda x: x.get('confidence', 0))
                
                edge = GraphEdge(
                    source_id=source,
                    target_id=target,
                    relationship_type=best_edge_data.get('relationship_type', 'unknown'),
                    properties=best_edge_data.get('properties', {}),
                    weight=best_edge_data.get('weight', 0.0),
                    confidence=best_edge_data.get('confidence', 0.0)
                )
                edges.append(edge)
                total_weight += edge.weight
                confidence_scores.append(edge.confidence)
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return GraphPath(
            nodes=nodes,
            edges=edges,
            path_type=path_type,
            total_weight=total_weight,
            confidence=avg_confidence
        )
    
    def _find_longest_paths(self, topo_order: List[str]) -> List[List[str]]:
        """Find longest paths using topological ordering."""
        # Initialize distances
        distances = {node: 0 for node in self.graph.nodes()}
        predecessors = {node: None for node in self.graph.nodes()}
        
        # Calculate longest paths
        for node in topo_order:
            for successor in self.graph.successors(node):
                edge_data = self.graph.get_edge_data(node, successor)
                best_edge = max(edge_data.values(), key=lambda x: x.get('weight', 0))
                weight = best_edge.get('weight', 0)
                
                if distances[node] + weight > distances[successor]:
                    distances[successor] = distances[node] + weight
                    predecessors[successor] = node
        
        # Reconstruct longest paths
        longest_paths = []
        max_distance = max(distances.values())
        
        # Find all nodes with maximum distance
        end_nodes = [node for node, dist in distances.items() if dist == max_distance]
        
        for end_node in end_nodes:
            path = []
            current = end_node
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()
            longest_paths.append(path)
        
        return longest_paths
    
    def _find_critical_paths_with_cycles(self) -> List[GraphPath]:
        """Find critical paths when graph contains cycles."""
        critical_paths = []
        
        # Use strongly connected components to handle cycles
        sccs = list(nx.strongly_connected_components(self.graph))
        
        # Create condensed graph
        condensed = nx.condensation(self.graph, sccs)
        
        # Find longest paths in condensed graph
        try:
            topo_order = list(nx.topological_sort(condensed))
            longest_paths = self._find_longest_paths_condensed(condensed, topo_order, sccs)
            
            for path in longest_paths:
                graph_path = self._create_graph_path(path, 'critical')
                critical_paths.append(graph_path)
        except Exception as e:
            logger.error(f"Error finding critical paths with cycles: {e}")
        
        return critical_paths
    
    def _find_longest_paths_condensed(self, condensed_graph: nx.DiGraph, 
                                    topo_order: List[int], sccs: List[Set[str]]) -> List[List[str]]:
        """Find longest paths in condensed graph and expand back to original nodes."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated handling of SCCs
        longest_paths = []
        
        if topo_order:
            # Take the longest topological path
            path = topo_order
            expanded_path = []
            
            for scc_id in path:
                # Take one representative node from each SCC
                scc_nodes = list(sccs[scc_id])
                if scc_nodes:
                    expanded_path.append(scc_nodes[0])
            
            longest_paths.append(expanded_path)
        
        return longest_paths
    
    def _identify_bottlenecks(self) -> List[GraphNode]:
        """Identify bottleneck nodes in the graph."""
        bottlenecks = []
        
        try:
            # Calculate betweenness centrality
            centrality = nx.betweenness_centrality(self.graph)
            
            # Calculate in-degree and out-degree
            in_degrees = dict(self.graph.in_degree())
            out_degrees = dict(self.graph.out_degree())
            
            # Identify bottlenecks based on high centrality and degree
            for node_id, centrality_score in centrality.items():
                in_degree = in_degrees.get(node_id, 0)
                out_degree = out_degrees.get(node_id, 0)
                
                # High centrality or high degree indicates potential bottleneck
                if (centrality_score > 0.1 or in_degree > 3 or out_degree > 3):
                    node_data = self.graph.nodes[node_id]
                    bottleneck = GraphNode(
                        id=node_id,
                        label=node_data.get('label', node_id),
                        node_type=node_data.get('node_type', 'unknown'),
                        properties={
                            **node_data.get('properties', {}),
                            'centrality_score': centrality_score,
                            'in_degree': in_degree,
                            'out_degree': out_degree
                        },
                        metadata=node_data.get('metadata', {})
                    )
                    bottlenecks.append(bottleneck)
            
            # Sort by centrality score
            bottlenecks.sort(key=lambda x: x.properties.get('centrality_score', 0), reverse=True)
            
            return bottlenecks[:10]  # Return top 10 bottlenecks
            
        except Exception as e:
            logger.error(f"Error identifying bottlenecks: {e}")
            return []
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the graph."""
        circular_dependencies = []
        
        try:
            # Find strongly connected components with more than one node
            sccs = list(nx.strongly_connected_components(self.graph))
            
            for scc in sccs:
                if len(scc) > 1:
                    # This is a circular dependency
                    circular_dependencies.append(list(scc))
            
            # Also find simple cycles
            try:
                simple_cycles = list(nx.simple_cycles(self.graph))
                for cycle in simple_cycles:
                    if len(cycle) > 1 and cycle not in circular_dependencies:
                        circular_dependencies.append(cycle)
            except Exception as e:
                logger.warning(f"Error finding simple cycles: {e}")
            
            return circular_dependencies
            
        except Exception as e:
            logger.error(f"Error detecting circular dependencies: {e}")
            return []
    
    def _calculate_dependency_depth(self) -> Dict[str, int]:
        """Calculate dependency depth for each node."""
        dependency_depth = {}
        
        try:
            # Calculate shortest path lengths from all nodes
            for node in self.graph.nodes():
                # Find maximum depth from this node
                try:
                    lengths = nx.single_source_shortest_path_length(self.graph, node)
                    max_depth = max(lengths.values()) if lengths else 0
                    dependency_depth[node] = max_depth
                except Exception:
                    dependency_depth[node] = 0
            
            return dependency_depth
            
        except Exception as e:
            logger.error(f"Error calculating dependency depth: {e}")
            return {}
    
    def _perform_impact_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Perform impact analysis for each node."""
        impact_analysis = {}
        
        try:
            for node in self.graph.nodes():
                # Calculate various impact metrics
                successors = list(self.graph.successors(node))
                predecessors = list(self.graph.predecessors(node))
                
                # Calculate downstream impact (how many nodes this affects)
                downstream_nodes = set()
                try:
                    for successor in successors:
                        reachable = nx.descendants(self.graph, successor)
                        downstream_nodes.update(reachable)
                except Exception:
                    pass
                
                # Calculate upstream dependencies (how many nodes affect this)
                upstream_nodes = set()
                try:
                    for predecessor in predecessors:
                        ancestors = nx.ancestors(self.graph, predecessor)
                        upstream_nodes.update(ancestors)
                except Exception:
                    pass
                
                node_data = self.graph.nodes[node]
                impact_analysis[node] = {
                    'direct_successors': len(successors),
                    'direct_predecessors': len(predecessors),
                    'downstream_impact': len(downstream_nodes),
                    'upstream_dependencies': len(upstream_nodes),
                    'node_type': node_data.get('node_type', 'unknown'),
                    'criticality': self._assess_node_criticality(node, len(downstream_nodes), len(upstream_nodes))
                }
            
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Error performing impact analysis: {e}")
            return {}
    
    def _assess_node_criticality(self, node: str, downstream_count: int, upstream_count: int) -> str:
        """Assess the criticality of a node based on its impact."""
        # High downstream impact or high upstream dependencies indicate criticality
        if downstream_count > 10 or upstream_count > 10:
            return 'critical'
        elif downstream_count > 5 or upstream_count > 5:
            return 'high'
        elif downstream_count > 2 or upstream_count > 2:
            return 'medium'
        else:
            return 'low'

class GraphVisualizationPreparer:
    """Prepares graph data for visualization."""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
    
    def prepare_visualization_data(self, layout_algorithm: str = 'spring',
                                 filter_criteria: Dict[str, Any] = None) -> GraphVisualizationData:
        """
        Prepare graph data for visualization.
        
        Args:
            layout_algorithm: Layout algorithm to use ('spring', 'circular', 'hierarchical')
            filter_criteria: Optional criteria to filter nodes/edges
            
        Returns:
            GraphVisualizationData object
        """
        try:
            # Apply filters if specified
            filtered_graph = self._apply_filters(filter_criteria) if filter_criteria else self.graph
            
            # Calculate layout
            layout = self._calculate_layout(filtered_graph, layout_algorithm)
            
            # Prepare nodes for visualization
            vis_nodes = self._prepare_nodes_for_visualization(filtered_graph, layout)
            
            # Prepare edges for visualization
            vis_edges = self._prepare_edges_for_visualization(filtered_graph)
            
            # Prepare metadata
            metadata = self._prepare_visualization_metadata(filtered_graph)
            
            return GraphVisualizationData(
                nodes=vis_nodes,
                edges=vis_edges,
                layout={'algorithm': layout_algorithm, 'positions': layout},
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error preparing visualization data: {e}")
            raise
    
    def _apply_filters(self, filter_criteria: Dict[str, Any]) -> nx.MultiDiGraph:
        """Apply filters to the graph."""
        filtered_graph = self.graph.copy()
        
        # Filter by node type
        if 'node_types' in filter_criteria:
            allowed_types = filter_criteria['node_types']
            nodes_to_remove = [node for node, data in filtered_graph.nodes(data=True)
                             if data.get('node_type') not in allowed_types]
            filtered_graph.remove_nodes_from(nodes_to_remove)
        
        # Filter by confidence threshold
        if 'min_confidence' in filter_criteria:
            min_confidence = filter_criteria['min_confidence']
            edges_to_remove = [(u, v, k) for u, v, k, data in filtered_graph.edges(keys=True, data=True)
                             if data.get('confidence', 0) < min_confidence]
            filtered_graph.remove_edges_from(edges_to_remove)
        
        # Filter by relationship type
        if 'relationship_types' in filter_criteria:
            allowed_types = filter_criteria['relationship_types']
            edges_to_remove = [(u, v, k) for u, v, k, data in filtered_graph.edges(keys=True, data=True)
                             if data.get('relationship_type') not in allowed_types]
            filtered_graph.remove_edges_from(edges_to_remove)
        
        return filtered_graph
    
    def _calculate_layout(self, graph: nx.MultiDiGraph, algorithm: str) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions for visualization."""
        if algorithm == 'spring':
            return nx.spring_layout(graph, k=1, iterations=50)
        elif algorithm == 'circular':
            return nx.circular_layout(graph)
        elif algorithm == 'hierarchical':
            return self._hierarchical_layout(graph)
        else:
            # Default to spring layout
            return nx.spring_layout(graph)
    
    def _hierarchical_layout(self, graph: nx.MultiDiGraph) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout based on node hierarchy levels."""
        positions = {}
        
        # Group nodes by hierarchy level
        level_groups = defaultdict(list)
        for node, data in graph.nodes(data=True):
            level = data.get('properties', {}).get('hierarchy_level', 0)
            level_groups[level].append(node)
        
        # Position nodes by level
        y_spacing = 1.0
        x_spacing = 1.0
        
        for level, nodes in level_groups.items():
            y = level * y_spacing
            for i, node in enumerate(nodes):
                x = (i - len(nodes) / 2) * x_spacing
                positions[node] = (x, y)
        
        return positions
    
    def _prepare_nodes_for_visualization(self, graph: nx.MultiDiGraph, 
                                       layout: Dict[str, Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Prepare nodes for visualization."""
        vis_nodes = []
        
        for node, data in graph.nodes(data=True):
            x, y = layout.get(node, (0, 0))
            
            vis_node = {
                'id': node,
                'label': data.get('label', node),
                'x': x,
                'y': y,
                'type': data.get('node_type', 'unknown'),
                'color': data.get('color', '#cccccc'),
                'shape': data.get('shape', 'circle'),
                'size': self._calculate_node_size(graph, node),
                'properties': data.get('properties', {}),
                'metadata': data.get('metadata', {})
            }
            
            vis_nodes.append(vis_node)
        
        return vis_nodes
    
    def _calculate_node_size(self, graph: nx.MultiDiGraph, node: str) -> float:
        """Calculate node size based on its importance."""
        # Base size
        size = 10.0
        
        # Increase size based on degree
        degree = graph.degree(node)
        size += min(degree * 2, 20)  # Cap at 20 additional units
        
        # Increase size based on node properties
        node_data = graph.nodes[node]
        properties = node_data.get('properties', {})
        
        if properties.get('criticality_level') == 'critical':
            size += 10
        elif properties.get('criticality_level') == 'high':
            size += 5
        
        return size
    
    def _prepare_edges_for_visualization(self, graph: nx.MultiDiGraph) -> List[Dict[str, Any]]:
        """Prepare edges for visualization."""
        vis_edges = []
        
        for source, target, key, data in graph.edges(keys=True, data=True):
            vis_edge = {
                'id': key,
                'source': source,
                'target': target,
                'type': data.get('relationship_type', 'unknown'),
                'weight': data.get('weight', 1.0),
                'confidence': data.get('confidence', 0.0),
                'color': self._get_edge_color(data.get('relationship_type', 'unknown')),
                'width': max(1, data.get('confidence', 0.0) * 5),  # Width based on confidence
                'properties': data.get('properties', {})
            }
            
            vis_edges.append(vis_edge)
        
        return vis_edges
    
    def _get_edge_color(self, relationship_type: str) -> str:
        """Get color for edge based on relationship type."""
        color_map = {
            'depends_prerequisite': '#FF5722',
            'depends_parallel': '#4CAF50',
            'depends_downstream': '#2196F3',
            'depends_conditional': '#FF9800',
            'requires_preventive_maintenance': '#9C27B0',
            'requires_corrective_maintenance': '#F44336',
            'required_for_function': '#607D8B',
            'requires_mandatory_compliance': '#E91E63'
        }
        
        return color_map.get(relationship_type, '#999999')
    
    def _prepare_visualization_metadata(self, graph: nx.MultiDiGraph) -> Dict[str, Any]:
        """Prepare metadata for visualization."""
        node_types = defaultdict(int)
        relationship_types = defaultdict(int)
        
        for node, data in graph.nodes(data=True):
            node_types[data.get('node_type', 'unknown')] += 1
        
        for source, target, key, data in graph.edges(keys=True, data=True):
            relationship_types[data.get('relationship_type', 'unknown')] += 1
        
        return {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'node_type_counts': dict(node_types),
            'relationship_type_counts': dict(relationship_types),
            'is_connected': nx.is_weakly_connected(graph),
            'has_cycles': not nx.is_directed_acyclic_graph(graph)
        }

class GraphQueryInterface:
    """Interface for complex relationship queries on the knowledge graph."""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
    
    def query_relationships(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute complex relationship queries on the knowledge graph.
        
        Args:
            query: Query specification dictionary
            
        Returns:
            List of query results
        """
        try:
            query_type = query.get('type', 'find_nodes')
            
            if query_type == 'find_nodes':
                return self._find_nodes(query)
            elif query_type == 'find_paths':
                return self._find_paths(query)
            elif query_type == 'find_neighbors':
                return self._find_neighbors(query)
            elif query_type == 'analyze_subgraph':
                return self._analyze_subgraph(query)
            else:
                raise ValueError(f"Unknown query type: {query_type}")
                
        except Exception as e:
            logger.error(f"Error executing graph query: {e}")
            raise
    
    def _find_nodes(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find nodes matching specified criteria."""
        results = []
        
        criteria = query.get('criteria', {})
        
        for node, data in self.graph.nodes(data=True):
            if self._matches_criteria(data, criteria):
                results.append({
                    'node_id': node,
                    'label': data.get('label', node),
                    'type': data.get('node_type', 'unknown'),
                    'properties': data.get('properties', {}),
                    'metadata': data.get('metadata', {})
                })
        
        return results
    
    def _find_paths(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find paths between nodes."""
        results = []
        
        source = query.get('source')
        target = query.get('target')
        max_length = query.get('max_length', 10)
        
        if source and target and source in self.graph and target in self.graph:
            try:
                paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
                
                for path in paths:
                    path_info = {
                        'path': path,
                        'length': len(path) - 1,
                        'edges': []
                    }
                    
                    # Get edge information
                    for i in range(len(path) - 1):
                        edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                        if edge_data:
                            best_edge = max(edge_data.values(), key=lambda x: x.get('confidence', 0))
                            path_info['edges'].append({
                                'source': path[i],
                                'target': path[i + 1],
                                'type': best_edge.get('relationship_type', 'unknown'),
                                'confidence': best_edge.get('confidence', 0.0)
                            })
                    
                    results.append(path_info)
                    
            except nx.NetworkXNoPath:
                pass
        
        return results
    
    def _find_neighbors(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find neighbors of specified nodes."""
        results = []
        
        node_id = query.get('node_id')
        direction = query.get('direction', 'both')  # 'in', 'out', 'both'
        max_distance = query.get('max_distance', 1)
        
        if node_id and node_id in self.graph:
            neighbors = set()
            
            if direction in ['out', 'both']:
                # Get outgoing neighbors
                for distance in range(1, max_distance + 1):
                    if distance == 1:
                        neighbors.update(self.graph.successors(node_id))
                    else:
                        # Get neighbors at specific distance
                        current_neighbors = set()
                        for neighbor in neighbors:
                            current_neighbors.update(self.graph.successors(neighbor))
                        neighbors.update(current_neighbors)
            
            if direction in ['in', 'both']:
                # Get incoming neighbors
                for distance in range(1, max_distance + 1):
                    if distance == 1:
                        neighbors.update(self.graph.predecessors(node_id))
                    else:
                        # Get neighbors at specific distance
                        current_neighbors = set()
                        for neighbor in neighbors:
                            current_neighbors.update(self.graph.predecessors(neighbor))
                        neighbors.update(current_neighbors)
            
            # Prepare results
            for neighbor in neighbors:
                node_data = self.graph.nodes[neighbor]
                results.append({
                    'node_id': neighbor,
                    'label': node_data.get('label', neighbor),
                    'type': node_data.get('node_type', 'unknown'),
                    'properties': node_data.get('properties', {}),
                    'relationship_to_source': self._get_relationship_to_source(node_id, neighbor)
                })
        
        return results
    
    def _analyze_subgraph(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze a subgraph based on specified criteria."""
        results = []
        
        node_ids = query.get('node_ids', [])
        
        if node_ids:
            # Create subgraph
            subgraph = self.graph.subgraph(node_ids)
            
            # Analyze subgraph properties
            analysis = {
                'node_count': subgraph.number_of_nodes(),
                'edge_count': subgraph.number_of_edges(),
                'is_connected': nx.is_weakly_connected(subgraph),
                'has_cycles': not nx.is_directed_acyclic_graph(subgraph),
                'density': nx.density(subgraph),
                'node_types': defaultdict(int),
                'relationship_types': defaultdict(int)
            }
            
            # Count node types
            for node, data in subgraph.nodes(data=True):
                analysis['node_types'][data.get('node_type', 'unknown')] += 1
            
            # Count relationship types
            for source, target, key, data in subgraph.edges(keys=True, data=True):
                analysis['relationship_types'][data.get('relationship_type', 'unknown')] += 1
            
            results.append(analysis)
        
        return results
    
    def _matches_criteria(self, node_data: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if node data matches specified criteria."""
        for key, value in criteria.items():
            if key == 'node_type':
                if node_data.get('node_type') != value:
                    return False
            elif key == 'properties':
                properties = node_data.get('properties', {})
                for prop_key, prop_value in value.items():
                    if properties.get(prop_key) != prop_value:
                        return False
            elif key in node_data:
                if node_data[key] != value:
                    return False
        
        return True
    
    def _get_relationship_to_source(self, source: str, target: str) -> Dict[str, Any]:
        """Get relationship information between source and target nodes."""
        if self.graph.has_edge(source, target):
            edge_data = self.graph.get_edge_data(source, target)
            best_edge = max(edge_data.values(), key=lambda x: x.get('confidence', 0))
            return {
                'direction': 'outgoing',
                'type': best_edge.get('relationship_type', 'unknown'),
                'confidence': best_edge.get('confidence', 0.0)
            }
        elif self.graph.has_edge(target, source):
            edge_data = self.graph.get_edge_data(target, source)
            best_edge = max(edge_data.values(), key=lambda x: x.get('confidence', 0))
            return {
                'direction': 'incoming',
                'type': best_edge.get('relationship_type', 'unknown'),
                'confidence': best_edge.get('confidence', 0.0)
            }
        else:
            return {'direction': 'none', 'type': 'unknown', 'confidence': 0.0}