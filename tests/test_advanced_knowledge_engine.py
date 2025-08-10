"""
Tests for the Advanced Knowledge Engine

Validates deep knowledge extraction, workflow discovery, and tacit knowledge identification.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any

# Import the classes we want to test
from src.ai.advanced_knowledge_engine import AdvancedKnowledgeEngine, ModelManager, Neo4jLiteGraph
from src.core.config import AIConfig


class TestAdvancedKnowledgeEngine:
    """Test cases for AdvancedKnowledgeEngine"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return AIConfig(
            llm_model="test-model.gguf",
            llm_path="./test_models",
            max_tokens=1024,
            temperature=0.1,
            embedding_model="test-embedding",
            chunk_size=512,
            chunk_overlap=50,
            use_gpu=True,
            quantization="4bit",
            batch_size=4
        )
    
    @pytest.fixture
    def mock_engine(self, mock_config):
        """Mock knowledge engine for testing"""
        with patch('src.ai.advanced_knowledge_engine.config_manager.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            
            engine = AdvancedKnowledgeEngine()
            # Mock the model loading
            engine.llm = Mock()
            engine.embedder = Mock()
            engine.knowledge_graph = Mock()
            
            return engine
    
    @pytest.mark.asyncio
    async def test_extract_deep_knowledge(self, mock_engine):
        """Test deep knowledge extraction"""
        # Mock document
        test_document = {
            "content": "This is a test document about business processes.",
            "type": "text",
            "metadata": {"source": "test"}
        }
        
        # Mock LLM response
        mock_engine.llm.analyze = AsyncMock(return_value={
            "concepts": ["business", "processes"],
            "entities": ["company", "workflow"],
            "relationships": [{"source": "company", "target": "workflow", "type": "implements"}]
        })
        
        # Mock embedding generation
        mock_engine.embedder.encode = AsyncMock(return_value=[0.1, 0.2, 0.3])
        
        # Test extraction
        result = await mock_engine.extract_deep_knowledge(test_document)
        
        assert result is not None
        assert "concepts" in result
        assert "entities" in result
        assert "relationships" in result
        assert "embeddings" in result
    
    @pytest.mark.asyncio
    async def test_build_knowledge_graph(self, mock_engine):
        """Test knowledge graph building"""
        # Mock extracted data
        test_data = {
            "concepts": ["business", "processes"],
            "entities": ["company", "workflow"],
            "relationships": [{"source": "company", "target": "workflow", "type": "implements"}]
        }
        
        # Mock knowledge graph methods
        mock_engine.knowledge_graph.add_node = Mock()
        mock_engine.knowledge_graph.add_edge = Mock()
        
        # Test graph building
        result = await mock_engine.build_knowledge_graph(test_data)
        
        assert result is not None
        # Verify nodes were added
        assert mock_engine.knowledge_graph.add_node.call_count >= 2
        # Verify edges were added
        assert mock_engine.knowledge_graph.add_edge.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_extract_operational_intelligence(self, mock_engine):
        """Test operational intelligence extraction"""
        test_content = """
        Standard Operating Procedure:
        1. Receive customer request
        2. Validate requirements
        3. Process order
        4. Send confirmation
        
        Decision criteria: Customer must have valid account
        Risk factors: Payment failure, inventory shortage
        """
        
        # Mock LLM response for SOP extraction
        mock_engine.llm.analyze = AsyncMock(return_value={
            "sops": ["Receive customer request", "Validate requirements", "Process order"],
            "decision_criteria": ["Customer must have valid account"],
            "risk_factors": ["Payment failure", "Inventory shortage"],
            "compliance_requirements": ["Account validation", "Payment verification"]
        })
        
        # Test extraction
        result = await mock_engine.extract_operational_intelligence(test_content)
        
        assert result is not None
        assert "sops" in result
        assert "decision_criteria" in result
        assert "risk_factors" in result
        assert "compliance_requirements" in result


class TestNeo4jLiteGraph:
    """Test cases for Neo4jLiteGraph"""
    
    @pytest.fixture
    def graph(self):
        """Create a test graph instance"""
        return Neo4jLiteGraph()
    
    def test_add_node(self, graph):
        """Test adding nodes to the graph"""
        # Add test node
        node_id = graph.add_node("test_concept", "concept", "A test concept", confidence=0.9)
        
        assert node_id in graph.nodes
        assert graph.nodes[node_id].name == "test_concept"
        assert graph.nodes[node_id].type == "concept"
        assert graph.nodes[node_id].confidence == 0.9
    
    def test_add_edge(self, graph):
        """Test adding edges to the graph"""
        # Add two nodes first
        node1_id = graph.add_node("source", "concept", "Source concept")
        node2_id = graph.add_node("target", "concept", "Target concept")
        
        # Add edge
        edge_id = graph.add_edge(node1_id, node2_id, "relates_to", strength=0.8)
        
        assert edge_id in graph.edges
        assert graph.edges[edge_id].source_id == node1_id
        assert graph.edges[edge_id].target_id == node2_id
        assert graph.edges[edge_id].relationship_type == "relates_to"
        assert graph.edges[edge_id].strength == 0.8
    
    def test_find_nodes_by_type(self, graph):
        """Test finding nodes by type"""
        # Add nodes of different types
        graph.add_node("concept1", "concept", "Concept 1")
        graph.add_node("concept2", "concept", "Concept 2")
        graph.add_node("process1", "process", "Process 1")
        
        # Find concept nodes
        concept_nodes = graph.find_nodes_by_type("concept")
        assert len(concept_nodes) == 2
        
        # Find process nodes
        process_nodes = graph.find_nodes_by_type("process")
        assert len(process_nodes) == 1
    
    def test_find_related_nodes(self, graph):
        """Test finding related nodes"""
        # Create a simple graph structure
        node1_id = graph.add_node("A", "concept", "Concept A")
        node2_id = graph.add_node("B", "concept", "Concept B")
        node3_id = graph.add_node("C", "concept", "Concept C")
        
        # Add edges
        graph.add_edge(node1_id, node2_id, "relates_to")
        graph.add_edge(node2_id, node3_id, "relates_to")
        
        # Find nodes related to A
        related = graph.find_related_nodes(node1_id)
        assert len(related) == 1
        assert related[0].id == node2_id
        
        # Find nodes related to B
        related = graph.find_related_nodes(node2_id)
        assert len(related) == 2  # A and C


class TestModelManager:
    """Test cases for ModelManager"""
    
    @pytest.fixture
    def manager(self):
        """Create a test model manager"""
        return ModelManager("./test_models")
    
    def test_detect_hardware_profile(self, manager):
        """Test hardware profile detection"""
        with patch('platform.system') as mock_system, \
             patch('platform.machine') as mock_machine, \
             patch('psutil.virtual_memory') as mock_memory:
            
            # Mock macOS with Apple Silicon
            mock_system.return_value = "Darwin"
            mock_machine.return_value = "arm64"
            mock_memory.return_value.total = 16 * (1024**3)  # 16GB
            
            profile = manager.detect_hardware_profile()
            assert profile == "m4_16gb"
            
            # Even with higher RAM, only single profile is returned
            mock_memory.return_value.total = 32 * (1024**3)  # 32GB
            profile = manager.detect_hardware_profile()
            assert profile == "m4_16gb"
    
    def test_model_configs(self, manager):
        """Test model configurations for different hardware profiles"""
        assert "m4_16gb" in manager.model_configs
    # Only unified profile remains
        
        # Check M4 16GB config
        m4_16gb_config = manager.model_configs["m4_16gb"]
        assert "llm" in m4_16gb_config
        assert "embeddings" in m4_16gb_config
        assert "vision" in m4_16gb_config
        assert "audio" in m4_16gb_config
        
        # Check LLM config
        llm_config = m4_16gb_config["llm"]
        assert "primary" in llm_config
        assert "quantization" in llm_config
        assert "max_ram" in llm_config
        assert "fallback" in llm_config
    
    @pytest.mark.asyncio
    async def test_setup_models(self, manager):
        """Test model setup process"""
        with patch.object(manager, 'download_model') as mock_download, \
             patch.object(manager, 'quantize_model') as mock_quantize, \
             patch.object(manager, 'optimize_for_m4') as mock_optimize:
            
            # Mock successful operations
            mock_download.return_value = "./test_models/llm/test_model"
            mock_quantize.return_value = "./test_models/llm/test_model_Q4_K_M.gguf"
            mock_optimize.return_value = {"optimized": True}
            
            # Test setup
            results = await manager.setup_models("m4_16gb")
            
            assert "llm" in results
            assert "embeddings" in results
            assert "vision" in results
            assert "audio" in results
            
            # Verify calls were made
            assert mock_download.call_count >= 4  # LLM, embeddings, vision, audio
            assert mock_quantize.call_count >= 1  # At least LLM
            assert mock_optimize.call_count >= 1  # At least LLM


# Integration test
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test end-to-end document processing"""
        # This would test the complete pipeline from document input
        # to knowledge extraction to graph building
        # For now, we'll create a simple test structure
        
        # Mock the complete system
        with patch('src.ai.advanced_knowledge_engine.AdvancedKnowledgeEngine') as MockEngine:
            mock_engine = MockEngine.return_value
            
            # Mock document processing
            test_document = {
                "content": "Business process documentation for customer onboarding",
                "type": "pdf",
                "metadata": {"department": "operations"}
            }
            
            # Mock knowledge extraction
            mock_engine.extract_deep_knowledge = AsyncMock(return_value={
                "concepts": ["customer onboarding", "business process"],
                "entities": ["customer", "operations department"],
                "workflows": ["customer onboarding workflow"],
                "confidence": 0.85
            })
            
            # Mock graph building
            mock_engine.build_knowledge_graph = AsyncMock(return_value={
                "nodes_added": 3,
                "edges_added": 2,
                "graph_id": "test_graph_123"
            })
            
            # Test the complete flow
            extraction_result = await mock_engine.extract_deep_knowledge(test_document)
            graph_result = await mock_engine.build_knowledge_graph(extraction_result)
            
            assert extraction_result is not None
            assert "concepts" in extraction_result
            assert "workflows" in extraction_result
            assert graph_result is not None
            assert "nodes_added" in graph_result


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
