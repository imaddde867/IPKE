"""
EXPLAINIUM - Advanced Knowledge Engine

A sophisticated, AI-powered knowledge processing system that extracts deep,
meaningful insights from company documents and tacit knowledge using local models.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
from datetime import datetime

# Core AI libraries
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np

# Local model support
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    Llama = None

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Internal imports
from src.logging_config import get_logger
from src.core.config import AIConfig

logger = get_logger(__name__)


@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph"""
    id: str
    type: str  # concept, person, process, system, requirement
    name: str
    description: str
    content: str
    metadata: Dict[str, Any]
    confidence: float
    created_at: datetime
    updated_at: datetime
    embeddings: Optional[np.ndarray] = None


@dataclass
class KnowledgeEdge:
    """Represents a relationship between knowledge nodes"""
    source_id: str
    target_id: str
    relationship_type: str  # depends_on, implements, requires, etc.
    strength: float
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class ExtractedInsight:
    """Represents a deep insight extracted from documents"""
    title: str
    description: str
    insight_type: str  # workflow, process, decision, compliance, risk
    confidence: float
    supporting_evidence: List[str]
    implications: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


class Neo4jLiteGraph:
    """Lightweight in-memory graph database for knowledge representation"""
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.node_types = set()
        self.relationship_types = set()
    
    def add_node(self, node: KnowledgeNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.node_types.add(node.type)
    
    def add_edge(self, edge: KnowledgeEdge) -> None:
        """Add an edge to the graph"""
        if edge.source_id in self.nodes and edge.target_id in self.nodes:
            self.edges.append(edge)
            self.relationship_types.add(edge.relationship_type)
    
    def get_nodes_by_type(self, node_type: str) -> List[KnowledgeNode]:
        """Get all nodes of a specific type"""
        return [node for node in self.nodes.values() if node.type == node_type]
    
    def get_connected_nodes(self, node_id: str) -> List[Tuple[KnowledgeNode, KnowledgeEdge]]:
        """Get all nodes connected to a specific node"""
        connected = []
        for edge in self.edges:
            if edge.source_id == node_id:
                target_node = self.nodes.get(edge.target_id)
                if target_node:
                    connected.append((target_node, edge))
            elif edge.target_id == node_id:
                source_node = self.nodes.get(edge.source_id)
                if source_node:
                    connected.append((source_node, edge))
        return connected
    
    def export_cytoscape(self) -> Dict[str, Any]:
        """Export graph for Cytoscape visualization"""
        nodes = []
        edges = []
        
        for node in self.nodes.values():
            nodes.append({
                'data': {
                    'id': node.id,
                    'label': node.name,
                    'type': node.type,
                    'description': node.description,
                    'confidence': node.confidence
                }
            })
        
        for edge in self.edges:
            edges.append({
                'data': {
                    'id': f"{edge.source_id}_{edge.target_id}_{edge.relationship_type}",
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'label': edge.relationship_type,
                    'strength': edge.strength
                }
            })
        
        return {
            'nodes': nodes,
            'edges': edges
        }


class ModelManager:
    """Manages local AI models with memory optimization for M4 Mac"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.models = {}
        self.model_cache = {}
        self.current_memory_usage = 0
        self.max_memory = 14 * 1024 * 1024 * 1024  # 14GB (leaving 2GB for system)
        
    async def load_quantized_model(self, model_name: str, quantization: str = "4bit") -> Any:
        """Load a quantized model optimized for M4 Mac"""
        cache_key = f"{model_name}_{quantization}"
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        try:
            if model_name == "mistral-7b" and LLAMA_AVAILABLE:
                model_path = f"{self.config.llm_path}/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
                if not Path(model_path).exists():
                    raise FileNotFoundError(f"Model not found: {model_path}")
                
                model = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,  # Use Apple Metal
                    n_ctx=4096,
                    n_batch=512,
                    n_threads=8,  # Optimized for M4
                    verbose=False
                )
                
                self.model_cache[cache_key] = model
                logger.info(f"Loaded quantized Mistral model: {model_name}")
                return model
                
            elif model_name == "phi-2":
                # Load Microsoft Phi-2 model
                model = self._load_phi_model()
                self.model_cache[cache_key] = model
                return model
                
            else:
                raise ValueError(f"Unsupported model: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _load_phi_model(self) -> Any:
        """Load Microsoft Phi-2 model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_name = "microsoft/phi-2"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            return {"model": model, "tokenizer": tokenizer}
            
        except Exception as e:
            logger.error(f"Failed to load Phi-2 model: {e}")
            raise
    
    async def load_embedding_model(self, model_name: str) -> Any:
        """Load embedding model for semantic search"""
        cache_key = f"embedding_{model_name}"
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        try:
            if model_name == "bge-small":
                model = SentenceTransformer('BAAI/bge-small-en-v1.5')
                self.model_cache[cache_key] = model
                return model
            else:
                raise ValueError(f"Unsupported embedding model: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        return self.current_memory_usage
    
    def cleanup_unused_models(self) -> None:
        """Clean up unused models to free memory"""
        # Implementation for model cleanup
        pass


class AdvancedKnowledgeEngine:
    """Advanced knowledge extraction engine using local AI models"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.knowledge_graph = Neo4jLiteGraph()
        self.llm = None
        self.embedder = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize the knowledge engine with models"""
        try:
            # Load core models
            self.llm = await self.model_manager.load_quantized_model(
                "mistral-7b", 
                self.config.quantization
            )
            self.embedder = await self.model_manager.load_embedding_model(
                self.config.embedding_model
            )
            self.initialized = True
            logger.info("Advanced Knowledge Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge engine: {e}")
            raise
    
    async def extract_deep_knowledge(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-pass knowledge extraction strategy"""
        if not self.initialized:
            await self.initialize()
        
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        # Multi-pass extraction
        results = {
            'extraction_timestamp': datetime.now().isoformat(),
            'document_id': document.get('id'),
            'passes': {}
        }
        
        # First pass: Extract key concepts, entities, relationships
        logger.info("Starting first pass: Concept extraction")
        concepts = await self._extract_concepts(content, metadata)
        results['passes']['concepts'] = concepts
        
        # Second pass: Identify workflows, processes, decision trees
        logger.info("Starting second pass: Workflow extraction")
        workflows = await self._extract_workflows(content, metadata)
        results['passes']['workflows'] = workflows
        
        # Third pass: Infer tacit knowledge, patterns, best practices
        logger.info("Starting third pass: Tacit knowledge extraction")
        tacit_knowledge = await self._extract_tacit_knowledge(content, metadata)
        results['passes']['tacit_knowledge'] = tacit_knowledge
        
        # Fourth pass: Cross-reference with existing knowledge base
        logger.info("Starting fourth pass: Knowledge integration")
        integration = await self._integrate_knowledge(results['passes'], metadata)
        results['passes']['integration'] = integration
        
        # Build knowledge graph
        await self._build_knowledge_graph(results['passes'], metadata)
        
        return results
    
    async def _extract_concepts(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key concepts and entities from content"""
        try:
            # Use LLM to extract concepts
            prompt = f"""
            Analyze the following business document and extract key concepts, entities, and relationships.
            Focus on identifying:
            1. Business processes and procedures
            2. Key stakeholders and roles
            3. Systems and technologies mentioned
            4. Compliance requirements
            5. Risk factors
            
            Document content:
            {content[:2000]}  # Limit content for prompt
            
            Provide your analysis in JSON format with the following structure:
            {{
                "concepts": [
                    {{
                        "name": "concept_name",
                        "type": "process|person|system|requirement|risk",
                        "description": "detailed description",
                        "confidence": 0.95
                    }}
                ],
                "relationships": [
                    {{
                        "source": "concept1",
                        "target": "concept2",
                        "type": "relationship_type",
                        "description": "relationship description"
                    }}
                ]
            }}
            """
            
            if isinstance(self.llm, Llama):
                response = self.llm(
                    prompt,
                    max_tokens=2048,
                    temperature=0.1,
                    stop=["</s>", "\n\n"]
                )
                result_text = response['choices'][0]['text']
            else:
                # Fallback for other model types
                result_text = "{}"
            
            # Parse JSON response
            try:
                concepts_data = json.loads(result_text)
                return concepts_data
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
                return {"concepts": [], "relationships": []}
            
        except Exception as e:
            logger.error(f"Error in concept extraction: {e}")
            return {"concepts": [], "relationships": []}
    
    async def _extract_workflows(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract workflows and business processes"""
        try:
            prompt = f"""
            Analyze the following business document to identify workflows, processes, and operational procedures.
            
            Extract:
            1. Step-by-step processes
            2. Decision points and criteria
            3. Roles and responsibilities
            4. Dependencies and prerequisites
            5. Success criteria and outcomes
            
            Document content:
            {content[:2000]}
            
            Provide analysis in JSON format:
            {{
                "workflows": [
                    {{
                        "name": "workflow_name",
                        "description": "workflow description",
                        "steps": ["step1", "step2"],
                        "decision_points": ["decision1"],
                        "roles": ["role1", "role2"],
                        "dependencies": ["dependency1"],
                        "success_criteria": ["criteria1"]
                    }}
                ]
            }}
            """
            
            if isinstance(self.llm, Llama):
                response = self.llm(
                    prompt,
                    max_tokens=2048,
                    temperature=0.1,
                    stop=["</s>", "\n\n"]
                )
                result_text = response['choices'][0]['text']
            else:
                result_text = "{}"
            
            try:
                workflows_data = json.loads(result_text)
                return workflows_data
            except json.JSONDecodeError:
                return {"workflows": []}
            
        except Exception as e:
            logger.error(f"Error in workflow extraction: {e}")
            return {"workflows": []}
    
    async def _extract_tacit_knowledge(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tacit knowledge and implicit patterns"""
        try:
            prompt = f"""
            Analyze the following business document to identify tacit knowledge, implicit patterns, and unstated assumptions.
            
            Look for:
            1. Implicit workflows and procedures
            2. Unstated business rules
            3. Organizational culture indicators
            4. Informal communication patterns
            5. Hidden dependencies and risks
            6. Best practices and lessons learned
            
            Document content:
            {content[:2000]}
            
            Provide analysis in JSON format:
            {{
                "tacit_knowledge": [
                    {{
                        "type": "implicit_workflow|business_rule|culture_indicator|best_practice",
                        "description": "detailed description",
                        "confidence": 0.85,
                        "evidence": "supporting evidence from text"
                    }}
                ]
            }}
            """
            
            if isinstance(self.llm, Llama):
                response = self.llm(
                    prompt,
                    max_tokens=2048,
                    temperature=0.1,
                    stop=["</s>", "\n\n"]
                )
                result_text = response['choices'][0]['text']
            else:
                result_text = "{}"
            
            try:
                tacit_data = json.loads(result_text)
                return tacit_data
            except json.JSONDecodeError:
                return {"tacit_knowledge": []}
            
        except Exception as e:
            logger.error(f"Error in tacit knowledge extraction: {e}")
            return {"tacit_knowledge": []}
    
    async def _integrate_knowledge(self, extracted_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate extracted knowledge with existing knowledge base"""
        try:
            # Generate embeddings for semantic search
            content_text = str(extracted_data)
            embeddings = self.embedder.encode(content_text)
            
            # Find similar existing knowledge
            similar_nodes = await self._find_similar_knowledge(embeddings)
            
            # Identify conflicts and gaps
            conflicts = await self._identify_conflicts(extracted_data, similar_nodes)
            gaps = await self._identify_knowledge_gaps(extracted_data, similar_nodes)
            
            return {
                "similar_knowledge": similar_nodes,
                "conflicts": conflicts,
                "gaps": gaps,
                "integration_score": 0.85
            }
            
        except Exception as e:
            logger.error(f"Error in knowledge integration: {e}")
            return {"similar_knowledge": [], "conflicts": [], "gaps": []}
    
    async def _find_similar_knowledge(self, embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """Find similar knowledge using semantic search"""
        similar_nodes = []
        
        for node in self.knowledge_graph.nodes.values():
            if node.embeddings is not None:
                similarity = np.dot(embeddings, node.embeddings) / (
                    np.linalg.norm(embeddings) * np.linalg.norm(node.embeddings)
                )
                if similarity > 0.7:  # Similarity threshold
                    similar_nodes.append({
                        "node_id": node.id,
                        "name": node.name,
                        "type": node.type,
                        "similarity": float(similarity)
                    })
        
        # Sort by similarity
        similar_nodes.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_nodes[:10]  # Return top 10
    
    async def _identify_conflicts(self, new_data: Dict[str, Any], existing_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify conflicts between new and existing knowledge"""
        conflicts = []
        
        # Simple conflict detection based on concept names
        new_concepts = set()
        if 'concepts' in new_data:
            for concept in new_data['concepts']:
                new_concepts.add(concept.get('name', '').lower())
        
        for node in existing_nodes:
            if node['name'].lower() in new_concepts:
                conflicts.append({
                    "type": "concept_overlap",
                    "existing_node": node,
                    "description": f"Concept '{node['name']}' already exists"
                })
        
        return conflicts
    
    async def _identify_knowledge_gaps(self, new_data: Dict[str, Any], existing_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify gaps in knowledge coverage"""
        gaps = []
        
        # Identify missing knowledge areas
        covered_areas = set()
        for node in existing_nodes:
            covered_areas.add(node['type'])
        
        expected_areas = {'process', 'person', 'system', 'requirement', 'risk'}
        missing_areas = expected_areas - covered_areas
        
        for area in missing_areas:
            gaps.append({
                "type": "missing_knowledge_area",
                "area": area,
                "description": f"No knowledge found for area: {area}"
            })
        
        return gaps
    
    async def _build_knowledge_graph(self, extracted_data: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Build interconnected knowledge representation"""
        try:
            # Create nodes from extracted concepts
            if 'concepts' in extracted_data:
                for concept in extracted_data['concepts']:
                    node_id = hashlib.md5(concept['name'].encode()).hexdigest()
                    
                    node = KnowledgeNode(
                        id=node_id,
                        type=concept['type'],
                        name=concept['name'],
                        description=concept['description'],
                        content=concept['name'],
                        metadata=concept,
                        confidence=concept.get('confidence', 0.8),
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    # Generate embeddings
                    if self.embedder:
                        node.embeddings = self.embedder.encode(concept['name'])
                    
                    self.knowledge_graph.add_node(node)
            
            # Create edges from relationships
            if 'relationships' in extracted_data:
                for rel in extracted_data['relationships']:
                    source_id = hashlib.md5(rel['source'].encode()).hexdigest()
                    target_id = hashlib.md5(rel['target'].encode()).hexdigest()
                    
                    edge = KnowledgeEdge(
                        source_id=source_id,
                        target_id=target_id,
                        relationship_type=rel['type'],
                        strength=0.8,
                        metadata=rel,
                        created_at=datetime.now()
                    )
                    
                    self.knowledge_graph.add_edge(edge)
            
            logger.info(f"Built knowledge graph with {len(self.knowledge_graph.nodes)} nodes and {len(self.knowledge_graph.edges)} edges")
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
    
    async def extract_operational_intelligence(self, content: str) -> Dict[str, Any]:
        """Extract operational intelligence using specialized prompts"""
        try:
            prompt = f"""
            Analyze the following business content to extract operational intelligence:
            
            Focus on:
            1. Standard Operating Procedures (SOPs)
            2. Decision criteria and business rules
            3. Compliance requirements and regulations
            4. Risk factors and mitigation strategies
            5. Performance metrics and KPIs
            6. Unstated assumptions and tribal knowledge
            
            Content:
            {content[:2000]}
            
            Provide detailed analysis in JSON format:
            {{
                "sops": ["sop1", "sop2"],
                "decision_criteria": ["criteria1", "criteria2"],
                "compliance_requirements": ["req1", "req2"],
                "risk_factors": ["risk1", "risk2"],
                "performance_metrics": ["metric1", "metric2"],
                "assumptions": ["assumption1", "assumption2"]
            }}
            """
            
            if isinstance(self.llm, Llama):
                response = self.llm(
                    prompt,
                    max_tokens=2048,
                    temperature=0.1,
                    stop=["</s>", "\n\n"]
                )
                result_text = response['choices'][0]['text']
            else:
                result_text = "{}"
            
            try:
                operational_data = json.loads(result_text)
                return operational_data
            except json.JSONDecodeError:
                return {}
            
        except Exception as e:
            logger.error(f"Error extracting operational intelligence: {e}")
            return {}
    
    def get_knowledge_graph(self) -> Neo4jLiteGraph:
        """Get the current knowledge graph"""
        return self.knowledge_graph
    
    def export_knowledge(self, format: str = "json") -> Dict[str, Any]:
        """Export knowledge in various formats"""
        if format == "cytoscape":
            return self.knowledge_graph.export_cytoscape()
        elif format == "json":
        return {
                "nodes": [node.__dict__ for node in self.knowledge_graph.nodes.values()],
                "edges": [edge.__dict__ for edge in self.knowledge_graph.edges],
                "metadata": {
                    "total_nodes": len(self.knowledge_graph.nodes),
                    "total_edges": len(self.knowledge_graph.edges),
                    "node_types": list(self.knowledge_graph.node_types),
                    "relationship_types": list(self.knowledge_graph.relationship_types)
                }
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")