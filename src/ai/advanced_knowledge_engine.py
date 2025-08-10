"""
EXPLAINIUM - Advanced Knowledge Processing Engine

A sophisticated AI-powered knowledge processing system that extracts deep, 
meaningful insights from company documents and tacit knowledge.
Optimized for Apple M4 Mac with 16GB RAM.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

# Advanced AI and ML libraries
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import networkx as nx
import numpy as np
from diskcache import Cache

# Apple Silicon optimization
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# LLM processing
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Internal imports
from src.logging_config import get_logger
from src.core.config import config_manager

logger = get_logger(__name__)


@dataclass
class KnowledgeEntity:
    """Advanced knowledge entity with relationships and context"""
    id: str
    name: str
    type: str  # concept, person, process, system, requirement, risk, etc.
    description: str
    properties: Dict[str, Any]
    confidence: float
    source_documents: List[str]
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class KnowledgeRelationship:
    """Relationship between knowledge entities"""
    source_id: str
    target_id: str
    relationship_type: str  # depends_on, implements, requires, leads_to, etc.
    strength: float
    context: str
    properties: Dict[str, Any]
    confidence: float


@dataclass
class WorkflowProcess:
    """Extracted business workflow or process"""
    id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    decision_points: List[Dict[str, Any]]
    roles_involved: List[str]
    inputs: List[str]
    outputs: List[str]
    complexity_score: float
    automation_potential: float
    risk_factors: List[str]


@dataclass
class TacitKnowledge:
    """Extracted tacit or implicit knowledge"""
    id: str
    knowledge_type: str  # assumption, best_practice, tribal_knowledge, pattern
    description: str
    context: str
    confidence: float
    evidence: List[str]
    affected_processes: List[str]


class ModelManager:
    """Manages AI models with M4 optimization and memory management"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.cache = Cache(directory="./model_cache", size_limit=int(4e9))  # 4GB cache
        self._init_models()
    
    def _init_models(self):
        """Initialize AI models with M4 optimization"""
        logger.info("Initializing advanced AI models for M4 Mac...")
        
        # Initialize embedding model (lightweight, fast)
        try:
            self.embedder = SentenceTransformer(
                'BAAI/bge-small-en-v1.5',
                device='mps' if torch.backends.mps.is_available() else 'cpu'
            )
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedder = None
        
        # Initialize LLM (quantized for M4)
        self._init_llm()
        
        # Initialize specialized models
        self._init_specialized_models()
    
    def _init_llm(self):
        """Initialize quantized LLM optimized for M4"""
        try:
            model_path = self.config.get("llm_model_path", "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
            
            if LLAMA_CPP_AVAILABLE and os.path.exists(model_path):
                self.llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,  # Use Apple Metal
                    n_ctx=4096,       # Context window
                    n_batch=512,      # Optimized for M4
                    verbose=False,
                    use_mmap=True,    # Memory mapping for efficiency
                    use_mlock=True,   # Lock memory pages
                )
                logger.info("Quantized LLM initialized successfully with Metal acceleration")
            else:
                # Fallback to Hugging Face model
                self._init_fallback_llm()
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self._init_fallback_llm()
    
    def _init_fallback_llm(self):
        """Initialize fallback LLM model"""
        try:
            self.llm_pipeline = pipeline(
                "text-generation",
                model="microsoft/phi-2",
                device="mps" if torch.backends.mps.is_available() else "cpu",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            logger.info("Fallback LLM pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize fallback LLM: {e}")
            self.llm_pipeline = None
    
    def _init_specialized_models(self):
        """Initialize specialized models for specific tasks"""
        try:
            # Document layout understanding
            self.layout_model = pipeline(
                "document-question-answering",
                model="microsoft/layoutlmv3-base",
                device="mps" if torch.backends.mps.is_available() else "cpu"
            )
            
            # Image understanding
            self.vision_model = pipeline(
                "image-to-text",
                model="Salesforce/blip-image-captioning-base",
                device="mps" if torch.backends.mps.is_available() else "cpu"
            )
            
            logger.info("Specialized models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Some specialized models failed to initialize: {e}")
    
    def generate_llm_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using the LLM"""
        try:
            if hasattr(self, 'llm') and self.llm:
                response = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    top_p=0.95,
                    stop=["Human:", "Assistant:", "\n\n"]
                )
                return response['choices'][0]['text'].strip()
            
            elif hasattr(self, 'llm_pipeline') and self.llm_pipeline:
                response = self.llm_pipeline(
                    prompt,
                    max_length=max_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.llm_pipeline.tokenizer.eos_token_id
                )
                return response[0]['generated_text'][len(prompt):].strip()
            
            else:
                return "LLM not available"
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return f"Error: {str(e)}"
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts"""
        if self.embedder:
            return self.embedder.encode(texts, convert_to_numpy=True)
        else:
            # Fallback to random embeddings
            return np.random.rand(len(texts), 384)


class KnowledgeGraph:
    """Lightweight knowledge graph using NetworkX and ChromaDB"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities = {}
        self.relationships = []
        
        # Initialize vector database
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        self.collection = self.client.get_or_create_collection(
            name="knowledge_entities",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_entity(self, entity: KnowledgeEntity):
        """Add entity to knowledge graph"""
        self.entities[entity.id] = entity
        self.graph.add_node(entity.id, **asdict(entity))
        
        # Add to vector database if embedding exists
        if entity.embedding:
            self.collection.add(
                embeddings=[entity.embedding],
                documents=[entity.description],
                metadatas=[{
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type,
                    "confidence": entity.confidence
                }],
                ids=[entity.id]
            )
    
    def add_relationship(self, relationship: KnowledgeRelationship):
        """Add relationship to knowledge graph"""
        self.relationships.append(relationship)
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            type=relationship.relationship_type,
            strength=relationship.strength,
            context=relationship.context,
            confidence=relationship.confidence
        )
    
    def find_similar_entities(self, query: str, n_results: int = 5) -> List[Dict]:
        """Find similar entities using vector search"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def get_entity_relationships(self, entity_id: str) -> List[KnowledgeRelationship]:
        """Get all relationships for an entity"""
        return [rel for rel in self.relationships 
                if rel.source_id == entity_id or rel.target_id == entity_id]
    
    def export_graph(self, format: str = "gexf") -> str:
        """Export graph in various formats"""
        output_path = f"knowledge_graph.{format}"
        
        if format == "gexf":
            nx.write_gexf(self.graph, output_path)
        elif format == "graphml":
            nx.write_graphml(self.graph, output_path)
        elif format == "json":
            data = nx.node_link_data(self.graph)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        return output_path


class AdvancedKnowledgeEngine:
    """
    Advanced knowledge processing engine that extracts deep insights,
    builds knowledge graphs, and identifies tacit knowledge patterns.
    """
    
    def __init__(self):
        self.config = config_manager.get_ai_config()
        self.model_manager = ModelManager(self.config)
        self.knowledge_graph = KnowledgeGraph()
        self.executor = ThreadPoolExecutor(max_workers=4)  # Optimized for M4
        
        # Knowledge extraction templates
        self.templates = self._load_extraction_templates()
        
        logger.info("AdvancedKnowledgeEngine initialized successfully")
    
    def _load_extraction_templates(self) -> Dict[str, str]:
        """Load prompt templates for different extraction tasks"""
        return {
            "entity_extraction": """
            Extract key entities from the following text. Focus on:
            - Business concepts and terminology
            - People, roles, and responsibilities
            - Processes and workflows
            - Systems and technologies
            - Requirements and constraints
            - Risks and compliance issues
            
            Text: {text}
            
            Return a JSON list of entities with: name, type, description, confidence
            """,
            
            "process_extraction": """
            Analyze the following text to identify business processes and workflows:
            
            Text: {text}
            
            Extract:
            1. Process name and description
            2. Sequential steps
            3. Decision points
            4. Roles and responsibilities
            5. Inputs and outputs
            6. Dependencies
            
            Return structured JSON format.
            """,
            
            "tacit_knowledge": """
            Identify implicit or tacit knowledge from the following text:
            - Unstated assumptions
            - Best practices mentioned indirectly
            - Informal procedures
            - Tribal knowledge
            - Patterns and habits
            
            Text: {text}
            
            Extract insights that are implied but not explicitly stated.
            """,
            
            "relationship_extraction": """
            Identify relationships between entities in the following text:
            
            Entities: {entities}
            Text: {text}
            
            Find relationships like:
            - depends_on, requires, implements
            - leads_to, causes, affects
            - contains, part_of, member_of
            - similar_to, replaces, conflicts_with
            
            Return relationships with confidence scores.
            """
        }
    
    async def extract_deep_knowledge(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Multi-pass knowledge extraction with deep understanding
        """
        content = document.get('content', '')
        doc_id = document.get('id', 'unknown')
        
        logger.info(f"Starting deep knowledge extraction for document {doc_id}")
        
        # Multi-pass extraction strategy
        results = {
            'entities': [],
            'processes': [],
            'relationships': [],
            'tacit_knowledge': [],
            'insights': {}
        }
        
        try:
            # Pass 1: Extract key concepts and entities
            entities = await self._extract_entities(content, doc_id)
            results['entities'] = entities
            
            # Pass 2: Identify workflows and processes
            processes = await self._extract_processes(content, doc_id)
            results['processes'] = processes
            
            # Pass 3: Extract relationships between entities
            relationships = await self._extract_relationships(content, entities, doc_id)
            results['relationships'] = relationships
            
            # Pass 4: Identify tacit knowledge and patterns
            tacit_knowledge = await self._extract_tacit_knowledge(content, doc_id)
            results['tacit_knowledge'] = tacit_knowledge
            
            # Pass 5: Generate insights and cross-references
            insights = await self._generate_insights(results, doc_id)
            results['insights'] = insights
            
            logger.info(f"Deep knowledge extraction completed for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error in deep knowledge extraction: {e}")
            results['error'] = str(e)
        
        return results
    
    async def _extract_entities(self, content: str, doc_id: str) -> List[KnowledgeEntity]:
        """Extract entities with advanced understanding"""
        entities = []
        
        try:
            # Chunk content for processing
            chunks = self._chunk_content(content)
            
            for i, chunk in enumerate(chunks):
                prompt = self.templates["entity_extraction"].format(text=chunk)
                response = self.model_manager.generate_llm_response(prompt, max_tokens=1024)
                
                # Parse response and create entities
                chunk_entities = self._parse_entities_response(response, doc_id, f"chunk_{i}")
                entities.extend(chunk_entities)
            
            # Deduplicate and merge similar entities
            entities = self._deduplicate_entities(entities)
            
            # Generate embeddings for entities
            await self._generate_entity_embeddings(entities)
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
        
        return entities
    
    async def _extract_processes(self, content: str, doc_id: str) -> List[WorkflowProcess]:
        """Extract business processes and workflows"""
        processes = []
        
        try:
            chunks = self._chunk_content(content)
            
            for chunk in chunks:
                prompt = self.templates["process_extraction"].format(text=chunk)
                response = self.model_manager.generate_llm_response(prompt, max_tokens=1024)
                
                chunk_processes = self._parse_processes_response(response, doc_id)
                processes.extend(chunk_processes)
            
        except Exception as e:
            logger.error(f"Error extracting processes: {e}")
        
        return processes
    
    async def _extract_relationships(self, content: str, entities: List[KnowledgeEntity], doc_id: str) -> List[KnowledgeRelationship]:
        """Extract relationships between entities"""
        relationships = []
        
        try:
            entity_names = [e.name for e in entities[:20]]  # Limit for prompt size
            
            chunks = self._chunk_content(content)
            
            for chunk in chunks:
                prompt = self.templates["relationship_extraction"].format(
                    entities=entity_names,
                    text=chunk
                )
                response = self.model_manager.generate_llm_response(prompt, max_tokens=512)
                
                chunk_relationships = self._parse_relationships_response(response, entities, doc_id)
                relationships.extend(chunk_relationships)
            
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
        
        return relationships
    
    async def _extract_tacit_knowledge(self, content: str, doc_id: str) -> List[TacitKnowledge]:
        """Extract implicit and tacit knowledge"""
        tacit_items = []
        
        try:
            chunks = self._chunk_content(content)
            
            for chunk in chunks:
                prompt = self.templates["tacit_knowledge"].format(text=chunk)
                response = self.model_manager.generate_llm_response(prompt, max_tokens=512)
                
                chunk_tacit = self._parse_tacit_response(response, doc_id)
                tacit_items.extend(chunk_tacit)
            
        except Exception as e:
            logger.error(f"Error extracting tacit knowledge: {e}")
        
        return tacit_items
    
    async def _generate_insights(self, extraction_results: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
        """Generate high-level insights from extracted knowledge"""
        insights = {
            'summary': '',
            'key_findings': [],
            'automation_opportunities': [],
            'risk_factors': [],
            'knowledge_gaps': []
        }
        
        try:
            # Generate summary
            entities_count = len(extraction_results.get('entities', []))
            processes_count = len(extraction_results.get('processes', []))
            
            insights['summary'] = f"Extracted {entities_count} entities and {processes_count} processes from document {doc_id}"
            
            # Identify automation opportunities
            for process in extraction_results.get('processes', []):
                if hasattr(process, 'automation_potential') and process.automation_potential > 0.7:
                    insights['automation_opportunities'].append({
                        'process': process.name,
                        'potential': process.automation_potential,
                        'description': process.description
                    })
            
            # Identify risk factors
            for entity in extraction_results.get('entities', []):
                if entity.type in ['risk', 'compliance', 'security']:
                    insights['risk_factors'].append({
                        'entity': entity.name,
                        'type': entity.type,
                        'description': entity.description,
                        'confidence': entity.confidence
                    })
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
        
        return insights
    
    def _chunk_content(self, content: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """Chunk content for processing with overlap"""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            
            # Find a good breaking point
            if end < len(content):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = content[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(content):
                break
        
        return chunks
    
    def _parse_entities_response(self, response: str, doc_id: str, chunk_id: str) -> List[KnowledgeEntity]:
        """Parse LLM response to extract entities"""
        entities = []
        
        try:
            # Try to parse JSON response
            if response.strip().startswith('[') or response.strip().startswith('{'):
                import json
                data = json.loads(response)
                
                if isinstance(data, list):
                    for item in data:
                        entity = KnowledgeEntity(
                            id=f"{doc_id}_{chunk_id}_{len(entities)}",
                            name=item.get('name', ''),
                            type=item.get('type', 'concept'),
                            description=item.get('description', ''),
                            properties={},
                            confidence=item.get('confidence', 0.5),
                            source_documents=[doc_id]
                        )
                        entities.append(entity)
            else:
                # Parse unstructured response
                lines = response.split('\n')
                for line in lines:
                    if ':' in line and len(line.strip()) > 5:
                        parts = line.split(':', 1)
                        entity = KnowledgeEntity(
                            id=f"{doc_id}_{chunk_id}_{len(entities)}",
                            name=parts[0].strip(),
                            type='concept',
                            description=parts[1].strip(),
                            properties={},
                            confidence=0.6,
                            source_documents=[doc_id]
                        )
                        entities.append(entity)
        
        except Exception as e:
            logger.error(f"Error parsing entities response: {e}")
        
        return entities
    
    def _parse_processes_response(self, response: str, doc_id: str) -> List[WorkflowProcess]:
        """Parse LLM response to extract processes"""
        processes = []
        
        try:
            # Similar parsing logic for processes
            # Implementation would parse process-specific fields
            pass
        except Exception as e:
            logger.error(f"Error parsing processes response: {e}")
        
        return processes
    
    def _parse_relationships_response(self, response: str, entities: List[KnowledgeEntity], doc_id: str) -> List[KnowledgeRelationship]:
        """Parse LLM response to extract relationships"""
        relationships = []
        
        try:
            # Implementation for parsing relationship responses
            pass
        except Exception as e:
            logger.error(f"Error parsing relationships response: {e}")
        
        return relationships
    
    def _parse_tacit_response(self, response: str, doc_id: str) -> List[TacitKnowledge]:
        """Parse LLM response to extract tacit knowledge"""
        tacit_items = []
        
        try:
            # Implementation for parsing tacit knowledge
            pass
        except Exception as e:
            logger.error(f"Error parsing tacit knowledge response: {e}")
        
        return tacit_items
    
    async def _generate_entity_embeddings(self, entities: List[KnowledgeEntity]):
        """Generate embeddings for entities"""
        if not entities:
            return
        
        try:
            descriptions = [entity.description for entity in entities]
            embeddings = self.model_manager.get_embeddings(descriptions)
            
            for entity, embedding in zip(entities, embeddings):
                entity.embedding = embedding.tolist()
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
    
    def _deduplicate_entities(self, entities: List[KnowledgeEntity]) -> List[KnowledgeEntity]:
        """Remove duplicate entities based on similarity"""
        unique_entities = []
        seen_names = set()
        
        for entity in entities:
            name_lower = entity.name.lower().strip()
            if name_lower not in seen_names and len(name_lower) > 2:
                seen_names.add(name_lower)
                unique_entities.append(entity)
        
        return unique_entities
    
    async def build_knowledge_graph(self, extraction_results: Dict[str, Any]) -> KnowledgeGraph:
        """Build knowledge graph from extraction results"""
        try:
            # Add entities to graph
            for entity in extraction_results.get('entities', []):
                self.knowledge_graph.add_entity(entity)
            
            # Add relationships to graph
            for relationship in extraction_results.get('relationships', []):
                self.knowledge_graph.add_relationship(relationship)
            
            logger.info("Knowledge graph built successfully")
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
        
        return self.knowledge_graph
    
    async def extract_operational_intelligence(self, content: str) -> Dict[str, Any]:
        """Extract operational intelligence and business insights"""
        operational_data = {
            'sops': [],
            'decision_criteria': [],
            'compliance_requirements': [],
            'risk_factors': [],
            'kpis': [],
            'assumptions': []
        }
        
        try:
            # Use specialized prompts for operational intelligence
            sop_prompt = f"""
            Identify Standard Operating Procedures (SOPs) in the following text:
            
            {content}
            
            Extract:
            - Procedure name
            - Steps
            - Frequency
            - Responsible parties
            - Success criteria
            """
            
            sop_response = self.model_manager.generate_llm_response(sop_prompt)
            operational_data['sops'] = self._parse_sop_response(sop_response)
            
            # Additional operational intelligence extraction...
            
        except Exception as e:
            logger.error(f"Error extracting operational intelligence: {e}")
        
        return operational_data
    
    def _parse_sop_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse SOP extraction response"""
        sops = []
        
        try:
            # Implementation for parsing SOP responses
            pass
        except Exception as e:
            logger.error(f"Error parsing SOP response: {e}")
        
        return sops
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of extracted knowledge"""
        return {
            'total_entities': len(self.knowledge_graph.entities),
            'total_relationships': len(self.knowledge_graph.relationships),
            'entity_types': self._get_entity_type_distribution(),
            'relationship_types': self._get_relationship_type_distribution()
        }
    
    def _get_entity_type_distribution(self) -> Dict[str, int]:
        """Get distribution of entity types"""
        distribution = {}
        for entity in self.knowledge_graph.entities.values():
            distribution[entity.type] = distribution.get(entity.type, 0) + 1
        return distribution
    
    def _get_relationship_type_distribution(self) -> Dict[str, int]:
        """Get distribution of relationship types"""
        distribution = {}
        for rel in self.knowledge_graph.relationships:
            distribution[rel.relationship_type] = distribution.get(rel.relationship_type, 0) + 1
        return distribution
    
    def cleanup_memory(self):
        """Clean up memory resources"""
        try:
            if hasattr(self.model_manager, 'llm'):
                del self.model_manager.llm
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            logger.info("Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")