"""
EXPLAINIUM - Knowledge Export and Visualization System

Exports knowledge graphs, generates documentation, and creates training materials
from extracted knowledge insights.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import csv
import yaml
from dataclasses import asdict

# Internal imports
from src.logging_config import get_logger
from src.ai.advanced_knowledge_engine import AdvancedKnowledgeEngine, Neo4jLiteGraph

logger = get_logger(__name__)


class KnowledgeExporter:
    """Export knowledge in various formats for visualization and analysis"""
    
    def __init__(self, knowledge_engine: AdvancedKnowledgeEngine):
        self.knowledge_engine = knowledge_engine
        self.export_formats = ['json', 'csv', 'yaml', 'cytoscape', 'markdown']
    
    async def export_knowledge_graph(self, format: str = "cytoscape") -> Dict[str, Any]:
        """Export knowledge graph for visualization tools"""
        try:
            if format not in self.export_formats:
                raise ValueError(f"Unsupported export format: {format}")
            
            knowledge_graph = self.knowledge_engine.get_knowledge_graph()
            
            if format == "cytoscape":
                return knowledge_graph.export_cytoscape()
            elif format == "json":
                return self._export_as_json(knowledge_graph)
            elif format == "csv":
                return self._export_as_csv(knowledge_graph)
            elif format == "yaml":
                return self._export_as_yaml(knowledge_graph)
            elif format == "markdown":
                return self._export_as_markdown(knowledge_graph)
            else:
                raise ValueError(f"Format {format} not implemented")
            
        except Exception as e:
            logger.error(f"Failed to export knowledge graph: {e}")
            return {"error": str(e)}
    
    def _export_as_json(self, knowledge_graph: Neo4jLiteGraph) -> Dict[str, Any]:
        """Export knowledge graph as JSON"""
        return {
            "export_timestamp": datetime.now().isoformat(),
            "format": "json",
            "nodes": [asdict(node) for node in knowledge_graph.nodes.values()],
            "edges": [asdict(edge) for edge in knowledge_graph.edges],
            "metadata": {
                "total_nodes": len(knowledge_graph.nodes),
                "total_edges": len(knowledge_graph.edges),
                "node_types": list(knowledge_graph.node_types),
                "relationship_types": list(knowledge_graph.relationship_types)
            }
        }
    
    def _export_as_csv(self, knowledge_graph: Neo4jLiteGraph) -> Dict[str, Any]:
        """Export knowledge graph as CSV data"""
        # Prepare nodes CSV
        nodes_data = []
        for node in knowledge_graph.nodes.values():
            nodes_data.append({
                'id': node.id,
                'type': node.type,
                'name': node.name,
                'description': node.description,
                'confidence': node.confidence,
                'created_at': node.created_at.isoformat(),
                'updated_at': node.updated_at.isoformat()
            })
        
        # Prepare edges CSV
        edges_data = []
        for edge in knowledge_graph.edges:
            edges_data.append({
                'source_id': edge.source_id,
                'target_id': edge.target_id,
                'relationship_type': edge.relationship_type,
                'strength': edge.strength,
                'created_at': edge.created_at.isoformat()
            })
            
            return {
            "format": "csv",
            "nodes": nodes_data,
            "edges": edges_data,
                "metadata": {
                "total_nodes": len(nodes_data),
                "total_edges": len(edges_data)
            }
        }
    
    def _export_as_yaml(self, knowledge_graph: Neo4jLiteGraph) -> Dict[str, Any]:
        """Export knowledge graph as YAML data"""
        return {
            "format": "yaml",
                    "export_timestamp": datetime.now().isoformat(),
            "knowledge_graph": {
                "nodes": [asdict(node) for node in knowledge_graph.nodes.values()],
                "edges": [asdict(edge) for edge in knowledge_graph.edges]
            }
        }
    
    def _export_as_markdown(self, knowledge_graph: Neo4jLiteGraph) -> Dict[str, Any]:
        """Export knowledge graph as Markdown documentation"""
        markdown_content = f"""# Knowledge Graph Export

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Nodes**: {len(knowledge_graph.nodes)}
- **Total Edges**: {len(knowledge_graph.edges)}
- **Node Types**: {', '.join(knowledge_graph.node_types)}
- **Relationship Types**: {', '.join(knowledge_graph.relationship_types)}

## Nodes

"""
        
        # Group nodes by type
        nodes_by_type = {}
        for node in knowledge_graph.nodes.values():
            if node.type not in nodes_by_type:
                nodes_by_type[node.type] = []
            nodes_by_type[node.type].append(node)
        
        # Generate markdown for each node type
        for node_type, nodes in nodes_by_type.items():
            markdown_content += f"### {node_type.title()} Nodes\n\n"
            
            for node in nodes:
                markdown_content += f"#### {node.name}\n"
                markdown_content += f"- **ID**: `{node.id}`\n"
                markdown_content += f"- **Description**: {node.description}\n"
                markdown_content += f"- **Confidence**: {node.confidence:.2f}\n"
                markdown_content += f"- **Created**: {node.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Generate markdown for relationships
        markdown_content += "## Relationships\n\n"
        
        for edge in knowledge_graph.edges:
            source_name = knowledge_graph.nodes[edge.source_id].name
            target_name = knowledge_graph.nodes[edge.target_id].name
            
            markdown_content += f"- **{source_name}** → *{edge.relationship_type}* → **{target_name}**\n"
            markdown_content += f"  - Strength: {edge.strength:.2f}\n"
            markdown_content += f"  - Created: {edge.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return {
            "format": "markdown",
            "content": markdown_content,
            "metadata": {
                "total_nodes": len(knowledge_graph.nodes),
                "total_edges": len(knowledge_graph.edges)
            }
        }
    
    async def generate_documentation(self, output_dir: str = "./docs") -> Dict[str, Any]:
        """Auto-generate documentation from extracted knowledge"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            knowledge_graph = self.knowledge_engine.get_knowledge_graph()
            
            # Generate different types of documentation
            docs_generated = {}
            
            # 1. Process Documentation
            process_docs = await self._generate_process_documentation(knowledge_graph)
            process_file = output_path / "process_documentation.md"
            with open(process_file, 'w') as f:
                f.write(process_docs)
            docs_generated['process_documentation'] = str(process_file)
            
            # 2. API Documentation
            api_docs = await self._generate_api_documentation(knowledge_graph)
            api_file = output_path / "api_documentation.md"
            with open(api_file, 'w') as f:
                f.write(api_docs)
            docs_generated['api_documentation'] = str(api_file)
            
            # 3. Workflow Diagrams (Mermaid format)
            workflow_docs = await self._generate_workflow_diagrams(knowledge_graph)
            workflow_file = output_path / "workflow_diagrams.md"
            with open(workflow_file, 'w') as f:
                f.write(workflow_docs)
            docs_generated['workflow_diagrams'] = str(workflow_file)
            
            # 4. Organizational Charts
            org_docs = await self._generate_organizational_charts(knowledge_graph)
            org_file = output_path / "organizational_charts.md"
            with open(org_file, 'w') as f:
                f.write(org_docs)
            docs_generated['organizational_charts'] = str(org_file)
            
            logger.info(f"Generated documentation in {output_path}")
            return {
                "status": "success",
                "output_directory": str(output_path),
                "files_generated": docs_generated
            }
            
        except Exception as e:
            logger.error(f"Failed to generate documentation: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _generate_process_documentation(self, knowledge_graph: Neo4jLiteGraph) -> str:
        """Generate process documentation from knowledge graph"""
        process_nodes = knowledge_graph.get_nodes_by_type('process')
        
        if not process_nodes:
            return "# Process Documentation\n\nNo process nodes found in knowledge graph.\n"
        
        markdown = "# Process Documentation\n\n"
        markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown += f"Total processes found: {len(process_nodes)}\n\n"
        
        for process in process_nodes:
            markdown += f"## {process.name}\n\n"
            markdown += f"{process.description}\n\n"
            markdown += f"**Confidence**: {process.confidence:.2f}\n"
            markdown += f"**Created**: {process.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Find related nodes
            connected = knowledge_graph.get_connected_nodes(process.id)
            if connected:
                markdown += "**Related Elements**:\n"
                for related_node, edge in connected:
                    markdown += f"- {related_node.name} ({edge.relationship_type})\n"
                markdown += "\n"
            
            markdown += "---\n\n"
        
        return markdown
    
    async def _generate_api_documentation(self, knowledge_graph: Neo4jLiteGraph) -> str:
        """Generate API documentation from knowledge graph"""
        system_nodes = knowledge_graph.get_nodes_by_type('system')
        
        if not system_nodes:
            return "# API Documentation\n\nNo system nodes found in knowledge graph.\n"
        
        markdown = "# API Documentation\n\n"
        markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown += f"Total systems found: {len(system_nodes)}\n\n"
        
        for system in system_nodes:
            markdown += f"## {system.name}\n\n"
            markdown += f"{system.description}\n\n"
            
            # Find related requirements
            connected = knowledge_graph.get_connected_nodes(system.id)
            requirements = [node for node, edge in connected if node.type == 'requirement']
            
            if requirements:
                markdown += "**Requirements**:\n"
                for req in requirements:
                    markdown += f"- {req.name}: {req.description}\n"
                markdown += "\n"
            
            markdown += "---\n\n"
        
        return markdown
    
    async def _generate_workflow_diagrams(self, knowledge_graph: Neo4jLiteGraph) -> str:
        """Generate workflow diagrams in Mermaid format"""
        process_nodes = knowledge_graph.get_nodes_by_type('process')
        
        if not process_nodes:
            return "# Workflow Diagrams\n\nNo process nodes found in knowledge graph.\n"
        
        markdown = "# Workflow Diagrams\n\n"
        markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown += "These diagrams are in Mermaid format and can be rendered in GitHub, GitLab, or other Markdown viewers.\n\n"
        
        for i, process in enumerate(process_nodes):
            markdown += f"## {process.name} Workflow\n\n"
            markdown += "```mermaid\n"
            markdown += "graph TD\n"
            
            # Create a simple workflow diagram
            markdown += f"    A[Start: {process.name}] --> B[Process Step 1]\n"
            markdown += f"    B --> C[Process Step 2]\n"
            markdown += f"    C --> D[Decision Point]\n"
            markdown += f"    D -->|Yes| E[Continue Process]\n"
            markdown += f"    D -->|No| F[Alternative Path]\n"
            markdown += f"    E --> G[End: {process.name}]\n"
            markdown += f"    F --> G\n"
            markdown += "```\n\n"
            
            markdown += f"**Description**: {process.description}\n\n"
            markdown += "---\n\n"
        
        return markdown
    
    async def _generate_organizational_charts(self, knowledge_graph: Neo4jLiteGraph) -> str:
        """Generate organizational charts from knowledge graph"""
        person_nodes = knowledge_graph.get_nodes_by_type('person')
        
        if not person_nodes:
            return "# Organizational Charts\n\nNo person nodes found in knowledge graph.\n"
        
        markdown = "# Organizational Charts\n\n"
        markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown += f"Total people found: {len(person_nodes)}\n\n"
        
        # Group people by relationships
        markdown += "## Organizational Structure\n\n"
        markdown += "```mermaid\n"
        markdown += "graph TD\n"
        
        for person in person_nodes:
            markdown += f"    {person.id.replace('-', '_')}[{person.name}]\n"
        
        # Add relationships
        for edge in knowledge_graph.edges:
            if (knowledge_graph.nodes[edge.source_id].type == 'person' and 
                knowledge_graph.nodes[edge.target_id].type == 'person'):
                source_id = edge.source_id.replace('-', '_')
                target_id = edge.target_id.replace('-', '_')
                markdown += f"    {source_id} -->|{edge.relationship_type}| {target_id}\n"
        
        markdown += "```\n\n"
        
        # Detailed information
        for person in person_nodes:
            markdown += f"### {person.name}\n\n"
            markdown += f"{person.description}\n\n"
            
            # Find related roles and responsibilities
            connected = knowledge_graph.get_connected_nodes(person.id)
            if connected:
                markdown += "**Related Elements**:\n"
                for related_node, edge in connected:
                    markdown += f"- {related_node.name} ({edge.relationship_type})\n"
                markdown += "\n"
            
            markdown += "---\n\n"
        
        return markdown
    
    async def create_training_materials(self, output_dir: str = "./training") -> Dict[str, Any]:
        """Generate training content from knowledge base"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            knowledge_graph = self.knowledge_engine.get_knowledge_graph()
            
            # Generate different types of training materials
            materials_generated = {}
            
            # 1. Training Manual
            training_manual = await self._generate_training_manual(knowledge_graph)
            manual_file = output_path / "training_manual.md"
            with open(manual_file, 'w') as f:
                f.write(training_manual)
            materials_generated['training_manual'] = str(manual_file)
            
            # 2. Quiz Questions
            quiz_questions = await self._generate_quiz_questions(knowledge_graph)
            quiz_file = output_path / "quiz_questions.md"
            with open(quiz_file, 'w') as f:
                f.write(quiz_questions)
            materials_generated['quiz_questions'] = str(quiz_file)
            
            # 3. Best Practices Guide
            best_practices = await self._generate_best_practices_guide(knowledge_graph)
            practices_file = output_path / "best_practices.md"
            with open(practices_file, 'w') as f:
                f.write(best_practices)
            materials_generated['best_practices'] = str(practices_file)
            
            # 4. Training Slides (Markdown format)
            training_slides = await self._generate_training_slides(knowledge_graph)
            slides_file = output_path / "training_slides.md"
            with open(slides_file, 'w') as f:
                f.write(training_slides)
            materials_generated['training_slides'] = str(slides_file)
            
            logger.info(f"Generated training materials in {output_path}")
            return {
                "status": "success",
                "output_directory": str(output_path),
                "files_generated": materials_generated
            }
            
        except Exception as e:
            logger.error(f"Failed to create training materials: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _generate_training_manual(self, knowledge_graph: Neo4jLiteGraph) -> str:
        """Generate training manual from knowledge graph"""
        markdown = "# Training Manual\n\n"
        markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown += "This training manual is automatically generated from the knowledge base.\n\n"
        
        # Table of Contents
        markdown += "## Table of Contents\n\n"
        markdown += "1. [Processes](#processes)\n"
        markdown += "2. [Systems](#systems)\n"
        markdown += "3. [Requirements](#requirements)\n"
        markdown += "4. [Risk Management](#risk-management)\n\n"
        
        # Processes Section
        process_nodes = knowledge_graph.get_nodes_by_type('process')
        if process_nodes:
            markdown += "## Processes\n\n"
            for process in process_nodes:
                markdown += f"### {process.name}\n\n"
                markdown += f"{process.description}\n\n"
                markdown += "**Key Steps**:\n"
                markdown += "1. Step 1: [Description]\n"
                markdown += "2. Step 2: [Description]\n"
                markdown += "3. Step 3: [Description]\n\n"
                markdown += "---\n\n"
        
        # Systems Section
        system_nodes = knowledge_graph.get_nodes_by_type('system')
        if system_nodes:
            markdown += "## Systems\n\n"
            for system in system_nodes:
                markdown += f"### {system.name}\n\n"
                markdown += f"{system.description}\n\n"
                markdown += "---\n\n"
        
        return markdown
    
    async def _generate_quiz_questions(self, knowledge_graph: Neo4jLiteGraph) -> str:
        """Generate quiz questions from knowledge graph"""
        markdown = "# Quiz Questions\n\n"
        markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Generate questions from different node types
        questions = []
        
        # Process questions
        process_nodes = knowledge_graph.get_nodes_by_type('process')
        for process in process_nodes:
            questions.append({
                'question': f"What is the main purpose of the {process.name} process?",
                'answer': process.description,
                'type': 'process'
            })
        
        # System questions
        system_nodes = knowledge_graph.get_nodes_by_type('system')
        for system in system_nodes:
            questions.append({
                'question': f"Describe the {system.name} system and its role.",
                'answer': system.description,
                'type': 'system'
            })
        
        # Format questions
        for i, q in enumerate(questions, 1):
            markdown += f"## Question {i}\n\n"
            markdown += f"**Type**: {q['type'].title()}\n\n"
            markdown += f"**Question**: {q['question']}\n\n"
            markdown += f"**Answer**: {q['answer']}\n\n"
            markdown += "---\n\n"
        
        return markdown
    
    async def _generate_best_practices_guide(self, knowledge_graph: Neo4jLiteGraph) -> str:
        """Generate best practices guide from knowledge graph"""
        markdown = "# Best Practices Guide\n\n"
        markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Extract best practices from different node types
        best_practices = []
        
        # Process best practices
        process_nodes = knowledge_graph.get_nodes_by_type('process')
        for process in process_nodes:
            if 'best' in process.description.lower() or 'practice' in process.description.lower():
                best_practices.append({
                    'category': 'Process',
                    'practice': process.name,
                    'description': process.description
                })
        
        # System best practices
        system_nodes = knowledge_graph.get_nodes_by_type('system')
        for system in system_nodes:
            if 'best' in system.description.lower() or 'practice' in system.description.lower():
                best_practices.append({
                    'category': 'System',
                    'practice': system.name,
                    'description': system.description
                })
        
        if best_practices:
            for practice in best_practices:
                markdown += f"## {practice['category']}: {practice['practice']}\n\n"
                markdown += f"{practice['description']}\n\n"
                markdown += "---\n\n"
        else:
            markdown += "No specific best practices identified in the knowledge base.\n\n"
            markdown += "Consider reviewing processes and systems for optimization opportunities.\n\n"
        
        return markdown
    
    async def _generate_training_slides(self, knowledge_graph: Neo4jLiteGraph) -> str:
        """Generate training slides in Markdown format"""
        markdown = "# Training Slides\n\n"
        markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown += "These slides can be converted to PowerPoint or other presentation formats.\n\n"
        
        slide_number = 1
        
        # Title slide
        markdown += f"## Slide {slide_number}: Title\n\n"
        markdown += "# Knowledge Base Training\n\n"
        markdown += "Understanding Our Organization's Knowledge\n\n"
        markdown += "---\n\n"
        slide_number += 1
        
        # Overview slide
        markdown += f"## Slide {slide_number}: Overview\n\n"
        markdown += "## What We'll Cover\n\n"
        markdown += f"- **{len(knowledge_graph.nodes)}** Knowledge Elements\n"
        markdown += f"- **{len(knowledge_graph.edges)}** Relationships\n"
        markdown += f"- **{len(knowledge_graph.node_types)}** Types of Knowledge\n\n"
        markdown += "---\n\n"
        slide_number += 1
        
        # Process overview
        process_nodes = knowledge_graph.get_nodes_by_type('process')
        if process_nodes:
            markdown += f"## Slide {slide_number}: Key Processes\n\n"
            markdown += "## Our Core Processes\n\n"
            for process in process_nodes[:5]:  # Limit to 5 for slide
                markdown += f"- {process.name}\n"
            markdown += "\n---\n\n"
            slide_number += 1
        
        # Systems overview
        system_nodes = knowledge_graph.get_nodes_by_type('system')
        if system_nodes:
            markdown += f"## Slide {slide_number}: Systems Overview\n\n"
            markdown += "## Key Systems\n\n"
            for system in system_nodes[:5]:  # Limit to 5 for slide
                markdown += f"- {system.name}\n"
            markdown += "\n---\n\n"
            slide_number += 1
        
        # Summary slide
        markdown += f"## Slide {slide_number}: Summary\n\n"
        markdown += "## Key Takeaways\n\n"
        markdown += "- Knowledge is interconnected\n"
        markdown += "- Processes drive operations\n"
        markdown += "- Systems support processes\n"
        markdown += "- Continuous improvement is key\n\n"
        markdown += "---\n\n"
        
        return markdown
    
    def save_to_file(self, data: Dict[str, Any], file_path: str, format: str = "json"):
        """Save exported data to a file"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format == "csv":
                if 'nodes' in data and 'edges' in data:
                    # Save nodes
                    nodes_file = file_path.parent / f"{file_path.stem}_nodes.csv"
                    with open(nodes_file, 'w', newline='') as f:
                        if data['nodes']:
                            writer = csv.DictWriter(f, fieldnames=data['nodes'][0].keys())
                            writer.writeheader()
                            writer.writerows(data['nodes'])
                    
                    # Save edges
                    edges_file = file_path.parent / f"{file_path.stem}_edges.csv"
                    with open(edges_file, 'w', newline='') as f:
                        if data['edges']:
                            writer = csv.DictWriter(f, fieldnames=data['edges'][0].keys())
                            writer.writeheader()
                            writer.writerows(data['edges'])
            elif format == "yaml":
                with open(file_path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, default_representer=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data saved to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save data to file: {e}")
            raise