#!/usr/bin/env python3
"""
EXPLAINIUM Demo Script

Demonstrates the advanced AI-powered knowledge extraction capabilities
of the transformed EXPLAINIUM system.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.ai.advanced_knowledge_engine import AdvancedKnowledgeEngine
from src.processors.processor import DocumentProcessor
from src.export.knowledge_export import KnowledgeExporter


async def demo_basic_extraction():
    """Demo basic knowledge extraction"""
    print("🧠 Demo 1: Basic Knowledge Extraction")
    print("=" * 50)
    
    # Sample business document
    sample_document = {
        "content": """
        Customer Onboarding Process
        
        Our customer onboarding workflow consists of the following steps:
        
        1. Initial Contact: Sales team receives inquiry
        2. Requirements Gathering: Technical team assesses needs
        3. Proposal Development: Solution design and pricing
        4. Contract Negotiation: Legal review and terms
        5. Implementation: Technical deployment and training
        6. Go-Live: Customer activation and support handoff
        
        Key Decision Points:
        - Customer must have valid business license
        - Technical requirements must be feasible
        - Budget must meet minimum threshold
        
        Risk Factors:
        - Technical complexity exceeding capabilities
        - Customer budget constraints
        - Timeline delays due to dependencies
        
        Compliance Requirements:
        - Data protection standards (GDPR, CCPA)
        - Industry-specific regulations
        - Internal security policies
        """,
        "type": "text",
        "metadata": {
            "department": "operations",
            "document_type": "process_documentation",
            "author": "Operations Team"
        }
    }
    
    try:
        # Initialize the advanced knowledge engine
        print("Initializing Advanced Knowledge Engine...")
        engine = AdvancedKnowledgeEngine()
        
        # Extract deep knowledge
        print("Extracting deep knowledge...")
        knowledge = await engine.extract_deep_knowledge(sample_document)
        
        print("\n📊 Extracted Knowledge:")
        print(f"• Concepts: {len(knowledge.get('concepts', []))}")
        print(f"• Entities: {len(knowledge.get('entities', []))}")
        print(f"• Relationships: {len(knowledge.get('relationships', []))}")
        print(f"• Workflows: {len(knowledge.get('workflows', []))}")
        print(f"• Confidence: {knowledge.get('confidence', 'N/A')}")
        
        # Build knowledge graph
        print("\n🔗 Building Knowledge Graph...")
        graph_result = await engine.build_knowledge_graph(knowledge)
        
        print(f"• Nodes added: {graph_result.get('nodes_added', 0)}")
        print(f"• Edges added: {graph_result.get('edges_added', 0)}")
        
        return knowledge, engine
        
    except Exception as e:
        print(f"❌ Error in basic extraction: {e}")
        return None, None


async def demo_operational_intelligence():
    """Demo operational intelligence extraction"""
    print("\n\n🎯 Demo 2: Operational Intelligence Extraction")
    print("=" * 50)
    
    try:
        # Initialize engine
        engine = AdvancedKnowledgeEngine()
        
        # Sample operational content
        operational_content = """
        Standard Operating Procedure: Customer Support Escalation
        
        Level 1: Frontline Support
        - Handle basic inquiries and common issues
        - Escalate complex technical problems to Level 2
        - Document all interactions in CRM system
        
        Level 2: Technical Support
        - Resolve advanced technical issues
        - Escalate business-critical problems to Level 3
        - Coordinate with development team for bug fixes
        
        Level 3: Management Escalation
        - Handle business-critical issues
        - Coordinate with executive team
        - Manage customer relationship for high-value accounts
        
        Decision Criteria:
        - Issue complexity score > 7/10
        - Customer tier (Gold/Platinum)
        - SLA breach risk > 80%
        - Revenue impact > $10,000
        
        Performance Metrics:
        - First response time: < 2 hours
        - Resolution time: < 24 hours
        - Customer satisfaction: > 4.5/5
        - Escalation rate: < 15%
        """
        
        print("Extracting operational intelligence...")
        ops_intel = await engine.extract_operational_intelligence(operational_content)
        
        print("\n📋 Operational Intelligence:")
        if 'sops' in ops_intel:
            print(f"• SOPs identified: {len(ops_intel['sops'])}")
        if 'decision_criteria' in ops_intel:
            print(f"• Decision criteria: {len(ops_intel['decision_criteria'])}")
        if 'risk_factors' in ops_intel:
            print(f"• Risk factors: {len(ops_intel['risk_factors'])}")
        if 'performance_metrics' in ops_intel:
            print(f"• Performance metrics: {len(ops_intel['performance_metrics'])}")
        
        return ops_intel
        
    except Exception as e:
        print(f"❌ Error in operational intelligence: {e}")
        return None


async def demo_knowledge_export():
    """Demo knowledge export capabilities"""
    print("\n\n📤 Demo 3: Knowledge Export")
    print("=" * 50)
    
    try:
        # Initialize exporter
        exporter = KnowledgeExporter()
        
        # Mock knowledge graph data
        mock_graph = {
            "nodes": [
                {"id": "1", "name": "Customer Onboarding", "type": "process"},
                {"id": "2", "name": "Sales Team", "type": "team"},
                {"id": "3", "name": "Technical Assessment", "type": "activity"}
            ],
            "edges": [
                {"source": "1", "target": "2", "type": "involves"},
                {"source": "1", "target": "3", "type": "includes"}
            ]
        }
        
        print("Exporting knowledge in different formats...")
        
        # Export as JSON
        json_export = exporter._export_as_json(mock_graph)
        print(f"• JSON export: {len(json_export)} nodes, {len(json_export.get('edges', []))} edges")
        
        # Export as Markdown
        markdown_export = exporter._export_as_markdown(mock_graph)
        print(f"• Markdown export: {len(markdown_export['content'])} characters")
        
        # Export as CSV
        csv_export = exporter._export_as_csv(mock_graph)
        print(f"• CSV export: {len(csv_export)} rows")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in knowledge export: {e}")
        return False


async def demo_model_management():
    """Demo model management capabilities"""
    print("\n\n🤖 Demo 4: Model Management")
    print("=" * 50)
    
    try:
        from src.ai.advanced_knowledge_engine import ModelManager
        
        # Initialize model manager
        manager = ModelManager("./demo_models")
        
        # Detect hardware profile
        profile = manager.detect_hardware_profile()
        print(f"• Detected hardware profile: {profile}")
        
        # Show model configurations
        config = manager.model_configs.get(profile, {})
        if 'llm' in config:
            llm_config = config['llm']
            print(f"• Primary LLM: {llm_config.get('primary', 'N/A')}")
            print(f"• Quantization: {llm_config.get('quantization', 'N/A')}")
            print(f"• Max RAM: {llm_config.get('max_ram', 'N/A')}")
        
        if 'embeddings' in config:
            emb_config = config['embeddings']
            print(f"• Embedding model: {emb_config.get('primary', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model management: {e}")
        return False


async def main():
    """Main demo function"""
    print("🚀 EXPLAINIUM Advanced AI-Powered Knowledge Extraction Demo")
    print("=" * 70)
    print("This demo showcases the transformed EXPLAINIUM system capabilities:")
    print("• Deep knowledge extraction using local LLMs")
    print("• Operational intelligence discovery")
    print("• Knowledge graph building")
    print("• Multi-format export")
    print("• Apple M4 optimization")
    print("=" * 70)
    
    try:
        # Run demos
        knowledge, engine = await demo_basic_extraction()
        
        if knowledge:
            await demo_operational_intelligence()
            await demo_knowledge_export()
            await demo_model_management()
        
        print("\n\n✅ Demo completed successfully!")
        print("\n🎉 EXPLAINIUM is now a sophisticated, AI-powered knowledge processing system!")
        print("\nTo get started:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Setup models: python scripts/model_manager.py --action setup")
        print("3. Run frontend: streamlit run src/frontend/knowledge_table.py")
        print("4. Process documents: python -m src.processors.processor")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nThis might be due to missing dependencies or models.")
        print("Please ensure you have installed all requirements and downloaded the AI models.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
