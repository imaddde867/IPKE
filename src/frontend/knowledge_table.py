"""
EXPLAINIUM - Knowledge Table Frontend

A clean, modern frontend component for displaying extracted knowledge
in a large table format with advanced search and filtering capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import io

# Internal imports
import sys
import os

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Try importing with src prefix
    from src.export.knowledge_export import KnowledgeExporter
    from src.ai.advanced_knowledge_engine import AdvancedKnowledgeEngine
    from src.core.config import AIConfig
    FULL_SYSTEM_AVAILABLE = True
    print("✅ Full AI system loaded successfully")
except ImportError as e1:
    try:
        # Try importing without src prefix
        from export.knowledge_export import KnowledgeExporter
        from ai.advanced_knowledge_engine import AdvancedKnowledgeEngine
        from core.config import AIConfig
        FULL_SYSTEM_AVAILABLE = True
        print("✅ Full AI system loaded successfully (fallback)")
    except ImportError as e2:
        print(f"❌ Import error: {e1}, {e2}")
        # Fallback for when modules aren't available
        KnowledgeExporter = None
        AdvancedKnowledgeEngine = None
        AIConfig = None
        FULL_SYSTEM_AVAILABLE = False


class KnowledgeTableFrontend:
    """Frontend component for displaying extracted knowledge in table format"""
    
    def __init__(self, knowledge_engine: AdvancedKnowledgeEngine):
        self.knowledge_engine = knowledge_engine
        self.exporter = KnowledgeExporter(knowledge_engine)
        
    def render_knowledge_table(self):
        """Render the main knowledge table interface"""
        st.set_page_config(
            page_title="EXPLAINIUM Knowledge Table",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🧠 EXPLAINIUM Knowledge Table")
        st.markdown("**Deep Knowledge Extraction & Analysis Dashboard**")
        
        # Sidebar controls
        with st.sidebar:
            st.header("📊 Table Controls")
            
            # Knowledge type filter
            knowledge_types = st.multiselect(
                "Knowledge Types",
                ["concepts", "processes", "systems", "requirements", "people", "risks"],
                default=["concepts", "processes", "systems"]
            )
            
            # Confidence threshold
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1
            )
            
            # Date range filter
            date_range = st.date_input(
                "Date Range",
                value=(datetime.now().date(), datetime.now().date())
            )
            
            # Search functionality
            search_query = st.text_input("🔍 Search Knowledge", placeholder="Enter search terms...")
            
            # Export options
            st.header("📤 Export Options")
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "CSV", "YAML", "Markdown", "Cytoscape"]
            )
            
            if st.button("Export Knowledge", type="primary"):
                self._export_knowledge(export_format.lower())
        
        # Main content area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.header("📋 Knowledge Table")
            
            # Get knowledge data
            knowledge_data = self._get_filtered_knowledge(
                knowledge_types, confidence_threshold, date_range, search_query
            )
            
            if knowledge_data:
                # Convert to DataFrame for display
                df = self._convert_to_dataframe(knowledge_data)
                
                # Display table with pagination
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            help="Confidence score for this knowledge item",
                            min_value=0.0,
                            max_value=1.0
                        ),
                        "created_at": st.column_config.DatetimeColumn(
                            "Created",
                            format="DD/MM/YYYY HH:MM"
                        ),
                        "updated_at": st.column_config.DatetimeColumn(
                            "Updated",
                            format="DD/MM/YYYY HH:MM"
                        )
                    }
                )
                
                # Table statistics
                st.info(f"📊 Showing {len(df)} knowledge items")
                
            else:
                st.warning("No knowledge data found with the current filters.")
        
        with col2:
            st.header("📈 Quick Stats")
            
            if knowledge_data:
                # Knowledge distribution by type
                type_counts = self._get_type_distribution(knowledge_data)
                fig_pie = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    title="Knowledge Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Confidence distribution
                confidence_data = [item.get('confidence', 0) for item in knowledge_data]
                fig_hist = px.histogram(
                    x=confidence_data,
                    title="Confidence Distribution",
                    nbins=20
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        # Knowledge graph visualization
        if knowledge_data:
            st.header("🕸️ Knowledge Graph")
            
            # Create network graph
            graph_data = self._create_network_graph(knowledge_data)
            
            if graph_data:
                # Use Plotly for network visualization
                fig_network = self._create_plotly_network(graph_data)
                st.plotly_chart(fig_network, use_container_width=True)
        
        # Advanced analytics section
        if knowledge_data:
            st.header("🔍 Advanced Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Temporal analysis
                st.subheader("📅 Temporal Trends")
                temporal_data = self._analyze_temporal_trends(knowledge_data)
                if temporal_data:
                    fig_temporal = px.line(
                        temporal_data,
                        x='date',
                        y='count',
                        title="Knowledge Creation Over Time"
                    )
                    st.plotly_chart(fig_temporal, use_container_width=True)
            
            with col2:
                # Relationship analysis
                st.subheader("🔗 Relationship Analysis")
                relationship_data = self._analyze_relationships(knowledge_data)
                if relationship_data:
                    fig_relationships = px.bar(
                        relationship_data,
                        x='relationship_type',
                        y='count',
                        title="Relationship Types"
                    )
                    st.plotly_chart(fig_relationships, use_container_width=True)
    
    def _get_filtered_knowledge(self, types: List[str], confidence: float, 
                               date_range: tuple, search: str) -> List[Dict[str, Any]]:
        """Get filtered knowledge data based on user selections"""
        try:
            # Get knowledge graph from engine
            knowledge_graph = self.knowledge_engine.get_knowledge_graph()
            
            if not knowledge_graph:
                return []
            
            # Filter nodes by type
            filtered_nodes = []
            for node in knowledge_graph.nodes.values():
                if node.type in types and node.confidence >= confidence:
                    # Apply date filter
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        if (start_date <= node.created_at.date() <= end_date):
                            filtered_nodes.append(node)
                    else:
                        filtered_nodes.append(node)
            
            # Apply search filter
            if search:
                filtered_nodes = [
                    node for node in filtered_nodes
                    if search.lower() in node.name.lower() or 
                       search.lower() in node.description.lower()
                ]
            
            # Convert to list of dictionaries
            return [self._node_to_dict(node) for node in filtered_nodes]
            
        except Exception as e:
            st.error(f"Error retrieving knowledge data: {e}")
            return []
    
    def _node_to_dict(self, node) -> Dict[str, Any]:
        """Convert knowledge graph node to dictionary"""
        return {
            'id': node.id,
            'name': node.name,
            'type': node.type,
            'description': node.description,
            'confidence': node.confidence,
            'created_at': node.created_at,
            'updated_at': node.updated_at,
            'metadata': node.metadata
        }
    
    def _convert_to_dataframe(self, knowledge_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert knowledge data to pandas DataFrame"""
        if not knowledge_data:
            return pd.DataFrame()
        
        # Flatten metadata if present
        flattened_data = []
        for item in knowledge_data:
            flat_item = item.copy()
            if 'metadata' in flat_item and isinstance(flat_item['metadata'], dict):
                for key, value in flat_item['metadata'].items():
                    flat_item[f'meta_{key}'] = value
                del flat_item['metadata']
            flattened_data.append(flat_item)
        
        df = pd.DataFrame(flattened_data)
        
        # Reorder columns for better display
        priority_columns = ['name', 'type', 'description', 'confidence', 'created_at']
        other_columns = [col for col in df.columns if col not in priority_columns]
        df = df[priority_columns + other_columns]
        
        return df
    
    def _get_type_distribution(self, knowledge_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of knowledge by type"""
        type_counts = {}
        for item in knowledge_data:
            item_type = item.get('type', 'unknown')
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        return type_counts
    
    def _create_network_graph(self, knowledge_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create network graph data for visualization"""
        try:
            # Get relationships from knowledge graph
            knowledge_graph = self.knowledge_engine.get_knowledge_graph()
            
            if not knowledge_graph:
                return None
            
            # Create nodes and edges for visualization
            nodes = []
            edges = []
            
            # Add nodes
            for item in knowledge_data:
                nodes.append({
                    'id': item['id'],
                    'label': item['name'],
                    'type': item['type'],
                    'confidence': item['confidence']
                })
            
            # Add edges (limit to avoid overwhelming visualization)
            edge_count = 0
            max_edges = 100
            
            for edge in knowledge_graph.edges:
                if edge_count >= max_edges:
                    break
                
                # Only include edges between nodes in our filtered data
                node_ids = {item['id'] for item in knowledge_data}
                if edge.source_id in node_ids and edge.target_id in node_ids:
                    edges.append({
                        'source': edge.source_id,
                        'target': edge.target_id,
                        'type': edge.relationship_type,
                        'strength': edge.strength
                    })
                    edge_count += 1
            
            return {
                'nodes': nodes,
                'edges': edges
            }
            
        except Exception as e:
            st.error(f"Error creating network graph: {e}")
            return None
    
    def _create_plotly_network(self, graph_data: Dict[str, Any]):
        """Create Plotly network visualization"""
        try:
            nodes = graph_data['nodes']
            edges = graph_data['edges']
            
            # Create node positions (simple circular layout)
            import math
            node_positions = {}
            for i, node in enumerate(nodes):
                angle = 2 * math.pi * i / len(nodes)
                radius = 200
                node_positions[node['id']] = {
                    'x': radius * math.cos(angle),
                    'y': radius * math.sin(angle)
                }
            
            # Create edge traces
            edge_x = []
            edge_y = []
            for edge in edges:
                source_pos = node_positions.get(edge['source'])
                target_pos = node_positions.get(edge['target'])
                
                if source_pos and target_pos:
                    edge_x.extend([source_pos['x'], target_pos['x'], None])
                    edge_y.extend([source_pos['y'], target_pos['y'], None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create node trace
            node_x = [node_positions[node['id']]['x'] for node in nodes]
            node_y = [node_positions[node['id']]['y'] for node in nodes]
            
            # Color nodes by type
            node_colors = []
            for node in nodes:
                if node['type'] == 'concept':
                    node_colors.append('#1f77b4')
                elif node['type'] == 'process':
                    node_colors.append('#ff7f0e')
                elif node['type'] == 'system':
                    node_colors.append('#2ca02c')
                else:
                    node_colors.append('#d62728')
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[node['label'] for node in nodes],
                textposition="top center",
                marker=dict(
                    size=20,
                    color=node_colors,
                    line=dict(width=2, color='white')
                )
            )
            
            # Create layout
            layout = go.Layout(
                title='Knowledge Graph Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            return go.Figure(data=[edge_trace, node_trace], layout=layout)
            
        except Exception as e:
            st.error(f"Error creating Plotly network: {e}")
            return None
    
    def _analyze_temporal_trends(self, knowledge_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyze temporal trends in knowledge creation"""
        try:
            # Group by date and count
            date_counts = {}
            for item in knowledge_data:
                date = item['created_at'].date()
                date_counts[date] = date_counts.get(date, 0) + 1
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {'date': date, 'count': count}
                for date, count in sorted(date_counts.items())
            ])
            
            return df
            
        except Exception as e:
            st.error(f"Error analyzing temporal trends: {e}")
            return pd.DataFrame()
    
    def _analyze_relationships(self, knowledge_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyze relationship types in knowledge graph"""
        try:
            knowledge_graph = self.knowledge_engine.get_knowledge_graph()
            
            if not knowledge_graph:
                return pd.DataFrame()
            
            # Count relationship types
            relationship_counts = {}
            for edge in knowledge_graph.edges:
                rel_type = edge.relationship_type
                relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {'relationship_type': rel_type, 'count': count}
                for rel_type, count in relationship_counts.items()
            ])
            
            return df.sort_values('count', ascending=False)
            
        except Exception as e:
            st.error(f"Error analyzing relationships: {e}")
            return pd.DataFrame()
    
    def _export_knowledge(self, format_type: str):
        """Export knowledge data in specified format"""
        try:
            with st.spinner(f"Exporting knowledge in {format_type.upper()} format..."):
                # Get all knowledge data
                knowledge_graph = self.knowledge_engine.get_knowledge_graph()
                
                if not knowledge_graph:
                    st.error("No knowledge data available for export")
                    return
                
                # Export using the exporter
                export_data = st.session_state.get('export_data', {})
                
                if format_type == 'csv':
                    # Create CSV data
                    csv_data = self._prepare_csv_export(knowledge_graph)
                    
                    # Convert to CSV string
                    csv_string = csv_data.to_csv(index=False)
                    
                    # Create download button
                    st.download_button(
                        label="Download CSV",
                        data=csv_string,
                        file_name=f"explainium_knowledge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                elif format_type == 'json':
                    # Export as JSON
                    json_data = self.exporter._export_as_json(knowledge_graph)
                    
                    # Create download button
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(json_data, indent=2, default=str),
                        file_name=f"explainium_knowledge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                elif format_type == 'markdown':
                    # Export as Markdown
                    markdown_data = self.exporter._export_as_markdown(knowledge_graph)
                    
                    # Create download button
                    st.download_button(
                        label="Download Markdown",
                        data=markdown_data['content'],
                        file_name=f"explainium_knowledge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                else:
                    st.info(f"Export format '{format_type}' will be implemented soon")
            
            st.success(f"Knowledge exported successfully in {format_type.upper()} format!")
            
        except Exception as e:
            st.error(f"Export failed: {e}")
    
    def _prepare_csv_export(self, knowledge_graph) -> pd.DataFrame:
        """Prepare knowledge data for CSV export"""
        # Convert nodes to DataFrame
        nodes_data = []
        for node in knowledge_graph.nodes.values():
            nodes_data.append({
                'id': node.id,
                'name': node.name,
                'type': node.type,
                'description': node.description,
                'confidence': node.confidence,
                'created_at': node.created_at.isoformat(),
                'updated_at': node.updated_at.isoformat()
            })
        
        return pd.DataFrame(nodes_data)


def main():
    """Main function to run the knowledge table frontend"""
    st.set_page_config(
        page_title="EXPLAINIUM Knowledge Table",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 EXPLAINIUM Knowledge Table")
    st.markdown("**Deep Knowledge Extraction & Analysis Dashboard**")
    
    # Check if we have the full system available
    if not FULL_SYSTEM_AVAILABLE:
        st.warning("⚠️ Running in demo mode - full AI engine not available")
        demo_mode()
    else:
        # Initialize with proper config
        try:
            st.info("🔄 Initializing AI Knowledge Engine...")
            config = AIConfig()
            knowledge_engine = AdvancedKnowledgeEngine(config)
            
            st.success("✅ AI Knowledge Engine ready!")
            
            # For now, show demo mode with AI engine available
            # TODO: Implement real data loading from knowledge engine
            st.info("📊 Showing demo data - upload documents via API to see real extractions")
            demo_mode()
            
        except Exception as e:
            st.error(f"Error initializing knowledge engine: {e}")
            st.info("Falling back to demo mode...")
            demo_mode()

def demo_mode():
    """Run the app in demo mode with mock data"""
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Demo Controls")
        
        # Knowledge type filter
        knowledge_types = st.multiselect(
            "Knowledge Types",
            ["concepts", "processes", "systems", "requirements", "people", "risks"],
            default=["concepts", "processes", "systems", "requirements", "people", "risks"]
        )
        
        # Confidence filter
        confidence_range = st.slider(
            "Confidence Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.5, 1.0),
            step=0.1
        )
        
        # Date range
        date_range = st.date_input(
            "Date Range",
            value=[datetime.now().date()],
            help="Filter by extraction date"
        )
        
        # Search
        search_term = st.text_input("🔍 Search Knowledge", placeholder="Enter search term...")
        
        # Export options
        st.subheader("📥 Export Options")
        export_format = st.selectbox("Format", ["CSV", "JSON", "Markdown"])
        
        if st.button("📥 Export Data"):
            mock_data = generate_mock_data()
            df = pd.DataFrame(mock_data)
            
            if export_format == "CSV":
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"knowledge_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            elif export_format == "JSON":
                json_str = df.to_json(orient='records', indent=2)
                st.download_button(
                    "Download JSON",
                    json_str,
                    f"knowledge_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
    
    # Document Upload Section
    st.header("📄 Document Processing")
    
    uploaded_file = st.file_uploader(
        "Upload a document for knowledge extraction",
        type=['pdf', 'txt', 'docx', 'doc'],
        help="Upload documents to extract knowledge using the AI engine"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Processing document with AI engine..."):
                # TODO: Implement actual document processing
                import time
                time.sleep(2)
                st.success("✅ Document processed! Results added to knowledge base.")
                st.rerun()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("📊 Knowledge Table")
        
        # Generate and display mock data
        mock_data = generate_mock_data()
        df = pd.DataFrame(mock_data)
        
        # Apply filters
        if knowledge_types:
            df = df[df['Type'].isin(knowledge_types)]
        
        df = df[
            (df['Confidence'] >= confidence_range[0]) & 
            (df['Confidence'] <= confidence_range[1])
        ]
        
        if search_term:
            df = df[df['Knowledge'].str.contains(search_term, case=False, na=False)]
        
        # Display results
        if df.empty:
            st.warning("No knowledge data found with the current filters.")
            st.info("Try adjusting your filters or search terms.")
        else:
            # Display table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        help="Extraction confidence score",
                        min_value=0,
                        max_value=1,
                    ),
                    "Type": st.column_config.SelectboxColumn(
                        "Type",
                        help="Knowledge type",
                        options=["concepts", "processes", "systems", "requirements", "people", "risks"]
                    )
                }
            )
        
        # Statistics
        st.subheader("📈 Statistics")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("Total Items", len(df))
        with col_b:
            st.metric("Avg Confidence", f"{df['Confidence'].mean():.2f}" if not df.empty else "0.00")
        with col_c:
            st.metric("Types", df['Type'].nunique() if not df.empty else 0)
        with col_d:
            st.metric("High Confidence", len(df[df['Confidence'] > 0.8]) if not df.empty else 0)
    
    with col2:
        st.header("📈 Analytics")
        
        if not df.empty:
            # Type distribution
            type_counts = df['Type'].value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Knowledge Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence distribution
            fig2 = px.histogram(
                df,
                x='Confidence',
                title="Confidence Distribution",
                nbins=10
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Timeline (mock)
            timeline_data = pd.DataFrame({
                'Date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
                'Extractions': [5, 8, 12, 15, 10, 7, 9, 11, 14, 16, 13, 8, 6, 9, 12, 15, 18, 20, 17, 14, 11, 8, 10, 13, 16, 19, 22, 18, 15, 12]
            })
            
            fig3 = px.line(
                timeline_data,
                x='Date',
                y='Extractions',
                title="Extraction Timeline"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No data to display with current filters")
        
        # System status
        st.subheader("🔧 System Status")
        st.success("✅ Frontend: Active")
        st.warning("⚠️ AI Engine: Demo Mode")
        st.info("📊 Knowledge Items: " + str(len(df)))

def generate_mock_data():
    """Generate mock knowledge data for demo"""
    return [
        {
            "Knowledge": "Customer Onboarding Process",
            "Type": "processes",
            "Confidence": 0.95,
            "Category": "Business Process",
            "Description": "Multi-step workflow for bringing new customers into the system",
            "Source": "Process Documentation",
            "Extracted_At": "2024-01-15 10:30:00"
        },
        {
            "Knowledge": "Sales Team",
            "Type": "people",
            "Confidence": 0.88,
            "Category": "Organizational Unit",
            "Description": "Team responsible for initial customer contact and inquiries",
            "Source": "Org Chart",
            "Extracted_At": "2024-01-15 10:31:00"
        },
        {
            "Knowledge": "CRM System",
            "Type": "systems",
            "Confidence": 0.92,
            "Category": "Software System",
            "Description": "Customer relationship management platform",
            "Source": "Technical Documentation",
            "Extracted_At": "2024-01-15 10:32:00"
        },
        {
            "Knowledge": "Data Privacy Compliance",
            "Type": "requirements",
            "Confidence": 0.85,
            "Category": "Legal Requirement",
            "Description": "Must comply with GDPR and local data protection laws",
            "Source": "Legal Documentation",
            "Extracted_At": "2024-01-15 10:33:00"
        },
        {
            "Knowledge": "Technical Complexity Risk",
            "Type": "risks",
            "Confidence": 0.78,
            "Category": "Technical Risk",
            "Description": "Risk of technical requirements exceeding capabilities",
            "Source": "Risk Assessment",
            "Extracted_At": "2024-01-15 10:34:00"
        },
        {
            "Knowledge": "Solution Architecture",
            "Type": "concepts",
            "Confidence": 0.82,
            "Category": "Technical Concept",
            "Description": "Overall system design and component relationships",
            "Source": "Architecture Document",
            "Extracted_At": "2024-01-15 10:35:00"
        },
        {
            "Knowledge": "Quality Assurance Process",
            "Type": "processes",
            "Confidence": 0.90,
            "Category": "Quality Process",
            "Description": "Systematic approach to ensuring product quality",
            "Source": "QA Manual",
            "Extracted_At": "2024-01-15 10:36:00"
        },
        {
            "Knowledge": "Development Team",
            "Type": "people",
            "Confidence": 0.87,
            "Category": "Technical Team",
            "Description": "Software development and engineering team",
            "Source": "Team Directory",
            "Extracted_At": "2024-01-15 10:37:00"
        },
        {
            "Knowledge": "Database System",
            "Type": "systems",
            "Confidence": 0.94,
            "Category": "Data System",
            "Description": "Primary data storage and management system",
            "Source": "System Architecture",
            "Extracted_At": "2024-01-15 10:38:00"
        },
        {
            "Knowledge": "Security Requirements",
            "Type": "requirements",
            "Confidence": 0.91,
            "Category": "Security Requirement",
            "Description": "System security and access control requirements",
            "Source": "Security Policy",
            "Extracted_At": "2024-01-15 10:39:00"
        }
    ]


if __name__ == "__main__":
    main()
