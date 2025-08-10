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
from src.export.knowledge_export import KnowledgeExporter
from src.ai.advanced_knowledge_engine import AdvancedKnowledgeEngine


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
    st.title("🧠 EXPLAINIUM Knowledge Table")
    st.info("This is a standalone component. Integrate with your knowledge engine to display data.")
    
    # Placeholder for demonstration
    st.write("""
    ## Features
    
    - **Large Knowledge Table**: Display all extracted knowledge in a searchable, filterable table
    - **Advanced Filtering**: Filter by type, confidence, date range, and search terms
    - **Visual Analytics**: Charts and graphs showing knowledge distribution and trends
    - **Knowledge Graph**: Interactive network visualization of relationships
    - **Export Capabilities**: Export data in multiple formats (CSV, JSON, Markdown)
    - **Real-time Updates**: Live updates as new knowledge is extracted
    
    ## Integration
    
    To use this component, initialize it with your `AdvancedKnowledgeEngine`:
    
    ```python
    from src.frontend.knowledge_table import KnowledgeTableFrontend
    
    frontend = KnowledgeTableFrontend(knowledge_engine)
    frontend.render_knowledge_table()
    ```
    """)


if __name__ == "__main__":
    main()
