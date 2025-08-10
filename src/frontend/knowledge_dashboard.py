"""
EXPLAINIUM - Knowledge Dashboard

Clean and simple frontend to display extracted knowledge in a large table format
with filtering, sorting, and basic visualization capabilities.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

# Dash components
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Internal imports
from src.logging_config import get_logger
from src.export.knowledge_export import KnowledgeExporter

logger = get_logger(__name__)


class KnowledgeDashboard:
    """
    Simple and clean dashboard for displaying extracted knowledge
    """
    
    def __init__(self, port: int = 8050, debug: bool = False):
        self.port = port
        self.debug = debug
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'
        ])
        
        # Initialize with empty data
        self.knowledge_data = pd.DataFrame()
        self.stats = {}
        
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info("Knowledge Dashboard initialized")
    
    def load_knowledge_data(self, knowledge_results: Dict[str, Any]):
        """Load knowledge extraction results into the dashboard"""
        try:
            # Convert knowledge results to DataFrame
            self.knowledge_data = self._convert_to_dataframe(knowledge_results)
            self.stats = self._calculate_stats(knowledge_results)
            
            logger.info(f"Loaded {len(self.knowledge_data)} knowledge items into dashboard")
            
        except Exception as e:
            logger.error(f"Error loading knowledge data: {e}")
            self.knowledge_data = pd.DataFrame()
            self.stats = {}
    
    def _convert_to_dataframe(self, knowledge_results: Dict[str, Any]) -> pd.DataFrame:
        """Convert knowledge results to a flattened DataFrame for table display"""
        table_data = []
        
        # Add entities
        for entity in knowledge_results.get('entities', []):
            row = {
                'ID': getattr(entity, 'id', ''),
                'Type': 'Entity',
                'Category': getattr(entity, 'type', 'unknown'),
                'Name': getattr(entity, 'name', ''),
                'Description': getattr(entity, 'description', ''),
                'Confidence': getattr(entity, 'confidence', 0),
                'Source Documents': '; '.join(getattr(entity, 'source_documents', [])),
                'Created At': getattr(entity, 'created_at', datetime.now()).strftime('%Y-%m-%d %H:%M:%S') if hasattr(entity, 'created_at') and entity.created_at else '',
                'Properties': json.dumps(getattr(entity, 'properties', {}), indent=2)
            }
            table_data.append(row)
        
        # Add processes
        for process in knowledge_results.get('processes', []):
            row = {
                'ID': getattr(process, 'id', ''),
                'Type': 'Process',
                'Category': 'business_process',
                'Name': getattr(process, 'name', ''),
                'Description': getattr(process, 'description', ''),
                'Confidence': getattr(process, 'complexity_score', 0),
                'Source Documents': 'process_extraction',
                'Created At': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Properties': json.dumps({
                    'steps': getattr(process, 'steps', []),
                    'roles_involved': getattr(process, 'roles_involved', []),
                    'inputs': getattr(process, 'inputs', []),
                    'outputs': getattr(process, 'outputs', [])
                }, indent=2)
            }
            table_data.append(row)
        
        # Add relationships
        for relationship in knowledge_results.get('relationships', []):
            row = {
                'ID': f"rel_{getattr(relationship, 'source_id', '')}_{getattr(relationship, 'target_id', '')}",
                'Type': 'Relationship',
                'Category': getattr(relationship, 'relationship_type', 'unknown'),
                'Name': f"{getattr(relationship, 'source_id', '')} → {getattr(relationship, 'target_id', '')}",
                'Description': getattr(relationship, 'context', ''),
                'Confidence': getattr(relationship, 'confidence', 0),
                'Source Documents': 'relationship_extraction',
                'Created At': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Properties': json.dumps({
                    'strength': getattr(relationship, 'strength', 0),
                    'properties': getattr(relationship, 'properties', {})
                }, indent=2)
            }
            table_data.append(row)
        
        # Add tacit knowledge
        for tacit in knowledge_results.get('tacit_knowledge', []):
            row = {
                'ID': getattr(tacit, 'id', ''),
                'Type': 'Tacit Knowledge',
                'Category': getattr(tacit, 'knowledge_type', 'unknown'),
                'Name': f"Tacit: {getattr(tacit, 'knowledge_type', 'Unknown')}",
                'Description': getattr(tacit, 'description', ''),
                'Confidence': getattr(tacit, 'confidence', 0),
                'Source Documents': 'tacit_extraction',
                'Created At': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Properties': json.dumps({
                    'context': getattr(tacit, 'context', ''),
                    'evidence': getattr(tacit, 'evidence', []),
                    'affected_processes': getattr(tacit, 'affected_processes', [])
                }, indent=2)
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Sort by confidence descending
        if not df.empty:
            df = df.sort_values(['Confidence'], ascending=False)
            
        return df
    
    def _calculate_stats(self, knowledge_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for the dashboard"""
        stats = {
            'total_entities': len(knowledge_results.get('entities', [])),
            'total_processes': len(knowledge_results.get('processes', [])),
            'total_relationships': len(knowledge_results.get('relationships', [])),
            'total_tacit_knowledge': len(knowledge_results.get('tacit_knowledge', [])),
            'avg_confidence': 0.0,
            'entity_types': {},
            'relationship_types': {},
            'high_confidence_items': 0
        }
        
        # Calculate entity type distribution
        for entity in knowledge_results.get('entities', []):
            entity_type = getattr(entity, 'type', 'unknown')
            stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
        
        # Calculate relationship type distribution
        for rel in knowledge_results.get('relationships', []):
            rel_type = getattr(rel, 'relationship_type', 'unknown')
            stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + 1
        
        # Calculate average confidence
        all_confidences = []
        for entity in knowledge_results.get('entities', []):
            if hasattr(entity, 'confidence'):
                all_confidences.append(entity.confidence)
        
        for rel in knowledge_results.get('relationships', []):
            if hasattr(rel, 'confidence'):
                all_confidences.append(rel.confidence)
        
        for tacit in knowledge_results.get('tacit_knowledge', []):
            if hasattr(tacit, 'confidence'):
                all_confidences.append(tacit.confidence)
        
        if all_confidences:
            stats['avg_confidence'] = sum(all_confidences) / len(all_confidences)
            stats['high_confidence_items'] = len([c for c in all_confidences if c > 0.8])
        
        return stats
    
    def _setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("EXPLAINIUM Knowledge Dashboard", 
                       className="text-center text-primary mb-4"),
                html.P("Advanced AI-powered knowledge extraction and visualization", 
                       className="text-center text-muted mb-4")
            ], className="container-fluid bg-light py-4"),
            
            # Statistics cards
            html.Div([
                html.Div([
                    self._create_stat_card("Total Entities", "total-entities", "primary"),
                    self._create_stat_card("Business Processes", "total-processes", "success"),
                    self._create_stat_card("Relationships", "total-relationships", "info"),
                    self._create_stat_card("Tacit Knowledge", "total-tacit", "warning"),
                    self._create_stat_card("Avg Confidence", "avg-confidence", "secondary"),
                    self._create_stat_card("High Confidence", "high-confidence", "danger")
                ], className="row mb-4")
            ], className="container-fluid"),
            
            # Filters and controls
            html.Div([
                html.Div([
                    html.H4("Filters & Controls", className="mb-3"),
                    
                    html.Div([
                        html.Div([
                            html.Label("Type Filter:", className="form-label"),
                            dcc.Dropdown(
                                id="type-filter",
                                options=[
                                    {"label": "All Types", "value": "all"},
                                    {"label": "Entities", "value": "Entity"},
                                    {"label": "Processes", "value": "Process"},
                                    {"label": "Relationships", "value": "Relationship"},
                                    {"label": "Tacit Knowledge", "value": "Tacit Knowledge"}
                                ],
                                value="all",
                                className="mb-3"
                            )
                        ], className="col-md-3"),
                        
                        html.Div([
                            html.Label("Category Filter:", className="form-label"),
                            dcc.Dropdown(
                                id="category-filter",
                                options=[{"label": "All Categories", "value": "all"}],
                                value="all",
                                className="mb-3"
                            )
                        ], className="col-md-3"),
                        
                        html.Div([
                            html.Label("Confidence Threshold:", className="form-label"),
                            dcc.Slider(
                                id="confidence-slider",
                                min=0.0,
                                max=1.0,
                                step=0.1,
                                value=0.0,
                                marks={i/10: f"{i/10:.1f}" for i in range(0, 11, 2)},
                                className="mb-3"
                            )
                        ], className="col-md-4"),
                        
                        html.Div([
                            html.Label("Search:", className="form-label"),
                            dcc.Input(
                                id="search-input",
                                type="text",
                                placeholder="Search names and descriptions...",
                                className="form-control mb-3"
                            )
                        ], className="col-md-2")
                    ], className="row")
                ], className="card-body")
            ], className="container-fluid"),
            
            # Main data table
            html.Div([
                html.Div([
                    html.H4("Knowledge Extraction Results", className="mb-3"),
                    html.Div(id="table-info", className="mb-3"),
                    
                    dash_table.DataTable(
                        id="knowledge-table",
                        columns=[
                            {"name": "Type", "id": "Type", "type": "text"},
                            {"name": "Category", "id": "Category", "type": "text"},
                            {"name": "Name", "id": "Name", "type": "text"},
                            {"name": "Description", "id": "Description", "type": "text"},
                            {"name": "Confidence", "id": "Confidence", "type": "numeric", "format": {"specifier": ".2f"}},
                            {"name": "Source", "id": "Source Documents", "type": "text"},
                            {"name": "Created", "id": "Created At", "type": "text"}
                        ],
                        data=[],
                        page_size=50,
                        sort_action="native",
                        sort_mode="multi",
                        filter_action="native",
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'fontFamily': 'Arial, sans-serif',
                            'fontSize': '14px',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                            'maxWidth': '300px'
                        },
                        style_header={
                            'backgroundColor': '#007bff',
                            'color': 'white',
                            'fontWeight': 'bold',
                            'textAlign': 'center'
                        },
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{Type} = Entity'},
                                'backgroundColor': '#e3f2fd'
                            },
                            {
                                'if': {'filter_query': '{Type} = Process'},
                                'backgroundColor': '#e8f5e8'
                            },
                            {
                                'if': {'filter_query': '{Type} = Relationship'},
                                'backgroundColor': '#fff3e0'
                            },
                            {
                                'if': {'filter_query': '{Type} = "Tacit Knowledge"'},
                                'backgroundColor': '#fce4ec'
                            },
                            {
                                'if': {'filter_query': '{Confidence} > 0.8'},
                                'fontWeight': 'bold',
                                'border': '2px solid #28a745'
                            }
                        ],
                        tooltip_data=[],
                        tooltip_duration=None,
                        export_format="csv",
                        export_headers="display"
                    )
                ], className="card-body")
            ], className="container-fluid"),
            
            # Visualizations
            html.Div([
                html.Div([
                    html.H4("Visualizations", className="mb-3"),
                    
                    html.Div([
                        html.Div([
                            dcc.Graph(id="type-distribution-chart")
                        ], className="col-md-6"),
                        
                        html.Div([
                            dcc.Graph(id="confidence-distribution-chart")
                        ], className="col-md-6")
                    ], className="row"),
                    
                    html.Div([
                        html.Div([
                            dcc.Graph(id="category-distribution-chart")
                        ], className="col-md-12")
                    ], className="row mt-4")
                ], className="card-body")
            ], className="container-fluid mt-4"),
            
            # Detail modal for selected items
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.H5("Item Details", className="modal-title"),
                            html.Button("×", type="button", className="btn-close", 
                                      **{"data-bs-dismiss": "modal"})
                        ], className="modal-header"),
                        
                        html.Div([
                            html.Div(id="detail-content")
                        ], className="modal-body"),
                        
                        html.Div([
                            html.Button("Close", type="button", className="btn btn-secondary",
                                      **{"data-bs-dismiss": "modal"})
                        ], className="modal-footer")
                    ], className="modal-content")
                ], className="modal-dialog modal-lg")
            ], className="modal fade", id="detail-modal", tabindex="-1"),
            
            # Storage for data
            dcc.Store(id="knowledge-data-store")
        ])
    
    def _create_stat_card(self, title: str, element_id: str, color: str) -> html.Div:
        """Create a statistics card"""
        return html.Div([
            html.Div([
                html.H5(title, className="card-title text-center"),
                html.H2("0", id=element_id, className=f"text-center text-{color}")
            ], className="card-body")
        ], className=f"col-md-2 mb-3")
    
    def _setup_callbacks(self):
        """Setup Dash callbacks for interactivity"""
        
        @self.app.callback(
            [Output("knowledge-table", "data"),
             Output("knowledge-table", "tooltip_data"),
             Output("table-info", "children"),
             Output("category-filter", "options")],
            [Input("type-filter", "value"),
             Input("category-filter", "value"),
             Input("confidence-slider", "value"),
             Input("search-input", "value"),
             Input("knowledge-data-store", "data")]
        )
        def update_table(type_filter, category_filter, confidence_threshold, search_text, stored_data):
            if stored_data is None or len(stored_data) == 0:
                return [], [], "No data available", [{"label": "All Categories", "value": "all"}]
            
            df = pd.DataFrame(stored_data)
            
            # Apply filters
            filtered_df = df.copy()
            
            # Type filter
            if type_filter != "all":
                filtered_df = filtered_df[filtered_df["Type"] == type_filter]
            
            # Category filter
            if category_filter != "all":
                filtered_df = filtered_df[filtered_df["Category"] == category_filter]
            
            # Confidence threshold
            filtered_df = filtered_df[filtered_df["Confidence"] >= confidence_threshold]
            
            # Search filter
            if search_text:
                search_mask = (
                    filtered_df["Name"].str.contains(search_text, case=False, na=False) |
                    filtered_df["Description"].str.contains(search_text, case=False, na=False)
                )
                filtered_df = filtered_df[search_mask]
            
            # Update category options based on current type filter
            if type_filter != "all":
                categories = df[df["Type"] == type_filter]["Category"].unique()
            else:
                categories = df["Category"].unique()
            
            category_options = [{"label": "All Categories", "value": "all"}]
            category_options.extend([{"label": cat, "value": cat} for cat in sorted(categories)])
            
            # Prepare tooltip data
            tooltip_data = []
            for _, row in filtered_df.iterrows():
                tooltip_data.append({
                    "Properties": {"value": row["Properties"], "type": "text"}
                })
            
            # Table info
            info_text = f"Showing {len(filtered_df)} of {len(df)} items"
            if confidence_threshold > 0:
                info_text += f" (confidence ≥ {confidence_threshold:.1f})"
            
            return filtered_df.to_dict('records'), tooltip_data, info_text, category_options
        
        @self.app.callback(
            [Output("total-entities", "children"),
             Output("total-processes", "children"),
             Output("total-relationships", "children"),
             Output("total-tacit", "children"),
             Output("avg-confidence", "children"),
             Output("high-confidence", "children")],
            [Input("knowledge-data-store", "data")]
        )
        def update_stats(stored_data):
            if stored_data is None:
                return "0", "0", "0", "0", "0.00", "0"
            
            df = pd.DataFrame(stored_data)
            
            entity_count = len(df[df["Type"] == "Entity"])
            process_count = len(df[df["Type"] == "Process"])
            relationship_count = len(df[df["Type"] == "Relationship"])
            tacit_count = len(df[df["Type"] == "Tacit Knowledge"])
            
            avg_confidence = df["Confidence"].mean() if not df.empty else 0
            high_confidence = len(df[df["Confidence"] > 0.8])
            
            return (
                str(entity_count),
                str(process_count),
                str(relationship_count),
                str(tacit_count),
                f"{avg_confidence:.2f}",
                str(high_confidence)
            )
        
        @self.app.callback(
            Output("type-distribution-chart", "figure"),
            [Input("knowledge-data-store", "data")]
        )
        def update_type_chart(stored_data):
            if stored_data is None:
                return px.bar(title="Type Distribution")
            
            df = pd.DataFrame(stored_data)
            type_counts = df["Type"].value_counts()
            
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Knowledge Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            return fig
        
        @self.app.callback(
            Output("confidence-distribution-chart", "figure"),
            [Input("knowledge-data-store", "data")]
        )
        def update_confidence_chart(stored_data):
            if stored_data is None:
                return px.histogram(title="Confidence Distribution")
            
            df = pd.DataFrame(stored_data)
            
            fig = px.histogram(
                df,
                x="Confidence",
                nbins=20,
                title="Confidence Score Distribution",
                color="Type",
                barmode="overlay"
            )
            fig.update_layout(height=400)
            return fig
        
        @self.app.callback(
            Output("category-distribution-chart", "figure"),
            [Input("knowledge-data-store", "data")]
        )
        def update_category_chart(stored_data):
            if stored_data is None:
                return px.bar(title="Category Distribution")
            
            df = pd.DataFrame(stored_data)
            category_counts = df["Category"].value_counts().head(20)  # Top 20 categories
            
            fig = px.bar(
                x=category_counts.values,
                y=category_counts.index,
                orientation='h',
                title="Top Categories",
                labels={'x': 'Count', 'y': 'Category'}
            )
            fig.update_layout(height=600)
            return fig
    
    def update_data(self, knowledge_results: Dict[str, Any]):
        """Update the dashboard with new knowledge data"""
        try:
            # Load new data
            self.load_knowledge_data(knowledge_results)
            
            # Update the stored data (this would trigger callbacks in a real app)
            return self.knowledge_data.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
            return []
    
    def run(self, host: str = "0.0.0.0", port: Optional[int] = None):
        """Run the dashboard server"""
        port = port or self.port
        
        logger.info(f"Starting Knowledge Dashboard on http://{host}:{port}")
        
        self.app.run_server(
            host=host,
            port=port,
            debug=self.debug,
            dev_tools_hot_reload=False
        )
    
    def get_app(self):
        """Get the Dash app instance for external server integration"""
        return self.app


def create_knowledge_dashboard(knowledge_results: Dict[str, Any], 
                             port: int = 8050, 
                             debug: bool = False) -> KnowledgeDashboard:
    """
    Create and initialize a knowledge dashboard with data
    """
    dashboard = KnowledgeDashboard(port=port, debug=debug)
    dashboard.load_knowledge_data(knowledge_results)
    
    # Set initial data in the store
    dashboard.app.layout.children.append(
        dcc.Store(
            id="knowledge-data-store",
            data=dashboard.knowledge_data.to_dict('records')
        )
    )
    
    return dashboard


if __name__ == "__main__":
    # Example usage
    sample_data = {
        'entities': [],
        'processes': [],
        'relationships': [],
        'tacit_knowledge': []
    }
    
    dashboard = create_knowledge_dashboard(sample_data)
    dashboard.run(debug=True)