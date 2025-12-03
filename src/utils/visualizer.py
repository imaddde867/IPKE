"""
Visualization utilities for IPKE.
Generates interactive Procedural Knowledge Graphs using PyVis.
"""
import networkx as nx
from pyvis.network import Network
import textwrap
import tempfile
import os
import re
from typing import Dict, Any, List, Optional, Tuple

from src.ai.types import ExtractionResult, ExtractedEntity


def _build_resource_catalog(result: ExtractionResult) -> Dict[str, str]:
    """
    Build a lookup dictionary from resource IDs to friendly canonical names.
    Parses resources_catalog from metadata if available.
    """
    catalog = {}
    if not result.metadata:
        return catalog
    
    resources_catalog = result.metadata.get("resources_catalog", {})
    if not resources_catalog:
        # Also check nested in relations or other keys
        resources_catalog = result.metadata.get("relations", {}).get("resources_catalog", {})
    
    # Parse tools, materials, documents, ppe, etc.
    for category in ["tools", "materials", "documents", "ppe"]:
        items = resources_catalog.get(category, [])
        for item in items:
            if isinstance(item, dict):
                res_id = item.get("id", "")
                canonical = item.get("canonical_name", "")
                if res_id and canonical:
                    catalog[res_id] = canonical
    
    return catalog


def _normalize_label(label: str, catalog: Dict[str, str]) -> str:
    """
    Normalize a label by:
    1. Looking up resource IDs in the catalog for friendly names
    2. Stripping technical prefixes (R_, T_, S_, etc.)
    3. Replacing underscores with spaces
    """
    if not label:
        return label
    
    # Check if this is a resource ID in our catalog
    if label in catalog:
        return catalog[label]
    
    # Strip common technical prefixes
    normalized = label
    prefix_patterns = [
        r'^R_',   # Resource prefix
        r'^T_',   # Tool prefix  
        r'^M_',   # Material prefix
        r'^Res_\w+_',  # Resource node prefix (Res_S1_)
        r'^Act_',  # Action prefix
        r'^Obj_\w+_',  # Object node prefix
    ]
    
    for pattern in prefix_patterns:
        normalized = re.sub(pattern, '', normalized)
    
    # Replace underscores with spaces
    normalized = normalized.replace('_', ' ')
    
    return normalized.strip()


def _build_graph_from_result(result: ExtractionResult, catalog: Dict[str, str] = None) -> nx.DiGraph:
    """Converts ExtractionResult into NetworkX graph."""
    G = nx.DiGraph()
    
    if catalog is None:
        catalog = {}
    
    # 1. Add Steps
    steps = result.steps if result.steps else []
    
    for step in steps:
        step_id = step.get("id", f"S{len(G.nodes)}")
        label = step.get("label", step_id)
        
        G.add_node(step_id, type="Step", label=label, title=textwrap.fill(label, 40))
        
        # Action
        action = step.get("action_verb")
        if action:
            act_id = f"Act_{step_id}"
            G.add_node(act_id, type="Action", label=action, title=f"Action: {action}")
            G.add_edge(step_id, act_id, relation="ACTION")
            
            # Object
            obj = step.get("action_object")
            if isinstance(obj, dict):
                obj_name = obj.get("canonical") or obj.get("surface_form")
            else:
                obj_name = obj
            
            if obj_name:
                obj_id = f"Obj_{step_id}_{str(obj_name).replace(' ', '_')}"
                G.add_node(obj_id, type="Asset", label=str(obj_name), title=f"Object: {obj_name}")
                G.add_edge(act_id, obj_id, relation="ACTS_ON")

        # Resources
        resources = step.get("resources", {})
        if isinstance(resources, dict):
            res_list = resources.get("tools", []) + resources.get("materials", [])
        elif isinstance(resources, list):
            res_list = resources
        else:
            res_list = []
            
        for res in res_list:
            res_name = res if isinstance(res, str) else res.get("name", str(res))
            # Resolve resource ID to friendly name using catalog
            friendly_name = _normalize_label(res_name, catalog)
            res_id = f"Res_{step_id}_{res_name.replace(' ', '_')}"
            G.add_node(res_id, type="Asset", label=friendly_name, title=f"Resource: {friendly_name}")
            G.add_edge(step_id, res_id, relation="REQUIRES")

    # 2. Sequence & Relations
    relations = {}
    if result.metadata:
        relations = result.metadata.get("relations", {})
    
    if relations and isinstance(relations, dict):
        # Process Gateways first
        gateways = relations.get("gateways", [])
        for gw in gateways:
            gw_id = gw.get("id")
            gw_type = gw.get("gateway_type", "XOR")
            if gw_id:
                # Use ? symbol for cleaner gateway display, with full type in tooltip
                G.add_node(gw_id, type="Gateway", label="?", title=f"Decision Point ({gw_type})")
                # Add branches as edges
                for branch_target in gw.get("branches", []):
                    G.add_edge(gw_id, branch_target, relation="NEXT")

        # Process Sequence
        sequence = relations.get("sequence", [])
        for link in sequence:
            u, v = link.get("from"), link.get("to")
            if u and v:
                G.add_edge(u, v, relation="NEXT")
    
    elif steps:
        # Fallback: linear assumption
        for i in range(len(steps) - 1):
            u = steps[i].get("id", f"S{i}")
            v = steps[i+1].get("id", f"S{i+1}")
            G.add_edge(u, v, relation="NEXT")

    return G

def generate_interactive_graph_html(result: ExtractionResult, height="600px", width="100%") -> str:
    """
    Generates the HTML for the interactive graph using a physics-based 'spider web' layout.
    Returns the HTML string.
    """
    # Build resource catalog for friendly name resolution
    catalog = _build_resource_catalog(result)
    
    G = _build_graph_from_result(result, catalog)
    
    if G.number_of_nodes() == 0:
        return "<div>No procedural steps extracted to visualize.</div>"

    # Convert to PyVis with layout=False to allow physics engine to handle positioning
    net = Network(height=height, width=width, directed=True, layout=False)
    
    # Colors & Shapes
    styles = {
        "Step": {"color": "#ADD8E6", "shape": "box", "size": 25},
        "Gateway": {"color": "#FFD700", "shape": "diamond", "size": 20},
        "Action": {"color": "#E6E6FA", "shape": "ellipse", "size": 15},
        "Asset": {"color": "#90EE90", "shape": "dot", "size": 10},
        "Constraint": {"color": "#FFB6C1", "shape": "dot", "size": 10}
    }
    
    for node, data in G.nodes(data=True):
        ntype = data.get("type", "Step")
        style = styles.get(ntype, styles["Step"])
        
        label = data.get("label", str(node))
        # Use textwrap.fill for multi-line labels instead of truncating with "..."
        # Wrap at 20 characters for readable multi-line display
        wrapped_label = textwrap.fill(label, width=20)
        
        net.add_node(
            node,
            label=wrapped_label,
            title=data.get("title", label),
            color=style["color"],
            shape=style["shape"],
            size=style.get("size", 15),
            physics=True, # Enable physics for spider layout
            font={"size": 14, "face": "arial", "multi": True}  # multi: True enables line breaks
        )

    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "")
        color = "black"
        dashes = False
        width = 1
        
        if rel == "NEXT":
            color = "#455a64"
            width = 2
        elif rel == "REQUIRES":
            color = "#90a4ae"
            dashes = True
        else:
            color = "#b0bec5"
            
        net.add_edge(u, v, title=rel, color=color, width=width, dashes=dashes)

    # Configure physics for a denser "spider web" layout with adjusted parameters
    # - gravitationalConstant: -80 (was -100) keeps nodes closer together
    # - springLength: 100 (was 200) creates shorter edges
    # - avoidOverlap: 0.3 (was 0.5) allows nodes to be positioned closer while preventing overlap
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.005,
          "springLength": 100,
          "springConstant": 0.05,
          "damping": 0.4,
          "avoidOverlap": 0.3
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 25
        }
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      }
    }
    """)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        tmp.seek(0)
        html_content = tmp.read().decode("utf-8")
    
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    
    # Inject JavaScript to stop physics simulation on node click
    stop_physics_script = """
    <script type="text/javascript">
    document.addEventListener('DOMContentLoaded', function() {
        // Wait for the network to be initialized
        var checkNetwork = setInterval(function() {
            if (typeof network !== 'undefined' && network !== null) {
                clearInterval(checkNetwork);
                // Stop physics when a node is clicked
                network.on('click', function(params) {
                    if (params.nodes.length > 0) {
                        network.setOptions({ physics: { enabled: false } });
                    }
                });
                // Also stop on drag end for better UX
                network.on('dragEnd', function(params) {
                    if (params.nodes.length > 0) {
                        network.setOptions({ physics: { enabled: false } });
                    }
                });
            }
        }, 100);
    });
    </script>
    """
    
    # Insert the script before closing </body> tag
    html_content = html_content.replace('</body>', stop_physics_script + '\n</body>')
    
    # Add fullscreen-friendly styles to ensure the graph fills the viewport
    fullscreen_styles = """
    <style>
    html, body {
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }
    #mynetwork {
        width: 100% !important;
        height: 100vh !important;
        border: none !important;
    }
    </style>
    """
    html_content = html_content.replace('</head>', fullscreen_styles + '\n</head>')
        
    return html_content