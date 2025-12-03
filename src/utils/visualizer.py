"""
Visualization utilities for IPKE.
Generates interactive Procedural Knowledge Graphs using PyVis.
"""
import networkx as nx
from pyvis.network import Network
import textwrap
import tempfile
import os
from typing import Dict, Any, List, Optional, Tuple

from src.ai.types import ExtractionResult, ExtractedEntity

def _build_graph_from_result(result: ExtractionResult) -> nx.DiGraph:
    """Converts ExtractionResult into NetworkX graph."""
    G = nx.DiGraph()
    
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
            res_id = f"Res_{step_id}_{res_name.replace(' ', '_')}"
            G.add_node(res_id, type="Asset", label=res_name, title=f"Resource: {res_name}")
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
                G.add_node(gw_id, type="Gateway", label=gw_type, title=f"Gateway: {gw_type}")
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
    G = _build_graph_from_result(result)
    
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
        short_label = textwrap.shorten(label, width=15, placeholder="...")
        
        net.add_node(
            node,
            label=short_label,
            title=data.get("title", label),
            color=style["color"],
            shape=style["shape"],
            size=style.get("size", 15),
            physics=True, # Enable physics for spider layout
            font={"size": 14, "face": "arial"}
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

    # Configure physics for a spacious "spider web" layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "centralGravity": 0.005,
          "springLength": 200,
          "springConstant": 0.05,
          "damping": 0.4,
          "avoidOverlap": 0.5
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
        
    return html_content