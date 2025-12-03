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

# --- Layout Logic (Adapted from generate_large_pkg.py) ---

def _calculate_spine_layout(G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    """
    Calculates x,y coordinates for a 'spine' layout.
    Spine (Steps) runs horizontally.
    Actions/Objects above.
    Resources/Constraints below.
    """
    pos = {}
    
    # Identify spine nodes (Steps and Gateways)
    spine_nodes = [n for n, d in G.nodes(data=True) if d.get("type") in ("Step", "Gateway")]
    
    if not spine_nodes:
        return nx.spring_layout(G) # Fallback

    # Determine start node (try S1 or first found)
    start_node = "S1"
    if start_node not in G:
        # Try to find a node with no incoming NEXT edges
        candidates = [n for n in spine_nodes if G.in_degree(n) == 0]
        if candidates:
            start_node = candidates[0]
        else:
            if spine_nodes:
                start_node = spine_nodes[0]
            else:
                return nx.spring_layout(G)

    # BFS for Rank/Order along spine
    ranks = {}
    queue = [(start_node, 0)]
    visited = {start_node}
    ranks[start_node] = 0
    
    while queue:
        curr, rank = queue.pop(0)
        # Next spine nodes
        next_nodes = [v for u, v, d in G.edges(data=True) 
                      if u == curr and d.get("relation") == "NEXT"]
        
        for nxt in next_nodes:
            if nxt not in visited:
                visited.add(nxt)
                ranks[nxt] = rank + 1
                queue.append((nxt, rank + 1))
    
    # Sort spine
    sorted_spine = sorted(ranks.keys(), key=lambda k: ranks[k])
    
    # Configuration
    x_cursor = 0
    base_y = 0
    step_width_base = 200  # Increased for PyVis pixel coordinates
    satellite_spacing = 120 
    y_level_spacing = 150
    
    # Calculate positions
    for node in sorted_spine:
        neighbors = list(G.neighbors(node))
        actions = []
        resources = []
        params = []
        
        for nbr in neighbors:
            if nbr in ranks: continue # Skip next spine nodes
            
            etype = G.nodes[nbr].get("type")
            relation = G.edges[node, nbr].get("relation")
            
            if etype == "Action":
                actions.append(nbr)
            elif etype == "Asset" and relation == "REQUIRES":
                resources.append(nbr)
            elif etype == "Constraint":
                params.append(nbr)
        
        # Calculate widths
        top_width = 0
        if actions:
            for act in actions:
                act_objs = [an for an in G.neighbors(act) if G.nodes[an].get("type") == "Asset"]
                act_width = max(1, len(act_objs)) * satellite_spacing
                top_width += act_width
        
        bottom_count = len(resources) + len(params)
        bottom_width = bottom_count * satellite_spacing
        
        total_width = max(step_width_base, top_width, bottom_width)
        
        # Center X
        center_x = x_cursor + (total_width / 2)
        pos[node] = (center_x, base_y)
        
        # --- TOP SECTOR (Actions) ---
        current_act_x = center_x - (top_width / 2) + (satellite_spacing / 2)
        for act in actions:
            act_objs = [an for an in G.neighbors(act) if G.nodes[an].get("type") == "Asset"]
            n_objs = len(act_objs)
            width_this_act = max(1, n_objs) * satellite_spacing
            
            act_center_x = current_act_x + (width_this_act / 2) - (satellite_spacing / 2)
            pos[act] = (act_center_x, base_y - y_level_spacing) 
            
            # Objects above action
            obj_start_x = current_act_x
            for i, obj in enumerate(act_objs):
                pos[obj] = (obj_start_x + (i * satellite_spacing), base_y - (y_level_spacing * 2))
            
            current_act_x += width_this_act

        # --- BOTTOM SECTOR (Resources/Params) ---
        start_res_x = center_x - (bottom_width / 2) + (satellite_spacing / 2)
        current_res_x = start_res_x
        
        for i, res in enumerate(resources):
            pos[res] = (current_res_x, base_y + y_level_spacing)
            current_res_x += satellite_spacing
            
        for i, p in enumerate(params):
            pos[p] = (current_res_x, base_y + (y_level_spacing * 1.5) if len(resources)%2==0 else base_y + y_level_spacing)
            current_res_x += satellite_spacing
            
        x_cursor += total_width + 100 # Gap between steps

    return pos

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
        # Could be dict or list depending on extraction quality
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
    # Check metadata for explicit relation structure
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
    Generates the HTML for the interactive graph.
    Returns the HTML string.
    """
    G = _build_graph_from_result(result)
    
    if G.number_of_nodes() == 0:
        return "<div>No procedural steps extracted to visualize.</div>"

    # Calculate layout
    pos = _calculate_spine_layout(G)
    
    # Convert to PyVis
    net = Network(height=height, width=width, directed=True, layout=False)
    
    # Colors & Shapes
    styles = {
        "Step": {"color": "#ADD8E6", "shape": "box"},
        "Gateway": {"color": "#FFD700", "shape": "diamond"},
        "Action": {"color": "#E6E6FA", "shape": "ellipse"},
        "Asset": {"color": "#90EE90", "shape": "box"},
        "Constraint": {"color": "#FFB6C1", "shape": "box"}
    }
    
    for node, data in G.nodes(data=True):
        x, y = pos.get(node, (0, 0))
        ntype = data.get("type", "Step")
        style = styles.get(ntype, styles["Step"])
        
        label = data.get("label", str(node))
        # Shorten label for visual clarity
        short_label = textwrap.shorten(label, width=15, placeholder="...")
        
        net.add_node(
            node,
            label=short_label,
            title=data.get("title", label), # Tooltip
            color=style["color"],
            shape=style["shape"],
            x=x, 
            y=y,
            physics=False, # Fixed position for spine layout
            font={"size": 16, "face": "arial"}
        )

    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "")
        color = "black"
        dashes = False
        
        if rel == "NEXT":
            color = "#455a64"
            width = 2
        elif rel == "REQUIRES":
            color = "#90a4ae"
            dashes = True
            width = 1
        else:
            color = "#b0bec5"
            width = 1
            
        net.add_edge(u, v, title=rel, color=color, width=width, dashes=dashes)

    # Configure options
    net.set_options("""
    {
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      },
      "physics": {
        "enabled": false
      }
    }
    """)
    
    # Return HTML
    # PyVis write_html writes to file. We want string.
    # We can write to temp file and read it back.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        tmp.seek(0)
        html_content = tmp.read().decode("utf-8")
    
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
        
    return html_content