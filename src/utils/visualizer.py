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
    """Converts ExtractionResult into NetworkX graph with a cleaner, more readable structure."""
    G = nx.DiGraph()
    
    if catalog is None:
        catalog = {}
    
    # 1. Add Steps as primary nodes
    steps = result.steps if result.steps else []
    
    for step in steps:
        step_id = step.get("id", f"S{len(G.nodes)}")
        label = step.get("label", step_id)
        action = step.get("action_verb", "")
        
        # Build a rich tooltip with all step details
        tooltip_parts = [f"<b>Step {step_id}</b>", f"<br><i>{label}</i>"]
        
        if action:
            tooltip_parts.append(f"<br><br><b>Action:</b> {action.upper()}")
        
        # Get object
        obj = step.get("action_object")
        if isinstance(obj, dict):
            obj_name = obj.get("canonical") or obj.get("surface_form")
        else:
            obj_name = obj
        if obj_name:
            tooltip_parts.append(f"<br><b>Target:</b> {obj_name}")
        
        # Get resources and resolve to friendly names
        resources = step.get("resources", {})
        if isinstance(resources, dict):
            tools = resources.get("tools", [])
            materials = resources.get("materials", [])
        else:
            tools, materials = [], []
        
        if tools:
            tool_names = [_normalize_label(t if isinstance(t, str) else t.get("name", str(t)), catalog) for t in tools]
            tooltip_parts.append(f"<br><b>Tools:</b> {', '.join(tool_names)}")
        
        if materials:
            mat_names = [_normalize_label(m if isinstance(m, str) else m.get("name", str(m)), catalog) for m in materials]
            tooltip_parts.append(f"<br><b>Materials:</b> {', '.join(mat_names)}")
        
        # Parameters
        params = step.get("parameters", [])
        if params:
            param_strs = [f"{p.get('name', '?')}: {p.get('value', '?')}" for p in params if isinstance(p, dict)]
            if param_strs:
                tooltip_parts.append(f"<br><b>Parameters:</b> {', '.join(param_strs)}")
        
        tooltip = "".join(tooltip_parts)
        
        # Create a concise but readable label: Action + short description
        if action:
            short_label = f"{action.upper()}\n{textwrap.fill(label, width=25)}"
        else:
            short_label = textwrap.fill(label, width=25)
        
        # Check flags for styling
        is_safety = step.get("flags", {}).get("safety_critical", False)
        
        G.add_node(
            step_id, 
            type="Step", 
            label=short_label, 
            title=tooltip,
            safety_critical=is_safety,
            has_resources=bool(tools or materials)
        )

    # 2. Sequence & Relations
    relations = {}
    if result.metadata:
        relations = result.metadata.get("relations", {})
    
    if relations and isinstance(relations, dict):
        # Process Gateways
        gateways = relations.get("gateways", [])
        for gw in gateways:
            gw_id = gw.get("id")
            gw_type = gw.get("gateway_type", "XOR")
            guard = gw.get("guard", {})
            condition = guard.get("condition", "") if isinstance(guard, dict) else ""
            
            if gw_id:
                tooltip = f"<b>Decision Point</b><br>Type: {gw_type}"
                if condition:
                    tooltip += f"<br>Condition: {condition}"
                
                G.add_node(gw_id, type="Gateway", label=f"⬦\n{gw_type}", title=tooltip)
                
                # Add branches as edges
                for branch_target in gw.get("branches", []):
                    G.add_edge(gw_id, branch_target, relation="BRANCH")

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
    Generates the HTML for an interactive, readable Procedural Knowledge Graph.
    Designed for demo/presentation quality with clear visual hierarchy.
    """
    # Build resource catalog for friendly name resolution
    catalog = _build_resource_catalog(result)
    
    G = _build_graph_from_result(result, catalog)
    
    if G.number_of_nodes() == 0:
        return "<div style='padding: 40px; text-align: center; font-family: Arial;'><h2>No procedural steps extracted to visualize.</h2></div>"

    # Use hierarchical layout for clearer flow
    net = Network(height=height, width=width, directed=True, layout=False)
    
    # Enhanced color palette - professional and readable
    COLORS = {
        "step_normal": {
            "background": "#4A90D9",  # Professional blue
            "border": "#2E6AB3",
            "highlight": {"background": "#6BA3E0", "border": "#4A90D9"}
        },
        "step_safety": {
            "background": "#E74C3C",  # Warning red for safety-critical
            "border": "#C0392B",
            "highlight": {"background": "#EC7063", "border": "#E74C3C"}
        },
        "step_resources": {
            "background": "#27AE60",  # Green for steps with resources
            "border": "#1E8449",
            "highlight": {"background": "#52BE80", "border": "#27AE60"}
        },
        "gateway": {
            "background": "#F39C12",  # Orange for decisions
            "border": "#D68910",
            "highlight": {"background": "#F5B041", "border": "#F39C12"}
        }
    }
    
    for node, data in G.nodes(data=True):
        ntype = data.get("type", "Step")
        label = data.get("label", str(node))
        
        if ntype == "Gateway":
            net.add_node(
                node,
                label=label,
                title=data.get("title", label),
                color=COLORS["gateway"],
                shape="diamond",
                size=50,
                font={"size": 18, "face": "Arial", "color": "#FFFFFF", "bold": True, "multi": True},
                borderWidth=3,
                shadow=True
            )
        else:
            # Determine step color based on attributes
            if data.get("safety_critical"):
                color = COLORS["step_safety"]
            elif data.get("has_resources"):
                color = COLORS["step_resources"]
            else:
                color = COLORS["step_normal"]
            
            net.add_node(
                node,
                label=label,
                title=data.get("title", label),
                color=color,
                shape="box",
                size=40,
                font={"size": 16, "face": "Arial", "color": "#FFFFFF", "multi": True, "align": "center"},
                borderWidth=3,
                borderWidthSelected=5,
                shadow={"enabled": True, "size": 10, "x": 3, "y": 3},
                margin={"top": 15, "bottom": 15, "left": 15, "right": 15},
                widthConstraint={"minimum": 180, "maximum": 280}
            )

    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "NEXT")
        
        if rel == "NEXT":
            net.add_edge(
                u, v,
                title="Sequence Flow",
                color={"color": "#34495E", "highlight": "#2C3E50"},
                width=3,
                arrows={"to": {"enabled": True, "scaleFactor": 1.2}},
                smooth={"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4}
            )
        elif rel == "BRANCH":
            net.add_edge(
                u, v,
                title="Decision Branch",
                color={"color": "#E67E22", "highlight": "#D35400"},
                width=3,
                dashes=[8, 4],
                arrows={"to": {"enabled": True, "scaleFactor": 1.2}},
                smooth={"type": "cubicBezier", "roundness": 0.5}
            )

    # Configure physics for a spider web layout - spreads nodes in all directions
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.02,
          "damping": 0.3,
          "avoidOverlap": 0.8
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "barnesHut",
        "stabilization": {
          "enabled": true,
          "iterations": 2000,
          "updateInterval": 25
        }
      },
      "layout": {
        "randomSeed": 42,
        "improvedLayout": true
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": {
          "enabled": true,
          "bindToWindow": false
        }
      },
      "edges": {
        "font": {
          "size": 12,
          "face": "Arial"
        }
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
    
    # Inject JavaScript for enhanced interactivity
    enhanced_script = """
    <script type="text/javascript">
    document.addEventListener('DOMContentLoaded', function() {
        var checkNetwork = setInterval(function() {
            if (typeof network !== 'undefined' && network !== null) {
                clearInterval(checkNetwork);
                
                // Stop physics on click for manual arrangement
                network.on('click', function(params) {
                    if (params.nodes.length > 0) {
                        network.setOptions({ physics: { enabled: false } });
                    }
                });
                
                // Stop physics on drag end
                network.on('dragEnd', function(params) {
                    if (params.nodes.length > 0) {
                        network.setOptions({ physics: { enabled: false } });
                    }
                });
                
                // Fit the graph nicely after stabilization
                network.on('stabilizationIterationsDone', function() {
                    network.fit({
                        padding: 50,
                        animation: {
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                });
            }
        }, 100);
    });
    </script>
    """
    
    # Add legend HTML
    legend_html = """
    <div id="graph-legend" style="
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        font-family: Arial, sans-serif;
        z-index: 1000;
        min-width: 200px;
    ">
        <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #2C3E50; border-bottom: 2px solid #EEE; padding-bottom: 10px;">
            📊 Knowledge Graph Legend
        </h3>
        <div style="display: flex; flex-direction: column; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 30px; height: 20px; background: #4A90D9; border-radius: 4px; border: 2px solid #2E6AB3;"></div>
                <span style="font-size: 13px; color: #34495E;">Process Step</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 30px; height: 20px; background: #27AE60; border-radius: 4px; border: 2px solid #1E8449;"></div>
                <span style="font-size: 13px; color: #34495E;">Step with Resources</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 30px; height: 20px; background: #E74C3C; border-radius: 4px; border: 2px solid #C0392B;"></div>
                <span style="font-size: 13px; color: #34495E;">⚠️ Safety Critical</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 20px; height: 20px; background: #F39C12; transform: rotate(45deg); border: 2px solid #D68910;"></div>
                <span style="font-size: 13px; color: #34495E; margin-left: 10px;">Decision Point</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; margin-top: 8px; padding-top: 8px; border-top: 1px solid #EEE;">
                <div style="width: 30px; height: 3px; background: #34495E;"></div>
                <span style="font-size: 13px; color: #34495E;">Sequence Flow</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 30px; height: 3px; background: #E67E22; border-style: dashed;"></div>
                <span style="font-size: 13px; color: #34495E;">Decision Branch</span>
            </div>
        </div>
        <p style="margin: 15px 0 0 0; font-size: 11px; color: #7F8C8D; font-style: italic;">
            💡 Hover over nodes for details<br>
            🖱️ Click & drag to rearrange
        </p>
    </div>
    """
    
    # Insert the script and legend before closing </body> tag
    html_content = html_content.replace('</body>', legend_html + enhanced_script + '\n</body>')
    
    # Add fullscreen-friendly styles
    fullscreen_styles = """
    <style>
    html, body {
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    #mynetwork {
        width: 100% !important;
        height: 100vh !important;
        border: none !important;
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    /* Navigation buttons styling */
    div.vis-navigation {
        background: rgba(255,255,255,0.9) !important;
        border-radius: 8px !important;
        padding: 5px !important;
    }
    </style>
    """
    html_content = html_content.replace('</head>', fullscreen_styles + '\n</head>')
        
    return html_content