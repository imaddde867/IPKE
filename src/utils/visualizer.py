"""
IPKE Procedural Knowledge Graph Visualizer

Generates interactive PKGs with:
- NEXT edges (sequence flow between steps)
- GUARD edges (constraint attachment to steps)
- Safety-critical highlighting
- Resource dependency visualization

Schema:
- Nodes: Step, Constraint, Gateway
- Edges: NEXT (temporal adjacency), GUARD (constraint→step)
"""

import networkx as nx
from pyvis.network import Network
import textwrap
import tempfile
import os
import re
from typing import Dict, Any, List

from src.ai.types import ExtractionResult


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCE CATALOG HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

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
        resources_catalog = result.metadata.get("relations", {}).get("resources_catalog", {})
    
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
    Normalize a label by looking up resource IDs and cleaning technical prefixes.
    """
    if not label:
        return label
    
    if label in catalog:
        return catalog[label]
    
    normalized = label
    prefix_patterns = [
        r'^R_', r'^T_', r'^M_', r'^Res_\w+_', r'^Act_', r'^Obj_\w+_',
    ]
    
    for pattern in prefix_patterns:
        normalized = re.sub(pattern, '', normalized)
    
    return normalized.replace('_', ' ').strip()


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_constraints_from_step(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all constraint objects from a step."""
    constraints = []
    constraint_data = step.get("constraints", {})
    
    constraint_types = {
        "precondition": {"color": "#E74C3C", "icon": "PRE"},
        "postcondition": {"color": "#27AE60", "icon": "POST"},
        "guard": {"color": "#F39C12", "icon": "IF"},
        "warning": {"color": "#C0392B", "icon": "WARN"},
        "acceptance_criteria": {"color": "#9B59B6", "icon": "ACC"},
    }
    
    for ctype, meta in constraint_types.items():
        items = constraint_data.get(ctype, [])
        for i, item in enumerate(items):
            if isinstance(item, str):
                constraints.append({
                    "id": f"{step.get('id', 'S?')}_{ctype}_{i}",
                    "text": item,
                    "type": ctype,
                    "color": meta["color"],
                    "icon": meta["icon"],
                    "step_id": step.get("id", "S?")
                })
            elif isinstance(item, dict):
                constraints.append({
                    "id": f"{step.get('id', 'S?')}_{ctype}_{i}",
                    "text": item.get("text", item.get("expression", str(item))),
                    "type": ctype,
                    "color": meta["color"],
                    "icon": meta["icon"],
                    "step_id": step.get("id", "S?")
                })
    
    return constraints


def _build_graph_from_result(result: ExtractionResult, catalog: Dict[str, str] = None) -> nx.DiGraph:
    """
    Converts ExtractionResult into NetworkX graph following IPKE thesis schema.
    
    Graph structure:
    - Step nodes (blue): Procedural actions
    - Constraint nodes (red/orange): Guards, preconditions, warnings
    - NEXT edges (solid): Sequence flow
    - GUARD edges (dashed): Constraint attachment
    """
    G = nx.DiGraph()
    
    if catalog is None:
        catalog = {}
    
    steps = result.steps if result.steps else []
    all_constraints = []
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 1. ADD STEP NODES
    # ─────────────────────────────────────────────────────────────────────────────
    for idx, step in enumerate(steps):
        step_id = step.get("id", f"S{idx + 1}")
        label = step.get("label", step_id)
        action = step.get("action_verb", "")
        
        # Build rich tooltip
        tooltip_parts = [f"<b>{step_id}: {action.upper() if action else 'ACTION'}</b>"]
        tooltip_parts.append(f"<br><br>{label}")
        
        # Get object
        obj = step.get("action_object")
        if isinstance(obj, dict):
            obj_name = obj.get("canonical") or obj.get("surface_form")
        else:
            obj_name = obj
        if obj_name:
            tooltip_parts.append(f"<br><br><b>Target:</b> {obj_name}")
        
        # Get resources
        resources = step.get("resources", {})
        tools = resources.get("tools", []) if isinstance(resources, dict) else []
        materials = resources.get("materials", []) if isinstance(resources, dict) else []
        
        if tools:
            tool_names = [_normalize_label(t if isinstance(t, str) else str(t), catalog) for t in tools[:3]]
            tooltip_parts.append(f"<br><b>Tools:</b> {', '.join(tool_names)}")
        
        if materials:
            mat_names = [_normalize_label(m if isinstance(m, str) else str(m), catalog) for m in materials[:3]]
            tooltip_parts.append(f"<br><b>Materials:</b> {', '.join(mat_names)}")
        
        # Get parameters
        params = step.get("parameters", [])
        if params:
            param_strs = [f"{p.get('name', '?')}: {p.get('value', '?')}" for p in params if isinstance(p, dict)]
            if param_strs:
                tooltip_parts.append(f"<br><b>Parameters:</b> {', '.join(param_strs)}")
        
        # Extract constraints for this step
        step_constraints = _extract_constraints_from_step(step)
        all_constraints.extend(step_constraints)
        
        if step_constraints:
            tooltip_parts.append(f"<br><br><b>Constraints:</b> {len(step_constraints)}")
        
        tooltip = "".join(tooltip_parts)
        
        # Create concise label
        if action:
            display_label = f"{action.upper()}\n{textwrap.fill(label[:60], width=22)}"
        else:
            display_label = textwrap.fill(label[:60], width=22)
        
        # Determine flags
        flags = step.get("flags", {})
        is_safety = flags.get("safety_critical", False)
        
        G.add_node(
            step_id, 
            node_type="step",
            label=display_label, 
            full_label=label,
            title=tooltip,
            action_verb=action,
            safety_critical=is_safety,
            has_resources=bool(tools or materials),
            has_constraints=len(step_constraints) > 0,
            order=idx
        )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 2. ADD CONSTRAINT NODES
    # ─────────────────────────────────────────────────────────────────────────────
    for constraint in all_constraints:
        c_id = constraint["id"]
        c_text = constraint["text"]
        c_type = constraint["type"]
        c_icon = constraint["icon"]
        
        display_label = f"[{c_icon}]\n{textwrap.fill(c_text[:40], width=18)}"
        
        tooltip = f"<b>{c_type.upper()}</b><br><br>{c_text}<br><br>Attached to: {constraint['step_id']}"
        
        G.add_node(
            c_id,
            node_type="constraint",
            constraint_type=c_type,
            label=display_label,
            full_label=c_text,
            title=tooltip,
            color=constraint["color"],
            attached_to=constraint["step_id"]
        )
        
        # Add GUARD edge (constraint → step)
        G.add_edge(c_id, constraint["step_id"], edge_type="GUARD", constraint_type=c_type)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 3. ADD SEQUENCE EDGES
    # ─────────────────────────────────────────────────────────────────────────────
    relations = {}
    if result.metadata:
        relations = result.metadata.get("relations", {})
    
    if relations and isinstance(relations, dict):
        # Process explicit sequence
        sequence = relations.get("sequence", [])
        for link in sequence:
            u, v = link.get("from"), link.get("to")
            if u and v and u in G.nodes and v in G.nodes:
                G.add_edge(u, v, edge_type="NEXT")
        
        # Process gateways (decision points)
        gateways = relations.get("gateways", [])
        for gw in gateways:
            gw_id = gw.get("id")
            gw_type = gw.get("gateway_type", "XOR")
            guard = gw.get("guard", {})
            condition = guard.get("condition", "") if isinstance(guard, dict) else ""
            
            if gw_id:
                tooltip = f"<b>DECISION: {gw_type}</b>"
                if condition:
                    tooltip += f"<br><br>Condition: {condition}"
                
                G.add_node(
                    gw_id, 
                    node_type="gateway",
                    gateway_type=gw_type,
                    label=f"?\n{gw_type}",
                    title=tooltip,
                    condition=condition
                )
                
                # Add branch edges
                for branch_target in gw.get("branches", []):
                    if branch_target in G.nodes:
                        G.add_edge(gw_id, branch_target, edge_type="BRANCH")
    
    # Fallback: linear sequence if no explicit relations
    if not any(G.edges(data=True)):
        step_ids = [s.get("id", f"S{i+1}") for i, s in enumerate(steps)]
        for i in range(len(step_ids) - 1):
            if step_ids[i] in G.nodes and step_ids[i+1] in G.nodes:
                G.add_edge(step_ids[i], step_ids[i+1], edge_type="NEXT")
    
    return G


# ═══════════════════════════════════════════════════════════════════════════════
# HTML GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_interactive_graph_html(result: ExtractionResult, height="600px", width="100%") -> str:
    """
    Generates elite-level interactive PKG visualization.
    
    Features:
    - Hierarchical layout optimized for procedural flow
    - Clear distinction between steps (blue) and constraints (red/orange)
    - NEXT edges (solid) vs GUARD edges (dashed)
    - Safety-critical highlighting
    - Professional legend
    - Interactive controls
    """
    catalog = _build_resource_catalog(result)
    G = _build_graph_from_result(result, catalog)
    
    if G.number_of_nodes() == 0:
        return """
        <div style='
            display: flex; 
            align-items: center; 
            justify-content: center; 
            height: 400px; 
            background: linear-gradient(135deg, #1a1a2e 0%, #252540 100%);
            border-radius: 20px;
            font-family: Inter, sans-serif;
        '>
            <div style='text-align: center; color: #888;'>
                <div style='font-size: 4rem; margin-bottom: 1rem;'>📊</div>
                <h2 style='color: #fff; margin: 0;'>No Procedural Steps Found</h2>
                <p>Upload a document or load demo data to visualize the PKG.</p>
            </div>
        </div>
        """

    net = Network(height=height, width=width, directed=True, layout=False)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # COLOR PALETTE - Professional Research Grade
    # ─────────────────────────────────────────────────────────────────────────────
    COLORS = {
        "step": {
            "background": "#4A90D9",
            "border": "#2E6AB3",
            "highlight": {"background": "#6BA3E0", "border": "#4A90D9"},
            "hover": {"background": "#5A9DE3", "border": "#4A90D9"}
        },
        "step_safety": {
            "background": "#E74C3C",
            "border": "#C0392B",
            "highlight": {"background": "#EC7063", "border": "#E74C3C"},
            "hover": {"background": "#EF5350", "border": "#E74C3C"}
        },
        "step_resources": {
            "background": "#27AE60",
            "border": "#1E8449",
            "highlight": {"background": "#52BE80", "border": "#27AE60"},
            "hover": {"background": "#4CAF50", "border": "#27AE60"}
        },
        "constraint_guard": {
            "background": "#F39C12",
            "border": "#D68910",
            "highlight": {"background": "#F5B041", "border": "#F39C12"}
        },
        "constraint_precondition": {
            "background": "#E74C3C",
            "border": "#C0392B",
            "highlight": {"background": "#EC7063", "border": "#E74C3C"}
        },
        "constraint_warning": {
            "background": "#C0392B",
            "border": "#922B21",
            "highlight": {"background": "#E74C3C", "border": "#C0392B"}
        },
        "constraint_default": {
            "background": "#9B59B6",
            "border": "#7D3C98",
            "highlight": {"background": "#AF7AC5", "border": "#9B59B6"}
        },
        "gateway": {
            "background": "#F39C12",
            "border": "#D68910",
            "highlight": {"background": "#F5B041", "border": "#F39C12"}
        }
    }
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ADD NODES
    # ─────────────────────────────────────────────────────────────────────────────
    for node, data in G.nodes(data=True):
        node_type = data.get("node_type", "step")
        label = data.get("label", str(node))
        tooltip = data.get("title", label)
        
        if node_type == "gateway":
            net.add_node(
                node,
                label=label,
                title=tooltip,
                color=COLORS["gateway"],
                shape="diamond",
                size=45,
                font={"size": 16, "face": "Inter, Arial", "color": "#FFFFFF", "bold": True, "multi": True},
                borderWidth=3,
                shadow={"enabled": True, "size": 8, "x": 2, "y": 2}
            )
        
        elif node_type == "constraint":
            c_type = data.get("constraint_type", "default")
            color_key = f"constraint_{c_type}" if f"constraint_{c_type}" in COLORS else "constraint_default"
            
            net.add_node(
                node,
                label=label,
                title=tooltip,
                color=COLORS[color_key],
                shape="ellipse",
                size=35,
                font={"size": 12, "face": "Inter, Arial", "color": "#FFFFFF", "multi": True, "align": "center"},
                borderWidth=2,
                borderWidthSelected=4,
                shadow={"enabled": True, "size": 6, "x": 2, "y": 2}
            )
        
        else:  # step
            if data.get("safety_critical"):
                color = COLORS["step_safety"]
            elif data.get("has_resources"):
                color = COLORS["step_resources"]
            else:
                color = COLORS["step"]
            
            net.add_node(
                node,
                label=label,
                title=tooltip,
                color=color,
                shape="box",
                size=40,
                font={"size": 14, "face": "Inter, Arial", "color": "#FFFFFF", "multi": True, "align": "center"},
                borderWidth=3,
                borderWidthSelected=5,
                shadow={"enabled": True, "size": 10, "x": 3, "y": 3},
                margin={"top": 12, "bottom": 12, "left": 12, "right": 12},
                widthConstraint={"minimum": 160, "maximum": 260}
            )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ADD EDGES
    # ─────────────────────────────────────────────────────────────────────────────
    for u, v, data in G.edges(data=True):
        edge_type = data.get("edge_type", "NEXT")
        
        if edge_type == "NEXT":
            net.add_edge(
                u, v,
                title="Sequence Flow (NEXT)",
                color={"color": "#4A90D9", "highlight": "#6BA3E0", "opacity": 0.9},
                width=3,
                arrows={"to": {"enabled": True, "scaleFactor": 1.2, "type": "arrow"}},
                smooth={"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.3}
            )
        
        elif edge_type == "GUARD":
            c_type = data.get("constraint_type", "guard")
            edge_color = "#F39C12" if c_type == "guard" else "#E74C3C"
            
            net.add_edge(
                u, v,
                title=f"Constraint Attachment ({c_type.upper()})",
                color={"color": edge_color, "highlight": "#F5B041", "opacity": 0.8},
                width=2,
                dashes=[6, 3],
                arrows={"to": {"enabled": True, "scaleFactor": 0.8, "type": "arrow"}},
                smooth={"type": "curvedCW", "roundness": 0.2}
            )
        
        elif edge_type == "BRANCH":
            net.add_edge(
                u, v,
                title="Decision Branch",
                color={"color": "#9B59B6", "highlight": "#AF7AC5", "opacity": 0.8},
                width=2,
                dashes=[8, 4],
                arrows={"to": {"enabled": True, "scaleFactor": 1.0}},
                smooth={"type": "cubicBezier", "roundness": 0.4}
            )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # PHYSICS & LAYOUT OPTIONS
    # ─────────────────────────────────────────────────────────────────────────────
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "hierarchicalRepulsion": {
          "centralGravity": 0.1,
          "springLength": 180,
          "springConstant": 0.02,
          "nodeDistance": 200,
          "damping": 0.3
        },
        "maxVelocity": 40,
        "minVelocity": 0.1,
        "solver": "hierarchicalRepulsion",
        "stabilization": {
          "enabled": true,
          "iterations": 1500,
          "updateInterval": 25,
          "fit": true
        }
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD",
          "sortMethod": "directed",
          "nodeSpacing": 200,
          "levelSeparation": 150,
          "treeSpacing": 250,
          "blockShifting": true,
          "edgeMinimization": true,
          "parentCentralization": true
        }
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
        },
        "multiselect": true,
        "selectable": true
      },
      "edges": {
        "font": {
          "size": 11,
          "face": "Inter, Arial",
          "strokeWidth": 3,
          "strokeColor": "#1a1a2e"
        }
      },
      "nodes": {
        "font": {
          "strokeWidth": 2,
          "strokeColor": "rgba(0,0,0,0.3)"
        }
      }
    }
    """)
    
    # Generate base HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        tmp.seek(0)
        html_content = tmp.read().decode("utf-8")
    
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    
    # ─────────────────────────────────────────────────────────────────────────────
    # CUSTOM STYLES
    # ─────────────────────────────────────────────────────────────────────────────
    custom_styles = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    * {
        box-sizing: border-box;
    }
    
    html, body {
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #0a0a15 0%, #1a1a2e 50%, #0f0f23 100%);
    }
    
    #mynetwork {
        width: 100% !important;
        height: 100vh !important;
        border: none !important;
        background: transparent !important;
    }
    
    /* Navigation buttons */
    div.vis-navigation {
        background: rgba(30, 30, 50, 0.95) !important;
        border-radius: 12px !important;
        padding: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }
    
    div.vis-button {
        background: rgba(74, 144, 217, 0.2) !important;
        border: 1px solid rgba(74, 144, 217, 0.4) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    div.vis-button:hover {
        background: rgba(74, 144, 217, 0.4) !important;
        transform: scale(1.05) !important;
    }
    
    /* Tooltip styling */
    div.vis-tooltip {
        background: rgba(20, 20, 35, 0.98) !important;
        border: 1px solid rgba(74, 144, 217, 0.3) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        color: #e0e0e0 !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5) !important;
        max-width: 350px !important;
        line-height: 1.5 !important;
    }
    </style>
    """
    
    # ─────────────────────────────────────────────────────────────────────────────
    # LEGEND HTML
    # ─────────────────────────────────────────────────────────────────────────────
    legend_html = """
    <div id="pkg-legend" style="
        position: fixed;
        top: 24px;
        right: 24px;
        background: rgba(20, 20, 35, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4);
        font-family: 'Inter', sans-serif;
        z-index: 1000;
        min-width: 220px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    ">
        <h3 style="
            margin: 0 0 16px 0; 
            font-size: 14px; 
            color: #fff; 
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        ">
            PKG Legend
        </h3>
        
        <div style="margin-bottom: 16px;">
            <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Nodes</div>
            
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <div style="width: 24px; height: 18px; background: #4A90D9; border-radius: 4px; border: 2px solid #2E6AB3;"></div>
                <span style="font-size: 12px; color: #ccc;">Process Step</span>
            </div>
            
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <div style="width: 24px; height: 18px; background: #27AE60; border-radius: 4px; border: 2px solid #1E8449;"></div>
                <span style="font-size: 12px; color: #ccc;">Step + Resources</span>
            </div>
            
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <div style="width: 24px; height: 18px; background: #E74C3C; border-radius: 4px; border: 2px solid #C0392B;"></div>
                <span style="font-size: 12px; color: #ccc;">Safety Critical</span>
            </div>
            
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <div style="width: 18px; height: 18px; background: #F39C12; border-radius: 50%; border: 2px solid #D68910;"></div>
                <span style="font-size: 12px; color: #ccc;">Constraint (GUARD)</span>
            </div>
            
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <div style="width: 16px; height: 16px; background: #F39C12; transform: rotate(45deg); border: 2px solid #D68910;"></div>
                <span style="font-size: 12px; color: #ccc; margin-left: 4px;">Decision Point</span>
            </div>
        </div>
        
        <div style="padding-top: 12px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
            <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Edges</div>
            
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <div style="width: 24px; height: 3px; background: #4A90D9; border-radius: 2px;"></div>
                <span style="font-size: 12px; color: #ccc;">NEXT (Sequence)</span>
            </div>
            
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <div style="width: 24px; height: 0; border-top: 3px dashed #F39C12;"></div>
                <span style="font-size: 12px; color: #ccc;">GUARD (Constraint)</span>
            </div>
        </div>
        
        <div style="
            margin-top: 16px; 
            padding-top: 12px; 
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 10px; 
            color: #666; 
            line-height: 1.4;
        ">
            <strong style="color: #888;">Controls:</strong><br>
            Scroll to zoom | Drag to pan<br>
            Click node to fix position
        </div>
    </div>
    
    <!-- Title Bar -->
    <div id="pkg-title" style="
        position: fixed;
        top: 24px;
        left: 24px;
        background: rgba(20, 20, 35, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 16px 24px;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4);
        font-family: 'Inter', sans-serif;
        z-index: 1000;
        border: 1px solid rgba(255, 255, 255, 0.08);
    ">
        <div style="font-size: 18px; font-weight: 700; color: #fff; margin-bottom: 4px;">
            Procedural Knowledge Graph
        </div>
        <div style="font-size: 12px; color: #888;">
            IPKE Extraction | Steps + Constraints Visualization
        </div>
    </div>
    """
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ENHANCED JAVASCRIPT
    # ─────────────────────────────────────────────────────────────────────────────
    enhanced_script = """
    <script type="text/javascript">
    document.addEventListener('DOMContentLoaded', function() {
        var checkNetwork = setInterval(function() {
            if (typeof network !== 'undefined' && network !== null) {
                clearInterval(checkNetwork);
                
                // Disable physics on interaction for manual arrangement
                network.on('click', function(params) {
                    if (params.nodes.length > 0) {
                        network.setOptions({ physics: { enabled: false } });
                    }
                });
                
                network.on('dragEnd', function(params) {
                    if (params.nodes.length > 0) {
                        network.setOptions({ physics: { enabled: false } });
                    }
                });
                
                // Fit graph after stabilization
                network.on('stabilizationIterationsDone', function() {
                    setTimeout(function() {
                        network.fit({
                            padding: 80,
                            animation: {
                                duration: 800,
                                easingFunction: 'easeInOutQuad'
                            }
                        });
                    }, 100);
                });
                
                // Double-click to re-enable physics
                network.on('doubleClick', function(params) {
                    if (params.nodes.length === 0) {
                        network.setOptions({ physics: { enabled: true } });
                    }
                });
            }
        }, 100);
    });
    </script>
    """
    
    # Inject everything
    html_content = html_content.replace('</head>', custom_styles + '\n</head>')
    html_content = html_content.replace('</body>', legend_html + enhanced_script + '\n</body>')
    
    return html_content
