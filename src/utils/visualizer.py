"""PKG Visualizer - Clean interactive graph visualization."""

import networkx as nx
from pyvis.network import Network
import textwrap
import tempfile
import os
import re
from typing import Dict, Any, List

from src.ai.types import ExtractionResult

THEME = {
    "step": {"bg": "#1E40AF", "border": "#1E3A8A"},
    "step_safety": {"bg": "#B91C1C", "border": "#991B1B"},
    "step_resources": {"bg": "#047857", "border": "#065F46"},
    "constraint_guard": {"bg": "#B45309", "border": "#92400E"},
    "constraint_precondition": {"bg": "#C2410C", "border": "#9A3412"},
    "constraint_postcondition": {"bg": "#15803D", "border": "#166534"},
    "constraint_warning": {"bg": "#B91C1C", "border": "#991B1B"},
    "gateway": {"bg": "#6D28D9", "border": "#5B21B6"},
    "resource_tools": {"bg": "#0EA5E9", "border": "#0284C7"},
    "resource_materials": {"bg": "#10B981", "border": "#059669"},
    "resource_documents": {"bg": "#93C5FD", "border": "#60A5FA"},
    "resource_ppe": {"bg": "#F59E0B", "border": "#D97706"},
    "parameter": {"bg": "#64748B", "border": "#475569"},
    "edge_next": "#1E40AF",
    "edge_guard": "#B45309",
    "edge_branch": "#6D28D9",
    "edge_uses": "#0EA5E9",
    "edge_param": "#64748B",
    "edge_alias": "#94A3B8",
}


def _build_resource_catalog(result: ExtractionResult) -> Dict[str, Dict[str, Any]]:
    """Build resource lookup from metadata (id -> {name, category})."""
    catalog: Dict[str, Dict[str, Any]] = {}
    if not result.metadata:
        return catalog
    
    resources = result.metadata.get("resources_catalog", {})
    if not resources:
        resources = result.metadata.get("relations", {}).get("resources_catalog", {})
    
    for category in ["tools", "materials", "documents", "ppe"]:
        for item in resources.get(category, []):
            if isinstance(item, dict) and item.get("id"):
                catalog[item["id"]] = {
                    "name": item.get("canonical_name", item["id"]),
                    "category": category,
                }
    return catalog


def _resource_name_to_id_map(catalog: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Map canonical resource names back to IDs (case-insensitive)."""
    name_to_id: Dict[str, str] = {}
    for rid, info in catalog.items():
        name = str(info.get("name") or "").strip()
        if name:
            name_to_id[name.lower()] = rid
    return name_to_id


def _normalize_label(label: str, catalog: Dict) -> str:
    """Clean up label text."""
    if not label:
        return label
    if label in catalog:
        return catalog[label].get("name", label)
    for pattern in [r'^R_', r'^T_', r'^M_', r'^Res_\w+_', r'^Act_', r'^Obj_\w+_']:
        label = re.sub(pattern, '', label)
    return label.replace('_', ' ').strip()


def _get_resource_name(res_id: str, catalog: Dict) -> str:
    """Get display name for resource."""
    if res_id in catalog:
        return catalog[res_id].get("name", res_id)
    return _normalize_label(res_id, catalog)


def _extract_constraints(step: Dict) -> List[Dict]:
    """Extract constraints from step."""
    constraints = []
    data = step.get("constraints", {})
    types = {
        "precondition": "PRE",
        "postcondition": "POST",
        "guard": "IF",
        "warning": "WARN",
        "acceptance_criteria": "ACC",
    }
    
    for ctype, icon in types.items():
        for i, item in enumerate(data.get(ctype, [])):
            text = item if isinstance(item, str) else item.get("text", str(item))
            constraints.append({
                "id": f"{step.get('id', 'S')}_{ctype}_{i}",
                "text": text,
                "type": ctype,
                "icon": icon,
                "step_id": step.get("id", "S")
            })
    return constraints


def _build_graph(result: ExtractionResult, catalog: Dict) -> nx.DiGraph:
    """Build NetworkX graph from extraction result."""
    G = nx.DiGraph()
    steps = result.steps or []
    all_constraints = []
    
    for idx, step in enumerate(steps):
        step_id = step.get("id", f"S{idx + 1}")
        label = step.get("label", step_id)
        action = step.get("action_verb", "")
        
        obj = step.get("action_object")
        obj_name = obj.get("canonical", "") if isinstance(obj, dict) else (obj or "")
        
        resources = step.get("resources", {})
        tools = resources.get("tools", []) if isinstance(resources, dict) else []
        materials = resources.get("materials", []) if isinstance(resources, dict) else []
        documents = resources.get("documents", []) if isinstance(resources, dict) else []
        ppe = resources.get("ppe", []) if isinstance(resources, dict) else []
        
        step_constraints = _extract_constraints(step)
        all_constraints.extend(step_constraints)
        
        flags = step.get("flags", {})
        is_safety = flags.get("safety_critical", False)
        
        tooltip = f"<b>{step_id}: {action.upper() if action else 'ACTION'}</b>"
        if is_safety:
            tooltip += " [SAFETY]"
        tooltip += f"<br>{label}"
        if obj_name:
            tooltip += f"<br>Target: {obj_name}"
        if tools or materials or documents or ppe:
            res_names = [_get_resource_name(r, catalog) for r in (tools + materials + documents + ppe)[:3]]
            tooltip += f"<br>Resources: {', '.join(res_names)}"
        
        action_str = action.upper().replace("_", " ") if action else ""
        wrapped = textwrap.fill(label[:35], width=16)
        display = f"{step_id}\n{action_str}\n{wrapped}" if action_str else f"{step_id}\n{wrapped}"
        
        G.add_node(
            step_id,
            node_type="step",
            label=display,
            title=tooltip,
            safety_critical=is_safety,
            has_resources=bool(tools or materials or documents or ppe),
            order=idx
        )

        # Resources and parameters
        name_to_id = _resource_name_to_id_map(catalog)

        def _ensure_res_node(res: Any, category: str) -> str:
            # Create or reuse a resource node, return node ID
            if isinstance(res, dict):
                rid = res.get("id") or res.get("canonical_id") or res.get("name") or res.get("canonical_name") or str(res)
                rname = res.get("canonical_name") or res.get("name") or str(rid)
            else:
                rid = name_to_id.get(str(res).lower(), str(res))
                rname = _get_resource_name(str(res), catalog)
            rid = str(rid)
            node_id = f"R:{rid}"
            if node_id not in G:
                colors = THEME.get(f"resource_{category}", THEME["resource_materials"])
                G.add_node(
                    node_id,
                    node_type="resource",
                    resource_category=category,
                    label=textwrap.fill(rname[:28], width=14),
                    title=f"{category.title()}<br>{rname}",
                )
            return node_id

        for r in tools:
            rid = _ensure_res_node(r, "tools")
            G.add_edge(step_id, rid, edge_type="USES")
        for r in materials:
            rid = _ensure_res_node(r, "materials")
            G.add_edge(step_id, rid, edge_type="USES")
        for r in documents:
            rid = _ensure_res_node(r, "documents")
            G.add_edge(step_id, rid, edge_type="USES")
        for r in ppe:
            rid = _ensure_res_node(r, "ppe")
            G.add_edge(step_id, rid, edge_type="USES")

        params = step.get("parameters", [])
        if isinstance(params, list):
            for p in params:
                if isinstance(p, dict):
                    pname = str(p.get("name") or "param").strip()
                    pval = str(p.get("value") or "").strip()
                    punit = str(p.get("unit") or "").strip()
                    plabel = f"{pname}\n{pval} {punit}".strip()
                else:
                    plabel = str(p)
                    pname = plabel
                pid = f"P:{step_id}:{pname or 'param'}"
                if pid not in G:
                    colors = THEME["parameter"]
                    G.add_node(
                        pid,
                        node_type="parameter",
                        label=textwrap.fill(plabel[:28], width=14),
                        title=f"Parameter<br>{plabel}",
                    )
                G.add_edge(step_id, pid, edge_type="HAS_PARAM")
    
    for c in all_constraints:
        wrapped = textwrap.fill(c["text"][:30], width=14)
        display = f"{c['icon']}\n{wrapped}"
        tooltip = f"<b>{c['type'].upper()}</b><br>{c['text']}<br>Step: {c['step_id']}"
        
        G.add_node(
            c["id"],
            node_type="constraint",
            constraint_type=c["type"],
            label=display,
            title=tooltip,
            attached_to=c["step_id"]
        )
        G.add_edge(c["id"], c["step_id"], edge_type="GUARD")
    
    relations = result.metadata.get("relations", {}) if result.metadata else {}
    
    if isinstance(relations, dict):
        for link in relations.get("sequence", []):
            u, v = link.get("from"), link.get("to")
            if u and v and u in G.nodes and v in G.nodes:
                G.add_edge(u, v, edge_type="NEXT")
        
        for gw in relations.get("gateways", []):
            gw_id = gw.get("id")
            gw_type = gw.get("gateway_type", "XOR")
            if gw_id:
                G.add_node(gw_id, node_type="gateway", label=f"{gw_type}\n{gw_id}", title=f"Gateway: {gw_type}")
                for branch in gw.get("branches", []):
                    if branch in G.nodes:
                        G.add_edge(gw_id, branch, edge_type="BRANCH")
    
    if not any(d.get("edge_type") == "NEXT" for _, _, d in G.edges(data=True)):
        step_ids = [s.get("id", f"S{i+1}") for i, s in enumerate(steps)]
        for i in range(len(step_ids) - 1):
            if step_ids[i] in G.nodes and step_ids[i+1] in G.nodes:
                G.add_edge(step_ids[i], step_ids[i+1], edge_type="NEXT")
    
    # Integrate extracted entities by aliasing to resources when possible
    try:
        # Build lookup of resource names -> node id
        res_name_to_node: Dict[str, str] = {}
        for nid, data in G.nodes(data=True):
            if data.get("node_type") == "resource":
                name = (data.get("title") or "").split("<br>")[-1].strip().lower()
                if name:
                    res_name_to_node[name] = nid

        for idx, ent in enumerate(getattr(result, "entities", []) or []):
            name = str(getattr(ent, "content", "") or "").strip()
            if not name:
                continue
            low = name.lower()
            node_id = f"E:{idx+1}"
            category = str(getattr(ent, "category", "entity") or "entity").lower()
            colors = THEME.get(f"resource_{category}", THEME["resource_materials"])
            G.add_node(
                node_id,
                node_type="entity",
                entity_category=category,
                label=textwrap.fill(name[:28], width=14),
                title=f"Entity: {category.title()}<br>{name}",
            )
            # Alias link to resource node if names match
            rid = res_name_to_node.get(low)
            if rid:
                G.add_edge(node_id, rid, edge_type="ALIAS_OF")
    except Exception:
        pass

    return G


def generate_interactive_graph_html(result: ExtractionResult, height="600px", width="100%") -> str:
    """Generate interactive PKG visualization HTML."""
    catalog = _build_resource_catalog(result)
    G = _build_graph(result, catalog)
    
    if G.number_of_nodes() == 0:
        return """<!DOCTYPE html><html><body style="display:flex;justify-content:center;align-items:center;height:100vh;font-family:sans-serif;color:#64748B"><h2>No Data</h2></body></html>"""
    
    steps = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "step")
    constraints = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "constraint")
    resources_ct = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "resource")
    params_ct = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "parameter")
    safety = sum(1 for _, d in G.nodes(data=True) if d.get("safety_critical"))
    
    net = Network(height=height, width=width, directed=True, layout=False)
    
    for node, data in G.nodes(data=True):
        node_type = data.get("node_type", "step")
        label = data.get("label", str(node))
        tooltip = data.get("title", label)
        
        if node_type == "gateway":
            colors = THEME["gateway"]
            net.add_node(node, label=label, title=tooltip,
                color={"background": colors["bg"], "border": colors["border"]},
                shape="diamond", size=35, borderWidth=2,
                font={"size": 11, "color": "#FFF", "multi": True, "align": "center"})
        
        elif node_type == "constraint":
            c_type = data.get("constraint_type", "guard")
            colors = THEME.get(f"constraint_{c_type}", THEME["constraint_guard"])
            net.add_node(node, label=label, title=tooltip,
                color={"background": colors["bg"], "border": colors["border"]},
                shape="ellipse", size=24, borderWidth=2,
                font={"size": 9, "color": "#FFF", "multi": True, "align": "center"})
        
        elif node_type == "resource":
            category = data.get("resource_category", "materials")
            colors = THEME.get(f"resource_{category}", THEME["resource_materials"])
            net.add_node(node, label=label, title=tooltip,
                color={"background": colors["bg"], "border": colors["border"]},
                shape="database", size=20, borderWidth=1.5,
                font={"size": 10, "color": "#0F172A", "multi": True, "align": "center"})
        
        elif node_type == "parameter":
            colors = THEME["parameter"]
            net.add_node(node, label=label, title=tooltip,
                color={"background": colors["bg"], "border": colors["border"]},
                shape="triangle", size=18, borderWidth=1.5,
                font={"size": 10, "color": "#FFF", "multi": True, "align": "center"})
        
        else:
            if data.get("safety_critical"):
                colors = THEME["step_safety"]
            elif data.get("has_resources"):
                colors = THEME["step_resources"]
            else:
                colors = THEME["step"]
            
            net.add_node(node, label=label, title=tooltip,
                color={"background": colors["bg"], "border": colors["border"]},
                shape="box", size=30, borderWidth=2,
                font={"size": 10, "color": "#FFF", "multi": True, "align": "center"},
                margin=8, widthConstraint={"minimum": 100, "maximum": 160})
    
    for u, v, data in G.edges(data=True):
        edge_type = data.get("edge_type", "NEXT")
        if edge_type == "NEXT":
            net.add_edge(u, v, color={"color": THEME["edge_next"]}, width=2,
                arrows={"to": {"enabled": True, "scaleFactor": 0.7}},
                smooth={"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.2})
        elif edge_type == "GUARD":
            net.add_edge(u, v, color={"color": THEME["edge_guard"]}, width=1.5, dashes=[5, 3],
                arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
                smooth={"type": "curvedCW", "roundness": 0.1})
        elif edge_type == "BRANCH":
            net.add_edge(u, v, color={"color": THEME["edge_branch"]}, width=2, dashes=[6, 3],
                arrows={"to": {"enabled": True, "scaleFactor": 0.6}},
                smooth={"type": "cubicBezier", "roundness": 0.2})
        elif edge_type == "USES":
            net.add_edge(u, v, color={"color": THEME["edge_uses"]}, width=1.5,
                arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
                smooth={"type": "curvedCCW", "roundness": 0.15})
        elif edge_type == "HAS_PARAM":
            net.add_edge(u, v, color={"color": THEME["edge_param"]}, width=1.2, dashes=[2,3],
                arrows={"to": {"enabled": True, "scaleFactor": 0.4}},
                smooth={"type": "curvedCCW", "roundness": 0.1})
        elif edge_type == "ALIAS_OF":
            net.add_edge(u, v, color={"color": THEME["edge_alias"]}, width=1.0, dashes=[2,2],
                arrows={"to": {"enabled": True, "scaleFactor": 0.3}},
                smooth={"type": "curvedCCW", "roundness": 0.05})
    
    net.set_options("""{
        "physics": {"enabled": true, "hierarchicalRepulsion": {"nodeDistance": 160, "springLength": 180}, "solver": "hierarchicalRepulsion", "stabilization": {"iterations": 1500}},
        "layout": {"hierarchical": {"enabled": true, "direction": "UD", "sortMethod": "directed", "nodeSpacing": 160, "levelSeparation": 120}},
        "interaction": {"hover": true, "tooltipDelay": 100, "navigationButtons": true, "keyboard": true}
    }""")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        tmp.seek(0)
        html = tmp.read().decode("utf-8")
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    
    styles = """<style>
    *{box-sizing:border-box}
    html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;font-family:-apple-system,sans-serif;background:#FFF}
    #mynetwork{width:100%!important;height:100vh!important;border:none!important;background:#FFF!important}
    div.vis-tooltip{background:#FFF!important;border:1px solid #E2E8F0!important;border-radius:6px!important;padding:10px!important;font-size:12px!important;box-shadow:0 2px 8px rgba(0,0,0,0.1)!important}
    .panel{position:fixed;background:#FFF;border-radius:6px;padding:12px 16px;border:1px solid #E2E8F0;font-size:11px;z-index:1000}
    </style>"""
    
    stats_panel = f"""<div class="panel" style="top:12px;left:12px">
        <div style="font-weight:600;color:#0F172A;margin-bottom:8px">PKG</div>
        <div style="display:flex;gap:16px">
            <div><span style="font-size:18px;font-weight:700;color:#1E40AF">{steps}</span><br><span style="color:#64748B">Steps</span></div>
            <div><span style="font-size:18px;font-weight:700;color:#B45309">{constraints}</span><br><span style="color:#64748B">Constraints</span></div>
            <div><span style="font-size:18px;font-weight:700;color:#0EA5E9">{resources_ct}</span><br><span style="color:#64748B">Resources</span></div>
            <div><span style="font-size:18px;font-weight:700;color:#64748B">{params_ct}</span><br><span style="color:#64748B">Parameters</span></div>
            <div><span style="font-size:18px;font-weight:700;color:#B91C1C">{safety}</span><br><span style="color:#64748B">Safety</span></div>
        </div>
    </div>"""
    
    legend = """<div class="panel" style="top:12px;right:12px">
        <div style="font-weight:600;color:#64748B;margin-bottom:8px">LEGEND</div>
        <div style="display:flex;flex-direction:column;gap:4px">
            <div style="display:flex;align-items:center;gap:6px"><div style="width:16px;height:10px;background:#1E40AF;border-radius:2px"></div><span style="color:#475569">Step</span></div>
            <div style="display:flex;align-items:center;gap:6px"><div style="width:16px;height:10px;background:#047857;border-radius:2px"></div><span style="color:#475569">+Resources</span></div>
            <div style="display:flex;align-items:center;gap:6px"><div style="width:16px;height:10px;background:#B91C1C;border-radius:2px"></div><span style="color:#475569">Safety</span></div>
            <div style="display:flex;align-items:center;gap:6px"><div style="width:10px;height:10px;background:#B45309;border-radius:50%"></div><span style="color:#475569">Constraint</span></div>
        </div>
    </div>"""
    
    controls = """<div class="panel" style="bottom:12px;left:12px;color:#94A3B8">Scroll: zoom | Drag: pan | F: fit</div>"""
    
    script = """<script>
    (function(){var i=setInterval(function(){if(typeof network!=='undefined'){clearInterval(i);
        network.on('click',function(p){if(p.nodes.length>0)network.setOptions({physics:{enabled:false}})});
        network.on('doubleClick',function(p){if(p.nodes.length===0)network.setOptions({physics:{enabled:true}})});
        network.on('stabilizationIterationsDone',function(){setTimeout(function(){network.fit({padding:50})},100)});
        document.addEventListener('keydown',function(e){if(e.key==='f'||e.key==='F')network.fit()});
    }},100)})();
    </script>"""
    
    html = html.replace('</head>', styles + '</head>')
    legend2 = """<div class=\"panel\" style=\"top:120px;right:12px\">\n        <div style=\"font-weight:600;color:#64748B;margin-bottom:8px\">NODES</div>\n        <div style=\"display:flex;flex-direction:column;gap:4px\">\n            <div style=\"display:flex;align-items:center;gap:6px\"><div style=\"width:14px;height:10px;background:#0EA5E9;border-radius:2px\"></div><span style=\"color:#475569\">Resource</span></div>\n            <div style=\"display:flex;align-items:center;gap:6px\"><div style=\"width:0;height:0;border-left:8px solid transparent;border-right:8px solid transparent;border-bottom:14px solid #64748B\"></div><span style=\"color:#475569\">Parameter</span></div>\n        </div>\n    </div>"""
    html = html.replace('</body>', stats_panel + legend + legend2 + controls + script + '</body>')
    
    return html
