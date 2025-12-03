import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os
import textwrap

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def create_procedural_graph(data):
    G = nx.DiGraph()
    
    # Helper map for resources
    resource_map = {}
    if "resources_catalog" in data:
        for cat, items in data["resources_catalog"].items():
            for item in items:
                resource_map[item["id"]] = item["canonical_name"]

    # 1. Process Steps
    for step in data.get("steps", []):
        step_id = step["id"]
        step_label = textwrap.fill(step.get("label", step_id), width=20)
        
        # Main Step Node
        G.add_node(step_id, type="Step", label=step_id, full_label=step_label)
        
        # Action Node
        action_verb = step.get("action_verb")
        if action_verb:
            action_node_id = f"Act_{step_id}"
            G.add_node(action_node_id, type="Action", label=action_verb)
            G.add_edge(step_id, action_node_id, relation="ACTION")
            
            # Action Object (Asset/Object)
            obj = step.get("action_object")
            if obj and obj.get("canonical"):
                obj_label = obj["canonical"]
                obj_node_id = f"Obj_{step_id}_{obj_label.replace(' ', '_')}"
                G.add_node(obj_node_id, type="Asset", label=obj_label)
                G.add_edge(action_node_id, obj_node_id, relation="ACTS_ON")
        
        # Resources (Tools/Materials) -> Requires
        resources = step.get("resources", {})
        all_res = resources.get("tools", []) + resources.get("materials", [])
        for res_id in all_res:
            res_name = resource_map.get(res_id, res_id)
            res_node_unique_id = f"Res_{step_id}_{res_id}"
            G.add_node(res_node_unique_id, type="Asset", label=res_name)
            G.add_edge(step_id, res_node_unique_id, relation="REQUIRES")

        # Parameters -> Constraint/Param
        for param in step.get("parameters", []):
            p_name = param.get("name")
            p_val = param.get("value")
            label = f"{p_name}: {p_val}"
            param_node_id = f"Param_{step_id}_{p_name}"
            G.add_node(param_node_id, type="Constraint", label=label)
            G.add_edge(step_id, param_node_id, relation="REQUIRES")

    # 2. Process Relations (Sequence & Gateways)
    relations = data.get("relations", {})
    
    # Sequence
    for seq in relations.get("sequence", []):
        u, v = seq["from"], seq["to"]
        G.add_edge(u, v, relation="NEXT")
        
    # Gateways
    gateways = {g["id"]: g for g in relations.get("gateways", [])}
    for g_id, g_data in gateways.items():
        if g_id not in G:
            G.add_node(g_id, type="Gateway", label=g_data.get("gateway_type", "GW"))
        
        for branch_target in g_data.get("branches", []):
             G.add_edge(g_id, branch_target, relation="NEXT")

    return G

def calculate_thesis_layout(G):
    pos = {}
    
    # Identify spine nodes (Steps and Gateways)
    spine_nodes = [n for n, d in G.nodes(data=True) if d.get("type") in ("Step", "Gateway")]
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Spine Nodes: {len(spine_nodes)}")

    # Linearize the spine using BFS/Traversal logic to determine order
    start_node = "S1"
    if start_node not in G and spine_nodes:
        start_node = spine_nodes[0]

    # Determine Order (rank)
    ranks = {}
    queue = [(start_node, 0)]
    visited = {start_node}
    ranks[start_node] = 0
    
    while queue:
        curr, rank = queue.pop(0)
        # Find next spine nodes
        next_nodes = [v for u, v, d in G.edges(data=True) 
                      if u == curr and d.get("relation") == "NEXT"]
        
        for nxt in next_nodes:
            if nxt not in visited:
                visited.add(nxt)
                ranks[nxt] = rank + 1
                queue.append((nxt, rank + 1))
    
    # Sort spine nodes by rank
    sorted_spine = sorted(ranks.keys(), key=lambda k: ranks[k])
    print(f"Spine Length (sorted): {len(sorted_spine)}")
    
    # Configuration
    x_cursor = 0
    base_y = 0
    step_width_base = 2.0
    satellite_spacing = 1.5
    
    # Calculate positions
    for node in sorted_spine:
        # Get satellites
        neighbors = G[node]
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
        
        # Determine width
        top_width = 0
        if actions:
            for act in actions:
                act_objs = [an for an in G[act] if G.nodes[an].get("type") == "Asset"]
                act_width = max(1, len(act_objs)) * satellite_spacing
                top_width += act_width
        else:
            top_width = 0
            
        bottom_count = len(resources) + len(params)
        bottom_width = bottom_count * satellite_spacing
        
        total_width = max(step_width_base, top_width, bottom_width)
        
        # Center position for the Step
        center_x = x_cursor + (total_width / 2)
        pos[node] = (center_x, base_y)
        
        # Place Satellites
        # --- TOP SECTOR (Actions & Objects) ---
        current_act_x = center_x - (top_width / 2) + (satellite_spacing / 2)
        for act in actions:
            act_objs = [an for an in G[act] if G.nodes[an].get("type") == "Asset"]
            n_objs = len(act_objs)
            width_this_act = max(1, n_objs) * satellite_spacing
            
            act_center_x = current_act_x + (width_this_act / 2) - (satellite_spacing / 2)
            pos[act] = (act_center_x, base_y + 3) # Action level
            
            # Place Objects above Action
            obj_start_x = current_act_x
            for i, obj in enumerate(act_objs):
                pos[obj] = (obj_start_x + (i * satellite_spacing), base_y + 5) # Object level
            
            current_act_x += width_this_act
            
        # --- BOTTOM SECTOR (Resources & Params) ---
        start_res_x = center_x - (bottom_width / 2) + (satellite_spacing / 2)
        current_res_x = start_res_x
        for i, res in enumerate(resources):
            pos[res] = (current_res_x, base_y - 3)
            current_res_x += satellite_spacing
            
        for i, p in enumerate(params):
            pos[p] = (current_res_x, base_y - 4 if len(resources) % 2 == 0 else base_y - 3.5)
            current_res_x += satellite_spacing

        # Advance cursor
        x_cursor += total_width + 1.0 
    
    print(f"Total Layout Width: {x_cursor}")
    return pos

def draw_thesis_figure(G, pos, output_file):
    # 1. Calculate Bounds
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width_units = max_x - min_x
    height_units = max_y - min_y
    
    # Margins
    margin_x = 5
    margin_y = 2
    
    # 2. Dynamic Figure Size
    # Target scale: 0.5 inches per unit width to fit text
    scale_factor = 0.5 
    fig_width = (width_units + 2*margin_x) * scale_factor
    fig_height = max(10, (height_units + 2*margin_y) * scale_factor) # Min height 10
    
    # 3. Check Limits
    dpi = 300
    max_pixels = 60000 # Safe limit below 65536
    
    # Calculate pixel dimensions
    px_w = fig_width * dpi
    px_h = fig_height * dpi
    
    if px_w > max_pixels or px_h > max_pixels:
        print(f"Warning: Calculated size {int(px_w)}x{int(px_h)} exceeds limit.")
        # Downscale DPI to fit
        scaling_needed = max(px_w / max_pixels, px_h / max_pixels)
        dpi = int(dpi / scaling_needed)
        print(f"Adjusting DPI to {dpi} to fit image.")
    
    print(f"Figure Size: {fig_width:.1f}x{fig_height:.1f} inches @ {dpi} DPI")

    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    plt.axis('off')
    
    # Set explicit limits
    ax.set_xlim(min_x - margin_x, max_x + margin_x)
    ax.set_ylim(min_y - margin_y, max_y + margin_y)

    # Style Config
    styles = {
        "Step": {"boxstyle": "round,pad=0.5", "fc": "#ffffff", "ec": "#005b96", "alpha": 1.0, "fontsize": 10, "fontweight": "bold"},
        "Gateway": {"boxstyle": "darrow,pad=0.3", "fc": "#ffcc00", "ec": "#cc9900", "alpha": 1.0, "fontsize": 9},
        "Action": {"boxstyle": "round,pad=0.3", "fc": "#e3f2fd", "ec": "#90caf9", "alpha": 1.0, "fontsize": 9},
        "Asset": {"boxstyle": "round4,pad=0.3", "fc": "#e8f5e9", "ec": "#a5d6a7", "alpha": 1.0, "fontsize": 8},
        "Constraint": {"boxstyle": "round,pad=0.3", "fc": "#ffebee", "ec": "#ef9a9a", "alpha": 1.0, "fontsize": 8}
    }

    # Draw Edges
    # 1. Spine Edges
    spine_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "NEXT"]
    for u, v in spine_edges:
        if u in pos and v in pos:
            p1 = pos[u]
            p2 = pos[v]
            is_loop = p2[0] < p1[0]
            connectionstyle = "arc3,rad=0.5" if is_loop else "arc3,rad=0.0"
            ls = 'dashed' if is_loop else 'solid'
            color = '#d32f2f' if is_loop else '#455a64'
            
            ax.annotate("", xy=p2, xytext=p1, 
                        arrowprops=dict(arrowstyle="->", color=color, lw=2, 
                                        shrinkA=15, shrinkB=15, 
                                        connectionstyle=connectionstyle, ls=ls))

    # 2. Satellite Edges
    satellite_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("relation") != "NEXT"]
    for u, v, d in satellite_edges:
        if u in pos and v in pos:
            rel = d.get("relation")
            color = '#90a4ae'
            style = 'dashed' if rel == "REQUIRES" else 'solid'
            ax.annotate("", xy=pos[v], xytext=pos[u], 
                        arrowprops=dict(arrowstyle="-", color=color, lw=1, 
                                        shrinkA=10, shrinkB=10, ls=style))

    # Draw Nodes
    for node, data in G.nodes(data=True):
        if node not in pos: continue
        x, y = pos[node]
        ntype = data.get("type", "Step")
        style = styles.get(ntype, styles["Step"])
        label = textwrap.fill(data.get("label", node), width=15)
        
        ax.text(x, y, label, ha='center', va='center', 
                bbox=dict(boxstyle=style["boxstyle"], fc=style["fc"], ec=style["ec"], alpha=style["alpha"]),
                fontsize=style.get("fontsize", 9),
                fontweight=style.get("fontweight", "normal"),
                color='#263238', zorder=10)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#ffffff', edgecolor='#005b96', label='Procedure Step'),
        mpatches.Patch(facecolor='#ffcc00', edgecolor='#cc9900', label='Gateway'),
        mpatches.Patch(facecolor='#e3f2fd', edgecolor='#90caf9', label='Action'),
        mpatches.Patch(facecolor='#e8f5e9', edgecolor='#a5d6a7', label='Asset'),
        mpatches.Patch(facecolor='#ffebee', edgecolor='#ef9a9a', label='Constraint'),
        mlines.Line2D([], [], color='#455a64', lw=2, label='Sequence'),
        mlines.Line2D([], [], color='#d32f2f', lw=2, ls='dashed', label='Rework Loop')
    ]
    ax.legend(handles=legend_elements, loc='upper left', title="Legend", fontsize=12)

    plt.title("Extracted Procedural Knowledge Graph: 3M OEM SOP", fontsize=20, pad=20)
    # No tight_layout, rely on explicit sizing
    plt.savefig(output_file, dpi=dpi)
    print(f"Graph saved to {output_file}")

if __name__ == "__main__":
    json_path = "datasets/archive/gold_human/3M_OEM_SOP.json"
    output_png = "generate_large_pkg.png"
    
    data = load_data(json_path)
    if data:
        G = create_procedural_graph(data)
        pos = calculate_thesis_layout(G)
        draw_thesis_figure(G, pos, output_png)