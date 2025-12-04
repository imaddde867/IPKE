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

def calculate_thesis_layout(G, max_row_width=45, row_gap=2.5):
    """Lay the process spine over multiple rows so the figure fits on a page."""
    pos = {}

    # Identify spine nodes (Steps and Gateways)
    spine_nodes = [n for n, d in G.nodes(data=True) if d.get("type") in ("Step", "Gateway")]
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Spine Nodes: {len(spine_nodes)}")

    # Linearize the spine using BFS/Traversal logic to determine order
    start_node = "S1"
    if start_node not in G and spine_nodes:
        start_node = spine_nodes[0]

    ranks = {}
    if spine_nodes:
        queue = [(start_node, 0)]
        visited = {start_node}
        ranks[start_node] = 0

        while queue:
            curr, rank = queue.pop(0)
            next_nodes = [v for u, v, d in G.edges(data=True)
                          if u == curr and d.get("relation") == "NEXT"]

            for nxt in next_nodes:
                if nxt not in visited:
                    visited.add(nxt)
                    ranks[nxt] = rank + 1
                    queue.append((nxt, rank + 1))

    remaining = [n for n in spine_nodes if n not in ranks]
    if remaining:
        next_rank = max(ranks.values()) + 1 if ranks else 0
        for n in remaining:
            ranks[n] = next_rank
            next_rank += 1

    sorted_spine = sorted(ranks.keys(), key=lambda k: ranks[k])
    print(f"Spine Length (sorted): {len(sorted_spine)}")

    # Layout configuration tuned for print readability
    step_width_base = 6.0
    satellite_spacing = 1.8
    row_spacing = 14.0
    action_offset = 4.0
    object_offset = 6.5
    resource_offset = -4.0
    param_offset = -5.5

    # Pre-compute satellite information & widths for each step/gateway
    layout_info = {}
    for node in sorted_spine:
        neighbors = G[node]
        actions_info = []
        resources = []
        params = []

        for nbr in neighbors:
            if nbr in ranks:
                continue  # Skip NEXT links along the spine

            etype = G.nodes[nbr].get("type")
            relation = G.edges[node, nbr].get("relation")

            if etype == "Action":
                act_objs = [an for an in G[nbr] if G.nodes[an].get("type") == "Asset"]
                actions_info.append({
                    "id": nbr,
                    "objects": act_objs,
                    "width": max(1, len(act_objs)) * satellite_spacing
                })
            elif etype == "Asset" and relation == "REQUIRES":
                resources.append(nbr)
            elif etype == "Constraint":
                params.append(nbr)

        top_width = sum(info["width"] for info in actions_info)
        bottom_count = len(resources) + len(params)
        bottom_width = bottom_count * satellite_spacing
        total_width = max(step_width_base, top_width, bottom_width)

        layout_info[node] = {
            "actions": actions_info,
            "resources": resources,
            "params": params,
            "width": total_width,
            "top_width": top_width,
            "bottom_width": bottom_width
        }

    # Wrap the spine into multiple rows while respecting the max width
    rows = []
    current_row = []
    current_width = 0.0
    for node in sorted_spine:
        node_width = layout_info[node]["width"]
        addition = node_width if not current_row else row_gap + node_width
        if current_row and (current_width + addition) > max_row_width:
            rows.append(current_row)
            current_row = [node]
            current_width = node_width
        else:
            current_row.append(node)
            current_width += addition

    if current_row:
        rows.append(current_row)

    print(f"Rows generated: {len(rows)} (max row width {max_row_width} units)")

    # Position each row centered on the canvas
    for row_idx, row_nodes in enumerate(rows):
        row_y = -(row_idx * row_spacing)
        row_width = sum(layout_info[n]["width"] for n in row_nodes)
        row_width += row_gap * (len(row_nodes) - 1) if len(row_nodes) > 1 else 0
        cursor_x = -row_width / 2.0

        for node in row_nodes:
            info = layout_info[node]
            node_center_x = cursor_x + (info["width"] / 2.0)
            pos[node] = (node_center_x, row_y)

            # --- TOP SECTOR (Actions & Objects) ---
            if info["actions"]:
                top_width = max(info["top_width"], satellite_spacing)
                current_act_x = node_center_x - (top_width / 2.0) + (satellite_spacing / 2.0)
            else:
                current_act_x = node_center_x

            for action in info["actions"]:
                width_this_act = max(satellite_spacing, action["width"])
                act_center_x = current_act_x + (width_this_act / 2.0) - (satellite_spacing / 2.0)
                pos[action["id"]] = (act_center_x, row_y + action_offset)

                obj_start_x = current_act_x
                for idx, obj in enumerate(action["objects"]):
                    pos[obj] = (obj_start_x + (idx * satellite_spacing), row_y + object_offset)

                current_act_x += width_this_act

            # --- BOTTOM SECTOR (Resources & Params) ---
            if info["bottom_width"] > 0:
                start_res_x = node_center_x - (info["bottom_width"] / 2.0) + (satellite_spacing / 2.0)
            else:
                start_res_x = node_center_x

            current_res_x = start_res_x
            for res in info["resources"]:
                pos[res] = (current_res_x, row_y + resource_offset)
                current_res_x += satellite_spacing

            for param in info["params"]:
                pos[param] = (current_res_x, row_y + param_offset)
                current_res_x += satellite_spacing

            cursor_x += info["width"] + row_gap

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
    margin_y = 4
    
    # 2. Dynamic Figure Size tuned for thesis layout proportions
    scale_factor = 0.35
    fig_width = min(14, max(11, (width_units + 2 * margin_x) * scale_factor))
    fig_height = min(18, max(9, (height_units + 2 * margin_y) * scale_factor))
    
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
    ax.set_facecolor('#f9fafb')
    plt.axis('off')
    
    # Set explicit limits
    ax.set_xlim(min_x - margin_x, max_x + margin_x)
    ax.set_ylim(min_y - margin_y, max_y + margin_y)

    # Style Config
    styles = {
        "Step": {"boxstyle": "round,pad=0.5", "fc": "#ffffff", "ec": "#005b96", "alpha": 1.0, "fontsize": 11, "fontweight": "bold"},
        "Gateway": {"boxstyle": "darrow,pad=0.35", "fc": "#ffeb99", "ec": "#cc9900", "alpha": 1.0, "fontsize": 10},
        "Action": {"boxstyle": "round,pad=0.3", "fc": "#e3f2fd", "ec": "#64b5f6", "alpha": 1.0, "fontsize": 10},
        "Asset": {"boxstyle": "round4,pad=0.3", "fc": "#e8f5e9", "ec": "#81c784", "alpha": 1.0, "fontsize": 9},
        "Constraint": {"boxstyle": "round,pad=0.3", "fc": "#ffebee", "ec": "#ef9a9a", "alpha": 1.0, "fontsize": 9}
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
        mpatches.Patch(facecolor='#ffeb99', edgecolor='#cc9900', label='Gateway'),
        mpatches.Patch(facecolor='#e3f2fd', edgecolor='#64b5f6', label='Action'),
        mpatches.Patch(facecolor='#e8f5e9', edgecolor='#81c784', label='Asset'),
        mpatches.Patch(facecolor='#ffebee', edgecolor='#ef9a9a', label='Constraint'),
        mlines.Line2D([], [], color='#455a64', lw=2, label='Sequence'),
        mlines.Line2D([], [], color='#d32f2f', lw=2, ls='dashed', label='Rework Loop')
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        title="Legend",
        fontsize=11,
        frameon=True,
        framealpha=0.95
    )

    plt.title("Extracted Procedural Knowledge Graph: 3M OEM SOP", fontsize=18, pad=18)
    plt.tight_layout()
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
