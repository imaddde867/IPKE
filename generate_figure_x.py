import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Style Constants
COLORS = {
    "Step": "#ADD8E6",       # Light Blue
    "Action": "#F0F0F0",     # Light Gray
    "Asset": "#90EE90",      # Light Green
    "Constraint": "#FFB6C1", # Light Pink
    "Role": "#FFD700"        # Gold
}
NODE_SHAPES = {
    "Step": "o", "Action": "o", "Asset": "s", "Constraint": "s", "Role": "d"
}
NODE_SIZES = {
    "Step": 3000, "Action": 1800, "Asset": 1500, "Constraint": 2000, "Role": 1500
}
EDGE_STYLES = {
    "NEXT":         {"color": "black",    "style": "solid",  "width": 2.0, "arrow": 20, "label": "NEXT"},
    "GUARD":        {"color": "red",      "style": "dashed", "width": 2.0, "arrow": 20, "label": "GUARD"},
    "REQUIRES":     {"color": "dimgray",  "style": "dotted", "width": 2.0, "arrow": 15, "label": "REQUIRES"},
    "ACTS_ON":      {"color": "green",    "style": "solid",  "width": 2.0, "arrow": 15, "label": "ACTS_ON"},
    "PERFORMED_BY": {"color": "orange",   "style": "solid",  "width": 1.5, "arrow": 15, "label": "PERFORMED_BY"},
    "ACTION":       {"color": "gray",     "style": "solid",  "width": 1.0, "arrow": 10, "label": "ACTION"}
}

def generate_figure(output_file="figure_X_procedural_KG.png"):
    G = nx.DiGraph()

    # --- Data Definition ---
    nodes = {
        "Step": ["Step1", "Step2", "Step3", "Step4"],
        "Action": ["VerifyClosed", "CheckOilLevel", "ConfirmNoPersonnel", "PressStart"],
        "Asset": ["V-23", "P-101"],
        "Constraint": ["oil_level > 80%", "no_personnel_in_zone"],
        "Role": ["Operator"]
    }

    edges = [
        ("Step1", "Step2", "NEXT"), ("Step2", "Step3", "NEXT"), ("Step3", "Step4", "NEXT"),
        ("Step2", "oil_level > 80%", "REQUIRES"), ("Step3", "no_personnel_in_zone", "REQUIRES"),
        ("no_personnel_in_zone", "Step4", "GUARD"),
        ("VerifyClosed", "V-23", "ACTS_ON"), ("PressStart", "P-101", "ACTS_ON"),
        ("VerifyClosed", "Operator", "PERFORMED_BY"), ("CheckOilLevel", "Operator", "PERFORMED_BY"),
        ("ConfirmNoPersonnel", "Operator", "PERFORMED_BY"), ("PressStart", "Operator", "PERFORMED_BY"),
        ("Step1", "VerifyClosed", "ACTION"), ("Step2", "CheckOilLevel", "ACTION"),
        ("Step3", "ConfirmNoPersonnel", "ACTION"), ("Step4", "PressStart", "ACTION")
    ]

    pos = {
        "Step1": (0, 0), "Step2": (3, 0), "Step3": (6, 0), "Step4": (9, 0),
        "VerifyClosed": (0, 1.5), "CheckOilLevel": (3, 1.5), "ConfirmNoPersonnel": (6, 1.5), "PressStart": (9, 1.5),
        "V-23": (0, 3), "P-101": (9, 3), "Operator": (4.5, 3.5),
        "oil_level > 80%": (3, -1.5), "no_personnel_in_zone": (6, -1.5)
    }

    # --- Graph Construction ---
    for ntype, names in nodes.items():
        G.add_nodes_from(names, type=ntype)
    for u, v, rel in edges:
        G.add_edge(u, v, relation=rel)

    # --- Plotting ---
    plt.figure(figsize=(14, 9))
    ax = plt.gca()
    plt.axis('off')

    # Draw Nodes
    for ntype, names in nodes.items():
        nx.draw_networkx_nodes(G, pos, nodelist=names, node_color=COLORS[ntype],
                               node_shape=NODE_SHAPES[ntype], node_size=NODE_SIZES[ntype],
                               edgecolors='black', linewidths=1)

    # Draw Labels
    labels = {n: n.replace("_", "\n") if " > " not in n else n for n in G.nodes()}
    labels["oil_level > 80%"] = "Oil Level\n> 80%"
    labels["no_personnel_in_zone"] = "No Personnel\nIn Zone"
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold")

    # Draw Edges
    for rel, style in EDGE_STYLES.items():
        edgelist = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == rel]
        if not edgelist: continue
        
        conn = "arc3,rad=-0.2" if rel == "PERFORMED_BY" else "arc3,rad=0.2" if rel == "GUARD" else "arc3,rad=0"
        nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color=style["color"],
                               style=style["style"], width=style["width"], arrowsize=style["arrow"],
                               connectionstyle=conn)
        
        # Edge Labels
        if rel not in ["ACTION"]: # Skip obvious ones if needed
            nx.draw_networkx_edge_labels(G, pos, edge_labels={e: style["label"] for e in edgelist},
                                         font_color=style["color"], font_size=7, label_pos=0.7 if rel == "GUARD" else 0.5)

    # --- Legend ---
    legend_nodes = [mpatches.Patch(color=c, label=t) for t, c in COLORS.items()]
    legend_edges = [mlines.Line2D([], [], color=s["color"], ls=s["style"], lw=2, label=s["label"]) 
                    for r, s in EDGE_STYLES.items()]

    plt.legend(handles=legend_nodes, loc='lower left', title="Node Types", frameon=True)
    # Add second legend manually
    leg2 = plt.legend(handles=legend_edges, loc='lower right', title="Edge Relations", frameon=True)
    ax.add_artist(plt.legend(handles=legend_nodes, loc='lower left', title="Node Types", frameon=True))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file}")

if __name__ == "__main__":
    generate_figure()
