import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_procedural_kg_figure(output_filename="figure_X_procedural_KG.png"):
    """
    Generates a high-readability figure for a procedural knowledge graph.
    """
    G = nx.DiGraph()

    # Define nodes with clearer line breaks for large text
    step_nodes = [
        "Step 1:\nVerify V-23\nClosed",
        "Step 2:\nCheck Oil\nLevel",
        "Step 3:\nConfirm No\nPersonnel",
        "Step 4:\nPress Start"
    ]
    condition_parameter_nodes = [
        "Param:\nOil Level\n> 80%",
        "Cond:\nNo Personnel\nIn Zone"
    ]
    equipment_action_nodes = [
        "Equip:\nV-23",
        "Equip:\nP-101",
        "Action:\nVerify\nClosed",
        "Action:\nPress\nStart"
    ]

    # Add nodes
    for node in step_nodes:
        G.add_node(node, type='step', color='#87CEEB') # SkyBlue
    for node in condition_parameter_nodes:
        G.add_node(node, type='constraint', color='#FA8072') # Salmon
    for node in equipment_action_nodes:
        G.add_node(node, type='other', color='#D3D3D3') # LightGray

    # Define edges
    edges = [
        ("Step 1:\nVerify V-23\nClosed", "Step 2:\nCheck Oil\nLevel", {'type': 'Precedes', 'label': 'NEXT', 'style': 'solid'}),
        ("Step 2:\nCheck Oil\nLevel", "Step 3:\nConfirm No\nPersonnel", {'type': 'Precedes', 'label': 'NEXT', 'style': 'solid'}),
        ("Step 3:\nConfirm No\nPersonnel", "Step 4:\nPress Start", {'type': 'Precedes', 'label': 'NEXT', 'style': 'solid'}),

        ("Step 2:\nCheck Oil\nLevel", "Param:\nOil Level\n> 80%", {'type': 'Requires', 'label': 'REQUIRES', 'style': 'solid'}),
        ("Step 3:\nConfirm No\nPersonnel", "Cond:\nNo Personnel\nIn Zone", {'type': 'Requires', 'label': 'REQUIRES', 'style': 'solid'}),

        ("Cond:\nNo Personnel\nIn Zone", "Step 4:\nPress Start", {'type': 'Guards', 'label': 'GUARDS', 'style': 'dashed'}),

        ("Action:\nVerify\nClosed", "Equip:\nV-23", {'type': 'ActsOn', 'label': 'ACTS ON', 'style': 'solid'}),
        ("Action:\nPress\nStart", "Equip:\nP-101", {'type': 'ActsOn', 'label': 'ACTS ON', 'style': 'solid'}),

        ("Step 1:\nVerify V-23\nClosed", "Action:\nVerify\nClosed", {'type': 'ContainsAction', 'label': 'CONTAINS', 'style': 'dotted'}),
        ("Step 4:\nPress Start", "Action:\nPress\nStart", {'type': 'ContainsAction', 'label': 'CONTAINS', 'style': 'dotted'}),
    ]

    G.add_edges_from([(u, v, data) for u, v, data in edges])

    # Get node colors
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    
    # Layout settings
    pos = nx.spring_layout(G, k=1.8, iterations=100, seed=42) # Increased k further

    # Create figure
    plt.figure(figsize=(26, 20)) # Even larger figure size

    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=12000, edgecolors='black', linewidths=1.0, alpha=1.0) # Increased node size
    
    # Draw Node Labels
    nx.draw_networkx_labels(G, pos, font_size=18, font_weight='bold', font_color='black') # Increased label font size

    # Draw Edges
    solid_edges = [ (u,v) for u,v,data in G.edges(data=True) if data['style'] == 'solid' ]
    dashed_edges = [ (u,v) for u,v,data in G.edges(data=True) if data['style'] == 'dashed' ]
    dotted_edges = [ (u,v) for u,v,data in G.edges(data=True) if data['style'] == 'dotted' ]

    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, edge_color='#555555', width=2.5, arrowstyle='-|>', arrowsize=40, min_source_margin=45, min_target_margin=45) # Increased width and arrowsize
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, edge_color='#8B0000', style='dashed', width=2.5, arrowstyle='-|>', arrowsize=40, min_source_margin=45, min_target_margin=45) # Increased width and arrowsize
    nx.draw_networkx_edges(G, pos, edgelist=dotted_edges, edge_color='#006400', style='dotted', width=2.5, arrowstyle='-|>', arrowsize=40, min_source_margin=45, min_target_margin=45) # Increased width and arrowsize

    # Draw Edge Labels
    edge_labels = { (u,v): G[u][v]['label'] for u,v in G.edges() }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=14, font_weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)) # Increased edge label font size

    # Legend
    node_legend_patches = [
        mpatches.Patch(facecolor='#87CEEB', edgecolor='black', label='Step'),
        mpatches.Patch(facecolor='#FA8072', edgecolor='black', label='Constraint (Condition/Param)'),
        mpatches.Patch(facecolor='#D3D3D3', edgecolor='black', label='Equipment/Action')
    ]
    edge_legend_patches = [
        mpatches.Patch(color='#555555', label='Precedes / Requires / Acts On'),
        mpatches.Patch(color='#8B0000', linestyle='--', label='Guards'),
        mpatches.Patch(color='#006400', linestyle=':', label='Contains Action')
    ]

    plt.legend(handles=node_legend_patches + edge_legend_patches, loc='lower right', fontsize=16, framealpha=1.0, fancybox=True, shadow=True) # Increased legend font size

    plt.title("Procedural Knowledge Graph Example: 'Pump Startup'", size=28, weight='bold', pad=20) # Increased title font size
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Figure saved to {output_filename}")

if __name__ == "__main__":
    generate_procedural_kg_figure()
