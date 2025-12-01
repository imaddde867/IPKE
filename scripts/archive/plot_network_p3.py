import json
import networkx as nx
import matplotlib.pyplot as plt

# Load P3 Data
path = "logs/prompting_grid/P3_two_stage/3M_OEM_SOP/predictions.json"
try:
    with open(path) as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: Run the P3 experiment first.")
    exit()

G = nx.DiGraph()

# Add Steps
for s in data.get('steps', []):
    # Use short ID for label to keep it clean
    G.add_node(s['id'], color='lightblue', node_type='step')

# Add Constraints and link them
for c in data.get('constraints', []):
    c_id = c.get('id', 'C?')
    G.add_node(c_id, color='#ffcccb', node_type='constraint') # Light red
    
    # Link to steps
    attached = c.get('attached_to') or c.get('steps') or []
    if isinstance(attached, str): attached = [attached]
    
    for target in attached:
        if target in G.nodes:
            G.add_edge(c_id, target, color='red', style='dashed')

# Add Procedural Sequence (S1->S2->S3...)
steps = sorted(data.get('steps', []), key=lambda x: x.get('order', 0))
for i in range(len(steps)-1):
    G.add_edge(steps[i]['id'], steps[i+1]['id'], color='blue', style='solid')

# Layout and Draw
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.9, iterations=50)

node_colors = [nx.get_node_attributes(G, 'color')[n] for n in G.nodes]
edge_colors = [G[u][v].get('color', 'black') for u,v in G.edges]
edge_styles = [G[u][v].get('style', 'solid') for u,v in G.edges]

nx.draw(G, pos, 
        node_color=node_colors, 
        edge_color=edge_colors, 
        style=edge_styles,
        with_labels=True, 
        node_size=600, 
        font_size=8, 
        font_weight='bold',
        arrowsize=15)

# Custom Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', label='Procedural Step', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffcccb', label='Safety Constraint', markersize=10),
    Line2D([0], [0], color='blue', lw=2, label='Sequence Flow'),
    Line2D([0], [0], color='red', lw=2, linestyle='--', label='Logic/Guard'),
]
plt.legend(handles=legend_elements, loc='upper left')

plt.title("Extracted Knowledge Graph Structure (3M SOP - P3 Strategy)")
plt.savefig('p3_network_graph.png', dpi=300)
print("Graph saved as p3_network_graph.png")
