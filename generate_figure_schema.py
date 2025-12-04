"""
Generate Figure: IPKE Procedural Knowledge Graph Schema Example

Visualizes the minimal schema for industrial procedural knowledge representation:
- Nodes: Step, Action, Equipment, Condition, Parameter
- Edges: Precedes, Requires, Guards, ActsOn

Example: "Before starting pump P-101: (1) Verify valve V-23 is closed; 
         (2) Check oil level > 80%; (3) Confirm no personnel in hazard zone; (4) Press Start."
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

def create_pkg_schema_figure():
    """Create the procedural knowledge graph schema visualization."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 8)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Color scheme (thesis-aligned)
    COLORS = {
        'step': '#3498db',           # Blue - Steps
        'action': '#9b59b6',         # Purple - Actions
        'equipment': '#27ae60',      # Green - Equipment/Assets
        'condition': '#e74c3c',      # Red - Conditions/Guards
        'parameter': '#f39c12',      # Orange - Parameters/Thresholds
        'next': '#2c3e50',           # Dark - NEXT/Precedes edges
        'guard': '#c0392b',          # Dark red - GUARD edges
        'requires': '#7f8c8d',       # Gray - REQUIRES edges
        'actson': '#16a085',         # Teal - ACTS_ON edges
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NODE POSITIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Steps (main horizontal flow)
    steps = {
        'S1': (1, 4),
        'S2': (3.5, 4),
        'S3': (6, 4),
        'S4': (8.5, 4),
    }
    
    # Actions (below steps)
    actions = {
        'A1': (1, 2.2),      # VerifyClosed
        'A2': (3.5, 2.2),    # CheckLevel
        'A3': (6, 2.2),      # ConfirmClear
        'A4': (8.5, 2.2),    # PressStart
    }
    
    # Equipment (bottom)
    equipment = {
        'E1': (0, 0.5),      # V-23 (valve)
        'E2': (9.5, 0.5),    # P-101 (pump)
    }
    
    # Parameters/Conditions (above steps)
    conditions = {
        'P1': (3.5, 6.2),    # oil_level > 80%
        'C1': (6, 6.2),      # no_personnel_in_zone
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DRAW NODES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def draw_node(x, y, label, sublabel, color, shape='o', size=0.55):
        """Draw a node with label."""
        if shape == 'o':
            circle = plt.Circle((x, y), size, color=color, ec='white', linewidth=2, zorder=10)
            ax.add_patch(circle)
        elif shape == 's':  # square for equipment
            rect = mpatches.FancyBboxPatch(
                (x - size, y - size*0.7), size*2, size*1.4,
                boxstyle="round,pad=0.05", facecolor=color, edgecolor='white', linewidth=2, zorder=10
            )
            ax.add_patch(rect)
        elif shape == 'd':  # diamond for conditions
            diamond = mpatches.RegularPolygon(
                (x, y), numVertices=4, radius=size*1.1, orientation=np.pi/4,
                facecolor=color, edgecolor='white', linewidth=2, zorder=10
            )
            ax.add_patch(diamond)
        elif shape == 'h':  # hexagon for parameters
            hexagon = mpatches.RegularPolygon(
                (x, y), numVertices=6, radius=size*1.0,
                facecolor=color, edgecolor='white', linewidth=2, zorder=10
            )
            ax.add_patch(hexagon)
        
        # Labels
        ax.text(x, y + 0.05, label, ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white', zorder=11)
        ax.text(x, y - 0.25, sublabel, ha='center', va='center', fontsize=7, 
                color='white', zorder=11)
    
    # Draw Steps (circles)
    step_labels = [
        ('Step 1', 'Verify'),
        ('Step 2', 'Check'),
        ('Step 3', 'Confirm'),
        ('Step 4', 'Start'),
    ]
    for (sid, pos), (label, sublabel) in zip(steps.items(), step_labels):
        draw_node(pos[0], pos[1], label, sublabel, COLORS['step'], 'o', 0.6)
    
    # Draw Actions (circles, smaller)
    action_labels = [
        ('Action', 'VerifyClosed'),
        ('Action', 'CheckLevel'),
        ('Action', 'ConfirmClear'),
        ('Action', 'PressStart'),
    ]
    for (aid, pos), (label, sublabel) in zip(actions.items(), action_labels):
        draw_node(pos[0], pos[1], label, sublabel, COLORS['action'], 'o', 0.5)
    
    # Draw Equipment (rectangles)
    draw_node(equipment['E1'][0], equipment['E1'][1], 'Equipment', 'V-23', COLORS['equipment'], 's', 0.6)
    draw_node(equipment['E2'][0], equipment['E2'][1], 'Equipment', 'P-101', COLORS['equipment'], 's', 0.6)
    
    # Draw Parameter (hexagon)
    draw_node(conditions['P1'][0], conditions['P1'][1], 'Parameter', 'oil > 80%', COLORS['parameter'], 'h', 0.55)
    
    # Draw Condition (diamond)
    draw_node(conditions['C1'][0], conditions['C1'][1], 'Condition', 'zone clear', COLORS['condition'], 'd', 0.55)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DRAW EDGES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def draw_edge(start, end, color, style='-', label='', curved=0, offset_start=0.6, offset_end=0.6):
        """Draw an edge with optional label."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Normalize and offset
        ux, uy = dx/dist, dy/dist
        x1 = start[0] + ux * offset_start
        y1 = start[1] + uy * offset_start
        x2 = end[0] - ux * offset_end
        y2 = end[1] - uy * offset_end
        
        linestyle = '-' if style == '-' else '--'
        linewidth = 2.5 if style == '-' else 2
        
        if curved != 0:
            # Curved arrow
            mid_x = (x1 + x2) / 2 + curved
            mid_y = (y1 + y2) / 2 + abs(curved) * 0.3
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                      linestyle=linestyle,
                                      connectionstyle=f'arc3,rad={curved*0.3}'),
                       zorder=5)
        else:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                      linestyle=linestyle),
                       zorder=5)
        
        # Edge label
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2 + 0.2
            ax.text(mid_x, mid_y, label, ha='center', va='bottom', fontsize=8,
                   fontstyle='italic', color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                            edgecolor='none', alpha=0.8))
    
    # PRECEDES/NEXT edges (Step → Step) - solid arrows
    draw_edge(steps['S1'], steps['S2'], COLORS['next'], '-', 'NEXT')
    draw_edge(steps['S2'], steps['S3'], COLORS['next'], '-', 'NEXT')
    draw_edge(steps['S3'], steps['S4'], COLORS['next'], '-', 'NEXT')
    
    # REQUIRES edges (Step → Parameter/Condition) - gray dashed
    draw_edge(steps['S2'], conditions['P1'], COLORS['requires'], '--', 'REQUIRES', offset_start=0.6, offset_end=0.55)
    draw_edge(steps['S3'], conditions['C1'], COLORS['requires'], '--', 'REQUIRES', offset_start=0.6, offset_end=0.6)
    
    # GUARD edges (Condition → Step) - red dashed
    draw_edge(conditions['P1'], steps['S4'], COLORS['guard'], '--', 'GUARD', curved=0.3, offset_start=0.55, offset_end=0.6)
    draw_edge(conditions['C1'], steps['S4'], COLORS['guard'], '--', 'GUARD', curved=-0.15, offset_start=0.6, offset_end=0.6)
    
    # Step → Action edges (implicit execution)
    for sid, aid in [('S1', 'A1'), ('S2', 'A2'), ('S3', 'A3'), ('S4', 'A4')]:
        draw_edge(steps[sid], actions[aid], '#7f8c8d', '-', '', offset_start=0.6, offset_end=0.5)
    
    # ACTS_ON edges (Action → Equipment)
    draw_edge(actions['A1'], equipment['E1'], COLORS['actson'], '-', 'ACTS_ON', curved=0.2, offset_start=0.5, offset_end=0.5)
    draw_edge(actions['A4'], equipment['E2'], COLORS['actson'], '-', 'ACTS_ON', curved=-0.2, offset_start=0.5, offset_end=0.5)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEGEND
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Node legend
    legend_y = 7.5
    legend_items = [
        (0.5, 'Step', COLORS['step'], 'o'),
        (2.0, 'Action', COLORS['action'], 'o'),
        (3.5, 'Equipment', COLORS['equipment'], 's'),
        (5.2, 'Condition', COLORS['condition'], 'd'),
        (7.0, 'Parameter', COLORS['parameter'], 'h'),
    ]
    
    for x, label, color, shape in legend_items:
        if shape == 'o':
            circle = plt.Circle((x, legend_y), 0.25, color=color, ec='white', linewidth=1.5)
            ax.add_patch(circle)
        elif shape == 's':
            rect = mpatches.FancyBboxPatch((x-0.25, legend_y-0.18), 0.5, 0.36,
                                           boxstyle="round,pad=0.02", facecolor=color, 
                                           edgecolor='white', linewidth=1.5)
            ax.add_patch(rect)
        elif shape == 'd':
            diamond = mpatches.RegularPolygon((x, legend_y), numVertices=4, radius=0.28,
                                              orientation=np.pi/4, facecolor=color, 
                                              edgecolor='white', linewidth=1.5)
            ax.add_patch(diamond)
        elif shape == 'h':
            hexagon = mpatches.RegularPolygon((x, legend_y), numVertices=6, radius=0.28,
                                              facecolor=color, edgecolor='white', linewidth=1.5)
            ax.add_patch(hexagon)
        
        ax.text(x + 0.4, legend_y, label, ha='left', va='center', fontsize=9)
    
    # Edge legend
    edge_legend_y = -0.5
    edge_items = [
        (1.0, 'NEXT (Precedes)', COLORS['next'], '-'),
        (3.5, 'GUARD', COLORS['guard'], '--'),
        (5.5, 'REQUIRES', COLORS['requires'], '--'),
        (7.8, 'ACTS_ON', COLORS['actson'], '-'),
    ]
    
    for x, label, color, style in edge_items:
        linestyle = '-' if style == '-' else '--'
        ax.plot([x-0.3, x+0.3], [edge_legend_y, edge_legend_y], 
               color=color, linestyle=linestyle, linewidth=2.5)
        ax.annotate('', xy=(x+0.35, edge_legend_y), xytext=(x+0.2, edge_legend_y),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.text(x + 0.5, edge_legend_y, label, ha='left', va='center', fontsize=9)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TITLE & CAPTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Title
    ax.text(5, 8.5, 'IPKE Procedural Knowledge Graph Schema', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Source procedure text
    ax.text(5, -1.3, 
           '"Before starting pump P-101: (1) Verify valve V-23 is closed; (2) Check oil level > 80%;'
           '\n(3) Confirm no personnel in hazard zone; (4) Press Start."',
           ha='center', va='top', fontsize=9, style='italic', color='#555')
    
    plt.tight_layout()
    return fig


def main():
    """Generate and save the figure."""
    fig = create_pkg_schema_figure()
    
    # Save in multiple formats
    output_path = 'assets/figure_pkg_schema'
    fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    fig.savefig(f'{output_path}.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✓ Saved: {output_path}.png")
    print(f"✓ Saved: {output_path}.pdf")
    
    plt.show()


if __name__ == "__main__":
    main()
