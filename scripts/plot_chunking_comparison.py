"""
Script to generate the chunking method comparison chart.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Style Constants
COLORS = ['#C0C0C0', '#708090', '#2F4F4F', '#4169E1'] # Silver, Slate, Charcoal, Royal Blue
TEXT_COLOR = '#333333'
FONT_SIZE_LABEL = 11
FONT_SIZE_TITLE = 12

def plot_chunking_comparison(output_path: str = "assets/chunking_comparison_chart.png") -> None:
    metrics = ["Procedural Fidelity Φ", "Adjacency F1", "Constraint Cov.", "Step F1", "Kendall τ"]
    
    # Data: {Method: [Phi, Adj, Const, Step, Tau]}
    data = {
        "Fixed":                [0.253, 0.333, 0.075, 0.126, 0.889],
        "Fixed + Overlap":      [0.311, 0.000, 0.192, 0.146, 0.856],
        "Breakpoint Semantic":  [0.260, 0.190, 0.117, 0.094, 0.870],
        "Dual Semantic":        [0.362, 0.797, 0.350, 0.093, 0.797]
    }

    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ((method, scores), color) in enumerate(zip(data.items(), COLORS)):
        offset = width * i
        ax.bar(x + offset - (width * 1.5), scores, width, label=method, color=color, zorder=3)

    # Styling
    ax.set_ylabel('Score (0-1)', fontsize=FONT_SIZE_TITLE, color=TEXT_COLOR, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=FONT_SIZE_LABEL, color=TEXT_COLOR)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_LABEL, color=TEXT_COLOR)
    
    # Minimalist Spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['bottom'].set_linewidth(0.8)
    
    ax.grid(axis='y', linestyle='-', alpha=0.2, zorder=0)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=False, shadow=False, ncol=4, frameon=False, fontsize=FONT_SIZE_LABEL)

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    plot_chunking_comparison()
