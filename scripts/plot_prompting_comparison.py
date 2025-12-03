"""
Script to generate the prompting strategy comparison chart.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Style Constants
COLORS = ['#C0C0C0', '#708090', '#2F4F4F', '#4169E1']
TEXT_COLOR = '#333333'
FONT_SIZE_LABEL = 11
FONT_SIZE_TITLE = 12

def plot_prompting_comparison(output_path: str = "assets/prompting_comparison_chart.png") -> None:
    metrics = ["Procedural Fidelity Φ", "Adjacency F1", "Constraint Cov.", "Step F1", "Kendall τ"]
    
    data = {
        "P0 (Baseline)":  [0.253, 0.176, 0.033, 0.305, 0.723],
        "P1 (Few-Shot)":  [0.407, 0.314, 0.342, 0.313, 0.708],
        "P2 (CoT)":       [0.000, 0.000, 0.000, 0.000, 0.000],
        "P3 (Two-Stage)": [0.611, 0.048, 0.708, 0.377, 0.716]
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
    plot_prompting_comparison()
