"""
Script to generate the prompting strategy comparison chart (Figure 9).
Visualizes the performance of different prompting strategies (P0-P3) across multiple metrics
for the thesis results section.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_prompting_comparison(output_path: str = "assets/prompting_comparison_chart.png") -> None:
    """
    Generates a grouped bar chart comparing prompting strategies.
    """
    # Metrics to display (Consistent with chunking comparison order)
    metrics = [
        "Procedural Fidelity Φ",
        "Adjacency F1",
        "Constraint Cov.",
        "Step F1",
        "Kendall τ"
    ]
    
    # Data from the results table
    # Reordered values to match the metric order: [Phi, Adj, Const, Step, Tau]
    data = {
        "P0 (Baseline)":  [0.253, 0.176, 0.033, 0.305, 0.723],
        "P1 (Few-Shot)":  [0.407, 0.314, 0.342, 0.313, 0.708],
        "P2 (CoT)":       [0.000, 0.000, 0.000, 0.000, 0.000],
        "P3 (Two-Stage)": [0.611, 0.048, 0.708, 0.377, 0.716]
    }

    # Define colors: Professional grayscale gradient for baselines, Royal Blue for main method
    colors = [
        '#C0C0C0',  # P0 (Silver)
        '#708090',  # P1 (Slate Grey)
        '#2F4F4F',  # P2 (Charcoal)
        '#4169E1'   # P3 (Royal Blue Accent)
    ]

    # Setup plot
    x = np.arange(len(metrics))
    width = 0.2
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6))

    for (method_name, scores), color in zip(data.items(), colors):
        offset = width * multiplier
        # Calculate position: center the group of bars on the tick
        rects = ax.bar(x + offset - (width * 1.5), scores, width, 
                       label=method_name, color=color, zorder=3)
        multiplier += 1

    # Styling
    ax.set_ylabel('Score (0-1)', fontsize=12, color='#333333', labelpad=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, color='#333333')
    
    # Clean up y-axis
    ax.tick_params(axis='y', labelsize=11, color='#333333')
    ax.set_ylim(0, 1.0)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['bottom'].set_linewidth(0.8)

    # Add faint horizontal grid lines
    ax.grid(axis='y', linestyle='-', alpha=0.2, zorder=0)

    # Legend configuration
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=False, shadow=False, ncol=4, frameon=False, fontsize=11)

    plt.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    plot_prompting_comparison()
