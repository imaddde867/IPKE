"""
Script to generate the chunking method comparison chart.
Visualizes the performance of different chunking strategies across multiple metrics
for the thesis results section.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_chunking_comparison(output_path: str = "assets/chunking_comparison_chart.png") -> None:
    """
    Generates a grouped bar chart comparing chunking methods.
    """
    # Metrics to display (Reordered to highlight DSC strengths first)
    metrics = [
        "Procedural Fidelity Φ",
        "Adjacency F1",
        "Constraint Cov.",
        "Step F1",
        "Kendall τ"
    ]
    
    # Data from the results table
    # Using a dictionary to map method names to their scores across the metrics above
    # Reordered values to match the new metric order: [Phi, Adj, Const, Step, Tau]
    data = {
        "Fixed":                [0.253, 0.333, 0.075, 0.126, 0.889],
        "Fixed + Overlap":      [0.311, 0.000, 0.192, 0.146, 0.856],
        "Breakpoint Semantic":  [0.260, 0.190, 0.117, 0.094, 0.870],
        "Dual Semantic":        [0.362, 0.797, 0.350, 0.093, 0.797]
    }

    # Define colors: Professional grayscale gradient for baselines, Royal Blue for main method
    colors = [
        '#C0C0C0',  # Fixed (Silver)
        '#708090',  # Fixed + Overlap (Slate Grey)
        '#2F4F4F',  # Breakpoint (Charcoal)
        '#4169E1'   # Dual Semantic (Royal Blue Accent)
    ]

    # Setup plot
    x = np.arange(len(metrics))  # label locations
    width = 0.2  # width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars for each method
    for (method_name, scores), color in zip(data.items(), colors):
        offset = width * multiplier
        # Calculate position: center the group of bars on the tick
        # Total width of group = width * 4
        # Start = x - (total_width/2) + offset + width/2
        # Simplified: x + offset - (width * 1.5)
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

    # Add faint horizontal grid lines for readability
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
    plot_chunking_comparison()
