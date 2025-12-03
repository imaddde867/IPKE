"""
Script to generate Figure 13: Distribution of dominant error types.
Produces the horizontal bar chart for the thesis results section.
"""

import matplotlib.pyplot as plt
import os

def plot_error_distribution(output_path: str = "assets/fig13_error_distribution.png") -> None:
    """
    Generates the error distribution bar chart.
    """
    # Data from the manual evaluation phase
    error_types = [
        "Ambiguous conditional phrasing",
        "Cross-chunk references",
        "Implicit step omissions",
        "Hallucinated constraints"
    ]
    shares = [0.33, 0.27, 0.21, 0.19]

    # Matplotlib plots bottom-to-top, so reverse the lists to keep the order above
    labels = error_types[::-1]
    values = shares[::-1]

    # Half-page width for the thesis layout
    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars = ax.barh(labels, values, color='#2078b4', height=0.6, zorder=3)

    # Clean up the frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.spines['bottom'].set_color('#333333')
    ax.spines['bottom'].set_linewidth(0.8)

    ax.set_xlim(0, 0.4)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xticklabels(['0%', '10%', '20%', '30%', '40%'], fontsize=11)
    
    # Clean y-axis (labels only, no ticks)
    ax.tick_params(axis='y', which='both', length=0, labelsize=12)
    ax.tick_params(axis='x', labelsize=11, color='#333333')

    # Label bars with exact percentages
    for bar in bars:
        width = bar.get_width()
        x_pos = width + 0.005
        y_pos = bar.get_y() + bar.get_height() / 2
        
        ax.text(x_pos, y_pos, f'{width:.0%}', 
                va='center', ha='left', 
                fontsize=11, color='#333333')

    ax.set_xlabel('Share of observed errors (%)', fontsize=12, labelpad=10, color='#333333')
    
    ax.set_title('Figure 13. Distribution of dominant error types', 
                 fontsize=14, fontweight='bold', 
                 loc='left', pad=20, color='#111111')

    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    plot_error_distribution()
