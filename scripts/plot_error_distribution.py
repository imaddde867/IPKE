"""
Script to generate Figure 13: Distribution of dominant error types.
"""

import matplotlib.pyplot as plt
import os

# Style Constants
BAR_COLOR = '#2078b4'
TEXT_COLOR = '#333333'
FONT_SIZE_LABEL = 11
FONT_SIZE_TITLE = 12

def plot_error_distribution(output_path: str = "assets/fig13_error_distribution.png") -> None:
    error_types = [
        "Ambiguous conditional phrasing",
        "Cross-chunk references",
        "Implicit step omissions",
        "Hallucinated constraints"
    ]
    shares = [0.33, 0.27, 0.21, 0.19]

    # Reverse for horizontal bar chart (top-to-bottom)
    labels = error_types[::-1]
    values = shares[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, values, color=BAR_COLOR, height=0.6, zorder=3)

    # Styling
    ax.set_xlim(0, 0.4)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xticklabels(['0%', '10%', '20%', '30%', '40%'], fontsize=FONT_SIZE_LABEL)
    
    ax.tick_params(axis='y', which='both', length=0, labelsize=12)
    ax.tick_params(axis='x', labelsize=FONT_SIZE_LABEL, color=TEXT_COLOR)

    # Minimalist Spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['bottom'].set_linewidth(0.8)

    # Value Labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f'{width:.0%}', 
                va='center', ha='left', fontsize=FONT_SIZE_LABEL, color=TEXT_COLOR)

    ax.set_xlabel('Share of observed errors (%)', fontsize=FONT_SIZE_TITLE, labelpad=10, color=TEXT_COLOR)
    ax.set_title('Figure 13. Distribution of dominant error types', 
                 fontsize=14, fontweight='bold', loc='left', pad=20, color='#111111')

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    plot_error_distribution()
