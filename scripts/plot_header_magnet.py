"""Plot a semantic signal with a structural "header magnet" bonus."""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Synthetic procedural text reused from the Signal Dip chart
SENTENCES: List[str] = [
    "Preheat the oven to 350°F and line a round cake pan with parchment paper.",
    "Whisk flour, sugar, and cocoa powder until the dry mixture looks uniform",
    "Fold in eggs, melted butter, and vanilla to create a glossy cake batter.",
    "Pour the batter into the prepared pan and bake until the center springs back.",
    "Guide the car onto the shoulder safely and switch on the hazard lights.",
    "Position the jack, raise the car slightly, and loosen the lug nuts on the flat tire.",
    "Remove the damaged wheel, mount the spare onto the studs, and hand-tighten the nuts.",
    "Lower the vehicle, torque the lug nuts firmly, and stow the jack and tire kit."
]

OUTPUT_PATH = "assets/header_magnet.png"
HEADER_TRANSITION_INDEX = 4  # Transition 4→5 acts as the structural anchor


def compute_adjacent_similarities(sentences: List[str]) -> Tuple[List[str], np.ndarray]:
    """Encode sentences with SBERT and compute cosine similarity between neighbors."""

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

    transitions: List[str] = []
    similarities: List[float] = []

    for idx in range(len(embeddings) - 1):
        pair_similarity = cosine_similarity(
            embeddings[idx : idx + 1],
            embeddings[idx + 1 : idx + 2],
        )[0][0]
        similarities.append(float(pair_similarity))
        transitions.append(f"{idx}→{idx + 1}")

    return transitions, np.array(similarities)


def plot_header_magnet(transitions: List[str], values: np.ndarray, output_path: str) -> None:
    """Visualize the semantic signal and highlight the structural header bonus."""

    x_positions = np.arange(len(values))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#CCCCCC",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        x_positions,
        values,
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        markersize=8,
    )
    ax.fill_between(x_positions, values, color="#1f77b4", alpha=0.08)
    ax.margins(x=0.02)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(transitions, fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(x_positions[0], x_positions[-1])

    ax.set_xlabel("Sentence Transition Index (k → k+1)", fontsize=13)
    ax.set_ylabel("Cosine Similarity Score", fontsize=13)
    ax.set_title("Header Magnet: Structural Bonus on Semantic Signal", fontsize=18, pad=18)

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Annotation for cohesive (Topic A) segment
    cohesive_idx = 1
    ax.annotate(
        r"Cohesive Phase (High $c(u,v)$)",
        xy=(cohesive_idx, values[cohesive_idx]),
        xytext=(cohesive_idx + 0.7, min(values[cohesive_idx] + 0.15, 0.95)),
        arrowprops=dict(arrowstyle="->", color="#2E8B57", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2E8B57", lw=0.8),
        fontsize=10,
        color="#2E8B57",
    )

    # Highlight the natural semantic dip at transition 3→4
    dip_idx = int(np.argmin(values))
    ax.annotate(
        "Semantic Dip at 3→4",
        xy=(dip_idx, values[dip_idx]),
        xytext=(dip_idx - 1.2, max(values[dip_idx] - 0.25, 0.05)),
        arrowprops=dict(arrowstyle="->", color="#B22222", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#B22222", lw=0.8),
        fontsize=10,
        color="#B22222",
    )

    # Draw the structural header bonus at transition 4→5
    header_x = x_positions[HEADER_TRANSITION_INDEX]
    ax.axvline(header_x, color="#2E8B57", linestyle="--", linewidth=2, alpha=0.9)
    ax.annotate(
        r"+$\beta$ Bonus (Structural Anchor)",
        xy=(header_x, values[HEADER_TRANSITION_INDEX]),
        xytext=(header_x + 0.6, 0.9),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#2E8B57", lw=1.3),
        color="#2E8B57",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2E8B57", lw=0.9),
    )

    # Final note that the boundary is forced because of the header
    ax.annotate(
        "Header magnet fixes boundary despite earlier dip",
        xy=(header_x, values[HEADER_TRANSITION_INDEX]),
        xytext=(header_x + 0.7, values[HEADER_TRANSITION_INDEX] - 0.25),
        arrowprops=dict(arrowstyle="->", color="#006d77", lw=1.4),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#006d77", lw=0.8),
        fontsize=10,
        color="#006d77",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Header magnet figure saved to {output_path}")


def main() -> None:
    transitions, similarities = compute_adjacent_similarities(SENTENCES)
    plot_header_magnet(transitions, similarities, OUTPUT_PATH)


if __name__ == "__main__":
    main()
