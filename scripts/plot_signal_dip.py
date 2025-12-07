"""Generate a semantic cohesion "signal dip" visualization."""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Eight-step procedural text with an intentional topic shift after sentence 3
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

OUTPUT_PATH = "assets/signal_dip.png"


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


def plot_signal_dip(transitions: List[str], values: np.ndarray, output_path: str) -> None:
    """Plot the cosine similarity drop that marks a topic shift."""

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

    fig, ax = plt.subplots(figsize=(6, 6))
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
    ax.set_xticks(x_positions)
    ax.set_xticklabels(transitions, fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(x_positions[0], x_positions[-1])

    ax.set_xlabel("Sentence Transition Index (k → k+1)", fontsize=12)
    ax.set_ylabel("Cosine Similarity Score", fontsize=12)
    ax.set_title("Signal Dip from Topic Shift in Procedural Text", fontsize=14, pad=12)

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Annotation for cohesive phase among the recipe sentences
    cohesive_idx = 1
    cohesive_value = values[cohesive_idx]
    ax.annotate(
        r"Cohesive Phase (High $c(u,v)$)",
        xy=(cohesive_idx, cohesive_value),
        xytext=(cohesive_idx + 0.6, cohesive_value + 0.18),
        arrowprops=dict(arrowstyle="->", color="#2E8B57", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2E8B57", lw=0.8),
        fontsize=11,
        color="#2E8B57",
    )

    # Highlight the topic boundary where similarity is lowest
    dip_idx = int(np.argmin(values))
    dip_value = float(values[dip_idx])
    ax.annotate(
        "Topic Shift / Boundary",
        xy=(dip_idx, dip_value),
        xytext=(dip_idx + 0.8, dip_value - 0.25),
        arrowprops=dict(arrowstyle="->", color="#B22222", lw=2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B22222", lw=0.8),
        fontsize=11,
        color="#B22222",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Signal dip figure saved to {output_path}")


def main() -> None:
    transitions, similarities = compute_adjacent_similarities(SENTENCES)
    plot_signal_dip(transitions, similarities, OUTPUT_PATH)


if __name__ == "__main__":
    main()
