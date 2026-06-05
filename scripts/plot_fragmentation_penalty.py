"""Visualize how different fragmentation penalties affect segmentation granularity."""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

PANEL_CONFIG: List[Tuple[str, Sequence[int]]] = [
    (r"Low $\lambda$: Micro-fragmentation", [1, 1, 2, 1, 1, 2, 1, 1]),
    (r"Optimal $\lambda$: Maximized J(S)", [3, 3, 2, 2]),
    (r"High $\lambda$: Loss of Nuance", [6, 4]),
]

OUTPUT_PATH = "assets/fragmentation_penalty.png"

PALETTE = [
    "#002C54",
    "#005F73",
    "#0A9396",
    "#94D2BD",
    "#E9D8A6",
    "#EE9B00",
    "#CA6702",
    "#BB3E03",
]


def cumulative_spans(lengths: Sequence[int]) -> Iterable[Tuple[int, int]]:
    """Yield (start, length) tuples that cover the 10-sentence index range."""

    cursor = 0
    for length in lengths:
        yield cursor, length
        cursor += length


def draw_segmentation(ax: plt.Axes, lengths: Sequence[int], title: str) -> None:
    """Render colored blocks that illustrate chunk boundaries."""

    spans = list(cumulative_spans(lengths))
    ax.broken_barh(
        spans,
        (0, 1),
        facecolors=[PALETTE[i % len(PALETTE)] for i in range(len(spans))],
        edgecolors="white",
        linewidth=2,
    )

    for idx, (start, length) in enumerate(spans):
        if length <= 0:
            continue
        if length == 1:
            label = f"S{start + 1}"
        else:
            label = f"S{start + 1}-S{start + length}"
        ax.text(
            start + length / 2,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="white" if idx % 2 == 0 else "#0f172a",
            fontweight="bold",
        )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, pad=12)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#94a3b8")


def plot_fragmentation_panels(output_path: str = OUTPUT_PATH) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, (title, lengths) in zip(axes, PANEL_CONFIG):
        draw_segmentation(ax, lengths, title)

    axes[-1].set_xticks(np.arange(1, 11, 1))
    axes[-1].set_xticklabels([f"S{i}" for i in range(1, 11)], fontsize=10)
    axes[-1].set_xlabel("Sentence Index", fontsize=12)

    fig.suptitle(
        "Fragmentation Penalty $\\lambda$ and Resulting Segmentations",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Fragmentation comparison saved to {output_path}")


def main() -> None:
    plot_fragmentation_panels()


if __name__ == "__main__":
    main()
