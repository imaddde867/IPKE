import matplotlib.pyplot as plt
import numpy as np


def main():
    # Hard-coded metrics from macro-tier-a-comparison-prompting.md
    strategies = [
        "P0 (Baseline)",
        "P1 (Few-Shot)",
        "P2 (CoT)",
        "P3 (Two-Stage)",
    ]
    step_f1 = [0.305, 0.313, 0.0, 0.377]
    a_score = [0.253, 0.407, 0.0, 0.611]
    constraint_cov = [0.033, 0.342, 0.0, 0.708]

    x = np.arange(len(strategies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, step_f1, width, label="Step F1", color="#a6cee3")
    rects2 = ax.bar(x, a_score, width, label="Overall A-Score", color="#1f78b4")
    rects3 = ax.bar(x + width, constraint_cov, width, label="Constraint Coverage", color="#e31a1c")

    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_title("Impact of Prompting Strategy on Extraction Quality (Macro Tier A)")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig("prompting_comparison_chart.png", dpi=300)
    print("Chart saved as prompting_comparison_chart.png")


if __name__ == "__main__":
    main()
