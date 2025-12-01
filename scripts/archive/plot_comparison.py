import matplotlib.pyplot as plt

# Models and strategies
models = [
    "Mistral-7B (P0)",
    "Llama-3-8B (P0)",
    "Llama-3-70B (P0)",
    "Mistral-7B (P3)",
    "Llama-3-8B (P3)",
    "Llama-3-70B (P3)",
]

# Procedural Fidelity Score Φ (per model/strategy) – from thesis results
phi = [
    0.253,  # Mistral-7B P0
    0.174,  # Llama-3-8B P0
    0.187,  # Llama-3-70B P0
    0.699,  # Mistral-7B P3
    0.376,  # Llama-3-8B P3
    0.439,  # Llama-3-70B P3
]

# Approx VRAM/compute cost (GB)
vram = [
    14,  # Mistral-7B
    18,  # Llama-3-8B
    40,  # Llama-3-70B
    14,  # Mistral-7B (same model, different prompt)
    18,  # Llama-3-8B (same)
    40,  # Llama-3-70B (same)
]

# Color by strategy: red = P0, green = P3
colors = ["red", "red", "red", "green", "green", "green"]

plt.figure(figsize=(10, 6))
plt.scatter(vram, phi, s=180, c=colors, alpha=0.8, edgecolors="black")

# Annotate each point
for x, y, label in zip(vram, phi, models):
    plt.annotate(label, (x + 0.7, y), fontsize=9, weight="bold")

plt.title("Efficiency Frontier: Model Scale vs. Procedural Fidelity Φ", fontsize=14)
plt.xlabel("Approximate VRAM for Inference (GB)", fontsize=12)
plt.ylabel("Procedural Fidelity Score Φ", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)

# Arrow highlighting the "smart jump" from Mistral-7B P0 → P3
plt.annotate(
    "",
    xy=(14, 0.699),
    xytext=(14, 0.253),
    arrowprops=dict(arrowstyle="->", linewidth=2),
)
plt.text(
    15,
    0.50,
    "Effect of P3 strategy\n(+~0.45 Φ on same 7B model)",
    color="green",
    fontsize=10,
)

# Manual legend for strategies
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", label="P0 (Zero-shot)",
           markerfacecolor="red", markeredgecolor="black", markersize=10),
    Line2D([0], [0], marker="o", color="w", label="P3 (Two-Stage)",
           markerfacecolor="green", markeredgecolor="black", markersize=10),
]
plt.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
plt.savefig("efficiency_frontier_phi.png", dpi=300)
print("Saved efficiency_frontier_phi.png")
