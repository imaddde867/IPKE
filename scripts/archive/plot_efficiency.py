import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Mistral-7B (P0)', 'Llama-3-8B (P0)', 'Llama-3-70B (P0)', 'Mistral-7B (P3)']
# A-Scores from your data
performance = [0.317, 0.174, 0.187, 0.699] 
# Approximate VRAM usage (GB) for inference
compute_cost = [14, 16, 40, 14] 

plt.figure(figsize=(10, 6))
plt.scatter(compute_cost, performance, s=200, c=['red', 'red', 'red', 'green'], alpha=0.7, edgecolors='black')

# Annotate points
for i, txt in enumerate(models):
    plt.annotate(txt, (compute_cost[i]+1, performance[i]), fontsize=10, weight='bold')

plt.title('The Efficiency Frontier: Algorithmic Strategy vs. Compute Scale', fontsize=14)
plt.xlabel('Computational Cost (VRAM GB)', fontsize=12)
plt.ylabel('Extraction Quality (A-Score)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Draw an arrow showing the "Elite" jump
plt.annotate('', xy=(14, 0.699), xytext=(14, 0.317),
             arrowprops=dict(facecolor='green', shrink=0.05))
plt.text(15, 0.5, 'Effect of P3 Strategy\n(+120% Performance)', color='green', fontsize=11)

plt.savefig('efficiency_frontier.png', dpi=300)
print("Saved efficiency_frontier.png")
