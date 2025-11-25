import matplotlib.pyplot as plt
import numpy as np

# Data from your reval results (3M SOP)
strategies = ['P0 (Zero-Shot)', 'P1 (Few-Shot)', 'P2 (CoT)', 'P3 (Two-Stage)']
step_f1 = [0.553, 0.603, 0.0, 0.619]
constraint_cov = [0.0, 0.0, 0.0, 0.75]
a_score = [0.317, 0.324, 0.0, 0.699]

x = np.arange(len(strategies))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, step_f1, width, label='Step F1', color='#a6cee3')
rects2 = ax.bar(x, a_score, width, label='Overall A-Score', color='#1f78b4')
rects3 = ax.bar(x + width, constraint_cov, width, label='Constraint Coverage', color='#e31a1c')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Score (0.0 - 1.0)')
ax.set_title('Impact of Prompting Strategy on Extraction Quality (3M SOP)')
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.legend(loc='upper left')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add labels on top of P3 bars to highlight success
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('prompting_comparison_chart.png', dpi=300)
print("Chart saved as prompting_comparison_chart.png")
