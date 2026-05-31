import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('./results.csv.xlsx')

# 2. Modern minimalist styling setup
bg_color = '#F8F9FA' # Off-white modern background
fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
ax = fig.add_subplot(111, projection='3d', facecolor=bg_color)

# Remove harsh axis panes/backgrounds
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Remove the default axis borders/spines
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.line.set_color((1.0, 1.0, 1.0, 0.0))

# Soften the grid
ax.grid(color='#DEE2E6', linestyle='-', linewidth=0.5, alpha=0.5)

# 3. Plot the trajectories
# Real Trajectory (Ground Truth) -> Clean dark tone
ax.plot(df['gt_x'], df['gt_y'], df['gt_z'], 
        label='Real Trajectory', color='#212529', linewidth=2, alpha=0.9)

# Estimated Trajectory -> Vibrant modern accent color
ax.plot(df['pos_x'], df['pos_y'], df['agl'], 
        label='Estimated Trajectory', color='#339AF0', linewidth=2, alpha=0.8)

# 4. Typography and Labels
font_options = {'family': 'sans-serif', 'color': '#495057', 'size': 10}
ax.set_xlabel('X Position', fontdict=font_options, labelpad=10)
ax.set_ylabel('Y Position', fontdict=font_options, labelpad=10)
ax.set_zlabel('Altitude', fontdict=font_options, labelpad=10)

# Minimalist Legend (no harsh box frame)
ax.legend(frameon=False, fontsize=11, loc='upper right')

# 5. Render and Save
plt.tight_layout()
plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight', facecolor=bg_color)
plt.show()