from pathlib import Path as FilePath

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.path import Path

plt.rcParams['font.family'] = 'Arial'

# Define spatial domain boundaries
west = 0
east = 143.2
south = 0
north = 1
top = 110.0
bottom = 90.0

# Define grid spacings
dx = np.concatenate((0.1 * 1.095415574 ** np.arange(48, 0, -1), np.full(532, 0.1)))
dy = np.array([1.0])
dz = np.concatenate((0.05 * 1.09505408 ** np.arange(25, 0, -1), np.full(300, 0.05)))

# Compute grid dimensions
nx = len(dx)
ny = len(dy)
nz = len(dz)

# Compute cell edges coordinates
x = west + np.cumsum(dx)
y = south + np.cumsum(dy)
z = bottom + np.cumsum(dz)
x = np.insert(x, 0, 0)
y = np.insert(y, 0, 0)
z = np.insert(z, 0, 0)

# Compute cell center coordinates
#x = west + np.cumsum(dx) - 0.5 * dx
#y = south + np.cumsum(dy) - 0.5 * dy
#z = bottom + np.cumsum(dz) - 0.5 * dz

# Create a meshgrid for visualization (x-z plane, as y has only one layer)
X, Z = np.meshgrid(x, z)


# Upper bound of Ringold
data1 = np.array([
    [0.091684, 96.808883],
    [9.832057, 96.808883],
    [16.045743, 97.060789],
    [55.427077, 98.572226],
    [82.129133, 99.411913],
    [93.716818, 99.663820],
    [99.426692, 99.747788],
    [105.472440, 99.831757],
    [111.266283, 99.831757],
    [118.487593, 99.747788],
    [123.105873, 99.747788],
    [128.710786, 99.243976],
    [134.798519, 98.719171],
    [139.101916, 98.614210],
    [143.195392, 98.614210]
])

# Bottom of Alluvium
data2 = np.array([
    [83.177412, 109.977453],
    [88.233089, 107.899778],
    [95.574209, 105.683591],
    [103.192353, 103.398148],
    [112.334125, 101.528240],
    [121.268130, 100.212378],
    [128.710786, 99.243976],
    [134.798519, 98.719171],
    [139.101916, 98.614210],
    [143.195392, 98.614210]
])

# Bottom of River
data3 = np.array([
    [83.317430, 110.025249],
    [93.773696, 107.876701],
    [101.866559, 106.874046],
    [107.381165, 106.086245],
    [112.895771, 104.725498],
    [121.275108, 103.364751],
    [131.659755, 101.860767],
    [138.964818, 101.216203],
    [143.190295, 100.786494]
])

# Axis limits
x_min, x_max = 0, 142.3
y_min, y_max = 90, 110

# Create figure and axis
fig, ax = plt.subplots(figsize=(9.23, 2.5))


for xi in x:
    ax.plot([xi, xi], [z[0], z[-1]], color='gray', linewidth=0.2)
for zi in z:
    ax.plot([x[0], x[-1]], [zi, zi], color='gray', linewidth=0.2)




# Region 1: Below data1
region1 = Polygon(np.vstack(([[x_min, y_min]], data1, [[x_max, y_min]])), closed=True)

# Region 2: Between y = 110 and data1, bounded left by x = 0 and right by data2
region2 = Polygon(np.vstack(([[x_min, y_max]], data2, data1[::-1], [[x_min, y_min]])), closed=True)

# Region 3: Between data2 and data3
region3 = Polygon(np.vstack((data2, data3[::-1])), closed=True)

# Add regions 1–3 using solid colors
patches = [region1, region2, region3]
colors = ['#FBCBD6', '#56FDFD', '#FAFB2D']
collection = PatchCollection(patches, facecolors=colors, edgecolor='none', alpha=0.9)
ax.add_collection(collection)

# --- Region 4: Gradient between data3 and y = 106.962333, x ≥ 100 ---
data3_clip = data3[data3[:, 0] >= 100]
region4_coords = np.vstack((
    data3_clip,
    [[data3_clip[-1, 0], 106.962333]],
    [[data3_clip[0, 0], 106.962333]]
))
region4_path = Path(region4_coords)
region4_patch = PathPatch(region4_path, transform=ax.transData)

# Create vertical gradient image
gradient = np.linspace(0, 1, 60).reshape(-1, 1)
gradient = np.repeat(gradient, 500, axis=1)

# Display gradient image and clip it to Region 4
x_start = data3_clip[0, 0]
x_end = data3_clip[-1, 0]
y_top = 106.962333
y_bottom = data3_clip[:, 1].min()

img = ax.imshow(
    gradient,
    extent=(x_start, x_end, y_bottom, y_top),
    origin='lower',
    cmap='Blues_r',
    alpha=1.0,
    aspect='auto',
    
)


img.set_clip_path(region4_patch)


plt.plot([105, 142.3], [106.962333, 106.96233], 'b:')

plt.plot([113, 142.3], [105.232333, 105.232333], 'b:')

plt.plot([117, 142.3], [104.362333, 104.362333], 'b:')

# Annotation
y_top = 106.962333
y_bottom = 104.362333
x_arrow = 138
# Vertical double-headed arrow
ax.annotate(
    "",
    xy=(x_arrow, y_top+0.5),
    xytext=(x_arrow, y_bottom-0.5),
    arrowprops=dict(
        arrowstyle="<->",
        color="b",
        linewidth=1.4,
        mutation_scale=11,
    ),
    zorder=10,
)

'''
# Annotation
ax.text(
    x_arrow - 1.5,
    (y_top + y_bottom) / 2,
    "Fluctuation zone",
    color="k",
    fontsize=12,
    rotation=00,
    ha="right",
    va="center",
    zorder=10,
    #weight='bold'
)
'''

# Plot dashed lines for interfaces
ax.plot(data1[:, 0], data1[:, 1], color='black', linestyle='--', linewidth=1.5)
#ax.plot(data2[:-4, 0], data2[:-4, 1], color='black', linestyle='--', linewidth=1.5)
ax.plot(data3[:, 0], data3[:, 1], color='black', linestyle='--', linewidth=1.5)

# Add text labels
ax.text(50, 105, 'Hanford', ha='center', va='center', fontsize=16, weight='bold', color='w')
ax.text(70, 94, 'Ringold', ha='center', va='center', fontsize=16, weight='bold', color='w')
ax.text(113, 103, 'Alluvium', ha='center', va='center', fontsize=16, weight='bold', color='w')
ax.text(137, 102.5, 'River', ha='center', va='center', fontsize=16, weight='bold', color='w')





# Axis and display settings
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_ylabel('Z-direction (m)', fontname='Arial', fontsize=14)
ax.set_xlabel('X-direction (m)', fontname='Arial', fontsize=14)

ax.set_xticks([0, 20, 40, 60, 80, 100, 120, 140])
ax.set_yticks([90, 100, 110])
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

ax.grid(False)

plt.tight_layout()

project_root = FilePath(__file__).resolve().parents[1]
output_dir = project_root / "figures" / "generated"
output_dir.mkdir(parents=True, exist_ok=True)
output_stem = output_dir / "figure03_model_cross_section"
fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)
