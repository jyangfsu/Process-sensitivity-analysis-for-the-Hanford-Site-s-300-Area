# -*- coding: utf-8 -*-
"""Create a standalone legend for the Hanford site/model-domain figure."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "figures" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

handles = [
    Line2D(
        [], [], linestyle="none", marker="o",
        markerfacecolor="#F6C4D8", markeredgecolor="#F6C4D8",
        markersize=5.5, label="Monitoring Well",
    ),
    Line2D(
        [], [], linestyle="none", marker="^",
        markerfacecolor="#43C5D6", markeredgecolor="#43C5D6",
        markersize=6.0, label="River gauge",
    ),
    Line2D(
        [], [], linestyle="-", color="#FF1E1E",
        linewidth=1.8, label="2D model domain",
    ),
]

fig, ax = plt.subplots(figsize=(2.15, 0.82))
ax.axis("off")
ax.legend(
    handles=handles,
    loc="center",
    frameon=False,
    handlelength=2.2,
    handletextpad=0.75,
    borderaxespad=0.0,
    labelspacing=0.28,
    fontsize=10,
)

output_stem = OUTPUT_DIR / "site_legend"
fig.savefig(
    output_stem.with_suffix(".png"),
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
    facecolor="white",
)
fig.savefig(
    OUTPUT_DIR / "site_legend_transparent.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
    transparent=True,
)
fig.savefig(
    output_stem.with_suffix(".pdf"),
    bbox_inches="tight",
    pad_inches=0.02,
)
fig.savefig(
    output_stem.with_suffix(".svg"),
    bbox_inches="tight",
    pad_inches=0.02,
)
plt.close(fig)
