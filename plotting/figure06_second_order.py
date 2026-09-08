# -*- coding: utf-8 -*-
"""Second-order process interactions with spatial maps and distributions.

Complete standalone v5_cc script. It uses corrected interaction arrays and
does not import or execute an earlier plotting script. Each panel combines a
spatial field with a gray distribution sample, an emphasized 5th--95th
percentile interval and median, and a lower-center Med./Neg. annotation.
"""

from io import BytesIO
from pathlib import Path
import csv
import sys
import tarfile

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.colors import BoundaryNorm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from constvars import ntimes, nx, nz, time_dict, x, z, xx, zz


BASE_DIR = PROJECT_ROOT / "data" / "processed"
ARCHIVE_DIR = PROJECT_ROOT / "data" / "material_archives"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_STEM = OUTPUT_DIR / "second_order_interaction_v5_cc"
MATERIAL_FILENAME = "T3_Slice_material.h5"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.linewidth": 0.60,
    "xtick.major.width": 0.60,
    "ytick.major.width": 0.60,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def load_array(filename):
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required interaction array not found: {path}")
    values = np.load(path)
    expected = (ntimes, nx * nz)
    if values.shape != expected:
        raise ValueError(f"{filename}: found {values.shape}, expected {expected}")
    return values


def read_material_grid(configuration):
    archive_path = ARCHIVE_DIR / f"{configuration}.tar"
    member_name = f"{configuration}/{MATERIAL_FILENAME}"
    with tarfile.open(archive_path, "r") as archive:
        member = archive.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(f"{member_name!r} missing from {archive_path}")
        raw = member.read()
    with h5py.File(BytesIO(raw), "r") as hdf:
        material = hdf["Materials"]["Material Ids"][:]
    return material.reshape((nz, nx))


def extract_contour_segments(mask):
    temp_fig, temp_ax = plt.subplots(figsize=(2, 1))
    contour_set = temp_ax.contour(x, z, np.asarray(mask, float), levels=[0.5])
    segments = [
        np.asarray(segment).copy()
        for segment in contour_set.allsegs[0]
        if len(segment) > 1
    ]
    plt.close(temp_fig)
    return segments


def extract_lower_material5_boundary(material):
    segments = extract_contour_segments(material == 5)
    if not segments:
        raise RuntimeError("No material-5 boundary was found.")
    contour = max(segments, key=len)
    split = int(np.argmin(contour[:, 0]))
    branches = [contour[:split + 1], contour[split:]]
    branches = [branch for branch in branches if len(branch) > 1]
    return min(branches, key=lambda branch: np.mean(branch[:, 1]))


def centers_to_edges(centers):
    centers = np.asarray(centers, float)
    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def build_background_grid_segments():
    x_edges = centers_to_edges(x)
    z_edges = centers_to_edges(z)
    x_edges = x_edges[(x_edges >= 0.0) & (x_edges <= 142.3)]
    z_edges = z_edges[(z_edges >= 95.0) & (z_edges <= 110.0)]
    segments = [[(xe, 95.0), (xe, 110.0)] for xe in x_edges]
    segments.extend([[(0.0, ze), (142.3, ze)] for ze in z_edges])
    return segments


INSET_XLIM = (-0.15, 0.50)
INSET_XTICKS = [0.0, 0.25, 0.50]
MAX_SCATTER_POINTS = 350


def finite_statistics(values):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return finite, np.nan, np.nan, np.nan, np.nan
    median = float(np.median(finite))
    q05, q95 = np.quantile(finite, [0.05, 0.95])
    negative = 100.0 * float(np.mean(finite < 0.0))
    return finite, median, float(q05), float(q95), negative


def deterministic_sample(values, maximum, seed):
    if values.size <= maximum:
        return values
    rng = np.random.default_rng(seed)
    return values[rng.choice(values.size, maximum, replace=False)]


def add_distribution_inset(ax, values, seed):
    finite, median, q05, q95, negative = finite_statistics(values)
    inset = ax.inset_axes([0.045, 0.615, 0.56, 0.205], zorder=20)
    inset.set_facecolor((1.0, 1.0, 1.0, 0.92))
    for spine in inset.spines.values():
        spine.set_linewidth(0.42)
        spine.set_color("0.30")
    inset.set_xlim(*INSET_XLIM)
    inset.set_ylim(-0.24, 0.24)
    inset.set_yticks([])
    inset.set_xticks(INSET_XTICKS)
    inset.tick_params(axis="x", labelsize=8.2, length=1.8, pad=1.0)
    inset.axvline(0.0, color="0.35", lw=0.45, ls=":", zorder=1)

    if finite.size:
        sample = deterministic_sample(finite, MAX_SCATTER_POINTS, seed)
        rng = np.random.default_rng(seed + 1000)
        jitter = rng.uniform(-0.105, 0.105, sample.size)
        inset.scatter(sample, jitter, color="0.48", s=3.2,
                      linewidths=0.0, alpha=0.42, rasterized=True, zorder=3)
        inset.hlines(0.0, q05, q95, color="black", lw=1.25, zorder=6)
        inset.vlines([q05, q95], -0.080, 0.080, color="black", lw=0.95,
                     zorder=6)
        inset.vlines(median, -0.155, 0.155, color="black", lw=1.55,
                     zorder=7)

    return finite, median, q05, q95, negative


interaction_data = [
    load_array("ts_Interact_CF_cc.npy"),
    load_array("ts_Interact_CH_cc.npy"),
    load_array("ts_Interact_CR_cc.npy"),
    load_array("ts_Interact_HF_cc.npy"),
    load_array("ts_Interact_RF_cc.npy"),
    load_array("ts_Interact_RH_cc.npy"),
]
interaction_labels = [
    r"$PS_{CF}$", r"$PS_{CH}$", r"$PS_{CR}$",
    r"$PS_{FH}$", r"$PS_{FR}$", r"$PS_{HR}$",
]
row_letters = list("abcdef")

material_grids = [read_material_grid(i) for i in range(1, 6)]
alluvial_boundaries = [
    extract_lower_material5_boundary(material) for material in material_grids
]
reference_lower_segments = extract_contour_segments(material_grids[2] == 4)
river_bottom = np.asarray([
    [83.317430, 110.025249], [93.773696, 107.876701],
    [101.866559, 106.874046], [107.381165, 106.086245],
    [112.895771, 104.725498], [121.275108, 103.364751],
    [131.659755, 101.860767], [138.964818, 101.216203],
    [143.190295, 100.786494],
])

cmap = plt.cm.jet
levels = np.linspace(0.0, 1.0, 21)
norm = BoundaryNorm(levels, cmap.N, clip=True)
fig, axes = plt.subplots(6, 3, figsize=(10.6, 9.5), sharex=True, sharey=True)

summary_rows = []
mesh = None
for row, (values, label_text) in enumerate(zip(interaction_data,
                                                interaction_labels)):
    for col in range(ntimes):
        ax = axes[row, col]
        raw_values = values[col]
        field = np.ma.masked_invalid(
            np.clip(raw_values.reshape((nx, nz)).T, 0.0, 1.0)
        )
        mesh = ax.contourf(xx, zz, field, levels=levels, cmap=cmap,
                           norm=norm, extend="neither")
        for boundary in alluvial_boundaries:
            ax.plot(boundary[:, 0], boundary[:, 1], color="black", lw=0.55,
                    zorder=7)
        for segment in reference_lower_segments:
            ax.plot(segment[:, 0], segment[:, 1], color="black", lw=0.75,
                    zorder=7)
        ax.plot(river_bottom[:, 0], river_bottom[:, 1], color="black",
                lw=0.75, zorder=7)
        ax.plot([105, 142.3], [106.962333, 106.962333], "b:", lw=1.05)
        ax.plot([113, 142.3], [105.232333, 105.232333], "b:", lw=1.05)
        ax.plot([117, 142.3], [104.362333, 104.362333], "b:", lw=1.05)

        finite, median, q05, q95, negative = add_distribution_inset(
            ax, raw_values, seed=100 * row + col,
        )
        statistics_label = ax.text(
            0.50, 0.12,
            f"Med.={median:.3f}; Neg.={negative:.2f}%",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11.0, fontweight="normal", color="black", zorder=24,
        )
        statistics_label.set_path_effects([
            path_effects.Stroke(linewidth=1.6, foreground="white"),
            path_effects.Normal(),
        ])
        summary_rows.append({
            "panel": f"{row_letters[row]}{col + 1}",
            "index": label_text.replace("$", ""),
            "time": time_dict[str(col)].replace("Time = ", ""),
            "n_finite": int(finite.size), "median": median,
            "negative_percent": negative, "q05": q05, "q95": q95,
            "minimum": float(np.min(finite)) if finite.size else np.nan,
            "maximum": float(np.max(finite)) if finite.size else np.nan,
        })

        ax.set_xlim(0.0, 142.3)
        ax.set_ylim(95.0, 110.0)
        ax.set_xticks([10, 50, 90, 130])
        ax.set_yticks([95, 100, 105])
        ax.tick_params(axis="both", labelsize=8.2, pad=1.8,
                       labelbottom=(row == 5))
        if row == 0:
            ax.set_title(time_dict[str(col)].replace("Time = ", ""),
                         fontsize=13, pad=9)
        if col == 0:
            ax.set_ylabel(f"{label_text}\nZ-direction (m)", fontsize=10.5,
                          labelpad=6)
        ax.text(0.880, 0.960, f"({row_letters[row]}{col + 1})",
                transform=ax.transAxes, ha="left", va="top", fontsize=12,
                color="black", zorder=22, fontweight="bold")

fig.subplots_adjust(left=0.095, right=0.897, bottom=0.092, top=0.944,
                    wspace=0.045, hspace=0.085)
fig.supxlabel("X-direction (m)", fontsize=11, y=0.047)

cbar_ax = fig.add_axes([0.918, 0.19, 0.016, 0.64])
cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="vertical",
                    boundaries=levels, ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
                    spacing="uniform", drawedges=True)
cbar.set_ticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
cbar.ax.tick_params(labelsize=10.5, length=2.5, pad=2.5)
cbar.outline.set_linewidth(0.55)
cbar.solids.set_edgecolor("0.45")
cbar.solids.set_linewidth(0.22)
cbar.set_label("Second-order process sensitivity index", fontsize=12,
               labelpad=6)

fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)

summary_path = OUTPUT_STEM.with_name(OUTPUT_STEM.name +
                                     "_distribution_summary.csv")
with summary_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Saved figure files to: {OUTPUT_STEM}")
print(f"Saved distribution summary to: {summary_path}")
