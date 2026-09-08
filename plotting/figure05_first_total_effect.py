# -*- coding: utf-8 -*-
"""Figure 4: spatial process sensitivities with lower-panel statistics.

This is a complete, standalone plotting script.  It does not execute or import
any earlier plotting script.  The spatial maps use the corrected first-order
and total-effect arrays. Each panel contains a compact distribution inset and
a black median/negative-value annotation positioned in the lower center.

Raw finite values are used for the inset statistics.  Only the spatial color
map is clipped to the theoretical [0, 1] display range.  This distinction
allows small negative estimator residuals to remain quantified in the
companion CSV summary.
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


# -----------------------------------------------------------------------------
# Paths and publication defaults
# -----------------------------------------------------------------------------
BASE_DIR = PROJECT_ROOT / "data" / "processed"
ARCHIVE_DIR = PROJECT_ROOT / "data" / "material_archives"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_STEM = OUTPUT_DIR / "first_and_total_effect_four_processes_v7_cc"
MATERIAL_FILENAME = "T3_Slice_material.h5"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",
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


# -----------------------------------------------------------------------------
# Data and geometry helpers
# -----------------------------------------------------------------------------
def load_array(filename):
    """Load one corrected sensitivity array from the script directory."""
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required sensitivity array not found: {path}")
    values = np.load(path)
    expected_shape = (ntimes, nx * nz)
    if values.shape != expected_shape:
        raise ValueError(
            f"Unexpected shape for {filename}: {values.shape}; "
            f"expected {expected_shape}"
        )
    return values


def read_material_grid(configuration):
    """Read a material grid directly from an archived model realization."""
    archive_path = ARCHIVE_DIR / f"{configuration}.tar"
    member_name = f"{configuration}/{MATERIAL_FILENAME}"

    with tarfile.open(archive_path, "r") as archive:
        member = archive.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(
                f"{member_name!r} was not found in {archive_path}"
            )
        raw_hdf5 = member.read()

    with h5py.File(BytesIO(raw_hdf5), "r") as hdf:
        material = hdf["Materials"]["Material Ids"][:]
    return material.reshape((nz, nx))


def extract_contour_segments(mask):
    """Extract contour vertices without retaining a temporary figure."""
    temp_fig, temp_ax = plt.subplots(figsize=(2, 1))
    contour_set = temp_ax.contour(
        x, z, np.asarray(mask, dtype=float), levels=[0.5]
    )
    segments = [
        np.asarray(segment).copy()
        for segment in contour_set.allsegs[0]
        if len(segment) > 1
    ]
    plt.close(temp_fig)
    return segments


def extract_lower_material5_boundary(material):
    """Extract the configuration-dependent lower branch of material 5."""
    segments = extract_contour_segments(material == 5)
    if not segments:
        raise RuntimeError("No material-5 boundary was found.")

    contour = max(segments, key=len)
    split_index = int(np.argmin(contour[:, 0]))
    branches = [contour[:split_index + 1], contour[split_index:]]
    branches = [branch for branch in branches if len(branch) > 1]
    return min(branches, key=lambda branch: np.mean(branch[:, 1]))


def centers_to_edges(centers):
    """Convert monotonically increasing cell centers to cell edges."""
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def build_background_grid_segments():
    """Build line segments for the nonuniform computational grid."""
    x_edges = centers_to_edges(x)
    z_edges = centers_to_edges(z)
    x_edges = x_edges[(x_edges >= 0.0) & (x_edges <= 142.3)]
    z_edges = z_edges[(z_edges >= 95.0) & (z_edges <= 110.0)]

    segments = [
        [(x_edge, 95.0), (x_edge, 110.0)] for x_edge in x_edges
    ]
    segments.extend([
        [(0.0, z_edge), (142.3, z_edge)] for z_edge in z_edges
    ])
    return segments


# -----------------------------------------------------------------------------
# Distribution inset
# -----------------------------------------------------------------------------
INSET_XLIM = (-0.10, 1.00)
INSET_XTICKS = [0.0, 0.5, 1.0]
MAX_SCATTER_POINTS = 350


def finite_statistics(values):
    """Return finite raw values and the summary statistics shown in the inset."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return finite, np.nan, np.nan, np.nan, np.nan

    median = float(np.median(finite))
    q05, q95 = np.quantile(finite, [0.05, 0.95])
    negative_percent = 100.0 * float(np.mean(finite < 0.0))
    return finite, median, float(q05), float(q95), negative_percent


def deterministic_sample(values, maximum, seed):
    """Select a reproducible subset for the inset scatterplot."""
    if values.size <= maximum:
        return values
    rng = np.random.default_rng(seed)
    indices = rng.choice(values.size, size=maximum, replace=False)
    return values[indices]


def add_distribution_inset(parent_ax, values, seed):
    """Add gray scatter and emphasized summary lines on a white background."""
    finite, median, q05, q95, negative_percent = finite_statistics(values)

    # The inland upper-left part of each map carries little spatial signal and
    # provides a consistent location for the quantitative inset.
    inset = parent_ax.inset_axes([0.045, 0.615, 0.56, 0.205], zorder=20)
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
        inset.scatter(
            sample,
            jitter,
            color="0.48",
            s=3.2,
            linewidths=0.0,
            alpha=0.42,
            rasterized=True,
            zorder=3,
        )

        # Draw the statistical markers above the scatter so that they remain
        # legible even where the sampled values are dense.
        inset.hlines(0.0, q05, q95, color="black", lw=1.25, zorder=6)
        inset.vlines(
            [q05, q95], -0.080, 0.080,
            color="black", lw=0.95, zorder=6,
        )
        inset.vlines(
            median, -0.155, 0.155,
            color="black", lw=1.55, zorder=7,
        )

    return median, negative_percent


# -----------------------------------------------------------------------------
# Load corrected results and static model geometry
# -----------------------------------------------------------------------------
first_order = [
    load_array("ts_SI_CS_cc.npy"),
    load_array("ts_SI_GF_bin_cc.npy"),
    load_array("ts_SI_HT_cc.npy"),
    load_array("ts_SI_RT_bin_cc.npy"),
]
total_effect = [
    load_array("ts_ST_CS_cc.npy"),
    load_array("ts_ST_GF_bin_cc.npy"),
    load_array("ts_ST_HT_cc.npy"),
    load_array("ts_ST_RT_bin_cc.npy"),
]

row_data = first_order + total_effect
row_labels = [
    r"$PS_C$", r"$PS_F$", r"$PS_H$", r"$PS_R$",
    r"$PS_{TC}$", r"$PS_{TF}$", r"$PS_{TH}$", r"$PS_{TR}$",
]
row_letters = list("abcdefgh")

material_grids = [read_material_grid(i) for i in range(1, 6)]
aquifer_boundaries = [
    extract_lower_material5_boundary(material) for material in material_grids
]
reference_lower_segments = extract_contour_segments(material_grids[2] == 4)

river_bottom = np.asarray([
    [83.317430, 110.025249],
    [93.773696, 107.876701],
    [101.866559, 106.874046],
    [107.381165, 106.086245],
    [112.895771, 104.725498],
    [121.275108, 103.364751],
    [131.659755, 101.860767],
    [138.964818, 101.216203],
    [143.190295, 100.786494],
])


# -----------------------------------------------------------------------------
# Main 8 x 3 figure
# -----------------------------------------------------------------------------
cmap = plt.cm.jet
levels = np.linspace(0.0, 1.0, 21)
norm = BoundaryNorm(levels, cmap.N, clip=True)

fig, axes = plt.subplots(
    8,
    3,
    figsize=(10.6, 12.1),
    sharex=True,
    sharey=True,
)

summary_rows = []
mesh = None

for row, (values, process_label) in enumerate(zip(row_data, row_labels)):
    for col in range(3):
        ax = axes[row, col]
        raw_values = values[col, :]
        field = raw_values.reshape((nx, nz)).T
        field = np.ma.masked_invalid(np.clip(field, 0.0, 1.0))

        mesh = ax.contourf(
            xx,
            zz,
            field,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="neither",
        )

        for boundary in aquifer_boundaries:
            ax.plot(
                boundary[:, 0],
                boundary[:, 1],
                color="black",
                lw=0.55,
                zorder=7,
            )

        for segment in reference_lower_segments:
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                color="black",
                lw=0.75,
                zorder=7,
            )

        ax.plot(
            river_bottom[:, 0],
            river_bottom[:, 1],
            color="black",
            lw=0.75,
            zorder=7,
        )

        # Characteristic elevations in the river-stage fluctuation zone.
        ax.plot([105.0, 142.3], [106.962333, 106.962333], "b:", lw=1.05)
        ax.plot([113.0, 142.3], [105.232333, 105.232333], "b:", lw=1.05)
        ax.plot([117.0, 142.3], [104.362333, 104.362333], "b:", lw=1.05)

        median, negative_percent = add_distribution_inset(
            ax, raw_values, seed=100 * row + col
        )

        # Keep the statistical annotation out of the inset. It is centered in
        # the lower part of each map, where it remains easy to compare across
        # rows and time points.
        statistics_label = ax.text(
            0.50,
            0.12,
            f"Med.={median:.3f}; Neg.={negative_percent:.2f}%",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11.0,
            fontweight="normal",
            color="black",
            zorder=24,
        )
        statistics_label.set_path_effects([
            path_effects.Stroke(linewidth=1.6, foreground="white"),
            path_effects.Normal(),
        ])

        finite = raw_values[np.isfinite(raw_values)]
        summary_rows.append({
            "panel": f"{row_letters[row]}{col + 1}",
            "index": process_label.replace("$", ""),
            "time": time_dict[str(col)].replace("Time = ", ""),
            "n_finite": int(finite.size),
            "median": median,
            "negative_percent": negative_percent,
            "q05": float(np.quantile(finite, 0.05)) if finite.size else np.nan,
            "q95": float(np.quantile(finite, 0.95)) if finite.size else np.nan,
            "minimum": float(np.min(finite)) if finite.size else np.nan,
            "maximum": float(np.max(finite)) if finite.size else np.nan,
        })

        ax.set_xlim(0.0, 142.3)
        ax.set_ylim(95.0, 110.0)
        ax.set_xticks([10, 50, 90, 130])
        ax.set_yticks([95, 100, 105])

        if row == 0:
            ax.set_title(
                time_dict[str(col)].replace("Time = ", ""),
                fontsize=13,
                pad=9,
            )

        if col == 0:
            ax.set_ylabel(
                f"{process_label}\nZ-direction (m)",
                fontsize=10.5,
                labelpad=6,
            )

        ax.tick_params(
            axis="both",
            labelsize=8.2,
            pad=1.8,
            labelbottom=(row == 7),
        )

        # Panel identifiers remain outside the inset. Geological unit labels
        # from v3 are intentionally omitted from the first panel.
        ax.text(
            0.880,
            0.960,
            f"({row_letters[row]}{col + 1})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            color="black",
            zorder=22,
            fontweight='bold'
        )


fig.subplots_adjust(
    left=0.083,
    right=0.897,
    bottom=0.078,
    top=0.954,
    wspace=0.045,
    hspace=0.085,
)
fig.supxlabel("X-direction (m)", fontsize=11, y=0.043)

cbar_ax = fig.add_axes([0.918, 0.185, 0.016, 0.635])
cbar = fig.colorbar(
    mesh,
    cax=cbar_ax,
    orientation="vertical",
    boundaries=levels,
    ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    spacing="uniform",
    drawedges=True,
)
cbar.set_ticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
cbar.ax.tick_params(labelsize=10.5, length=2.5, pad=2.5)
cbar.outline.set_linewidth(0.55)
cbar.solids.set_edgecolor("0.45")
cbar.solids.set_linewidth(0.22)
cbar.set_label("Process sensitivity index", fontsize=12, labelpad=6)

# Export the figure in both review and publication formats.
fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)

# Save the exact inset summaries used in the figure for source-data auditing.
summary_path = OUTPUT_STEM.with_name(
    OUTPUT_STEM.name + "_distribution_summary.csv"
)
with summary_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Saved figure files to: {OUTPUT_STEM}")
print(f"Saved distribution summary to: {summary_path}")
