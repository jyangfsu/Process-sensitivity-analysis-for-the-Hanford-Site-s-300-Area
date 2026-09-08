# -*- coding: utf-8 -*-
"""Plot dominant and least-influential processes using corrected indices.

Rows represent simulation times of 8, 10, and 12 weeks. The left column
shows the process with the highest first-order process sensitivity index,
whereas the right column shows the process with the lowest total-effect
process sensitivity index. Each spatial panel includes:

1. the spatial process classification;
2. the selected process and index value at P1-P7; and
3. the evaluated physical-area proportion assigned to each process.

Version v13: area-percentage labels use two decimal places.
All calculations and other visual settings are unchanged from v12.

Corrections relative to v11
---------------------------
1. Read all eight corrected ``*_cc.npy`` first-order and total-effect arrays.
2. Recompute process classifications rather than reusing hard-coded results.
3. Compute proportions with physical cell areas on the nonuniform grid instead
   of unweighted grid-cell counts.
4. Remove silent clipping of P1-P7 sensitivity values.
5. Exclude a cell from ranking if any compared index is nonfinite or outside
   its theoretical [0, 1] range. This prevents residual negative estimates from
   being mislabeled as evidence that flow is the least-influential process.
6. Export the plotted classifications and point values as a source-data NPZ.
"""

from io import BytesIO
from pathlib import Path
import sys
import tarfile

import h5py
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from constvars import ntimes, nx, nz, x, z, xx, zz


# -----------------------------------------------------------------------------
# Paths and visual settings
# -----------------------------------------------------------------------------

BASE_DIR = PROJECT_ROOT / "data" / "processed"
ARCHIVE_DIR = PROJECT_ROOT / "data" / "material_archives"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATERIAL_FILENAME = "T3_Slice_material.h5"
TIME_LABELS = ["8 weeks", "10 weeks", "12 weeks"]
PROCESS_NAMES = ["Climate", "Flow", "Heat", "Reaction"]
PROCESS_ABBREVIATIONS = ["C", "F", "H", "R"]
PROCESS_COLORS = ["#5B8DB8", "#4FAF8B", "#E69F4C", "#A879B2"]

PROCESS_CMAP = ListedColormap(PROCESS_COLORS)
PROCESS_NORM = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), 4)

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# -----------------------------------------------------------------------------
# Sensitivity-index data
# -----------------------------------------------------------------------------

def load_array(filename):
    """Load one sensitivity-index array from the script directory."""
    return np.load(BASE_DIR / filename)


FIRST_ORDER = [
    load_array("ts_SI_CS_cc.npy"),
    load_array("ts_SI_GF_bin_cc.npy"),
    load_array("ts_SI_HT_cc.npy"),
    load_array("ts_SI_RT_bin_cc.npy"),
]

TOTAL_EFFECT = [
    load_array("ts_ST_CS_cc.npy"),
    load_array("ts_ST_GF_bin_cc.npy"),
    load_array("ts_ST_HT_cc.npy"),
    load_array("ts_ST_RT_bin_cc.npy"),
]


def classify_processes(arrays, selection):
    """Classify each valid cell by its maximum or minimum process index."""
    stack = np.stack(arrays, axis=0)
    result = np.full((ntimes, stack.shape[2]), np.nan)

    for time_index in range(ntimes):
        values = stack[:, time_index, :]
        valid = (
            np.all(np.isfinite(values), axis=0)
            & np.all(values >= 0.0, axis=0)
            & np.all(values <= 1.0, axis=0)
        )
        if selection == "maximum":
            result[time_index, valid] = np.argmax(
                values[:, valid],
                axis=0,
            )
        elif selection == "minimum":
            result[time_index, valid] = np.argmin(
                values[:, valid],
                axis=0,
            )
        else:
            raise ValueError("selection must be 'maximum' or 'minimum'")

    return result


HIGHEST_FIRST = classify_processes(FIRST_ORDER, "maximum")
LOWEST_TOTAL = classify_processes(TOTAL_EFFECT, "minimum")
MAP_FIELDS = [HIGHEST_FIRST, LOWEST_TOTAL]

# Grid indices corresponding to P1-P7. P5 and P6 are located exactly 1 and
# 2 m below P4 at the same x coordinate. The nominal location 5 m below P4
# lies outside the active model domain, so P7 uses the closest vertically
# aligned valid cell, approximately 4.55 m below P4.
POINT_GRID_INDICES = [
    (220, 243),
    (255, 229),
    (280, 219),
    (301, 211),
    (301, 191),
    (301, 171),
    (301, 120),
]

# Derive plotted coordinates from the same grid indices used for extraction.
POINT_COORDINATES = np.asarray([
    [x[ix], z[iz]]
    for ix, iz in POINT_GRID_INDICES
])


def extract_point_values(arrays):
    """Extract the four process indices at P1-P7 for each time."""
    values = np.empty((ntimes, 7, 4), dtype=float)

    for time_index in range(ntimes):
        for process_index, array in enumerate(arrays):
            field = array[time_index].reshape((nx, nz))
            for point_index, (ix, iz) in enumerate(POINT_GRID_INDICES):
                values[time_index, point_index, process_index] = field[ix, iz]

    return values


FIRST_POINT_VALUES = extract_point_values(FIRST_ORDER)
TOTAL_POINT_VALUES = extract_point_values(TOTAL_EFFECT)


def select_point_processes(point_values, selection):
    """Select the process and corresponding value at every time and point."""
    candidate_values = point_values.copy()
    invalid = (
        ~np.isfinite(candidate_values)
        | (candidate_values < 0.0)
        | (candidate_values > 1.0)
    )
    candidate_values[invalid] = np.nan
    if np.any(np.all(np.isnan(candidate_values), axis=2)):
        raise ValueError("At least one P1-P7 location has no valid process index.")

    if selection == "maximum":
        process_ids = np.nanargmax(candidate_values, axis=2)
    elif selection == "minimum":
        process_ids = np.nanargmin(candidate_values, axis=2)
    else:
        raise ValueError("selection must be 'maximum' or 'minimum'")

    selected_values = np.take_along_axis(
        candidate_values,
        process_ids[:, :, np.newaxis],
        axis=2,
    )[:, :, 0]
    return process_ids, selected_values


POINT_RESULTS = [
    select_point_processes(FIRST_POINT_VALUES, "maximum"),
    select_point_processes(TOTAL_POINT_VALUES, "minimum"),
]

# Physical area of every nonuniform grid cell. The flattening order matches the
# sensitivity arrays, which are reshaped as (nx, nz) throughout this script.
def cell_widths(centers):
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.diff(edges)


CELL_AREAS = (cell_widths(x)[:, np.newaxis] * cell_widths(z)[np.newaxis, :]).ravel()


def compute_area_percent(classification):
    """Area-weight each mutually exclusive process classification."""
    percentages = np.zeros((ntimes, 4), dtype=float)
    for time_index in range(ntimes):
        classes = classification[time_index]
        valid = np.isfinite(classes)
        total_area = np.sum(CELL_AREAS[valid])
        if total_area <= 0.0:
            percentages[time_index, :] = np.nan
            continue
        for process_id in range(4):
            percentages[time_index, process_id] = (
                100.0
                * np.sum(CELL_AREAS[valid & (classes == process_id)])
                / total_area
            )
    return percentages


DOMINANT_AREA_PERCENT = compute_area_percent(HIGHEST_FIRST)
INFERIOR_AREA_PERCENT = compute_area_percent(LOWEST_TOTAL)

AREA_PERCENT_SETS = [DOMINANT_AREA_PERCENT, INFERIOR_AREA_PERCENT]


# -----------------------------------------------------------------------------
# Material boundaries and background grid
# -----------------------------------------------------------------------------

def read_material_grid(configuration):
    """Read one material grid directly from its archived HDF5 file."""
    archive_path = ARCHIVE_DIR / f"{configuration}.tar"
    member_name = f"{configuration}/{MATERIAL_FILENAME}"

    with tarfile.open(archive_path, "r") as archive:
        member = archive.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(
                f"{member_name!r} was not found in {archive_path}"
            )
        raw_hdf5 = member.read()

    with h5py.File(BytesIO(raw_hdf5), "r") as hdf_file:
        material = hdf_file["Materials"]["Material Ids"][:]

    return material.reshape((nz, nx))


def extract_contour_segments(mask):
    """Extract contour vertices without retaining a temporary figure."""
    temp_fig, temp_ax = plt.subplots()
    contour_set = temp_ax.contour(
        x,
        z,
        np.asarray(mask, dtype=float),
        levels=[0.5],
    )
    segments = [
        np.asarray(segment).copy()
        for segment in contour_set.allsegs[0]
        if len(segment) > 1
    ]
    plt.close(temp_fig)
    return segments


def extract_lower_material5_boundary(material):
    """Extract the lower alluvium boundary from one configuration."""
    segments = extract_contour_segments(material == 5)
    if not segments:
        raise RuntimeError("No material-5 boundary was found.")

    contour = max(segments, key=len)
    split_index = int(np.argmin(contour[:, 0]))
    branches = [contour[:split_index + 1], contour[split_index:]]
    branches = [branch for branch in branches if len(branch) > 1]
    return min(branches, key=lambda branch: np.mean(branch[:, 1]))


def centers_to_edges(centers):
    """Convert one-dimensional cell centers to cell edges."""
    centers = np.asarray(centers)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def build_background_grid_segments():
    """Construct lightweight line segments for the numerical grid."""
    x_edges = centers_to_edges(x)
    z_edges = centers_to_edges(z)
    x_edges = x_edges[(x_edges >= 0.0) & (x_edges <= 142.3)]
    z_edges = z_edges[(z_edges >= 95.0) & (z_edges <= 110.0)]

    segments = [
        [(x_edge, 95.0), (x_edge, 110.0)]
        for x_edge in x_edges
    ]
    segments.extend([
        [(0.0, z_edge), (142.3, z_edge)]
        for z_edge in z_edges
    ])
    return segments


BACKGROUND_GRID = build_background_grid_segments()
MATERIAL_GRIDS = [read_material_grid(i) for i in range(1, 6)]
ALLUVIAL_BOUNDARIES = [
    extract_lower_material5_boundary(material)
    for material in MATERIAL_GRIDS
]
REFERENCE_LOWER_SEGMENTS = extract_contour_segments(
    MATERIAL_GRIDS[2] == 4
)

RIVER_BOTTOM = np.asarray([
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
# Panel-drawing functions
# -----------------------------------------------------------------------------

def draw_domain_boundaries(ax):
    """Add grid lines, five alluvium boundaries, and model-domain edges."""
    ax.add_collection(LineCollection(
        BACKGROUND_GRID,
        colors="0.45",
        linewidths=0.10,
        alpha=0.24,
        zorder=5,
        rasterized=True,
    ))

    for boundary in ALLUVIAL_BOUNDARIES:
        ax.plot(
            boundary[:, 0],
            boundary[:, 1],
            color="black",
            linewidth=0.55,
            zorder=7,
        )

    for segment in REFERENCE_LOWER_SEGMENTS:
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color="black",
            linewidth=0.75,
            zorder=7,
        )

    ax.plot(
        RIVER_BOTTOM[:, 0],
        RIVER_BOTTOM[:, 1],
        color="black",
        linewidth=0.75,
        zorder=7,
    )

    for x_start, elevation in [
        (105, 106.962333),
        (113, 105.232333),
        (117, 104.362333),
    ]:
        ax.plot(
            [x_start, 142.3],
            [elevation, elevation],
            color="black",
            linestyle=":",
            linewidth=0.75,
            zorder=8,
        )


def draw_spatial_map(ax, field):
    """Draw one spatial process-classification map."""
    plot_field = field.reshape((nx, nz)).T
    plot_field = np.ma.masked_invalid(plot_field)

    ax.pcolormesh(
        xx,
        zz,
        plot_field,
        cmap=PROCESS_CMAP,
        norm=PROCESS_NORM,
        shading="auto",
        rasterized=True,
    )
    draw_domain_boundaries(ax)

    for point_x, point_z in POINT_COORDINATES:
        ax.scatter(
            point_x,
            point_z,
            marker="s",
            s=17.5,
            facecolor="white",
            edgecolor="black",
            linewidth=0.25,
            zorder=25,
        )

    ax.set_xlim(0, 142.3)
    ax.set_ylim(95, 110)
    ax.set_xticks([10, 50, 90, 130])
    ax.set_yticks([95, 100, 105])
    ax.set_yticks([95, 100, 105], ['95' ,'100', '105'], fontsize=7)


def draw_point_strip(ax, process_ids, values):
    """Draw the P1-P7 classifications and the downward ordering arrow."""
    cell_width = 0.72
    ax.set_xlim(0, 1)
    ax.set_ylim(7, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    for point_index in range(7):
        process_id = int(process_ids[point_index])
        value = float(values[point_index])

        ax.add_patch(Rectangle(
            (0, point_index),
            cell_width,
            1,
            facecolor=PROCESS_COLORS[process_id],
            edgecolor="white",
            linewidth=0.70,
        ))
        ax.text(
            cell_width / 2.0,
            point_index + 0.50,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8.0,
        )

    ax.add_patch(Rectangle(
        (0, 0),
        cell_width,
        7,
        facecolor="none",
        edgecolor="0.20",
        linewidth=0.70,
        zorder=5,
    ))
    ax.annotate(
        "",
        xy=(0.88, 6.35),
        xytext=(0.88, 0.65),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "0.20",
            "linewidth": 0.85,
            "mutation_scale": 8,
        },
    )
    ax.text(0.88, 0.36, "  P1", ha="center", va="center", fontsize=8)
    ax.text(0.88, 6.66, "  P7", ha="center", va="center", fontsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_area_proportion(ax, percentages, show_ylabel=False):
    """Draw the mutually exclusive process-classification area proportions."""
    left = 0.0

    for process_id, percentage in enumerate(percentages):
        ax.barh(
            0,
            percentage,
            left=left,
            height=0.58,
            color=PROCESS_COLORS[process_id],
            edgecolor="white",
            linewidth=0.50,
        )
        if percentage >= 5.0:
            ax.text(
                np.clip(left + percentage / 2.0, 7.5, 92.5),
                0,
                f"{percentage:.2f}%",
                ha="center",
                va="center",
                fontsize=8,
            )
        left += percentage

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.55)

    if show_ylabel:
        ax.set_ylabel(
            "Evaluated\narea (%)",
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=7,
            linespacing=1.05,
        )


# -----------------------------------------------------------------------------
# Figure assembly
# -----------------------------------------------------------------------------

COLUMN_TITLES = [
    r"Dominant process: highest $PS_K$",
    r"Least-influential process: lowest $PS_{TK}$",
]
COLUMN_LETTERS = ["a", "b"]

fig = plt.figure(figsize=(7.4, 5.4))
outer = fig.add_gridspec(
    3,
    2,
    left=0.135,
    right=0.970,
    bottom=0.115,
    top=0.925,
    wspace=0.145,
    hspace=0.265,
)

map_axes = np.empty((3, 2), dtype=object)
area_axes = np.empty((3, 2), dtype=object)

for time_index in range(3):
    for column_index in range(2):
        process_ids, selected_values = POINT_RESULTS[column_index]
        cell = outer[time_index, column_index].subgridspec(
            2,
            2,
            width_ratios=[0.865, 0.135],
            height_ratios=[0.145, 0.855],
            wspace=0.018,
            hspace=0.0,
        )

        area_ax = fig.add_subplot(cell[0, 0])
        area_axes[time_index, column_index] = area_ax
        draw_area_proportion(
            area_ax,
            AREA_PERCENT_SETS[column_index][time_index],
            show_ylabel=(column_index == 0),
        )

        if time_index == 0:
            area_ax.set_title(
                COLUMN_TITLES[column_index],
                fontsize=11.2,
                pad=10,
            )

        blank_ax = fig.add_subplot(cell[0, 1])
        blank_ax.axis("off")

        shared_axis = (
            map_axes[0, 0]
            if (time_index, column_index) != (0, 0)
            else None
        )
        map_ax = fig.add_subplot(
            cell[1, 0],
            sharex=shared_axis,
            sharey=shared_axis,
        )
        map_axes[time_index, column_index] = map_ax
        draw_spatial_map(
            map_ax,
            MAP_FIELDS[column_index][time_index],
        )
        map_ax.tick_params(
            axis="both",
            labelsize=8,
            pad=2,
            labelbottom=True,
        )
        map_ax.set_xticks([10, 50, 90, 130])
        map_ax.set_xticklabels(["10", "50", "90", "130"], fontsize=8)
        plt.setp(map_ax.get_xticklabels(), visible=True)
        map_ax.set_xlabel("X-direction (m)", fontsize=8.5, labelpad=2)

        if column_index == 0:
            map_ax.set_ylabel(
                "Z-direction (m)",
                fontsize=9,
                labelpad=7,
            )
        else:
            map_ax.tick_params(labelleft=False)

        map_ax.text(
            0.025,
            0.955,
            f"({COLUMN_LETTERS[column_index]}{time_index + 1})",
            transform=map_ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            fontweight="bold",
            zorder=30,
        )

        point_ax = fig.add_subplot(cell[1, 1])
        draw_point_strip(
            point_ax,
            process_ids[time_index],
            selected_values[time_index],
        )


legend_handles = [
    Patch(
        facecolor=color,
        edgecolor="black",
        linewidth=0.55,
        label=name,
    )
    for color, name in zip(PROCESS_COLORS, PROCESS_NAMES)
]

# Independent row labels separate simulation time from the spatial y-axis.
for time_index, time_label in enumerate(TIME_LABELS):
    group_top = area_axes[time_index, 0].get_position().y1
    group_bottom = map_axes[time_index, 0].get_position().y0
    fig.text(
        0.022,
        0.5 * (group_top + group_bottom),
        time_label,
        ha="center",
        va="center",
        rotation=90,
        fontsize=11.2,
    )

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.555, 0.002),
    ncol=4,
    frameon=False,
    fontsize=9,
    handletextpad=0.40,
    handlelength=1.8,
    handleheight=0.8,
    columnspacing=1.30,
)

output_stem = OUTPUT_DIR / "most_influential_process_integrated_panels_v13"
fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)

np.savez_compressed(
    OUTPUT_DIR / "most_influential_process_integrated_panels_v13_source_data.npz",
    process_names=np.asarray(PROCESS_NAMES),
    time_labels=np.asarray(TIME_LABELS),
    point_grid_indices=np.asarray(POINT_GRID_INDICES),
    point_coordinates=POINT_COORDINATES,
    first_order_point_values=FIRST_POINT_VALUES,
    total_effect_point_values=TOTAL_POINT_VALUES,
    dominant_point_process_ids=POINT_RESULTS[0][0],
    dominant_point_values=POINT_RESULTS[0][1],
    least_influential_point_process_ids=POINT_RESULTS[1][0],
    least_influential_point_values=POINT_RESULTS[1][1],
    dominant_area_percent=DOMINANT_AREA_PERCENT,
    least_influential_area_percent=INFERIOR_AREA_PERCENT,
    dominant_classification=HIGHEST_FIRST,
    least_influential_classification=LOWEST_TOTAL,
)

print(f"Saved figure files to: {output_stem}")
print("Dominant process evaluated-area percentages (C, F, H, R):")
print(np.round(DOMINANT_AREA_PERCENT, 3))
print("Least-influential process evaluated-area percentages (C, F, H, R):")
print(np.round(INFERIOR_AREA_PERCENT, 3))
