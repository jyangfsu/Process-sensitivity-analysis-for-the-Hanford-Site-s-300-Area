# -*- coding: utf-8 -*-
"""Compact pathway-resolved contributions to each process total effect.

Each panel in the 3-row by 4-column layout decomposes one process total effect at one time into
the exclusive pathways that contain that process: one first-order component,
three second-order pathways, three third-order pathways, and one fourth-order
pathway. Small jittered points show an area-weighted sample of grid-cell
fractions. Large circles show area-weighted pathway medians. Each
order, rather than each pathway circle, has one shared 5th--95th percentile
interval and one horizontal median line.

Pairwise P values compare order-level contributions aggregated within coarse
spatial blocks. Paired Wilcoxon tests with Holm correction reduce both spatial
pseudo-replication and multiplicity.
"""

from pathlib import Path
import csv
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import blended_transform_factory
import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from constvars import ntimes, nx, nz, time_dict, x, z


BASE_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_STEM = OUTPUT_DIR / "order_pathway_contributions_v12_cc"

TOTAL_EFFECT_TOLERANCE = 1.0e-6
MAX_SCATTER_POINTS = 85
N_BLOCKS_X = 8
N_BLOCKS_Z = 4
MIN_CELLS_PER_BLOCK = 20

# Muted blue--violet palette adapted from the supplied reference figure.
# The six pairwise pathways follow a three-color edge-coloring of the four
# processes, so the three pairwise terms shown in any panel remain distinct.
# The four third-order pathways use all four hues.
MUTED_VIOLET = "#6F6996"
MUTED_BLUE = "#6F9FC7"
PALE_PERIWINKLE = "#B7CBE5"
DUSTY_MAUVE = "#A47799"

NEUTRAL_COLOR = 'grey'#MUTED_VIOLET

PATHWAY_COLORS = {
    "CF": MUTED_BLUE,
    "CH": PALE_PERIWINKLE,
    "CR": DUSTY_MAUVE,
    "FH": DUSTY_MAUVE,
    "FR": PALE_PERIWINKLE,
    "HR": MUTED_BLUE,
    "CFH": MUTED_BLUE,
    "CFR": DUSTY_MAUVE,
    "CHR": PALE_PERIWINKLE,
    "FHR": MUTED_VIOLET,
}

# Within each process column, the paired second- and third-order pathways in
# the compact three-row legend use the same color. Colors are therefore local
# to a process column rather than global identifiers of a pathway.
PANEL_PATHWAY_COLORS = {
    "C": {
        "CF": MUTED_BLUE, "CFH": MUTED_BLUE,
        "CH": PALE_PERIWINKLE, "CHR": PALE_PERIWINKLE,
        "CR": DUSTY_MAUVE, "CFR": DUSTY_MAUVE,
    },
    "F": {
        "CF": MUTED_BLUE, "CFH": MUTED_BLUE,
        "FH": DUSTY_MAUVE, "FHR": DUSTY_MAUVE,
        "FR": PALE_PERIWINKLE, "CFR": PALE_PERIWINKLE,
    },
    "H": {
        "CH": PALE_PERIWINKLE, "CHR": PALE_PERIWINKLE,
        "FH": DUSTY_MAUVE, "CFH": DUSTY_MAUVE,
        "HR": MUTED_BLUE, "FHR": MUTED_BLUE,
    },
    "R": {
        "CR": DUSTY_MAUVE, "CHR": DUSTY_MAUVE,
        "FR": PALE_PERIWINKLE, "CFR": PALE_PERIWINKLE,
        "HR": MUTED_BLUE, "FHR": MUTED_BLUE,
    },
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    
 
    "mathtext.fontset": "dejavusans",
    
    "font.size":8.0,
    "axes.linewidth": 0.72,
    "xtick.major.width": 0.70,
    "ytick.major.width": 0.70,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_array(filename):
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


def centers_to_widths(centers):
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.abs(np.diff(edges))


def weighted_quantile(values, weights, probabilities):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return np.full(probabilities.shape, np.nan)
    sorting = np.argsort(values)
    values = values[sorting]
    weights = weights[sorting]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return np.interp(
        probabilities, cumulative, values,
        left=values[0], right=values[-1],
    )


def format_p_value(p_value):
    if not np.isfinite(p_value):
        return r"$P=\mathrm{NA}$"
    if p_value < 0.001:
        return r"$P<0.001$"
    return rf"$P={p_value:.3f}$"


def spatial_block_order_values(order_arrays, denominator, valid_mask,
                               cell_area, block_ids):
    """Return paired order-level contributions for coarse spatial blocks."""
    block_rows = []
    for block_id in np.unique(block_ids[valid_mask]):
        block_mask = valid_mask & (block_ids == block_id)
        if np.count_nonzero(block_mask) < MIN_CELLS_PER_BLOCK:
            continue
        block_weights = cell_area[block_mask]
        block_denominator = np.sum(block_weights * denominator[block_mask])
        if not np.isfinite(block_denominator) or block_denominator <= 0.0:
            continue
        row = [
            np.sum(block_weights * order_array[block_mask]) / block_denominator
            for order_array in order_arrays
        ]
        if np.all(np.isfinite(row)):
            block_rows.append(row)

    block_rows = np.asarray(block_rows, dtype=float)
    if block_rows.ndim != 2:
        return np.empty((0, len(order_arrays)), dtype=float)
    return block_rows


def holm_adjust(raw_p_values):
    """Holm-adjust a one-dimensional sequence of finite P values."""
    raw = np.asarray(raw_p_values, dtype=float)
    adjusted = np.full(raw.shape, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(raw))
    if finite_indices.size == 0:
        return adjusted
    ordered = finite_indices[np.argsort(raw[finite_indices])]
    running_max = 0.0
    m = ordered.size
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (m - rank) * raw[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def pairwise_order_tests(block_values):
    """Run all six paired order comparisons and apply Holm correction."""
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    raw_p_values = []
    for first, second in pairs:
        if block_values.shape[0] < 4:
            raw_p_values.append(np.nan)
            continue
        difference = block_values[:, first] - block_values[:, second]
        if np.allclose(difference, 0.0):
            raw_p_values.append(1.0)
            continue
        try:
            result = wilcoxon(
                block_values[:, first], block_values[:, second],
                alternative="two-sided", zero_method="wilcox",
            )
            raw_p_values.append(float(result.pvalue))
        except ValueError:
            raw_p_values.append(np.nan)
    adjusted = holm_adjust(raw_p_values)
    return {
        pair: (float(raw), float(adj))
        for pair, raw, adj in zip(pairs, raw_p_values, adjusted)
    }


first_order = {
    "C": load_array("ts_SI_CS_cc.npy"),
    "F": load_array("ts_SI_GF_bin_cc.npy"),
    "H": load_array("ts_SI_HT_cc.npy"),
    "R": load_array("ts_SI_RT_bin_cc.npy"),
}
total_effect = {
    "C": load_array("ts_ST_CS_cc.npy"),
    "F": load_array("ts_ST_GF_bin_cc.npy"),
    "H": load_array("ts_ST_HT_cc.npy"),
    "R": load_array("ts_ST_RT_bin_cc.npy"),
}
second_order = {
    "CF": load_array("ts_Interact_CF_cc.npy"),
    "CH": load_array("ts_Interact_CH_cc.npy"),
    "CR": load_array("ts_Interact_CR_cc.npy"),
    "FH": load_array("ts_Interact_HF_cc.npy"),
    "FR": load_array("ts_Interact_RF_cc.npy"),
    "HR": load_array("ts_Interact_RH_cc.npy"),
}
third_order = {
    "CFH": load_array("ts_Interact_CHF_cc.npy"),
    "CFR": load_array("ts_Interact_CRF_cc.npy"),
    "CHR": load_array("ts_Interact_CRH_cc.npy"),
    "FHR": load_array("ts_Interact_RHF_cc.npy"),
}
fourth_order = load_array("ts_Interact_CFHR_cc.npy")

processes = ["C", "F", "H", "R"]
process_names = {"C": "Climate\n", "F": "Flow\n", "H": "Heat\n", "R": "Reaction\n"}
second_members = {
    process: [name for name in second_order if process in name]
    for process in processes
}
third_members = {
    process: [name for name in third_order if process in name]
    for process in processes
}

dx = centers_to_widths(x)
dz = centers_to_widths(z)
cell_area = np.outer(dx, dz).reshape(-1)
x_coordinates = np.repeat(np.asarray(x, dtype=float), nz)
z_coordinates = np.tile(np.asarray(z, dtype=float), nx)
if not (cell_area.size == x_coordinates.size == z_coordinates.size == nx * nz):
    raise RuntimeError("Grid geometry does not match the sensitivity arrays.")

x_block = np.clip(
    np.digitize(
        x_coordinates,
        np.linspace(np.min(x_coordinates), np.max(x_coordinates), N_BLOCKS_X + 1),
    ) - 1,
    0, N_BLOCKS_X - 1,
)
z_block = np.clip(
    np.digitize(
        z_coordinates,
        np.linspace(np.min(z_coordinates), np.max(z_coordinates), N_BLOCKS_Z + 1),
    ) - 1,
    0, N_BLOCKS_Z - 1,
)
block_ids = z_block * N_BLOCKS_X + x_block


fig, axes = plt.subplots(
    3, 4, figsize=(7.2, 7.2), sharex=True, sharey=True,
)
summary_rows = []
test_rows = []

for col, process in enumerate(processes):
    for row in range(ntimes):
        ax = axes[row, col]
        denominator = total_effect[process][row]
        category_groups = [
            [(process, first_order[process][row], NEUTRAL_COLOR)],
            [(name, second_order[name][row], PANEL_PATHWAY_COLORS[process][name])
             for name in second_members[process]],
            [(name, third_order[name][row], PANEL_PATHWAY_COLORS[process][name])
             for name in third_members[process]],
            [("CFHR", fourth_order[row], NEUTRAL_COLOR)],
        ]

        valid = (
            np.isfinite(denominator)
            & (denominator > TOTAL_EFFECT_TOLERANCE)
            & np.isfinite(cell_area)
            & (cell_area > 0.0)
        )
        for group in category_groups:
            for _, component, _ in group:
                valid &= np.isfinite(component)
        if not np.any(valid):
            raise RuntimeError(
                f"No valid cells for process {process} at time index {col}."
            )

        weights = cell_area[valid]
        denominator_valid = denominator[valid]
        sample_count = min(MAX_SCATTER_POINTS, denominator_valid.size)
        sample_rng = np.random.default_rng(1000 + 100 * col + row)
        sample_indices = sample_rng.choice(
            denominator_valid.size,
            size=sample_count,
            replace=False,
            p=weights / np.sum(weights),
        )

        order_arrays = [
            np.sum([component for _, component, _ in group], axis=0)
            for group in category_groups
        ]

        for order_number, (group, order_array) in enumerate(
            zip(category_groups, order_arrays), start=1
        ):
            offsets = (
                np.array([0.0]) if len(group) == 1
                else np.linspace(-0.070, 0.070, len(group))
            )

            # One shared interval and median per interaction order.
            order_fraction = 100.0 * order_array[valid] / denominator_valid
            order_q05, order_median, order_q95 = weighted_quantile(
                order_fraction, weights, [0.05, 0.50, 0.95]
            )
            # Half-width doubled from V7 so all three horizontal summary
            # strokes (lower cap, median, and upper cap) are twice as long.
            cap_half_width = 0.250
            # The shared 5th--95th percentile interval and median are the
            # primary summary marks and must remain above every point/circle.
            ax.vlines(order_number, order_q05, order_q95,
                      color="black", lw=0.82, zorder=20)
            ax.hlines(
                [order_q05, order_q95],
                order_number - 0.5 * cap_half_width,
                order_number + 0.5 * cap_half_width,
                color="black", lw=0.82, zorder=20,
            )
            ax.hlines(
                order_median,
                order_number - cap_half_width,
                order_number + cap_half_width,
                color="black", lw=1.12, zorder=21,
            )

            for category_index, ((name, component, color), offset) in enumerate(
                zip(group, offsets)
            ):
                component_valid = component[valid]
                fraction = 100.0 * component_valid / denominator_valid
                q05, median, q95 = weighted_quantile(
                    fraction, weights, [0.05, 0.50, 0.95]
                )
                integrated = 100.0 * (
                    np.sum(weights * component_valid)
                    / np.sum(weights * denominator_valid)
                )
                x_position = order_number + offset

                jitter_rng = np.random.default_rng(
                    5000 + 1000 * col + 100 * row
                    + 10 * order_number + category_index
                )
                jitter = np.clip(
                    jitter_rng.normal(0.0, 0.028, sample_count), -0.070, 0.070
                )
                ax.scatter(
                    order_number + jitter,
                    fraction[sample_indices],
                    s=5.8,
                    color=color,
                    alpha=0.30,
                    edgecolors="none",
                    rasterized=True,
                    zorder=1,
                )

                # Large circle: area-weighted median pathway contribution.
                ax.scatter(
                    x_position, median,
                    s=66, marker="o", facecolor=color, edgecolor="black",
                    linewidth=0.82, alpha=0.82, zorder=4,
                )

                summary_rows.append({
                    "process": process_names[process],
                    "time": time_dict[str(row)].replace("Time = ", ""),
                    "order": order_number,
                    "pathway": name,
                    "n_cells": int(np.sum(valid)),
                    "weighted_q05_percent": q05,
                    "weighted_median_percent": median,
                    "weighted_q95_percent": q95,
                    "circle_value_percent": median,
                    "circle_statistic": "area-weighted median",
                    "area_integrated_contribution_percent": integrated,
                })

        # Pairwise order comparisons follow the bracket logic of the supplied
        # template. All six comparisons are Holm-adjusted and saved; five
        # planned contrasts are displayed above the plotting region.
        block_values = spatial_block_order_values(
            order_arrays, denominator, valid, cell_area, block_ids,
        )
        order_tests = pairwise_order_tests(block_values)
        for (first, second), (raw_p, adjusted_p) in order_tests.items():
            test_rows.append({
                "process": process_names[process],
                "time": time_dict[str(row)].replace("Time = ", ""),
                "order_1": first + 1,
                "order_2": second + 1,
                "test": "Paired Wilcoxon test of spatial-block order contributions",
                "n_blocks": int(block_values.shape[0]),
                "raw_p_value": raw_p,
                "holm_adjusted_p_value": adjusted_p,
            })

        displayed_comparisons = [
            ((0, 1), 0.90),
            ((2, 3), 0.90),
            ((1, 2), 1.025),
            ((1, 3), 1.15),
            ((0, 2), 1.275),
        ]
        p_transform = blended_transform_factory(ax.transData, ax.transAxes)
        for (first, second), bracket_y in displayed_comparisons:
            adjusted_p = order_tests[(first, second)][1]
            x_start, x_end = first + 1, second + 1
            ax.plot(
                [x_start + 0.08, x_end - 0.08],
                [bracket_y, bracket_y],
                color="black", lw=0.70, clip_on=False,
                transform=p_transform,
            )
            ax.text(
                0.5 * (x_start + x_end), bracket_y + 0.015,
                format_p_value(adjusted_p),
                ha="center", va="bottom", fontsize=6.8,
                transform=p_transform, clip_on=False,
            )

        ax.set_xlim(0.72, 4.28)
        ax.set_ylim(-5.0, 75.0)
        ax.set_box_aspect(1.0)
        ax.set_xticks([1, 2, 3, 4], ["First-", "Second-", "Third-", "Fourth-"], rotation=30)
        ax.set_yticks([0.0, 25.0, 50.0, 75.0])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.tick_params(axis="both", labelsize=7.3, pad=2.0)
        ax.text(
            -0.275, 1.30, f"({chr(ord('a') + row)}{col + 1})",
            transform=ax.transAxes, ha="left", va="center", fontsize=9.0,
            fontweight="bold", clip_on=False,
        )
        if col == 0:
            ax.set_ylabel(
                time_dict[str(row)].replace("Time = ", ""),
                fontsize=9.4, labelpad=6, 
            )

#fig.supxlabel("First- and Interaction- order", fontsize=9.4, y=0.025)
fig.supylabel("Area-weighted contribution to $PS_{TK}$ (%)", fontsize=9.4, y=0.55, x=-0.01, va='center', ha='center')

fig.subplots_adjust(
    left=0.075, right=0.990, bottom=0.310, top=0.895,
    wspace=0.6, hspace=0.60,
)

# Each column has a compact, three-row pathway key. Each row groups one
# second-order and one third-order pathway; first- and fourth-order components
# are deliberately omitted because both use the same neutral summary symbol.
column_legend_pairs = {
    "C": [("CF", "CFH"), ("CH", "CHR"), ("CR", "CFR")],
    "F": [("CF", "CFH"), ("FH", "FHR"), ("FR", "CFR")],
    "H": [("CH", "CHR"), ("FH", "CFH"), ("HR", "FHR")],
    "R": [("CR", "CHR"), ("FR", "CFR"), ("HR", "FHR")],
}


def pathway_marker(process, pathway):
    return Line2D(
        [0], [0], marker="o", linestyle="none", markersize=4.8,
        markerfacecolor=PANEL_PATHWAY_COLORS[process][pathway],
        markeredgecolor="black",
        markeredgewidth=0.50,
    )


# Column headings and pathway keys are placed beneath their corresponding
# columns, below the rotated x tick labels and above the shared x-axis title.
for col, process in enumerate(processes):
    position = axes[0, col].get_position()
    column_center = 0.5 * (position.x0 + position.x1)
    fig.text(
        column_center, 0.250,
        process_names[process], #fontweight='bold',
        ha="center", va="top", fontsize=10.0,
    )
    pathway_pairs = column_legend_pairs[process]
    paired_handles = [
        (pathway_marker(process, second), pathway_marker(process, third))
        for second, third in pathway_pairs
    ]
    paired_labels = [
        rf"$PS_{{{second}}}$  /  $PS_{{{third}}}$"
        for second, third in pathway_pairs
    ]
    fig.legend(
        handles=paired_handles,
        labels=paired_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.30)},
        loc="upper center",
        bbox_to_anchor=(column_center, 0.225),
        ncol=1,
        frameon=False,
        fontsize=7.5,
        labelspacing=0.22,
        handlelength=2.0,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)

summary_path = OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_summary.csv")
with summary_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

test_path = OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_spatial_block_tests.csv")
with test_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(test_rows[0].keys()))
    writer.writeheader()
    writer.writerows(test_rows)

print(f"Saved figure files to: {OUTPUT_STEM}")
print(f"Saved pathway summary to: {summary_path}")
print(f"Saved spatial-block tests to: {test_path}")
