"""Schematic showing how binning replaces nested conditional sampling.

The figure is designed for a double-column manuscript layout.  It contrasts
the model evaluations required by direct nested estimation with the local
replication created by binning a common paired ensemble.  The final panel maps
the same operation to first-order, interaction, and total-effect indices.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'



PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "figures" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_STEM = OUTPUT_DIR / "binning_method_schematic_v8"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        #"font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        #"font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        
        
        "font.size": 7.2,
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


COLORS = {
    "ink": "#25313C",
    "muted": "#687684",
    "line": "#AEB8C2",
    "light": "#F4F6F8",
    "nested": "#D98B73",
    "nested_light": "#F6E3DC",
    "blue": "#4F86B5",
    "blue_light": "#DCEAF4",
    "teal": "#4E9C8A",
    "teal_light": "#DDEFEA",
    "gold": "#D9A441",
    "gold_light": "#F5EACF",
}


def rounded_box(ax, xy, width, height, text, facecolor, edgecolor=None,
                fontsize=7.0, weight="normal", textcolor=None, radius=0.02,
                linewidth=0.8, zorder=2):
    """Draw a rounded rectangle with centered text in axes coordinates."""
    if edgecolor is None:
        edgecolor = facecolor
    if textcolor is None:
        textcolor = COLORS["ink"]
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=textcolor,
        zorder=zorder + 1,
    )
    return patch


def arrow(ax, start, end, color=None, linewidth=1.0, mutation_scale=8,
          connectionstyle="arc3"):
    """Draw an arrow in axes coordinates."""
    if color is None:
        color = COLORS["muted"]
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
        clip_on=False,
        zorder=5,
    )
    ax.add_patch(patch)
    return patch


def panel_label(ax, label):
    """Panel labels are included directly in panel titles."""
    return None


def draw_nested_panel(ax):
    
    
    """Panel a: conventional nested conditional sampling."""
    panel_label(ax, "a")
    ax.text(
        0.02,
        0.98,
        "(a) Direct nested estimation",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.02,
        0.915,
        #r'Retain $M_{A_j},\theta_{A_j}^{(i)}$; average over $\boldsymbol{\theta}_{\sim A}$, then $\mathbf{M}_{\sim A}$',
        #r'Retain $M_{A_j},\theta_{A_j}^{(i)}$; average over $\mathbf{\mathrm{\theta}}_{\sim A}$, then $\mathbf{M}_{\sim A}$',
        #r'Retain $M_{A_j},\theta_{A_j}^{(i)}$; average over ' + 'θ' + r'$_{\sim A}$, then $\mathbf{M}_{\sim A}$',
        r'Retain $M_{A_j},\theta_{A_j}^{(i)}$; average over ' + r'$\mathbf{θ}$' + r'$_{\sim A}$, then $\mathbf{M}_{\sim A}$',
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        #fontweight='bold',
        color=COLORS["muted"],
    )

    x_positions = [0.12, 0.32, 0.52, 0.72]
    y_positions = np.linspace(0.33, 0.73, 5)
    
   

    for idx, x_pos in enumerate(x_positions, start=1):
        rounded_box(
            ax,
            (x_pos - 0.055, 0.79),
            0.11,
            0.07,
            rf"$\theta_{{A_j}}^{{({idx})}}$",
            COLORS["nested_light"],
            COLORS["nested"],
            fontsize=6.7,
            radius=0.012,
        )
        
    
    
        for sample_index, y_pos in enumerate(y_positions):
            ax.plot(
                x_pos,
                y_pos,
                marker="o",
                markersize=3.8,
                markerfacecolor=COLORS["nested"],
                markeredgecolor="white",
                markeredgewidth=0.35,
                transform=ax.transAxes,
                zorder=4,
            )
            ax.plot(
                [x_pos, x_pos],
                [0.785, y_pos + 0.012],
                color=COLORS["line"],
                linewidth=0.55,
                transform=ax.transAxes,
                zorder=1,
            )
    
    ax.text(
        0.02,
        0.57,
        r"samples of $\boldsymbol{\theta}_{\sim A}$",
        rotation=90,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        color=COLORS["muted"],
    )

    arrow(ax, (0.735, 0.54), (0.82, 0.54), COLORS["nested"], 1.15, 9)
    ax.text(
        0.90,
        0.77,
        "model-averaged\nconditional means",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color=COLORS["muted"],
    )
    for idx, y_pos in enumerate([0.67, 0.56, 0.45, 0.34], start=1):
        rounded_box(
            ax,
            (0.83, y_pos - 0.032),
            0.14,
            0.064,
            rf"$\widehat{{m}}_{{A_j}}^{{({idx})}}$",
            COLORS["nested_light"],
            COLORS["nested"],
            fontsize=6.2,
            radius=0.010,
        )

    ax.set_axis_off()


def draw_binning_panel(ax):
    """Panel b: local replication from a common paired ensemble."""
    panel_label(ax, "b")
    ax.text(
        0.02,
        0.98,
        r"(b) Binning the retained subset " + r"$\mathbf{θ}_{A}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.02,
        0.915,
        r"Each $\theta_A$ is paired with one " + r"$\mathbf{θ}_{\sim A}$" + " realization",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=COLORS["muted"],
    )

    plot_left, plot_bottom = 0.10, 0.36
    plot_width, plot_height = 0.63, 0.43
    bin_colors = ["#DCEAF4", "#C8DFEE", "#B4D4E8", "#9FC9E1"]
    point_colors = ["#5C93BE", "#4F86B5", "#3E789F", "#2F688B"]

    for bin_index in range(4):
        x0 = plot_left + bin_index * plot_width / 4
        ax.add_patch(
            Rectangle(
                (x0, plot_bottom),
                plot_width / 4,
                plot_height,
                transform=ax.transAxes,
                facecolor=bin_colors[bin_index],
                edgecolor="white",
                linewidth=0.8,
                zorder=0,
            )
        )
        ax.text(
            x0 + plot_width / 8,
            plot_bottom + plot_height + 0.015,
            rf"$\mathcal{{B}}_{{A,{bin_index + 1}}}$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=COLORS["ink"],
        )

    theta_a = np.array(
        [0.03, 0.08, 0.14, 0.20, 0.23, 0.28,
         0.30, 0.36, 0.40, 0.44, 0.48, 0.49,
         0.52, 0.57, 0.63, 0.67, 0.70, 0.73,
         0.77, 0.82, 0.87, 0.91, 0.95, 0.98]
    )
    theta_other = np.array(
        [0.76, 0.18, 0.52, 0.34, 0.88, 0.64,
         0.10, 0.46, 0.81, 0.26, 0.61, 0.93,
         0.39, 0.71, 0.16, 0.84, 0.55, 0.29,
         0.96, 0.43, 0.68, 0.22, 0.79, 0.49]
    )
    bins = np.minimum((theta_a * 4).astype(int), 3)
    x_plot = plot_left + theta_a * plot_width
    y_plot = plot_bottom + theta_other * plot_height

    for bin_index in range(4):
        select = bins == bin_index
        ax.scatter(
            x_plot[select],
            y_plot[select],
            s=17,
            color=point_colors[bin_index],
            edgecolor="white",
            linewidth=0.35,
            transform=ax.transAxes,
            zorder=3,
        )

    ax.plot(
        [plot_left, plot_left + plot_width],
        [plot_bottom, plot_bottom],
        color=COLORS["ink"],
        linewidth=0.7,
        transform=ax.transAxes,
    )
    ax.plot(
        [plot_left, plot_left],
        [plot_bottom, plot_bottom + plot_height],
        color=COLORS["ink"],
        linewidth=0.7,
        transform=ax.transAxes,
    )
    ax.text(
        plot_left + plot_width / 2,
        plot_bottom - 0.055,
        r"conditioning parameter $\theta_A$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.7,
        color=COLORS["ink"],
    )
    ax.text(
        plot_left - 0.065,
        plot_bottom + plot_height / 2,
        r"associated $\boldsymbol{\theta}_{\sim A}$",
        rotation=90,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.7,
        color=COLORS["ink"],
    )

    arrow(ax, (0.75, 0.58), (0.82, 0.58), COLORS["blue"], 1.0, 8)
    ax.text(
        0.91,
        0.835,
        "INNER CALCULATION\nmodel-averaged bin means\n"
        "(average over " + r"$\mathbf{θ}_{\sim A}$" + r" , then $\mathbf{M}_{\sim A}$)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color=COLORS["muted"],
    )

    mean_y = [0.71, 0.61, 0.51, 0.41]
    for idx, y_pos in enumerate(mean_y):
        rounded_box(
            ax,
            (0.84, y_pos - 0.035),
            0.13,
            0.065,
            rf"$\widehat{{m}}_{{A,{idx + 1}}}$",
            bin_colors[idx],
            point_colors[idx],
            fontsize=6.4,
            radius=0.010,
        )

    arrow(ax, (0.905, 0.365), (0.905, 0.225), COLORS["teal"], 0.9, 8)
    ax.text(
        0.52,
        0.155,
        r"OUTER CALCULATION: average $\widehat m_{A,h}^{2}(M_A)$ across bins and $M_A$"
        "\n" r"subtract $\widehat\mu^2$, normalize by $\widehat V(\Delta)$, and obtain $\widehat{PS}_A$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.95,
        color=COLORS["teal"],
        fontweight="bold",
    )

    ax.set_axis_off()


def draw_complement_panel(ax):
    """Panel c: regroup the identical ensemble by the complementary subset."""
    panel_label(ax, "c")
    ax.text(
        0.02, 0.98, r"(c) Binning the complementary subset $\mathbf{θ}_{\sim A}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.2,
        fontweight="bold", color=COLORS["ink"],
    )
    ax.text(
        0.02, 0.915,
        r"Each $\mathbf{θ}_{\sim A}$ is paired with one $\theta_A$ realization",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
        color=COLORS["muted"],
    )

    plot_left, plot_bottom = 0.10, 0.36
    plot_width, plot_height = 0.63, 0.43
    bin_colors = ["#DDEFEA", "#CBE5DE", "#B8DBD1", "#A4D0C4"]
    point_colors = ["#5AA692", "#4E9C8A", "#3F8B79", "#327966"]

    for bin_index in range(4):
        y0 = plot_bottom + bin_index * plot_height / 4
        ax.add_patch(
            Rectangle(
                (plot_left, y0), plot_width, plot_height / 4,
                transform=ax.transAxes, facecolor=bin_colors[bin_index],
                edgecolor="white", linewidth=0.8, zorder=0,
            )
        )
        ax.text(
            plot_left + 0.012, y0 + plot_height / 8,
            rf"$\mathcal{{B}}_{{\sim A,{bin_index + 1}}}$",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=5.9, color=COLORS["ink"],
        )

    # Identical paired realizations and coordinates as panel b.
    theta_a = np.array(
        [0.03, 0.08, 0.14, 0.20, 0.23, 0.28,
         0.30, 0.36, 0.40, 0.44, 0.48, 0.49,
         0.52, 0.57, 0.63, 0.67, 0.70, 0.73,
         0.77, 0.82, 0.87, 0.91, 0.95, 0.98]
    )
    theta_other = np.array(
        [0.76, 0.18, 0.52, 0.34, 0.88, 0.64,
         0.10, 0.46, 0.81, 0.26, 0.61, 0.93,
         0.39, 0.71, 0.16, 0.84, 0.55, 0.29,
         0.96, 0.43, 0.68, 0.22, 0.79, 0.49]
    )
    bins = np.minimum((theta_other * 4).astype(int), 3)
    x_plot = plot_left + theta_a * plot_width
    y_plot = plot_bottom + theta_other * plot_height

    for bin_index in range(4):
        select = bins == bin_index
        ax.scatter(
            x_plot[select], y_plot[select], s=17,
            color=point_colors[bin_index], edgecolor="white",
            linewidth=0.35, transform=ax.transAxes, zorder=3,
        )

    ax.plot([plot_left, plot_left + plot_width], [plot_bottom, plot_bottom],
            color=COLORS["ink"], linewidth=0.7, transform=ax.transAxes)
    ax.plot([plot_left, plot_left], [plot_bottom, plot_bottom + plot_height],
            color=COLORS["ink"], linewidth=0.7, transform=ax.transAxes)
    ax.text(
        plot_left + plot_width / 2, plot_bottom - 0.055,
        r"associated parameter $\theta_A$", transform=ax.transAxes,
        ha="center", va="top", fontsize=6.7, color=COLORS["ink"],
    )
    ax.text(
        plot_left - 0.095, plot_bottom + plot_height / 2,
        r"conditioning subset $\boldsymbol{\theta}_{\sim A}$",
        rotation=90, transform=ax.transAxes, ha="center", va="center",
        fontsize=6.5, color=COLORS["ink"],
    )

    arrow(ax, (0.75, 0.58), (0.82, 0.58), COLORS["teal"], 1.0, 8)
    ax.text(
        0.91, 0.795,
        "INNER CALCULATION\nmodel-averaged bin means\n"
        "(average over " + r"$\mathbf{θ}_{\sim A}$" + r" , then $\mathbf{M}_{\sim A}$)",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=6.1, color=COLORS["muted"],
    )

    mean_y = [0.71, 0.61, 0.51, 0.41]
    for idx, y_pos in enumerate(mean_y):
        rounded_box(
            ax, (0.84, y_pos - 0.035), 0.13, 0.065,
            rf"$\widehat{{m}}_{{\sim A,{idx + 1}}}$",
            bin_colors[idx], point_colors[idx], fontsize=6.2,
            radius=0.010,
        )

    arrow(ax, (0.905, 0.365), (0.905, 0.225), COLORS["teal"], 0.9, 8)
    ax.text(
        0.52, 0.145,
        r"OUTER CALCULATION: average $\widehat m_{\sim A,h}^{2}(M_{\sim A})$ across bins and $M_{\sim A}$"
        "\n" r"subtract $\widehat\mu^2$, normalize by $\widehat V(\Delta)$, and take the complement to obtain $\widehat{PS}_{T,A}$",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=5.9, color=COLORS["teal"], fontweight="bold",
    )
    ax.set_axis_off()


def draw_index_panel(ax):
    """Panel d: retained subsets directly yield the complete index set."""
    panel_label(ax, "d")
    ax.text(
        0.02, 0.98, "(d) Changing the retained subset estimates all indices",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.2,
        fontweight="bold", color=COLORS["ink"],
    )
    ax.text(
        0.02, 0.915,
        r"Colored blocks are retained and binned; gray blocks are averaged",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.7,
        color=COLORS["muted"],
    )

    process_fill = {
        "A": COLORS["blue_light"],
        "B": COLORS["teal_light"],
        "C": COLORS["gold_light"],
    }
    process_edge = {
        "A": COLORS["blue"],
        "B": COLORS["teal"],
        "C": COLORS["gold"],
    }

    rows = [
        (
            r"$\widehat{PS}_A$", {"A"},
            "first-order contribution of $A$",
            "bin $A$; average $B$ and $C$",
        ),
        (
            r"$\widehat{PS}_{AB}$", {"A", "B"},
            r"subtract $\widehat{PS}_A$ and $\widehat{PS}_B$ from the joint contribution",
            "jointly bin $A$ and $B$; average $C$",
        ),
        (
            r"$\widehat{PS}_{ABC}$", {"A", "B", "C"},
            r"$\widehat{PS}_{ABC}=1-\sum_K\widehat{PS}_K-\sum_{K<L}\widehat{PS}_{KL}$",
            "retain A, B, and C; apply variance closure",
        ),
    ]
    y_values = [0.69, 0.47, 0.25]

    for (joint_label, retained, equation, note), y in zip(rows, y_values):
        ax.text(0.04, y + 0.055, joint_label, transform=ax.transAxes,
                ha="left", va="center", fontsize=7.2, fontweight="bold",
                color=COLORS["ink"])

        for idx, process in enumerate(["A", "B", "C"]):
            x = 0.22 + idx * 0.105
            is_retained = process in retained
            rounded_box(
                ax, (x, y), 0.082, 0.11, process,
                process_fill[process] if is_retained else COLORS["light"],
                edgecolor=process_edge[process] if is_retained else COLORS["line"],
                fontsize=7.0, weight="bold" if is_retained else "normal",
                textcolor=COLORS["ink"] if is_retained else COLORS["muted"],
                radius=0.010,
            )

        ax.text(0.57, y + 0.070, equation, transform=ax.transAxes,
                ha="left", va="center", fontsize=6.6, color=COLORS["ink"])
        ax.text(0.57, y + 0.025, note, transform=ax.transAxes,
                ha="left", va="center", fontsize=5.8, color=COLORS["muted"])

    ax.plot([0.04, 0.96], [0.205, 0.205], transform=ax.transAxes,
            color=COLORS["line"], linewidth=0.65)
    ax.text(
        0.04, 0.165, "Total effect of $A$ collects every exclusive term containing $A$:",
        transform=ax.transAxes, ha="left", va="center", fontsize=6.2,
        color=COLORS["muted"],
    )

    total_terms = [
        r"$\widehat{PS}_A$", r"$\widehat{PS}_{AB}$",
        r"$\widehat{PS}_{AC}$", r"$\widehat{PS}_{ABC}$",
    ]
    for idx, term in enumerate(total_terms):
        x = 0.10 + idx * 0.205
        rounded_box(
            ax, (x, 0.055), 0.16, 0.075, term,
            COLORS["blue_light"], edgecolor=COLORS["blue"],
            fontsize=6.5, radius=0.010,
        )
        if idx < len(total_terms) - 1:
            ax.text(x + 0.182, 0.092, "+", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7.2,
                    color=COLORS["muted"])

    ax.set_axis_off()


def build_figure():
    fig = plt.figure(figsize=(6.2, 6.75))
    grid = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.045, right=0.98, bottom=0.045, top=0.965,
        wspace=0.1, hspace=-0.05,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    draw_nested_panel(ax_a)
    draw_binning_panel(ax_b)
    draw_complement_panel(ax_c)
    draw_index_panel(ax_d)

    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 400},
    }.items():
        fig.savefig(
            str(OUTPUT_STEM) + suffix,
            bbox_inches="tight",
            pad_inches=0.035,
            **kwargs,
        )
    return fig


if __name__ == "__main__":
    figure = build_figure()
    plt.close(figure)
    print(f"Saved figure files to {OUTPUT_STEM}.[svg|pdf|png]")
