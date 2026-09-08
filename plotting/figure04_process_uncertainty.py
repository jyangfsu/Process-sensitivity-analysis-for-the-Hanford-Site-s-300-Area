"""Four-process uncertainty framework and complete sensitivity-index inventory.

Panel (a) shows how uncertainty enters the coupled reactive-transport model.
Panel (b) identifies the 15 mutually exclusive variance components and four
total-effect indices evaluated at every retained grid cell and time, and shows
the equivalent complementary and component-sum calculations of each total effect.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


W, H = 958.0, 930.0

mpl.rcParams.update(
    {
     "font.family": "sans-serif",
     #"font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
     #"font.family": "sans-serif",
     "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
     
  
     "mathtext.fontset": "dejavusans",
     
        "font.size": 12,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


COLORS = {
    "climate": "#9BD5C0",
    "climate_title": "#79CDB0",
    "flow": "#A8CFDF",
    "flow_title": "#91C8DF",
    "heat": "#FAC0B7",
    "heat_title": "#F4AAA2",
    "reaction": "#D2B3D3",
    "reaction_title": "#C7A6CA",
    "parameter": "#00B956",
    "equation": "#383838",
    "model_gray": "#A9A9A9",
    "output": "#ADE39C",
}


def rounded_box(
    ax,
    x,
    y,
    width,
    height,
    text,
    facecolor="white",
    edgecolor="black",
    textcolor="black",
    linewidth=1.7,
    fontsize=12.5,
    fontstyle="normal",
    fontweight="normal",
    radius=6,
    zorder=4,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size={}".format(radius),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=textcolor,
        fontsize=fontsize,
        fontstyle=fontstyle,
        fontweight=fontweight,
        linespacing=1.35,
        zorder=zorder + 1,
    )
    return patch


def output_ellipse(ax, x, y, width, height, text):
    patch = Ellipse(
        (x, y),
        width,
        height,
        facecolor=COLORS["output"],
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=12.5,
        fontstyle="normal",
        linespacing=1.35,
        zorder=4,
    )
    return patch


def line(ax, points, linewidth=1.7, zorder=2):
    xs, ys = zip(*points)
    ax.plot(
        xs,
        ys,
        color="black",
        linewidth=linewidth,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=zorder,
    )


def arrow(ax, start, end, linewidth=1.7, mutation_scale=12, zorder=3):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color="black",
        shrinkA=0,
        shrinkB=0,
        connectionstyle="arc3,rad=0",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def build_figure():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    # Reserve a dedicated title line above panel (a).
    ax.set_ylim(H, -30)
    ax.axis("off")

    # Process regions
    ax.add_patch(
        Rectangle((35, 38), 378, 241, facecolor=COLORS["climate"], edgecolor="none")
    )
    ax.add_patch(
        Rectangle((437, 39), 510, 241, facecolor=COLORS["flow"], edgecolor="none")
    )
    ax.add_patch(
        Rectangle((35, 317), 378, 141, facecolor=COLORS["heat"], edgecolor="none")
    )
    ax.add_patch(
        Rectangle((661, 317), 285, 254, facecolor=COLORS["reaction"], edgecolor="none")
    )

    # Region titles
    title_kw = dict(fontsize=12.5, fontweight="bold", ha="left", va="bottom")
    ax.text(55, 31, "Climate Process", color=COLORS["climate_title"], **title_kw)
    ax.text(437, 32, "Flow Process", color=COLORS["flow_title"], **title_kw)
    ax.text(35, 311, "Heat Process", color=COLORS["heat_title"], **title_kw)
    ax.text(661, 312, "Reaction Process", color=COLORS["reaction_title"], **title_kw)

    # Climate alternatives
    climate_y = [48, 86, 124, 163, 201, 239]
    for i, y in enumerate(climate_y, start=1):
        rounded_box(ax, 47, y, 55, 28, "C{}".format(i), fontsize=11.5, radius=5)

    rounded_box(
        ax,
        165,
        86,
        190,
        33,
        "River Stage Level",
        facecolor=COLORS["model_gray"],
        edgecolor=COLORS["model_gray"],
        textcolor="white",
        fontsize=12,
        radius=6,
    )
    rounded_box(
        ax,
        165,
        143,
        190,
        33,
        "Well Water Level",
        facecolor=COLORS["model_gray"],
        edgecolor=COLORS["model_gray"],
        textcolor="white",
        fontsize=12,
        radius=6,
    )
    rounded_box(
        ax,
        165,
        200,
        190,
        33,
        "Temperature",
        facecolor=COLORS["model_gray"],
        edgecolor=COLORS["model_gray"],
        textcolor="white",
        fontsize=12,
        radius=6,
    )

    # Climate connector bus and arrows
    line(ax, [(102, 62), (134, 62), (134, 253)])
    for y in [100, 138, 177, 215, 253]:
        line(ax, [(102, y), (134, y)])
    arrow(ax, (134, 102), (165, 102))
    arrow(ax, (134, 159), (165, 159))
    arrow(ax, (134, 216), (165, 216))

    # Flow alternatives and permeability model
    flow_g_y = [54, 99, 144, 189, 234]
    for i, y in enumerate(flow_g_y, start=1):
        rounded_box(ax, 863, y, 54, 28, "G{}".format(i), fontsize=11.5, radius=5)

    rounded_box(ax, 666, 111, 150, 34, "Homogenous", fontsize=12.5, radius=6)
    rounded_box(ax, 666, 178, 150, 34, "Heterogenous", fontsize=12.5, radius=6)
    rounded_box(
        ax,
        482,
        138,
        138,
        56,
        "Permeability\nField",
        facecolor=COLORS["parameter"],
        edgecolor=COLORS["parameter"],
        textcolor="white",
        fontsize=12,
        radius=9,
    )
    rounded_box(
        ax,
        482,
        215,
        138,
        56,
        "Richard\nEquation",
        facecolor=COLORS["equation"],
        edgecolor=COLORS["equation"],
        textcolor="white",
        fontsize=12,
        radius=9,
    )

    # G-model connector bus
    line(ax, [(863, 68), (840, 68), (840, 248), (863, 248)])
    for y in [113, 158, 203]:
        line(ax, [(840, y), (863, y)])
    arrow(ax, (840, 128), (816, 128))
    arrow(ax, (840, 195), (816, 195))

    # Model alternatives to permeability field
    line(ax, [(666, 128), (644, 128), (644, 195), (666, 195)])
    arrow(ax, (644, 166), (620, 166))
    arrow(ax, (551, 194), (551, 215))

    # Climate forcing to flow equation
    line(ax, [(355, 102), (425, 102), (425, 243)])
    line(ax, [(355, 159), (425, 159)])
    arrow(ax, (425, 243), (482, 243))

    # Heat alternatives and governing equation
    rounded_box(ax, 47, 328, 158, 56, "Heat\nDependent", fontsize=12, radius=10)
    rounded_box(ax, 47, 396, 158, 56, "Heat\nIndependent", fontsize=12, radius=10)
    rounded_box(
        ax,
        252,
        363,
        151,
        56,
        "Heat Transfer\nEquation",
        facecolor=COLORS["equation"],
        edgecolor=COLORS["equation"],
        textcolor="white",
        fontsize=12,
        radius=9,
    )

    # Climate temperature to heat alternatives
    line(ax, [(179, 233), (179, 328)])
    arrow(ax, (179, 306), (179, 328))

    # Heat alternatives to heat equation
    line(ax, [(205, 356), (228, 356), (228, 424), (205, 424)])
    arrow(ax, (228, 389), (252, 389))

    # Output fields
    output_ellipse(ax, 550, 355, 202, 57, "Velocity Field")
    output_ellipse(ax, 550, 437, 202, 58, "Temperature\nField")
    output_ellipse(ax, 550, 530, 202, 76, "OC\nConsumption\nRate")

    # Governing equations to output fields
    arrow(ax, (551, 271), (551, 326))
    line(ax, [(449, 355), (425, 355), (425, 391)])
    arrow(ax, (425, 391), (403, 391))
    line(ax, [(327, 419), (327, 437)])
    arrow(ax, (327, 437), (449, 437))

    # Reaction nodes
    rounded_box(
        ax,
        728,
        327,
        199,
        57,
        "ADR\nEquation",
        facecolor=COLORS["equation"],
        edgecolor=COLORS["equation"],
        textcolor="white",
        fontsize=12,
        radius=9,
    )
    rounded_box(
        ax,
        674,
        411,
        130,
        57,
        "Reaction\nRates",
        facecolor=COLORS["parameter"],
        edgecolor=COLORS["parameter"],
        textcolor="white",
        fontsize=12,
        radius=9,
    )

    # Flow and heat fields to reaction model
    arrow(ax, (651, 355), (728, 355))
    arrow(ax, (651, 437), (674, 437))
    arrow(ax, (760, 411), (760, 384))
    line(ax, [(827, 384), (827, 530), (651, 530)])
    arrow(ax, (680, 530), (651, 530))

    # Legend
    legend_y = [471, 508, 545]
    legend_faces = ["white", COLORS["parameter"], COLORS["equation"]]
    legend_edges = ["black", COLORS["parameter"], COLORS["equation"]]
    legend_labels = ["Model Uncertainty", "Parameter Uncertainty", "Governing Equation"]
    for y, fc, ec, label in zip(
        legend_y, legend_faces, legend_edges, legend_labels
    ):
        rounded_box(
            ax,
            35,
            y,
            54,
            27,
            "",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.7,
            radius=4,
        )
        ax.text(
            99,
            y + 13.5,
            label,
            ha="left",
            va="center",
            fontsize=11.5,
            fontstyle="normal",
        )

    # Match panel (a) typography to the hierarchy used in panel (b):
    # 14 pt panel title, 11.5 pt process headers, 10.5--11.5 pt nodes,
    # and 9--10.5 pt supporting labels.
    for text_artist in ax.texts:
        old_size = text_artist.get_fontsize()
        if old_size >= 12.5:
            text_artist.set_fontsize(11.5)
        elif old_size >= 12.0:
            text_artist.set_fontsize(11.0)
        elif old_size >= 11.5:
            text_artist.set_fontsize(10.5)

    ax.text(
        35, -8,
        "(a) Conceptual representation of the four processes",
        ha="left", va="center", fontsize=14, fontweight="bold",
        color="#25313C",
    )

    # ------------------------------------------------------------------
    # Panel (b): inventory of the complete process-sensitivity index set.
    # ------------------------------------------------------------------
    divider_y = 604
    ax.plot([20, W - 20], [divider_y, divider_y], color="#B7BEC5", linewidth=1.0)
    ax.text(35, 628, "(b) Complete set of process sensitivity indices",
            ha="left", va="center", fontsize=14, fontweight="bold",
            color="#25313C")
    ax.text(W - 35, 628, "19 indices at each retained grid cell and time",
            ha="right", va="center", fontsize=10.5, color="#687684")

    ax.text(35, 658, "15 mutually exclusive variance components",
            ha="left", va="center", fontsize=11.5, fontweight="bold",
            color="#25313C")
    ax.text(648, 658, "Four total-effect",
            ha="left", va="center", fontsize=10.5, fontweight="bold",
            color="#25313C")
    ax.plot([625, 625], [648, 934], color="#C2C9D0", linewidth=1.0)

    order_rows = [
        ("Four first-order",
         [r"$PS_C$", r"$PS_F$", r"$PS_H$", r"$PS_R$"],
         684, "#DCEAF4", "#4F86B5"),
        ("Six second-order",
         [r"$PS_{CF}$", r"$PS_{CH}$", r"$PS_{CR}$",
          r"$PS_{FH}$", r"$PS_{FR}$", r"$PS_{HR}$"],
         729, "#DDEFEA", "#4E9C8A"),
        ("Four third-order",
         [r"$PS_{CFH}$", r"$PS_{CFR}$", r"$PS_{CHR}$", r"$PS_{FHR}$"],
         774, "#F5EACF", "#D9A441"),
        ("One fourth-order",
         [r"$PS_{CFHR}$"],
         819, "#F6E3DC", "#D98B73"),
    ]

    for label, terms, y, face, edge in order_rows:
        ax.text(42, y + 15, label, ha="left", va="center", fontweight="bold",
                fontsize=10.5, color="#25313C")
        start_x = 165
        gap = 74
        for idx, term in enumerate(terms):
            rounded_box(
                ax, start_x + idx * gap, y, 62, 30, term,
                facecolor=face, edgecolor=edge, linewidth=1.3,
                fontsize=11.2, radius=5,
            )

    total_terms = [
        (r"$PS_{TC}$", 649, 681), (r"$PS_{TF}$", 724, 681),
        (r"$PS_{TH}$", 799, 681), (r"$PS_{TR}$", 874, 681),
    ]
    for term, x, y in total_terms:
        rounded_box(
            ax, x, y, 67, 31, term,
            facecolor="#E8E1EF", edgecolor="#9A78A6",
            linewidth=1.3, fontsize=11.2, radius=5,
        )

    rounded_box(
        ax, 649, 730, 294, 45,
        #r"$PS_{T,K}^{\mathrm{comp}}=1-"
        #r"\dfrac{V_{\mathbf{X}_{\sim K}}"
        #r"\!\left[E_{\mathbf{X}_{K}}(\Delta\mid\mathbf{X}_{\sim K})\right]}"
        #r"{V(\Delta)}$",
        
        r"$PS_{TK}=1-"
        r"\dfrac{V_{\mathbf{X}_{\sim K}}"
        r" \!\left[E_{\mathbf{X}_{K}}(\Delta\mid\mathbf{X}_{\sim K})\right]}"
        r"{V(\Delta)}$",
        
        facecolor="#F3F5F7", edgecolor="#9AA8B5",
        linewidth=1.2, fontsize=9.8, radius=6,
    )

    rounded_box(
        ax, 649, 796, 294, 43,
        #r"$PS_{TK}^{\mathrm{sum}}=\sum_{u:\,K\in u}PS_u$",
        r"$PS_{TK}=\sum_{u:\,K\in u}PS_u$",
        facecolor="#EEF4F8", edgecolor="#5E91B8",
        linewidth=1.2, fontsize=11.2, radius=6,
    )

    ax.text(796, 875,
            "Agreement checks component consistency\nand variance closure",
            ha="center", va="center", fontsize=10.5, color="#687684",
            linespacing=1.25)

    ax.text(35, 884,
            "$C$, climate; $F$, flow; $H$, heat; $R$, reaction. "
            "Indices were evaluated at 8, 10, and 12 weeks.",
            ha="left", va="center", fontsize=10.5, color="#687684")

    return fig


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "figures", "generated")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    stem = os.path.join(output_dir, "process_uncertainty_framework_v4")

    fig = build_figure()
    fig.savefig(stem + ".png", dpi=100, facecolor="white")
    fig.savefig(stem + "_600dpi.png", dpi=600, facecolor="white")
    fig.savefig(stem + ".pdf", facecolor="white")
    fig.savefig(stem + ".svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
