"""Journal-style pseudocode for concurrent binning estimation (ABC example).

Version 12 follows the revised manuscript equations. It removes the auxiliary
J notation and directly estimates first-order, interaction, and total-effect
process sensitivity indices from a common model-and-parameter ensemble. Python-style indentation
defines loop scope; ``end for`` statements are intentionally omitted. Bold
parameter vectors use upright Greek symbols throughout.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "figures" / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

INK = "#111111"
BLUE = "#2584C6"
RULE = "#9A9A9A"


# (indent, role, text); roles: section, statement, compute, result.
LINES = [
    (0, "section", "# Step 1: Generate the model-and-parameter ensemble"),
    (0, "statement", r"for $M_A$ in $\mathbf{M}_A$:"),
    (1, "statement", r"for $M_B$ in $\mathbf{M}_B$:"),
    (2, "statement", r"for $M_C$ in $\mathbf{M}_C$:"),
    (3, "statement", r"for $(\mathbf{θ}_A,\mathbf{θ}_B,"
                       r"\mathbf{θ}_C)$ in joint realizations conditional on $(M_A,M_B,M_C)$:"),
    (4, "compute", r"$\Delta\leftarrow$ run model with $(M_A,M_B,M_C,"
                    r"\mathbf{θ}_A,\mathbf{θ}_B,\mathbf{θ}_C)$"),
    (0, "compute", r"compute $\widehat E(\Delta)$ and $\widehat V(\Delta)$ from all ensemble outputs using model averaging"),

    (0, "section", r"# Step 2: Estimate first-order process sensitivity indices"),
    (0, "statement", r"for $M_A$ in $\mathbf{M}_A$:"),
    (1, "statement", r"for $B_{A,h}$ in bins of $\mathbf{θ}_A\mid M_A$:"),
    (2, "statement", r"# Inner expectation and model averaging over $M_{\sim A}$"),
    (2, "compute", r"$\widehat m_{A,h}(M_A)\leftarrow\sum_{M_{\sim A}}w(M_{\sim A})"
                    r"\overline{\Delta}(\mathbf{θ}_A\in B_{A,h},M_A,M_{\sim A})$"),
    (0, "compute", r"# Outer calculation over bins and model averaging over $M_A$"),
    (0, "compute", r"$\widehat{PS}_A\leftarrow\{\sum_{M_A}w(M_A)\sum_h p_{A,h}(M_A)"
                    r"\widehat m_{A,h}^{2}(M_A)-[\widehat E(\Delta)]^{2}\}/\widehat V(\Delta)$"),
    (0, "result", r"repeat with exchanged subscripts to obtain "
                   r"$\widehat{PS}_B$ and $\widehat{PS}_C$"),

    (0, "section", r"# Step 3: Estimate second-order process-interaction indices"),
    (0, "statement", r"for $M_{AB}$ in $\mathbf{M}_{AB}$:"),
    (1, "statement", r"for $B_{AB,h}$ in joint bins of $\mathbf{θ}_{AB}\mid M_{AB}$:"),
    (2, "statement", r"# Inner expectation and model averaging over $M_C$"),
    (2, "compute", r"$\widehat m_{AB,h}(M_{AB})\leftarrow\sum_{M_C}w(M_C)"
                    r"\overline{\Delta}(\mathbf{θ}_{AB}\in B_{AB,h},M_{AB},M_C)$"),
    (0, "compute", r"# Outer calculation over joint bins and model averaging over $M_{AB}$"),
    (0, "compute", r"$\widehat{PS}_{AB}\leftarrow\{\sum_{M_{AB}}w(M_{AB})\sum_h "
                    r"p_{AB,h}(M_{AB})\widehat m_{AB,h}^{2}(M_{AB})"
                    r"-[\widehat E(\Delta)]^{2}\}/\widehat V(\Delta)"
                    r"-\widehat{PS}_A-\widehat{PS}_B$"),
    (0, "result", r"repeat with exchanged subscripts to obtain "
                   r"$\widehat{PS}_{AC}$ and $\widehat{PS}_{BC}$"),

    (0, "section", "# Step 4: Estimate third-order and total-effect indices"),
    (0, "compute", r"$\widehat{PS}_{ABC}\leftarrow 1-\widehat{PS}_A-\widehat{PS}_B-"
                    r"\widehat{PS}_C-\widehat{PS}_{AB}-\widehat{PS}_{AC}-\widehat{PS}_{BC}$"),
    (0, "compute", r"$\widehat{PS}_{TA}\leftarrow1-\{\sum_{M_{BC}}w(M_{BC})\sum_h "
                    r"p_{BC,h}(M_{BC})\widehat m_{BC,h}^{2}(M_{BC})"
                    r"-[\widehat E(\Delta)]^{2}\}/\widehat V(\Delta)$"),
    (0, "result", r"verify $\widehat{PS}_{TA}=\widehat{PS}_A+\widehat{PS}_{AB}+"
                   r"\widehat{PS}_{AC}+\widehat{PS}_{ABC}$"),
    (0, "result", r"repeat with exchanged subscripts to obtain "
                   r"$\widehat{PS}_{TB}$ and $\widehat{PS}_{TC}$"),
]


# A compact single-column algorithm plate matching Version 8.
fig = plt.figure(figsize=(4.15, 5.90), facecolor="white")
ax = fig.add_axes([0.035, 0.025, 0.93, 0.95])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Publication-style header.
ax.plot([0, 1], [0.992, 0.992], color=RULE, lw=0.65)
ax.text(0.003, 0.978, "Input:", fontsize=6.5, fontweight="bold",
        ha="left", va="center", color=INK)
ax.text(0.105, 0.978,
        r"Process-model alternatives and weights, parameter realizations, and bin assignments",
        fontsize=6.0, ha="left", va="center", color=INK)
ax.plot([0, 1], [0.958, 0.958], color=RULE, lw=0.45)

body_top = 0.956
body_bottom = 0.058
line_h = (body_top - body_bottom) / len(LINES)
number_x = 0.032
text_x = 0.062
indent_step = 0.034

for number, (indent, role, content) in enumerate(LINES, start=1):
    y = body_top - (number - 0.5) * line_h
    ax.text(number_x, y, f"{number}:", fontsize=6.0, color=INK,
            ha="right", va="center")
    is_comment = content.lstrip().startswith("#")
    color = BLUE if is_comment else INK
    weight = "bold" if role == "section" else "normal"
    ax.text(text_x + indent * indent_step, y, content,
            fontsize=6.0 if role == "section" else 5.65,
            color=color, fontweight=weight, fontstyle="normal",
            ha="left", va="center")

ax.plot([0, 1], [0.044, 0.044], color=RULE, lw=0.45)
ax.text(0.003, 0.027, "Output:", fontsize=5.9, fontweight="bold",
        ha="left", va="center", color=INK)
ax.text(
    0.112, 0.027,
    r"$\widehat{PS}_A$, $\widehat{PS}_B$, $\widehat{PS}_C$, "
    r"$\widehat{PS}_{AB}$, $\widehat{PS}_{AC}$, $\widehat{PS}_{BC}$, "
    r"$\widehat{PS}_{ABC}$, $\widehat{PS}_{TA}$, "
    r"$\widehat{PS}_{TB}$, and $\widehat{PS}_{TC}$",
    fontsize=6.5,
    ha="left",
    va="center",
    color=INK,
)
ax.plot([0, 1], [0.009, 0.009], color=RULE, lw=0.65)

stem = OUT_DIR / "binning_pseudocode_ABC_V12"
fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
