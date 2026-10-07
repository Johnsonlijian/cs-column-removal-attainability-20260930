"""Graphical abstract for the Computers & Structures submission.

Rebuilt so that the graphic carries the paper's actual headline: the
zero-sway screen's exactness is decided by bay count and capacity pattern, and
the interval margin is the certificate that decides it.

Fixes a defect in the previous revision, which rendered a literal ``\n`` in the
graphical abstract because the label was a raw string containing ``\n``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)

NAVY = "#17324D"
TEAL = "#1D7A84"
RED = "#C4473A"
GOLD = "#D59A28"
SLATE = "#637381"
PALE = "#EAF3F4"
INK = "#20252B"
GREEN = "#2E8B6E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})

fig = plt.figure(figsize=(7.2, 4.05), dpi=220, facecolor="white")
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.12], wspace=0.24)

# ---------------------------------------------------------------- panel 1
ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.02, 0.94, "Question", color=NAVY, fontsize=10, weight="bold")

# three-bay frame; B = 1 is the pathological case, drawn separately below
for x in [0.14, 0.34, 0.54, 0.74]:
    ax.plot([x, x], [0.38, 0.78], color=SLATE, lw=1.2)
    ax.plot([x - 0.035, x + 0.035], [0.36, 0.36], color=SLATE, lw=1.0)
for y in [0.38, 0.58, 0.78]:
    ax.plot([0.14, 0.74], [y, y], color=SLATE, lw=1.2)
# removed column
ax.plot([0.74, 0.74], [0.40, 0.56], color="white", lw=4, zorder=2)
ax.plot([0.74, 0.74], [0.40, 0.56], color=RED, lw=1.7, ls=(0, (3, 2)), zorder=3)
ax.add_patch(FancyArrowPatch((0.86, 0.74), (0.86, 0.52), arrowstyle="-|>",
                             mutation_scale=11, lw=1.8, color=GOLD))
ax.text(0.88, 0.63, "global\nsway", color=GOLD, fontsize=7.4, va="center")

ax.add_patch(FancyBboxPatch((0.07, 0.06), 0.86, 0.20,
                            boxstyle="round,pad=0.015",
                            facecolor=PALE, edgecolor=TEAL, lw=1.1))
ax.text(0.50, 0.16,
        "Is the fast zero-sway screen\nthe true first-order limit?",
        ha="center", va="center", fontsize=8.2, color=INK)

# ---------------------------------------------------------------- panel 2
ax = fig.add_subplot(gs[0, 1])
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.02, 0.94, "Certificate", color=NAVY, fontsize=10, weight="bold")

xs = [0.11, 0.30, 0.49, 0.68]
for i, x in enumerate(xs):
    ax.add_patch(plt.Circle((x, 0.50), 0.042, facecolor="white",
                            edgecolor=NAVY, lw=1.5, zorder=3))
    ax.text(x, 0.50, str(i), ha="center", va="center", fontsize=8,
            color=NAVY, weight="bold", zorder=4)
    if i < len(xs) - 1:
        ax.add_patch(FancyArrowPatch((x + 0.048, 0.50), (xs[i + 1] - 0.048, 0.50),
                                     arrowstyle="-|>", mutation_scale=9,
                                     lw=1.4, color=SLATE, zorder=2))
ax.text(0.80, 0.50, "story\nchain", color=SLATE, fontsize=7.2,
        va="center", ha="left")

ax.add_patch(FancyBboxPatch((0.06, 0.68), 0.88, 0.15,
                            boxstyle="round,pad=0.02",
                            facecolor=PALE, edgecolor=TEAL, lw=1.2))
ax.text(0.50, 0.755, r"all interval margins $\geq 0$",
        ha="center", va="center", fontsize=9, color=TEAL, weight="bold")

ax.text(0.50, 0.30, "screen is EXACT", ha="center", va="center",
        fontsize=8.6, color=GREEN, weight="bold")
ax.text(0.50, 0.20, "otherwise a sway mechanism\nbeats it, and the margin\n"
                    "names the story interval",
        ha="center", va="center", fontsize=7.6, color=RED)
ax.text(0.50, 0.04, r"decided in $O(H)$ per removal",
        ha="center", va="center", fontsize=7.6, color=INK)

# ---------------------------------------------------------------- panel 3
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.02, 0.94, "Finding", color=NAVY, fontsize=10, weight="bold")

rows = [
    ("Single bay", "0 / 96 exact", RED),
    ("Two bays and up", "192 / 192 exact", GREEN),
    ("Realistic upgrades", "0 broke", TEAL),
    ("Independent LP checks", "912", GOLD),
]
for i, (txt, val, col) in enumerate(rows):
    y = 0.78 - i * 0.165
    ax.add_patch(Rectangle((0.03, y - 0.055), 0.94, 0.115,
                           facecolor=col, alpha=0.13, edgecolor=col, lw=1.0))
    ax.text(0.06, y, txt, color=INK, fontsize=7.6, va="center")
    ax.text(0.955, y, val, color=col, fontsize=7.8, va="center",
            ha="right", weight="bold")

ax.text(0.50, 0.055, "catalogue sections, $F_y$ = 345 MPa",
        ha="center", va="center", fontsize=7.2, color=SLATE)

fig.suptitle(
    "The zero-sway column-removal screen is not always exact:\n"
    "an interval-margin certificate decides when it can be trusted",
    fontsize=10.4, weight="bold", color=NAVY, y=1.005, linespacing=1.35,
)

fig.savefig(OUT / "fig_graphical_abstract.pdf", bbox_inches="tight",
            facecolor="white")
fig.savefig(OUT / "fig_graphical_abstract.svg", bbox_inches="tight",
            facecolor="white")
fig.savefig(OUT / "fig_graphical_abstract.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(OUT / "fig_graphical_abstract.pdf")
