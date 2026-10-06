"""Graphical abstract for Computers & Structures optional upload."""
from __future__ import annotations

from pathlib import Path

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

fig = plt.figure(figsize=(7.2, 4.0), dpi=220, facecolor="white")
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0], wspace=0.22)

# Panel 1: question
ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.02, 0.92, "Question", color=NAVY, fontsize=10, weight="bold")
for x in [0.18, 0.42, 0.66]:
    ax.plot([x, x], [0.18, 0.72], color=SLATE, lw=1.2)
    ax.plot([x - 0.04, x + 0.04], [0.16, 0.16], color=SLATE, lw=1.0)
for y in [0.18, 0.45, 0.72]:
    ax.plot([0.18, 0.66], [y, y], color=SLATE, lw=1.2)
ax.plot([0.66, 0.66], [0.20, 0.43], color="white", lw=4, zorder=2)
ax.plot([0.66, 0.66], [0.20, 0.43], color=RED, lw=1.6, ls=(0, (4, 3)), zorder=3)
ax.add_patch(FancyArrowPatch((0.72, 0.58), (0.72, 0.38), arrowstyle="-|>", mutation_scale=11, lw=1.8, color=GOLD))
ax.text(0.02, 0.08, "When is a zero-sway\nlocal screen exact?", color=INK, fontsize=9)

# Panel 2: certificate
ax = fig.add_subplot(gs[0, 1])
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.02, 0.92, "Certificate", color=NAVY, fontsize=10, weight="bold")
xs = [0.12, 0.32, 0.52, 0.72]
for i, x in enumerate(xs):
    ax.add_patch(plt.Circle((x, 0.52), 0.045, facecolor="white", edgecolor=NAVY, lw=1.5))
    ax.text(x, 0.52, str(i), ha="center", va="center", fontsize=8, color=NAVY, weight="bold")
    if i < len(xs) - 1:
        ax.add_patch(FancyArrowPatch((x + 0.05, 0.52), (xs[i + 1] - 0.05, 0.52), arrowstyle="-|>", mutation_scale=9, lw=1.4, color=SLATE))
ax.add_patch(FancyBboxPatch((0.18, 0.66), 0.52, 0.14, boxstyle="round,pad=0.02", facecolor=PALE, edgecolor=TEAL, lw=1.2))
ax.text(0.44, 0.73, r"all interval margins $\geq 0$", ha="center", va="center", fontsize=9, color=TEAL, weight="bold")
ax.text(0.02, 0.08, r"$\lambda_{\rm loc}=\lambda_{\rm full}$ iff\ncertificate passes", color=INK, fontsize=9)

# Panel 3: evidence
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text(0.02, 0.92, "Evidence", color=NAVY, fontsize=10, weight="bold")
items = [
    ("912 LP checks", TEAL),
    ("21% / 56% gaps", RED),
    ("Pushdown verdict", GOLD),
]
for i, (txt, col) in enumerate(items):
    y = 0.68 - i * 0.18
    ax.add_patch(Rectangle((0.06, y - 0.05), 0.88, 0.11, facecolor=col, alpha=0.15, edgecolor=col, lw=1.0))
    ax.text(0.10, y, txt, color=INK, fontsize=9, va="center")
ax.text(0.02, 0.08, "Frozen 2,880-frame\nconditional study", color=INK, fontsize=9)

fig.suptitle(
    "Exactness certificates for zero-sway column-removal limits",
    fontsize=11,
    weight="bold",
    color=NAVY,
    y=0.98,
)
fig.savefig(OUT / "fig_graphical_abstract.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig_graphical_abstract.svg", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig_graphical_abstract.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUT / "fig_graphical_abstract.pdf")
