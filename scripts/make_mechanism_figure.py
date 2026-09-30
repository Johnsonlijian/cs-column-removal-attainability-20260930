"""Create a vector-first mechanism figure for the C&S release package."""
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

fig = plt.figure(figsize=(12.2, 4.25), dpi=220, facecolor="white")
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.05], wspace=0.28)

# Panel (a): canonical frame
ax = fig.add_subplot(gs[0, 0])
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-0.35, 1.8)
ax.set_ylim(-0.35, 2.55)
ax.text(-0.42, 2.48, "(a) Canonical frame", color=NAVY, fontsize=11, weight="bold", va="top")
for x in [0, 1]:
    ax.plot([x, x], [0, 2], color=SLATE, lw=1.3, zorder=1)
    ax.plot([x - 0.11, x + 0.11], [-0.02, -0.02], color=SLATE, lw=1.0)
for y in [0, 1, 2]:
    ax.plot([0, 1], [y, y], color=SLATE, lw=1.3, zorder=1)
# remove first story exterior column at x=1
ax.plot([1, 1], [0.03, 0.97], color="white", lw=5.0, zorder=3)
ax.plot([1, 1], [0.03, 0.97], color=RED, lw=2.0, ls=(0, (5, 3)), zorder=4)
ax.text(1.08, 0.46, "removed\ncolumn", color=RED, fontsize=9, va="center")
# local mechanism motion
ax.add_patch(FancyArrowPatch((1.0, 1.05), (1.0, 0.72), arrowstyle="-|>", mutation_scale=12, lw=2.0, color=TEAL))
ax.add_patch(FancyArrowPatch((1.0, 2.02), (1.0, 1.65), arrowstyle="-|>", mutation_scale=12, lw=2.0, color=TEAL))
ax.text(-0.37, 1.08, r"local $w_r=0$", color=TEAL, fontsize=9, rotation=90, va="center")
# sway indication
ax.add_patch(FancyArrowPatch((1.45, 0.98), (1.45, 0.58), arrowstyle="-|>", mutation_scale=12, lw=2.0, color=GOLD))
ax.text(1.49, 0.78, "global\nsway", color=GOLD, fontsize=9, va="center")
ax.text(0.0, -0.22, r"$\lambda_{\rm loc}$ versus $\lambda_{\rm full}$", color=INK, fontsize=9)

# Panel (b): interval certificate
ax = fig.add_subplot(gs[0, 1])
ax.axis("off")
ax.set_xlim(-0.2, 5.2)
ax.set_ylim(-0.2, 3.1)
ax.text(-0.18, 3.04, "(b) Interval certificate", color=NAVY, fontsize=11, weight="bold", va="top")
ax.text(0.0, 2.67, "two-state story chain", color=SLATE, fontsize=9)
# chain nodes
xs = [0.55, 1.75, 2.95, 4.15]
for i, x in enumerate(xs):
    ax.add_patch(plt.Circle((x, 1.75), 0.14, facecolor="white", edgecolor=NAVY, lw=1.8))
    ax.text(x, 1.75, str(i), ha="center", va="center", fontsize=9, color=NAVY, weight="bold")
    if i < len(xs)-1:
        ax.add_patch(FancyArrowPatch((x + 0.18, 1.75), (xs[i+1] - 0.18, 1.75), arrowstyle="-|>", mutation_scale=10, lw=1.7, color=SLATE))
        ax.text((x+xs[i+1])/2, 1.93, rf"$V_{i+1}$", ha="center", fontsize=9, color=SLATE)
# zero-sway path
ax.plot(xs, [1.75]*4, color=TEAL, lw=5.0, alpha=0.22, solid_capstyle="round")
ax.text(0.04, 1.30, "local path", color=TEAL, fontsize=9)
# interval box above
ax.add_patch(FancyBboxPatch((1.27, 2.10), 2.56, 0.42, boxstyle="round,pad=0.03,rounding_size=0.05", facecolor=PALE, edgecolor=TEAL, lw=1.3))
ax.text(2.55, 2.31, r"interval $[a,b]$", ha="center", va="center", color=TEAL, fontsize=9, weight="bold")
ax.add_patch(FancyArrowPatch((2.55, 2.08), (2.55, 1.93), arrowstyle="-|>", mutation_scale=10, lw=1.2, color=TEAL))
# equation box
ax.add_patch(FancyBboxPatch((0.10, 0.30), 4.72, 0.62, boxstyle="round,pad=0.05,rounding_size=0.06", facecolor="#FFF8E7", edgecolor=GOLD, lw=1.3))
ax.text(2.46, 0.67, r"$\Delta[a,b]=\text{entry}+\sum\text{interior}+\text{exit}$", ha="center", va="center", fontsize=11, color=INK)
ax.text(2.46, 0.43, r"all margins $\geq 0$ $\Longleftrightarrow$ local screen is exact", ha="center", va="center", fontsize=8.5, color=INK)

# Panel (c): strict separation
ax = fig.add_subplot(gs[0, 2])
ax.set_xlim(-0.3, 5.4)
ax.set_ylim(0, 22.5)
ax.spines[["top", "right"]].set_visible(False)
ax.spines["left"].set_color(SLATE)
ax.spines["bottom"].set_color(SLATE)
ax.tick_params(colors=SLATE, labelsize=8)
ax.set_xticks([0.9, 2.3, 3.7])
ax.set_xticklabels(["$k=1$", "$k=2$", "$k=5$"], color=INK)
ax.set_ylabel("first-order capacity", color=INK, fontsize=9)
ax.set_title("(c) Strict separation", loc="left", color=NAVY, fontsize=11, weight="bold", pad=10)
ks = [1, 2, 5]
local = [4, 8, 20]
full = [4, 5, 5]
for x, a, b in zip([0.9, 2.3, 3.7], local, full):
    ax.plot([x-0.20, x-0.20], [0, a], color=TEAL, lw=7, solid_capstyle="butt")
    ax.plot([x+0.20, x+0.20], [0, b], color=RED, lw=7, solid_capstyle="butt")
    ax.text(x-0.20, a+0.28, f"{a:g}", color=TEAL, fontsize=8.5, ha="center")
    ax.text(x+0.20, b+0.28, f"{b:g}", color=RED, fontsize=8.5, ha="center")
ax.text(0.68, 21.0, r"$\lambda_{\rm loc}=4k$", color=TEAL, fontsize=9)
ax.text(3.15, 21.0, r"$\lambda_{\rm full}=\min(4k,5)$", color=RED, fontsize=9)
ax.legend([plt.Line2D([0], [0], color=TEAL, lw=6), plt.Line2D([0], [0], color=RED, lw=6)], ["local", "complete"], frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.90))
ax.text(0.0, -0.13, "capacity strengthening can raise the restricted bound\nwhile the global bottleneck remains", transform=ax.transAxes, fontsize=8.5, color=INK, va="top")

fig.savefig(OUT / "fig_mechanism_certificate.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig_mechanism_certificate.svg", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig_mechanism_certificate.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUT / "fig_mechanism_certificate.pdf")
