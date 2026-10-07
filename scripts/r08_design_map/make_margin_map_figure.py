"""Figure for the screen-exactness boundary: the margin surface.

Panel (a) is the worst signed interval margin over the plane of the two
cut-adjacent beam capacities. The zero contour is the exactness boundary, and
the balanced diagonal is its ridge.
Panel (b) follows the balanced diagonal against the beam-to-column capacity
ratio, showing the margin falling monotonically to exactly zero.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DATA = PKG / "data"
OUT = PKG / "figures"
OUT.mkdir(exist_ok=True)

PRINT_W_IN = 142.5 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12.0,
    "axes.titlesize": 12.5,
    "axes.labelsize": 12.0,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.5,
    "axes.unicode_minus": False,
    "axes.linewidth": 0.7,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.015,
})

LOCAL = "#B4472B"
FULL = "#1F5F7A"
ACC = "#C9A227"


def main() -> None:
    payload = json.loads((DATA / "margin_map.json").read_text(encoding="utf-8"))
    f = np.array(payload["factors"], float)
    Z = np.array(payload["worst_margin"], float)

    fig = plt.figure(figsize=(PRINT_W_IN, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.42)

    # ---- (a) margin surface ----------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    lim = float(np.nanmax(np.abs(Z)))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    im = ax.imshow(Z, cmap="RdBu_r", norm=norm, origin="lower",
                   extent=[np.log10(f[0]), np.log10(f[-1]),
                           np.log10(f[0]), np.log10(f[-1])],
                   aspect="auto")
    ax.contour(np.log10(f), np.log10(f), Z, levels=[0.0], colors=[ACC],
               linewidths=1.6)
    ax.plot([np.log10(f[0]), np.log10(f[-1])],
            [np.log10(f[0]), np.log10(f[-1])], ls=":", color="0.25", lw=1.1)
    ticks = np.log10([0.5, 1.0, 2.0, 4.0, 6.0])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0.5", "1", "2", "4", "6"])
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0.5", "1", "2", "4", "6"])
    ax.set_xlabel("beam capacity right of cut ($\\times$ ref)")
    ax.set_ylabel("beam capacity left of cut ($\\times$ ref)")
    ax.set_title("(a) Worst signed margin; gold line is the boundary")
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label("worst margin")
    ax.text(np.log10(0.55), np.log10(5.0), "screen exact",
            fontsize=10.0, color="white", ha="left", va="top")
    ax.text(np.log10(4.2), np.log10(0.55), "certified failure",
            fontsize=10.0, color="#3B0A0A", ha="right", va="bottom")

    # ---- (b) balanced diagonal against capacity level --------------------
    ax = fig.add_subplot(gs[1, 0])
    diag = np.array([Z[i, i] for i in range(len(f))])
    ax.plot(f, diag, "o-", color=FULL, ms=5, lw=1.7)
    ax.axhline(0.0, color=ACC, lw=1.3)
    ax.text(f[0], 24.0, "boundary: margin = 0", fontsize=10.6, color=ACC)
    for x, y in zip(f, diag):
        if y in (diag.max(), 0.0) or x in (f[0], f[-1]):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 9 if y >= 0 else -16), ha="center",
                        fontsize=10.0, color=FULL)
    ax.set_xscale("log")
    ax.set_xlabel("balanced beam capacity ($\\times$ ref)")
    ax.set_ylabel("worst margin\n(balanced)")
    ax.set_title("(b) Balanced diagonal: margin falls to exactly zero")
    ax.grid(True, alpha=0.22, which="both")

    for ext in ("pdf", "svg", "png"):
        # Reserve margins: the rotated colourbar label and the y-axis labels
        # otherwise run past the canvas edge, which the overflow checker flags
        # and LaTeX would clip.
        fig.subplots_adjust(left=0.19, right=0.86, top=0.95, bottom=0.09)
        fig.savefig(OUT / f"fig_margin_map.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print(OUT / "fig_margin_map.pdf")


if __name__ == "__main__":
    main()
