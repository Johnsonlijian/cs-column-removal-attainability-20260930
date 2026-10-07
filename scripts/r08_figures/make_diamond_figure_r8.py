"""Figure for the declared diamond-domain boundary probe.

Redrawn at the width the manuscript actually prints at (142.5 mm). The earlier
revision was authored at 9.6 in and included at 142.5 mm, which put every label
between 4.7 and 5.9 pt on the page. All plotted values come from the same
``data/diamond_domain_probe.json`` as before.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)

PRINT_W_IN = 142.5 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11.0,
    "axes.titlesize": 11.5,
    "axes.labelsize": 11.0,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.0,
    # ASCII hyphen: matplotlib's U+2212 is emitted as a separate tiny glyph
    # span, which breaks automated legibility checks.
    "axes.unicode_minus": False,
    "axes.linewidth": 0.7,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.015,
})

NAVY = "#17324D"
TEAL = "#1D7A84"
RED = "#C4473A"
SLATE = "#637381"
INK = "#20252B"


def main() -> None:
    payload = json.loads((DATA / "diamond_domain_probe.json")
                         .read_text(encoding="utf-8"))

    # Canvas is set to the aspect ratio the manuscript prints at (142.5 mm
    # wide). Two panels at 11 pt cannot share a 142.5 x 129 mm canvas without
    # collisions, so the layout is 142.5 x 156 mm and is included with no
    # reduction; printed point size therefore equals the size set here.
    fig = plt.figure(figsize=(PRINT_W_IN, 6.15))
    gs = fig.add_gridspec(2, 1, hspace=0.30)

    # ---- (a) analytical removal -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    rows = payload["canonical"]
    ks = [r["k"] for r in rows]
    x = np.arange(len(ks))
    width = 0.36
    moment = [r["complete_moment"] for r in rows]
    diamond = [r["static_diamond"] for r in rows]
    ax.bar(x - width / 2, moment, width, color=TEAL, label="moment complete")
    ax.bar(x + width / 2, diamond, width, color=RED,
           label="diamond static limit")
    for xi, m, d in zip(x, moment, diamond):
        drop = (m - d) / m * 100.0
        ax.text(xi, max(m, d) + 0.18, f"$-{drop:.1f}\\%$", ha="center",
                fontsize=10.0, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$k={k:g}$" for k in ks])
    ax.set_ylim(0, 7.4)
    ax.set_ylabel("limit multiplier")
    ax.set_title("(a) Analytical removal $(1,1)$", loc="left", color=NAVY)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6,
              borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)
    ax.grid(True, axis="y", alpha=0.20)

    # ---- (b) one baseline frame per stratification cell -------------------
    ax = fig.add_subplot(gs[1, 0])
    ens = payload["ensemble_one_removal_per_cell"]
    regimes = ["balanced", "weak_base", "axial"]
    labels = ["balanced", "weak base", "low column moment"]
    med, hi = [], []
    for reg in regimes:
        vals = np.array([100 * r["relative_diamond_drop"] for r in ens
                         if r["regime"] == reg])
        med.append(float(np.median(vals)))
        hi.append(float(np.max(vals)))
    xs = np.arange(len(regimes))
    ax.bar(xs, med, color=TEAL, width=0.60, label="median drop")
    ax.scatter(xs, hi, color=RED, zorder=3, s=34, label="maximum drop")
    for i, (m, h) in enumerate(zip(med, hi)):
        ax.text(i - 0.03, m + 2.0, f"{m:.1f}", ha="center", fontsize=10.6,
                color=INK)
        ax.text(i + 0.06, h + 2.0, f"{h:.1f}", ha="left", fontsize=10.0,
                color=RED)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 124)
    ax.set_ylabel("relative drop vs moment strip (%)")
    ax.set_title("(b) One baseline frame per stratification cell",
                 loc="left", color=NAVY)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6,
              borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)
    ax.grid(True, axis="y", alpha=0.20)

    for ext in ("pdf", "svg", "png"):
        # The canvas is already exactly the printed size, so no bounding-box
        # adjustment is wanted: savefig.bbox is left unset above and this call
        # uses the full figure box. A "tight" box would grow the canvas to fit
        # labels and hand LaTeX a wider page that it then shrinks, undoing the
        # intended point sizes.
        fig.savefig(OUT / f"fig_diamond_boundary.{ext}", dpi=300,
                    bbox_inches=None, pad_inches=0.0,
                    facecolor="white")
    plt.close(fig)
    print(OUT / "fig_diamond_boundary.pdf")


if __name__ == "__main__":
    main()
