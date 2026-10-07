"""Figure for the axial-demand envelope.

Shows the engineering question the referee panel raised: does activating
axial-moment interaction destroy the screen-validity split found on catalogue
sections? The envelope answers it with the certificate itself.

Panels:
(a) fraction of designs exact at every removal against the prescribed column
    axial demand ratio dcr = P/(A Fy), by bay count;
(b) median and minimum worst interval margin against dcr over the multi-bay
    designs, with the certificate boundary at zero;
(c) maximum relative local-over-complete gap over the designs the screen
    already fails, showing that interaction scales the consequence of a
    failure rather than creating or removing one.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DATA = PKG / "data"
OUT = PKG / "figures"
OUT.mkdir(exist_ok=True)

PRINT_W_IN = 142.5 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "axes.unicode_minus": False,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.0,
})

LOCAL = "#B4472B"
FULL = "#1F5F7A"
ACC = "#C9A227"
GREEN = "#2E8B6E"


def main() -> None:
    payload = json.loads((DATA / "axial_demand_envelope.json")
                         .read_text(encoding="utf-8"))
    rows = payload["rows"]
    dcrs = sorted({r["dcr"] for r in rows})
    bays = [1, 2, 3]

    fig = plt.figure(figsize=(PRINT_W_IN, 7.4))
    gs = fig.add_gridspec(3, 1, hspace=0.52)

    # ---- (a) exact-at-every-removal fraction ------------------------------
    ax = fig.add_subplot(gs[0, 0])
    colours = {1: LOCAL, 2: FULL, 3: GREEN}
    for B in bays:
        frac = []
        for d in dcrs:
            sub = [r for r in rows if r["dcr"] == d and r["B"] == B]
            frac.append(100.0 * sum(1 for r in sub if r["exact_all"]) / len(sub))
        ax.plot(dcrs, frac, "o-", color=colours[B], ms=4.5, lw=1.6,
                label=f"$B={B}$")
    ax.set_ylim(-6, 108)
    ax.set_xlabel("column axial demand ratio $P/(A F_y)$")
    ax.set_ylabel("designs exact at\nevery removal (%)")
    ax.set_title("(a) The screen-validity split is unchanged by axial demand")
    ax.legend(frameon=False, ncol=3, loc="center right", handlelength=1.6)
    ax.grid(True, alpha=0.22)

    # ---- (b) worst margin over multi-bay designs --------------------------
    ax = fig.add_subplot(gs[1, 0])
    med, lo = [], []
    for d in dcrs:
        vals = [r["worst_margin"] for r in rows
                if r["dcr"] == d and r["B"] >= 2]
        med.append(float(np.median(vals)))
        lo.append(float(np.min(vals)))
    ax.plot(dcrs, med, "o-", color=FULL, ms=4.5, lw=1.6, label="median")
    ax.plot(dcrs, lo, "s--", color=LOCAL, ms=4.0, lw=1.4, label="minimum")
    ax.axhline(0.0, color=ACC, lw=1.2)
    ax.text(0.02, 12.0, "certificate boundary $\\Delta_{\\min}=0$",
            fontsize=9.0, color=ACC)
    ax.set_xlabel("column axial demand ratio $P/(A F_y)$")
    ax.set_ylabel("worst interval margin")
    ax.set_title("(b) Multi-bay margins fall with axial demand but stay "
                 "positive in all 45 designs")
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)
    ax.grid(True, alpha=0.22)

    # ---- (c) consequence of an existing failure ---------------------------
    ax = fig.add_subplot(gs[2, 0])
    gaps = []
    for d in dcrs:
        vals = [r["max_relative_gap"] for r in rows
                if r["dcr"] == d and not r["exact_all"]]
        gaps.append(max(vals) if vals else np.nan)
    ax.plot(dcrs, gaps, "o-", color=LOCAL, ms=4.5, lw=1.6)
    for d, g in zip(dcrs, gaps):
        if d in (dcrs[0], dcrs[len(dcrs) // 2], dcrs[-1]) and not np.isnan(g):
            ax.annotate(f"{g:.1f}", (d, g), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=9.0, color=LOCAL)
    ax.set_xlabel("column axial demand ratio $P/(A F_y)$")
    ax.set_ylabel("max relative\ngap (local/complete)")
    ax.set_title("(c) Interaction does not create failures; it scales the "
                 "consequence of a failure")
    ax.grid(True, alpha=0.22)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig_axial_envelope.{ext}", dpi=300,
                    facecolor="white")
    plt.close(fig)
    print(OUT / "fig_axial_envelope.pdf")


if __name__ == "__main__":
    main()
