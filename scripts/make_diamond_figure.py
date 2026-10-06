"""Figure for the declared diamond-domain boundary probe."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)

NAVY = "#17324D"
TEAL = "#1D7A84"
RED = "#C4473A"
SLATE = "#637381"
INK = "#20252B"


def main() -> None:
    payload = json.loads((DATA / "diamond_domain_probe.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), dpi=220, facecolor="white")

    ax = axes[0]
    rows = payload["canonical"]
    ks = [r["k"] for r in rows]
    x = np.arange(len(ks))
    width = 0.36
    moment = [r["complete_moment"] for r in rows]
    diamond = [r["static_diamond"] for r in rows]
    ax.bar(x - width / 2, moment, width, color=TEAL, label="moment complete")
    ax.bar(x + width / 2, diamond, width, color=RED, label="diamond static")
    ax.set_xticks(x, [f"$k={k:g}$" for k in ks])
    ax.set_ylim(0, 6.4)
    ax.set_ylabel("limit multiplier", color=INK)
    ax.set_title("(a) Analytical removal $(1,1)$", loc="left", color=NAVY, fontsize=10, weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)

    ax = axes[1]
    ens = payload["ensemble_one_removal_per_cell"]
    regimes = ["balanced", "weak_base", "axial"]
    labels = ["balanced", "weak base", "low column\nmoment"]
    med = []
    hi = []
    for reg in regimes:
        vals = np.array([100 * r["relative_diamond_drop"] for r in ens if r["regime"] == reg])
        med.append(float(np.median(vals)))
        hi.append(float(np.max(vals)))
    xs = np.arange(len(regimes))
    ax.bar(xs, med, color=TEAL, width=0.62, label="median drop")
    ax.scatter(xs, hi, color=RED, zorder=3, label="maximum drop")
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("relative drop of diamond vs moment strip (%)", color=INK)
    ax.set_title("(b) One baseline frame per cell", loc="left", color=NAVY, fontsize=10, weight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)
    for i, v in enumerate(med):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=8, color=INK)

    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig_diamond_boundary.{ext}", bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(OUT / "fig_diamond_boundary.pdf")


if __name__ == "__main__":
    main()
