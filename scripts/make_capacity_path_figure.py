"""Figure: where exactness breaks along the removal-position grid."""
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
GOLD = "#D59A28"
SLATE = "#637381"
INK = "#20252B"

ORDER = [
    "ground|exterior",
    "ground|interior_bay",
    "interior_story|exterior",
    "interior_story|interior_bay",
    "roof|exterior",
    "roof|interior_bay",
]
LABELS = [
    "Ground / exterior",
    "Ground / interior",
    "Mid / exterior",
    "Mid / interior",
    "Roof / exterior",
    "Roof / interior",
]


def bars(ax, payload, treatment, title):
    rows = payload["treatments"][treatment]["by_position_class"]
    vals = [100 * rows[k]["gap_fraction"] for k in ORDER]
    xs = np.arange(len(ORDER))
    ax.bar(xs, vals, color=TEAL, edgecolor=NAVY, linewidth=0.8, width=0.72)
    ax.set_xticks(xs, LABELS, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Gap instance fraction (%)", fontsize=9, color=INK)
    ax.set_ylim(0, max(vals) * 1.18 + 0.5)
    ax.set_title(title, loc="left", fontsize=10, weight="bold", color=NAVY, pad=8)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)
    ax.tick_params(colors=SLATE)


def main() -> None:
    payload = json.loads((DATA / "capacity_path_analysis.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), dpi=220, facecolor="white")
    bars(
        axes[0],
        payload,
        "componentwise_strengthening",
        "(a) Memberwise strengthening",
    )
    bars(
        axes[1],
        payload,
        "geometric_mean_one_redistribution",
        "(b) Product-one redistribution",
    )
    fig.suptitle(
        "Removal-position capacity path (baseline-clean instances only)",
        fontsize=11,
        weight="bold",
        color=NAVY,
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig_capacity_path.{ext}", bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(OUT / "fig_capacity_path.pdf")


if __name__ == "__main__":
    main()
