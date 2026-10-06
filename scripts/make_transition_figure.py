"""Render the corrected capacity-pattern transition figure."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIGURES = HERE.parent / "figures"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "savefig.dpi": 300, "axes.spines.top": False,
    "axes.spines.right": False,
})


def aggregate(payload, treatment, Hs, Bs):
    z = np.full((len(Hs), len(Bs)), np.nan)
    for i, H in enumerate(Hs):
        for j, B in enumerate(Bs):
            rows = []
            base = 0
            for key, value in payload["groups"].items():
                if f"H{H}|B{B}" in key:
                    base += value["baseline_clean_frames"]
                    rows.append(value["treatments"][treatment])
            if rows and base:
                z[i, j] = sum(r["frames_with_new_gap"] for r in rows) / base
    return z


def heat(ax, z, Hs, Bs, title):
    im = ax.imshow(z, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(Bs)), Bs)
    ax.set_yticks(range(len(Hs)), Hs)
    ax.set_xlabel("bays $B$")
    ax.set_ylabel("stories $H$")
    ax.set_title(title)
    for i in range(len(Hs)):
        for j in range(len(Bs)):
            if np.isfinite(z[i, j]):
                ax.text(j, i, f"{z[i,j]:.2f}", ha="center", va="center",
                        color="white" if z[i, j] > 0.55 else "black", fontsize=8)
    return im


def main():
    c = json.loads((DATA / "canonical_case.json").read_text(encoding="utf-8"))
    p = json.loads((DATA / "final_capacity_pattern_gate.json").read_text(encoding="utf-8"))
    Hs, Bs = [2, 4, 6, 8, 12], [1, 2, 3, 4]
    fig, ax = plt.subplots(2, 2, figsize=(7.5, 5.8), constrained_layout=True)
    rows = c["rows"]
    k = np.array([r["k"] for r in rows])
    q = [r["checks"][1] for r in rows]
    local = np.array([r["local"] for r in q])
    full = np.array([r["chain"] for r in q])
    static = np.array([r["static"] for r in q])
    a = ax[0, 0]
    a.plot(k, local, "o-", lw=2, color="#C4473A", label="local bound")
    a.plot(k, full, "o-", lw=2, color="#1D7A84", label="complete chain")
    a.scatter(k, static, marker="x", s=28, color="#20252B", label="static LP")
    a.axvline(1.25, color="#777", ls=":", lw=.9)
    a.set_xscale("log")
    a.set_xlabel("capacity multiplier $k$")
    a.set_ylabel("capacity")
    a.set_title("(a) Exact two-story boundary")
    a.legend(frameon=False, fontsize=7, loc="upper left")
    a.text(.04, .07, r"$\lambda_L=4k$; $\lambda_F=\min(4k,5)$", transform=a.transAxes, fontsize=8)
    z1 = aggregate(p, "componentwise_strengthening", Hs, Bs)
    z2 = aggregate(p, "geometric_mean_one_redistribution", Hs, Bs)
    im1 = heat(ax[0, 1], z1, Hs, Bs, "(b) Memberwise strengthening")
    im2 = heat(ax[1, 0], z2, Hs, Bs, "(c) Exact product-one redistribution")
    fig.colorbar(im1, ax=ax[0, 1], fraction=.046, pad=.04, label="new-gap frame fraction")
    fig.colorbar(im2, ax=ax[1, 0], fraction=.046, pad=.04, label="new-gap frame fraction")
    records = p["margin_sample"]["records"]
    d = ax[1, 1]
    pos = [r for r in records if r["margin"] >= -1e-10]
    neg = [r for r in records if r["margin"] < -1e-10]
    d.scatter([r["margin"] for r in pos], [max(float(r["gap"]), 1e-10) for r in pos], s=8, alpha=.22, c="#245A9A", label="nonnegative margin")
    d.scatter([r["margin"] for r in neg], [max(float(r["gap"]), 1e-10) for r in neg], s=8, alpha=.32, c="#C43D3A", label="negative margin")
    d.axvline(0, color="#555", ls="--", lw=.8)
    d.set_yscale("log")
    d.set_xlabel("minimum interval margin")
    d.set_ylabel("local minus complete capacity")
    d.set_title("(d) Archived 6,000-record margin sample")
    d.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Capacity patterns can open a global-sway mechanism after local screening passes", fontsize=12, fontweight="bold", y=1.02)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"fig_strengthening_phase.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote corrected transition figure")


if __name__ == "__main__":
    main()
